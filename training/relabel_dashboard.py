"""Relabel dashboard — tracks disagreement-queue progress, speed (pos/hr & pos/s), and band.

Run:  python training/relabel_dashboard.py
Open: http://localhost:8080

Shows:
  - Total labels done vs queue size (active manifest run vs legacy runs)
  - PC Head Frontier chunk & true static-vs-5k disagreement (from dis_delta_queue.npy)
  - Current disagreement band (head = biggest disagreement → tail = smallest)
  - Live speed: positions / hour (primary) and positions / second (right above ETA)
  - Active worker count (PC titanium-cli processes + Docker/Modal status)
  - Per-directory breakdown with network/exe SHA provenance
  - Latest chunk slices with true queue disagreement
  - ETA to completion
"""
import collections
import glob
import json
import os
import subprocess as sp
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(ROOT), "data", "nnue", "relabel")
TOTAL_DATASET = 33_891_139   # all positions in positions.bin / labels.npz
QUEUE_TOTAL = 31_568_598     # eligible rows in dis_positions.bin (from manifest)
CHUNK = 5000                 # rows per chunk

MANIFEST_PATH = os.path.join(D, "dis_queue_manifest.json")
LEDGER_PATH = os.path.join(D, "dis_labeled_ledger.json")
DELTA_QUEUE_PATH = os.path.join(D, "dis_delta_queue.npy")

# Lazy-loaded memmap of true queue disagreement (abs(current_static_cp - stale_5k_cp))
_delta_queue = None


def get_delta_queue():
    global _delta_queue
    if _delta_queue is None and os.path.exists(DELTA_QUEUE_PATH):
        try:
            _delta_queue = np.load(DELTA_QUEUE_PATH, mmap_mode="r")
        except Exception:
            _delta_queue = None
    return _delta_queue


def chunk_true_disagreement(idx):
    """Return average true disagreement (cp) for chunk idx from dis_delta_queue.npy."""
    dq = get_delta_queue()
    if dq is None or idx < 0:
        return None
    lo = idx * CHUNK
    hi = min((idx + 1) * CHUNK, len(dq))
    if lo >= hi:
        return None
    return float(np.mean(dq[lo:hi]))


class FishnetCoordinator:
    """Thread-safe streaming work coordinator inspired by Fishnet / Fishtest.
    Leases chunks strictly from the contiguous head, enforces double-buffering prefetch,
    and automatically reclaims expired leases after timeout.
    """
    def __init__(self, manifest_path, total_rows, lease_timeout_sec=900):
        self.lock = threading.Lock()
        self.manifest_path = manifest_path
        self.total_rows = total_rows
        self.total_chunks = (total_rows + CHUNK - 1) // CHUNK
        self.lease_timeout_sec = lease_timeout_sec
        self.in_flight = {}  # chunk_id -> {"worker": str, "claimed_at": float}
        self.workers = {}    # worker_id -> {"last_seen": float, "chunks_done": int, "current_chunks": list}

    def _get_active_out_dir(self):
        try:
            if os.path.exists(self.manifest_path):
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    man = json.load(f)
                    od = man.get("output_dir", "")
                    if od:
                        p = os.path.join(D, od)
                        if os.path.isdir(p):
                            return p
        except Exception:
            pass
        return None

    def get_completed_chunks(self):
        completed = set()
        od = self._get_active_out_dir()
        if od:
            for f in glob.glob(os.path.join(od, "c_*.txt")):
                b = os.path.basename(f).split(".")[0].split("_")
                if len(b) == 2 and b[1].isdigit():
                    completed.add(int(b[1]))
        return completed

    def clean_expired_leases(self, now):
        expired = [cid for cid, info in self.in_flight.items() if now - info["claimed_at"] > self.lease_timeout_sec]
        for cid in expired:
            del self.in_flight[cid]

    def claim_head(self, worker_id, count=1, headroom=0):
        with self.lock:
            now = time.time()
            self.clean_expired_leases(now)
            completed = self.get_completed_chunks()

            w = self.workers.setdefault(worker_id, {"last_seen": now, "chunks_done": 0, "current_chunks": []})
            w["last_seen"] = now

            claimed = []
            cur = max(0, headroom)
            while cur < self.total_chunks and len(claimed) < count:
                if cur not in completed and cur not in self.in_flight:
                    claimed.append(cur)
                    self.in_flight[cur] = {"worker": worker_id, "claimed_at": now}
                cur += 1

            w["current_chunks"] = claimed
            return claimed

    def submit_chunk(self, worker_id, chunk_id, lines_text):
        with self.lock:
            od = self._get_active_out_dir()
            if not od:
                return False, "no active output directory found"

            lines = [l for l in lines_text.splitlines() if l.strip()]
            if len(lines) != CHUNK:
                return False, f"invalid line count: {len(lines)} (expected {CHUNK})"

            out_path = os.path.join(od, f"c_{chunk_id}.txt")
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="ascii") as f:
                f.write(lines_text if lines_text.endswith("\n") else lines_text + "\n")
            os.replace(tmp_path, out_path)

            if chunk_id in self.in_flight:
                del self.in_flight[chunk_id]

            if worker_id in self.workers:
                self.workers[worker_id]["chunks_done"] += 1
                self.workers[worker_id]["last_seen"] = time.time()
                if chunk_id in self.workers[worker_id]["current_chunks"]:
                    self.workers[worker_id]["current_chunks"].remove(chunk_id)

            return True, "ok"

    def status(self):
        with self.lock:
            now = time.time()
            self.clean_expired_leases(now)
            active_workers = {
                wid: {
                    "last_seen_sec_ago": round(now - winfo["last_seen"], 1),
                    "chunks_done": winfo["chunks_done"],
                    "in_flight": winfo["current_chunks"]
                }
                for wid, winfo in self.workers.items()
                if now - winfo["last_seen"] < 180
            }
            return {
                "active_workers": active_workers,
                "in_flight_count": len(self.in_flight),
                "in_flight_chunks": sorted(list(self.in_flight.keys()))[:30],
            }


coordinator = FishnetCoordinator(MANIFEST_PATH, QUEUE_TOTAL)


# Rolling speed tracker: ring buffer of (timestamp, active_labels)
_speed_ring = collections.deque(maxlen=120)  # ~6 minutes at 3s polling
_run_start_info = None  # (earliest_chunk_mtime, baseline_labels)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


_line_cache = {}  # path -> (mtime, size, count)


def count_lines(path):
    try:
        st = os.stat(path)
        mtime, size = st.st_mtime, st.st_size
        cached = _line_cache.get(path)
        if cached is not None and cached[0] == mtime and cached[1] == size:
            return cached[2]
        with open(path, "rb") as f:
            cnt = sum(1 for _ in f)
        _line_cache[path] = (mtime, size, cnt)
        return cnt
    except Exception:
        return 0


def slice_disagreement(path, idx):
    """Return (avg_disagreement_cp, line_count) for a chunk file.
    Uses dis_delta_queue.npy when available for exact static-vs-5k disagreement,
    falling back to abs(score) from the file.
    """
    cnt = count_lines(path)
    true_dis = chunk_true_disagreement(idx)
    if true_dis is not None:
        return true_dis, cnt
    total = 0
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        total += abs(int(parts[0]))
                        n += 1
                    except ValueError:
                        pass
    except Exception:
        return 0.0, cnt
    return (total / n if n > 0 else 0.0), cnt


def count_workers():
    """Count running titanium-cli processes reliably on Windows."""
    try:
        out = sp.run(
            ["tasklist", "/FI", "IMAGENAME eq titanium-cli.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
            stdin=sp.DEVNULL,
        ).stdout
        count = sum(1 for line in out.splitlines() if "titanium-cli" in line.lower())
        if count > 0:
            return count
    except Exception:
        pass
    try:
        out = sp.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-Process titanium-cli -ErrorAction SilentlyContinue).Count"],
            capture_output=True, text=True, timeout=5,
            stdin=sp.DEVNULL,
        ).stdout.strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0


def chunk_index_from_name(name):
    """Extract chunk index from filename like c_42.txt or c_0_000005000.txt"""
    base = name.rsplit(".", 1)[0]
    parts = base.split("_")
    try:
        if len(parts) == 2:
            return int(parts[1])
        elif len(parts) >= 3:
            row = int(parts[2])
            return row // CHUNK
        return -1
    except ValueError:
        return -1


def read_dir_manifest(d):
    man = {}
    p = os.path.join(d, "MANIFEST.txt")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.rstrip("\r\n").split("=", 1)
                        man[k] = v
        except Exception:
            pass
    return man


def snapshot():
    active_out_name = "out_dis_remaining_e3f8dc6e_e11827d1_d8a135ce"
    same_hash_skipped = 2_322_541
    exe_sha = "e3f8dc6e67e86d2a"
    net_sha = "e11827d1f9010fd0"
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                m = json.load(f)
            active_out_name = m.get("output_dir", active_out_name)
            same_hash_skipped = int(m.get("same_hash_rows_skipped", same_hash_skipped))
            exe_sha = m.get("exe_sha256", exe_sha)[:16]
            net_sha = m.get("net_sha256", net_sha)[:16]
        except Exception:
            pass

    out_dirs = [
        os.path.join(D, "out_dis"),
        os.path.join(D, "out_dis_persistent"),
    ] + sorted(glob.glob(os.path.join(D, "out_dis_remaining_*")))

    per_dir = {}
    active_labels = 0
    legacy_labels = 0
    tot_dis = 0.0
    tot_n = 0
    latest = []
    active_chunk_indices = set()
    earliest_mtime = None
    newest_mtime = 0.0

    for d in out_dirs:
        if not os.path.isdir(d):
            continue
        bname = os.path.basename(d)
        is_active = (bname == active_out_name)
        files = glob.glob(os.path.join(d, "c_*.txt"))
        n_files = len(files)
        dir_lines = 0
        dman = read_dir_manifest(d)

        for p in files:
            cnt = count_lines(p)
            dir_lines += cnt
            if is_active:
                try:
                    m = os.path.getmtime(p)
                    if earliest_mtime is None or m < earliest_mtime:
                        earliest_mtime = m
                    if m > newest_mtime:
                        newest_mtime = m
                    name = os.path.basename(p)
                    idx = chunk_index_from_name(name)
                    if idx >= 0 and cnt >= CHUNK:
                        active_chunk_indices.add(idx)
                    avg_dis, _ = slice_disagreement(p, idx)
                    if cnt > 0:
                        tot_dis += avg_dis * cnt
                        tot_n += cnt
                    latest.append((name, m, cnt, avg_dis, idx))
                except Exception:
                    pass

        if is_active:
            active_labels += dir_lines
        else:
            legacy_labels += dir_lines

        per_dir[bname] = {
            "files": n_files,
            "labels": dir_lines,
            "active": is_active,
            "exe_sha": dman.get("exe_sha", "—"),
            "net_sha": dman.get("net_sha", "—"),
        }

    # Include out_v2 in breakdown as well
    out_v2_dir = os.path.join(D, "out_v2")
    if os.path.isdir(out_v2_dir):
        v2_files = glob.glob(os.path.join(out_v2_dir, "c_*.txt"))
        v2_lines = sum(count_lines(p) for p in v2_files)
        v2_man = read_dir_manifest(out_v2_dir)
        per_dir["out_v2 (excluded from queue)"] = {
            "files": len(v2_files),
            "labels": v2_lines,
            "active": False,
            "exe_sha": v2_man.get("exe_sha", "13210c04bbcefe80"),
            "net_sha": v2_man.get("net_sha", "e11827d1f9010fd0"),
        }

    latest.sort(key=lambda x: x[1], reverse=True)
    latest = latest[:10]

    avg_disagreement = (tot_dis / tot_n) if tot_n > 0 else 0.0

    # Total relabeled across all same-net runs = 2,322,541 (already excluded when building dis_positions.bin) + active_labels
    total_relabeled_all = same_hash_skipped + active_labels

    # Contiguous head frontier: highest chunk index before the first gap from 0,
    # or the max chunk index in the PC head region (< 1200).
    total_rows = QUEUE_TOTAL
    total_chunks = (total_rows + CHUNK - 1) // CHUNK

    pc_head_chunks = [i for i in active_chunk_indices if i < 1200]
    cloud_chunks = [i for i in active_chunk_indices if i >= 1200]

    # Find all contiguous ranges of completed chunks
    sorted_active = sorted(active_chunk_indices)
    completed_ranges = []
    if sorted_active:
        s = e = sorted_active[0]
        for x in sorted_active[1:]:
            if x == e + 1:
                e = x
            else:
                completed_ranges.append((s, e))
                s = e = x
        completed_ranges.append((s, e))

    # Contiguous solid head: highest chunk index before the first gap from 0
    cur_head = 0
    while cur_head in active_chunk_indices:
        cur_head += 1
    contiguous_head_chunk = cur_head - 1  # e.g. 108 if 109 is missing
    contiguous_head_row = (contiguous_head_chunk + 1) * CHUNK if contiguous_head_chunk >= 0 else 0
    contiguous_head_pct = min(100.0, 100.0 * contiguous_head_row / total_rows) if total_rows > 0 else 0.0
    contiguous_head_dis = chunk_true_disagreement(contiguous_head_chunk) or 0.0

    # Max frontier reached across all pools (cloud/modal)
    max_chunk_reached = max(sorted_active) if sorted_active else 0
    max_row_reached = min(total_rows, (max_chunk_reached + 1) * CHUNK)
    max_pct_reached = min(100.0, 100.0 * max_row_reached / total_rows) if total_rows > 0 else 0.0
    deepest_dis = chunk_true_disagreement(max_chunk_reached) or 0.0
    tail_distance_pct = round(max(0.0, 100.0 - max_pct_reached), 2)

    # Disagreement at queue head vs tail
    head_dis = chunk_true_disagreement(0) or 1513.0
    tail_dis = chunk_true_disagreement(total_chunks - 1) or 0.0

    # Band description showing journey from Head to Tail:
    band_name = f"Head: {head_dis:.0f} cp -> Current: {contiguous_head_dis:.0f} cp -> Deepest: {deepest_dis:.0f} cp -> Tail: {tail_dis:.1f} cp"
    band_detail = (
        f"Contiguous Solid Head: chunk {contiguous_head_chunk:,} ({contiguous_head_pct:.1f}% from head, ~{contiguous_head_dis:.0f} cp) | "
        f"Deepest Cloud Reach: chunk {max_chunk_reached:,} ({max_pct_reached:.1f}% towards tail, ~{deepest_dis:.0f} cp) | "
        f"Remaining to Tail: {tail_distance_pct:.1f}%"
    )

    # Speed calculation: combine rolling window with sustained run rate so it never reads 0 between chunk completions
    now = time.time()
    _speed_ring.append((now, active_labels))
    speed_pos_s = 0.0
    if len(_speed_ring) >= 2:
        oldest_t, oldest_n = _speed_ring[0]
        dt = now - oldest_t
        dn = active_labels - oldest_n
        if dt > 10 and dn > 0:
            speed_pos_s = dn / dt

    # Baseline: 78 chunks (390,000 labels) were carried over from 13210c04 + pre-relaunch chunks.
    # Measure sustained speed from chunks written in the last 30 minutes if rolling window is cold.
    if speed_pos_s <= 0.0:
        recent_Cutoff = now - 1800.0
        recent_files = [
            m for (_, m, cnt, _, _) in latest if m >= recent_Cutoff and cnt >= CHUNK
        ]
        if len(recent_files) >= 2:
            dt_r = max(60.0, max(recent_files) - min(recent_files))
            speed_pos_s = (len(recent_files) * CHUNK) / dt_r
        else:
            speed_pos_s = 85.0  # baseline 8-core PC rate until next chunk lands

    speed_pos_hr = speed_pos_s * 3600.0
    remaining = max(0, QUEUE_TOTAL - active_labels)
    eta_hours = (remaining / speed_pos_s / 3600.0) if (speed_pos_s > 0 and remaining > 0) else None

    workers = count_workers()

    # Check ledger for stamped provenance count
    stamped_samehash = same_hash_skipped + active_labels
    if os.path.exists(LEDGER_PATH):
        try:
            with open(LEDGER_PATH, encoding="utf-8") as f:
                led = json.load(f)
            stamped_samehash = max(stamped_samehash, led.get("total_samehash_excluded_rows", stamped_samehash))
        except Exception:
            pass

    dataset_covered = same_hash_skipped + active_labels

    return {
        "dataset_total": TOTAL_DATASET,
        "queue_total": QUEUE_TOTAL,
        "labels_total": active_labels,
        "same_hash_skipped": same_hash_skipped,
        "legacy_labels_total": legacy_labels,
        "dataset_covered": dataset_covered,
        "stamped_samehash": stamped_samehash,
        "exe_sha": exe_sha,
        "net_sha": net_sha,
        "pct_of_dataset": round(100.0 * dataset_covered / TOTAL_DATASET, 2),
        "pct_of_queue": round(100.0 * active_labels / QUEUE_TOTAL, 2),
        "avg_disagreement": round(avg_disagreement, 1),
        "dirs": per_dir,
        "band_name": band_name,
        "band_detail": band_detail,
        "frontier_chunk": contiguous_head_chunk,
        "frontier_row": contiguous_head_row,
        "frontier_pct": round(contiguous_head_pct, 2),
        "frontier_dis": round(contiguous_head_dis, 1),
        "max_chunk_reached": max_chunk_reached,
        "max_row_reached": max_row_reached,
        "max_pct_reached": round(max_pct_reached, 2),
        "deepest_dis": round(deepest_dis, 1),
        "tail_distance_pct": tail_distance_pct,
        "head_dis": round(head_dis, 1),
        "tail_dis": round(tail_dis, 1),
        "completed_ranges": completed_ranges,
        "pc_chunks_done": len(pc_head_chunks),
        "cloud_chunks_done": len(cloud_chunks),
        "total_chunks": total_chunks,
        "latest_slices": [(n, cnt, round(a, 1), idx) for n, _, cnt, a, idx in latest],
        "workers": workers,
        "fishnet": coordinator.status(),
        "speed_pos_s": round(speed_pos_s, 1),
        "speed_pos_hr": int(round(speed_pos_hr)),
        "eta_hours": round(eta_hours, 1) if eta_hours is not None else None,
        "updated_at": time.strftime("%H:%M:%S"),
    }


class H(BaseHTTPRequestHandler):
    def _send_json(self, status_code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)

            if parsed.path == "/api/progress":
                self._send_json(200, snapshot())
            elif parsed.path == "/api/fishnet/claim":
                worker_id = qs.get("worker", ["anon"])[0]
                count = int(qs.get("count", ["1"])[0])
                headroom = int(qs.get("headroom", ["0"])[0])
                claimed = coordinator.claim_head(worker_id, count=count, headroom=headroom)
                self._send_json(200, {"ok": True, "chunks": claimed, "count": len(claimed)})
            elif parsed.path == "/api/fishnet/status":
                self._send_json(200, coordinator.status())
            elif parsed.path == "/" or parsed.path == "/index.html":
                html = open(os.path.join(os.path.dirname(__file__), "relabel_dashboard.html"), "r",
                            encoding="utf-8").read().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception:
            import traceback
            traceback.print_exc()
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/fishnet/submit":
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                worker_id = payload.get("worker", "anon")
                chunk_id = int(payload.get("chunk_id", -1))
                lines = payload.get("lines", "")
                ok, msg = coordinator.submit_chunk(worker_id, chunk_id, lines)
                self._send_json(200 if ok else 400, {"ok": ok, "msg": msg, "chunk_id": chunk_id})
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            self._send_json(500, {"ok": False, "error": str(e)})

    def log_message(self, *a):
        pass


def main():
    srv = ThreadedHTTPServer(("127.0.0.1", 8080), H)
    print("dashboard: http://127.0.0.1:8080", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
