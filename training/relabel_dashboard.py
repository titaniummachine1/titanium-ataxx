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
import time
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


# Rolling speed tracker: ring buffer of (timestamp, active_labels)
_speed_ring = collections.deque(maxlen=120)  # ~6 minutes at 3s polling
_run_start_info = None  # (earliest_chunk_mtime, baseline_labels)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
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

    latest.sort(key=lambda x: x[1], reverse=True)
    latest = latest[:10]

    avg_disagreement = (tot_dis / tot_n) if tot_n > 0 else 0.0

    # Contiguous head frontier: highest chunk index before the first gap from 0,
    # or the max chunk index in the PC head region (< 1200).
    total_rows = QUEUE_TOTAL
    total_chunks = (total_rows + CHUNK - 1) // CHUNK

    pc_head_chunks = [i for i in active_chunk_indices if i < 1200]
    cloud_chunks = [i for i in active_chunk_indices if i >= 1200]

    frontier_chunk = max(pc_head_chunks) if pc_head_chunks else (max(active_chunk_indices) if active_chunk_indices else -1)
    frontier_row = (frontier_chunk + 1) * CHUNK if frontier_chunk >= 0 else 0
    frontier_pct = min(100.0, 100.0 * frontier_row / total_rows) if total_rows > 0 else 0.0

    # True disagreement at frontier from dis_delta_queue.npy
    frontier_dis = chunk_true_disagreement(frontier_chunk) or (latest[0][3] if latest else 0.0)

    # Band based on true disagreement percentiles of dis_delta_queue.npy:
    # 0-2%: >500 cp (HEAD), 2-10%: 250-500 cp (UPPER HEAD), 10-25%: 150-250 cp (UPPER MIDDLE),
    # 25-50%: 80-150 cp (MIDDLE), 50-75%: 35-80 cp (LOWER MIDDLE), 75-90%: 14-35 cp (LOWER), 90-100%: <14 cp (TAIL)
    band_name = "not started"
    band_detail = ""
    if frontier_chunk >= 0:
        frac = frontier_row / max(1, total_rows)
        if frac < 0.02:
            band_name = "EXTREME HEAD (Top 2%)"
            band_detail = f"wildest disagreement — static vs 5k gap ~500–2400 cp (now ~{frontier_dis:.0f} cp)"
        elif frac < 0.10:
            band_name = "UPPER HEAD (Top 2–10%)"
            band_detail = f"very high disagreement — gap ~250–500 cp (now ~{frontier_dis:.0f} cp)"
        elif frac < 0.25:
            band_name = "UPPER MIDDLE (10–25%)"
            band_detail = f"high disagreement — gap ~150–250 cp (now ~{frontier_dis:.0f} cp)"
        elif frac < 0.50:
            band_name = "MIDDLE (25–50%)"
            band_detail = f"moderate disagreement — gap ~80–150 cp (now ~{frontier_dis:.0f} cp)"
        elif frac < 0.75:
            band_name = "LOWER MIDDLE (50–75%)"
            band_detail = f"low disagreement — gap ~35–80 cp (now ~{frontier_dis:.0f} cp)"
        elif frac < 0.90:
            band_name = "LOWER (75–90%)"
            band_detail = f"minor disagreement — gap ~14–35 cp (now ~{frontier_dis:.0f} cp)"
        else:
            band_name = "TAIL (90–100%)"
            band_detail = f"near-agreement — static eval already within ~{frontier_dis:.0f} cp of search"

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

    # Fallback / sustained rate from current active run if rolling window is between chunk writes
    if speed_pos_s <= 0.0 and earliest_mtime is not None and active_labels > 0:
        # Chunks 0..7 were written around 19:03; estimate elapsed since first chunk minus ~420s first-chunk time
        elapsed = max(60.0, now - (earliest_mtime - 400.0))
        speed_pos_s = active_labels / elapsed

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
            stamped_samehash = led.get("total_samehash_excluded_rows", stamped_samehash)
        except Exception:
            pass

    dataset_covered = same_hash_skipped + active_labels

    return {
        "dataset_total": TOTAL_DATASET,
        "queue_total": QUEUE_TOTAL,
        "labels_total": active_labels,
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
        "frontier_chunk": frontier_chunk,
        "frontier_row": frontier_row,
        "frontier_pct": round(frontier_pct, 2),
        "frontier_dis": round(frontier_dis, 1),
        "pc_chunks_done": len(pc_head_chunks),
        "cloud_chunks_done": len(cloud_chunks),
        "total_chunks": total_chunks,
        "latest_slices": [(n, cnt, round(a, 1), idx) for n, _, cnt, a, idx in latest],
        "workers": workers,
        "speed_pos_s": round(speed_pos_s, 1),
        "speed_pos_hr": int(round(speed_pos_hr)),
        "eta_hours": round(eta_hours, 1) if eta_hours is not None else None,
        "updated_at": time.strftime("%H:%M:%S"),
    }


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == "/api/progress":
                data = snapshot()
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/" or self.path == "/index.html":
                html = open(os.path.join(os.path.dirname(__file__), "relabel_dashboard.html"), "r",
                            encoding="utf-8").read().encode()
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

    def log_message(self, *a):
        pass


def main():
    srv = ThreadedHTTPServer(("127.0.0.1", 8080), H)
    print("dashboard: http://127.0.0.1:8080", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
