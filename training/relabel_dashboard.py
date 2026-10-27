"""Relabel dashboard — tracks disagreement-queue progress, speed, and band.

Run:  python training/relabel_dashboard.py
Open: http://localhost:8080

Shows:
  - Total labels done vs queue size
  - Frontier chunk (how far into the disagreement-ranked queue)
  - Current disagreement band (head = biggest disagreement → tail = smallest)
  - Live speed (pos/sec) with rolling average
  - Active worker count (titanium-cli processes)
  - Per-directory breakdown
  - Latest chunk slices with avg disagreement
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

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(ROOT), "data", "nnue", "relabel")
TOTAL_DATASET = 33_891_139   # all positions in positions.bin / labels.npz
QUEUE_TOTAL = 31_568_598     # eligible rows in dis_positions.bin (from manifest)
CHUNK = 5000                 # rows per chunk

# Scan these directories for c_*.txt output files
OUT_DIRS = [
    os.path.join(D, "out_dis"),
    os.path.join(D, "out_dis_persistent"),
] + sorted(glob.glob(os.path.join(D, "out_dis_remaining_*")))

# Rolling speed tracker: ring buffer of (timestamp, total_labels)
_speed_ring = collections.deque(maxlen=60)  # ~3 minutes at 3s polling


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def slice_disagreement(path):
    """Return (avg_abs_score, count) for a chunk file.
    Column layout: score depth nodes [disagreement]
    Column 3 (disagreement) = |static eval - search score|, but may not exist
    in older formats. Fall back to abs(score) in column 0.
    """
    total = 0
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    # prefer disagreement column (col 3)
                    try:
                        total += abs(int(parts[3]))
                        n += 1
                        continue
                    except ValueError:
                        pass
                if len(parts) >= 1:
                    try:
                        total += abs(int(parts[0]))
                    except ValueError:
                        pass
                    n += 1
    except Exception:
        return 0, 0
    if n == 0:
        return 0, 0
    return total / n, n


def count_workers():
    """Count running titanium-cli processes."""
    try:
        out = sp.run(["tasklist", "/FI", "IMAGENAME eq titanium-cli.exe", "/NH"],
                     capture_output=True, text=True, timeout=5).stdout
        count = sum(1 for line in out.splitlines() if "titanium-cli" in line)
        return count
    except Exception:
        return 0


def chunk_index_from_name(name):
    """Extract chunk index from filename like c_42.txt or c_0_000005000.txt"""
    base = name.rsplit(".", 1)[0]  # strip .txt
    parts = base.split("_")
    # c_{idx}.txt → parts = ["c", "42"]
    # c_{wid}_{lo}.txt → parts = ["c", "0", "000005000"]
    try:
        if len(parts) == 2:
            return int(parts[1])
        elif len(parts) >= 3:
            # row-based naming from old format: convert row to chunk index
            row = int(parts[2])
            return row // CHUNK
        return -1
    except ValueError:
        return -1


def snapshot():
    per_dir = {}
    total_labels = 0
    tot_dis = 0.0
    tot_n = 0
    latest = []
    all_chunk_indices = []
    frontier_chunk = -1
    frontier_dis = 0.0

    for d in OUT_DIRS:
        if not os.path.isdir(d):
            continue
        files = glob.glob(os.path.join(d, "c_*.txt"))
        n_files = len(files)
        lines = sum(count_lines(p) for p in files)
        total_labels += lines
        per_dir[os.path.basename(d)] = {
            "files": n_files,
            "labels": lines,
        }
        for p in files:
            try:
                m = os.path.getmtime(p)
                avg_dis, cnt = slice_disagreement(p)
                name = os.path.basename(p)
                idx = chunk_index_from_name(name)
                if idx >= 0:
                    all_chunk_indices.append(idx)
                if cnt > 0:
                    tot_dis += avg_dis * cnt
                    tot_n += cnt
                latest.append((name, m, cnt, avg_dis, idx))
            except Exception:
                pass

    latest.sort(key=lambda x: x[1], reverse=True)
    latest = latest[:10]

    avg_disagreement = (tot_dis / tot_n) if tot_n > 0 else 0.0

    # Frontier: highest chunk index completed → tells us how far into the queue
    total_rows = QUEUE_TOTAL
    total_chunks = (total_rows + CHUNK - 1) // CHUNK
    if all_chunk_indices:
        frontier_chunk = max(all_chunk_indices)
    frontier_row = (frontier_chunk + 1) * CHUNK if frontier_chunk >= 0 else 0
    frontier_pct = min(100.0, 100.0 * frontier_row / total_rows) if total_rows > 0 else 0.0

    # Band: based on frontier position in the disagreement-ranked queue
    # The queue is sorted descending by |static_eval - 5k_search_score|
    band_name = "not started"
    band_detail = ""
    if frontier_chunk >= 0:
        frac = frontier_row / max(1, total_rows)
        if frac < 0.05:
            band_name = "HEAD"
            band_detail = "highest disagreement — static eval wildly wrong (~800–2000+ cp off)"
        elif frac < 0.15:
            band_name = "UPPER HEAD"
            band_detail = "very high disagreement (~400–900 cp off)"
        elif frac < 0.30:
            band_name = "UPPER MIDDLE"
            band_detail = "high disagreement (~200–500 cp off)"
        elif frac < 0.50:
            band_name = "MIDDLE"
            band_detail = "moderate disagreement (~80–250 cp off)"
        elif frac < 0.70:
            band_name = "LOWER MIDDLE"
            band_detail = "low disagreement (~30–100 cp off)"
        elif frac < 0.90:
            band_name = "LOWER"
            band_detail = "minor disagreement (~10–40 cp off)"
        else:
            band_name = "TAIL"
            band_detail = "near-agreement — static eval already close to search"

    # Most recent chunk's actual disagreement
    if latest:
        frontier_dis = latest[0][3]

    # Speed tracking
    now = time.time()
    _speed_ring.append((now, total_labels))
    speed_pos_s = 0.0
    eta_hours = None
    if len(_speed_ring) >= 2:
        oldest_t, oldest_n = _speed_ring[0]
        dt = now - oldest_t
        dn = total_labels - oldest_n
        if dt > 0 and dn >= 0:
            speed_pos_s = dn / dt
            remaining = QUEUE_TOTAL - total_labels
            if speed_pos_s > 0 and remaining > 0:
                eta_hours = remaining / speed_pos_s / 3600.0

    # Worker count
    workers = count_workers()

    return {
        "dataset_total": TOTAL_DATASET,
        "queue_total": QUEUE_TOTAL,
        "labels_total": total_labels,
        "pct_of_dataset": round(100.0 * total_labels / TOTAL_DATASET, 2),
        "pct_of_queue": round(100.0 * total_labels / QUEUE_TOTAL, 2),
        "avg_disagreement": round(avg_disagreement, 1),
        "dirs": per_dir,
        "band_name": band_name,
        "band_detail": band_detail,
        "frontier_chunk": frontier_chunk,
        "frontier_row": frontier_row,
        "frontier_pct": round(frontier_pct, 2),
        "frontier_dis": round(frontier_dis, 1),
        "total_chunks": total_chunks,
        "latest_slices": [(n, cnt, round(a, 1), idx) for n, _, cnt, a, idx in latest],
        "workers": workers,
        "speed_pos_s": round(speed_pos_s, 1),
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
        except Exception as e:
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
