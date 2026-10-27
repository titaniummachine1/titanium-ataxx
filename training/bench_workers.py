#!/usr/bin/env python3
"""Clean benchmark: runs N labelbin workers concurrently for fixed duration,
measures pos/s/core to find optimal worker count."""
import glob
import msvcrt
import os
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(ROOT), "data", "nnue", "relabel")
BIN = D + "/dis_positions.bin"
OUT = D + "/out_dis_bench"
COUNTER = D + "/next_row_bench.txt"
EXE = r"C:\gitProjects\main-ataxx\target\release\titanium-cli.exe"
NODES = 5000
CHUNK = 5000


def claim_next():
    os.makedirs(os.path.dirname(COUNTER), exist_ok=True)
    if not os.path.exists(COUNTER):
        with open(COUNTER, "w") as f:
            f.write("0")
    total = os.path.getsize(BIN) // 25
    with open(COUNTER, "r+b") as f:
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        try:
            f.seek(0)
            raw = f.read().strip()
            lo = int(raw) if raw else 0
            if lo >= total:
                return None
            hi = min(lo + CHUNK, total)
            f.seek(0)
            f.truncate()
            f.write(str(hi).encode())
            return lo, hi
        finally:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def worker(wid, stop_time, result):
    local_rows = 0
    local_chunks = 0
    while time.time() < stop_time:
        rng = claim_next()
        if rng is None:
            break
        lo, hi = rng
        out = f"{OUT}/c_{wid}_{lo:09d}.txt"
        cmd = [EXE, "labelbin", "--bin", BIN, "--lo", str(lo), "--hi", str(hi),
               "--nodes", str(NODES), "--tuple", "none", "--network", "bundled",
               "--threads", "1"]
        tmp = out + ".tmp"
        # create empty tmp file, then close handle before subprocess uses it
        with open(tmp, "wb") as _f:
            pass
        with open(tmp, "wb") as of:
            p = subprocess.Popen(cmd, stdout=of, stderr=subprocess.PIPE, text=True)
            _, err = p.communicate()
        if p.returncode == 0:
            os.replace(tmp, out)
            local_rows += hi - lo
            local_chunks += 1
        else:
            print(f"worker {wid}: chunk {lo}..{hi} FAILED rc={p.returncode}", flush=True)
            with open(COUNTER, "r+b") as f:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    f.seek(0)
                    cur = int(f.read().strip() or "0")
                    f.seek(0)
                    f.truncate()
                    f.write(str(max(lo, cur - (hi - lo))).encode())
                finally:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    result["rows"] = local_rows
    result["chunks"] = local_chunks


def run_benchmark(workers, duration):
    os.makedirs(OUT, exist_ok=True)
    # clean old bench files
    for f in glob.glob(OUT + "/c_*.txt") + glob.glob(OUT + "/c_*.tmp"):
        try:
            os.remove(f)
        except Exception:
            pass
    if os.path.exists(COUNTER):
        os.remove(COUNTER)
    results = {}
    stop_time = time.time() + duration
    threads = []
    for w in range(workers):
        t = threading.Thread(target=worker, args=(w, stop_time, results))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    total_rows = sum(r.get("rows", 0) for r in results.values())
    total_chunks = sum(r.get("chunks", 0) for r in results.values())
    elapsed = duration
    pos_per_sec = total_rows / elapsed if elapsed > 0 else 0
    pos_per_sec_per_core = pos_per_sec / workers if workers > 0 else 0
    print(f"\n{'='*50}")
    print(f"WORKERS: {workers}, DURATION: {duration}s")
    print(f"Total rows: {total_rows:,} ({total_chunks} chunks)")
    print(f"POS/SEC total: {pos_per_sec:.0f}")
    print(f"POS/SEC/core:  {pos_per_sec_per_core:.0f}")
    print(f"{'='*50}\n")
    return pos_per_sec, pos_per_sec_per_core


if __name__ == "__main__":
    import argparse
    import glob as _glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--duration", type=int, default=60)
    a = ap.parse_args()
    run_benchmark(a.workers, a.duration)
