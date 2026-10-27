#!/usr/bin/env python3
"""Parallel per-core engine worker for disagreement-queue relabelling.

Spawns as a standalone process (one per CPU core).  All workers share an
atomic chunk counter via msvcrt file-locking so rows are never double-
labelled.  Output goes to the manifest's output_dir, in the same naming
convention as relabel_overnight.py (c_{chunk_idx}.txt) so progress is
fully visible to the dashboard and restarts seamlessly skip done chunks.

Usage (8-core machine, one terminal per core):
    python training/run_engines_only.py 0
    python training/run_engines_only.py 1
    ...
    python training/run_engines_only.py 7

Or launch all from a single shell:
    python training/run_engines_only.py --all --workers 8
"""
import argparse
import hashlib
import json
import msvcrt
import multiprocessing
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(ROOT), "data", "nnue", "relabel")
BIN = os.path.join(D, "dis_positions.bin")
EXE = r"C:\gitProjects\main-ataxx\target\release\titanium-cli.exe"
MANIFEST_PATH = os.path.join(D, "dis_queue_manifest.json")
COUNTER = os.path.join(D, "next_chunk.txt")
NODES = 5000
CHUNK = 5000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    """Load and validate the queue manifest.  Returns (manifest_dict, out_dir)."""
    assert os.path.exists(MANIFEST_PATH), f"manifest not found: {MANIFEST_PATH}"

    lock_path = os.path.join(D, "dis_queue_rebuild.lock")
    assert not os.path.exists(lock_path), "disagreement queue is being rebuilt; refusing to label"

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    exe_sha = sha256_file(EXE)
    manifest_exe = manifest.get("exe_sha256", "")
    assert exe_sha == manifest_exe, (
        f"exe sha mismatch:\n  running: {exe_sha[:16]}\n  manifest: {manifest_exe[:16]}"
    )

    net_sha_manifest = manifest.get("net_sha256", "")
    # net is bundled inside the exe — the manifest records the net sha that
    # was embedded at build time, so we trust the exe sha match above.

    out_name = manifest.get("output_dir", "")
    assert out_name.startswith("out_dis_remaining_"), f"bad output_dir in manifest: {out_name!r}"
    assert "/" not in out_name and "\\" not in out_name, f"unsafe output_dir: {out_name!r}"

    out_dir = os.path.join(D, out_name)
    os.makedirs(out_dir, exist_ok=True)
    return manifest, out_dir


def total_chunks():
    total_rows = os.path.getsize(BIN) // 25
    return (total_rows + CHUNK - 1) // CHUNK


def claim_next_chunk():
    """Atomically claim the next chunk index.  Returns chunk_idx or None."""
    n_chunks = total_chunks()
    os.makedirs(os.path.dirname(COUNTER), exist_ok=True)
    if not os.path.exists(COUNTER):
        with open(COUNTER, "w") as f:
            f.write("0")

    with open(COUNTER, "r+b") as f:
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        try:
            f.seek(0)
            raw = f.read().strip()
            idx = int(raw) if raw else 0
            if idx >= n_chunks:
                return None
            f.seek(0)
            f.truncate()
            f.write(str(idx + 1).encode())
            return idx
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def chunk_is_valid(path, want_lines):
    """Check if a chunk file exists and has the correct number of valid lines."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="ascii") as f:
            count = 0
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    return False
                int(parts[0])
                int(parts[1])
                int(parts[2])
                count += 1
                if count > want_lines:
                    return False
            return count == want_lines
    except (OSError, UnicodeError, ValueError):
        return False


def worker(wid, out_dir):
    """Main worker loop: claim chunks, run labelbin, write output."""
    total_rows = os.path.getsize(BIN) // 25
    n_chunks = total_chunks()
    done = 0
    skipped = 0
    rows = 0
    t0 = time.time()

    while True:
        idx = claim_next_chunk()
        if idx is None:
            break

        lo = idx * CHUNK
        hi = min((idx + 1) * CHUNK, total_rows)
        want = hi - lo
        out_path = os.path.join(out_dir, f"c_{idx}.txt")

        # Skip if already done (crash-resume)
        if chunk_is_valid(out_path, want):
            skipped += 1
            continue

        # Invalidate partial/corrupt file if present
        if os.path.exists(out_path):
            bad = out_path + f".invalid.{int(time.time())}"
            os.replace(out_path, bad)
            print(f"w{wid}: renamed corrupt {os.path.basename(out_path)} -> {os.path.basename(bad)}", flush=True)

        cmd = [
            EXE, "labelbin", "--bin", BIN,
            "--lo", str(lo), "--hi", str(hi),
            "--nodes", str(NODES),
            "--tuple", "none", "--network", "bundled",
            "--threads", "1",
        ]

        tmp = out_path + ".tmp"
        try:
            with open(tmp, "w") as of:
                result = subprocess.run(cmd, stdout=of, stderr=subprocess.PIPE, text=True)
        except Exception as e:
            print(f"w{wid}: chunk {idx} (rows {lo}..{hi}) EXCEPTION: {e}", flush=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            time.sleep(5)
            continue

        if result.returncode != 0:
            print(f"w{wid}: chunk {idx} (rows {lo}..{hi}) FAILED rc={result.returncode} "
                  f"stderr={result.stderr[-300:]}", flush=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            time.sleep(5)
            continue

        # Validate before promoting
        if not chunk_is_valid(tmp, want):
            line_count = 0
            try:
                with open(tmp, "r") as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                pass
            print(f"w{wid}: chunk {idx} validation failed (got {line_count} lines, want {want})", flush=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            time.sleep(5)
            continue

        os.replace(tmp, out_path)
        done += 1
        rows += want
        dt = time.time() - t0
        rate = rows / dt if dt > 0 else 0
        print(f"w{wid}: chunk {idx} done ({lo}..{hi}) | {done} done, {skipped} skipped, "
              f"{rows} rows in {dt:.0f}s = {rate:.1f} pos/s", flush=True)

    print(f"w{wid}: finished — {done} chunks done, {skipped} skipped, {rows} rows total", flush=True)


def run_worker(wid, out_dir):
    """Entry point for multiprocessing.Process."""
    try:
        worker(wid, out_dir)
    except KeyboardInterrupt:
        print(f"w{wid}: interrupted", flush=True)
    except Exception as e:
        print(f"w{wid}: fatal error: {e}", flush=True)
        raise


def main():
    ap = argparse.ArgumentParser(description="Per-core relabel worker")
    ap.add_argument("worker_id", nargs="?", type=int, default=None,
                    help="Worker ID (0-based) when running one worker per terminal")
    ap.add_argument("--all", action="store_true",
                    help="Launch all workers from this single process")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of workers (only with --all)")
    args = ap.parse_args()

    manifest, out_dir = load_manifest()
    total_rows = os.path.getsize(BIN) // 25
    n_chunks = total_chunks()
    exe_sha_short = manifest["exe_sha256"][:16]
    net_sha_short = manifest["net_sha256"][:16]
    print(f"relabel: {total_rows} positions, {n_chunks} chunks, "
          f"exe {exe_sha_short}, net {net_sha_short}", flush=True)
    print(f"output -> {out_dir}", flush=True)

    if args.all:
        n = args.workers
        print(f"launching {n} workers...", flush=True)
        procs = []
        for wid in range(n):
            p = multiprocessing.Process(target=run_worker, args=(wid, out_dir), daemon=False)
            p.start()
            procs.append(p)
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            print("\ninterrupted — waiting for workers to finish current chunk...", flush=True)
            for p in procs:
                p.join(timeout=600)
        print("all workers finished.", flush=True)
    else:
        wid = args.worker_id if args.worker_id is not None else 0
        run_worker(wid, out_dir)


if __name__ == "__main__":
    main()
