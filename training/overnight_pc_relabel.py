"""Reliable overnight PC relabeler for Titanium Ataxx.

Processes positions from data/nnue/relabel/dis_positions.bin in order of highest
disagreement between static evaluation and search, labeling them with the latest
strongest main NNUE (5000 nodes/pos) strictly on local PC compute (free, zero cloud).

Chunks are 5000 rows each (c_<chunk_idx>.txt). Output is written atomically via .tmp
files, fully resumable across restarts, and tracked with a MANIFEST.json.
"""
import argparse
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = r"C:\gitProjects\titanium-ataxx"
MAIN = r"C:\gitProjects\main-ataxx"
DEFAULT_BIN = os.path.join(ROOT, "data", "nnue", "relabel", "dis_positions.bin")
DEFAULT_ROWS_MAP = os.path.join(ROOT, "data", "nnue", "relabel", "dis_rows.npy")
DEFAULT_EXE = os.path.join(MAIN, "target", "release", "titanium-cli.exe")
DEFAULT_NET = os.path.join(MAIN, "data", "nnue", "nnue_main.tnn")
LOGS_DIR = os.path.join(ROOT, "logs")
CHUNK_SIZE = 5000
RECORD_BYTES = 25


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    except OSError:
        return -1


def parse_args():
    p = argparse.ArgumentParser(description="Reliable overnight PC relabeler")
    p.add_argument("--bin", default=DEFAULT_BIN, help="Input positions binary file")
    p.add_argument("--exe", default=DEFAULT_EXE, help="Path to titanium-cli executable")
    p.add_argument("--nnue", default=DEFAULT_NET, help="Path to NNUE network file")
    p.add_argument("--out-dir", default=None, help="Output directory (auto-derived if omitted)")
    p.add_argument("--threads", type=int, default=6, help="Worker threads (default: 6, leaves 2 cores free)")
    p.add_argument("--nodes", type=int, default=5000, help="Search node limit per position")
    p.add_argument("--first-chunk", type=int, default=0, help="First chunk index to process")
    p.add_argument("--last-chunk", type=int, default=None, help="Last chunk index (inclusive, optional)")
    p.add_argument("--dry-run", action="store_true", help="Validate configuration and exit without running")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 1. Validate files
    for path, name in [(args.bin, "Input bin"), (args.exe, "Engine exe"), (args.nnue, "NNUE net")]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {name} not found at {path}")

    bin_size = os.path.getsize(args.bin)
    if bin_size % RECORD_BYTES != 0:
        sys.exit(f"ERROR: {args.bin} size ({bin_size}) is not a multiple of {RECORD_BYTES} bytes")
    total_positions = bin_size // RECORD_BYTES
    total_chunks = (total_positions + CHUNK_SIZE - 1) // CHUNK_SIZE

    print("Computing hashes...", flush=True)
    exe_sha = sha256_file(args.exe)
    net_sha = sha256_file(args.nnue)

    # Use queue hash from manifest if available to avoid re-reading 789 MB on every invocation
    manifest_source = os.path.join(ROOT, "data", "nnue", "relabel", "dis_queue_manifest.json")
    bin_sha = None
    if os.path.exists(manifest_source):
        try:
            with open(manifest_source, "r", encoding="utf-8") as f:
                data = json.load(f)
                bin_sha = data.get("queue_sha256")
        except Exception:
            bin_sha = None
    if bin_sha is None:
        bin_sha = sha256_file(args.bin)

    print(f"Engine EXE: {args.exe} ({exe_sha[:16]})")
    print(f"NNUE Net:   {args.nnue} ({net_sha[:16]})")
    print(f"Input BIN:  {args.bin} ({bin_sha[:16]}, {total_positions:,} positions, {total_chunks:,} chunks)")

    # 2. Determine output directory
    if args.out_dir is None:
        out_dirname = f"out_dis_remaining_{exe_sha[:8]}_{net_sha[:8]}_{bin_sha[:8]}"
        out_dir = os.path.join(ROOT, "data", "nnue", "relabel", out_dirname)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    manifest = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "exe_path": args.exe,
        "exe_sha256": exe_sha,
        "net_path": args.nnue,
        "net_sha256": net_sha,
        "bin_path": args.bin,
        "bin_sha256": bin_sha,
        "total_positions": total_positions,
        "chunk_size": CHUNK_SIZE,
        "nodes": args.nodes,
        "threads": args.threads,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 3. Dry run verification
    if args.dry_run:
        print("[DRY-RUN] Running 5-position smoke test...")
        test_cmd = [
            args.exe, "labelbin",
            "--bin", args.bin,
            "--lo", "0",
            "--hi", "5",
            "--nodes", str(args.nodes),
            "--tuple", "none",
            "--network", args.nnue,
            "--threads", "1",
        ]
        r = subprocess.run(test_cmd, cwd=MAIN, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"[DRY-RUN] Engine test failed with code {r.returncode}:\n{r.stderr}")
        lines = r.stdout.strip().splitlines()
        if len(lines) != 5:
            sys.exit(f"[DRY-RUN] Expected 5 lines, got {len(lines)}")
        print(f"[DRY-RUN] Smoke test PASSED. First row output: {lines[0]}")
        print("[DRY-RUN] Configuration verified successfully.")
        return

    # Graceful stop handling
    stop_requested = False
    def sig_handler(signum, frame):
        nonlocal stop_requested
        print("\nStop requested. Will exit cleanly after current chunk completes...", flush=True)
        stop_requested = True
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    first_chunk = max(0, args.first_chunk)
    last_chunk = total_chunks - 1 if args.last_chunk is None else min(total_chunks - 1, args.last_chunk)

    print(f"\nStarting relabel loop: chunks {first_chunk} to {last_chunk} ({args.threads} threads, {args.nodes} nodes/pos)...", flush=True)

    heartbeat_path = os.path.join(LOGS_DIR, "overnight_pc_relabel.heartbeat")
    start_time = time.time()
    total_labeled_this_session = 0

    for k in range(first_chunk, last_chunk + 1):
        if stop_requested:
            print("Clean stop executed.", flush=True)
            break

        lo = k * CHUNK_SIZE
        hi = min(lo + CHUNK_SIZE, total_positions)
        expected_rows = hi - lo
        chunk_file = os.path.join(out_dir, f"c_{k}.txt")

        # Skip if already complete
        if os.path.exists(chunk_file):
            if count_lines(chunk_file) == expected_rows:
                continue
            else:
                print(f"Chunk {k} ({chunk_file}) is incomplete or corrupted. Re-running...", flush=True)

        t_chunk_start = time.time()
        cmd = [
            args.exe, "labelbin",
            "--bin", args.bin,
            "--lo", str(lo),
            "--hi", str(hi),
            "--nodes", str(args.nodes),
            "--tuple", "none",
            "--network", args.nnue,
            "--threads", str(args.threads),
        ]

        tmp_file = chunk_file + ".tmp"
        with open(tmp_file, "w", encoding="ascii", newline="\n") as fo:
            proc = subprocess.run(cmd, cwd=MAIN, stdout=fo, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            print(f"ERROR: Chunk {k} failed (code {proc.returncode}):\n{proc.stderr[-300:]}", flush=True)
            time.sleep(2)
            continue

        lines_written = count_lines(tmp_file)
        if lines_written != expected_rows:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            print(f"ERROR: Chunk {k} wrote {lines_written} lines, expected {expected_rows}. Discarding.", flush=True)
            continue

        os.replace(tmp_file, chunk_file)
        chunk_duration = time.time() - t_chunk_start
        chunk_rate = expected_rows / max(0.001, chunk_duration)
        total_labeled_this_session += expected_rows
        elapsed_total = time.time() - start_time
        avg_rate = total_labeled_this_session / max(0.001, elapsed_total)

        status_msg = (
            f"[{time.strftime('%H:%M:%S')}] Chunk {k:05d}/{total_chunks} "
            f"({lo:,}..{hi:,}) done in {chunk_duration:.1f}s ({chunk_rate:.1f} pos/s) | "
            f"Session: {total_labeled_this_session:,} pos, avg {avg_rate:.1f} pos/s"
        )
        print(status_msg, flush=True)

        # Heartbeat update
        try:
            with open(heartbeat_path, "w", encoding="utf-8") as hb:
                hb.write(f"{status_msg}\n")
        except OSError:
            pass

    print(f"\nOvernight relabel session finished. Total positions labeled: {total_labeled_this_session:,}.", flush=True)


if __name__ == "__main__":
    main()
