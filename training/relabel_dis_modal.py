"""Modal relabel runner for disagreement queue (dis_positions.bin).

Takes slices of chunks starting from cloud_next_chunk.txt or specified range.
Each container receives raw 25-byte position chunks, runs the engine with
--threads 1 per core in parallel (ThreadPoolExecutor), validates lines,
and returns the chunk results directly to write to out_dis_remaining_...

Cost: ~$0.047 per core-hour ($0.0000131/core-s + $0.00000222/GiB-s).
Budget cap: stops submitting batches once budget is reached.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import time

import modal

ROOT = "C:/gitProjects/titanium-ataxx"
ENGINE_SRC = os.environ.get("ENGINE_SRC", "C:/gitProjects/main-ataxx")
D = os.path.join(ROOT, "data", "nnue", "relabel")
BIN = os.path.join(D, "dis_positions.bin")
MANIFEST_PATH = os.path.join(D, "dis_queue_manifest.json")
SPEND_CSV = os.path.join(ROOT, "logs", "modal_spend.csv")
CH = 5000
RECORD_BYTES = 25

IGNORE = [
    "target", "logs", "web", "training", "docs", "downloads", "botzone",
    "**/*.pt", "**/*.npz", "node_modules", ".git",
    "data/nnue/dis_positions.bin", "data/nnue/positions.bin", "data/nnue/positions_s*.bin",
    "data/nnue/*.npy", "data/nnue/labels.npz", "data/nnue/out_*", "data/nnue/relabel*",
]
CARGO = "/root/.cargo/bin/cargo"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .add_local_dir(ENGINE_SRC, "/eng", copy=True, ignore=IGNORE)
    .run_commands("cd /eng && %s build -p titanium-cli --release" % CARGO)
)
app = modal.App("titanium-relabel-modal", image=image)


@app.function(cpu=8.0, memory=4096, timeout=1800)
def relabel_batch(items):
    """Items: list of (chunk_idx, raw_bytes, want_rows)
    Runs 8 parallel titanium-cli processes (--threads 1) inside container.
    """
    import concurrent.futures
    import os
    import subprocess
    import tempfile

    results = []

    def process_one(item):
        chunk_idx, raw_bytes, want_rows = item
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tf.write(raw_bytes)
            tmp_bin = tf.name

        try:
            cmd = [
                "/eng/target/release/titanium-cli", "labelbin",
                "--bin", tmp_bin,
                "--lo", "0", "--hi", str(want_rows),
                "--nodes", "5000",
                "--tuple", "none",
                "--network", "bundled",
                "--threads", "1",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            valid = (r.returncode == 0 and len(lines) == want_rows)
            return chunk_idx, valid, r.stdout
        finally:
            try:
                os.remove(tmp_bin)
            except OSError:
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process_one, item) for item in items]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    return results


def load_manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    out_name = manifest["output_dir"]
    out_dir = os.path.join(D, out_name)
    os.makedirs(out_dir, exist_ok=True)
    return manifest, out_dir


def read_chunk_bytes(chunk_idx, total_rows):
    lo = chunk_idx * CH
    hi = min((chunk_idx + 1) * CH, total_rows)
    want = hi - lo
    with open(BIN, "rb") as f:
        f.seek(lo * RECORD_BYTES)
        data = f.read(want * RECORD_BYTES)
    return chunk_idx, data, want


def find_first_unlabeled_head_chunk(out_dir, start_after=0):
    """Find the lowest chunk index not yet completed in out_dir, starting at or above start_after."""
    import glob
    existing = set()
    for f in glob.glob(os.path.join(out_dir, "c_*.txt")):
        b = os.path.basename(f).split(".")[0].split("_")
        if len(b) == 2 and b[1].isdigit():
            existing.add(int(b[1]))
    cur = max(0, start_after)
    while cur in existing:
        cur += 1
    return cur


def run_modal_relabel(budget_usd=10.0, start_chunk=None, max_chunks=None):
    manifest, out_dir = load_manifest()
    total_rows = os.path.getsize(BIN) // RECORD_BYTES
    n_chunks = (total_rows + CH - 1) // CH

    # If start_chunk is not explicitly given, start strictly from the HEAD:
    # Read the PC local worker counter and give PC a 16-chunk headroom so they don't collide.
    if start_chunk is None:
        pc_cur = 0
        try:
            with open(os.path.join(D, "next_chunk.txt"), "r") as pf:
                pc_cur = int(pf.read().strip())
        except Exception:
            pass
        start_chunk = find_first_unlabeled_head_chunk(out_dir, start_after=pc_cur + 16)
        print(f"Auto-selected head chunk {start_chunk} (PC counter={pc_cur}, headroom=16)", flush=True)

    # Modal chunk range: say 800 chunks = 4,000,000 positions
    # 4M positions @ ~25 pos/s/core / 8 cores ≈ 20 pos/s/vCPU = ~2.5 hrs on 16 vCPUs ≈ ~$2.50-$4.00
    if max_chunks is None:
        # Default ~600 chunks = 3M positions ≈ $3-$4 budget
        max_chunks = int(budget_usd / 0.007)  # approx ~600-1000 chunks
        max_chunks = min(max_chunks, 1000)

    end_chunk = min(n_chunks, start_chunk + max_chunks)
    print(f"Modal relabel plan: chunks {start_chunk}..{end_chunk} ({end_chunk - start_chunk} chunks, "
          f"{(end_chunk - start_chunk)*5000:,} rows), budget=${budget_usd:.2f}", flush=True)

    # Advance cloud counter so Docker doesn't step on these chunks
    try:
        with open(os.path.join(D, "cloud_next_chunk.txt"), "w") as cf:
            cf.write(str(end_chunk))
        print(f"Advanced cloud_next_chunk.txt to {end_chunk}", flush=True)
    except Exception as e:
        print(f"Warning: could not advance cloud counter: {e}", flush=True)

    # Batch into groups of 8 chunks per container
    batches = []
    current_batch = []
    for c in range(start_chunk, end_chunk):
        out_path = os.path.join(out_dir, f"c_{c}.txt")
        if os.path.exists(out_path):
            continue
        item = read_chunk_bytes(c, total_rows)
        current_batch.append(item)
        if len(current_batch) == 8:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    print(f"Prepared {len(batches)} containers (8 chunks/container). Launching on Modal...", flush=True)
    t0 = time.time()
    done_chunks = 0
    with app.run():
        for res_list in relabel_batch.map(batches, order_outputs=False):
            for chunk_idx, valid, stdout in res_list:
                if not valid:
                    print(f"Chunk {chunk_idx} failed validation on Modal!", flush=True)
                    continue
                out_path = os.path.join(out_dir, f"c_{chunk_idx}.txt")
                tmp_path = out_path + ".tmp"
                with open(tmp_path, "w", encoding="ascii") as of:
                    of.write(stdout)
                os.replace(tmp_path, out_path)
                done_chunks += 1
                if done_chunks % 16 == 0:
                    dt = time.time() - t0
                    rows = done_chunks * CH
                    rate = rows / dt if dt > 0 else 0
                    print(f"Modal: {done_chunks} chunks done ({rows:,} rows, {dt:.0f}s, {rate:.1f} pos/s)", flush=True)

    dt = time.time() - t0
    # Stamp provenance
    try:
        import relabel_dis_cloud
        relabel_dis_cloud.stamp_provenance()
    except Exception:
        pass

    est_cost = (dt / 3600.0) * (len(batches) * 8 * 0.047)  # upper bound
    print(f"Modal relabel complete: {done_chunks} chunks ({done_chunks*CH:,} rows) in {dt:.0f}s. Est cost: ~${est_cost:.2f}", flush=True)
    try:
        with open(SPEND_CSV, "a", encoding="utf-8") as sf:
            sf.write(f"{time.strftime('%Y-%m-%d %H:%M')},modal-batch,8,4,0.38,{int(dt/60)},{est_cost:.2f},chunks {start_chunk}-{end_chunk}\n")
    except Exception:
        pass


@app.local_entrypoint()
def main(budget: float = 8.0, start_chunk: int = None, max_chunks: int = None):
    run_modal_relabel(budget_usd=budget, start_chunk=start_chunk, max_chunks=max_chunks)


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    count = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_modal_relabel(budget_usd=budget, start_chunk=start, max_chunks=count)
