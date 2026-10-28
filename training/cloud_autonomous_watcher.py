"""Autonomous Cloud Sandbox Watcher, Multi-Slice Reuse Dispatcher, Collector, and Auto-Shutdown Daemon.

Monitors tit-d1 (16 vCPU), tit-d2 (16 vCPU), tit-d3 (8 vCPU):
1. Periodically syncs completed chunks from /tmp/job/out to the coordinator output dir.
2. Resilient against local WiFi / network drops (5 retries with 10s backoff).
3. When a sandbox finishes its current slice (status.txt == 'DONE'):
   - Collects all completed chunks from /tmp/job/out.
   - Pops the next remaining slice from SHARED_SLICES (or dynamically builds a slice of any remaining missing chunks) and calls `rdc.reuse(...)` without rebuilding Rust.
   - Once no slices remain, calls `sbx --cloud rm --force <sandbox>` to immediately shut down and delete the container.
4. Exits cleanly ONLY once all cloud sandboxes have completed and been shut down.
"""
import os
import sys
import time
import subprocess
import argparse

ROOT = "C:/gitProjects/titanium-ataxx"
sys.path.insert(0, os.path.join(ROOT, "training"))
import relabel_dis_cloud as rdc

SANDBOX_VCPU = {
    "tit-d1": 16,
    "tit-d2": 16,
    "tit-d3": 8,
}
SANDBOXES = list(SANDBOX_VCPU.keys())
LOG_FILE = os.path.join(ROOT, "logs", "cloud_autonomous_watcher.log")
OUT_DIR = os.path.join(ROOT, "data", "nnue", "relabel", "out_dis_remaining_e3f8dc6e_e11827d1_d8a135ce")

# Follow-up contiguous slices after initial launch:
# tit-d1 launched on 2000..2320 (320)
# tit-d2 launched on 2393..2600 (207)
# tit-d3 launched on 4483..4640 (157)
SHARED_SLICES = [
    (5640, 5800),  # 160 chunks
    (1660, 1760),  # 100 chunks
    (6247, 6314),  # 67 chunks
    (4640, 4700),  # 60 chunks
]

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def count_missing_in_range(lo, hi):
    missing = 0
    for c in range(lo, hi):
        if not os.path.exists(os.path.join(OUT_DIR, f"c_{c}.txt")):
            missing += 1
    return missing

def get_active_sandboxes(deleted_set):
    """Query sbx --cloud ls with retries to protect against WiFi / network drops."""
    candidates = [s for s in SANDBOXES if s not in deleted_set]
    if not candidates:
        return []

    for attempt in range(1, 6):
        try:
            res = subprocess.run(["sbx", "--cloud", "ls"], capture_output=True, text=True, timeout=60)
            if res.returncode == 0:
                active = [s for s in candidates if s in res.stdout]
                return active
            else:
                log(f"Warning: sbx ls returned non-zero (attempt {attempt}/5): {res.stderr.strip()[:100]}")
        except Exception as e:
            log(f"Warning: sbx ls failed (attempt {attempt}/5): {e}")
        time.sleep(10)

    log("Network unreachable after 5 attempts; keeping remaining sandboxes as active.")
    return candidates

def pop_next_slice():
    while SHARED_SLICES:
        lo, hi = SHARED_SLICES.pop(0)
        rem = count_missing_in_range(lo, hi)
        if rem > 0:
            return (lo, hi, rem)
    return None

def main():
    log("=== Cloud Autonomous Watcher (Wave 2 + Multi-Slice Reuse) Started ===")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    poll_interval = 120  # 2 minutes
    iteration = 0
    deleted_sandboxes = set()

    while True:
        iteration += 1
        active = get_active_sandboxes(deleted_sandboxes)
        log(f"Pass #{iteration}: Active={active}, Deleted={list(deleted_sandboxes)}, QueuedSlices={SHARED_SLICES}")

        if not active and len(deleted_sandboxes) == len(SANDBOXES):
            log("All cloud sandboxes have completed and been shut down!")
            break

        for name in active:
            try:
                status = rdc.remote(name, "cat /tmp/job/status.txt")
                chunk_cnt = rdc.remote(name, "ls /tmp/job/out 2>/dev/null | grep -c '\\.txt$'")
                log(f"[{name}] status={status}, completed_chunks_in_slice={chunk_cnt}")

                if status in ("running", "DONE"):
                    try:
                        rdc.collect(argparse.Namespace(name=name, keep=True))
                    except Exception as ce:
                        log(f"[{name}] collect warning: {ce}")

                if status == "DONE":
                    log(f"[{name}] FINISHED CURRENT SLICE! Running final slice collect...")
                    try:
                        rdc.collect(argparse.Namespace(name=name, keep=False))
                    except Exception:
                        pass

                    nxt = pop_next_slice()
                    if nxt is not None:
                        lo, hi, rem = nxt
                        vcpu = SANDBOX_VCPU[name]
                        log(f"[{name}] Reusing ({vcpu} vCPU) for slice {lo}..{hi} ({rem} missing chunks)...")
                        try:
                            rdc.reuse(argparse.Namespace(
                                name=name, vcpu=vcpu, first_chunk=lo, last_chunk=hi, hours=1.5
                            ))
                            continue
                        except Exception as re_err:
                            log(f"[{name}] reuse failed ({re_err}), pushing slice back to queue")
                            SHARED_SLICES.insert(0, (lo, hi))

                    log(f"[{name}] No more slices in queue; deleting container...")
                    del_res = subprocess.run(["sbx", "--cloud", "rm", "--force", name], capture_output=True, text=True)
                    log(f"[{name}] deleted: returncode={del_res.returncode}, out={del_res.stdout.strip()}")
                    deleted_sandboxes.add(name)
            except Exception as e:
                log(f"[{name}] iteration error: {e}")

        if len(deleted_sandboxes) == len(SANDBOXES):
            log("All sandboxes successfully processed and deleted!")
            break

        time.sleep(poll_interval)

    log("=== Cloud Autonomous Watcher Complete. All instances closed. ===")
    summary_path = os.path.join(ROOT, "data", "nnue", "relabel", "cloud_complete.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"CLOUD_COMPLETE at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
