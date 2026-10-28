"""Autonomous Cloud Sandbox Watcher, Collector, and Auto-Shutdown Daemon.

Monitors tit-d1, tit-d2, tit-d3:
1. Periodically syncs completed chunks from /tmp/job/out to the coordinator output dir.
2. Resilient against local WiFi / network drops (does not terminate on transient errors).
3. When a sandbox finishes its slice (status.txt == 'DONE'), performs a final collect and
   calls `sbx --cloud rm --force <sandbox>` to immediately shut down and delete the container.
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

SANDBOXES = ["tit-d1", "tit-d2", "tit-d3"]
LOG_FILE = os.path.join(ROOT, "logs", "cloud_autonomous_watcher.log")
OUT_DIR = os.path.join(ROOT, "data", "nnue", "relabel", "out_dis_remaining_e3f8dc6e_e11827d1_d8a135ce")

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

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

    # If all 5 attempts failed due to network error, assume remaining sandboxes are still running
    log("Network unreachable after 5 attempts; keeping remaining sandboxes as active.")
    return candidates

def main():
    log("=== Cloud Autonomous Watcher Started ===")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    poll_interval = 120 # 2 minutes
    iteration = 0
    deleted_sandboxes = set()
    
    while True:
        iteration += 1
        active = get_active_sandboxes(deleted_sandboxes)
        log(f"Pass #{iteration}: Active cloud sandboxes: {active} (deleted: {list(deleted_sandboxes)})")
        
        if not active and len(deleted_sandboxes) == len(SANDBOXES):
            log("All cloud sandboxes have completed and been shut down!")
            break
            
        for name in active:
            try:
                # 1. Check status
                status = rdc.remote(name, "cat /tmp/job/status.txt")
                chunk_cnt = rdc.remote(name, "ls /tmp/job/out 2>/dev/null | grep -c '\\.txt$'")
                log(f"[{name}] status={status}, completed_chunks={chunk_cnt}")
                
                # 2. Collect chunks
                try:
                    c_args = argparse.Namespace(name=name, keep=True)
                    rdc.collect(c_args)
                except Exception as ce:
                    log(f"[{name}] collect warning: {ce}")
                
                # 3. If finished, final collect and delete container
                if status == "DONE":
                    log(f"[{name}] HAS FINISHED ITS SLICE! Running final collect and deleting container...")
                    try:
                        rdc.collect(argparse.Namespace(name=name, keep=False))
                    except Exception:
                        pass
                    
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
