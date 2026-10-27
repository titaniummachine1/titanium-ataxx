"""Docker cloud sandbox relabeler + budget-capped watcher for dis_positions.bin.

One sandbox = one slice of whole 5000-row chunks. Inside the sandbox: one single-thread
labelbin process per vCPU (xargs -P), never one multi-threaded process.
Output files are c_<global chunk index>.txt, identical to PC/Modal workers, and every
collected chunk is stamped into dis_samehash_rows.bin + dis_labeled_ledger.json so any
future queue rebuild or restart knows the exact exe_sha + net_sha provenance per row.

Commands:
  launch:  python training/relabel_dis_cloud.py launch --name tit-d1 --vcpu 16 --first-chunk 1200 --last-chunk 1760 --hours 4
  reuse:   python training/relabel_dis_cloud.py reuse --name tit-d1 --vcpu 16 --first-chunk 2600 --last-chunk 3050 --hours 2.5
  status:  python training/relabel_dis_cloud.py status --name tit-d1
  collect: python training/relabel_dis_cloud.py collect --name tit-d1 [--keep]
  stamp:   python training/relabel_dis_cloud.py stamp
  watch:   python training/relabel_dis_cloud.py watch --budget-usd 15.0
"""
import argparse
import glob
import json
import msvcrt
import os
import re
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT = "C:/gitProjects/titanium-ataxx"
MAIN = "C:/gitProjects/main-ataxx"
D = ROOT + "/data/nnue/relabel"
BIN = D + "/dis_positions.bin"
MANIFEST = D + "/dis_queue_manifest.json"
CLOUD_COUNTER = D + "/cloud_next_chunk.txt"
SAMEHASH_BIN = D + "/dis_samehash_rows.bin"
LEDGER_JSON = D + "/dis_labeled_ledger.json"
DIS_ROWS = D + "/dis_rows.npy"
SPEND_CSV = ROOT + "/logs/docker_spend.csv"
WATCH_LOG = D + "/watch_dis_cloud.log"
CH = 5000
RECORD_BYTES = 25
VERIFY_ROWS = 100
VERIFY_MIN_IDENTICAL = 95
IMAGE = "docker.io/library/rust:1.97"
PRICE_PER_VCPU_HR = 0.07

JOB_SH = """#!/bin/bash
cd /tmp/job
tar xf cand.tar
(cd cand && cargo build -p titanium-cli --release -q 2> ../build.err)
sha256sum cand/target/release/titanium-cli > hashes.txt
echo built > status.txt
EXE=/tmp/job/cand/target/release/titanium-cli
$EXE labelbin --bin verify.bin --lo 0 --hi @VERIFY_ROWS@ --nodes 5000 --tuple none --network bundled --threads 1 2>/dev/null | tr -d '\\r' > verify_out.txt
head -n @VERIFY_ROWS@ ref_c0.txt | tr -d '\\r' > verify_ref.txt
SAME=$(paste -d'|' verify_ref.txt verify_out.txt | awk -F'|' '$1==$2' | wc -l)
echo "identical_lines=$SAME" > verify.txt
if [ "$SAME" -lt @VERIFY_MIN@ ]; then echo VERIFY_FAIL > status.txt; exit 1; fi
mkdir -p out
echo running > status.txt
seq 0 @LAST_K@ | EXE=$EXE xargs -P @VCPU@ -I{} bash /tmp/job/one.sh {}
echo DONE > status.txt
"""

JOB_REUSE_SH = """#!/bin/bash
cd /tmp/job
EXE=/tmp/job/cand/target/release/titanium-cli
mkdir -p out
echo running > status.txt
seq 0 @LAST_K@ | EXE=$EXE xargs -P @VCPU@ -I{} bash /tmp/job/one.sh {}
echo DONE > status.txt
"""

ONE_SH = """#!/bin/bash
cd /tmp/job
k=$1
g=$((@FIRST@ + k))
f=out/c_$g.txt
[ -f "$f" ] && exit 0
[ "$(date +%s)" -ge @DEADLINE@ ] && exit 0
lo=$((k * @CH@))
hi=$((lo + @CH@))
[ "$hi" -gt @ROWS@ ] && hi=@ROWS@
$EXE labelbin --bin slice.bin --lo $lo --hi $hi --nodes 5000 --tuple none --network bundled --threads 1 > $f.tmp 2>/dev/null
n=$(wc -l < $f.tmp)
if [ "$n" -eq $((hi - lo)) ]; then mv $f.tmp $f; else rm -f $f.tmp; fi
"""


def run(args_list, check=True, timeout=600):
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    result = subprocess.run(
        args_list, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
    )
    if check and result.returncode != 0:
        sys.exit("command failed: %s\n%s" % (" ".join(args_list)[:200], result.stderr[-400:]))
    return result


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    out_name = manifest["output_dir"]
    assert out_name.startswith("out_dis_remaining_"), "bad output_dir"
    return manifest, D + "/" + out_name


def total_chunks():
    total_rows = os.path.getsize(BIN) // RECORD_BYTES
    return (total_rows + CH - 1) // CH


def claim_cloud_chunks(n_chunks, start_default=2600):
    """Atomically claim a contiguous range [first_chunk, last_chunk) from cloud_next_chunk.txt."""
    max_chunks = total_chunks()
    os.makedirs(os.path.dirname(CLOUD_COUNTER), exist_ok=True)
    if not os.path.exists(CLOUD_COUNTER):
        with open(CLOUD_COUNTER, "w") as f:
            f.write(str(start_default))

    with open(CLOUD_COUNTER, "r+b") as f:
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        try:
            f.seek(0)
            raw = f.read().strip()
            lo = int(raw) if raw else start_default
            if lo >= max_chunks:
                return None
            hi = min(lo + n_chunks, max_chunks)
            f.seek(0)
            f.truncate()
            f.write(str(hi).encode())
            return lo, hi
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def fill(template, values):
    text = template
    for key, value in values.items():
        text = text.replace("@" + key + "@", str(value))
    return text


def write_slice(path, first_row, last_row):
    with open(BIN, "rb") as src, open(path, "wb") as dst:
        src.seek(first_row * RECORD_BYTES)
        left = (last_row - first_row) * RECORD_BYTES
        while left > 0:
            block = src.read(min(left, 64 << 20))
            assert block, "dis_positions.bin ended early"
            dst.write(block)
            left -= len(block)


def stamp_provenance():
    """Update dis_samehash_rows.bin, dis_labeled_ledger.json, and MANIFEST.txt with all valid chunks."""
    manifest, out_dir = load_manifest()
    total_rows = os.path.getsize(BIN) // RECORD_BYTES
    dis_rows = np.fromfile(DIS_ROWS, dtype=np.int64)
    assert len(dis_rows) == total_rows, "dis_rows length mismatch"

    valid_chunks = []
    valid_row_ids = []
    for path in glob.glob(out_dir + "/c_*.txt"):
        m = re.fullmatch(r"c_(\d+)\.txt", os.path.basename(path))
        if not m:
            continue
        idx = int(m.group(1))
        lo = idx * CH
        if lo >= total_rows:
            continue
        hi = min(lo + CH, total_rows)
        want = hi - lo
        try:
            with open(path, "rb") as f:
                lines = f.read().splitlines()
            if len(lines) == want and all(len(line.split()) >= 3 for line in lines):
                valid_chunks.append(idx)
                valid_row_ids.append(dis_rows[lo:hi])
        except OSError:
            pass

    valid_chunks.sort()
    new_rows = np.concatenate(valid_row_ids) if valid_row_ids else np.empty(0, dtype=np.int64)

    # Merge with existing dis_samehash_rows.bin so prior exclusions + new chunks are all recorded
    if os.path.exists(SAMEHASH_BIN):
        old_samehash = np.fromfile(SAMEHASH_BIN, dtype=np.int64)
        merged = np.unique(np.concatenate([old_samehash, new_rows]))
    else:
        merged = np.unique(new_rows)

    tmp_bin = SAMEHASH_BIN + ".tmp"
    merged.astype(np.int64).tofile(tmp_bin)
    os.replace(tmp_bin, SAMEHASH_BIN)

    # Write human/machine-readable provenance ledger
    ledger = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine_commit": manifest.get("engine_commit"),
        "exe_sha256": manifest.get("exe_sha256"),
        "net_sha256": manifest.get("net_sha256"),
        "queue_sha256": manifest.get("queue_sha256"),
        "output_dir": os.path.basename(out_dir),
        "valid_chunks_count": len(valid_chunks),
        "valid_labels_in_current_dir": int(len(new_rows)),
        "total_samehash_excluded_rows": int(len(merged)),
    }
    tmp_json = LEDGER_JSON + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp_json, LEDGER_JSON)

    # Ensure MANIFEST.txt in out_dir also records updated count + hashes
    man_path = out_dir + "/MANIFEST.txt"
    man_lines = [
        f"engine_commit={manifest.get('engine_commit', '')}",
        f"exe_sha={manifest.get('exe_sha256', '')[:16]}",
        f"net_sha={manifest.get('net_sha256', '')[:16]}",
        f"exe_sha256={manifest.get('exe_sha256', '')}",
        f"net_sha256={manifest.get('net_sha256', '')}",
        f"queue_sha256={manifest.get('queue_sha256', '')}",
        "queue_manifest=dis_queue_manifest.json",
        "nodes=5000",
        "reset=per_position",
        f"verified_chunks={len(valid_chunks)}",
        f"verified_labels={len(new_rows)}",
        f"updated={time.strftime('%Y-%m-%dT%H:%M:%S')}",
    ]
    with open(man_path + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(man_lines) + "\n")
    os.replace(man_path + ".tmp", man_path)
    return ledger


def launch(a):
    manifest, out_dir = load_manifest()
    total_rows = os.path.getsize(BIN) // RECORD_BYTES
    first_row = a.first_chunk * CH
    last_row = min(a.last_chunk * CH, total_rows)
    assert first_row < last_row, "empty slice"
    rows = last_row - first_row
    chunks = (rows + CH - 1) // CH
    ref_c0 = out_dir + "/c_0.txt"
    assert os.path.exists(ref_c0), "PC reference chunk c_0.txt missing"

    work = D + "/cloud_" + a.name
    os.makedirs(work, exist_ok=True)
    write_slice(work + "/slice.bin", first_row, last_row)
    write_slice(work + "/verify.bin", 0, VERIFY_ROWS)
    with open(ref_c0, "rb") as src, open(work + "/ref_c0.txt", "wb") as dst:
        dst.write(src.read())
    commit = manifest["engine_commit"]
    run(["git", "-C", MAIN, "archive", "--format=tar", "--prefix=cand/", "-o", work + "/cand.tar", commit])

    deadline = int(time.time() + a.hours * 3600)
    values = {
        "VERIFY_ROWS": VERIFY_ROWS, "VERIFY_MIN": VERIFY_MIN_IDENTICAL, "LAST_K": chunks - 1,
        "VCPU": a.vcpu, "FIRST": a.first_chunk, "DEADLINE": deadline, "CH": CH, "ROWS": rows,
    }
    with open(work + "/job.sh", "w", newline="\n") as f:
        f.write(fill(JOB_SH, values))
    with open(work + "/one.sh", "w", newline="\n") as f:
        f.write(fill(ONE_SH, values))

    ttl_minutes = int(a.hours * 60 + 60)
    worst_case = a.vcpu * PRICE_PER_VCPU_HR * ttl_minutes / 60.0
    print("create %s: %d vCPU, chunks %d..%d (%d rows), ttl %dm, worst case $%.2f" %
          (a.name, a.vcpu, a.first_chunk, a.last_chunk, rows, ttl_minutes, worst_case), flush=True)
    run(["sbx", "--cloud", "create", "--name", a.name, "--image-ref", IMAGE, "--cpus", str(a.vcpu),
         "--memory", "%dg" % (a.vcpu * 2), "--ttl", "%dm" % ttl_minutes, "--on-timeout", "delete",
         "--platform", "linux/amd64"])
    run(["sbx", "--cloud", "exec", a.name, "mkdir", "-p", "/tmp/job"])
    for name in ("cand.tar", "slice.bin", "verify.bin", "ref_c0.txt", "job.sh", "one.sh"):
        run(["sbx", "--cloud", "cp", work + "/" + name, "%s:/tmp/job/%s" % (a.name, name)])
    run(["sbx", "--cloud", "exec", a.name, "bash", "-c",
         "cd /tmp/job && (nohup setsid bash job.sh > job.log 2>&1 < /dev/null &); sleep 3; pgrep -f job.sh"])
    print("started", a.name, "deadline unix", deadline, flush=True)


def reuse(a):
    """Give an existing built sandbox a new slice without rebuilding."""
    total_rows = os.path.getsize(BIN) // RECORD_BYTES
    first_row = a.first_chunk * CH
    last_row = min(a.last_chunk * CH, total_rows)
    assert first_row < last_row, "empty slice"
    rows = last_row - first_row
    chunks = (rows + CH - 1) // CH

    work = D + "/cloud_" + a.name
    os.makedirs(work, exist_ok=True)
    write_slice(work + "/slice.bin", first_row, last_row)

    deadline = int(time.time() + a.hours * 3600)
    values = {
        "LAST_K": chunks - 1, "VCPU": a.vcpu, "FIRST": a.first_chunk,
        "DEADLINE": deadline, "CH": CH, "ROWS": rows,
    }
    with open(work + "/job.sh", "w", newline="\n") as f:
        f.write(fill(JOB_REUSE_SH, values))
    with open(work + "/one.sh", "w", newline="\n") as f:
        f.write(fill(ONE_SH, values))

    add_ttl = int(a.hours * 60 + 30)
    run(["sbx", "--cloud", "ttl", "+%dm" % add_ttl, a.name], check=False)
    run(["sbx", "--cloud", "exec", a.name, "bash", "-c",
         "cd /tmp/job && rm -f status.txt && mv out out_prev_$(date +%s) 2>/dev/null; mkdir -p out"])
    for fn in ("slice.bin", "job.sh", "one.sh"):
        run(["sbx", "--cloud", "cp", work + "/" + fn, "%s:/tmp/job/%s" % (a.name, fn)])
    run(["sbx", "--cloud", "exec", a.name, "bash", "-c",
         "cd /tmp/job && (nohup setsid bash job.sh > job.log 2>&1 < /dev/null &); sleep 2; pgrep -f job.sh"])
    print("reused %s for chunks %d..%d (%d rows)" % (a.name, a.first_chunk, a.last_chunk, rows), flush=True)


def remote(name, command):
    return run(["sbx", "--cloud", "exec", name, "bash", "-c", command], check=False, timeout=90).stdout.strip()


def status(a):
    print("status  :", remote(a.name, "cat /tmp/job/status.txt"))
    print("verify  :", remote(a.name, "cat /tmp/job/verify.txt"))
    print("chunks  :", remote(a.name, "ls /tmp/job/out 2>/dev/null | grep -c '\\.txt$'"))
    print("engines :", remote(a.name, "pgrep -c -f 'labelbin --bin slice.bin'"))
    print("build   :", remote(a.name, "tail -n 3 /tmp/job/build.err"))
    print("hashes  :", remote(a.name, "cat /tmp/job/hashes.txt"))


def collect(a):
    manifest, out_dir = load_manifest()
    work = D + "/cloud_" + a.name
    os.makedirs(work, exist_ok=True)
    state = remote(a.name, "cat /tmp/job/status.txt")
    print(a.name, "status:", state, flush=True)
    if state == "VERIFY_FAIL":
        sys.exit("verify failed; nothing to collect")
    remote(a.name, "cd /tmp/job && tar cf results.tar --exclude='*.tmp' out verify.txt hashes.txt")
    tar_path = work + "/results.tar"
    if os.path.exists(tar_path):
        os.remove(tar_path)
    run(["sbx", "--cloud", "cp", "%s:/tmp/job/results.tar" % a.name, tar_path], check=False)
    if not os.path.exists(tar_path):
        print(a.name, "no results.tar fetched", flush=True)
        return 0
    with tarfile.open(tar_path) as t:
        t.extractall(work)
    moved = 0
    rejected = 0
    total_rows = os.path.getsize(BIN) // RECORD_BYTES
    out_sub = work + "/out"
    if os.path.isdir(out_sub):
        for name in sorted(os.listdir(out_sub)):
            if not name.startswith("c_") or not name.endswith(".txt"):
                continue
            index = int(name[2:-4])
            want = min((index + 1) * CH, total_rows) - index * CH
            src = out_sub + "/" + name
            with open(src, "rb") as f:
                lines = f.read().splitlines()
            columns_ok = all(len(line.split()) >= 3 for line in lines)
            if len(lines) != want or not columns_ok:
                rejected += 1
                continue
            dst = out_dir + "/" + name
            if os.path.exists(dst):
                os.remove(src)
                continue
            os.replace(src, dst)
            moved += 1
    ledger = stamp_provenance()
    print("moved %d new chunks into %s (rejected %d, total verified in dir: %d)" %
          (moved, os.path.basename(out_dir), rejected, ledger["valid_chunks_count"]), flush=True)
    return moved


def watch(a):
    """Autonomous watcher for tit-d1, tit-d2, tit-d3 capped at --budget-usd."""
    sandboxes = {
        "tit-d1": {"vcpu": 16, "created": time.time() - 900, "active": True},
        "tit-d2": {"vcpu": 16, "created": time.time() - 660, "active": True},
        "tit-d3": {"vcpu": 8,  "created": time.time() - 660, "active": True},
    }
    # Write initial cloud_next_chunk.txt = 2600 if not present
    if not os.path.exists(CLOUD_COUNTER):
        with open(CLOUD_COUNTER, "w") as f:
            f.write("2600")

    def log_msg(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(WATCH_LOG, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    log_msg(f"watcher started: budget=${a.budget_usd:.2f}, sandboxes={list(sandboxes.keys())}")
    pass_num = 0
    while any(s["active"] for s in sandboxes.values()):
        pass_num += 1
        now = time.time()
        # Calculate total spend across all sandboxes so far
        total_spend = 0.0
        for name, info in sandboxes.items():
            end_t = now if info["active"] else info["deleted_at"]
            hrs = max(0.0, (end_t - info["created"]) / 3600.0)
            total_spend += hrs * info["vcpu"] * PRICE_PER_VCPU_HR

        ls_out = run(["sbx", "--cloud", "ls"], check=False, timeout=60).stdout
        for name, info in list(sandboxes.items()):
            if not info["active"]:
                continue
            if name not in ls_out:
                log_msg(f"{name} no longer in sbx ls; marking inactive")
                info["active"] = False
                info["deleted_at"] = now
                continue

            st = remote(name, "cat /tmp/job/status.txt")
            chk = remote(name, "ls /tmp/job/out 2>/dev/null | grep -c '\\.txt$'")
            eng = remote(name, "pgrep -c -f 'labelbin --bin slice.bin'")
            log_msg(f"pass {pass_num}: {name} ({info['vcpu']} vCPU) status={st} chunks={chk} engines={eng} spend_total=${total_spend:.2f}")

            # Pull completed chunks every pass so dashboard & local dir stay fresh!
            try:
                c_args = argparse.Namespace(name=name, keep=True)
                collect(c_args)
            except Exception as e:
                log_msg(f"{name} collect warning: {e}")

            # Check if budget reached or slice done
            active_vcpus = sum(s["vcpu"] for s in sandboxes.values() if s["active"])
            burn_rate_per_hr = active_vcpus * PRICE_PER_VCPU_HR
            remaining_budget = a.budget_usd - total_spend

            if remaining_budget <= 0.35:
                log_msg(f"budget ${a.budget_usd:.2f} reached (spent ${total_spend:.2f}); final collect & deleting {name}")
                try:
                    collect(argparse.Namespace(name=name, keep=False))
                except Exception:
                    pass
                run(["sbx", "--cloud", "rm", "--force", name], check=False)
                info["active"] = False
                info["deleted_at"] = time.time()
                hrs = (info["deleted_at"] - info["created"]) / 3600.0
                actual_cost = hrs * info["vcpu"] * PRICE_PER_VCPU_HR
                with open(SPEND_CSV, "a", encoding="utf-8") as sf:
                    sf.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{info['vcpu']},{info['vcpu']*2},"
                             f"{info['vcpu']*PRICE_PER_VCPU_HR:.2f},0,0,"
                             f"DELETED after ~{int(hrs*60)} min (actual ~${actual_cost:.2f})\n")
                continue

            if st == "DONE":
                # Estimate how many hours of budget remain if all active sandboxes share it
                hours_left = remaining_budget / max(0.5, burn_rate_per_hr)
                if hours_left < 0.35:
                    log_msg(f"{name} DONE and only {hours_left:.2f}h budget left; deleting {name}")
                    run(["sbx", "--cloud", "rm", "--force", name], check=False)
                    info["active"] = False
                    info["deleted_at"] = time.time()
                    hrs = (info["deleted_at"] - info["created"]) / 3600.0
                    actual_cost = hrs * info["vcpu"] * PRICE_PER_VCPU_HR
                    with open(SPEND_CSV, "a", encoding="utf-8") as sf:
                        sf.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{info['vcpu']},{info['vcpu']*2},"
                                 f"{info['vcpu']*PRICE_PER_VCPU_HR:.2f},0,0,"
                                 f"DELETED after ~{int(hrs*60)} min (actual ~${actual_cost:.2f})\n")
                    continue

                # Claim a new slice sized to ~min(hours_left, 2.0) hours at ~16 chunks/hr/vCPU
                slice_hours = min(hours_left, 2.0)
                n_claim = max(16, int(info["vcpu"] * 15 * slice_hours))
                claimed = claim_cloud_chunks(n_claim)
                if not claimed:
                    log_msg(f"{name} DONE and queue exhausted; deleting {name}")
                    run(["sbx", "--cloud", "rm", "--force", name], check=False)
                    info["active"] = False
                    info["deleted_at"] = time.time()
                    continue
                c_lo, c_hi = claimed
                log_msg(f"reusing {name} for chunks {c_lo}..{c_hi} ({slice_hours:.2f}h)")
                try:
                    r_args = argparse.Namespace(
                        name=name, vcpu=info["vcpu"],
                        first_chunk=c_lo, last_chunk=c_hi, hours=slice_hours,
                    )
                    reuse(r_args)
                    with open(SPEND_CSV, "a", encoding="utf-8") as sf:
                        sf.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{info['vcpu']},{info['vcpu']*2},"
                                 f"{info['vcpu']*PRICE_PER_VCPU_HR:.2f},{int(slice_hours*60+30)},0,"
                                 f"reused for chunks {c_lo}-{c_hi}\n")
                except Exception as e:
                    log_msg(f"{name} reuse failed: {e}")

        if not any(s["active"] for s in sandboxes.values()):
            break
        time.sleep(300)

    stamp_provenance()
    log_msg("watcher finished; all sandboxes stopped and provenance stamped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["launch", "reuse", "status", "collect", "stamp", "watch"])
    ap.add_argument("--name", default="")
    ap.add_argument("--vcpu", type=int, default=16)
    ap.add_argument("--first-chunk", type=int, default=0)
    ap.add_argument("--last-chunk", type=int, default=0)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=15.0)
    args = ap.parse_args()
    if args.cmd == "stamp":
        print(json.dumps(stamp_provenance(), indent=2))
    else:
        {"launch": launch, "reuse": reuse, "status": status, "collect": collect, "watch": watch}[args.cmd](args)
