"""Docker cloud sandbox relabeler for the disagreement queue (dis_positions.bin).

One sandbox = one slice of whole 5000-row chunks. Inside the sandbox: one single-thread
labelbin process per vCPU (xargs -P), never one multi-threaded process.
Output files are c_<global chunk index>.txt, the same names the PC workers write, so the PC
skips any chunk a sandbox already finished.

  launch:  python training/relabel_dis_cloud.py launch --name tit-d1 --vcpu 16 --first-chunk 1200 --last-chunk 1760 --hours 4
  status:  python training/relabel_dis_cloud.py status --name tit-d1
  collect: python training/relabel_dis_cloud.py collect --name tit-d1 [--keep]   (--keep = pull what exists, leave it running)
  delete:  sbx --cloud rm --force tit-d1     (after collect; log the spend in logs/docker_spend.csv)

The engine is built inside the sandbox from the exact commit in dis_queue_manifest.json.
Before labeling, the sandbox relabels the first 100 queue rows and compares them with the PC's
c_0.txt. If fewer than 95 lines are identical it stops (status VERIFY_FAIL) and labels nothing.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = "C:/gitProjects/titanium-ataxx"
MAIN = "C:/gitProjects/main-ataxx"
D = ROOT + "/data/nnue/relabel"
BIN = D + "/dis_positions.bin"
MANIFEST = D + "/dis_queue_manifest.json"
CH = 5000
RECORD_BYTES = 25
VERIFY_ROWS = 100
VERIFY_MIN_IDENTICAL = 95
IMAGE = "docker.io/library/rust:1.97"

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


def run(args_list, check=True):
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    result = subprocess.run(args_list, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit("command failed: %s\n%s" % (" ".join(args_list)[:200], result.stderr[-400:]))
    return result


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    out_name = manifest["output_dir"]
    assert out_name.startswith("out_dis_remaining_"), "bad output_dir"
    return manifest, D + "/" + out_name


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
    worst_case = a.vcpu * 0.07 * ttl_minutes / 60.0
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


def remote(name, command):
    return run(["sbx", "--cloud", "exec", name, "bash", "-c", command], check=False).stdout.strip()


def status(a):
    print("status  :", remote(a.name, "cat /tmp/job/status.txt"))
    print("verify  :", remote(a.name, "cat /tmp/job/verify.txt"))
    print("chunks  :", remote(a.name, "ls /tmp/job/out 2>/dev/null | grep -c '\\.txt$'"))
    print("engines :", remote(a.name, "pgrep -c -f 'labelbin --bin slice.bin'"))
    print("build   :", remote(a.name, "tail -n 3 /tmp/job/build.err"))
    print("hashes  :", remote(a.name, "cat /tmp/job/hashes.txt"))


def collect(a):
    import tarfile
    manifest, out_dir = load_manifest()
    work = D + "/cloud_" + a.name
    state = remote(a.name, "cat /tmp/job/status.txt")
    print(a.name, "status:", state)
    if state == "VERIFY_FAIL":
        sys.exit("verify failed; nothing to collect")
    remote(a.name, "cd /tmp/job && tar cf results.tar --exclude='*.tmp' out verify.txt hashes.txt")
    tar_path = work + "/results.tar"
    if os.path.exists(tar_path):
        os.remove(tar_path)
    run(["sbx", "--cloud", "cp", "%s:/tmp/job/results.tar" % a.name, tar_path])
    with tarfile.open(tar_path) as t:
        t.extractall(work)
    moved = 0
    rejected = 0
    total_rows = os.path.getsize(BIN) // RECORD_BYTES
    for name in sorted(os.listdir(work + "/out")):
        if not name.startswith("c_") or not name.endswith(".txt"):
            continue
        index = int(name[2:-4])
        want = min((index + 1) * CH, total_rows) - index * CH
        src = work + "/out/" + name
        with open(src, "rb") as f:
            lines = f.read().splitlines()
        columns_ok = all(len(line.split()) >= 3 for line in lines)
        if len(lines) != want or not columns_ok:
            rejected += 1
            continue
        dst = out_dir + "/" + name
        if os.path.exists(dst):
            continue
        os.replace(src, dst)
        moved += 1
    print("moved %d chunks into %s, rejected %d" % (moved, out_dir, rejected), flush=True)
    if not a.keep:
        print("collected; delete with: sbx --cloud rm --force %s" % a.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["launch", "status", "collect"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--vcpu", type=int, default=16)
    ap.add_argument("--first-chunk", type=int, default=0)
    ap.add_argument("--last-chunk", type=int, default=0)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    {"launch": launch, "status": status, "collect": collect}[args.cmd](args)
