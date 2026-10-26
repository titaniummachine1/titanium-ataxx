"""Relabel driver: the plain big net (nnue_t1ep4 c4829350, tuple none) at 5000 nodes per position, state reset per row.

  pack:    python training/relabel_pack.py --subset N          (positions.bin + rows.npy)
  plan:    python training/relabel_run.py plan --pc-rate 24 --cloud 4 --cloud-vcpu 16 --vcpu-rate 4
  pc:      python training/relabel_run.py pc --lo A --hi B [--threads 7]     (resumable: skips finished chunks)
  cloud:   python training/relabel_run.py cloud --name sbN --lo A --hi B --vcpu 16 --ttl 8h
  collect: python training/relabel_run.py collect --name sbN
  merge:   python training/relabel_run.py merge             (all chunks -> data/nnue/relabel/labels.npz)
Chunks are CH rows each (5000), one output file per chunk (out/c_<firstrow>.txt: 'score depth nodes' per row), so a
killed run resumes and every chunk is independent. The PC can take over any chunk a sandbox did not finish.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

ROOT = "C:/gitProjects/titanium-ataxx"
D = os.environ.get("RELABEL_DIR", ROOT + "/data/nnue/relabel")
OUT = D + "/out"
EXE = "C:/gitProjects/main-ataxx/target/release/titanium-cli.exe"
NET = ROOT + "/data/nnue/champions/STAGING_NNUE_t1ep4_c4829350.tnn"
CH = 5000


def total():
    return os.path.getsize(D + "/positions.bin") // 25


def plan(a):
    n = total()
    w = [("pc", a.pc_rate)] + [("cloud%d" % i, a.cloud_vcpu * a.vcpu_rate) for i in range(a.cloud)]
    tot = sum(r for _, r in w)
    print("%d positions, %.0f pos/s total -> %.1f h" % (n, tot, n / tot / 3600))
    lo, cloud_rows = 0, 0
    for i, (name, r) in enumerate(w):
        hi = n if i == len(w) - 1 else min(n, lo + int(round(n * r / tot / CH)) * CH)
        print("  %-7s rows %d..%d (%d rows, %.1f h)" % (name, lo, hi, hi - lo, (hi - lo) / r / 3600))
        if name != "pc":
            cloud_rows += hi - lo
        lo = hi
    print("cloud cost ~ $%.0f at $0.07/vCPU-h" % (cloud_rows / (a.vcpu_rate * 3600) * 0.07))


def pc(a):
    os.makedirs(OUT, exist_ok=True)
    hi = min(a.hi, total())
    for lo in range(a.lo, hi, CH):
        if a.deadline and time.time() >= a.deadline:
            print("deadline reached", flush=True)
            break
        f = "%s/c_%d.txt" % (OUT, lo)
        if os.path.exists(f):
            continue
        h = min(lo + CH, hi)
        r = subprocess.run([EXE, "labelbin", "--bin", D + "/positions.bin", "--lo", str(lo), "--hi", str(h), "--nodes", "5000",
                            "--tuple", "none", "--nnue", NET, "--threads", str(a.threads)], capture_output=True, text=True)
        if r.returncode or len(r.stdout.splitlines()) != h - lo:
            sys.exit("chunk %d failed: %s" % (lo, r.stderr[-300:]))
        open(f + ".tmp", "w").write(r.stdout)
        os.replace(f + ".tmp", f)
        print("chunk", lo, "done", flush=True)


def sh(cmd):
    print("+", cmd[:150], flush=True)
    return subprocess.run(cmd, shell=True, env=dict(os.environ, MSYS_NO_PATHCONV="1"))


JOB = """#!/bin/bash
cd /tmp/job && tar xf cand.tar && (cd cand && cargo build -p titanium-cli --release -q 2>/dev/null)
sha256sum cand/target/release/titanium-cli nnue.tnn > hashes.txt
mkdir -p out; echo built > status.txt
N=$(( $(stat -c %%s slice.bin) / 25 ))
for ((i=0; i<N; i+=%(ch)d)); do
  [ $(date +%%s) -ge %(deadline)d ] && break
  h=$((i+%(ch)d)); [ $h -gt $N ] && h=$N
  f=out/c_$((i+%(lo)d)).txt
  [ -f $f ] || { ./cand/target/release/titanium-cli labelbin --bin slice.bin --lo $i --hi $h --nodes 5000 --tuple none --nnue nnue.tnn --threads %(vcpu)d > $f.tmp 2>/dev/null && [ $(wc -l < $f.tmp) -eq $((h-i)) ] && mv $f.tmp $f; }
done
echo DONE > status.txt
"""


_NL = chr(10)
JOB_REUSE = ("#!/bin/bash" + _NL + "cd /tmp/job" + _NL + "mkdir -p out; echo built > status.txt" + _NL
             + JOB.split("mkdir -p out; echo built > status.txt" + _NL, 1)[1])


def reuse(a):
    """Give a finished sandbox a new slice: upload slice.bin + job.sh, move the old out/ aside, restart the job.
    The sandbox keeps its engine and net, so there is no create, no upload of the 28 MB net and no build."""
    work = D + "/cloud_" + a.name
    hi = min(a.hi, total())
    with open(D + "/positions.bin", "rb") as f, open(work + "/slice.bin", "wb") as g:
        f.seek(a.lo * 25)
        left = (hi - a.lo) * 25
        while left:
            b = f.read(min(left, 64 << 20))
            g.write(b)
            left -= len(b)
    open(work + "/job.sh", "w", newline=_NL).write(JOB_REUSE % dict(ch=CH, lo=a.lo, vcpu=a.vcpu, deadline=a.deadline or 2**40))
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    def ex(cmd):
        r = subprocess.run(["sbx", "--cloud", "exec", a.name, "bash", "-c", cmd], env=env, capture_output=True, text=True)
        if r.returncode:
            sys.exit("reuse: remote command failed: %s %s" % (cmd[:60], r.stderr[-200:]))
    ex("cd /tmp/job && rm -f status.txt && mv out out_prev_$(date +%s) && mkdir out")
    for fn in ("slice.bin", "job.sh"):
        r = subprocess.run(["sbx", "--cloud", "cp", work + "/" + fn, "%s:/tmp/job/%s" % (a.name, fn)], env=env)
        if r.returncode:
            sys.exit("reuse: upload of %s failed" % fn)
    ex("cd /tmp/job && (nohup setsid bash job.sh > job.log 2>&1 < /dev/null &)")
    print("reused", a.name, "rows", a.lo, hi, flush=True)


def cloud(a):
    work = D + "/cloud_" + a.name
    os.makedirs(work, exist_ok=True)
    hi = min(a.hi, total())
    with open(D + "/positions.bin", "rb") as f, open(work + "/slice.bin", "wb") as g:
        f.seek(a.lo * 25)
        left = (hi - a.lo) * 25
        while left:
            b = f.read(min(left, 64 << 20))
            g.write(b)
            left -= len(b)
    if a.prebuilt:  # ready-made Linux binary (zig build, verified bit-identical labels): the sandbox skips its cargo build
        import tarfile
        with tarfile.open(work + "/cand.tar", "w") as t:
            ti = t.gettarinfo(a.prebuilt, arcname="cand/target/release/titanium-cli")
            ti.mode = 0o755
            with open(a.prebuilt, "rb") as fb:
                t.addfile(ti, fb)
    else:
        sh("git -C C:/gitProjects/main-ataxx archive --format=tar --prefix=cand/ -o %s/cand.tar HEAD" % work)
    open(work + "/job.sh", "w", newline="\n").write(JOB % dict(ch=CH, lo=a.lo, vcpu=a.vcpu, deadline=a.deadline or 2**40))
    sh("sbx --cloud create --name %s --image-ref docker.io/library/rust:1.97 --cpus %d --memory %dg --ttl %s --on-timeout delete --platform linux/amd64"
       % (a.name, a.vcpu, a.vcpu * 2, a.ttl))
    sh("sbx --cloud exec %s mkdir -p /tmp/job" % a.name)
    for fn, src in (("cand.tar", work + "/cand.tar"), ("slice.bin", work + "/slice.bin"), ("job.sh", work + "/job.sh"), ("nnue.tnn", NET)):
        sh("sbx --cloud cp %s %s:/tmp/job/%s" % (src, a.name, fn))
    # fully detached (own session, no stdin): a job tied to the exec stream died when the stream closed (tit-rl1)
    subprocess.run(["sbx", "--cloud", "exec", a.name, "bash", "-c",
                    "cd /tmp/job && (nohup setsid bash job.sh > job.log 2>&1 < /dev/null &); sleep 3; pgrep -f labelbin || pgrep -f cargo"],
                   env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    print("started", a.name, "rows", a.lo, hi)


def collect(a):
    work = D + "/cloud_" + a.name
    st = subprocess.run("sbx --cloud exec %s cat /tmp/job/status.txt" % a.name, shell=True, capture_output=True, text=True,
                        env=dict(os.environ, MSYS_NO_PATHCONV="1")).stdout
    print(a.name, "status:", st.strip())
    if "DONE" not in st and not a.force:
        sys.exit("not finished (use --force to take what exists)")
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    # list args, no shell: Windows cmd.exe would mangle the quoting of the remote command
    subprocess.run(["sbx", "--cloud", "exec", a.name, "bash", "-c",
                    "cd /tmp/job && tar cf results.tar --exclude='*.tmp' out hashes.txt"], env=env)
    if os.path.exists(work + "/results.tar"):
        os.remove(work + "/results.tar")
    subprocess.run(["sbx", "--cloud", "cp", "%s:/tmp/job/results.tar" % a.name, work + "/results.tar"], env=env)
    if not os.path.exists(work + "/results.tar"):
        sys.exit("could not fetch results.tar from %s" % a.name)
    import tarfile
    with tarfile.open(work + "/results.tar") as t:
        t.extractall(work)
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(work + "/out/c_*.txt"):
        os.replace(f, OUT + "/" + os.path.basename(f))
    if not a.keep:
        sh("sbx --cloud rm --force %s" % a.name)


def merge(a):
    n = total()
    sc, dp, nd = np.zeros(n, np.int32), np.zeros(n, np.int16), np.zeros(n, np.int32)
    have = np.zeros(n, bool)
    for f in sorted(glob.glob(OUT + "/c_*.txt")):
        lo = int(os.path.basename(f)[2:-4])
        v = np.loadtxt(f, dtype=np.int64, ndmin=2)
        sc[lo:lo + len(v)], dp[lo:lo + len(v)], nd[lo:lo + len(v)] = v[:, 0], v[:, 1], v[:, 2]
        have[lo:lo + len(v)] = True
    print("labelled %d / %d rows" % (have.sum(), n))
    if not have.all() and not a.partial:
        sys.exit("missing chunks - not writing labels.npz (use --partial: positions.bin is shuffled, so the labelled rows are a random sample)")
    np.savez(D + "/labels.npz", rows=np.load(D + "/rows.npy"), have=have, score=sc, depth=dp, nodes=nd)
    print("wrote", D + "/labels.npz", "score range", sc[have].min(), sc[have].max(), "mean depth %.2f" % dp[have].mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "pc", "cloud", "reuse", "collect", "merge"])
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=1 << 60)
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--name", default="")
    ap.add_argument("--vcpu", type=int, default=16)
    ap.add_argument("--ttl", default="8h")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prebuilt", default="", help="cloud: path of a Linux titanium-cli to ship instead of building from source")
    ap.add_argument("--keep", action="store_true", help="collect: pull finished chunks but leave the sandbox running")
    ap.add_argument("--partial", action="store_true", help="merge: write labels.npz with a 'have' mask")
    ap.add_argument("--deadline", type=int, default=0, help="unix time after which no new chunk is started (pc, cloud)")
    ap.add_argument("--pc-rate", type=float, default=22)
    ap.add_argument("--cloud", type=int, default=4)
    ap.add_argument("--cloud-vcpu", type=int, default=16)
    ap.add_argument("--vcpu-rate", type=float, default=4)
    a = ap.parse_args()
    {"plan": plan, "pc": pc, "cloud": cloud, "reuse": reuse, "collect": collect, "merge": merge}[a.cmd](a)
