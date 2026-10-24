"""Gate an NNUE checkpoint against main's S57 at fixed 5k nodes (quality; speed is judged separately by time gates).

1. export the .pt to TNN1 and verify Rust == torch on 1000 positions (refuses to gate on any mismatch)
2. 8 legs x 200 games, paired book lines 2000-2799 from both seats, in-process (--opp self), same exe both sides:
   candidate side = --nnue <tnn> --tuple none, champion side = --opp-tuple S57 (the exe's unchanged classic path)
3. 20-game determinism tripwire on leg 0 (must replay bit-identically, else the run is VOID)
4. pooled summary via gate_summary.py. Nothing is merged: the verdict is the user's (iron laws 3/4/10).
Law 8: the exe must be built from a COMMITTED exp/nnue ref; the script records exe/net hashes and the git ref.
Usage: python training/gate_nnue.py --pt data/nnue/nnue_pbt/best_teacher.pt --tag pbt_t1
"""
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = "C:/gitProjects/titanium-ataxx"
EXE = "C:/gitProjects/nnue-ataxx/target/release/titanium-cli.exe"
CHAMP = "data/nnue/final_v2_wdl_ep8.tup"
BOOK = "data/nnue/book50k.txt"


def sha(p):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return h[:8].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--drop", default="")
    ap.add_argument("--legs", type=int, default=8)
    ap.add_argument("--games", type=int, default=200)
    a = ap.parse_args()
    tnn = "%s/data/nnue/nnue_%s.tnn" % (ROOT, a.tag)
    exp = [sys.executable, "training/export_nnue.py", "--pt", a.pt, "--out", tnn] + (["--drop", a.drop] if a.drop else [])
    subprocess.run(exp, cwd=ROOT, check=True)
    par = subprocess.run([EXE, "nneval", "--nnue", tnn, "--boards", tnn + ".ref.txt"], capture_output=True, text=True)
    print(par.stdout.strip())
    if par.returncode != 0:
        sys.exit("PARITY FAILED -> not gating")
    ref = subprocess.run(["git", "-C", "C:/gitProjects/nnue-ataxx", "log", "-1", "--format=%h"], capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", "C:/gitProjects/nnue-ataxx", "status", "--porcelain", "crates"],
                           capture_output=True, text=True).stdout.strip()
    print("exe %s (exp/nnue %s%s) cand %s champ %s" % (sha(EXE), ref, " DIRTY" if dirty else "", sha(tnn),
                                                       sha(ROOT + "/" + CHAMP)))
    if dirty:
        print("WARNING: exp/nnue has uncommitted crate changes -> result is DIAGNOSTIC ONLY (law 8)")
    files, procs = [], []
    for i in range(a.legs):
        f = "logs/gate_%s_p%d.txt" % (a.tag, i)
        files.append(f)
        procs.append(subprocess.Popen([EXE, "match", "--games", str(a.games), "--nodes", "5000", "--opp", "self",
                                       "--opp-nodes", "5000", "--tuple", "none", "--nnue", tnn, "--opp-tuple", CHAMP,
                                       "--book", BOOK, "--book-start", str(2000 + i * a.games // 2),
                                       "--start-game", "1", "--out", f], cwd=ROOT,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    trip = "logs/gate_%s_trip.txt" % a.tag
    subprocess.run([EXE, "match", "--games", "20", "--nodes", "5000", "--opp", "self", "--opp-nodes", "5000",
                    "--tuple", "none", "--nnue", tnn, "--opp-tuple", CHAMP, "--book", BOOK, "--book-start", "2000",
                    "--start-game", "1", "--out", trip], cwd=ROOT, stdout=subprocess.DEVNULL)
    g = lambda p: [l for l in open(ROOT + "/" + p, errors="ignore").read().splitlines() if l.startswith("-- game")][:20]
    print("tripwire:", "BIT-IDENTICAL" if g(files[0]) == g(trip) else "MISMATCH -> RUN VOID")
    pooled = "%s/logs/gate_%s_pooled.txt" % (ROOT, a.tag)
    with open(pooled, "w") as fo:
        for f in files:
            fo.write(open(ROOT + "/" + f, errors="ignore").read())
    print(subprocess.run([sys.executable, "training/gate_summary.py", pooled], cwd=ROOT, capture_output=True,
                         text=True).stdout)


if __name__ == "__main__":
    main()
