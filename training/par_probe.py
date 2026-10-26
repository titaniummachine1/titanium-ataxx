"""Throughput probe for PURE NODE-LIMIT gates (5k nodes, --time 0): how many matches to run at once on this machine.
Results of such gates do not depend on speed, so the only goal is games per minute (hyperthreads may help); time-control
gates stay at one match per physical core (timing noise). Runs N matches of --games games each concurrently, for every N
in --pars, and prints games/min. Same engines and command shape as training/gate_net_vs_main.py.
Usage: python training/par_probe.py [--pars 4,6,8,10,12] [--games 4] [--exe X] [--net NET] [--book BOOK]
"""
import argparse
import os
import subprocess
import tempfile
import time

ROOT = os.environ.get("TI_ROOT", "C:/gitProjects/titanium-ataxx")
MAIN = os.environ.get("TI_MAIN", "C:/gitProjects/main-ataxx")
SUFFIX = ".exe" if os.name == "nt" else ""

ap = argparse.ArgumentParser()
ap.add_argument("--pars", default="4,6,8,10,12")
ap.add_argument("--games", type=int, default=4, help="per match process (even)")
ap.add_argument("--exe", default=MAIN + "/target/release/titanium-cli" + SUFFIX)
ap.add_argument("--net", default=MAIN + "/data/nnue/nnue_main.tnn")
ap.add_argument("--book", default=ROOT + "/data/nnue/book50k.txt")
a = ap.parse_args()
tmp = tempfile.mkdtemp()
for n in [int(x) for x in a.pars.split(",")]:
    t = time.perf_counter()
    ps = []
    for k in range(n):
        cmd = [a.exe, "match", "--games", str(a.games), "--nodes", "5000", "--time", "0", "--opp-nodes", "5000", "--tuple", "none",
               "--nnue", a.net, "--opp", a.exe + " serve", "--opp-tuple", "none", "--opp-nnue", a.net, "--book", a.book,
               "--book-start", str(20000 + k * a.games // 2), "--start-game", "1", "--out", "%s/p%d_%d.txt" % (tmp, n, k)]
        ps.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in ps:
        p.wait()
    dt = time.perf_counter() - t
    print("par %2d: %3d games in %6.1f s = %5.1f games/min" % (n, n * a.games, dt, 60 * n * a.games / dt), flush=True)
