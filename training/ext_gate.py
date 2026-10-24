"""Gate current Titanium (main exe + S57 net) against an external UAI engine with the repo's `match` harness.

Classic rules, paired book (each line from both seats), `uainewgame` per game (harness does it),
N shards in parallel on disjoint book lines, paired-bootstrap CI over all shards.
Example:
  python training/ext_gate.py --tag moon100 --opp-cmd "C:/.../Moonbird.exe" --time 100 --games 100
"""
import argparse
import glob
import math
import os
import random
import re
import subprocess
import sys

ROOT = "C:/gitProjects/titanium-ataxx"
EXE = "C:/gitProjects/main-ataxx/target/release/titanium-cli.exe"
NET = "data/nnue/final_v2_wdl_ep8.tup"
PAT = re.compile(r"^-- game \d+: book (\d+) titanium=(\w) opp=(\w): (titanium|opponent) wins (\d+)-(\d+) \((\d+) plies\)")
DRAW = re.compile(r"^-- game \d+: book (\d+) titanium=(\w) opp=(\w): draw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--opp-cmd", required=True)
    ap.add_argument("--time", type=int, default=None, help="ms per move, both sides")
    ap.add_argument("--nodes", type=int, default=None, help="titanium nodes (opp gets --opp-time)")
    ap.add_argument("--opp-time", type=int, default=1000)
    ap.add_argument("--opp-nodes", type=int, default=None, help="opp nodes per move (equal-node gates)")
    ap.add_argument("--exe", default=EXE, help="Titanium binary under test (default: main)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--book-base", type=int, default=0)
    a = ap.parse_args()
    per = a.games // a.shards
    per -= per % 2
    lines = per // 2
    procs = []
    for i in range(a.shards):
        out = "logs/gate_%s_s%d.txt" % (a.tag, i)
        cmd = [a.exe, "match", "--games", str(per), "--tuple", NET, "--opp-tuple", NET, "--opp", a.opp_cmd,
               "--book", "data/nnue/book50k.txt", "--book-start", str(a.book_base + i * lines),
               "--start-game", "1", "--out", out]
        if a.time:
            cmd += ["--time", str(a.time), "--opp-time", str(a.time)]
        else:
            cmd += ["--nodes", str(a.nodes)] + (["--opp-nodes", str(a.opp_nodes)] if a.opp_nodes else ["--opp-time", str(a.opp_time)])
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=open("logs/gate_%s_s%d.stdout" % (a.tag, i), "w"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    W = L = D = 0
    seat = {"x": [0, 0], "o": [0, 0]}
    pairs = []
    for f in sorted(glob.glob("logs/gate_%s_s*.txt" % a.tag)):
        by = {}
        for line in open(f, encoding="utf-8", errors="ignore"):
            line = line.strip()
            m = PAT.match(line)
            if m:
                w = m.group(4) == "titanium"
                W += w
                L += not w
                seat[m.group(2)][0 if w else 1] += 1
                by.setdefault(int(m.group(1)), []).append(1.0 if w else 0.0)
                continue
            m = DRAW.match(line)
            if m:
                D += 1
                by.setdefault(int(m.group(1)), []).append(0.5)
        pairs += [sum(v) / len(v) for v in by.values()]
    n = W + L + D
    s = (W + 0.5 * D) / n if n else float("nan")
    random.seed(1)
    bs = sorted(sum(random.choices(pairs, k=len(pairs))) / len(pairs) for _ in range(4000)) if pairs else [0, 0]
    lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]

    def elo(x):
        x = min(max(x, 1e-4), 1 - 1e-4)
        return -400 * math.log10(1 / x - 1)

    print("%s: games %d Titanium W-L-D %d-%d-%d score %.3f elo %+.0f CI[%.3f,%.3f] eloCI[%+.0f,%+.0f] seats(x W/L)%s (o W/L)%s"
          % (a.tag, n, W, L, D, s, elo(s), lo, hi, elo(lo), elo(hi), seat["x"], seat["o"]))


if __name__ == "__main__":
    main()
