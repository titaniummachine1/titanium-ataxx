"""D662: per-move TIME MARGINS of the real Botzone driver + engine library under background CPU load (test harness around botzone/sim_local.py).
margin = limit - (time from request written to marker read); positive = answered before the limit, negative = LATE (a forfeit risk on Botzone).
usage: python driver_margin.py --repo <worktree> --lib <dll/so> --net <tnn> [--games 4] [--scale 0.25] [--load 0,4,8] [--moves 12] [--out report.txt]
Runs N long-session games per load level (busy-loop processes at normal priority), prints percentiles of the margins and the count of late moves.
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--lib", required=True)
ap.add_argument("--net", required=True)
ap.add_argument("--games", type=int, default=4)
ap.add_argument("--scale", type=float, default=0.25)
ap.add_argument("--load", default="0,4,8")
ap.add_argument("--moves", type=int, default=12)
ap.add_argument("--out", default="driver_margin.txt")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "botzone"))
import sim_local  # noqa: E402


class Args:
    pass


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


lines = []
for load in [int(x) for x in a.load.split(",")]:
    busy = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(load)]
    margins = []
    try:
        for gi in range(a.games):
            ns = Args()
            ns.lib, ns.net, ns.scale, ns.tt_bits, ns.mode = a.lib, a.net, a.scale, 22, "long"
            ns.max_driver_moves = a.moves
            sim_local.random.seed(100 + gi)
            res = sim_local.run_game(ns, gi % 2 == 0)
            margins += res[2][1:]  # skip the first (longer-limit) move; res[2] = margins list
    finally:
        for b in busy:
            b.kill()
    late = [m for m in margins if m < 0]
    line = ("load %d busy procs | n=%d moves (limit %.0f ms) | margin ms: min %.1f p1 %.1f p5 %.1f p50 %.1f max %.1f | LATE %d/%d (worst %.1f ms) | within 10 ms: %d"
            % (load, len(margins), 4000 * a.scale, min(margins), pct(margins, 0.01), pct(margins, 0.05), pct(margins, 0.5), max(margins),
               len(late), len(margins), min(margins), len([m for m in margins if m < 10])))
    print(line)
    lines.append(line)
open(a.out, "w").write("\n".join(lines) + "\n")
