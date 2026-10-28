"""How deep do Titanium (main) and Moonbird search at equal TIME and equal NODES? READ-ONLY.
Titanium: `titanium-cli bestmove <board> --time MS` (prints depth/nodes) and `label --nodes N` (score depth nodes).
Moonbird: UAI `go movetime MS` / `go nodes N`, last `info depth` line. Same 40 mid-game val positions for both.
Usage: python training/search_depth_probe.py [n] [ms] [nodes]
"""
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
from ataxx_move import board_str  # noqa: E402
from ataxx_cpp_match import to_fen  # noqa: E402

TI = "C:/gitProjects/nnue-ataxx/target/release/titanium-cli.exe"  # exp/nnue: main 653d9a1 search + label --time
TUP = "C:/gitProjects/titanium-ataxx/data/nnue/final_v2_wdl_ep8.tup"
MOON = "C:/gitProjects/titanium-ataxx/scripts/moonbird/Moonbird-1.1.0-windows-amd64.exe"


def moon(fens, go):
    p = subprocess.Popen([MOON], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, encoding="utf-8", errors="replace")
    p.stdin.write("uai\nisready\n")
    p.stdin.flush()
    out = []
    for fen in fens:
        p.stdin.write("uainewgame\nposition fen %s\n%s\n" % (fen, go))
        p.stdin.flush()
        last = None
        while True:
            l = p.stdout.readline()
            if l.startswith("info depth"):
                last = l
            if l.startswith("bestmove"):
                break
        m = re.search(r"depth\s+(\d+).*?nodes\s+(\d+)", last or "")
        out.append((int(m.group(1)), int(m.group(2))) if m else (0, 0))
    p.stdin.write("quit\n")
    p.stdin.flush()
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    z = np.load("C:/gitProjects/titanium-ataxx/data/nnue/final/final_val_v3.npz")
    o0, o1, st, blk = z["occ0"][:200000], z["occ1"][:200000], z["stm"][:200000], z["blk"][:200000]
    emp = 49 - np.array([bin(int(a) | int(b)).count("1") for a, b in zip(o0, o1)])
    idx = np.random.default_rng(2).choice(np.nonzero((emp >= 14) & (emp <= 34) & (blk == 0))[0], n, replace=False)
    boards = [board_str(int(o0[j]), int(o1[j]), 0, int(st[j])) for j in idx]
    fens = [to_fen((int(o0[j]), int(o1[j]), 0, int(st[j]))) for j in idx]
    def ti(b, extra):
        # fresh process per position (clean TT, like Moonbird's uainewgame)
        r = subprocess.run([TI, "label", "--tuple", TUP] + extra, input=b + chr(10), capture_output=True, text=True).stdout.split()
        return int(r[1]), int(r[2])
    ti_t = [ti(b, ["--time", str(ms)]) for b in boards]
    ti_n = [ti(b, ["--nodes", str(nodes)]) for b in boards]
    mb_t = moon(fens, "go movetime %d" % ms)
    mb_n = moon(fens, "go nodes %d" % nodes)
    f = lambda v, k: np.mean([x[k] for x in v])
    print("%d positions (14-34 empties)" % n)
    print("  at %d ms : Titanium depth %.2f nodes %.0f | Moonbird depth %.2f nodes %.0f" % (ms, f(ti_t, 0), f(ti_t, 1), f(mb_t, 0), f(mb_t, 1)))
    print("  at %d nodes: Titanium depth %.2f | Moonbird depth %.2f" % (nodes, f(ti_n, 0), f(mb_n, 0)))


if __name__ == "__main__":
    main()
