"""D484: single-process nps of two engine commands on real mid-game positions (no concurrent games).

Positions = gate game records (games.jsonl: start + moves) cut at given plies. Each engine runs `serve`,
gets `position fen <start> moves ...` + `go movetime T`, and the `info ... nodes N` line is read.
Engines run one after the other (never at the same time), so there is no cache/memory contention.

usage: python -I nps_ingame_pattern_vs_champ.py <games.jsonl> <movetime_ms> <n_games> "<cmdA>" "<cmdB>" [depth]
  with [depth]: `go depth D` instead (D487: nodes-to-depth of two move orderings; movetime_ms is then ignored for the search)
"""
import json, subprocess, sys, re, statistics

games_path, mt, ng = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cmds = sys.argv[4:6]
depth = int(sys.argv[6]) if len(sys.argv) > 6 else 0
go = ("go depth %d\n" % depth) if depth else ("go movetime %d\n" % mt)
plies = [10, 25, 40, 55, 70]
pos = []
for i, line in enumerate(open(games_path)):
    if i >= ng:
        break
    g = json.loads(line)
    mv = [m["move"] for m in g["moves"]]
    for p in plies:
        if p < len(mv) - 2:
            pos.append((p, "fen " + g["start"] + (" moves " + " ".join(mv[:p]) if p else "")))


def run(cmd):
    pr = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    out = []
    for p, ps in pos:
        pr.stdin.write("uainewgame\nposition " + ps + "\n" + go)
        pr.stdin.flush()
        nodes = None
        while True:
            l = pr.stdout.readline()
            if not l:
                break
            m = re.search(r"nodes (\d+)", l)
            if m:
                nodes = int(m.group(1))
            if l.startswith("bestmove") or (nodes is not None and re.match(r"^[a-g][1-7]", l.strip())) or l.strip() == "0000":
                break
        out.append(nodes or 0)
    pr.stdin.write("quit\n")
    pr.stdin.flush()
    pr.wait(timeout=10)
    return out


res = [run(c) for c in cmds]
print("positions", len(pos), "movetime", mt)
by = {}
for (p, _), a, b in zip(pos, *res):
    by.setdefault(p, []).append((a, b))
for p in sorted(by):
    a = sum(x for x, _ in by[p]); b = sum(y for _, y in by[p])
    print("ply %3d: n=%2d  A %7.0f knps  B %7.0f knps  B/A %.3f" % (p, len(by[p]), a / len(by[p]) / mt, b / len(by[p]) / mt, b / max(a, 1)))
ta, tb = sum(res[0]), sum(res[1])
if depth:
    print("FIXED DEPTH %d: total nodes A %d  B %d  B/A %.4f (lower = better ordering)" % (depth, ta, tb, tb / ta))
print("ALL: A %.0f knps  B %.0f knps  B/A %.3f" % (ta / len(pos) / mt, tb / len(pos) / mt, tb / ta))
print("per-position B/A median %.3f" % statistics.median([b / a for a, b in zip(*res) if a]))
