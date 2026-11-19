"""Missed big captures in REAL Botzone games (user bug report D427).

    python -I docs/analysis/botzone_missed_captures.py logs/botzone/*.json [--min 6] [--gap 4] [--free 3]

Replays each match JSON (?lite=true), Botzone coordinates (x, y), player 0 starts at (0,0)/(6,6), player 1 at (6,0)/(0,6).
For every move of Titanium (and, as a control, of the opponent): best capture available (max enemy stones flipped by landing
on an empty square the mover can reach), what was played, and the opponent's best recapture after the big capture.
A 'missed free capture' = best >= MIN flips, played <= best - GAP, and after taking it the opponent recaptures <= FREE.
Prints per match + totals and every case with Titanium's debug (depth, nodes, score).
"""
import glob
import json
import sys

N = 7


def ring(p, d):
    x, y = p
    for dx in range(-d, d + 1):
        for dy in range(-d, d + 1):
            if max(abs(dx), abs(dy)) == d and 0 <= x + dx < N and 0 <= y + dy < N:
                yield (x + dx, y + dy)


def flips(b, to, side):
    return sum(1 for q in ring(to, 1) if b.get(q) == 1 - side)


def best_capture(b, side):
    best = (-1, None)
    for p, v in b.items():
        if v != side:
            continue
        for d in (1, 2):
            for q in ring(p, d):
                if q not in b:
                    f = flips(b, q, side)
                    if f > best[0]:
                        best = (f, (p if d == 2 else None, q))
    return best


def play(b, frm, to, side):
    b = dict(b)
    if frm is not None and max(abs(frm[0] - to[0]), abs(frm[1] - to[1])) == 2:
        del b[frm]
    b[to] = side
    for q in ring(to, 1):
        if b.get(q) == 1 - side:
            b[q] = side
    return b


def name(p):
    return chr(97 + p[0]) + str(p[1] + 1)


args = sys.argv[1:]
MIN, GAP, FREE = 6, 4, 3
files = []
i = 0
while i < len(args):
    if args[i] in ("--min", "--gap", "--free"):
        v = int(args[i + 1])
        MIN, GAP, FREE = (v, GAP, FREE) if args[i] == "--min" else (MIN, v, FREE) if args[i] == "--gap" else (MIN, GAP, v)
        i += 2
    else:
        files += glob.glob(args[i])
        i += 1

tot = {"titanium": [0, 0, 0], "opponent": [0, 0, 0]}
cases = []
for fn in sorted(files):
    m = json.load(open(fn, encoding="utf-8"))
    names = [p["name"] for p in m["players"]]
    ti = next((k for k, n in enumerate(names) if "titanium" in n.lower()), None)
    if ti is None:
        continue
    b = {(0, 0): 0, (6, 6): 0, (6, 0): 1, (0, 6): 1}
    ply = 0
    for e in m["logs"]:
        for side in (0, 1):
            d = e.get(str(side))
            if not isinstance(d, dict) or "response" not in d:
                continue
            r = d["response"]
            ply += 1
            if r.get("x0", -1) < 0:
                continue  # pass
            frm, to = (r["x0"], r["y0"]), (r["x1"], r["y1"])
            who = "titanium" if side == ti else "opponent"
            bf, bm = best_capture(b, side)
            pf = flips(b, to, side)
            t = tot[who]
            t[0] += 1
            if bf >= MIN and pf <= bf - GAP:
                after = play(b, bm[0] if bm[0] else next(p for p in ring(bm[1], 1) if b.get(p) == side), bm[1], side)
                recap = best_capture(after, 1 - side)[0]
                if recap <= FREE:
                    t[1] += 1
                    nb_ = play(b, frm, to, side)
                    cases.append((fn.replace("\\", "/").split("/")[-1][:24], names[1 - ti][:24], ply, who, f"{name(frm)}{name(to)}" if max(abs(frm[0]-to[0]), abs(frm[1]-to[1])) == 2 else name(to), pf, name(bm[1]), bf, recap, d.get("debug", "")[:60], 49 - len(b)))
            b = play(b, frm, to, side)

for who, (n, miss, _) in tot.items():
    print(f"{who}: moves {n} | missed FREE capture (best >= {MIN}, recapture <= {FREE}, played <= best-{GAP}): {miss} ({100.0 * miss / max(1, n):.2f}%)")
print("\nmatch | opponent | ply | mover | played (flips) | best square (flips) | opp recapture after taking | debug | empties")
for c in cases:
    print(" | ".join(map(str, c)))
