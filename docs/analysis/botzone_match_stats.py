"""Per-move stats of Titanium in a Botzone match JSON (curl -H 'X-Requested-With: XMLHttpRequest' .../match/<id>?lite=true).

    python -I docs/analysis/botzone_match_stats.py logs/botzone/<id>.json

Titanium's debug field is "d<depth> n<nodes> t<ms> s<score cp> <net>/<bz-index>" (hardened driver). Prints the player index,
the result, then depth / nodes / time / score for every Titanium move and the median/max, plus the opponent's time per move.
"""
import json
import statistics
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
players = [p["name"] for p in m["players"]]
me = next(i for i, n in enumerate(players) if "titanium" in n.lower())
opp = 1 - me
fin = m["logs"][-1]
print("players:", players, "| Titanium = player", me)
print("finish:", json.dumps(fin.get("output", {}).get("content")), json.dumps(fin.get("output", {}).get("display")))
rows, otimes = [], []
ply = 0
for e in m["logs"]:
    for side in ("0", "1"):
        if side in e and isinstance(e[side], dict) and "verdict" in e[side]:
            ply += 1
            d = e[side]
            if int(side) == me:
                dbg = d.get("debug", "")
                tok = dbg.split()
                try:
                    depth = int(tok[0][1:]); nodes = int(tok[1][1:]); t = int(tok[2][1:]); sc = int(tok[3][1:])
                except Exception:
                    depth = nodes = t = sc = None
                rows.append((ply, depth, nodes, t, sc, d.get("time"), d.get("verdict"), d.get("raw")))
            else:
                otimes.append(d.get("time"))
print("ply depth nodes eng_ms score botzone_ms verdict move")
for r in rows:
    print(*r)
ok = [r for r in rows if r[1] is not None]
if ok:
    print("titanium moves", len(rows), "| median depth", statistics.median(r[1] for r in ok), "| median nodes", int(statistics.median(r[2] for r in ok)),
          "| median botzone ms", statistics.median(r[5] for r in rows), "| max botzone ms", max(r[5] for r in rows))
    print("verdicts", sorted({r[6] for r in rows}))
if otimes:
    print("opponent moves", len(otimes), "| median ms", statistics.median(otimes), "| max ms", max(otimes))
