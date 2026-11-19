"""Jump share and eval behaviour of candidate vs baseline in ti sprt games.jsonl files (D413 TSTG verification).

    python -I docs/analysis/tstg_jump_share.py <games.jsonl> [...]

Per file: games, candidate points, share of JUMP moves (4-char move, not 0000) for each mover in the first 15 own moves and
overall, mean stone count after ply 30, and the median |candidate score + next baseline score| (sign disagreement of the two
evals on consecutive plies; large = the candidate's eval scale or sign differs from main's).
"""
import json
import statistics
import sys


def stats(path):
    games = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = {"games": len(games), "cand_pts": sum(g.get("candidate_points", 0) for g in games)}
    for who in ("candidate", "baseline"):
        early, allm = [], []
        for g in games:
            own = [m for m in g["moves"] if m["mover"] == who and m["move"] != "0000"]
            early += [len(m["move"]) == 4 for m in own[:15]]
            allm += [len(m["move"]) == 4 for m in own]
        out[who + "_jump_first15"] = sum(early) / max(1, len(early))
        out[who + "_jump_all"] = sum(allm) / max(1, len(allm))
    dis = []
    for g in games:
        mv = g["moves"]
        for a, b in zip(mv, mv[1:]):
            if a["mover"] == "candidate" and b["mover"] == "baseline" and a.get("mate") is None and b.get("mate") is None:
                dis.append(abs(a["score"] + b["score"]))
    out["median_|cand+base|_cp"] = statistics.median(dis) if dis else None
    cs = [m["score"] for g in games for m in g["moves"] if m["mover"] == "candidate" and m.get("mate") is None]
    bs = [m["score"] for g in games for m in g["moves"] if m["mover"] == "baseline" and m.get("mate") is None]
    out["cand_median_score"] = statistics.median(cs) if cs else None
    out["base_median_score"] = statistics.median(bs) if bs else None
    cn = [m["nodes"] for g in games for m in g["moves"] if m["mover"] == "candidate"]
    bn = [m["nodes"] for g in games for m in g["moves"] if m["mover"] == "baseline"]
    cd = [m["depth"] for g in games for m in g["moves"] if m["mover"] == "candidate"]
    bd = [m["depth"] for g in games for m in g["moves"] if m["mover"] == "baseline"]
    out["median_nodes c/b"] = (statistics.median(cn), statistics.median(bn))
    out["median_depth c/b"] = (statistics.median(cd), statistics.median(bd))
    return out


for p in sys.argv[1:]:
    s = stats(p)
    print(p.replace("\\", "/").split("/")[-1])
    print("  " + " | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
