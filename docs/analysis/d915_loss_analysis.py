"""D915: why does candidate B lose games vs main? Reads a ti gate games.jsonl (paired book, both seats per opening).
Reports: (1) opening balance as both engines saw it on their first move; (2) whether the opening decided the pair (same colour won
both games); (3) the ply where each game was decided (loser's own eval first <= -200 cp, never back above 0) for candidate wins vs
candidate losses; (4) score of the candidate by game length; (5) the lost-from-even games: loser eval >= 0 at ply 60+ and still lost.
Usage: python -I d915_loss_analysis.py GAMES.jsonl [more.jsonl ...]
"""
import json
import sys
from collections import Counter, defaultdict


def load(paths):
    g = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            g += [json.loads(l) for l in f if l.strip()]
    return g


def own_eval(m):
    if m.get("mate") is not None:
        return 20000 if m["mate"] > 0 else -20000
    return m["score"]


def decided_ply(game, who):
    """first ply from which `who`'s own eval stays <= -200 for the rest of its moves (None if never)."""
    ev = [(m["ply"], own_eval(m)) for m in game["moves"] if m["mover"] == who]
    dp = None
    for ply, e in ev:
        if e <= -200:
            if dp is None:
                dp = ply
        elif e > 0:
            dp = None
    return dp


def main(paths):
    games = load(paths)
    n = len(games)
    pts = sum(g["candidate_points"] for g in games)
    print(f"games {n}, candidate score {pts}/{n} = {pts / n:.3f}")

    # (1) opening balance: first move eval of each engine (mover's view), by book position
    first = defaultdict(list)
    for g in games:
        for m in g["moves"][:2]:
            first[g["book"]].append(own_eval(m) if m["side"] == "x" else -own_eval(m))  # x's view
    ab = sorted(abs(sum(v) / len(v)) for v in first.values())
    print(f"(1) opening eval (x view, mean of both engines' first moves) over {len(ab)} openings: "
          f"median |cp| {ab[len(ab) // 2]:.0f}, 90% {ab[int(len(ab) * .9)]:.0f}, max {ab[-1]:.0f}; "
          f"> 100 cp: {sum(a > 100 for a in ab)}")

    # (2) pair outcomes
    pairs = defaultdict(list)
    for g in games:
        pairs[g["pair"]].append(g)
    kinds = Counter()
    seat = Counter()
    for p, gs in pairs.items():
        if len(gs) != 2:
            continue
        s = int(round(2 * sum(g["candidate_points"] for g in gs)))
        kinds[{0: "LL", 1: "LD/DL", 2: "split", 3: "WD", 4: "WW"}[s]] += 1
        if s == 2:
            # who won each game: the same colour both times = the opening decided it
            winners = []
            for g in gs:
                cand_won = g["candidate_points"] == 1
                winners.append(g["candidate_seat"] if cand_won else ("o" if g["candidate_seat"] == "x" else "x"))
            seat["same colour won both (opening decided)" if winners[0] == winners[1] else "same ENGINE won both"] += 1
    print(f"(2) pairs: {dict(kinds)}; split pairs: {dict(seat)}")
    # opening eval of LL / WW / split pairs
    for kind_s, lab in ((0, "LL"), (4, "WW"), (2, "split")):
        vals = [abs(sum(first[gs[0]['book']]) / len(first[gs[0]['book']])) for gs in pairs.values()
                if len(gs) == 2 and int(round(2 * sum(g["candidate_points"] for g in gs))) == kind_s]
        if vals:
            print(f"    {lab}: {len(vals)} pairs, mean |opening cp| {sum(vals) / len(vals):.0f}")

    # (3) decision ply
    dec = {"cand_won": [], "cand_lost": []}
    for g in games:
        if g["candidate_points"] == 0.5:
            continue
        loser = "baseline" if g["candidate_points"] == 1 else "candidate"
        d = decided_ply(g, loser)
        dec["cand_won" if loser == "baseline" else "cand_lost"].append(d)
    for k, v in dec.items():
        known = sorted(x for x in v if x is not None)
        buckets = Counter(min(x // 20, 4) for x in known)
        lab = ["0-19", "20-39", "40-59", "60-79", "80+"]
        print(f"(3) {k}: {len(v)} games, decided ply median {known[len(known) // 2] if known else '-'}; "
              + ", ".join(f"{lab[b]}: {buckets[b]}" for b in range(5)) + f"; never below -200 for the loser: {len(v) - len(known)}")

    # (4) score by game length
    by_len = defaultdict(lambda: [0, 0])
    for g in games:
        L = len(g["moves"])
        b = "<60" if L < 60 else "60-79" if L < 80 else "80-99" if L < 100 else "100+"
        by_len[b][0] += g["candidate_points"]
        by_len[b][1] += 1
    print("(4) candidate score by game length: " + ", ".join(f"{b} {v[0]}/{v[1]} ({v[0] / v[1]:.3f})"
                                                           for b, v in sorted(by_len.items())))

    # (5) late collapses: the loser's own eval was >= 0 at some ply >= 60 and the game was lost
    late = Counter()
    for g in games:
        if g["candidate_points"] == 0.5:
            continue
        loser = "baseline" if g["candidate_points"] == 1 else "candidate"
        if any(m["mover"] == loser and m["ply"] >= 60 and own_eval(m) >= 0 for m in g["moves"]):
            late[loser] += 1
    print(f"(5) lost although the loser's own eval was >= 0 at ply 60+: candidate {late['candidate']}, baseline {late['baseline']}")


if __name__ == "__main__":
    main(sys.argv[1:])
