"""What makes a stone valuable to KataGo's value head? COLOUR-FLIP occlusion (occupancy unchanged), READ-ONLY.

Deleting a stone (probe_kata_deep.py values) also creates a hole the mover can land on next turn, so it measures hole
creation as much as stone value. Here every stone is flipped to the other colour instead: the board stays full where it
was full, material moves by 2, and the change of P(win)-P(loss) tells how much THAT stone's colour is worth to Kata.
Grouped by: owner, empty neighbours (safety), attackable (the other side can land next to it), own-colour neighbours
(group support), edge/corner, phase. Positions: Kata's own self-play, exact stored inputs.
Usage: python training/probe_kata_flip.py [n_positions]
"""
import sys

import numpy as np

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P  # noqa: E402
import probe_kata_deep as D  # noqa: E402

FULL = int(P.FULL)
R1 = [int(x) for x in P.R1]
R2 = [int(x) for x in P.R2]


def pc(x):
    return bin(x).count("1")


def land(a, empty):
    m = 0
    for k in range(49):
        if (a >> k) & 1:
            m |= R1[k] | R2[k]
    return m & empty


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    PL, GL, TD, FIN, F2, F3, us_a, en_a = D.load_positions(n, seed=31)
    _, plain, _ = D.build(taps_ffn=(), attn=False)
    rows = []
    for i in range(len(PL)):
        base = D.value_of(plain.run(["out_value"], {"input_spatial": PL[i:i + 1].reshape(1, 22, 7, 7), "input_global": GL[i:i + 1]})[0])[0]
        stones = [s for s in range(49) if PL[i, 1, s] or PL[i, 2, s]]
        batch = np.repeat(PL[i:i + 1], len(stones), 0)
        for j, s in enumerate(stones):
            batch[j, 1, s], batch[j, 2, s] = PL[i, 2, s], PL[i, 1, s]
        v = D.value_of(plain.run(["out_value"], {"input_spatial": batch.reshape(-1, 22, 7, 7),
                                                 "input_global": np.repeat(GL[i:i + 1], len(stones), 0)})[0])
        u, e = int(us_a[i]), int(en_a[i])
        empty = ~(u | e) & FULL
        lu, le = land(u, empty), land(e, empty)
        nemp = pc(empty)
        for j, s in enumerate(stones):
            mine = bool(PL[i, 1, s])
            own, foe = (u, e) if mine else (e, u)
            att = le if mine else lu                  # squares the OTHER side can land on
            rows.append((1 if mine else 0, base - v[j] if mine else v[j] - base, pc(R1[s] & empty), 1 if (R1[s] & att) else 0,
                         pc(R1[s] & own), 1 if (s // 7 in (0, 6) or s % 7 in (0, 6)) else 0, 1 if s in (0, 6, 42, 48) else 0, nemp))
        if (i + 1) % 25 == 0:
            print("flip %d/%d" % (i + 1, len(PL)), flush=True)
    R = np.array(rows, float)
    mine, val, enb, att, sup, edge, corner, nemp = (R[:, k] for k in range(8))
    print("\nvalue of a stone's COLOUR = drop of the owner's P(win)-P(loss) when it is flipped (both sides, owner perspective)")
    print("  mover's stones mean %.3f (n=%d) | enemy stones mean %.3f (n=%d)" % (val[mine == 1].mean(), (mine == 1).sum(), val[mine == 0].mean(), (mine == 0).sum()))
    print("by empty neighbours (0 = fully enclosed):  mover | enemy")
    for k in range(9):
        a, b = (mine == 1) & (enb == k), (mine == 0) & (enb == k)
        if a.sum() + b.sum() < 20:
            continue
        print("  %d   %.3f (n=%4d) | %.3f (n=%4d)" % (k, val[a].mean() if a.any() else np.nan, a.sum(), val[b].mean() if b.any() else np.nan, b.sum()))
    print("by attackable (the other side can land next to it):")
    for f in (0, 1):
        a, b = (mine == 1) & (att == f), (mine == 0) & (att == f)
        print("  attackable=%d   mover %.3f (n=%d) | enemy %.3f (n=%d)" % (f, val[a].mean(), a.sum(), val[b].mean(), b.sum()))
    print("by own-colour neighbours (support):")
    for lo, hi in ((0, 1), (2, 3), (4, 5), (6, 8)):
        a = (sup >= lo) & (sup <= hi)
        print("  %d-%d   %.3f (n=%d)" % (lo, hi, val[a].mean(), a.sum()))
    print("edge %.3f (n=%d) | corner %.3f (n=%d) | interior %.3f (n=%d)" % (val[(edge == 1) & (corner == 0)].mean(), ((edge == 1) & (corner == 0)).sum(),
                                                                         val[corner == 1].mean(), (corner == 1).sum(), val[edge == 0].mean(), (edge == 0).sum()))
    print("by phase (empties):")
    for lo, hi in ((8, 15), (16, 24), (25, 38)):
        a = (nemp >= lo) & (nemp <= hi)
        print("  %d-%d   %.3f (n=%d)" % (lo, hi, val[a].mean(), a.sum()))
    # linear model of a stone's value: how much do safety / attackability / support explain?
    F = [np.ones(len(val)), mine]
    for k in range(1, 9):
        F.append((enb == k).astype(float))
    F += [att, sup, edge, corner, nemp / 49.0]
    F = np.stack(F, 1)
    cut = int(len(val) * 0.7)
    w = np.linalg.lstsq(F[:cut], val[:cut], rcond=None)[0]
    r2 = 1 - ((val[cut:] - F[cut:] @ w) ** 2).mean() / ((val[cut:] - val[cut:].mean()) ** 2).mean()
    print("R2 of stone colour value from (owner, empty-neighbour count, attackable, support, edge, corner, phase): %.3f" % r2)


if __name__ == "__main__":
    main()
