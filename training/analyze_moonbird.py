"""Moonbird 1.1.0 eval (tag 1.1.0 = our test binary), re-implemented in numpy from its evaluation.nim + params, then ablated.

Families (perspective of the side to move, Moonbird units, k = 567.89 per factor 10 in odds):
  pairs : 80 pairs of 2x2 blocks (H +2, V +14, D +16, A +12), quaternary cells, position-specific tables
  env   : every stone: (own neighbours, enemy neighbours, blockers) in its 8-ring, per square; + for stm stones, - for opp
  mob   : empty squares within distance 2 of a side's stones (clone + jump reach), per phase (= occupied count)
  turn  : constant side-to-move bonus
Positions: the 100k stratified KataGo calibration rows (relabel_moonbird.py --part calib) -> we also have Moonbird's own
1k-node search score there, which validates the re-implementation (static eval must correlate strongly with it).
Ablation: drop one family, refit the logistic scale against KataGo TD h1, report the error increase per empties band.
"""
import glob

import numpy as np

ROOT = "C:/gitProjects/titanium-ataxx"
# Moonbird 1.1.0 = the binary we actually play (tag 1.1.0): ternary tables, env without blockers, 2.58 MB.
# (data/engines/moonbird_src is a NEWER unreleased checkout with 4-state cells and a 25.7 MB file: not the engine we test.)
P = np.fromfile(ROOT + "/data/engines/moonbird_1.1.0/res/params/default.bin", dtype="<i2").astype(np.float64)
assert P.size == 4 * 49 * 6561 + 49 * 81 + 2500 + 1
PST = P[:4 * 49 * 6561].reshape(4, 49, 6561)
R = P[4 * 49 * 6561:]
ENV = R[:49 * 81].reshape(49, 9, 9)
MOB = R[49 * 81:49 * 81 + 2500].reshape(50, 50)
TURN = R[-1]
K = 567.89


def to_moon(b):
    """our index row*7+col (row 0 = top) -> Moonbird a1=0 (rank 1 = bottom): (6-row)*7+col"""
    out = np.zeros_like(b)
    for i in range(49):
        r, c = divmod(i, 7)
        out |= ((b >> np.uint64(i)) & np.uint64(1)) << np.uint64((6 - r) * 7 + c)
    return out


def bit(b, s):
    return ((b >> np.uint64(s)) & np.uint64(1)).astype(np.int64)


FILE_A = sum(1 << (r * 7) for r in range(7))
FILE_G = sum(1 << (r * 7 + 6) for r in range(7))
FULL = (1 << 49) - 1


def singles(b):
    b = b.astype(np.uint64)
    right = ((b << np.uint64(1)) | (b << np.uint64(8)) | (b >> np.uint64(6))) & np.uint64(~FILE_A & FULL)
    left = ((b >> np.uint64(1)) | (b << np.uint64(6)) | (b >> np.uint64(8))) & np.uint64(~FILE_G & FULL)
    return (right | left | (b << np.uint64(7)) | (b >> np.uint64(7))) & np.uint64(FULL)


def popc(x):
    x = x.astype(np.uint64)
    c = np.zeros(len(x), np.int64)
    for i in range(49):
        c += ((x >> np.uint64(i)) & np.uint64(1)).astype(np.int64)
    return c


RING1 = [int(singles(np.array([1 << s], np.uint64))[0]) for s in range(49)]


def families(us, en):
    n = len(us)
    occ = us | en
    # level-1 2x2 indices for anchors with file<=5 and rank<=5, cells [a2,b2,a1,b1] = offsets +7,+8,0,+1
    L = np.zeros((n, 49), np.int64)
    for f in range(6):
        for r in range(6):
            s = r * 7 + f
            idx = np.zeros(n, np.int64)
            mult = 1
            for off in (7, 8, 0, 1):
                idx += mult * (bit(us, s + off) * 1 + bit(en, s + off) * 2)
                mult *= 3
            L[:, s] = idx
    pairs = np.zeros(n)
    specs = [(0, 2, [r * 7 + f for r in range(6) for f in range(4)]),
             (1, 14, [r * 7 + f for r in range(4) for f in range(6)]),
             (2, 16, [r * 7 + f for r in range(4) for f in range(4)]),
             (3, 12, [r * 7 + f for r in range(4) for f in range(2, 6)])]
    for d, off, sqs in specs:
        for s in sqs:
            pairs += PST[d, s][L[:, s] + 81 * L[:, s + off]]
    env = np.zeros(n)
    for s in range(49):
        ring = np.uint64(RING1[s])
        for own, foe, sign in ((us, en, 1.0), (en, us, -1.0)):
            has = bit(own, s) == 1
            o, e = popc(own & ring), popc(foe & ring)
            env += sign * np.where(has, ENV[s][o, e], 0.0)
    ph = np.clip(popc(occ), 0, 49)
    empty = ~occ & np.uint64(FULL)
    mob = MOB[ph, popc(singles(singles(us)) & empty)] - MOB[ph, popc(singles(singles(en)) & empty)]
    return dict(pairs=pairs, env=env, mob=mob, turn=np.full(n, TURN))


def fit_mse(x, p, w):
    best = None
    for s in np.exp(np.linspace(np.log(50), np.log(5000), 200)):
        e = np.average((1 / (1 + np.exp(-x / s)) - p) ** 2, weights=w)
        if best is None or e < best[0]:
            best = (e, s)
    return best


def main():
    fs = sorted(glob.glob(ROOT + "/data/nnue/final/moon1000/calib_c*.npz"))
    row = np.concatenate([np.load(f)["row"] for f in fs])
    cp1k = np.concatenate([np.load(f)["cp"] for f in fs]).astype(np.float64)
    z = np.load(ROOT + "/data/nnue/final/final_train_v3.npz")
    o0, o1, stm = z["occ0"][row], z["occ1"][row], z["stm"][row]
    us = np.where(stm == 0, o0, o1)
    en = np.where(stm == 0, o1, o0)
    us, en = to_moon(us), to_moon(en)
    pk = z["kata_td_win"][row, 0].astype(np.float64) + 0.5 * z["kata_td_draw"][row, 0].astype(np.float64)
    w = z["wsum"][row].astype(np.float64)
    emp = 49 - popc(us | en)
    F = families(us, en)
    tot = sum(F.values())
    print("rows %d | validation: corr(static re-impl, Moonbird 1k search) = %.3f" % (len(row), np.corrcoef(tot, cp1k)[0, 1]))
    print("family contribution std (Moonbird units):", {k: round(float(v.std()), 1) for k, v in F.items()})
    bands = [(0, 8), (9, 15), (16, 24), (25, 35), (36, 49)]
    print("\nMSE vs KataGo TD h1 after refitting the scale (lower = better); 'drop X' = eval without family X")
    print("band   full    " + "  ".join("drop_%-5s" % k for k in ("pairs", "env", "mob", "turn")) + "  only_mob  only_pairs  material")
    for lo, hi in bands:
        m = (emp >= lo) & (emp <= hi)
        full = fit_mse(tot[m], pk[m], w[m])[0]
        drops = [fit_mse((tot - F[k])[m], pk[m], w[m])[0] for k in ("pairs", "env", "mob", "turn")]
        om = fit_mse((F["mob"] + F["turn"])[m], pk[m], w[m])[0]
        op = fit_mse((F["pairs"] + F["turn"])[m], pk[m], w[m])[0]
        mat = fit_mse((100.0 * (popc(us) - popc(en)) + TURN)[m], pk[m], w[m])[0]
        print("%2d-%-2d %.4f   " % (lo, hi, full) + "  ".join("%.4f    " % d for d in drops) + "  %.4f    %.4f      %.4f" % (om, op, mat))


if __name__ == "__main__":
    main()
