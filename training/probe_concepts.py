"""Concept probe: which board concepts explain each engine's value?

Positions: the 100k stratified KataGo calibration rows (relabel_moonbird.py --part calib), where we have
  kata   = KataGo 200-visit search value (TD h1)           [strongest]
  ataxx  = Ataxx_CPP raw net value (1 forward pass)
  moon   = Moonbird 1.1.0 static eval (exact re-implementation, analyze_moonbird.py)
  moon1k = Moonbird 1k-node search
  ti5k   = Titanium 5k-node search (titanium_cp)
Each target is put on a logit scale. Per empties band we fit least squares on standardized concepts and report:
  R2 of material alone, R2 of all concepts, and for each concept the R2 LOST when it is left out (unique contribution).
Concepts are computed backwards with whole-board bit operations (no move generation), side-to-move relative.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import analyze_moonbird as AM  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
FULL = np.uint64((1 << 49) - 1)
pc = np.bitwise_count


def ring_masks(dist):
    out = []
    for s in range(49):
        r, f = divmod(s, 7)
        m = 0
        for rr in range(7):
            for ff in range(7):
                if max(abs(rr - r), abs(ff - f)) == dist:
                    m |= 1 << (rr * 7 + ff)
        out.append(np.uint64(m))
    return out


R1, R2 = ring_masks(1), ring_masks(2)


def spread(b, rings):
    out = np.zeros_like(b)
    for s in range(49):
        has = (b >> np.uint64(s)) & np.uint64(1)
        out |= np.where(has == 1, rings[s], np.uint64(0))
    return out


def concepts(us, en):
    occ = us | en
    empty = ~occ & FULL
    clone_u, clone_e = spread(us, R1) & empty, spread(en, R1) & empty
    jump_u = spread(us, R2) & empty & ~clone_u
    jump_e = spread(en, R2) & empty & ~clone_e
    land_u, land_e = clone_u | jump_u, clone_e | jump_e
    cap_u = np.zeros(len(us), np.int64)      # best conversion count on a square I can land on
    cap_e = np.zeros(len(us), np.int64)
    capc_u = np.zeros(len(us), np.int64)     # best by clone
    att_u = np.zeros(len(us), np.int64)      # sum of conversion values over my landing squares
    att_e = np.zeros(len(us), np.int64)
    for s in range(49):
        bit = np.uint64(1) << np.uint64(s)
        gu = pc(en & R1[s]).astype(np.int64)  # what I convert landing on s
        ge = pc(us & R1[s]).astype(np.int64)  # what the enemy converts landing on s
        lu = (land_u & bit) != 0
        le = (land_e & bit) != 0
        cap_u = np.maximum(cap_u, np.where(lu, gu, 0))
        cap_e = np.maximum(cap_e, np.where(le, ge, 0))
        capc_u = np.maximum(capc_u, np.where((clone_u & bit) != 0, gu, 0))
        att_u += np.where(lu, gu, 0)
        att_e += np.where(le, ge, 0)
    exposed_u = pc(us & spread(land_e, R1)).astype(np.int64)   # my stones next to a square the enemy can land on
    exposed_e = pc(en & spread(land_u, R1)).astype(np.int64)
    safe_u = pc(us & ~spread(empty, R1)).astype(np.int64)      # no empty neighbour: cannot be converted now
    safe_e = pc(en & ~spread(empty, R1)).astype(np.int64)
    only_u = pc(land_u & ~land_e).astype(np.int64)             # empties only I can reach ("territory")
    only_e = pc(land_e & ~land_u).astype(np.int64)
    nu, ne, nemp = pc(us).astype(np.int64), pc(en).astype(np.int64), pc(empty).astype(np.int64)
    return {
        "material": nu - ne,
        "clone_reach": pc(clone_u).astype(np.int64) - pc(clone_e).astype(np.int64),
        "jump_reach": pc(jump_u).astype(np.int64) - pc(jump_e).astype(np.int64),
        "best_cap_me": cap_u, "best_cap_opp": cap_e, "best_clone_cap_me": capc_u,
        "attack_sum": att_u - att_e,
        "exposed": exposed_u - exposed_e,
        "safe": safe_u - safe_e,
        "only_reach": only_u - only_e,
        "stuck_me": (pc(land_u) == 0).astype(np.int64), "stuck_opp": (pc(land_e) == 0).astype(np.int64),
        "parity": nemp % 2,
    }


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def r2(X, y, w):
    X1 = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X1 * sw[:, None], y * sw, rcond=None)
    res = y - X1 @ beta
    return 1 - np.average(res ** 2, weights=w) / np.average((y - np.average(y, weights=w)) ** 2, weights=w)


def main():
    fs = sorted(glob.glob(ROOT + "/data/nnue/final/moon1000/calib_c*.npz"))
    row = np.concatenate([np.load(f)["row"] for f in fs])
    cp1k = np.concatenate([np.load(f)["cp"] for f in fs]).astype(np.float64)
    z = np.load(ROOT + "/data/nnue/final/final_train_v3.npz")
    stm = z["stm"][row]
    o0, o1 = z["occ0"][row], z["occ1"][row]
    us, en = np.where(stm == 0, o0, o1), np.where(stm == 0, o1, o0)
    w = z["wsum"][row].astype(np.float64)
    C = concepts(us, en)
    F = AM.families(AM.to_moon(us), AM.to_moon(en))
    targets = {
        "kata": logit(z["kata_td_win"][row, 0].astype(np.float64) + 0.5 * z["kata_td_draw"][row, 0]),
        "ataxx": logit(z["ataxx_p_win"][row].astype(np.float64) + 0.5 * z["ataxx_p_draw"][row]),
        "moon": sum(F.values()) / AM.K * np.log(10),
        "moon1k": np.clip(cp1k, -3000, 3000) / AM.K * np.log(10),
        "ti5k": np.clip(z["titanium_cp"][row].astype(np.float64), -1500, 1500) / 400.0,
    }
    names = list(C)
    X = np.column_stack([C[k] for k in names]).astype(np.float64)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    emp = 49 - pc(us | en).astype(np.int64)
    for lo, hi in [(0, 8), (9, 15), (16, 24), (25, 35), (36, 49)]:
        m = (emp >= lo) & (emp <= hi)
        print("\n=== empties %d-%d (%d rows): R2 material-only / all concepts; then R2 lost when a concept is left out"
              % (lo, hi, m.sum()))
        print("%-18s" % "concept" + "".join("%9s" % t for t in targets))
        full = {t: r2(X[m], y[m], w[m]) for t, y in targets.items()}
        mat = {t: r2(X[m][:, [0]], y[m], w[m]) for t, y in targets.items()}
        print("%-18s" % "material only" + "".join("%9.3f" % mat[t] for t in targets))
        print("%-18s" % "ALL concepts" + "".join("%9.3f" % full[t] for t in targets))
        for j, nm in enumerate(names):
            keep = [i for i in range(len(names)) if i != j]
            lost = {t: full[t] - r2(X[m][:, keep], y[m], w[m]) for t, y in targets.items()}
            print("%-18s" % nm + "".join("%9.3f" % lost[t] for t in targets))


if __name__ == "__main__":
    main()
