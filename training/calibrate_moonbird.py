"""Fit Moonbird cp -> win probability on KataGo rows, so Moonbird and KataGo targets share one scale.

p_moon = sigmoid(cp / s_band), s_band fitted per empties band by weighted least squares against KataGo TD h1
(the best-calibrated label we have, see docs/KATA_FIRST_V3.md). Also prints how good Moonbird is as a label
(MSE vs the real result next to KataGo's own) -- diagnostic only.
Fails (exit 2) if the fit is degenerate: scale at the grid edge, or corr(moon, kata) < 0.2 in any band.
Usage: python training/calibrate_moonbird.py --nodes 1000   -> data/nnue/final/moon1000/calibration.json
"""
import argparse
import glob
import json
import sys

import numpy as np

FINAL = "C:/gitProjects/titanium-ataxx/data/nnue/final"
BANDS = [(0, 8), (9, 15), (16, 24), (25, 35), (36, 49)]
GRID = np.exp(np.linspace(np.log(30), np.log(5000), 400))


def fit(cp, p, w):
    best = None
    for s in GRID:
        q = 1.0 / (1.0 + np.exp(-cp / s))
        e = np.average((q - p) ** 2, weights=w)
        if best is None or e < best[1]:
            best = (s, e)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1000)
    a = ap.parse_args()
    d = "%s/moon%d" % (FINAL, a.nodes)
    fs = sorted(glob.glob(d + "/calib_c*.npz"))
    assert fs, "no calibration chunks in " + d
    row = np.concatenate([np.load(f)["row"] for f in fs])
    cp = np.concatenate([np.load(f)["cp"] for f in fs]).astype(np.float64)
    z = np.load(FINAL + "/final_train_v3.npz")
    assert (z["teacher_kind"][row] == 1).all()
    pk = z["kata_td_win"][row, 0].astype(np.float64) + 0.5 * z["kata_td_draw"][row, 0].astype(np.float64)
    out = z["outcome_p"][row].astype(np.float64)
    w = z["wsum"][row].astype(np.float64)
    occ = z["occ0"][row] | z["occ1"][row]
    emp = 49 - np.array([int(x).bit_count() for x in occ])
    ok = np.isfinite(cp)
    rep = dict(nodes=a.nodes, rows=int(len(row)), nan=int((~ok).sum()), bands=[])
    bad = []
    s_all, e_all = fit(cp[ok], pk[ok], w[ok])
    rep["global_scale"] = float(s_all)
    print("rows %d nan %d global scale %.0f mse_vs_kata %.4f" % (len(row), (~ok).sum(), s_all, e_all))
    print("band     n      scale  mse_vs_kata corr(moon,kata) | vs result: moon  kata")
    for lo, hi in BANDS:
        m = ok & (emp >= lo) & (emp <= hi)
        s, e = fit(cp[m], pk[m], w[m])
        q = 1.0 / (1.0 + np.exp(-cp[m] / s))
        c = float(np.corrcoef(cp[m], pk[m])[0, 1])
        mr = float(np.average((q - out[m]) ** 2, weights=w[m]))
        kr = float(np.average((pk[m] - out[m]) ** 2, weights=w[m]))
        print("%2d-%-2d %7d %8.0f %10.4f %12.3f    | %.4f %.4f" % (lo, hi, m.sum(), s, e, c, mr, kr))
        rep["bands"].append(dict(lo=lo, hi=hi, n=int(m.sum()), scale=float(s), mse_vs_kata=float(e),
                                 corr_moon_kata=c, mse_vs_result_moon=mr, mse_vs_result_kata=kr))
        if s <= GRID[1] or s >= GRID[-2] or c < 0.2:   # Pearson of clipped cp vs p is low by nature (0.4-0.6 seen)
            bad.append((lo, hi, s, c))
    with open(d + "/calibration.json", "w") as f:
        json.dump(rep, f, indent=1)
    if bad:
        print("DEGENERATE FIT in bands:", bad)
        sys.exit(2)
    print("wrote", d + "/calibration.json")


if __name__ == "__main__":
    main()
