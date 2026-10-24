"""What does S57's STATIC eval miss compared with KataGo's value? READ-ONLY.

S57 static eval = the trainer's exact forward (tuple TUP7 + hand base eval, as nnue_baseline.py), on the same 100k
stratified calibration rows as screen_features.py (its cached features are reused). Per empties band:
  1. MSE vs Kata win prob of S57 alone (logistic scale refit), Moonbird static, Ataxx_CPP, Titanium 5k search.
  2. S57 + one feature group as a linear correction fitted on the logit (S57's own output is an input with a free weight):
     the error removed = knowledge S57 does NOT already have. Groups as in screen_features.py.
  3. Single features ranked by error removed on top of S57.
Usage: python training/s57_residual.py [--tup data/nnue/final_v2_wdl_ep8.tup]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnue_baseline as NB  # noqa: E402
import screen_features as SF  # noqa: E402
import train_stream as S  # noqa: E402

T = S.T
ROOT = "C:/gitProjects/titanium-ataxx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tup", default=ROOT + "/data/nnue/final_v2_wdl_ep8.tup")
    a = ap.parse_args()
    fs = sorted(glob.glob(ROOT + "/data/nnue/final/moon1000/calib_c*.npz"))
    row = np.concatenate([np.load(f)["row"] for f in fs])
    rng = np.random.default_rng(3)                                   # identical row selection to screen_features.py
    row = np.sort(rng.choice(row, min(100000, len(row)), replace=False))
    z = np.load(ROOT + "/data/nnue/final/final_train_v3.npz")
    row = row[z["blk"][row] == 0]
    o0, o1, blk, stm = (z[k][row] for k in ("occ0", "occ1", "blk", "stm"))
    o0, o1, blk = o0.astype(np.uint64), o1.astype(np.uint64), blk.astype(np.uint64)
    p = z["kata_td_win"][row, 0].astype(np.float64) + 0.5 * z["kata_td_draw"][row, 0].astype(np.float64)
    ataxx = z["ataxx_p_win"][row].astype(np.float64) + 0.5 * z["ataxx_p_draw"][row].astype(np.float64)
    ti5k = np.clip(z["titanium_cp"][row].astype(np.float64), -1500, 1500) / 400.0
    bias, W, D = NB.weights(a.tup)
    cp = np.zeros(len(row))
    for lo in range(0, len(row), 50_000):
        hi = min(len(row), lo + 50_000)
        f = S.features_for(o0[lo:hi], o1[lo:hi], blk[lo:hi], stm[lo:hi])
        SP, DE = S.materialize32(*f, [0, 1, 2, 3])
        base = T.base_eval_stm(o0[lo:hi], o1[lo:hi], blk[lo:hi], stm[lo:hi])
        cp[lo:hi] = W[SP].sum(1) + D[DE].sum(1) + bias + base
    s57 = cp / 400.0
    print("rows %d | S57 static: corr with Kata logit %.3f, with Titanium 5k search %.3f" % (
        len(row), np.corrcoef(s57, SF.logit(p))[0, 1], np.corrcoef(s57, ti5k)[0, 1]))
    cache = ROOT + "/data/nnue/screen_feats_100000.npz"
    zz = np.load(cache)
    F = {k: zz[k] for k in zz.files}
    assert len(F["material"]) == len(row), "feature cache rows differ"
    us = np.where(stm == 0, o0, o1)
    en = np.where(stm == 0, o1, o0)
    emp = 49 - SF.pc(us | en).astype(np.int64)
    y = SF.logit(p)
    test = np.zeros(len(row), bool)
    test[np.random.default_rng(4).choice(len(row), len(row) // 4, replace=False)] = True
    groups = list(SF.GROUPS)
    for lo, hi in SF.BANDS:
        m = (emp >= lo) & (emp <= hi)
        tr, te = m & ~test, m & test
        print("\n================ empties %d-%d (train %d, test %d) ================" % (lo, hi, tr.sum(), te.sum()))
        mt = (F["moon_pairs"] + F["moon_env"] + F["moon_mob"] + F["moon_turn"]) / SF.AM.K * np.log(10)
        print("   %-40s %.4f" % ("S57 STATIC eval (scale refit)", SF.scale_fit_mse(s57[tr], p[tr], s57[te], p[te])))
        print("   %-40s %.4f" % ("Moonbird static eval (scale refit)", SF.scale_fit_mse(mt[tr], p[tr], mt[te], p[te])))
        print("   %-40s %.4f" % ("Titanium 5k search (scale refit)", SF.scale_fit_mse(ti5k[tr], p[tr], ti5k[te], p[te])))
        print("   %-40s %.4f" % ("Ataxx_CPP raw value", float(((ataxx[te] - p[te]) ** 2).mean())))
        X0 = s57[:, None]
        base = SF.mse_p(SF.lin_fit_predict(X0[tr], y[tr], X0[te]), p[te])
        print("   %-40s %.4f" % ("S57 static, linear refit (reference)", base))
        for g in groups:
            X, _ = SF.cols(F, [g])
            X = np.column_stack([X0, X])
            print("   S57 + %-33s %.4f   (removes %.4f)" % (g, SF.mse_p(SF.lin_fit_predict(X[tr], y[tr], X[te]), p[te]),
                                                        base - SF.mse_p(SF.lin_fit_predict(X[tr], y[tr], X[te]), p[te])))
        X, _ = SF.cols(F, groups)
        X = np.column_stack([X0, X])
        allm = SF.mse_p(SF.lin_fit_predict(X[tr], y[tr], X[te]), p[te])
        print("   %-40s %.4f   (removes %.4f)" % ("S57 + ALL groups (linear)", allm, base - allm))
        Xt = SF.gbm_fit_predict(X[tr], y[tr], X[te])
        print("   %-40s %.4f" % ("S57 + ALL groups (trees)", SF.mse_p(Xt, p[te])))
        single = []
        for k in F:
            if F[k].std() == 0:
                continue
            X = np.column_stack([X0, F[k]])
            single.append((base - SF.mse_p(SF.lin_fit_predict(X[tr], y[tr], X[te]), p[te]), k))
        single.sort(reverse=True)
        print("   top single features on top of S57: " + ", ".join("%s %.4f" % (k, v) for v, k in single[:10]))


if __name__ == "__main__":
    main()
