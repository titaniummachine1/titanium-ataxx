"""Is S57's endgame gap to Kata KNOWLEDGE or the RULE difference? READ-ONLY.

Kata's self-play mixes end rules (~70% PASSSCORING = Botzone-like 'stuck side loses the empties', ~30% PASSCONTINUE =
classic pass). Its global input carries rule flags; column 2 is on for ~28% of rows (candidate PASSCONTINUE flag).
  1. Toggle ONLY global column 2 on the same positions and measure the change of Kata's value, by empties: if the flag is the
     end rule, the change should be ~0 early and large late.
  2. S57 static eval (trainer forward) vs Kata TD target on raw rows, split by the flag, per empties band: if S57 (classic) is
     much closer on flag rows in the endgame, the gap is largely the rule.
Usage: python training/rule_gap_probe.py [n_rows]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnue_baseline as NB  # noqa: E402
import probe_kata_deep as D  # noqa: E402
import screen_features as SF  # noqa: E402
import train_stream as S  # noqa: E402

T = S.T
ROOT = "C:/gitProjects/titanium-ataxx"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    rng = np.random.default_rng(12)
    fs = sorted(glob.glob(D.KDIR + "/*/tdata/*.npz"))
    PL, GL, P = [], [], []
    for f in [fs[i] for i in rng.choice(len(fs), 16, replace=False)]:
        z = np.load(f)
        pl = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
        gi = z["globalInputNC"]
        g = z["globalTargetsNC"]
        ok = np.nonzero((gi[:, 0] == 0) & (pl[:, 3].sum(1) == 0))[0]
        take = rng.choice(ok, min(n // 16, len(ok)), replace=False)
        PL.append(pl[take])
        GL.append(gi[take])
        P.append(g[take, 4] + 0.5 * g[take, 6])
    PL, GL, P = np.concatenate(PL), np.concatenate(GL).astype(np.float32), np.concatenate(P).astype(np.float64)
    w = 1 << np.arange(49, dtype=np.uint64)
    us = (PL[:, 1].astype(np.uint64) * w).sum(1).astype(np.uint64)
    en = (PL[:, 2].astype(np.uint64) * w).sum(1).astype(np.uint64)
    emp = 49 - PL[:, 1].sum(1) - PL[:, 2].sum(1)
    flag = GL[:, 2] > 0.5
    print("rows %d, flag(col2) on %.3f" % (len(P), flag.mean()))
    # 1. toggle test on a subset
    _, plain, _ = D.build(taps_ffn=(), attn=False)
    sub = rng.choice(len(P), 1500, replace=False)
    X = PL[sub].astype(np.float32).reshape(-1, 22, 7, 7)
    g0, g1 = GL[sub].copy(), GL[sub].copy()
    g0[:, 2] = 0
    g1[:, 2] = 1
    v0 = np.concatenate([D.value_of(plain.run(["out_value"], {"input_spatial": X[i:i + 50], "input_global": g0[i:i + 50]})[0]) for i in range(0, len(sub), 50)])
    v1 = np.concatenate([D.value_of(plain.run(["out_value"], {"input_spatial": X[i:i + 50], "input_global": g1[i:i + 50]})[0]) for i in range(0, len(sub), 50)])
    d = np.abs(v1 - v0)
    print("\n1. Kata value change when ONLY global column 2 is toggled (mean |dV|, V = P(win)-P(loss)):")
    for lo, hi in ((0, 8), (9, 15), (16, 24), (25, 35), (36, 49)):
        m = (emp[sub] >= lo) & (emp[sub] <= hi)
        if m.sum():
            print("   empties %2d-%2d  %.4f  (n=%d)" % (lo, hi, d[m].mean(), m.sum()))
    # 2. S57 static on the same raw rows
    bias, W, Dw = NB.weights(ROOT + "/data/nnue/final_v2_wdl_ep8.tup")
    zero = np.zeros(len(us), np.uint64)
    stm = np.zeros(len(us), np.int64)
    cp = np.zeros(len(us))
    for lo in range(0, len(us), 20000):
        hi = min(len(us), lo + 20000)
        f = S.features_for(us[lo:hi], en[lo:hi], zero[lo:hi], stm[lo:hi])
        SP, DE = S.materialize32(*f, [0, 1, 2, 3])
        base = T.base_eval_stm(us[lo:hi], en[lo:hi], zero[lo:hi], stm[lo:hi])
        cp[lo:hi] = W[SP].sum(1) + Dw[DE].sum(1) + bias + base
    s57 = cp / 400.0
    test = rng.random(len(P)) < 0.3
    print("\n2. S57 static vs Kata TD (MSE in win prob, scale refit per group), flag off / flag on:")
    for lo, hi in ((0, 8), (9, 15), (16, 24), (25, 35), (36, 49)):
        line = "   empties %2d-%2d " % (lo, hi)
        for fl in (False, True):
            m = (emp >= lo) & (emp <= hi) & (flag == fl)
            tr, te = m & ~test, m & test
            if te.sum() < 100:
                line += "  n/a      "
                continue
            line += "  %s %.4f (n=%d, const %.4f)" % ("on " if fl else "off", SF.scale_fit_mse(s57[tr], P[tr], s57[te], P[te]), te.sum(),
                                                     float(((P[te] - P[tr].mean()) ** 2).mean()))
        print(line)


if __name__ == "__main__":
    main()
