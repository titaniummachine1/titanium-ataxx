"""S57 (current main net, tuple TUP7 + hand base eval) scored on the held-out val split with the SAME metrics as
train_nnue.py (mse vs base-stage target, vs teacher-stage target on Kata rows, vs KataGo TD, vs the game result),
so NNUE numbers have a reference. Uses the trainer's own forward (train_stream: features_for + materialize32 +
base_eval_stm, win prob = sigmoid((base + tuple) / 400)).
Usage: python training/nnue_baseline.py [--tup data/nnue/final_v2_wdl_ep8.tup]
"""
import argparse
import ctypes
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_stream as S  # noqa: E402

T = S.T
ROOT = "C:/gitProjects/titanium-ataxx"


def weights(path):
    b0, ws0, wps0, rest0, cc0, pa0 = T.load_tup(path)
    W = np.zeros(S.SP_N, np.float64)
    D = np.zeros(S.DE_N, np.float64)
    W[S.SP_OFF["s"]:S.SP_OFF["s"] + 36 * 81] = ws0.reshape(-1)
    for d, nm in enumerate(["H", "V", "D1", "D2"]):
        W[S.SP_OFF[nm]:S.SP_OFF[nm] + 6561] = wps0[d].reshape(-1)
    if rest0 is not None:
        wq, weu, wet, wmu, wmt = rest0
        W[S.SP_OFF["q2"]:S.SP_OFF["q2"] + 36 * 256] = wq.reshape(-1)
        W[S.SP_OFF["envus"]:S.SP_OFF["envus"] + 35721] = weu.reshape(-1)
        W[S.SP_OFF["envthem"]:S.SP_OFF["envthem"] + 35721] = wet.reshape(-1)
        W[S.SP_OFF["mobus"]:S.SP_OFF["mobus"] + 2500] = wmu.reshape(-1)
        W[S.SP_OFF["mobthem"]:S.SP_OFF["mobthem"] + 2500] = wmt.reshape(-1)
    if cc0 is not None:
        wcu, wct, wvu, wvt = cc0
        D[S.DE_OFF["capnus"]:S.DE_OFF["capnus"] + 9] = wcu.reshape(-1)
        D[S.DE_OFF["capnthem"]:S.DE_OFF["capnthem"] + 9] = wct.reshape(-1)
        D[S.DE_OFF["convus"]:S.DE_OFF["convus"] + 256] = wvu.reshape(-1)
        D[S.DE_OFF["convthem"]:S.DE_OFF["convthem"] + 256] = wvt.reshape(-1)
    if pa0 is not None:
        W[S.SP_OFF["pa"]:S.SP_OFF["pa"] + 80 * 6561] = pa0.reshape(-1)
    return float(b0), W, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tup", default=ROOT + "/data/nnue/final_v2_wdl_ep8.tup")
    a = ap.parse_args()
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    bias, W, D = weights(a.tup)
    z = np.load(ROOT + "/data/nnue/final/final_val_v3.npz")
    o0, o1, blk, stm = (z[k] for k in ("occ0", "occ1", "blk", "stm"))
    o0, o1, blk = o0.astype(np.uint64), o1.astype(np.uint64), blk.astype(np.uint64)
    n = len(stm)
    p = np.zeros(n)
    for lo in range(0, n, 100_000):
        hi = min(n, lo + 100_000)
        f = S.features_for(o0[lo:hi], o1[lo:hi], blk[lo:hi], stm[lo:hi])
        SP, DE = S.materialize32(*f, [0, 1, 2, 3])
        base = T.base_eval_stm(o0[lo:hi], o1[lo:hi], blk[lo:hi], stm[lo:hi])
        cp = W[SP].sum(1) + D[DE].sum(1) + bias + base
        p[lo:hi] = 1 / (1 + np.exp(-cp / 400.0))
        print("%d/%d" % (hi, n), flush=True)
    w = z["wsum"].astype(np.float64)
    out = z["outcome_p"].astype(np.float64)
    ti = 1 / (1 + np.exp(-np.clip(z["titanium_cp"].astype(np.float64), -1500, 1500) / 400))
    base_t = 0.75 * ti + 0.25 * out
    k = z["teacher_kind"] == 1
    teach_t = 0.75 * z["teacher_p3"].astype(np.float64) + 0.25 * out
    kata = z["kata_td_win"][:, 0].astype(np.float64) + 0.5 * z["kata_td_draw"][:, 0].astype(np.float64)
    m = lambda x, y, mask=slice(None): float(np.average((x[mask] - y[mask]) ** 2, weights=w[mask]))
    res = dict(net=os.path.basename(a.tup), base_stage=dict(mse_target=m(p, base_t), mse_outcome=m(p, out),
                                                             mse_kata_td=m(p, kata, k)),
               teacher_stage=dict(mse_target=m(p, teach_t, k), mse_outcome=m(p, out, k), mse_kata_td=m(p, kata, k)))
    print(json.dumps(res, indent=1))
    with open(ROOT + "/data/nnue/nnue_baseline_s57.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
