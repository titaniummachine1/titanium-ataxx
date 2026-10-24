"""What is each net still missing vs KataGo? READ-ONLY. Held-out val Kata rows (sampled), per empties band:
  * MSE vs Kata TD of: S57 (ENGINE-exact, data/nnue/s57/val.f32, fixed main 653d9a1) and an NNUE checkpoint (trainer forward)
  * residual analysis: logit(Kata) - (scale-refit net logit) regressed on each concept group of screen_features.py:
    error a group removes = knowledge the net does NOT have
  * agreement: where the NNUE beats S57 most (by group, by band)
Usage: python training/nets_residual.py --ckpt data/nnue/nnue_teacher_v1_ep1.pt [--n 60000]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_features as SF  # noqa: E402
import train_nnue as N  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=60000)
    a = ap.parse_args()
    va = N.Split("val", "teacher", None)
    rows = np.sort(np.random.default_rng(6).choice(va.rows, min(a.n, len(va.rows)), replace=False))
    sd = torch.load(a.ckpt, weights_only=False)["model"]
    v2 = "s57w" in sd
    W = sd["ft.weight"].shape[1]
    model = N.NNUE2(W=W, pairwise=sd["l1.weight"].shape[1] == W) if v2 else N.NNUE(W=W)
    model.load_state_dict(sd)
    model.eval()
    ys = []
    with torch.no_grad():
        for i in range(0, len(rows), 50000):
            r = rows[i:i + 50000]
            ix, off, bk, _, _ = va.batch(r)
            ys.append(model(ix, off, bk, va.extra(r)[2]).numpy())
    nn = np.concatenate(ys).astype(np.float64)
    s57 = va.s57[rows].astype(np.float64)
    p = va.kata[rows].astype(np.float64)
    z = np.load(ROOT + "/data/nnue/final/final_val_v3.npz")
    o0, o1, st = z["occ0"][rows], z["occ1"][rows], z["stm"][rows]
    us, en = np.where(st == 0, o0, o1).astype(np.uint64), np.where(st == 0, o1, o0).astype(np.uint64)
    emp = 49 - SF.pc(us | en).astype(np.int64)
    print("rows %d (val Kata rows) | net %s (%s, W %d)" % (len(rows), os.path.basename(a.ckpt), "v2" if v2 else "v1", W), flush=True)
    F = SF.build_features(us, en)
    y = SF.logit(p)
    test = np.random.default_rng(4).random(len(rows)) < 0.3
    for lo, hi in ((0, 8),) + tuple(SF.BANDS):
        m = (emp >= lo) & (emp <= hi)
        tr, te = m & ~test, m & test
        if te.sum() < 300:
            continue
        line = "\n=== empties %d-%d (test %d) | MSE vs Kata: S57 %.4f | NNUE %.4f" % (
            lo, hi, te.sum(), SF.scale_fit_mse(s57[tr], p[tr], s57[te], p[te]), SF.scale_fit_mse(nn[tr], p[tr], nn[te], p[te]))
        print(line)
        for name, base in (("S57", s57), ("NNUE", nn), ("S57+NNUE", None)):
            X0 = base[:, None] if base is not None else np.column_stack([s57, nn])
            ref = SF.mse_p(SF.lin_fit_predict(X0[tr], y[tr], X0[te]), p[te])
            out = "   %-9s linear refit %.4f | error removed by adding a group:" % (name, ref)
            for g in SF.GROUPS:
                X, _ = SF.cols(F, [g])
                X = np.column_stack([X0, X])
                out += " %s %.4f" % (g, ref - SF.mse_p(SF.lin_fit_predict(X[tr], y[tr], X[te]), p[te]))
            print(out)


if __name__ == "__main__":
    main()
