"""Score a futility small net (smallnet_train.py) as the pre-filter of the futility step on held-out recorded calls
(sn_eval.npz: every futility call of 400 roots disjoint from the training roots, search byte-identical to main).
Reports the residual vs the big net and, per margin M, the share of big-net evals the filter skips and the share of the
moves the big net keeps that the filter prunes wrongly; the hand+parent filter of exp/fut-cheap (D108) is the baseline.
Usage: python training/smallnet_eval.py data/nnue/smallnet/sn_w32.pt [more .pt]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import smallnet_train as S  # noqa: E402

z = dict(np.load(S.D + "/sn_eval.npz"))
alpha, nn, cached, nd0 = z["alpha"].astype(np.int64), z["nn"].astype(np.int64), z["cached"], z["nd0"].astype(np.int64)
tz = np.load(S.D + "/sn_train.npz")
seen = np.isin(z["own"] ^ (z["opp"] * np.uint64(0x9E3779B97F4A7C15)), tz["own"] ^ (tz["opp"] * np.uint64(0x9E3779B97F4A7C15)))
print("eval calls %d, of them in the training set (same position) %.1f%%" % (len(nn), 100 * seen.mean()))


def red(ce):
    x = alpha + ce
    q = np.where(x >= 0, x // 57, -((-x) // 57))
    return np.clip(q, 0, 64)


prune = nd0 - red(nn) <= 0
unc = cached == 0


def report(name, pred, margins):
    res = (nn - pred)
    print("\n%s: residual sd %.0f cp, |res| q90 %.0f q99 %.0f" % (name, res.std(), np.quantile(abs(res), .9), np.quantile(abs(res), .99)))
    for mg in margins:
        cheap = nd0 - red(pred - mg) <= 0
        print("  margin %3d: big-net evals skipped %5.1f%% | wrong prunes %.2f%% of the moves the big net keeps"
              % (mg, 100 * (cheap & unc).sum() / unc.sum(), 100 * (cheap & ~prune).sum() / (~prune).sum()))


hand_pred = (357 * z["hand"].astype(np.int64) - 542 * z["pst"].astype(np.int64)) // 1000 + 248
report("hand+parent (exp/fut-cheap D108)", hand_pred, (300, 350, 400))
rows = S.Rows(z, np.arange(len(nn)))
for p in sys.argv[1:]:
    st = torch.load(p, map_location="cpu", weights_only=False)
    a = st["args"]
    m = S.SmallNet(a["width"], a["hidden"], not a["no_hand"])
    m.load_state_dict(st["model"])
    pred = np.round(S.evaluate(m, rows) * 400).astype(np.int64)
    report("%s (W %d H %d hand %s, val %.5f)" % (p, a["width"], a["hidden"], not a["no_hand"], st["val"]), pred,
           (50, 100, 150, 200, 250, 300))
