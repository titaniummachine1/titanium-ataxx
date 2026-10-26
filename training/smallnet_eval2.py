"""Calibrated score of futility small nets (D114): every head of every checkpoint is shifted by s cp (the engine's bound
lo + s) and the share of big-net evals the filter skips is read off at FIXED wrong-prune rates, so nets with different
inputs, losses or tau compare on the only thing that matters: speed at equal pruning errors. Same held-out calls and
definitions as smallnet_eval.py (sn_eval.npz: 400 disjoint roots; wrong prunes = % of the moves the big net keeps).
Reference: the deployed net (sn_q32.pt head tau 0.02, shift 0) = 68.8% skipped at 1.43% wrong prunes.
Usage: python training/smallnet_eval2.py a.pt [b.pt ...]   (prints one line per head; --json for the best head)
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import smallnet_train as S  # noqa: E402

RATES = (0.5, 1.0, 1.43, 2.0, 3.0)


def load_eval():
    z = dict(np.load(S.D + "/sn_eval.npz"))
    ex = np.load(S.D + "/sn_eval_ex.npy") if os.path.exists(S.D + "/sn_eval_ex.npy") else None
    return z, ex


def red(alpha, ce):
    x = alpha + ce
    q = np.where(x >= 0, x // 57, -((-x) // 57))
    return np.clip(q, 0, 64)


def score_heads(m, a, z, ex):
    """{tau: {rate: skip%}} for one model (a = its args)"""
    rows = S.Rows(z, np.arange(len(z["nn"])), ex=ex if a.get("extra") else None, alpha_in=bool(a.get("alpha_in")))
    out = np.round(S.evaluate(m, rows) * 400).astype(np.int64)
    if out.ndim == 1:
        out = out[:, None]
    alpha, nn, nd0 = z["alpha"].astype(np.int64), z["nn"].astype(np.int64), z["nd0"].astype(np.int64)
    unc = z["cached"] == 0
    keep = ~(nd0 - red(alpha, nn) <= 0)
    taus = [float(x) for x in a["quantiles"].split(",")] if a.get("quantiles") else [0.5]
    res = {}
    for i, t in enumerate(taus):
        sk, wr = [], []
        for s in range(-400, 401, 5):
            cheap = nd0 - red(alpha, out[:, i] + s) <= 0
            sk.append(100 * (cheap & unc).sum() / unc.sum())
            wr.append(100 * (cheap & keep).sum() / keep.sum())
        sk, wr = np.array(sk), np.array(wr)
        o = np.argsort(wr)
        res[t] = {r: float(np.interp(r, wr[o], sk[o])) for r in RATES}
    return res


def load_model(p):
    st = torch.load(p, map_location="cpu", weights_only=False)
    a = st["args"]
    taus = [float(x) for x in a["quantiles"].split(",")] if a.get("quantiles") else []
    K = (S.NEX if a.get("extra") else 0) + (1 if a.get("alpha_in") else 0)
    m = S.SmallNet(a["width"], a["hidden"], not a["no_hand"], max(1, len(taus)), K)
    m.load_state_dict(st["model"])
    return m, a, st["val"]


def main():
    js = "--json" in sys.argv
    paths = [p for p in sys.argv[1:] if p != "--json"]
    z, ex = load_eval()
    best = {}
    for p in paths:
        m, a, val = load_model(p)
        res = score_heads(m, a, z, ex)
        for t, r in res.items():
            if not js:
                print("%-34s tau %-5g skip%% at wrong-prune " % (os.path.basename(p), t)
                      + "  ".join("%.2f%%: %5.1f" % (k, v) for k, v in r.items()), flush=True)
            if r[1.43] > best.get("skip", -1):
                best = {"path": p, "tau": t, "skip": r[1.43], "all": r}
    if js:
        print(json.dumps(best))


if __name__ == "__main__":
    main()
