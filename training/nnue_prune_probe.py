"""No-training ablation probe for a v1 NNUE (.pt from train_nnue.py): what can be cut before any retrain.
Measured on the val split (mse_target, the metric D92 found to track Elo):
  1. feature-group knock-out (inference without the group's rows);
  2. FT neuron statistics (dead / always saturated / live) and mean-ablation importance of every FT neuron;
  3. greedy structured pruning: drop the k least important neurons, fold their mean activation into the L1 bias
     (exactly what a narrower engine net would compute), for several k;
  4. int8 feature transformer (per-tensor scale) instead of int16.
Usage: python training/nnue_prune_probe.py --pt data/nnue/modal_ckpt/s1c_hg/net.pt [--rows 200000]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import train_nnue as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pt", required=True)
ap.add_argument("--rows", type=int, default=200000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(4)

st = torch.load(a.pt, map_location="cpu", weights_only=False)
sd = st["model"]
W = sd["ftb"].shape[0]
m = T.NNUE(W=W, L1=sd["l1.bias"].shape[0] // T.NB, L2=sd["l2.bias"].shape[0] // T.NB)
m.load_state_dict(sd)
m.eval()
L1, L2, NB = m.L1, m.L2, T.NB

va = T.Split("val", "base", None)
rng = np.random.default_rng(1)
rows = np.sort(rng.choice(len(va.target), size=min(a.rows, len(va.target)), replace=False))
w = va.w[rows]
tgt = va.target[rows]
res = {"pt": a.pt, "rows": int(len(rows)), "width": W}


# neuron ranking uses half A, every pruning result is measured on the other half B (no selection bias)
selA = np.arange(len(rows)) % 2 == 0
selB = ~selA


def mse(p, sel=None):
    if sel is None:
        return float(np.average((p - tgt) ** 2, weights=w))
    return float(np.average((p[sel] - tgt[sel]) ** 2, weights=w[sel]))


def batches(keep=None, bs=50000):
    va.keep = keep
    for i in range(0, len(rows), bs):
        yield i, va.batch(rows[i:i + bs])
    va.keep = None


# activations once: a_us, a_them (n x W, clamped) + psqt term + bucket
A_us = np.empty((len(rows), W), np.float32)
A_th = np.empty((len(rows), W), np.float32)
PS = np.empty(len(rows), np.float32)
BK = va.bucket[rows]
with torch.no_grad():
    for i, (ix, off, bk, _, _) in batches():
        a_ = (m.ft(ix, off) + m.ftb).clamp(0, 1)
        A_us[i:i + len(bk)] = a_[0::2].numpy()
        A_th[i:i + len(bk)] = a_[1::2].numpy()
        ps = m.psqt(ix, off)
        ar = torch.arange(len(bk))
        PS[i:i + len(bk)] = (0.5 * (ps[0::2] - ps[1::2])[ar, bk]).numpy()

W1 = sd["l1.weight"].numpy().reshape(NB, L1, 2 * W)
B1 = sd["l1.bias"].numpy().reshape(NB, L1)
W2 = sd["l2.weight"].numpy().reshape(NB, L2, L1)
B2 = sd["l2.bias"].numpy().reshape(NB, L2)
WO = sd["out.weight"].numpy()
BO = sd["out.bias"].numpy()
# per-row bucket slices of the L1 weights, gathered once (n x L1 x 2W would be too big: use per-bucket groups)
groups = [np.nonzero(BK == b)[0] for b in range(NB)]


def head_pre(Aus, Ath, b1=B1):
    """L1 pre-activation per row (n x L1)."""
    h = np.empty((len(rows), L1), np.float32)
    for b, g in enumerate(groups):
        if len(g):
            h[g] = Aus[g] @ W1[b, :, :W].T + Ath[g] @ W1[b, :, W:].T + b1[b]
    return h


def head_rest(h1pre):
    p = np.empty(len(rows), np.float32)
    for b, g in enumerate(groups):
        if len(g):
            h1 = np.clip(h1pre[g], 0, 1)
            h2 = np.clip(h1 @ W2[b].T + B2[b], 0, 1)
            p[g] = h2 @ WO[b] + BO[b]
    return 1 / (1 + np.exp(-(p + PS)))


H1 = head_pre(A_us, A_th)
base = mse(head_rest(H1))
res["base_mse"] = base
print("base val mse_target %.6f on %d rows (stored val of the checkpoint: %s)" % (base, len(rows), st.get("val")))

# 1. feature-group knock-out
res["group_knockout"] = {}
for gname, (lo, hi) in T.GROUPS.items():
    keep = np.ones(T.NUM_FEATS, bool)
    keep[lo:hi] = False
    p = np.empty(len(rows), np.float32)
    with torch.no_grad():
        for i, b in batches(keep):
            ix, off, bk = b[0], b[1], b[2]
            p[i:i + len(bk)] = torch.sigmoid(m(ix, off, bk)).numpy()
    v = mse(p)
    res["group_knockout"][gname] = v
    print("knock out %s: mse %.6f (%+.1f%%)" % (gname, v, 100 * (v / base - 1)))

# 2. neuron statistics + mean-ablation importance (one FT neuron feeds input j of both perspectives)
both = np.concatenate([A_us, A_th])
dead = (both <= 0).mean(0)
sat = (both >= 1).mean(0)
mean_act = both.mean(0)
res["dead_frac_hist"] = np.histogram(dead, bins=[0, .5, .9, .99, .999, 1.0001])[0].tolist()
res["never_live"] = int(((dead + sat) >= 0.9999).sum())
print("FT neurons: never live (always 0 or always 1) %d / %d; dead-fraction histogram [0,.5,.9,.99,.999,1]: %s"
      % (res["never_live"], W, res["dead_frac_hist"]))
imp = np.empty(W)
baseA, baseB = mse(head_rest(H1), selA), mse(head_rest(H1), selB)
res["baseA"], res["baseB"] = baseA, baseB
for j in range(W):
    d = np.zeros((len(rows), L1), np.float32)
    for b, g in enumerate(groups):
        if len(g):
            d[g] = np.outer(mean_act[j] - A_us[g, j], W1[b, :, j]) + np.outer(mean_act[j] - A_th[g, j], W1[b, :, W + j])
    imp[j] = mse(head_rest(H1 + d), selA) - baseA
order = np.argsort(imp)
res["importance_quantiles"] = np.quantile(imp / baseA * 100, [0, .1, .25, .5, .75, .9, 1]).tolist()
print("per-neuron mean-ablation cost, %% of val: quantiles 0/10/25/50/75/90/100 = %s"
      % " ".join("%.3f" % x for x in res["importance_quantiles"]))

# 3. greedy structured pruning with the constant folded into the L1 bias
res["prune"] = {}
for k in (8, 16, 32, 48, 64, 96, 128, 160, 192, 224):
    cut = order[:k]
    Aus, Ath = A_us.copy(), A_th.copy()
    Aus[:, cut] = mean_act[cut]
    Ath[:, cut] = mean_act[cut]
    p = head_rest(head_pre(Aus, Ath))
    v = mse(p, selB)
    res["prune"][int(W - k)] = [v, mse(p, selA)]
    print("width %3d (drop %3d least important, folded): held-out mse %.6f (%+.2f%%), ranking half %+.2f%%"
          % (W - k, k, v, 100 * (v / baseB - 1), 100 * (mse(p, selA) / baseA - 1)))

# 4. int8 feature transformer (per-tensor scale), bias in the same scale
ftw = m.ft.weight.detach().clone()
ftb = m.ftb.detach().clone()
res["int8"] = {}
for name, qs in (("int16 QA1024 (engine now)", 1024.0), ("int8 max-scale", 127.0 / float(ftw.abs().max())),
                 ("int8 scale 128", 128.0), ("int8 scale 64", 64.0)):
    over = float((ftw.abs() * qs > 127).float().mean()) if "int8" in name else 0.0
    lim = 127 if "int8" in name else 32767
    with torch.no_grad():
        m.ft.weight.copy_((ftw * qs).round().clamp(-lim, lim) / qs)
        m.ftb.copy_((ftb * qs).round() / qs)
        p = np.empty(len(rows), np.float32)
        for i, b in batches():
            p[i:i + len(b[2])] = torch.sigmoid(m(b[0], b[1], b[2])).numpy()
    v = mse(p)
    res["int8"][name] = [v, over]
    print("%-28s scale %7.1f clipped %.4f%%: mse %.6f (%+.2f%%)" % (name, qs, 100 * over, v, 100 * (v / base - 1)))
with torch.no_grad():
    m.ft.weight.copy_(ftw)
    m.ftb.copy_(ftb)

if a.out:
    res["importance"] = imp.tolist()
    res["mean_act"] = mean_act.tolist()
    json.dump(res, open(a.out, "w"), indent=1)
