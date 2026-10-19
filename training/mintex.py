"""Mini-Texel for hand-eval weights (tempo/contact/multicap) on blended labels.
Features from train_tuple's exact eval port; grid search (deterministic);
target = 0.5*outcome + 0.5*sigmoid(deep/400)  (lambda-blend).
Usage: python training/mintex.py [--lamb F]
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_tuple as tt

LAMBD = float(sys.argv[sys.argv.index('--lamb') + 1]) if '--lamb' in sys.argv else 0.5

rows = [json.loads(l) for l in open('C:/gitProjects/logs/deep_fresh_out.jsonl')]
n = len(rows)
occ0 = np.zeros(n, dtype=np.uint64)
occ1 = np.zeros(n, dtype=np.uint64)
blk = np.zeros(n, dtype=np.uint64)
stm = np.zeros(n, dtype=np.int64)
for i, d in enumerate(rows):
    parts = d['fen'].split(' ')
    sq = 0
    for ch in parts[0]:
        if ch == '/':
            continue
        if ch.isdigit():
            sq += int(ch)
            continue
        bit = np.uint64(1) << np.uint64(sq)
        if ch == 'x':
            occ0[i] |= bit
        elif ch == 'o':
            occ1[i] |= bit
        elif ch == '#':
            blk[i] |= bit
        sq += 1
    stm[i] = 0 if parts[1] == 'x' else 1

# --- features (stm-relative), mirroring base_eval_stm ---
c0 = tt.popcnt(occ0).astype(np.float64)
c1 = tt.popcnt(occ1).astype(np.float64)
mat = 100.0 * (c0 - c1)
pst = np.zeros(n)
for sq in range(49):
    bit = ((occ0 >> np.uint64(sq)) & np.uint64(1)).astype(np.float64) \
        - ((occ1 >> np.uint64(sq)) & np.uint64(1)).astype(np.float64)
    pst += float(tt.PST[sq]) * bit
empty = ~(occ0 | occ1 | blk) & tt.FULL
lb = empty & (tt.dist_union(occ1, 1) | tt.jump_union(occ1))
lw = empty & (tt.dist_union(occ0, 1) | tt.jump_union(occ0))
cb = tt.popcnt(occ0 & tt.dist_union(lb, 1)).astype(np.float64)
cw = tt.popcnt(occ1 & tt.dist_union(lw, 1)).astype(np.float64)
mb = np.zeros(n)
mw = np.zeros(n)
for sq in range(49):
    isb = ((lb >> np.uint64(sq)) & np.uint64(1)).astype(np.float64)
    isw = ((lw >> np.uint64(sq)) & np.uint64(1)).astype(np.float64)
    nb = tt.popcnt(np.uint64(tt.RING1[sq]) & occ0).astype(np.float64)
    nw = tt.popcnt(np.uint64(tt.RING1[sq]) & occ1).astype(np.float64)
    mb += isb * np.clip(nb - 1, 0, None)
    mw += isw * np.clip(nw - 1, 0, None)
flip = np.where(stm == 1, -1.0, 1.0)
base = (mat + pst) * flip          # fixed part (no tempo/E2/E4)
feat_c = (-cb + cw) * flip         # E2 stm-relative count
feat_m = (-mb + mw) * flip         # E4 stm-relative exposure
deep = np.array([float(d['deep']) for d in rows])
ov = np.array([float(d['outcome']) for d in rows])
targ = LAMBD * ov + (1.0 - LAMBD) / (1.0 + np.exp(-deep / 400.0))


def loss(t, c, m):
    s = 1.0 / (1.0 + np.exp(-(base + t + c * feat_c + m * feat_m) / 400.0))
    return float(np.mean((s - targ) ** 2))


l0 = loss(150.0, 15.0, 20.0)
print('lamb=%.2f n=%d default-loss=%.6f' % (LAMBD, n, l0), flush=True)

best = (l0, 150.0, 15.0, 20.0)
for t in range(0, 401, 20):
    # vectorize (c, m) mesh per tempo
    cs = np.arange(0, 41, 2, dtype=np.float64)
    ms = np.arange(0, 41, 2, dtype=np.float64)
    C, M = np.meshgrid(cs, ms, indexing='ij')
    S = 1.0 / (1.0 + np.exp(-(base[None, None, :]
                                 + t + C[:, :, None] * feat_c[None, None, :]
                                 + M[:, :, None] * feat_m[None, None, :]) / 400.0))
    L = np.mean((S - targ[None, None, :]) ** 2, axis=2)
    j = np.unravel_index(np.argmin(L), L.shape)
    if L[j] < best[0]:
        best = (float(L[j]), float(t), float(cs[j[0]]), float(ms[j[1]]))
print('coarse best loss=%.6f tempo=%.0f contact=%.0f multicap=%.0f' % best, flush=True)

# refine around best
_, bt, bc, bm = best
best2 = best
for t in np.arange(max(0, bt - 20), min(400, bt + 20) + 1, 5):
    for c in np.arange(max(0, bc - 4), min(60, bc + 4) + 0.5, 1):
        for m in np.arange(max(0, bm - 4), min(60, bm + 4) + 0.5, 1):
            l = loss(t, c, m)
            if l < best2[0]:
                best2 = (l, float(t), float(c), float(m))
print('REFINED loss=%.6f tempo=%.0f contact=%.0f multicap=%.0f' % best2, flush=True)
print('improvement over defaults: %.2f%%' % (100 * (l0 - best2[0]) / l0))
