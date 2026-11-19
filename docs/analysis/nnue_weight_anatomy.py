"""D452 weight anatomy of a TNN1/TNN3 net (read-only; numpy only, no training). User: 'report how the weight analysis went,
look into the weights, find what is redundant'.

    python -I docs/analysis/nnue_weight_anatomy.py [data/nnue/nnue_main.tnn]

Layout = titaniumtest src/train.rs from_tnn_bytes. Feature ranges = crates/titanium/src/nnfeat.rs constants.
Reports: (1) weight energy per input group (and per row), dead rows; (2) singular-value spectrum of the FT matrix
(effective dimensions used by the 256 accumulator neurons); (3) us/them redundancy: for every accumulator neuron j, the
L1 weights reading the OWN-side copy vs the ENEMY-side copy (corr; ~-1 = the enemy half is a mirror, ~0 = new information);
(4) L1/L2 weight usage; (5) per-group contribution of the psqt shortcut.
"""
import struct
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/nnue/nnue_main.tnn"
d = open(path, "rb").read()
magic = d[:4]
v2, has_dis = magic == b"TNN2", magic == b"TNN3"
nf, w, l1, l2, nb, groups = struct.unpack_from("<6I", d, 4)
pairwise = (v2 or has_dis) and struct.unpack_from("<I", d, 28)[0] != 0
p = 32 if (v2 or has_dis) else 28


def take(n):
    global p
    a = np.frombuffer(d, dtype="<f4", count=n, offset=p).astype(np.float64)
    p += 4 * n
    return a


din = w if pairwise else 2 * w
out_in = l1 + l2 if v2 else l2
ft = take(nf * w).reshape(nf, w)
ftb = take(w)
psqt = take(nf * nb).reshape(nf, nb)
l1w = take(nb * l1 * din).reshape(nb, l1, din)
l1b = take(nb * l1)
l2w = take(nb * l2 * l1).reshape(nb, l2, l1)
l2b = take(nb * l2)
outw = take(nb * out_in).reshape(nb, out_in)
print(f"{path}: {magic.decode()} feats {nf} width {w} l1 {l1} l2 {l2} buckets {nb} groups {groups:#x} pairwise {pairwise}")

SQ, G1 = 49, 262
G2_OFF = SQ * G1
G2_NONE = G2_OFF + SQ * SQ * 2
G3_OFF = G2_NONE + 1
G5_OFF = G3_OFF + SQ * 2 * 81
G6_OFF = G5_OFF + SQ * 9
G6_PAR = G6_OFF + 2 * 8 * 10
ranges = {"G1 square state": (0, G2_OFF), "G2 anchor x stone": (G2_OFF, G3_OFF), "G3 stone ring": (G3_OFF, G5_OFF),
          "G5 2-ply exchange": (G5_OFF, G6_OFF), "G6 global": (G6_OFF, nf)}
assert G6_PAR + 16 == nf, (G6_PAR, nf)

print("\n(1) FT weight energy by input group (sum of squared weights; rows = features)")
norms = np.sqrt((ft ** 2).sum(1))
tot = (ft ** 2).sum()
for name, (a, b) in ranges.items():
    e = (ft[a:b] ** 2).sum()
    dead = int((norms[a:b] < 1e-6).sum())
    tiny = int((norms[a:b] < 0.02 * np.median(norms[norms > 1e-6])).sum())
    print(f"  {name:<20} rows {b - a:>6} | energy {100 * e / tot:5.1f}% | mean row norm {norms[a:b].mean():.4f} | dead rows {dead:>5} ({100 * dead / (b - a):4.1f}%) | near-zero {tiny}")
print(f"  table: {nf} x {w} = {nf * w * 2 / 1e6:.1f} MB as i16; dead rows overall {int((norms < 1e-6).sum())}")

print("\n(2) singular values of the FT matrix (rows centred): energy captured by the top-k directions")
s = np.linalg.svd(ft - ft.mean(0), compute_uv=False)
e = np.cumsum(s ** 2) / (s ** 2).sum()
for k in (8, 16, 32, 64, 96, 128, 192):
    print(f"  top {k:>3} of {w}: {100 * e[k - 1]:5.1f}%")
eff = (s.sum() ** 2) / (s ** 2).sum()
print(f"  participation ratio (effective number of dimensions): {eff:.1f}")

print("\n(3) OWN vs ENEMY accumulator copies: L1 weights reading neuron j from each side (all buckets x l1 outputs)")
if not pairwise:
    own, enemy = l1w[:, :, :w], l1w[:, :, w:]
    c = np.array([np.corrcoef(own[:, :, j].ravel(), enemy[:, :, j].ravel())[0, 1] for j in range(w)])
    no = np.sqrt((own ** 2).sum((0, 1)))
    ne = np.sqrt((enemy ** 2).sum((0, 1)))
    print(f"  corr(own_j, enemy_j) over neurons: mean {c.mean():+.3f} | median {np.median(c):+.3f} | <-0.8 (mirror): {(c < -0.8).sum()} | -0.8..-0.3: {((c >= -0.8) & (c < -0.3)).sum()} | |c|<0.3 (independent): {(abs(c) < 0.3).sum()} | >0.3: {(c > 0.3).sum()}")
    print(f"  weight norm own {no.sum():.1f} vs enemy {ne.sum():.1f} (ratio enemy/own {ne.sum() / no.sum():.2f})")
    # how well the enemy half is explained by a per-neuron scalar times the own half (least squares, all buckets/outputs)
    num = (own * enemy).sum((0, 1))
    den = (own * own).sum((0, 1)) + 1e-12
    resid = ((enemy - own * (num / den)) ** 2).sum() / (enemy ** 2).sum()
    print(f"  enemy L1 weights explained by (scalar x own) per neuron: {100 * (1 - resid):.1f}% (high = the enemy copy adds little the own copy lacks)")

print("\n(4) hidden layers")
print(f"  L1 |w| mean {abs(l1w).mean():.4f}; fraction |w| < 1e-3: {100 * (abs(l1w) < 1e-3).mean():.1f}%")
print(f"  L2 |w| mean {abs(l2w).mean():.4f}; fraction |w| < 1e-3: {100 * (abs(l2w) < 1e-3).mean():.1f}%")
print(f"  output weights per bucket (mean |w|): {np.round(abs(outw).mean(1), 3).tolist()}")

print("\n(5) psqt shortcut (linear term added to the output) by group: mean |psqt| per row")
for name, (a, b) in ranges.items():
    print(f"  {name:<20} {abs(psqt[a:b]).mean():.4f}")
