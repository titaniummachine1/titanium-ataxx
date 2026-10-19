#!/usr/bin/env python3
"""Phase-bucket bisection on PIECES-on-board (monotonic game clock in ataxx).
Tries k = 2,4,8,16 equal-width buckets over observed piece range and reports
balance (min share) + distinctness (neighbor outcome-rate gaps).
Optimal = finest k where every bucket is trainable (>=5%) and neighbors
differ. Run on any npz with occ0/occ1/(out|outcome).
Usage: python training/bisect_phases.py data/nnue/cache12M_clean.npz
"""
import argparse
import os
import sys

import numpy as np


def load(path):
    z = np.load(path, mmap_mode='r')
    keys = list(z.keys())
    n = len(z[keys[0]])
    o0 = np.asarray(z['occ0']).ravel()[:n]
    o1 = np.asarray(z['occ1']).ravel()[:n]
    c0 = np.unpackbits(np.ascontiguousarray(o0).view(np.uint8)).reshape(n, 64).sum(1)
    c1 = np.unpackbits(np.ascontiguousarray(o1).view(np.uint8)).reshape(n, 64).sum(1)
    out = None
    for k in ('outcome', 'out'):
        if k in keys:
            out = np.asarray(z[k]).ravel()[:n].astype(np.float64)
            break
    return (c0 + c1).astype(np.int64), out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    a = ap.parse_args()
    pieces, out, n = load(a.npz)
    lo, hi = int(pieces.min()), int(pieces.max())
    print('%s n=%d pieces range %d..%d' % (os.path.basename(a.npz), n, lo, hi))
    if out is not None:
        # normalize outcome to [0,1] if it looks like [-1,1]
        if out.min() < 0:
            out = (out + 1.0) / 2.0
    for k in (2, 4, 8, 16):
        edges = np.linspace(lo, hi + 1, k + 1, dtype=int)
        print('--- k=%d ---' % k)
        shares, means, ok = [], [], True
        for b in range(k):
            m = (pieces >= edges[b]) & (pieces < edges[b + 1])
            c = int(m.sum())
            shares.append(c / n)
            om = float(out[m].mean()) if out is not None and c else float('nan')
            means.append(om)
            print('  bucket %2d pieces %2d-%2d: n=%8d (%5.1f%%) mean_out=%.3f' %
                  (b, edges[b], edges[b + 1] - 1, c, 100 * c / n, om))
        minshare = min(shares)
        gaps = [abs(means[i + 1] - means[i]) for i in range(k - 1)
                if out is not None]
        print('  min_share=%.1f%% max_neighbor_gap=%.3f' %
              (100 * minshare, max(gaps) if gaps else float('nan')))


if __name__ == '__main__':
    main()
