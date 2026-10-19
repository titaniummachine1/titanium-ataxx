#!/usr/bin/env python3
"""Bucket-count bisection via variance explained (R^2).
For k in 2,4,8,16(,23): bucket positions by piece count (equal width),
compute R^2 = between-bucket variance / total variance of the target.
Optimal k = finest k where (a) every bucket >=5% of rows AND (b) R^2 keeps
climbing (elbow); a k whose extra buckets add no R^2 is wasted.
Target: stm-relative outcome if present else sigmoid(deep/400).
Usage: python training/bisect_r2.py data/nnue/selfplay50k_part1.npz
"""
import argparse
import math
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--ks', default='2,4,8,16')
    a = ap.parse_args()
    z = np.load(a.npz, mmap_mode='r')
    keys = list(z.keys())
    n = len(z[keys[0]])
    o0 = np.asarray(z['occ0']).ravel()[:n]
    o1 = np.asarray(z['occ1']).ravel()[:n]
    pc = np.unpackbits(np.ascontiguousarray(o0).view(np.uint8)).reshape(n, 64).sum(1) \
        + np.unpackbits(np.ascontiguousarray(o1).view(np.uint8)).reshape(n, 64).sum(1)
    pc = pc.astype(np.int64)
    if 'outcome' in keys:
        t = np.asarray(z['outcome']).ravel()[:n].astype(np.float64)
        tgt = 'outcome'
    elif 'deep' in keys:
        d = np.asarray(z['deep']).ravel()[:n].astype(np.float64)
        t = 1.0 / (1.0 + np.exp(-d / 400.0))
        tgt = 'sigmoid(deep)'
    else:
        # fall back: stm-relative material diff as pseudo-target
        t = (np.unpackbits(np.ascontiguousarray(o0).view(np.uint8)).reshape(n, 64).sum(1).astype(np.float64)
             - np.unpackbits(np.ascontiguousarray(o1).view(np.uint8)).reshape(n, 64).sum(1).astype(np.float64))
        tgt = 'matdiff'
    tot_var = float(t.var())
    print('%s n=%d pieces %d..%d target=%s total_var=%.5f' %
          (os.path.basename(a.npz), n, int(pc.min()), int(pc.max()), tgt, tot_var))
    prev_r2 = 0.0
    for k in [int(x) for x in a.ks.split(',')]:
        lo, hi = int(pc.min()), int(pc.max())
        edges = np.linspace(lo, hi + 1, k + 1, dtype=int)
        ssb, shares, means = 0.0, [], []
        ok = True
        detail = []
        for b in range(k):
            m = (pc >= edges[b]) & (pc < edges[b + 1])
            c = int(m.sum())
            shares.append(c / n)
            if c < 0.05 * n:
                ok = False
            mu = float(t[m].mean()) if c else float('nan')
            means.append(mu)
            ssb += c * (mu - float(t.mean())) ** 2 if c else 0.0
            detail.append('%d-%d:%.1f%%/%.3f' % (edges[b], edges[b + 1] - 1, 100 * c / n, mu))
        r2 = ssb / (n * tot_var) if tot_var > 0 else 0.0
        print('k=%2d R2=%.4f dR2=%+.4f min_share=%.1f%% %s %s' %
              (k, r2, r2 - prev_r2, 100 * min(shares),
               'OK' if ok else 'STARVED', ' | '.join(detail)))
        prev_r2 = r2


if __name__ == '__main__':
    main()
