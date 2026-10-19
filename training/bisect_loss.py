#!/usr/bin/env python3
"""Loss-driven bisection of the pieces axis (CART-style, 1D).

Greedy: repeatedly split the interval that yields the biggest MSE drop.
Stop when best gain < min_gain OR either child < min_count.
Reports boundaries = the bisect verdict, plus cumulative R^2.
This is the honest version of "more buckets where it gains intelligence":
splits land where the target actually changes, not on a uniform grid.

Usage: python training/bisect_loss.py data/nnue/cache12M_clean.npz [--target teacher|outcome] [--min-gain 0.0005] [--min-count 50000]
"""
import argparse
import os

import numpy as np


def load(path, target):
    z = np.load(path, mmap_mode='r')
    keys = list(z.keys())
    n = len(z[keys[0]])
    o0 = np.asarray(z['occ0']).ravel()[:n]
    o1 = np.asarray(z['occ1']).ravel()[:n]
    pc = (np.unpackbits(np.ascontiguousarray(o0).view(np.uint8)).reshape(n, 64).sum(1)
          + np.unpackbits(np.ascontiguousarray(o1).view(np.uint8)).reshape(n, 64).sum(1)).astype(np.int64)
    if target == 'outcome':
        key = 'outcome' if 'outcome' in keys else 'out'
        t = np.asarray(z[key]).ravel()[:n].astype(np.float64)
        if t.min() < 0:
            t = (t + 1.0) / 2.0
    else:
        t = np.asarray(z['teacher']).ravel()[:n].astype(np.float64)
    return pc, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--target', default='teacher', choices=['teacher', 'outcome'])
    ap.add_argument('--min-gain', type=float, default=0.0005)
    ap.add_argument('--min-count', type=int, default=50000)
    ap.add_argument('--max-splits', type=int, default=24)
    ap.add_argument('--drop-mates', type=float, default=0.0,
                    help='drop rows with abs(target) >= this (mates are search-handled; eval should not bucket on them)')
    a = ap.parse_args()
    pc, t = load(a.npz, a.target)
    if a.drop_mates > 0:
        keep = np.abs(t) < a.drop_mates
        pc, t = pc[keep], t[keep]
        print('dropped mates: kept %d rows' % len(pc))
    n = len(pc)
    order = np.argsort(pc, kind='stable')
    sp = pc[order].astype(np.float64)
    st = t[order]
    pref = np.zeros(n + 1)
    pref2 = np.zeros(n + 1)
    np.cumsum(st, out=pref[1:])
    np.cumsum(st * st, out=pref2[1:])
    tot_var = (pref2[n] - pref[n] ** 2 / n) / n
    print('%s n=%d target=%s total_var=%.3f' %
          (os.path.basename(a.npz), n, a.target, tot_var), flush=True)

    # intervals as index ranges into sorted arrays; boundaries in pieces
    intervals = [(0, n)]
    bounds = []
    cum_r2 = 0.0
    # distinct piece values change at few indices: evaluate transitions
    # exactly (no stride sampling -- it misses the ~45 change points).
    trans = np.where(np.diff(sp) != 0)[0] + 1
    while len(intervals) - 1 < a.max_splits:
        best = None  # (gain, interval_idx, split_pos, piece_val)
        for ii, (lo, hi) in enumerate(intervals):
            if hi - lo < 2 * a.min_count:
                continue
            mu = (pref[hi] - pref[lo]) / (hi - lo)
            cand = trans[(trans > lo + a.min_count - 1) & (trans < hi - a.min_count + 1)]
            # transition p splits [lo,p) and [p,hi); require both sides big
            cand = [int(p) for p in cand if p - lo >= a.min_count and hi - p >= a.min_count]
            for p in cand:
                lc = p - lo
                lmu = (pref[p] - pref[lo]) / lc
                rc = hi - p
                rmu = (pref[hi] - pref[p]) / rc
                g = lc * (lmu - mu) ** 2 + rc * (rmu - mu) ** 2
                gain = g / (n * tot_var)
                if best is None or gain > best[0]:
                    best = (gain, ii, p, float(sp[p]))
        if best is None or best[0] < a.min_gain:
            print('stop: best gain %.5f < %.4f' %
                  (best[0] if best else -1, a.min_gain))
            break
        gain, ii, bp, pv = best
        cum_r2 += gain
        lo, hi = intervals.pop(ii)
        intervals.extend([(lo, bp), (bp, hi)])
        intervals.sort()
        bounds.append(int(pv))
        print('split %2d at pieces>=%.0f gain=%.5f cumR2=%.4f buckets=%d' %
              (len(bounds), pv, gain, cum_r2, len(intervals)), flush=True)
    bounds = sorted(set(bounds))
    print('VERDICT k=%d boundaries(>=): %s' % (len(bounds) + 1, bounds))


if __name__ == '__main__':
    main()
