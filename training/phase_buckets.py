#!/usr/bin/env python3
"""Phase buckets on PIECES-on-board (monotonic game clock, user-ordered).
Bucket count is NOT fixed: use --buckets k (equal-width over observed range)
and pick k by the bisect rule (finest k with every bucket >=5% of rows AND
neighboring buckets behaviorally distinct). See bisect_phases.py.
Usage:
  python training/phase_buckets.py --census data/nnue/X.npz
  python training/phase_buckets.py --split data/nnue/X.npz --buckets 8 \
      --outdir data/nnue/phX  (writes b0..b{k-1}.npz + census.txt)
"""
import argparse
import os

import numpy as np


def bucket_edges(pieces, k):
    lo, hi = int(pieces.min()), int(pieces.max())
    edges = np.linspace(lo, hi + 1, k + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(k)]


def popcnt(a):
    a = np.ascontiguousarray(a)
    return np.unpackbits(a.view(np.uint8)).reshape(len(a), 64).sum(1)


def census(path, k=8):
    z = np.load(path, mmap_mode='r')
    n = len(z[list(z.keys())[0]])
    pieces = popcnt(np.asarray(z['occ0']).ravel()[:n]) + \
        popcnt(np.asarray(z['occ1']).ravel()[:n])
    print('%s n=%d pieces %d..%d' % (os.path.basename(path), n,
                                     int(pieces.min()), int(pieces.max())))
    out = []
    for bi, (lo, hi) in enumerate(bucket_edges(pieces, k)):
        m = (pieces >= lo) & (pieces <= hi)
        out.append((bi, lo, hi, int(m.sum())))
        print('  b%d pieces %2d-%2d: %8d (%5.1f%%)' %
              (bi, lo, hi, int(m.sum()), 100 * m.sum() / max(n, 1)))
    return out


def split(path, outdir, k):
    os.makedirs(outdir, exist_ok=True)
    z = np.load(path, mmap_mode='r')
    keys = list(z.keys())
    n = len(z[keys[0]])
    o0 = np.asarray(z['occ0']).ravel()[:n]
    o1 = np.asarray(z['occ1']).ravel()[:n]
    pieces = popcnt(o0) + popcnt(o1)
    with open(os.path.join(outdir, 'census.txt'), 'w') as cf:
        cf.write('source=%s n=%d k=%d key=pieces\n' % (path, n, k))
        for bi, (lo, hi) in enumerate(bucket_edges(pieces, k)):
            m = np.where((pieces >= lo) & (pieces <= hi))[0]
            kw = {kk: np.asarray(z[kk]).ravel()[:n][m] for kk in keys}
            fp = os.path.join(outdir, 'b%d.npz' % bi)
            np.savez_compressed(fp, **kw)
            cf.write('b%d pieces %d-%d: %d (%.1f%%)\n' %
                     (bi, lo, hi, len(m), 100 * len(m) / max(n, 1)))
            print('wrote %s (%d)' % (fp, len(m)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', default='')
    ap.add_argument('--split', default='')
    ap.add_argument('--outdir', default='')
    ap.add_argument('--buckets', type=int, default=8)
    a = ap.parse_args()
    if a.census:
        census(a.census, a.buckets)
    if a.split:
        assert a.outdir
        split(a.split, a.outdir, a.buckets)


if __name__ == '__main__':
    main()
