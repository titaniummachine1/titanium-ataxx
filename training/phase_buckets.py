#!/usr/bin/env python3
"""Phase census + 4-bucket split by empty squares (position-intrinsic).
P0 open 40-49, P1 early-mid 28-39, P2 late-mid 14-27, P3 end 0-13.
Usage:
  python training/phase_buckets.py --census data/nnue/cache12M_clean.npz
  python training/phase_buckets.py --split data/nnue/cache12M_clean.npz
    --outdir data/nnue/phases12m   (writes p0..p3.npz + census.txt)
"""
import argparse
import os

import numpy as np

BUCKETS = [(40, 49, 'P0-open'), (28, 39, 'P1-emid'),
           (14, 27, 'P2-lmid'), (0, 13, 'P3-end')]


def empties(o0, o1):
    n = len(o0)
    c0 = np.unpackbits(np.ascontiguousarray(o0).view(np.uint8)).reshape(n, 64).sum(1)
    c1 = np.unpackbits(np.ascontiguousarray(o1).view(np.uint8)).reshape(n, 64).sum(1)
    return 49 - c0 - c1


def census(path):
    z = np.load(path, mmap_mode='r')
    o0 = np.asarray(z['occ0']).ravel()
    o1 = np.asarray(z['occ1']).ravel()
    n = len(o0)
    e = empties(o0, o1)
    print('%s n=%d' % (os.path.basename(path), n))
    out = {}
    for lo, hi, nm in BUCKETS:
        m = (e >= lo) & (e <= hi)
        out[nm] = int(m.sum())
        print('  %s empties %2d-%2d: %8d (%5.1f%%)' %
              (nm, lo, hi, int(m.sum()), 100 * m.mean()))
    return out


def split(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    z = np.load(path, mmap_mode='r')
    keys = list(z.keys())
    n = len(z[keys[0]])
    o0 = np.asarray(z['occ0']).ravel()[:n]
    o1 = np.asarray(z['occ1']).ravel()[:n]
    e = empties(o0, o1)
    with open(os.path.join(outdir, 'census.txt'), 'w') as cf:
        cf.write('source=%s n=%d\n' % (path, n))
        for bi, (lo, hi, nm) in enumerate(BUCKETS):
            m = np.where((e >= lo) & (e <= hi))[0]
            kw = {k: np.asarray(z[k]).ravel()[:n][m] for k in keys}
            fp = os.path.join(outdir, 'p%d.npz' % bi)
            np.savez_compressed(fp, **kw)
            cf.write('%s empties %d-%d: %d (%.1f%%)\n' %
                     (nm, lo, hi, len(m), 100 * len(m) / max(n, 1)))
            print('%s -> %d rows' % (fp, len(m)), flush=True)
    print('census written', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', default='')
    ap.add_argument('--split', default='')
    ap.add_argument('--outdir', default='')
    a = ap.parse_args()
    if a.census:
        census(a.census)
    if a.split:
        assert a.outdir
        split(a.split, a.outdir)


if __name__ == '__main__':
    main()
