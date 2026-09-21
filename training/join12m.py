"""Join base_*.npz + lab_*.out -> one cache with teacher column.

Overnight step 3 of the 12.4M relabel. Aborts (no output file) if any
chunk is missing or its .out line count != base rows (orphan-write rule).

Usage: python training/join12m.py --dir data/nnue/relab12m --out data/nnue/cache12M.npz
"""
import argparse
import glob
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/nnue/relab12m')
    ap.add_argument('--out', default='data/nnue/cache12M.npz')
    a = ap.parse_args()

    bases = sorted(glob.glob(os.path.join(a.dir, 'base_*.npz')))
    assert bases, 'no base_*.npz in %s' % a.dir
    parts = []
    total = 0
    for b in bases:
        tag = os.path.basename(b)[5:7]
        lab = os.path.join(a.dir, 'lab_%s.out' % tag)
        if not os.path.exists(lab):
            print('ABORT: missing %s' % lab, flush=True)
            sys.exit(1)
        z = np.load(b)
        n = len(z['occ0'])
        t = np.loadtxt(lab, usecols=0, dtype=np.float32, ndmin=1)
        if len(t) != n:
            print('ABORT: %s has %d scores, want %d' % (lab, len(t), n), flush=True)
            sys.exit(1)
        parts.append((z, t))
        total += n
        print('chunk %s rows %d teacher mean %+.1f' % (tag, n, float(t.mean())), flush=True)
    occ0 = np.concatenate([z['occ0'] for z, _ in parts])
    occ1 = np.concatenate([z['occ1'] for z, _ in parts])
    blk = np.concatenate([z['blk'] for z, _ in parts])
    stm = np.concatenate([z['stm'] for z, _ in parts])
    out = np.concatenate([z['out'] for z, _ in parts])
    sc = np.concatenate([z['sc'] for z, _ in parts])
    wsum = np.concatenate([z['wsum'] for z, _ in parts])
    teacher = np.concatenate([t for _, t in parts]).astype(np.float32)
    assert len(teacher) == total
    print('teacher20k: n=%d mean %+.1f std %.1f rails +%.3f -%.3f'
          % (total, float(teacher.mean()), float(teacher.std()),
             float((teacher >= 1500).mean()), float((teacher <= -1500).mean())), flush=True)
    np.savez_compressed(a.out, occ0=occ0, occ1=occ1, blk=blk,
                        stm=stm, out=out, sc=sc, teacher=teacher, wsum=wsum)
    print('wrote %s %.1fMB' % (a.out, os.path.getsize(a.out) / 1e6), flush=True)


if __name__ == '__main__':
    main()
