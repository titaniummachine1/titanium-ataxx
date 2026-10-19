#!/usr/bin/env python3
"""Join per-worker selfplay chunks into one npz + phase report."""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import re
    # chunks are CUMULATIVE snapshots: keep only the latest per worker
    latest = {}
    for fp in glob.glob(os.path.join(a.indir, 'chunk_*.npz')):
        m = re.search(r'chunk_w(\d+)_(?:n(\d+)|final)', os.path.basename(fp))
        if not m:
            continue
        w, n = int(m.group(1)), m.group(2)
        key = (w, int(n) if n else 10 ** 9)
        if key[0] not in latest or key > latest[key[0]][0]:
            latest[key[0]] = (key, fp)
    files = [fp for (_, fp) in sorted(latest.values())]
    print('using latest per worker:', [os.path.basename(f) for f in files])
    o0, o1, bl, st, ou = [], [], [], [], []
    for fp in files:
        try:
            z = np.load(fp)
        except Exception as ex:
            print('skip %s: %s' % (fp, ex))
            continue
        if len(z['occ0']) == 0:
            continue
        o0.append(z['occ0'])
        o1.append(z['occ1'])
        bl.append(z['blk'])
        st.append(z['stm'])
        ou.append(z['outcome'])
    o0 = np.concatenate(o0).astype(np.uint64)
    o1 = np.concatenate(o1).astype(np.uint64)
    bl = np.concatenate(bl).astype(np.uint64)
    st = np.concatenate(st).astype(np.int64)
    ou = np.concatenate(ou).astype(np.float64)
    np.savez_compressed(a.out, occ0=o0, occ1=o1, blk=bl, stm=st, outcome=ou)
    print('joined %d positions -> %s' % (len(o0), a.out))
    # phase census (empties buckets)
    emp = 49 - (np.unpackbits(o0.view(np.uint8)).reshape(len(o0), 64).sum(1) +
                np.unpackbits(o1.view(np.uint8)).reshape(len(o0), 64).sum(1))
    bins = [(40, 49, 'P0-open'), (28, 39, 'P1-emid'), (14, 27, 'P2-lmid'), (0, 13, 'P3-end')]
    for lo, hi, nm in bins:
        m = (emp >= lo) & (emp <= hi)
        print('%s empties %2d-%2d: %6d (%.1f%%)' % (nm, lo, hi, int(m.sum()), 100 * m.mean()))


if __name__ == '__main__':
    main()
