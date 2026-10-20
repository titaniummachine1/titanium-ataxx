import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_hard_chunks import emit_fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--top', type=int, default=1200)
    ap.add_argument('--min-wsum', type=float, default=1.0)
    ap.add_argument('--out-dir', default='data/nnue/golden_src')
    ap.add_argument('--chunks', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    z = np.load(a.cache)
    occ0 = z['occ0'].astype(np.uint64)
    occ1 = z['occ1'].astype(np.uint64)
    blk = z['blk'].astype(np.uint64)
    stm = z['stm'].astype(np.uint64)
    n = len(occ0)
    teacher = z['teacher'].astype(np.float64)
    wsum = z['wsum'].astype(np.float64) if 'wsum' in z else np.ones(n)
    ok = np.isfinite(teacher) & (np.abs(teacher) <= 5000.0) & (wsum >= a.min_wsum)
    print('rows %d ok %d' % (n, int(ok.sum())), flush=True)

    h = (occ0 * np.uint64(0x9E3779B97F4A7C15)) ^ (occ1 * np.uint64(0xC2B2AE3D27D4EB4F))
    h ^= (blk * np.uint64(0x165667B19E3779F9)) ^ (stm * np.uint64(0x27D4EB2F165667C5))
    order = np.flatnonzero(ok)
    order = order[np.argsort(-wsum[order], kind='stable')]
    hs = h[order]
    _, first = np.unique(hs, return_index=True)
    uniq = order[first]
    uniq = uniq[np.argsort(-wsum[uniq], kind='stable')]
    keep = uniq[:min(a.top, len(uniq))]
    keep.sort()
    print('unique %d selected %d (hash-dedup)' % (len(uniq), len(keep)), flush=True)

    per = (len(keep) + a.chunks - 1) // a.chunks
    for c in range(a.chunks):
        part = keep[c * per:(c + 1) * per]
        if not len(part):
            break
        o0 = occ0[part]
        o1 = occ1[part]
        bl = blk[part]
        st = stm[part].astype(np.uint8)
        np.savez_compressed(os.path.join(a.out_dir, 'base_%02d.npz' % c),
                            occ0=o0, occ1=o1, blk=bl, stm=st,
                            out=z['out'][part], sc=z['sc'][part].astype(np.float32),
                            wsum=z['wsum'][part].astype(np.float32))
        with open(os.path.join(a.out_dir, 'lab_%02d.txt' % c), 'w') as f:
            emit_fens(o0, o1, bl, st, f)
        print('chunk %02d rows %d' % (c, len(part)), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
