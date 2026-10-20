import argparse
import os
import sys

import numpy as np


def col(z, name, n, dtype):
    return z[name].astype(dtype) if name in z else np.zeros(n, dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bulk', default='data/nnue/cache12M.npz')
    ap.add_argument('--golden', required=True)
    ap.add_argument('--out', default='data/nnue/cache12M_golden.npz')
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--dose', type=float, default=0.03)
    a = ap.parse_args()

    b = np.load(a.bulk, mmap_mode='r')
    nb = len(b['occ0'])
    g = np.load(a.golden)
    ng = len(g['occ0'])
    print('bulk %d rows, golden %d rows, repeat %d, target dose %.3f' % (nb, ng, a.repeat, a.dose), flush=True)

    reps = a.repeat
    nrep = ng * reps
    gw = a.dose * nb / max(nrep * (1.0 - a.dose), 1.0)
    print('golden per-row wsum weight %.2f (dose share ~%.3f)' % (gw, a.dose), flush=True)

    def cat(name, dtype):
        out = np.empty(nb + nrep, dtype=dtype)
        out[:nb] = b[name]
        src = g[name].astype(dtype)
        out[nb:] = np.tile(src, reps)
        return out

    occ0 = cat('occ0', np.uint64)
    occ1 = cat('occ1', np.uint64)
    blk = cat('blk', np.uint64)
    stm = cat('stm', np.uint8)
    out_col = cat('out', np.float32)
    sc = cat('sc', np.float32)
    teacher = cat('teacher', np.float32)

    wsum = np.empty(nb + nrep, dtype=np.float32)
    wsum[:nb] = b['wsum'].astype(np.float32) if 'wsum' in b else 1.0
    wsum[nb:] = np.tile(g['wsum'].astype(np.float32), reps) * gw

    np.savez_compressed(a.out, occ0=occ0, occ1=occ1, blk=blk, stm=stm,
                        out=out_col, sc=sc, teacher=teacher, wsum=wsum)
    print('wrote %s %.1fMB rows %d' % (a.out, os.path.getsize(a.out) / 1e6, len(occ0)), flush=True)


if __name__ == '__main__':
    main()
