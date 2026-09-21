"""Dump cache bitboards -> Board::from_str text lines for the `label` command.

Line format: 49 chars top->bottom (.xo#) + ' ' + b/w. stm 0 = black = 'b'.
Chunks the output so long label runs checkpoint safely.

Usage: python3 training/dump_fens.py --cache data/nnue/cache400k.npz --out data/nnue/lab400k --chunks 4
  -> data/nnue/lab400k_0.txt .. lab400k_3.txt
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache400k.npz')
    ap.add_argument('--out', default='data/nnue/lab400k')
    ap.add_argument('--chunks', type=int, default=4)
    a = ap.parse_args()

    z = np.load(a.cache)
    o0 = z['occ0'].astype(object)
    o1 = z['occ1'].astype(object)
    blk = z['blk'].astype(object)
    stm = z['stm']
    n = len(o0)
    print('rows %d' % n, flush=True)

    per = (n + a.chunks - 1) // a.chunks
    for c in range(a.chunks):
        lo, hi = c * per, min(n, (c + 1) * per)
        if lo >= hi:
            break
        path = '%s_%d.txt' % (a.out, c)
        with open(path, 'w') as f:
            for i in range(lo, hi):
                a0, a1, ab = int(o0[i]), int(o1[i]), int(blk[i])
                f.write(''.join(
                    '#' if (ab >> sq) & 1 else 'x' if (a0 >> sq) & 1
                    else 'o' if (a1 >> sq) & 1 else '.'
                    for sq in range(49)))
                f.write(' b\n' if stm[i] == 0 else ' w\n')
        print('wrote %s rows %d..%d' % (path, lo, hi), flush=True)


if __name__ == '__main__':
    main()
