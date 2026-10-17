#!/usr/bin/env python3
"""Arena opening book from cache rows (Moonbird-method analogue for our
no-blocker variant): balanced early positions, both colors alive.
Their method: symmetric blocker configs + balance(depth2, imbalance 0)
expansion. Ours: real cache rows, material |diff|<=2, 4-16 stones.

Usage: python3 training/make_book.py --cache data/nnue/cache400kR.npz \
         --out data/nnue/book100.txt --n 100 --seed 7
Line format: 49 chars top->bottom (.xo#) + ' ' + b/w (Board::from_str).
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache400kR.npz')
    ap.add_argument('--out', default='data/nnue/book100.txt')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--min-stones', type=int, default=4)
    ap.add_argument('--max-stones', type=int, default=16)
    ap.add_argument('--max-diff', type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    z = np.load(a.cache)
    o0 = z['occ0'].astype(object)
    o1 = z['occ1'].astype(object)
    blk = z['blk'].astype(object)
    stm = z['stm']
    n = len(o0)
    nb = np.array([bin(int(x)).count('1') for x in o0])
    nw = np.array([bin(int(x)).count('1') for x in o1])
    tot = nb + nw
    ok = ((tot >= a.min_stones) & (tot <= a.max_stones)
          & (np.abs(nb - nw) <= a.max_diff) & (nb > 0) & (nw > 0))
    idx = np.flatnonzero(ok)
    print('candidates %d/%d' % (len(idx), n), flush=True)
    sel = rng.choice(idx, size=min(a.n, len(idx)), replace=False)
    with open(a.out, 'w') as f:
        for i in sel:
            a0, a1, ab = int(o0[i]), int(o1[i]), int(blk[i])
            s = ''.join('#' if (ab >> sq) & 1 else 'x' if (a0 >> sq) & 1
                        else 'o' if (a1 >> sq) & 1 else '.' for sq in range(49))
            f.write(s + (' b\n' if int(stm[i]) == 0 else ' w\n'))
    print('wrote %s (%d positions)' % (a.out, len(sel)), flush=True)


if __name__ == '__main__':
    main()
