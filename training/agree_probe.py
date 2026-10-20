import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_full as T
import train_stream as S
from make_hard_chunks import flat_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', default='data/nnue/align20k.npz')
    ap.add_argument('--nets', nargs='+', required=True)
    a = ap.parse_args()
    z = np.load(a.probe)
    o0 = z['occ0'].astype(np.uint64)
    o1 = z['occ1'].astype(np.uint64)
    bl = z['blk'].astype(np.uint64)
    st = z['stm'].astype(np.uint8)
    teach = z['teacher'].astype(np.float64)
    base = T.base_eval_stm(o0, o1, bl, st)
    feats = S.features_for(o0, o1, bl, st)
    SP, DE = S.materialize32(*feats, [0, 1, 2, 3])
    del feats
    for p in a.nets:
        bias, W, D = flat_tables(p)
        pred = W[SP].sum(axis=1) + D[DE].sum(axis=1) + bias
        diff = np.abs(pred + base - teach)
        mse = float((((pred + base) - teach) ** 2).mean())
        print('%-28s bias %+.1f agree mean|d| %.1f  mse %.1f  corr %+.4f' % (
            os.path.basename(p), bias, float(diff.mean()), mse,
            float(np.corrcoef(pred + base, teach)[0, 1])), flush=True)


if __name__ == '__main__':
    main()
