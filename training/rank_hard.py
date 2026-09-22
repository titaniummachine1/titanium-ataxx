"""Rank corpus positions by optimizer weight = |residual| * wsum_norm.

Round-2 input for selective deepening: top-K rows are where the trained net
is most wrong on the most-visited positions -> relabel exactly these at
10k/20k nodes, merge back, retrain. Chunked (500k rows/block) to stay in RAM.

Usage: python training/rank_hard.py --net data/nnue/tupS26fly.tup --top 200000 --out data/nnue/hard200k.npy
"""
import argparse
import struct
import sys
import time

import numpy as np

sys.path.insert(0, 'training')
import train_tuple as T


def load_tup(path):
    d = open(path, 'rb').read()
    assert d[:4] == b'TUP4', path
    bias = struct.unpack('<i', d[4:8])[0]
    singles = np.frombuffer(d[8:8 + 2916], dtype=np.int8).astype(np.float64).reshape(36, 81)
    off = 8 + 2916 + 16
    pairs = []
    for _ in range(4):
        pairs.append(np.frombuffer(d[off:off + 6561], dtype=np.int8).astype(np.float64))
        off += 6561
    return bias, singles, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--outcome-weight', type=float, default=0.20)
    ap.add_argument('--top', type=int, default=200000)
    ap.add_argument('--out', default='data/nnue/hard200k.npy')
    a = ap.parse_args()
    bias, singles, pairs = load_tup(a.net)
    flat_s = singles.reshape(-1)
    z = np.load(a.cache, mmap_mode='r')
    n = len(z['occ0'])
    print('net %s rows %d' % (a.net, n), flush=True)

    raw_t = z['teacher'].astype(np.float64)
    ok = np.isfinite(raw_t) & (np.abs(raw_t) <= 5000.0)
    wsum = np.clip(z['wsum'].astype(np.float64), 0, None)
    wmean = max(wsum[ok].mean(), 1e-9)
    print('kept %d/%d (mate-spike dropped)' % (int(ok.sum()), n), flush=True)

    sw = np.clip((z['out'].astype(np.float64) * 2.0 - 1.0) * 1000.0, -1500.0, 1500.0)
    BS = 500000
    scores = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for lo in range(0, n, BS):
        hi = min(n, lo + BS)
        m = ok[lo:hi]
        if not m.any():
            continue
        o0 = z['occ0'][lo:hi][m].astype(np.uint64)
        o1 = z['occ1'][lo:hi][m].astype(np.uint64)
        bl = z['blk'][lo:hi][m].astype(np.uint64)
        st = z['stm'][lo:hi][m].astype(np.uint8)
        is_b = (st == 0)
        feats = T.block_indices(np.where(is_b, o0, o1), np.where(is_b, o1, o0))
        base = T.base_eval_stm(o0, o1, bl, st, simple=True)
        off = (np.arange(36, dtype=np.int64)[None, :] * 81 + feats)
        pred = flat_s[off].sum(axis=1) + bias
        for d in range(4):
            pp = [(x, y) for (dd, x, y) in T.PAIRS if dd == d]
            xa = np.array([x for x, y in pp], dtype=np.int64)
            ya = np.array([y for x, y in pp], dtype=np.int64)
            pred = pred + pairs[d][feats[:, xa] + 81 * feats[:, ya]].sum(axis=1)
        tgt = ((1.0 - a.outcome_weight) * np.clip(raw_t[lo:hi][m], -1500.0, 1500.0)
               + a.outcome_weight * sw[lo:hi][m])
        resid = np.abs(tgt - (base + pred))
        blk_scores = (resid * (wsum[lo:hi][m] / wmean)).astype(np.float32)
        scores[lo:hi][m] = blk_scores
        print('rank block %d..%d (%.0fs)' % (lo, hi, time.time() - t0), flush=True)
    k = min(a.top, n)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    np.save(a.out, idx.astype(np.int64))
    print('wrote %s top-%d mean_score %.1f wsum_mass %.3f' %
          (a.out, k, float(scores[idx].mean()),
           float(wsum[idx].sum() / max(wsum.sum(), 1e-9))), flush=True)


if __name__ == '__main__':
    main()
