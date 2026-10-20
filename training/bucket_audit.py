import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_full as T
import train_stream as S


def popcnt64(a):
    a = a.astype(np.uint64)
    out = np.zeros(len(a), dtype=np.int64)
    x = a.copy()
    while True:
        nz = x != 0
        if not nz.any():
            break
        out += (x & np.uint64(1)).astype(np.int64)
        x >>= np.uint64(1)
    return out


def seg_sse(counts, sums, sqs, lo, hi):
    n = counts[lo:hi].sum()
    if n == 0:
        return 0.0
    s = sums[lo:hi].sum()
    q = sqs[lo:hi].sum()
    return float(q - s * s / n)


def best_segmentation(cnt, sm, sq, max_k):
    vals = list(range(1, 50))
    m = len(vals)
    INF = 1e100
    dp = np.full((max_k + 1, m + 1), INF)
    back = np.zeros((max_k + 1, m + 1), dtype=np.int64)
    dp[0][0] = 0.0
    for k in range(1, max_k + 1):
        for j in range(1, m + 1):
            best = INF
            bi = 0
            for i in range(k - 1, j):
                if dp[k - 1][i] >= INF:
                    continue
                cost = dp[k - 1][i] + seg_sse(cnt, sm, sq, vals[i] - 1, vals[j - 1])
                if cost < best:
                    best = cost
                    bi = i
            dp[k][j] = best
            back[k][j] = bi
    bounds = []
    j = m
    for k in range(max_k, 0, -1):
        i = back[k][j]
        bounds.append((vals[i], vals[j - 1]))
        j = i
    bounds.reverse()
    return bounds, dp[max_k][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--net', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--rows', type=int, default=2000000)
    ap.add_argument('--block', type=int, default=250000)
    ap.add_argument('--max-k', type=int, default=12)
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()

    b0, ws, wp, rest, cc, pa = T.load_tup(a.net)
    W = np.zeros(S.SP_N, np.float64)
    D = np.zeros(S.DE_N, np.float64)
    W[S.SP_OFF['s']:S.SP_OFF['s'] + 2916] = ws.reshape(-1)
    for d, nm in enumerate(['H', 'V', 'D1', 'D2']):
        W[S.SP_OFF[nm]:S.SP_OFF[nm] + 6561] = wp[d].reshape(-1)
    if rest is not None:
        wq, weu, wet, wmu, wmt = rest
        W[S.SP_OFF['q2']:S.SP_OFF['q2'] + 9216] = wq.reshape(-1)
        W[S.SP_OFF['envus']:S.SP_OFF['envus'] + 35721] = weu.reshape(-1)
        W[S.SP_OFF['envthem']:S.SP_OFF['envthem'] + 35721] = wet.reshape(-1)
        W[S.SP_OFF['mobus']:S.SP_OFF['mobus'] + 2500] = wmu.reshape(-1)
        W[S.SP_OFF['mobthem']:S.SP_OFF['mobthem'] + 2500] = wmt.reshape(-1)
    if cc is not None:
        wcu, wct, wvu, wvt = cc
        D[S.DE_OFF['capnus']:S.DE_OFF['capnus'] + 9] = wcu.reshape(-1)
        D[S.DE_OFF['capnthem']:S.DE_OFF['capnthem'] + 9] = wct.reshape(-1)
        D[S.DE_OFF['convus']:S.DE_OFF['convus'] + 256] = wvu.reshape(-1)
        D[S.DE_OFF['convthem']:S.DE_OFF['convthem'] + 256] = wvt.reshape(-1)
    if pa is not None:
        W[S.SP_OFF['pa']:S.SP_OFF['pa'] + 80 * 6561] = pa.reshape(-1)

    z = np.load(a.cache)
    n_all = len(z['occ0'])
    rng = np.random.RandomState(a.seed)
    sel = np.sort(rng.choice(n_all, min(a.rows, n_all), replace=False))
    print('sample %d rows' % len(sel), flush=True)

    res = np.zeros(len(sel), np.float32)
    ocp = np.zeros(len(sel), np.float32)
    pc = np.zeros(len(sel), np.int64)
    wpos = 0
    import time as _time
    t_start = _time.time()
    for lo in range(0, len(sel), a.block):
        idx = sel[lo:lo + a.block]
        o0 = z['occ0'][idx].astype(np.uint64)
        o1 = z['occ1'][idx].astype(np.uint64)
        bl = z['blk'][idx].astype(np.uint64)
        st = z['stm'][idx].astype(np.uint8)
        t = z['teacher'][idx].astype(np.float64)
        out = z['out'][idx].astype(np.float64) if 'out' in z else np.full(len(idx), 0.5)
        keep = np.isfinite(t) & (np.abs(t) <= 5000.0)
        o0, o1, bl, st, t, out = o0[keep], o1[keep], bl[keep], st[keep], t[keep], out[keep]
        feats = S.features_for(o0, o1, bl, st)
        SP, DE = S.materialize32(*feats, [0, 1, 2, 3])
        del feats
        pred = W[SP].sum(axis=1) + D[DE].sum(axis=1) + b0
        del SP, DE
        base = T.base_eval_stm(o0, o1, bl, st)
        # write back: this block may be smaller than a.block due to mate filter
        res[wpos:wpos + len(t)] = (np.clip(t, -1500, 1500) - (base + pred)).astype(np.float32)
        ocp[wpos:wpos + len(t)] = (np.clip((out * 2 - 1) * 1000, -1500, 1500) - (base + pred)).astype(np.float32)
        pc[wpos:wpos + len(t)] = popcnt64(o0 | o1)
        wpos += len(t)
        print('  block %d..%d kept %d (%.0fs)' % (lo, lo + len(idx), len(t),
              _time.time() - t_start), flush=True)

    res = res[:wpos]; ocp = ocp[:wpos]; pc = pc[:wpos]
    vals = np.arange(1, 50)
    for name, r in (('teacher', res), ('outcome', ocp)):
        cnt = np.array([np.sum(pc == v) for v in vals], dtype=np.float64)
        sm = np.array([r[pc == v].sum() for v in vals], dtype=np.float64)
        sq = np.array([(r[pc == v].astype(np.float64) ** 2).sum() for v in vals], dtype=np.float64)
        sst = float((r.astype(np.float64) ** 2).sum())
        print('\n== %s residual: var %.1f total rows %d' % (name, sst / max(len(r), 1), len(r)), flush=True)
        print('piececount  n     mean_resid', flush=True)
        for v, c, s in zip(vals, cnt, sm):
            if c > 0:
                print('%3d %9d %+9.1f' % (v, c, s / c), flush=True)
        print('\nbuckets  R2(within-remove)  boundaries (dp optimal)', flush=True)
        for k in range(2, a.max_k + 1):
            bounds, sse = best_segmentation(cnt, sm, sq, k)
            r2 = 1.0 - sse / max(sst, 1e-9)
            print('  k=%2d  R2 %.5f  %s' % (k, r2, ' '.join('%d-%d' % b for b in bounds)), flush=True)
        # stockfish style width 4
        bw = 4
        edges = list(range(1, 50, bw))
        sse = 0.0
        for e in edges:
            lo, hi = e, min(49, e + bw - 1)
            sse += seg_sse(cnt, sm, sq, lo - 1, hi)
        print('  width4 (SF-style) R2 %.5f' % (1 - sse / max(sst, 1e-9)), flush=True)


if __name__ == '__main__':
    main()
