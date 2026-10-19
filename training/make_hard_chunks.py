import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_full as T
import train_stream as S


def flat_tables(path):
    b0, ws, wp, rest, cc, pa = T.load_tup(path)
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
    return float(b0), W, D


def emit_fens(occ0, occ1, blk, stm, f):
    for i in range(len(occ0)):
        x0, x1, b, st = int(occ0[i]), int(occ1[i]), int(blk[i]), int(stm[i])
        f.write(''.join('#' if (b >> sq) & 1 else 'x' if (x0 >> sq) & 1
                        else 'o' if (x1 >> sq) & 1 else '.'
                        for sq in range(49)))
        f.write(' b\n' if st == 0 else ' w\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--net', required=True)
    ap.add_argument('--top', type=int, default=2000000)
    ap.add_argument('--chunks', type=int, default=8)
    ap.add_argument('--block', type=int, default=500000)
    ap.add_argument('--blend', type=float, default=0.9)
    ap.add_argument('--out-dir', default='data/nnue/relab_hard')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    bias, W, D = flat_tables(a.net)
    print('net %s bias %.1f W_nz %.3f D_nz %.3f' % (
        a.net, bias, (W != 0).mean(), (D != 0).mean()), flush=True)

    z = np.load(a.cache)
    n = len(z['occ0'])
    raw_t = z['teacher'].astype(np.float64)
    ok = np.isfinite(raw_t) & (np.abs(raw_t) <= 5000.0)
    sc = z['sc'].astype(np.float64)
    wsum = np.clip(z['wsum'].astype(np.float64), 0, None)
    wmean = max(wsum[ok].mean(), 1e-9)
    prime = np.clip(raw_t, -1500.0, 1500.0)
    legacy = np.clip(sc * 2000.0, -1500.0, 1500.0)
    tgt = ((1.0 - a.blend) * legacy + a.blend * prime).astype(np.float32)

    scores = np.full(n, -1e9, dtype=np.float32)
    t0 = time.time()
    for lo in range(0, n, a.block):
        hi = min(n, lo + a.block)
        m = ok[lo:hi]
        if not m.any():
            continue
        o0 = z['occ0'][lo:hi][m].astype(np.uint64)
        o1 = z['occ1'][lo:hi][m].astype(np.uint64)
        bl = z['blk'][lo:hi][m].astype(np.uint64)
        st = z['stm'][lo:hi][m].astype(np.uint8)
        feats = S.features_for(o0, o1, bl, st)
        SP, DE = S.materialize32(*feats, [0, 1, 2, 3])
        del feats
        pred = W[SP].sum(axis=1) + D[DE].sum(axis=1) + bias
        del SP, DE
        base = T.base_eval_stm(o0, o1, bl, st)
        resid = np.abs(tgt[lo:hi][m] - (base + pred))
        scores[lo:hi][m] = (resid.astype(np.float32) * (wsum[lo:hi][m] / wmean)).astype(np.float32)
        print('rank %d..%d (%.0fs)' % (lo, hi, time.time() - t0), flush=True)
    del W, D

    k = min(a.top, n)
    idx = np.argpartition(scores, -k)[-k:]
    idx = np.sort(idx)
    print('selected %d mean_score %.1f wsum_mass %.3f (%.0fs)' % (
        k, float(scores[idx].mean()), float(wsum[idx].sum() / max(wsum.sum(), 1e-9)),
        time.time() - t0), flush=True)
    del scores

    per = (k + a.chunks - 1) // a.chunks
    for c in range(a.chunks):
        lo, hi = c * per, min(k, (c + 1) * per)
        if lo >= hi:
            break
        rows = idx[lo:hi]
        tag = '%02d' % c
        o0 = z['occ0'][rows].astype(np.uint64)
        o1 = z['occ1'][rows].astype(np.uint64)
        bl = z['blk'][rows].astype(np.uint64)
        st = z['stm'][rows].astype(np.uint8)
        np.savez_compressed(os.path.join(a.out_dir, 'base_%s.npz' % tag),
                            occ0=o0, occ1=o1, blk=bl, stm=st,
                            out=z['out'][rows], sc=z['sc'][rows].astype(np.float32),
                            wsum=z['wsum'][rows].astype(np.float32))
        with open(os.path.join(a.out_dir, 'lab_%s.txt' % tag), 'w') as f:
            emit_fens(o0, o1, bl, st, f)
        print('chunk %s rows %d -> base/lab written' % (tag, hi - lo), flush=True)
    with open(os.path.join(a.out_dir, 'manifest.txt'), 'a') as man:
        man.write('top %d net %s blend %.2f\n' % (k, a.net, a.blend))
    print('done', flush=True)


if __name__ == '__main__':
    main()
