"""Feature audit: marginal signal of each eval family on cache data.

For every family (singles, pairs, q2, env, mob, capn, conv, pa) computes a
one-shot mean-probe: per-feature mean residual on train rows, aggregated per
val row, scale-fitted on train, R2 on val vs residual variance.

Answers: does this family carry signal the base eval misses? Cheap (one pass,
no SGD). Usage:
  python training/feature_audit.py --cache data/nnue/cache12M.npz --rows 400000
"""
import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import train_full as T


def probe(name, idx_tr, r_tr, idx_va, r_va, vocab, out):
    """One-shot mean probe. idx arrays (n,k) int64; r residual (n,)."""
    t0 = time.time()
    flat = idx_tr.reshape(-1)
    sums = np.bincount(flat, weights=np.repeat(r_tr, idx_tr.shape[1]), minlength=vocab)
    cnts = np.bincount(flat, minlength=vocab).astype(np.float64)
    mean = np.zeros(vocab)
    nz = cnts > 0
    mean[nz] = sums[nz] / cnts[nz]
    # mask feature 0 of constrained families is allowed to be nonzero here (probe)
    agg_tr = mean[idx_tr].sum(axis=1)
    agg_va = mean[idx_va].sum(axis=1)
    alpha = float(np.dot(agg_tr, r_tr) / max(np.dot(agg_tr, agg_tr), 1e-9))
    pred = alpha * agg_va
    sse = float(((pred - r_va) ** 2).sum())
    sst = float((r_va ** 2).sum())
    r2 = 1.0 - sse / max(sst, 1e-9)
    cov = float((cnts[idx_va] > 0).astype(np.float64).mean())
    act = float((cnts > 0).sum()) / vocab
    out.append((name, vocab, int(nz.sum()), act, cov, r2, alpha, time.time() - t0))
    print('%-10s vocab %8d  feat-alive %6.1f%%  val-cov %.3f  R2 %+.4f  alpha %.3f (%.1fs)'
          % (name, vocab, 100 * act, cov, r2, alpha, time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--rows', type=int, default=400000)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--split', type=float, default=0.875)
    ap.add_argument('--target', default='resid', choices=['resid', 'out'],
                    help='resid = teacher-base (what the net trains on); '
                         'out = real game result - base (outcome grounding signal).')
    a = ap.parse_args()

    z = np.load(a.cache)
    n_all = len(z['occ0'])
    rng = np.random.RandomState(a.seed)
    sel = np.sort(rng.choice(n_all, min(a.rows, n_all), replace=False))
    occ0 = z['occ0'][sel].astype(np.uint64)
    occ1 = z['occ1'][sel].astype(np.uint64)
    blk = z['blk'][sel].astype(np.uint64)
    stm = z['stm'][sel].astype(np.uint8)
    teacher = z['teacher'][sel].astype(np.float64)
    out_col = z['out'][sel].astype(np.float64) if 'out' in z else np.full(len(sel), 0.5)
    keep = np.isfinite(teacher) & (np.abs(teacher) <= 5000.0)
    occ0, occ1, blk, stm, teacher = occ0[keep], occ1[keep], blk[keep], stm[keep], teacher[keep]
    out_col = out_col[keep]
    n = len(occ0)
    print('sample %d rows (kept %d after mate filter) of %d' % (a.rows, n, n_all), flush=True)

    is_b = (stm == 0)
    own = np.where(is_b, occ0, occ1)
    enm = np.where(is_b, occ1, occ0)

    base = T.base_eval_stm(occ0, occ1, blk, stm)
    if a.target == 'out':
        outcome_cp = np.clip((out_col * 2.0 - 1.0) * 1000.0, -1500.0, 1500.0)
        resid = (outcome_cp - base).astype(np.float64)
        print('target OUT-base (real result): mean %+.1f std %.1f (out means %.3f, base mean %+.1f)'
              % (resid.mean(), resid.std(), out_col.mean(), base.mean()), flush=True)
    else:
        resid = (teacher - base).astype(np.float64)
        print('resid mean %+.1f std %.1f' % (resid.mean(), resid.std()), flush=True)

    cut = int(n * a.split)
    r_tr, r_va = resid[:cut], resid[cut:]

    t0 = time.time()
    F = T.block_indices(own, enm)
    Q = T.q2_indices(own, enm, blk)
    EU, ET = T.env_tables(own, enm, blk)
    MU, MT = T.mob_tables(own, enm, blk)
    CU, CT = T.capn_tables(own, enm, blk)
    VU, VT = T.conv_tables(own, enm, blk)
    print('feature compute %.0fs' % (time.time() - t0), flush=True)

    AR = np.arange(36, dtype=np.int64)[None, :]
    out = []
    probe('singles', AR * 81 + F[:cut], r_tr, AR * 81 + F[cut:], r_va, 36 * 81, out)
    XA = [np.array([x for x, y in pp], dtype=np.int64)
          for pp in [[(x, y) for (dd, x, y) in T.PAIRS if dd == d] for d in range(4)]]
    YA = [np.array([y for x, y in pp], dtype=np.int64)
          for pp in [[(x, y) for (dd, x, y) in T.PAIRS if dd == d] for d in range(4)]]
    pr_tr = np.concatenate([d * 6561 + F[:cut][:, XA[d]] + 81 * F[:cut][:, YA[d]] for d in range(4)], axis=1)
    pr_va = np.concatenate([d * 6561 + F[cut:][:, XA[d]] + 81 * F[cut:][:, YA[d]] for d in range(4)], axis=1)
    probe('pairs', pr_tr, r_tr, pr_va, r_va, 4 * 6561, out)
    probe('q2', AR * 256 + Q[:cut].astype(np.int64), r_tr, AR * 256 + Q[cut:].astype(np.int64), r_va, 36 * 256, out)
    probe('envus', EU[:cut].astype(np.int64), r_tr, EU[cut:].astype(np.int64), r_va, 49 * 729, out)
    probe('envthem', ET[:cut].astype(np.int64), r_tr, ET[cut:].astype(np.int64), r_va, 49 * 729, out)
    probe('mobus', MU[:cut].astype(np.int64).reshape(-1, 1), r_tr, MU[cut:].astype(np.int64).reshape(-1, 1), r_va, 2500, out)
    probe('mobthem', MT[:cut].astype(np.int64).reshape(-1, 1), r_tr, MT[cut:].astype(np.int64).reshape(-1, 1), r_va, 2500, out)
    probe('capnus', CU[:cut].astype(np.int64).reshape(-1, 1), r_tr, CU[cut:].astype(np.int64).reshape(-1, 1), r_va, 9, out)
    probe('capnthem', CT[:cut].astype(np.int64).reshape(-1, 1), r_tr, CT[cut:].astype(np.int64).reshape(-1, 1), r_va, 9, out)
    probe('convus', VU[:cut].astype(np.int64), r_tr, VU[cut:].astype(np.int64), r_va, 256, out)
    probe('convthem', VT[:cut].astype(np.int64), r_tr, VT[cut:].astype(np.int64), r_va, 256, out)
    PAX = np.array([x for (dd, x, y) in T.PAIRS], dtype=np.int64)
    PAY = np.array([y for (dd, x, y) in T.PAIRS], dtype=np.int64)
    PAOFF = np.arange(80, dtype=np.int64)[None, :] * 6561
    pa_tr = PAOFF + F[:cut][:, PAX] + 81 * F[:cut][:, PAY]
    pa_va = PAOFF + F[cut:][:, PAX] + 81 * F[cut:][:, PAY]
    probe('pa', pa_tr, r_tr, pa_va, r_va, 80 * 6561, out)

    print()
    print('%-10s %8s %10s %8s %+8s' % ('family', 'vocab', 'alive', 'valcov', 'R2'))
    for (nm, vocab, nz, act, cov, r2, alpha, dt) in out:
        print('%-10s %8d %9.1f%% %7.3f %+8.4f' % (nm, vocab, 100 * act, cov, r2))


if __name__ == '__main__':
    main()
