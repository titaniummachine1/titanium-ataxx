#!/usr/bin/env python3
"""Priority relabel queue: regret-blame x commonness.
Tier 1 = regret rows (have drops): score = blame * (1 + relfreq).
Tier 2 = base rows: score = relfreq (common positions need precision).
relfreq = mean index-count over the row's sparse patterns, counted over the
whole pool with the trainer's exact index layout. Highest first.
Usage: python training/prio_queue.py --top 2000 --out data/nnue/queue.txt
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, 'training')
import train_tuple as T

SP_OFF = {'s': 0, 'H': 2916, 'V': 9477, 'D1': 16038, 'D2': 22599,
          'q2': 29160, 'envus': 37921, 'envthem': 73642,
          'mobus': 109363, 'mobthem': 111863, 'pa': 114818}
SP_DIR = {0: 'H', 1: 'V', 2: 'D1', 3: 'D2'}
SP_N = 114818 + 80 * 6561
DIRS = [0, 1, 2, 3]


def build_feats(occ0, occ1, blk, stm):
    is_b = (stm == 0)
    own = np.where(is_b, occ0, occ1)
    enm = np.where(is_b, occ1, occ0)
    F = T.block_indices(own, enm)
    Q = T.q2_indices(own, enm, blk)
    EUS, ETH = T.env_tables(own, enm, blk)
    MUS, MTH = T.mob_tables(own, enm, blk)
    CUS, CTH = T.capn_tables(own, enm, blk)
    VUS, VTH = T.conv_tables(own, enm, blk)
    return F, Q, EUS, ETH, MUS, MTH, CUS, CTH, VUS, VTH


def materialize_sp(Fb, Qb, EUb, ETb, MUb, MTb):
    F = Fb.astype(np.int64)
    AR = np.arange(36, dtype=np.int64)[None, :]
    AR2 = np.arange(36, dtype=np.int64)[None, :] * 256
    XA = [np.array([x for x, y in pp], dtype=np.int64)
          for pp in [[(x, y) for (dd, x, y) in T.PAIRS if dd == d] for d in range(4)]]
    YA = [np.array([y for x, y in pp], dtype=np.int64)
          for pp in [[(x, y) for (dd, x, y) in T.PAIRS if dd == d] for d in range(4)]]
    PAX = np.array([x for (dd, x, y) in T.PAIRS], dtype=np.int64)
    PAY = np.array([y for (dd, x, y) in T.PAIRS], dtype=np.int64)
    PAOFF = np.arange(80, dtype=np.int64)[None, :] * 6561
    sp = [(AR * 81 + F)]
    for d in DIRS:
        sp.append((SP_OFF[SP_DIR[d]] + F[:, XA[d]] + 81 * F[:, YA[d]]).reshape(F.shape[0], -1))
    sp.append((AR2 + Qb.astype(np.int64)))
    sp.append((SP_OFF['envus'] + EUb.astype(np.int64)))
    sp.append((SP_OFF['envthem'] + ETb.astype(np.int64)))
    sp.append((SP_OFF['mobus'] + MUb.astype(np.int64)).reshape(F.shape[0], -1))
    sp.append((SP_OFF['mobthem'] + MTb.astype(np.int64)).reshape(F.shape[0], -1))
    sp.append((SP_OFF['pa'] + PAOFF + F[:, PAX] + 81 * F[:, PAY]))
    return np.concatenate(sp, axis=1).astype(np.int32)


def board_line(o0, o1, blk, stm):
    s = ''.join('#' if (blk >> sq) & 1 else 'x' if (o0 >> sq) & 1
                else 'o' if (o1 >> sq) & 1 else '.' for sq in range(49))
    return s + (' b' if int(stm) == 0 else ' w')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='data/nnue/cache400kR_graft.npz')
    ap.add_argument('--regret', default='data/nnue/regretloop_r*.npz,data/nnue/regret1m.npz')
    ap.add_argument('--top', type=int, default=2000)
    ap.add_argument('--out', default='data/nnue/queue.txt')
    a = ap.parse_args()
    reg_files = []
    for pat in a.regret.split(','):
        reg_files.extend(glob.glob(pat))
    print('regret files: %d' % len(reg_files), flush=True)

    counts = np.zeros(SP_N, dtype=np.float64)
    reg_rows = []  # (occ0,occ1,blk,stm,blame)
    print('pass 1: counting (base streamed)', flush=True)
    d = np.load(a.base)
    n = int(d['occ0'].shape[0])
    CH = 50000
    for s in range(0, n, CH):
        e = min(n, s + CH)
        F, Q, EUS, ETH, MUS, MTH, _, _, _, _ = build_feats(
            d['occ0'][s:e], d['occ1'][s:e], d['blk'][s:e], d['stm'][s:e])
        SP = materialize_sp(F.astype(np.int64) if F.dtype != np.int64 else F,
                            Q, EUS, ETH, MUS, MTH)
        counts += np.bincount(SP.reshape(-1).astype(np.int64), minlength=SP_N)
        del F, Q, EUS, ETH, MUS, MTH, SP
    for f in reg_files:
        r = np.load(f)
        m = int(r['occ0'].shape[0])
        w = r['wsum'].astype(np.float64) if 'wsum' in r else np.ones(m)
        for i in range(m):
            reg_rows.append((int(r['occ0'][i]), int(r['occ1'][i]), int(r['blk'][i]),
                             int(r['stm'][i]), float(w[i])))
        F, Q, EUS, ETH, MUS, MTH, _, _, _, _ = build_feats(
            r['occ0'], r['occ1'], r['blk'], r['stm'])
        SP = materialize_sp(F.astype(np.int64) if F.dtype != np.int64 else F,
                            Q, EUS, ETH, MUS, MTH)
        counts += np.bincount(SP.reshape(-1).astype(np.int64), minlength=SP_N)
        del F, Q, EUS, ETH, MUS, MTH, SP
    print('pass 1 done: %d regret rows' % len(reg_rows), flush=True)

    print('pass 2: scoring', flush=True)
    scored = []
    for (o0, o1, blk, stm, blame) in reg_rows:
        F, Q, EUS, ETH, MUS, MTH, _, _, _, _ = build_feats(
            np.array([o0], dtype=np.uint64), np.array([o1], dtype=np.uint64),
            np.array([blk], dtype=np.uint64), np.array([stm], dtype=np.uint8))
        SP = materialize_sp(F.astype(np.int64), Q, EUS, ETH, MUS, MTH)
        rel = float(counts[SP[0]].mean()) if SP.shape[1] else 0.0
        scored.append((blame * (1.0 + rel), blame, rel, o0, o1, blk, stm, 1))
    d = np.load(a.base)
    mx = 0.0
    rels = []
    for s in range(0, n, CH):
        e = min(n, s + CH)
        F, Q, EUS, ETH, MUS, MTH, _, _, _, _ = build_feats(
            d['occ0'][s:e], d['occ1'][s:e], d['blk'][s:e], d['stm'][s:e])
        SP = materialize_sp(F.astype(np.int64) if F.dtype != np.int64 else F,
                            Q, EUS, ETH, MUS, MTH)
        r = counts[SP].mean(axis=1)
        mx = max(mx, float(r.max()))
        rels.append(r)
        del F, Q, EUS, ETH, MUS, MTH, SP
    rels = np.concatenate(rels)
    rels = rels / mx if mx > 0 else rels
    for s in range(n):
        scored.append((float(rels[s]), 0.0, float(rels[s]),
                       int(d['occ0'][s]), int(d['occ1'][s]), int(d['blk'][s]),
                       int(d['stm'][s]), 0))
    scored.sort(key=lambda t: (t[7], t[0]), reverse=True)
    with open(a.out, 'w') as f:
        for t in scored[:a.top]:
            f.write('%s # score=%.3f blame=%.2f rel=%.3f tier=%d\n' % (
                board_line(t[3], t[4], t[5], t[6]), t[0], t[1], t[2], t[7]))
    print('wrote %s top=%d (tier1 regret rows: %d)' % (
        a.out, a.top, sum(1 for t in scored[:a.top] if t[7] == 1)), flush=True)


if __name__ == '__main__':
    main()
