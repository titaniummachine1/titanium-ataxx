#!/usr/bin/env python3
"""Secondary alignment dataset: mostly random, bell-weighted by commonness,
regret mixed in. Purpose: align eval with own deep search (agreement), not Elo.
- 14k uniform random from base cache
- 4k sampled proportional to sqrt(pattern-frequency) (soft bell)
- 2k top-regret rows by wsum from banked regret npzs
"""
import glob
import sys

import numpy as np

sys.path.insert(0, 'training')
import prio_queue as Q

N_U, N_F, N_R = 14000, 4000, 2000
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 12
rng = np.random.default_rng(seed)

d = np.load('data/nnue/cache400kR_graft.npz')
N = int(d['occ0'].shape[0])
print('base rows: %d' % N, flush=True)


def board_line(o0, o1, blk, stm):
    s = ''.join('#' if (blk >> sq) & 1 else 'x' if (o0 >> sq) & 1
                else 'o' if (o1 >> sq) & 1 else '.' for sq in range(49))
    return s + (' b' if int(stm) == 0 else ' w')


print('materializing for frequency...', flush=True)
F, Qb, EUS, ETH, MUS, MTH, _, _, _, _ = Q.build_feats(
    d['occ0'], d['occ1'], d['blk'], d['stm'])
SP = Q.materialize_sp(F.astype(np.int64), Qb, EUS, ETH, MUS, MTH)
del F, Qb, EUS, ETH, MUS, MTH
counts = np.bincount(SP.reshape(-1).astype(np.int64), minlength=Q.SP_N).astype(np.float64)
rel = counts[SP].mean(axis=1)
del SP, counts
p = np.sqrt(rel / rel.max())
p = p / p.sum()

uni = rng.choice(N, size=N_U, replace=False)
frq = rng.choice(N, size=N_F, replace=False, p=p)

reg = []
for f in glob.glob('data/nnue/regretloop_r*.npz') + ['data/nnue/regret1m.npz']:
    try:
        r = np.load(f)
    except OSError:
        continue
    w = r['wsum'].astype(np.float64)
    m = len(w)
    take = min(400, m)
    top = np.argsort(-w)[:take]
    for i in top:
        reg.append((float(w[i]),
                    (int(r['occ0'][i]), int(r['occ1'][i]), int(r['blk'][i]), int(r['stm'][i]))))
reg.sort(key=lambda t: -t[0])
reg = reg[:N_R]
print('regret candidates banked: picked top %d' % len(reg), flush=True)

rows = []
for i in uni:
    rows.append((int(d['occ0'][i]), int(d['occ1'][i]), int(d['blk'][i]), int(d['stm'][i]), 'u'))
for i in frq:
    rows.append((int(d['occ0'][i]), int(d['occ1'][i]), int(d['blk'][i]), int(d['stm'][i]), 'f'))
for _, tup in reg:
    rows.append((tup[0], tup[1], tup[2], tup[3], 'r'))
rng.shuffle(rows)
with open('data/nnue/align20k_boards.txt', 'w') as f:
    for (o0, o1, blk, stm, k) in rows:
        f.write('%s #%s\n' % (board_line(o0, o1, blk, stm), k))
np.savez_compressed('data/nnue/align20k_meta.npz',
                    occ0=np.array([r[0] for r in rows], np.uint64),
                    occ1=np.array([r[1] for r in rows], np.uint64),
                    blk=np.array([r[2] for r in rows], np.uint64),
                    stm=np.array([r[3] for r in rows], np.uint8),
                    kind=np.array([r[4] for r in rows]))
print('wrote align20k: %d rows (u=%d f=%d r=%d)' % (
    len(rows), sum(1 for r in rows if r[4] == 'u'),
    sum(1 for r in rows if r[4] == 'f'), sum(1 for r in rows if r[4] == 'r')), flush=True)
