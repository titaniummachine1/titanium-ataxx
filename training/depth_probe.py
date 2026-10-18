#!/usr/bin/env python3
"""Depth-reach survey: how deep do 500k/1M searches get per occupancy phase,
and how often is the result a solved (mate) score vs heuristic? Informs the
retrograde-from-terminal plan: which phases can supply GROUND TRUTH."""
import sys
import numpy as np

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
d = np.load('data/nnue/cache400kR_graft.npz')
occ = d['occ0'].astype(np.int64) | d['occ1'].astype(np.int64)


def pop(x):
    return bin(int(x)).count('1')


n = len(occ)
stones = np.array([pop(x) for x in occ])
bands = [(0, 15, 'early'), (15, 35, 'mid'), (35, 50, 'late')]
out = []
for lo, hi, name in bands:
    idx = np.where((stones >= lo) & (stones < hi))[0]
    sel = rng.choice(idx, size=min(30, len(idx)), replace=False)
    for i in sel:
        o0, o1, blk, stm = int(d['occ0'][i]), int(d['occ1'][i]), int(d['blk'][i]), int(d['stm'][i])
        s = ''.join('#' if (blk >> sq) & 1 else 'x' if (o0 >> sq) & 1 else 'o' if (o1 >> sq) & 1 else '.' for sq in range(49))
        out.append('%s %s %s' % (s, 'b' if stm == 0 else 'w', name))
with open('data/nnue/depthprobe_boards.txt', 'w') as f:
    f.write('\n'.join(out) + '\n')
print('wrote %d probe positions' % len(out))
