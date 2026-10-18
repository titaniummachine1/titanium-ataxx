#!/usr/bin/env python3
"""Build training npz from top-blame rows: weight = disagreement (need more
compute gets more focus), teacher = deep 1M score. Sane caps throughout."""
import numpy as np

d = np.load('data/nnue/blame_top.npz')
idx = d['idx']
boards = [l.strip().split()[0] for l in open('data/nnue/regretloop/befores.txt')]
metas = [l.strip().split() for l in open('data/nnue/regretloop/meta.txt')]
occ0, occ1, stm, teach, ws, out = [], [], [], [], [], []
for j, i in enumerate(idx):
    i = int(i)
    cells = boards[i]
    o0 = o1 = 0
    for sq, ch in enumerate(cells):
        if ch == 'x':
            o0 |= 1 << sq
        elif ch == 'o':
            o1 |= 1 << sq
    st = 0 if metas[i][3] == 'b' else 1
    drop = float(d['dis'][j])
    t = float(np.clip(d['deep'][j], -1500, 1500))
    occ0.append(o0)
    occ1.append(o1)
    stm.append(st)
    teach.append(t)
    ws.append(min(drop / 100.0, 10.0))
    out.append(0.0)
t = np.array(teach, dtype=np.float32)
w = np.array(ws, dtype=np.float64)
z = {'occ0': np.array(occ0, np.uint64), 'occ1': np.array(occ1, np.uint64),
     'blk': np.zeros(len(idx), np.uint64), 'stm': np.array(stm, np.uint8),
     'teacher': t, 'wsum': w, 'out': np.zeros(len(idx)),
     'sc': t.astype(np.float64)}
np.savez_compressed('data/nnue/blame339.npz', **z)
print('wrote blame339.npz n=%d teach mean %+.0f wsum mean %.2f max %.1f' % (
    len(idx), t.mean(), w.mean(), w.max()))
