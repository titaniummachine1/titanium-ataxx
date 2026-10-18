#!/usr/bin/env python3
"""Assemble self-distillation pilot npz: self-deep teachers, uniform weights
(the mixture IS the weighting)."""
import re
import numpy as np

pat = re.compile(r'^(-?\d+) (\d+) (\d+)$')


def read(p):
    out = []
    raw = open(p, 'rb').read()
    txt = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', errors='replace')
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if m:
            out.append(int(m.group(1)))
    return out


sc = []
for i in range(4):
    sc.extend(read('data/nnue/align2k_s%d.txt' % i))
sc = np.array(sc, dtype=np.float32)
m = np.load('data/nnue/align20k_meta.npz')
n = len(sc)
print('scores=%d meta=%d' % (n, len(m['occ0']))),
occ0, occ1 = m['occ0'][:n], m['occ1'][:n]
teach = np.clip(sc, -1500, 1500).astype(np.float32)
z = {'occ0': occ0.astype(np.uint64), 'occ1': occ1.astype(np.uint64),
     'blk': m['blk'][:n].astype(np.uint64), 'stm': m['stm'][:n].astype(np.uint8),
     'teacher': teach, 'wsum': np.ones(n),
     'out': np.zeros(n), 'sc': teach.astype(np.float64)}
np.savez_compressed('data/nnue/align2k.npz', **z)
print('wrote align2k.npz teach mean %+.0f std %.0f' % (teach.mean(), teach.std()))
