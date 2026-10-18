#!/usr/bin/env python3
"""Assemble full 20k alignment set: pilot 2k + 18k deep labels + meta."""
import re
import numpy as np

pat = re.compile(r'^(-?\d+) (\d+) (\d+)$')


def read(p):
    raw = open(p, 'rb').read()
    txt = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', errors='replace')
    out = []
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if m:
            out.append(int(m.group(1)))
    return out


sc = []
for i in range(4):
    sc.extend(read('data/nnue/align2k_s%d.txt' % i))
for i in range(4):
    sc.extend(read('data/nnue/align18k_s%d.txt' % i))
sc = np.array(sc, dtype=np.float32)
m = np.load('data/nnue/align20k_meta.npz')
n = len(sc)
assert n == len(m['occ0']), (n, len(m['occ0']))
teach = np.clip(sc, -1500, 1500).astype(np.float32)
z = {'occ0': m['occ0'][:n].astype(np.uint64), 'occ1': m['occ1'][:n].astype(np.uint64),
     'blk': m['blk'][:n].astype(np.uint64), 'stm': m['stm'][:n].astype(np.uint8),
     'teacher': teach, 'wsum': np.ones(n),
     'out': np.zeros(n), 'sc': teach.astype(np.float64)}
np.savez_compressed('data/nnue/align20k.npz', **z)
print('wrote align20k.npz n=%d teach mean %+.0f std %.0f' % (n, teach.mean(), teach.std()))
