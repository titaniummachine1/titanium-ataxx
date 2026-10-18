#!/usr/bin/env python3
"""Blame rank: for each mined position, disagreement = |deep(1M) - shallow(5k)|
= how much it NEEDS compute; times game impact (drop from meta/mined scores).
Outputs ranked list. No training, no heavy compute.
"""
import re
import numpy as np

pat = re.compile(r'^(-?\d+) (\d+) (\d+)$')


def read_scores(path):
    out = []
    raw = open(path, 'rb').read()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        txt = raw.decode('utf-16')
    else:
        txt = raw.decode('utf-8', errors='replace')
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if m:
            out.append(int(m.group(1)))
    return np.array(out, dtype=np.float64)


deep = []
import glob
for f in sorted(glob.glob('data/nnue/regret1m_b_*.txt')):
    deep.extend(read_scores(f))
deep = np.array(deep)
shallow = read_scores('data/nnue/regret5k_b.txt')
n = min(len(deep), len(shallow))
deep, shallow = deep[:n], shallow[:n]
dis = np.abs(deep - shallow)
order = np.argsort(-dis)
print('rows=%d deep_mean=%+.0f shallow_mean=%+.0f' % (n, deep.mean(), shallow.mean()))
print('disagreement: mean=%.0f med=%.0f p90=%.0f max=%.0f' % (
    dis.mean(), np.median(dis), np.quantile(dis, 0.9), dis.max()))
print('rows with dis>=500: %d (%.1f%%)' % ((dis >= 500).sum(), 100 * (dis >= 500).mean()))
boards = [l.strip() for l in open('data/nnue/regretloop/befores.txt')]
metas = [l.strip() for l in open('data/nnue/regretloop/meta.txt')]
with open('data/nnue/blame_ranked.txt', 'w') as f:
    for i in order:
        f.write('%s # dis=%d deep=%d shallow=%d\n' % (boards[i], int(dis[i]),
                                                      int(deep[i]), int(shallow[i])))
top = [int(i) for i in order[:339]]
np.savez_compressed('data/nnue/blame_top.npz',
                    idx=np.array(top, dtype=np.int64),
                    deep=deep[top], shallow=shallow[top],
                    dis=dis[top])
print('wrote data/nnue/blame_ranked.txt top5 dis:',
      [int(dis[i]) for i in order[:5]])
