#!/usr/bin/env python3
"""Build regret cache from mined blunders: top-K drops with deep labels.
teacher = clipped deep_before (+/-1500, keeps mate-blunders as direction),
wsum = min(drop/100, 10) (critical rows weigh more), out = 0 (stm lost).
Usage: python training/build_regret.py --top 1000 --min-drop 50
"""
import argparse
import os
import re

import numpy as np

pat = re.compile(r'^(-?\d+) (\d+) (\d+)$')


def read_scores(tag):
    out = []
    for i in range(4):
        txt = open('data/nnue/regret/s_%s_%d.txt' % (tag, i),
                   encoding='utf-16', errors='replace').read()
        out.extend(int(l.split()[0]) for l in txt.splitlines()
                   if pat.match(l.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=1000)
    ap.add_argument('--min-drop', type=int, default=50)
    ap.add_argument('--wcap', type=float, default=10.0)
    ap.add_argument('--wdiv', type=float, default=100.0)
    ap.add_argument('--out', default='data/nnue/regret1k.npz')
    a = ap.parse_args()
    sb = read_scores('b')
    sa = read_scores('a')
    assert len(sb) == len(sa)
    boards = [l.strip().split()[0]
              for l in open('data/nnue/regret/befores.txt')]
    metas = [l.strip() for l in open('data/nnue/regret/meta.txt')]
    assert len(boards) == len(sb) == len(metas), (len(boards), len(sb))
    scored = []
    for i, (b, s) in enumerate(zip(sb, sa)):
        drop = b + s
        if drop >= a.min_drop:
            scored.append((drop, i))
    scored.sort(reverse=True)
    pick = [i for _, i in scored[:a.top]]
    print('candidates>=%d: %d, picked top %d' % (a.min_drop, len(scored), len(pick)), flush=True)
    occ0, occ1, stm, teach, ws, outs = [], [], [], [], [], []
    for i in pick:
        cells = boards[i]
        o0 = o1 = 0
        for sq, ch in enumerate(cells):
            if ch == 'x':
                o0 |= 1 << sq
            elif ch == 'o':
                o1 |= 1 << sq
        side = metas[i].split()[3]
        st = 0 if side == 'b' else 1
        drop = sb[i] + sa[i]
        occ0.append(o0)
        occ1.append(o1)
        stm.append(st)
        teach.append(max(-1500, min(1500, sb[i])))
        ws.append(min(drop / a.wdiv, a.wcap))
        outs.append(0.0)
    d = {'occ0': np.array(occ0, dtype=np.uint64),
         'occ1': np.array(occ1, dtype=np.uint64),
         'blk': np.zeros(len(pick), dtype=np.uint64),
         'stm': np.array(stm, dtype=np.uint8),
         'teacher': np.array(teach, dtype=np.float32),
         'wsum': np.array(ws, dtype=np.float64),
         'out': np.array(outs, dtype=np.float64),
         'sc': np.array(teach, dtype=np.float64)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **d)
    t = np.array(teach, dtype=np.float64)
    print('wrote %s n=%d teacher mean %+.1f std %.1f wsum mean %.2f' % (
        a.out, len(pick), t.mean(), t.std(), np.mean(ws)), flush=True)


if __name__ == '__main__':
    main()
