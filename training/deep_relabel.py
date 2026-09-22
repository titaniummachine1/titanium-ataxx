"""Deep selective relabel: hard200k rows at high nodes, merged back to cache.

Round 2 after the 5k flywheel: relabel ONLY the optimizer-weight top rows
(|residual|*wsum — where the net is most wrong on visited positions) at
--nodes with the champ teacher, then overwrite those rows' teacher column
in a copy of cache12M. Retrain + re-gate from there.

Usage: python training/deep_relabel.py --nodes 20000 --in data/nnue/hard200k.npy --top 200000
  -> data/nnue/cache12M_deep20k.npz
Resumable: lab_deep/*.txt + .out skip-OK unless --redo.
"""
import argparse
import os
import subprocess
import sys

import numpy as np


def board_line(o0, o1, blk, stm):
    a0, a1, ab = int(o0), int(o1), int(blk)
    s = ''.join('#' if (ab >> sq) & 1 else 'x' if (a0 >> sq) & 1
                else 'o' if (a1 >> sq) & 1 else '.' for sq in range(49))
    return s + (' b\n' if int(stm) == 0 else ' w\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', type=int, default=20000)
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--in', dest='inp', default='data/nnue/hard200k.npy')
    ap.add_argument('--top', type=int, default=200000)
    ap.add_argument('--tuple', default='data/nnue/tupS25ow20.tup')
    ap.add_argument('--dir', default='data/nnue/lab_deep')
    ap.add_argument('--out', default='')
    ap.add_argument('--redo', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    idx = np.load(a.inp)[:a.top].astype(np.int64)
    print('hard rows %d nodes %d' % (len(idx), a.nodes), flush=True)

    z = np.load(a.cache, mmap_mode='r')
    n = len(idx)
    per = 25000
    chunks = [(i, idx[i:i + per]) for i in range(0, n, per)]
    print('chunks %d' % len(chunks), flush=True)
    scores = np.zeros(n, dtype=np.float32)
    for ci, (lo, cidx) in enumerate(chunks):
        txt = os.path.join(a.dir, 'deep_%02d.txt' % ci)
        out = os.path.join(a.dir, 'deep_%02d.out' % ci)
        if not (os.path.exists(txt) and not a.redo):
            with open(txt, 'w') as f:
                for r in cidx:
                    f.write(board_line(z['occ0'][r], z['occ1'][r], z['blk'][r], z['stm'][r]))
        if os.path.exists(out) and not a.redo:
            t = np.loadtxt(out, usecols=0, dtype=np.float32, ndmin=1)
            if len(t) == len(cidx):
                print('chunk %02d SKIP-OK' % ci, flush=True)
                scores[lo:lo + len(cidx)] = t
                continue
        cmd = ['target/release/titanium-cli.exe', 'label', '--nodes', str(a.nodes),
               '--tuple', a.tuple, '--tuple-native']
        with open(txt, 'rb') as fin, open(out, 'wb') as fout:
            p = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
        t = np.loadtxt(out, usecols=0, dtype=np.float32, ndmin=1)
        assert len(t) == len(cidx), (ci, len(t), len(cidx))
        scores[lo:lo + len(cidx)] = t
        print('chunk %02d OK %d mean %+.1f' % (ci, len(t), float(t.mean())), flush=True)

    out = a.out or ('data/nnue/cache12M_deep%dk.npz' % (a.nodes // 1000))
    print('loading cache for merge...', flush=True)
    c = np.load(a.cache)
    d = {k: c[k] for k in c.files}
    t = d['teacher'].astype(np.float64)
    t[idx] = scores.astype(np.float64)
    d['teacher'] = t.astype(np.float32)
    np.savez_compressed(out, **d)
    print('wrote %s %.1fMB deep rows %d mean %+.1f' %
          (out, os.path.getsize(out) / 1e6, len(idx), float(scores.mean())), flush=True)


if __name__ == '__main__':
    main()
