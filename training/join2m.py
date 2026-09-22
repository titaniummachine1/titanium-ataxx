"""Join 2M @20k label outputs + cache2m bitboards/wsum/sc -> cache2mR.npz.

Aborts (nonzero exit, no output file) unless every chunk has exactly 125000
score lines in input order. Run in background; check the log tail.
"""
import numpy as np
import os
import sys


def main():
    b = np.load('data/nnue/cache2m.npz')
    parts = []
    for c in range(8):
        for s in ('a', 'b'):
            p = 'data/nnue/lab2m_%d%s.out' % (c, s)
            d = np.loadtxt(p, usecols=0)
            if len(d) != 125000:
                print('ABORT: %s has %d lines, want 125000' % (p, len(d)), flush=True)
                sys.exit(1)
            parts.append(d)
    t = np.concatenate(parts)
    assert len(t) == 2000000
    print('teacher20k: mean %+.1f std %.1f rails +%.3f -%.3f'
          % (t.mean(), t.std(), (t >= 1500).mean(), (t <= -1500).mean()), flush=True)
    np.savez_compressed('data/nnue/cache2mR.npz', occ0=b['occ0'], occ1=b['occ1'],
                        blk=b['blk'], stm=b['stm'], out=b['out'], sc=b['sc'],
                        teacher=t.astype(np.float32), wsum=b['wsum'])
    print('wrote cache2mR %.1fMB' % (os.path.getsize('data/nnue/cache2mR.npz') / 1e6), flush=True)


if __name__ == '__main__':
    main()
