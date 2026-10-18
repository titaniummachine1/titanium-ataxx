#!/usr/bin/env python3
"""Self-distillation set: sample cache rows -> board lines for labeling
with OUR OWN net at depth (no moon graft). Keeps occ for npz assembly."""
import sys
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
rng = np.random.default_rng(seed)
d = np.load('data/nnue/cache400kR_graft.npz')
N = int(d['occ0'].shape[0])
idx = rng.choice(N, size=min(n, N), replace=False)
np.save('data/nnue/selfdist_idx.npy', idx)
with open('data/nnue/selfdist_boards.txt', 'w') as f:
    for i in idx:
        o0, o1, blk, stm = int(d['occ0'][i]), int(d['occ1'][i]), int(d['blk'][i]), int(d['stm'][i])
        s = ''.join('#' if (blk >> sq) & 1 else 'x' if (o0 >> sq) & 1
                    else 'o' if (o1 >> sq) & 1 else '.' for sq in range(49))
        f.write(s + (' b\n' if int(stm) == 0 else ' w\n'))
print('wrote %d boards + idx' % len(idx))
