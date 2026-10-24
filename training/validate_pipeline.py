"""Pipeline validation ONLY. No optimizer, no backward, no weight writes.

Checks:
1. All launch files exist (cache, sidecars+manifest, init net).
2. Sidecar manifest matches cache rows + trainer widths for --dirs.
3. Required WDL columns present in cache.
4. Forward-only smoke: build the exact EmbeddingBag model, run 2 batches
   no_grad from sidecar chunk 0, print shapes/timing. Nothing is updated,
   nothing is written. This is NOT a training run.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_stream as S
import torch

CACHE = 'data/nnue/final/final_train_v2.npz'
SCDIR = 'data/nnue/final/sidecar'
INIT = 'data/nnue/tupS56twostage.tup'
DIRS = '0,1,2,3'


def need(path):
    assert os.path.exists(path), 'MISSING: %s' % path
    print('ok: %s (%.1f GB)' % (path, os.path.getsize(path) / 1e9))


def main():
    for p in (CACHE, INIT, os.path.join(SCDIR, 'manifest.json')):
        need(p)
    man = json.load(open(os.path.join(SCDIR, 'manifest.json')))
    z = np.load(CACHE, mmap_mode='r')
    n = int(z['occ0'].shape[0])
    assert man['rows'] == n, 'sidecar rows %d != cache rows %d' % (man['rows'], n)
    assert man['dirs'] == DIRS
    active = [int(x) for x in DIRS.split(',')]
    assert man['WSP'] == S.sp_width(active) and man['WDE'] == S.de_width(active)
    for c in man['chunks']:
        assert os.path.exists(c['sp']) and os.path.exists(c['de']), 'missing chunk %d' % c['ci']
    print('ok: %d sidecar chunks cover %d rows' % (len(man['chunks']), n))
    for col in ('out', 'outcome_p', 'teacher', 'teacher_p', 'teacher_src', 'sc', 'wsum',
                'ataxx_p_win', 'ataxx_p_draw', 'ataxx_p_loss', 'ataxx_ply'):
        assert col in z, 'missing column %s' % col
    print('ok: WDL columns present; teacher_src mix:',
          {int(k): int((np.asarray(z['teacher_src']) == k).sum()) for k in (0, 1, 2, 3)})
    pw = np.asarray(z['ataxx_p_win'], dtype=np.float64)
    pdw = np.asarray(z['ataxx_p_draw'], dtype=np.float64)
    pl = np.asarray(z['ataxx_p_loss'], dtype=np.float64)
    rel = ~np.isnan(pw)
    ps = (pw + pdw + pl)[rel]
    print('ok: ataxx relabel present on %d rows, p-sum [%.6f, %.6f]'
          % (int(rel.sum()), ps.min(), ps.max()))
    assert abs(ps.min() - 1) < 1e-5 and abs(ps.max() - 1) < 1e-5

    # Forward-only smoke on sidecar chunk 0 (no_grad, no optimizer, no writes).
    c0 = man['chunks'][0]
    SP = np.load(c0['sp'], mmap_mode='r')
    DE = np.load(c0['de'], mmap_mode='r')
    torch.manual_seed(7)
    bags = torch.nn.ModuleList(
        [torch.nn.EmbeddingBag(S.SP_N, 1, mode='sum', sparse=False),
         torch.nn.EmbeddingBag(S.DE_N, 1, mode='sum', sparse=False)])
    for bg in bags:
        torch.nn.init.zeros_(bg.weight)
    bias = torch.nn.Parameter(torch.zeros(1))
    with torch.no_grad():
        t = time.time()
        for i in (0, 4096):
            Xsp = torch.from_numpy(np.asarray(SP[i:i + 4096]).astype(np.int64))
            Xde = torch.from_numpy(np.asarray(DE[i:i + 4096]).astype(np.int64))
            pred = bags[0](Xsp).squeeze(1) + bags[1](Xde).squeeze(1) + bias
            assert pred.shape == (min(4096, len(SP) - i),)
        print('ok: forward-only smoke 2x4096 batches in %.1fs (no updates, no writes)'
              % (time.time() - t))
    print('PIPELINE READY. Launch only on explicit user order: '
          'training\\run_v2_wdl.bat')


if __name__ == '__main__':
    main()
