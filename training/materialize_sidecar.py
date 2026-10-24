"""One-time materialization of tuple SP/DE sidecars from a final *.npz cache.

Per training epoch the trainer re-runs features_for()+materialize32() over all
32.5M rows (~16 min/epoch, pure deterministic NumPy, no search). This script pays
that cost ONCE: SP/DE int32 index blocks are written per chunk as .npy files
(mmap-able), plus a manifest with sha256. The trainer (--sidecar-dir) then loads
chunks instead of recomputing. Bit-identical: same functions, same inputs.

Resumable: skip-if-exists, tmp+rename writes, manifest rewritten atomically.
A kill costs at most the in-flight chunk. No training happens here.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_stream as S

T_base = S.T.base_eval_stm


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/final/final_train.npz')
    ap.add_argument('--out-dir', default='data/nnue/final/sidecar')
    ap.add_argument('--chunk-rows', type=int, default=800000)
    ap.add_argument('--dirs', default='0,1,2,3')
    ap.add_argument('--verify-only', action='store_true',
                    help='recompute 2 chunks fresh and compare against sidecars')
    ap.add_argument('--only-base', action='store_true',
                    help='build only precomputed base-eval sidecars (base_cNNN.npy float32); '
                         'skips SP/DE. Same chunk grid. One-time ~10min; kills the '
                         'startup base-eval pass in all future runs.')
    a = ap.parse_args()
    active = [int(x) for x in a.dirs.split(',')]
    assert set(active) <= {0, 1, 2, 3} and active, a.dirs
    os.makedirs(a.out_dir, exist_ok=True)
    mpath = os.path.join(a.out_dir, 'manifest.json')

    z = np.load(a.cache, mmap_mode='r')
    n = int(z['occ0'].shape[0])
    WSP, WDE = S.sp_width(active), S.de_width(active)
    chunks = [(lo, min(lo + a.chunk_rows, n)) for lo in range(0, n, a.chunk_rows)]
    print('rows %d chunks %d WSP %d WDE %d (~%.1f GB total)'
          % (n, len(chunks), WSP, WDE, n * (WSP + WDE) * 4 / 1e9), flush=True)

    if a.verify_only:
        man = json.load(open(mpath))
        assert man['dirs'] == a.dirs and man['WSP'] == WSP and man['WDE'] == WDE, 'manifest mismatch'
        rng = np.random.RandomState(7)
        for ci in sorted(rng.choice(len(chunks), min(2, len(chunks)), replace=False)):
            lo, hi = chunks[int(ci)]
            f = S.features_for(np.asarray(z['occ0'][lo:hi], dtype=np.uint64),
                               np.asarray(z['occ1'][lo:hi], dtype=np.uint64),
                               np.asarray(z['blk'][lo:hi], dtype=np.uint64),
                               np.asarray(z['stm'][lo:hi], dtype=np.uint8))
            SP, DE = S.materialize32(*f, active)
            del f
            gc.collect()
            e = man['chunks'][int(ci)]
            SPs = np.load(e['sp'], mmap_mode='r')
            DEs = np.load(e['de'], mmap_mode='r')
            assert SPs.shape == SP.shape and DEs.shape == DE.shape, 'shape drift chunk %d' % ci
            assert np.array_equal(np.asarray(SPs), SP) and np.array_equal(np.asarray(DEs), DE), \
                'PARITY FAIL chunk %d' % ci
            print('parity ok chunk %d/%d rows %d-%d' % (ci + 1, len(chunks), lo, hi), flush=True)
        print('verify-only: PARITY OK', flush=True)
        return

    if a.only_base:
        man = json.load(open(mpath))
        assert man['rows'] == n, 'cache rows changed since sidecar build'
        have = set()
        for b in man.get('base', []):
            if os.path.exists(b['path']):
                have.add(b['ci'])
        print('base: %d/%d chunks already banked' % (len(have), len(chunks)), flush=True)
        t0 = time.time()
        bl = man.get('base', [])
        for ci, (lo, hi) in enumerate(chunks):
            if ci in have:
                continue
            b = T_base(np.asarray(z['occ0'][lo:hi], dtype=np.uint64),
                       np.asarray(z['occ1'][lo:hi], dtype=np.uint64),
                       np.asarray(z['blk'][lo:hi], dtype=np.uint64),
                       np.asarray(z['stm'][lo:hi], dtype=np.uint8)).astype(np.float32)
            bp = os.path.join(a.out_dir, 'base_c%03d.npy' % ci)
            np.save(bp + '.tmp', b)
            os.replace(bp + '.tmp.npy', bp)
            del b
            gc.collect()
            bl.append({'ci': ci, 'lo': lo, 'hi': hi, 'path': bp,
                       'sha': sha256(bp)[:16]})
            bl.sort(key=lambda c: c['ci'])
            man['base'] = bl
            json.dump(man, open(mpath + '.tmp', 'w'), indent=1)
            os.replace(mpath + '.tmp', mpath)
            print('base chunk %d/%d rows %d-%d banked (%.0fs elapsed)'
                  % (ci + 1, len(chunks), lo, hi, time.time() - t0), flush=True)
        print('base sidecar complete (%.0fs)' % (time.time() - t0), flush=True)
        return

    man = {'cache': os.path.abspath(a.cache), 'dirs': a.dirs, 'WSP': WSP, 'WDE': WDE,
           'chunk_rows': a.chunk_rows, 'rows': n, 'chunks': []}
    if os.path.exists(mpath):
        try:
            old = json.load(open(mpath))
            if old.get('dirs') == a.dirs and old.get('rows') == n and old.get('WSP') == WSP:
                man['chunks'] = old['chunks']
        except Exception as e:
            print('manifest unreadable, rebuilding: %s' % e, flush=True)
    done = {c['ci'] for c in man['chunks']
            if os.path.exists(c['sp']) and os.path.exists(c['de'])}
    print('resuming: %d/%d chunks already banked' % (len(done), len(chunks)), flush=True)
    t0 = time.time()
    for ci, (lo, hi) in enumerate(chunks):
        if ci in done:
            continue
        f = S.features_for(np.asarray(z['occ0'][lo:hi], dtype=np.uint64),
                           np.asarray(z['occ1'][lo:hi], dtype=np.uint64),
                           np.asarray(z['blk'][lo:hi], dtype=np.uint64),
                           np.asarray(z['stm'][lo:hi], dtype=np.uint8))
        SP, DE = S.materialize32(*f, active)
        del f
        gc.collect()
        sp_p = os.path.join(a.out_dir, 'sp_c%03d.npy' % ci)
        de_p = os.path.join(a.out_dir, 'de_c%03d.npy' % ci)
        np.save(sp_p + '.tmp', SP)  # creates sp_cNNN.npy.tmp.npy
        os.replace(sp_p + '.tmp.npy', sp_p)
        np.save(de_p + '.tmp', DE)
        os.replace(de_p + '.tmp.npy', de_p)
        del SP, DE
        gc.collect()
        man['chunks'].append({'ci': ci, 'lo': lo, 'hi': hi, 'sp': sp_p, 'de': de_p,
                              'sha_sp': sha256(sp_p)[:16], 'sha_de': sha256(de_p)[:16]})
        man['chunks'].sort(key=lambda c: c['ci'])
        json.dump(man, open(mpath + '.tmp', 'w'), indent=1)
        os.replace(mpath + '.tmp', mpath)
        print('chunk %d/%d rows %d-%d banked (%.0fs elapsed)'
              % (ci + 1, len(chunks), lo, hi, time.time() - t0), flush=True)
    print('sidecar complete: %d chunks (%.0fs)' % (len(chunks), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
