"""Overnight moon-distill: moon-label hard200k -> train -> gate.

Best plan while you sleep (nothing else touches the box):
1. Let the running resweep finish (7 ow nets on fresh 12M, table in RESWEEP.txt).
2. Moon-label hard200k @5k with Moonbird UAI (foreign teacher, max diversity).
   Moonbird @5k on hard midgame rows runs ~1-3 rows/s single-process, so
   200k rows = ~20-50h SINGLE worker. Chunked into 8 x 25k with one UAI
   process each (8 workers, 4c/8t box) -> ~3-7h. Fits the night.
3. Train moon-teacher student (ow sweep 0.0/0.1/0.2 on moon labels) + gate
   100g @5k vs champ, LB>0 rule. Verdict in MOONDISTILL.txt.

Launch detached: python training/moon_distill.py
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

import numpy as np

EXE = 'scripts/moonbird/Moonbird-1.1.0-windows-amd64.exe'

CACHE = 'data/nnue/cache12M.npz'
HARD = 'data/nnue/hard200k.npy'
CHAMP = 'data/nnue/tupS25ow20.tup'
LOG = 'data/nnue/moondistill.log'
LABDIR = 'data/nnue/lab_moon'


def board_line(o0, o1, blk, stm):
    a0, a1, ab = int(o0), int(o1), int(blk)
    s = ''.join('#' if (ab >> sq) & 1 else 'x' if (a0 >> sq) & 1
                else 'o' if (a1 >> sq) & 1 else '.' for sq in range(49))
    return s + (' b\n' if int(stm) == 0 else ' w\n')


def label_chunk(txt, out, nodes):
    # one moon_label worker per chunk (own persistent UAI process inside)
    if os.path.exists(out):
        want = sum(1 for _ in open(txt, 'rb'))
        got = len(open(out, errors='ignore').read().strip().split())
        if got == want:
            return 'SKIP-OK %d' % got
    p = subprocess.run([sys.executable, 'training/moon_label.py', '--in', txt,
                        '--out', out, '--nodes', str(nodes)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return 'FAIL %s' % p.stderr[-300:]
    return 'OK'


def run(cmd, log):
    log.write('\n### %s\n' % ' '.join(cmd))
    log.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        log.write(line)
        log.flush()
    p.wait()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=200000)
    ap.add_argument('--nodes', type=int, default=5000)
    ap.add_argument('--ows', default='0.0,0.10,0.20')
    a = ap.parse_args()
    log = open(LOG, 'a')
    log.write('\n===== moondistill %s top=%d nodes=%d =====\n'
              % (time.strftime('%Y-%m-%d %H:%M:%S'), a.top, a.nodes))
    log.flush()

    # 1. dump hard boards
    os.makedirs(LABDIR, exist_ok=True)
    idx = np.load(HARD)[:a.top].astype(np.int64)
    z = np.load(CACHE, mmap_mode='r')
    per = 25000
    txts = []
    for ci in range(0, len(idx), per):
        txt = os.path.join(LABDIR, 'moon_%02d.txt' % (ci // per))
        if not os.path.exists(txt):
            with open(txt, 'w') as f:
                for r in idx[ci:ci + per]:
                    f.write(board_line(z['occ0'][r], z['occ1'][r], z['blk'][r], z['stm'][r]))
            log.write('dump %s %d rows\n' % (txt, min(per, len(idx) - ci)))
            log.flush()
        txts.append(txt)

    # 2. moon-label chunks in parallel (8 UAI workers, 4c/8t box).
    # Moonbird @5k hard rows ~1-3/s/worker -> 200k / 8 / 2 ≈ 3.5h.
    pairs = [(t, t.replace('.txt', '.out')) for t in txts]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(label_chunk, t, o, a.nodes): t for t, o in pairs}
        for fut in cf.as_completed(futs):
            txt = futs[fut]
            msg = fut.result()
            log.write('%s: %s\n' % (os.path.basename(txt), msg))
            log.flush()
            if msg.startswith('FAIL'):
                log.write('MOON LABEL FAILED %s\n' % txt)
                log.close()
                return

    # 3. join moon teacher cache
    parts = []
    for txt in txts:
        t = np.loadtxt(txt.replace('.txt', '.out'), dtype=np.float32, ndmin=1)
        parts.append(t)
    moon = np.concatenate(parts)
    assert len(moon) == len(idx), (len(moon), len(idx))
    log.write('moon teacher: n=%d mean %+.1f std %.1f\n' %
              (len(moon), float(moon.mean()), float(moon.std())))
    log.flush()
    c = np.load(CACHE)
    d = {k: c[k] for k in c.files}
    t = d['teacher'].astype(np.float64)
    t[idx] = moon.astype(np.float64)
    d['teacher'] = t.astype(np.float32)
    mc = 'data/nnue/cache12M_moon%dk.npz' % (len(idx) // 1000)
    np.savez_compressed(mc, **d)
    log.write('wrote %s %.1fMB\n' % (mc, os.path.getsize(mc) / 1e6))
    log.flush()

    # 4. train + gate each ow on the moon cache (sequential, box is busy enough)
    import sys as _sys
    _sys.path.insert(0, 'training')
    import flywheel as F
    rows = []
    start = 26000
    for ow in [float(x) for x in a.ows.split(',')]:
        out = 'data/nnue/tupS30moon%d.tup' % round(ow * 100)
        rc = run([sys.executable, 'training/train_tuple.py',
                  '--cache', mc, '--tcol', 'teacher', '--blend', '1.0',
                  '--outcome-weight', str(ow), '--dirs', '0,1', '--base', 'simple',
                  '--lr', '0.05', '--seed', '7',
                  '--plateau', '1e-4', '--max-epochs', '60',
                  '--ckpt', out.replace('.tup', '_ep{ep}.tup'), '--out', out], log)
        if rc != 0:
            log.write('ow %.2f TRAIN FAILED\n' % ow)
            log.flush()
            continue
        W, L, D, elo, lo, hi = F.run_gate(log, out, CHAMP, 100, start,
                                          'logs/s30_moon%d_100g.txt' % round(ow * 100))
        rows.append((ow, W, L, D, elo, lo, hi))
        start += 200
    table = '\n'.join('moon ow %.2f: %d-%d-%d elo %+.0f [LB %+.0f]' % r for r in rows)
    open('data/nnue/MOONDISTILL.txt', 'w').write(table + '\n')
    log.write('MOONDISTILL:\n%s\n' % table)
    best = max(rows, key=lambda r: r[5]) if rows else None
    log.write('best by LB: %s\n===== moondistill done =====\n' % (best,))
    log.close()


if __name__ == '__main__':
    main()
