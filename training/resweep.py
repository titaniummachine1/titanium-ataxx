"""Blend re-sweep driver on the refreshed corpus.

Runs K outcome-weights on cache12M (same recipe/seed, only ow differs),
then gates each 100g @5k vs the fixed teacher-champ and writes a table.
Launched detached after the flywheel finishes (or alongside on a schedule).

Usage: python training/resweep.py --ows 0.0,0.05,0.10,0.15,0.20,0.25,0.30
"""
import argparse
import subprocess
import sys

import flywheel as F

CACHE = 'data/nnue/cache12M.npz'
TEACHER = 'data/nnue/tupS25ow20.tup'
LOG = 'data/nnue/resweep.log'


def train(ow, out, log):
    cmd = [sys.executable, 'training/train_tuple.py',
           '--cache', CACHE, '--tcol', 'teacher', '--blend', '1.0',
           '--outcome-weight', str(ow), '--dirs', '0,1', '--base', 'simple',
           '--lr', '0.05', '--seed', '7',
           '--plateau', '1e-4', '--max-epochs', '60',
           '--ckpt', out.replace('.tup', '_ep{ep}.tup'), '--out', out]
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
    ap.add_argument('--ows', default='0.0,0.05,0.10,0.15,0.20,0.25,0.30')
    a = ap.parse_args()
    ows = [float(x) for x in a.ows.split(',')]
    log = open(LOG, 'a')
    rows = []
    start = 25000
    for ow in ows:
        out = 'data/nnue/tupS29ow%d.tup' % round(ow * 100)
        rc = train(ow, out, log)
        if rc != 0:
            log.write('ow %.2f TRAIN FAILED rc=%d\n' % (ow, rc))
            log.flush()
            continue
        W, L, D, elo, lo, hi = F.run_gate(log, out, TEACHER, 100, start,
                                          'logs/s29_ow%d_100g.txt' % round(ow * 100))
        rows.append((ow, W, L, D, elo, lo, hi))
        start += 200
        table = '\n'.join('ow %.2f: %d-%d-%d elo %+.0f [LB %+.0f]' % r for r in rows)
        open('data/nnue/RESWEEP.txt', 'w').write(table + '\n')
        log.write('RESWEEP so far:\n%s\n' % table)
        log.flush()
    best = max(rows, key=lambda r: r[5]) if rows else None
    log.write('RESWEEP best by LB: %s\n' % (best,))
    log.close()


if __name__ == '__main__':
    main()
