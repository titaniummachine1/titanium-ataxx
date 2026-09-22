"""Overnight flywheel: train cache12M ow0.20 to plateau, then auto-gate.

Step 1 (tonight): full-corpus epochs until val relative-improvement stalls
  twice below 1e-4 (your 4-digit rule), cap 60 epochs, every-epoch ckpt +
  best-val copied to out. Detached; appends to fly12M.log.
Step 2 (morning): 100g @5k new-net vs ow20-teacher + vs S13 base; the numbers
  decide merge, not val (val never crowns).

Launch (detached): python training/flywheel.py
Resume/morning gate: python training/flywheel.py --gate-only
"""
import argparse
import subprocess
import sys
import time

CACHE = 'data/nnue/cache12M.npz'
TEACHER = 'data/nnue/tupS25ow20.tup'
OUT = 'data/nnue/tupS26fly.tup'
CKPT = 'data/nnue/fly12M_ep{ep}.tup'
LOG = 'data/nnue/fly12M.log'
EXE = 'target/release/titanium-cli.exe'


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


def train(log):
    rc = run([sys.executable, 'training/train_tuple.py',
              '--cache', CACHE, '--tcol', 'teacher', '--blend', '1.0',
              '--outcome-weight', '0.20', '--dirs', '0,1', '--base', 'simple',
              '--lr', '0.05', '--seed', '7',
              '--plateau', '1e-4', '--max-epochs', '60',
              '--ckpt', CKPT, '--out', OUT], log)
    log.write('train rc=%d\n' % rc)
    log.flush()
    return rc


def gate(log, games=100, nodes=5000):
    for name, a, b in [('fly-vs-teacher', OUT, TEACHER),
                       ('fly-vs-base', None, None)]:
        cmd = [EXE, 'match', '--games', str(games), '--nodes', str(nodes)]
        if a:
            cmd += ['--tuple', a]
        cmd += ['--opp', EXE + ' serve', '--opp-nodes', str(nodes)]
        if b:
            cmd += ['--opp-tuple', b]
        cmd += ['--start-game', '23000' if 'teacher' in name else '23100',
                '--out', 'logs/s26_%s_%dg.txt' % (name, games)]
        rc = run(cmd, log)
        log.write('%s rc=%d\n' % (name, rc))
        log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate-only', action='store_true')
    ap.add_argument('--games', type=int, default=100)
    a = ap.parse_args()
    log = open(LOG, 'a')
    log.write('\n===== flywheel %s %s =====\n'
              % (time.strftime('%Y-%m-%d %H:%M:%S'), 'gate-only' if a.gate_only else 'train+gate'))
    log.flush()
    if not a.gate_only:
        if train(log) != 0:
            log.write('TRAIN FAILED, abort\n')
            log.close()
            return
    gate(log, games=a.games)
    log.write('===== flywheel done =====\n')
    log.close()


if __name__ == '__main__':
    main()
