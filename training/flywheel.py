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
import math
import subprocess
import sys
import time


def parse_match(path):
    W = L = D = 0
    for line in open(path, errors='ignore'):
        if line.startswith('-- game'):
            if 'titanium wins' in line:
                W += 1
            elif 'opponent wins' in line:
                L += 1
            else:
                D += 1
    return W, L, D


def elo_ci(W, L, D):
    """Elo point + 95% CI from game scores (1/0.5/0). Promotion iff lo > 0."""
    N = W + L + D
    assert N > 0, 'empty gate log'
    s = (W + D / 2.0) / N
    m2 = (W + D * 0.25) / N
    var = max((m2 - s * s) * N / max(N - 1, 1), 1e-12)
    se = math.sqrt(var / N)

    def f(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / x - 1)
    return f(s), f(s - 1.96 * se), f(s + 1.96 * se), s, se

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


def run_gate(log, tup, opp_tup, games, start, out):
    cmd = [EXE, 'match', '--games', str(games), '--nodes', '5000']
    if tup:
        cmd += ['--tuple', tup]
    cmd += ['--opp', EXE + ' serve', '--opp-nodes', '5000']
    if opp_tup:
        cmd += ['--opp-tuple', opp_tup]
    cmd += ['--start-game', str(start), '--out', out]
    rc = run(cmd, log)
    W, L, D = parse_match(out)
    elo, lo, hi, s, se = elo_ci(W, L, D)
    log.write('%s: %d-%d-%d elo %+.0f 95CI [%+.0f,%+.0f] s=%.3f se=%.4f rc=%d\n'
              % (out, W, L, D, elo, lo, hi, s, se, rc))
    log.flush()
    return W, L, D, elo, lo, hi


def promote_gate(log):
    """S26 promotion rule: new net vs champ (champ as baseline), 100g @5k.
    PROMOTE iff 95% lower bound > 0. Else statistical refine: extend to 200g
    and re-test. Then magnitude/regression guard vs S13 hand eval."""
    W, L, D, elo, lo, hi = run_gate(log, OUT, TEACHER, 100, 23000,
                                    'logs/s26_fly_vs_champ_100g.txt')
    verdict = None
    if lo > 0:
        verdict = 'PROMOTED-100g'
    else:
        log.write('champ-gate LB %+.0f <= 0: extending to 200g (statistical refine)\n' % lo)
        log.flush()
        W2, L2, D2, _, _, _ = run_gate(log, OUT, TEACHER, 100, 23200,
                                       'logs/s26_fly_vs_champ_xtra100g.txt')
        W, L, D = W + W2, L + L2, D + D2
        elo, lo, hi, s, se = elo_ci(W, L, D)
        log.write('champ-gate combined 200g: %d-%d-%d elo %+.0f 95CI [%+.0f,%+.0f]\n'
                  % (W, L, D, elo, lo, hi))
        log.flush()
        verdict = 'PROMOTED-200g' if lo > 0 else 'STALLED'
    Wb, Lb, Db, eb, lob, hib = run_gate(log, OUT, None, 100, 23400,
                                        'logs/s26_fly_vs_base_100g.txt')
    if eb < 0:
        verdict = 'REGRESSION-vs-base'
    with open('data/nnue/PROMOTION.txt', 'w') as f:
        f.write('net: %s\nteacher/champ: %s\n' % (OUT, TEACHER))
        f.write('vs-champ: %d-%d-%d elo %+.0f 95CI [%+.0f,%+.0f]\n' % (W, L, D, elo, lo, hi))
        f.write('vs-base: %d-%d-%d elo %+.0f 95CI [%+.0f,%+.0f]\n' % (Wb, Lb, Db, eb, lob, hib))
        f.write('verdict: %s\n' % verdict)
    log.write('PROMOTION VERDICT: %s\n' % verdict)
    log.flush()
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate-only', action='store_true')
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
    verdict = promote_gate(log)
    log.write('verdict %s, ranking hard positions for round 2\n' % verdict)
    log.flush()
    rc = run([sys.executable, 'training/rank_hard.py', '--net', OUT,
              '--top', '200000', '--out', 'data/nnue/hard200k.npy'], log)
    log.write('rank rc=%d\n' % rc)
    log.flush()
    log.write('===== flywheel done: %s =====\n' % verdict)
    log.close()


if __name__ == '__main__':
    main()
