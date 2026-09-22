"""Nightly strength loop (independent of the flywheel trainer).

Watches for new flywheel checkpoints and gates each against the current
champion at 5k nodes, LB>0 promotion rule. Run detached alongside
training/flywheel.py — it never trains, only plays games.

Promotion: challenger PROMOTES iff 100g Elo lower-bound > 0 vs champ.
Then the new champ becomes the baseline for the next file (ratchet).
Also guards vs S13 hand eval (REGRESSION flag).

Usage: python training/nightly.py            # watch loop (default)
       python training/nightly.py --once data/nnue/fly12M_ep3.tup
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import time

EXE = 'target/release/titanium-cli.exe'
WATCH = 'data/nnue/fly12M_ep*.tup'
CHAMP_FILE = 'data/nnue/CHAMPION.txt'
BASE_CHAMP = 'data/nnue/tupS25ow20.tup'


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
    N = W + L + D
    assert N > 0, 'empty gate log'
    s = (W + D / 2.0) / N
    m2 = (W + D * 0.25) / N
    var = max((m2 - s * s) * N / max(N - 1, 1), 1e-12)
    se = math.sqrt(var / N)

    def f(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / x - 1)
    return f(s), f(s - 1.96 * se), f(s + 1.96 * se)


def champ():
    if os.path.exists(CHAMP_FILE):
        c = open(CHAMP_FILE).read().strip()
        if c and os.path.exists(c):
            return c
    return BASE_CHAMP


def gate(chall, opp_tup, games, start, out):
    cmd = [EXE, 'match', '--games', str(games), '--nodes', '5000']
    if chall:
        cmd += ['--tuple', chall]
    cmd += ['--opp', EXE + ' serve', '--opp-nodes', '5000']
    if opp_tup:
        cmd += ['--opp-tuple', opp_tup]
    cmd += ['--start-game', str(start), '--out', out]
    subprocess.run(cmd, capture_output=True, text=True)
    W, L, D = parse_match(out)
    elo, lo, hi = elo_ci(W, L, D)
    return W, L, D, elo, lo, hi


def judge(chall, log):
    c = champ()
    tag = os.path.basename(chall).replace('.tup', '')
    W, L, D, elo, lo, hi = gate(chall, c, 100, 24000, 'logs/nite_%s_vs_champ.txt' % tag)
    log.write('%s vs champ %s: %d-%d-%d elo %+.0f [LB %+.0f] -> ' %
              (chall, os.path.basename(c), W, L, D, elo, lo))
    if lo > 0:
        Wb, Lb, Db, eb, lob, hib = gate(chall, None, 100, 24100,
                                        'logs/nite_%s_vs_base.txt' % tag)
        if eb < 0:
            log.write('REGRESSION vs base (%d-%d-%d)\n' % (Wb, Lb, Db))
        else:
            open(CHAMP_FILE, 'w').write(chall)
            log.write('PROMOTED (vs base %d-%d-%d %+.0f)\n' % (Wb, Lb, Db, eb))
    else:
        log.write('STALLED\n')
    log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', default='')
    a = ap.parse_args()
    log = open('data/nnue/nightly.log', 'a')
    if a.once:
        judge(a.once, log)
        log.close()
        return
    if not os.path.exists(CHAMP_FILE):
        open(CHAMP_FILE, 'w').write(BASE_CHAMP)
    done = set()
    log.write('nightly watch start %s champ=%s\n' % (time.strftime('%H:%M:%S'), champ()))
    log.flush()
    while True:
        for f in sorted(glob.glob(WATCH)):
            if f not in done and os.path.getsize(f) > 10000:
                done.add(f)
                judge(f, log)
        time.sleep(60)


if __name__ == '__main__':
    main()
