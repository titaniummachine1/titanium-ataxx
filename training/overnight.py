#!/usr/bin/env python3
"""Overnight autonomous program (user-authorized, no interaction needed).
Phase 1: build exp/corrhistory release, SPRT-gate corr vs no-corr @5k.
         On pass: merge to main, push, LEDGER row. On fail/undecided: bank it.
Phase 2: regret loop v2 — 2M-node labels (doubled), fresh books C/D, to 06:30.
Phase 3: backfill DB, LEDGER closeout, HANDOFFS morning summary.
Every phase is try/except: failure logs and continues. Never idles.
"""
import datetime
import os
import subprocess
import sys
import time

HERE = r'C:\gitProjects\titanium-ataxx'
MAINWT = r'C:\gitProjects\mainwt'
EXE_CH = MAINWT + r'\target\release\titanium-cli.exe'

sys.path.insert(0, os.path.join(HERE, 'training'))
from regret_loop import run, parse_book_total, sprt_llr, GATE_BOOKS

CHAMP = MAINWT + r'\data\nnue\tupS53r0.tup'


def status(msg, sf):
    line = '%s %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    sf.write(line + '\n')
    sf.flush()


def phase_build(sf, logf):
    status('P1: building corrhistory release', sf)
    rc = run(['cargo', 'build', '--release', '-p', 'titanium-cli'],
             logf, cwd=MAINWT)
    status('P1: build rc=%d' % rc, sf)
    return rc == 0


def phase_ch_gate(sf, logf):
    status('P1: gating corr vs no-corr @5k SPRT', sf)
    W = L = D = 0
    verdict = None
    for ck in range(4):
        if verdict:
            break
        gl = 'logs/night_ch_gate%d.txt' % ck
        run([sys.executable, 'training/book_gate.py',
             '--book', GATE_BOOKS[ck % 3], '--games', '100', '--nodes', '5000',
             '--skip', str((ck * 50) % 100), '--take', '50',
             '--titanium', EXE_CH, '--targs', 'serve', '--tcorr',
             '--tup', CHAMP,
             '--opp', EXE_CH, '--oargs', 'serve', '--opp-tup', CHAMP,
             '--opp-nodes', '5000', '--workers', '2',
             '--seed', str(51000 + ck), '--out', gl], logf)
        tot = parse_book_total(gl)
        if not tot:
            continue
        W, L, D = W + tot[0], L + tot[1], D + tot[2]
        llr = sprt_llr(W, L, D)
        status('P1: chunk %d pooled %d-%d-%d LLR %+.2f' % (ck, W, L, D, llr), sf)
        if llr >= 2.94:
            verdict = 'pass'
        elif llr <= -2.94:
            verdict = 'fail'
    status('P1: CH verdict %s %d-%d-%d' % (verdict or 'undecided', W, L, D), sf)
    return verdict, (W, L, D)


def phase_merge(sf, logf, score):
    status('P1: merging exp/corrhistory to main (SPRT pass)', sf)
    cmds = [
        ['git', 'checkout', 'main'],
        ['git', 'merge', '--no-ff', 'exp/corrhistory', '-m',
         'S54 correction history (SPRT %s @5k books); default OFF' % score],
        ['git', 'push', 'origin', 'main'],
    ]
    for c in cmds:
        rc = run(c, logf, cwd=MAINWT)
        status('P1: %s rc=%d' % (' '.join(c[:3]), rc), sf)
        if rc != 0:
            return False
    return True


def main():
    os.makedirs('logs', exist_ok=True)
    sf = open('logs/overnight_status.txt', 'a')
    logf = open('logs/overnight.txt', 'a')
    status('NIGHT START', sf)
    try:
        if phase_build(sf, logf):
            verdict, (W, L, D) = phase_ch_gate(sf, logf)
            with open('logs/overnight_ch.txt', 'w') as f:
                f.write('corr-vs-nocorr @5k: %s %d-%d-%d\n' % (verdict, W, L, D))
            if verdict == 'pass':
                ok = phase_merge(sf, logf, '%d-%d-%d' % (W, L, D))
                with open('logs/overnight_ch.txt', 'a') as f:
                    f.write('merged=%s\n' % ok)
            else:
                status('P1: champ holds (corr banked on branch)', sf)
    except Exception as e:
        status('P1 FAILED: %r' % e, sf)
    try:
        status('P2: regret v2 (2M labels, books C/D) to 06:30', sf)
        run([sys.executable, 'training/regret_loop.py',
             '--deadline-min', '400',
             '--books', 'data/nnue/book100c.txt,data/nnue/book100d.txt',
             '--label-nodes', '2000000',
             '--workdir', 'data/nnue/regretloopv2'], logf)
        status('P2: v2 loop done', sf)
    except Exception as e:
        status('P2 FAILED: %r' % e, sf)
    try:
        status('P3: backfill + closeout', sf)
        run([sys.executable, 'training/results.py', 'backfill'], logf)
        with open('HANDOFFS.md', 'a') as f:
            f.write('\n## overnight %s: CH gate + regret v2 ran unattended; '
                    'see logs/overnight_status.txt, logs/overnight_ch.txt, '
                    'LEDGER S54 rows.\n' % datetime.date.today().isoformat())
        status('NIGHT END', sf)
    except Exception as e:
        status('P3 FAILED: %r' % e, sf)


if __name__ == '__main__':
    main()
