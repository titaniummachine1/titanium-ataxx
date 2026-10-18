#!/usr/bin/env python3
"""Optimizer bracket, PARALLEL (user-authorized): 4 arms at once (4 cores).
Same init (R0), same data, same seed; only --opt differs. Each arm trains
then SPRT-gates vs R0 independently. No adoption; ranking only.
"""
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE, GATE_BOOKS

ARMS = [
    ('sgd', ['--opt', 'sgd', '--lr', '0.01']),
    ('adam', ['--opt', 'adam']),
    ('adamw', ['--opt', 'adamw']),
    ('ranger', ['--opt', 'radam', '--lookahead']),
    ('adam_fd', ['--opt', 'adam', '--freq-decay', '0.1']),
]
CHAMP = r'C:\gitProjects\mainwt\data\nnue\tupS53r0.tup'
DATA = 'data/nnue/regretloop_r9.npz'
WLOCK = threading.Lock()


def status(msg, sf):
    line = '%s %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with WLOCK:
        sf.write(line + '\n')
        sf.flush()


def arm(name, flags, sf, logf):
    cand = 'data/nnue/opt_%s.tup' % name
    status('ARM %s: training' % name, sf)
    rc = run([sys.executable, 'training/train_tuple.py', '--cache', DATA,
              '--tcol', 'teacher', '--init', CHAMP, '--epochs', '12',
              '--out', cand, '--seed', '7'] + flags, logf)
    if rc != 0 or not os.path.exists(cand):
        return (name, 'train-fail 0-0-0')
    W = L = D = 0
    verdict = None
    for ck in range(4):
        if verdict:
            break
        gl = 'logs/opt_%s_gate%d.txt' % (name, ck)
        run([sys.executable, 'training/book_gate.py',
             '--book', GATE_BOOKS[ck % 3],
             '--games', '100', '--nodes', '5000',
             '--skip', str((ck * 50) % 100), '--take', '50',
             '--titanium', EXE, '--targs', 'serve', '--tup', cand,
             '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
             '--opp-nodes', '5000', '--workers', '1',
             '--seed', str(31000 + ck), '--out', gl], logf)
        tot = parse_book_total(gl)
        if not tot:
            continue
        W, L, D = W + tot[0], L + tot[1], D + tot[2]
        llr = sprt_llr(W, L, D)
        status('ARM %s: chunk %d pooled %d-%d-%d LLR %+.2f' % (
            name, ck, W, L, D, llr), sf)
        if llr >= 2.94:
            verdict = 'pass'
        elif llr <= -2.94:
            verdict = 'fail'
    return (name, '%s %d-%d-%d' % (verdict or 'undecided', W, L, D))


def main():
    os.makedirs('logs', exist_ok=True)
    sf = open('logs/opt_bracket_status.txt', 'a')
    logf = open('logs/opt_bracket.txt', 'a')
    status('BRACKET PARALLEL START data=%s arms=%d' % (DATA, len(ARMS)), sf)
    import concurrent.futures as cf
    out = {}
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(arm, n, f, sf, logf): n for n, f in ARMS}
        for fut in cf.as_completed(futs):
            n, r = fut.result()
            out[n] = r
            status('ARM %s DONE: %s' % (n, r), sf)
    status('BRACKET DONE %s' % out, sf)


if __name__ == '__main__':
    main()
