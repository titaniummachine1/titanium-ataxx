#!/usr/bin/env python3
"""SPRT extension: align20k vs R0, chunks 8-19 on fresh books E-J.
Stops early on pass/fail (|LLR| >= 2.94). Cap 2000 games total.
"""
import glob
import re
import sys

sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE

CAND = r'C:\gitProjects\titanium-ataxx\data\nnue\tupS54align20k.tup'
CHAMP = r'C:\gitProjects\mainwt\data\nnue\tupS53r0.tup'
BOOKS = ['data/nnue/book100e.txt', 'data/nnue/book100f.txt',
         'data/nnue/book100g.txt', 'data/nnue/book100h.txt',
         'data/nnue/book100i.txt', 'data/nnue/book100j.txt']

W = L = D = 0
import glob as _g
for gl in sorted(_g.glob('logs/tupS54align20k_gate*.txt')) + ['logs/align20k_gate4.txt']:
    tot = parse_book_total(gl)
    if tot:
        W, L, D = W + tot[0], L + tot[1], D + tot[2]
print('prior: %d-%d-%d LLR %+.2f' % (W, L, D, sprt_llr(W, L, D)), flush=True)
logf = open('logs/align_ext.txt', 'a')
verdict = None
ck = 8
for bi, book in enumerate(BOOKS):
    for half in (0, 50):
        if verdict:
            break
        gl = 'logs/align20k_gate%d.txt' % ck
        run([sys.executable, 'training/book_gate.py', '--book', book,
             '--games', '100', '--nodes', '5000',
             '--skip', str(half), '--take', '50',
             '--titanium', EXE, '--targs', 'serve', '--tup', CAND,
             '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
             '--opp-nodes', '5000', '--workers', '2',
             '--seed', str(45000 + ck), '--out', gl], logf)
        tot = parse_book_total(gl)
        if tot:
            W, L, D = W + tot[0], L + tot[1], D + tot[2]
        llr = sprt_llr(W, L, D)
        msg = 'chunk %d (%s:%d) pooled %d-%d-%d LLR %+.2f' % (ck, book[-12:-4], half, W, L, D, llr)
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()
        if llr >= 2.94:
            verdict = 'pass'
        elif llr <= -2.94:
            verdict = 'fail'
        ck += 1
print('EXTENDED VERDICT: %s %d-%d-%d' % (verdict or 'undecided', W, L, D), flush=True)
