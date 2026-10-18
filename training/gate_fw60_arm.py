#!/usr/bin/env python3
"""SPRT gate: opt_adam_fw60 vs R0. No adoption; verdict only."""
import sys
sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE, GATE_BOOKS

CAND = r'C:\gitProjects\titanium-ataxx\data\nnue\opt_adam_fw60.tup'
CHAMP = r'C:\gitProjects\mainwt\data\nnue\tupS53r0.tup'

logf = open('logs/opt_fw60_gate.txt', 'a')
W = L = D = 0
verdict = None
for ck in range(4):
    if verdict:
        break
    gl = 'logs/opt_fw60_gate%d.txt' % ck
    run([sys.executable, 'training/book_gate.py', '--book', GATE_BOOKS[ck % 3],
         '--games', '100', '--nodes', '5000',
         '--skip', str((ck * 50) % 100), '--take', '50',
         '--titanium', EXE, '--targs', 'serve', '--tup', CAND,
         '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
         '--opp-nodes', '5000', '--workers', '1',
         '--seed', str(43000 + ck), '--out', gl], logf)
    tot = parse_book_total(gl)
    if not tot:
        continue
    W, L, D = W + tot[0], L + tot[1], D + tot[2]
    llr = sprt_llr(W, L, D)
    print('chunk %d pooled %d-%d-%d LLR %+.2f' % (ck, W, L, D, llr), flush=True)
    logf.write('chunk %d pooled %d-%d-%d LLR %+.2f\n' % (ck, W, L, D, llr))
    logf.flush()
    if llr >= 2.94:
        verdict = 'pass'
    elif llr <= -2.94:
        verdict = 'fail'
print('FW60-ARM VERDICT: %s %d-%d-%d' % (verdict or 'undecided', W, L, D), flush=True)
