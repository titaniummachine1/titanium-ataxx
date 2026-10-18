#!/usr/bin/env python3
"""SPRT gate: S54r0_1m (1M-label regret) vs R0. Chunked for watched running."""
import sys
sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE, GATE_BOOKS

CAND = sys.argv[2] if len(sys.argv) > 2 else r'C:\gitProjects\titanium-ataxx\data\nnue\tupS54r0_1m.tup'
CHAMP = r'C:\gitProjects\mainwt\data\nnue\tupS53r0.tup'
CK = int(sys.argv[1]) if len(sys.argv) > 1 else 0

logf = open('logs/s54_gate.txt', 'a')
import os
tag = os.path.splitext(os.path.basename(CAND))[0]
gl = 'logs/%s_gate%d.txt' % (tag, CK)
run([sys.executable, 'training/book_gate.py', '--book', GATE_BOOKS[CK % 3],
     '--games', '100', '--nodes', '5000',
     '--skip', str((CK * 50) % 100), '--take', '50',
     '--titanium', EXE, '--targs', 'serve', '--tup', CAND,
     '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
     '--opp-nodes', '5000', '--workers', '2',
     '--seed', str(60000 + CK), '--out', gl], logf)
tot = parse_book_total(gl)
print('chunk %d: %s' % (CK, tot), flush=True)
