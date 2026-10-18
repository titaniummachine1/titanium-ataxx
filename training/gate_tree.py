#!/usr/bin/env python3
"""SPRT gate: tree-trained vs R0."""
import sys
sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE, GATE_BOOKS

CAND = r'C:\gitProjects\titanium-ataxx\data\nnue\tupS54tree.tup'
CHAMP = r'C:\gitProjects\mainwt\data\nnue\tupS53r0.tup'
CK = int(sys.argv[1])
gl = 'logs/tree_gate%d.txt' % CK
run([sys.executable, 'training/book_gate.py', '--book', GATE_BOOKS[CK % 3],
     '--games', '100', '--nodes', '5000', '--skip', str((CK * 50) % 100),
     '--take', '50', '--titanium', EXE, '--targs', 'serve', '--tup', CAND,
     '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
     '--opp-nodes', '5000', '--workers', '2',
     '--seed', str(45000 + CK), '--out', gl], open('logs/tree_gate.txt', 'a'))
tot = parse_book_total(gl)
print('chunk %d: %s' % (CK, tot), flush=True)
