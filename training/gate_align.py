#!/usr/bin/env python3
"""SPRT gate: align2k self-distilled vs R0."""
import sys
sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE, GATE_BOOKS

CAND = r'C:\gitProjects\titanium-ataxx\data\nnue\upS54align2k.tup'
CHAMP = r'C:\gitProjects\mainwt\data\nnue\tupS53r0.tup'
CK = int(sys.argv[1])
# Fresh (book, offset) pairs only. align20k used A0 B0 C0 A50 (chunks 0-3).
SLOTS = [('data/nnue/book100b.txt', 50), ('data/nnue/book100c.txt', 50),
         ('data/nnue/book100d.txt', 0), ('data/nnue/book100d.txt', 50),
         ('data/nnue/book100e.txt', 0), ('data/nnue/book100e.txt', 50),
         ('data/nnue/book100f.txt', 0), ('data/nnue/book100f.txt', 50)]
BOOK, SKIP = SLOTS[(CK - 4) % len(SLOTS)]

logf = open('logs/align_gate.txt', 'a')
gl = 'logs/align20k_gate%d.txt' % CK
run([sys.executable, 'training/book_gate.py', '--book', BOOK,
     '--games', '100', '--nodes', '5000',
     '--skip', str(SKIP), '--take', '50',
     '--titanium', EXE, '--targs', 'serve', '--tup', CAND,
     '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
     '--opp-nodes', '5000', '--workers', '2',
     '--seed', str(44000 + CK), '--out', gl], logf)
tot = parse_book_total(gl)
print('chunk %d: %s' % (CK, tot), flush=True)
