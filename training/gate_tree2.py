#!/usr/bin/env python3
"""SPRT gate: tree2 (soft-backup) vs align. Chunked for watched running."""
import sys
sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE

CAND = 'C:/gitProjects/titanium-ataxx/data/nnue/upS54tree2.tup'
CHAMP = 'C:/gitProjects/mainwt/data/nnue/tupS54align.tup'
BOOKS = ['data/nnue/book100.txt', 'data/nnue/book100b.txt',
         'data/nnue/book100c.txt', 'data/nnue/book100d.txt']
CK = int(sys.argv[1])
gl = 'logs/tree2_gate%d.txt' % CK
logf = open('logs/tree2_gate.txt', 'a')
run([sys.executable, 'training/book_gate.py', '--book', BOOKS[CK % 4],
     '--games', '100', '--nodes', '5000', '--skip', str((CK * 50) % 100),
     '--take', '50', '--titanium', EXE, '--targs', 'serve', '--tup', CAND,
     '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
     '--opp-nodes', '5000', '--workers', '2', '--seed', str(47000 + CK),
     '--out', gl], logf)
tot = parse_book_total(gl)
print('chunk %d: %s' % (CK, tot), flush=True)
