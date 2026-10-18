#!/usr/bin/env python3
"""CH gate extension: corr vs no-corr, uncapped SPRT to +-2.94.
Continues from banked chunks 0-3 (A0 B0 C0 D0). Fresh: A50 B50 C50 D50,
then K L M N O full books. Stops on pass/fail.
NOTE: same-binary + --tcorr flag (pre two-binary rule); config proven by
per-log sha+corr headers. Two-binary rule applies to NEW gates."""
import glob
import re
import sys

sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE

CHAMP = 'C:/gitProjects/mainwt/data/nnue/tupS54align.tup'
BOOKS = [('data/nnue/book100.txt', 50), ('data/nnue/book100b.txt', 50),
         ('data/nnue/book100c.txt', 50), ('data/nnue/book100d.txt', 50),
         ('data/nnue/book100e.txt', 0), ('data/nnue/book100e.txt', 50),
         ('data/nnue/book100f.txt', 0), ('data/nnue/book100f.txt', 50),
         ('data/nnue/book100g.txt', 0), ('data/nnue/book100g.txt', 50),
         ('data/nnue/book100h.txt', 0), ('data/nnue/book100h.txt', 50),
         ('data/nnue/book100i.txt', 0), ('data/nnue/book100i.txt', 50),
         ('data/nnue/book100j.txt', 0), ('data/nnue/book100j.txt', 50),
         ('data/nnue/book100k.txt', 0), ('data/nnue/book100k.txt', 50),
         ('data/nnue/book100l.txt', 0), ('data/nnue/book100l.txt', 50),
         ('data/nnue/book100m.txt', 0), ('data/nnue/book100m.txt', 50),
         ('data/nnue/book100n.txt', 0), ('data/nnue/book100n.txt', 50),
         ('data/nnue/book100o.txt', 0), ('data/nnue/book100o.txt', 50)]

W = L = D = G = 0
seen = set()
for gl in sorted(glob.glob('logs/ch_gate*.txt')):
    if gl in seen:
        continue
    seen.add(gl)
    tot = parse_book_total(gl)
    if tot:
        W, L, D = W + tot[0], L + tot[1], D + tot[2]
        G += sum(tot)
print('CH-EXT START banked %d-%d-%d n=%d LLR %+.2f' % (W, L, D, G, sprt_llr(W, L, D)), flush=True)


def main():
    global W, L, D, G
    logf = open('logs/ch_ext.txt', 'a')
    ck = 4
    verdict = None
    for book, half in BOOKS:
        if verdict:
            break
        gl = 'logs/ch_ext_%d.txt' % ck
        run([sys.executable, 'training/book_gate.py', '--book', book,
             '--games', '100', '--nodes', '5000',
             '--skip', str(half), '--take', '50',
             '--titanium', EXE, '--targs', 'serve', '--tcorr', '--tup', CHAMP,
             '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
             '--opp-nodes', '5000', '--workers', '2', '--seed', str(50000 + ck),
             '--out', gl], logf)
        tot = parse_book_total(gl)
        if tot:
            W, L, D = W + tot[0], L + tot[1], D + tot[2]
            G = W + L + D
        llr = sprt_llr(W, L, D)
        msg = 'chunk %d (%s:%d) pooled %d-%d-%d n=%d LLR %+.2f' % (
            ck, book[-12:-4], half, W, L, D, G, llr)
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()
        ck += 1
        if llr >= 2.94:
            verdict = 'pass'
        elif llr <= -2.94:
            verdict = 'fail'
    print('CH-EXT VERDICT: %s %d-%d-%d n=%d' % (verdict or 'STILL OPEN', W, L, D, G), flush=True)


if __name__ == '__main__':
    main()
