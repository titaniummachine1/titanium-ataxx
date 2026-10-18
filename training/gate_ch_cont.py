#!/usr/bin/env python3
"""CH gate continuation: corr vs no-corr, uncapped SPRT to +-2.94.
Picks up banked totals from ch_gate* + ch_ext_*.txt, runs books P-T.
Same binary, same net, same flags — only fresh positions."""
import glob
import sys

sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXE

CHAMP = 'C:/gitProjects/mainwt/data/nnue/tupS54align.tup'
BOOKS = [('data/nnue/book100p.txt', 0), ('data/nnue/book100p.txt', 50),
         ('data/nnue/book100q.txt', 0), ('data/nnue/book100q.txt', 50),
         ('data/nnue/book100r.txt', 0), ('data/nnue/book100r.txt', 50),
         ('data/nnue/book100s.txt', 0), ('data/nnue/book100s.txt', 50),
         ('data/nnue/book100t.txt', 0), ('data/nnue/book100t.txt', 50)]


def main():
    W = L = D = 0
    seen = set()
    for pat in ('logs/ch_gate*.txt', 'logs/ch_ext_*.txt'):
        for gl in sorted(glob.glob(pat)):
            if gl in seen:
                continue
            seen.add(gl)
            tot = parse_book_total(gl)
            if tot:
                W, L, D = W + tot[0], L + tot[1], D + tot[2]
    G = W + L + D
    print('CH-CONT START banked %d-%d-%d n=%d LLR %+.2f' % (W, L, D, G, sprt_llr(W, L, D)), flush=True)
    logf = open('logs/ch_ext2.txt', 'a')
    ck = 30
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
             '--opp-nodes', '5000', '--workers', '2', '--seed', str(51000 + ck),
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
    print('CH-CONT VERDICT: %s %d-%d-%d n=%d' % (verdict or 'STILL OPEN', W, L, D, G), flush=True)


if __name__ == '__main__':
    main()
