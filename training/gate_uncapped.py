#!/usr/bin/env python3
"""Uncapped SPRT continuation: align20k vs R0. No game cap — runs until LLR
crosses +-2.94 or books exhaust (then MORE books get made, per fishtest).
Starts from banked totals; never replays a used (book, offset)."""
import glob
import re
import sys

sys.path.insert(0, 'training')
from regret_loop import run, parse_book_total, sprt_llr, EXEMPT_CHAMP if False else EXE

CAND = 'C:/gitProjects/titanium-ataxx/data/nnue/tupS54align20k.tup'
CHAMP = 'C:/gitProjects/mainwt/data/nnue/tupS54align.tup'

BOOKS = ['data/nnue/book100k.txt', 'data/nnue/book100l.txt',
         'data/nnue/book100m.txt', 'data/nnue/book100n.txt',
         'data/nnue/book100o.txt']

W = L = D = G = 0
seen = set()


def prior_total():
    W2 = L2 = D2 = G2 = 0
    for pat in ('logs/tupS54align_gate*.txt', 'logs/align20k_gate*.txt',
                'logs/align_fw*.txt'):
        for gl in sorted(glob.glob(pat)):
            if 'align_fw' in gl:
                continue
            m = re.search(r'_gate(\d+)\.txt$', gl)
            ck = int(m.group(1)) if m else -1
            if (gl, ck) in seen:
                continue
            seen.add((gl, ck))
            tot = parse_book_total(gl)
            if tot:
                W2, L2, D2 = W2 + tot[0], W2 + tot[1], D2 + tot[2]
                G2 += sum(tot)
    return W2, L2, D2, G2


def main():
    logf = open('logs/align_uncapped.txt', 'a')
    W, L, D, G = prior_total()
    msg = 'UNCAPPED START banked %d-%d-%d n=%d LLR %+.2f' % (W, L, D, G, sprt_llr(W, L, D))
    print(msg, flush=True)
    logf.write(msg + '\n')
    logf.flush()
    ck = 20
    verdict = None
    for book in BOOKS:
        for half in (0, 50):
            if verdict:
                break
            gl = 'logs/align_uncapped_%d.txt' % ck
            run([sys.executable, 'training/book_gate.py', '--book', book,
                 '--games', '100', '--nodes', '5000',
                 '--skip', str(half), '--take', '50',
                 '--titanium', EXE, '--targs', 'serve', '--tup', CAND,
                 '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP,
                 '--opp-nodes', '5000', '--workers', '2',
                 '--seed', str(48000 + ck), '--out', gl], logf)
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
    msg = 'UNCAPPED VERDICT: %s %d-%d-%d n=%d' % (verdict or 'STILL OPEN (make more books)', W, L, D, G)
    print(msg, flush=True)
    logf.write(msg + '\n')


if __name__ == '__main__':
    main()
