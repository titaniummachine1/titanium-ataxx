#!/usr/bin/env python3
"""Blame assignment: replay lost games, emit our-move positions for deep scoring.
Input: match log with move lists. Output: befores.txt + meta (game, ply, move,
side) + afters.txt (aligned: position AFTER our move, foe stm).
Board lines: 49 chars top->bottom (.xo) + ' ' + b/w (Board::from_str format).

Usage: python training/mine_blunders.py --log logs/s53_mine_5k.txt \
         --out data/nnue/regret --max-games 0
"""
import argparse
import os
import re


def parse_sq(name):
    f = ord(name[0].lower()) - ord('a')
    r = int(name[1])
    return (7 - r) * 7 + f


def neigh(sq):
    r, f = sq // 7, sq % 7
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            rr, ff = r + dr, f + df
            if 0 <= rr < 7 and 0 <= ff < 7:
                yield rr * 7 + ff


def apply(cells, turn, tok):
    piece = 1 if turn == 0 else 2
    tok = tok.strip()
    if tok == 'pass':
        return True
    if len(tok) == 2:
        to = parse_sq(tok)
        frm = None
    elif len(tok) == 4:
        frm, to = parse_sq(tok[:2]), parse_sq(tok[2:])
    else:
        return False
    if cells[to] != 0:
        return False
    if frm is None:
        src = [s for s in neigh(to) if cells[s] == piece]
        if not src:
            return False
    else:
        if cells[frm] != piece:
            return False
        d = max(abs(frm // 7 - to // 7), abs(frm % 7 - to % 7))
        if d not in (1, 2):
            return False
        if d == 2:
            cells[frm] = 0
    cells[to] = piece
    foe = 2 if turn == 0 else 1
    for s in neigh(to):
        if cells[s] == foe:
            cells[s] = piece
    return True


def board_line(cells, turn):
    s = ''.join('.' if v == 0 else 'x' if v == 1 else 'o' for v in cells)
    return s + (' b' if turn == 0 else ' w')


GM_RE = re.compile(r'^-- game (\d+): titanium=(x|o) opp=(x|o): (\S+) (\S+)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-games', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fb = open(os.path.join(a.out, 'befores.txt'), 'w')
    fa = open(os.path.join(a.out, 'afters.txt'), 'w')
    fm = open(os.path.join(a.out, 'meta.txt'), 'w')
    ng = nb = 0
    lines = open(a.log, encoding='utf-8', errors='replace').read().splitlines()
    i = 0
    while i < len(lines):
        m = GM_RE.match(lines[i])
        if not m:
            i += 1
            continue
        g, tcol, ocol, res1, res2 = m.groups()
        g = int(g)
        moves = lines[i + 1].strip().split() if i + 1 < len(lines) else []
        i += 2
        # titanium wins -> skip (we mine LOSSES); draws skip
        titan_won = (res1 == 'titanium' and res2 == 'wins') or \
                    (res1 == 'opponent' and False)
        if res1 == 'titanium' and res2 == 'wins':
            continue
        if res1 == 'draw' or res2 == 'draw':
            continue
        if a.max_games and ng >= a.max_games:
            break
        ng += 1
        tside = 0 if tcol == 'x' else 1
        # startpos: black a1 (42), white g7 (6)
        cells = [0] * 49
        cells[42] = 1
        cells[6] = 2
        turn = 0
        ply = 0
        for tok in moves:
            if tside == turn:
                # our move: record before + after
                fb.write(board_line(cells, turn) + '\n')
                ok = apply(cells, turn, tok)
                if not ok:
                    break
                fa.write(board_line(cells, 1 - turn) + '\n')
                fm.write('%d %d %s %s\n' % (g, ply, tok, 'b' if turn == 0 else 'w'))
                nb += 1
            else:
                if not apply(cells, turn, tok):
                    break
            turn = 1 - turn
            ply += 1
    print('games mined %d, our-move positions %d' % (ng, nb), flush=True)


if __name__ == '__main__':
    main()
