#!/usr/bin/env python3
"""Cross-protocol parity: Botzone-native exe vs UAI exe, full games.
Framing mirrors validated bz_native_test.py. One side is native (history
lines), the other is UAI (FEN + go movetime). Shared board via book_gate.
Usage: python training/match_bz_uai.py --a EXE --b EXE --games 100
"""
import argparse
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
import book_gate as bg
from place_bench import Engine

START = 'x.....o' + '.' * 35 + 'o.....x'
TIMES = {'bz': [], 'uai': []}


def bz_move(exe, history_req, history_resp, cur):
    n = len(history_resp) + 1
    lines = [str(n)]
    for r, s in zip(history_req, history_resp):
        lines.append('%d %d %d %d' % r)
        lines.append('%d %d %d %d' % s)
    lines.append('%d %d %d %d' % cur)
    t0 = time.time()
    p = subprocess.run([exe], input='\n'.join(lines) + '\n', capture_output=True,
                       text=True, timeout=30)
    TIMES['bz'].append(time.time() - t0)
    out = (p.stdout or '').strip().split()
    assert len(out) == 4, (p.stdout, (p.stderr or '')[-200:])
    return tuple(map(int, out))


def uai_move(eng, cells, turn, movetime):
    board = ''.join('.' if v == 0 else ('x' if v == 1 else 'o') for v in cells)
    rows = [board[r * 7:(r + 1) * 7] for r in range(7)]
    fen = []
    for r in rows:
        s, gap = '', 0
        for ch in r:
            if ch == '.':
                gap += 1
                continue
            if gap:
                s += str(gap)
                gap = 0
            s += ch
        if gap:
            s += str(gap)
        fen.append(s)
    fen = '/'.join(fen) + (' x 0 1' if turn == 0 else ' o 0 1')
    t0 = time.time()
    tok = eng.go(fen, 0, movetime)
    TIMES['uai'].append(time.time() - t0)
    assert tok and tok not in ('None', '0000', 'pass'), tok
    if len(tok) == 2:
        to = bg.parse_sq(tok)
        return (to % 7, to // 7, to % 7, to // 7, True)
    return (ord(tok[0]) - 97, 7 - int(tok[1]), ord(tok[2]) - 97, 7 - int(tok[3]), False)


def play_game(exe, eng, bz_is_black, movetime, seed):
    cells = [1 if c == 'x' else (2 if c == 'o' else 0) for c in START]
    b_req = [(-1, -1, -1, -1)]
    b_resp, w_req, w_resp = [], [], []
    cur = (-1, -1, -1, -1)
    turn_black = True
    ply = 0
    for ply in range(400):
        if bg.over(cells):
            break
        turn = 0 if turn_black else 1
        if not bg.has_moves(cells, turn):
            turn_black = not turn_black
            continue
        is_bz = (turn == 0) == bz_is_black
        if is_bz:
            if turn_black:
                mv = bz_move(exe, b_req, b_resp, cur)
            else:
                mv = bz_move(exe, w_req, w_resp, cur)
        else:
            mv4 = uai_move(eng, cells, turn, movetime)
            mv, is_clone = mv4[:4], mv4[4]
            # find a legal from-square for clones
            to = mv[3] * 7 + mv[2]
            if is_clone:
                frm = None
                for s in bg.neigh(to):
                    if cells[s] == (1 if turn == 0 else 2):
                        frm = s
                        break
                assert frm is not None
                mv = (frm % 7, frm // 7, mv[2], mv[3])
        if mv[2] < 0:
            turn_black = not turn_black
            continue
        frm, to = mv[1] * 7 + mv[0], mv[3] * 7 + mv[2]
        piece = 1 if turn == 0 else 2
        assert cells[frm] == piece and cells[to] == 0, (ply, mv, turn)
        d = max(abs(mv[0] - mv[2]), abs(mv[1] - mv[3]))
        assert d in (1, 2), (ply, mv)
        tok = bg.sq_name(to) if d == 1 else bg.sq_name(frm) + bg.sq_name(to)
        assert bg.apply(cells, turn, tok) == 'ok', (ply, mv, tok)
        if turn_black:
            b_resp.append(mv)
            w_req.append(mv)
        else:
            w_resp.append(mv)
            b_req.append(mv)
        cur = mv
        turn_black = not turn_black
    b, w = bg.counts(cells)
    if b == w:
        return 'draw', ply + 1
    win_black = b > w
    if win_black == bz_is_black:
        return 'bz', ply + 1
    return 'uai', ply + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--tuple', default=r'C:\gitProjects\engines\titan_champ.tup')
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--movetime', type=int, default=700)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    eng = Engine(a.b, ['serve', '--tuple', a.tuple], 'uai', [])
    eng.newgame()
    res = {'bz': 0, 'uai': 0, 'draw': 0}
    out = open(a.out, 'w')
    for g in range(a.games):
        try:
            w, plies = play_game(a.a, eng, g % 2 == 0, a.movetime, g)
            res[w] += 1
            out.write('game %d bz=%s: %s plies=%d\n' %
                      (g, 'black' if g % 2 == 0 else 'white', w, plies))
            out.flush()
        except Exception as e:
            out.write('game %d ERROR %s\n' % (g, str(e)[:200]))
            out.flush()
    eng.quit()
    print('RESULT bz=%d uai=%d draws=%d games=%d' %
          (res['bz'], res['uai'], res['draw'], a.games))
    print('bz times: n=%d max=%.2f mean=%.2f' %
          (len(TIMES['bz']), max(TIMES['bz']), st.mean(TIMES['bz'])))
    print('uai times: n=%d max=%.2f mean=%.2f' %
          (len(TIMES['uai']), max(TIMES['uai']), st.mean(TIMES['uai'])))
    out.write('RESULT bz=%d uai=%d draws=%d\n' % (res['bz'], res['uai'], res['draw']))
    out.close()


if __name__ == '__main__':
    main()
