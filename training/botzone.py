#!/usr/bin/env python3
"""Botzone simple-protocol bridge for titanium-cli (serve mode).

The judge re-executes this program every turn. stdin framing (mirrors the
official sample):
  turnID                      # our move number N (1-based)
  2*(N-1) history lines       # (opp, own) pairs for rounds 1..N-1
  1 current opp line          # opponent's just-played move
Each move line: x0 y0 x1 y1 (Botzone coords: x=file 0-6 L-to-R, y=0 TOP row;
-1 -1 -1 -1 = black-first marker / pass). We reply one move line.

Color: black iff requests[0] is all -1 (black moves first).
Mapping: Botzone (x,y) -> sq y*7+x -> algebraic chr(a+x)+str(7-y).
Engine: explicit `position fen` + `go movetime|nodes`; parse bestmove
(4-char from+to, 2-char clone dest, pass).
Repetition guard (Botzone loop rule): never play a move recreating a past
(cells, turn); fall back to a random legal move (stderr-logged).
No-move: reply -1 -1 -1 -1.

Usage: python training/botzone.py --exe target/release/titanium-cli.exe
         --tuple <tup> [--movetime 800 | --nodes N]
"""
import argparse
import random
import subprocess
import sys


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Engine:
    def __init__(self, exe, args):
        self.p = subprocess.Popen(
            [exe] + list(args), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _send(self, s):
        self.p.stdin.write((s + '\n').encode())
        self.p.stdin.flush()

    def _rl(self):
        return self.p.stdout.readline().decode('utf-8', errors='replace').strip()

    def _until(self, tok, limit=900):
        for _ in range(limit):
            if self._rl().startswith(tok):
                return
        raise RuntimeError('no ' + tok)

    def setup(self):
        self._send('uai')
        self._until('uaiok')
        self._send('isready')
        self._until('readyok')

    def query(self, fen, movetime=None, nodes=None):
        self._send('position fen ' + fen)
        self._send('go movetime %d' % movetime if movetime else 'go nodes %d' % nodes)
        for _ in range(7200):
            l = self._rl()
            if l.startswith('bestmove'):
                ps = l.split()
                return ps[1] if len(ps) > 1 else None
        return None

    def quit(self):
        try:
            self._send('quit')
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def sq_alg(sq):
    return chr(ord('a') + sq % 7) + str(7 - sq // 7)


def parse_alg(tok):
    return (7 - int(tok[1])) * 7 + (ord(tok[0].lower()) - ord('a'))


def to_fen(cells, turn):
    rows = []
    for r in range(7):
        row, gap = '', 0
        for f in range(7):
            v = cells[r * 7 + f]
            if v == 0:
                gap += 1
                continue
            if gap:
                row += str(gap)
                gap = 0
            row += 'x' if v == 1 else 'o'
        if gap:
            row += str(gap)
        rows.append(row)
    return '/'.join(rows) + (' x 0 1' if turn == 0 else ' o 0 1')


def neigh(sq, dist):
    r, f = sq // 7, sq % 7
    for dr in range(-dist, dist + 1):
        for df in range(-dist, dist + 1):
            if dr == 0 and df == 0:
                continue
            rr, ff = r + dr, f + df
            if 0 <= rr < 7 and 0 <= ff < 7:
                yield rr * 7 + ff


def legal_from(cells, piece, frm):
    """Legal landings for one piece (clone+jump)."""
    return [t for t in neigh(frm, 2) if cells[t] == 0]


def all_legal(cells, piece):
    out = []
    for s, v in enumerate(cells):
        if v != piece:
            continue
        for t in legal_from(cells, piece, s):
            out.append((s, t))
    return out


def apply(cells, piece, frm, to):
    cells = list(cells)
    foe = 3 - piece
    if max(abs(frm // 7 - to // 7), abs(frm % 7 - to % 7)) == 2:
        cells[frm] = 0
    cells[to] = piece
    for s in neigh(to, 1):
        if cells[s] == foe:
            cells[s] = piece
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True)
    ap.add_argument('--args', default='serve')
    ap.add_argument('--tuple', default='')
    ap.add_argument('--movetime', type=int, default=800)
    ap.add_argument('--nodes', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    eargs = a.args.split()
    if a.tuple:
        eargs += ['--tuple', a.tuple]

    data = sys.stdin.read().strip().split()
    if not data:
        return
    nums = list(map(int, data))
    turn_id = nums[0]
    rest = nums[1:]
    n_rounds = turn_id - 1
    assert len(rest) == 4 * (2 * n_rounds + 1), (len(rest), n_rounds)
    reqs, resps = [], []
    for i in range(n_rounds):
        reqs.append(tuple(rest[4 * (2 * i):4 * (2 * i) + 4]))
        resps.append(tuple(rest[4 * (2 * i + 1):4 * (2 * i + 1) + 4]))
    cur_opp = tuple(rest[4 * 2 * n_rounds:4 * 2 * n_rounds + 4])
    reqs.append(cur_opp)

    me = 1 if all(v < 0 for v in reqs[0]) else 2
    foe = 3 - me
    cells = [0] * 49
    for (x, y, v) in ((0, 0, 1), (6, 6, 1), (6, 0, 2), (0, 6, 2)):
        cells[y * 7 + x] = v
    seen = {(tuple(cells), 0)}
    turn = 0  # black moves first
    for i in range(turn_id):
        # opponent move i (pass lines flip the turn without board change;
        # the game-start -1 marker at reqs[0] is not a turn)
        x0, y0, x1, y1 = reqs[i]
        if x1 >= 0:
            cells = apply(cells, foe if me == 1 else 1,
                          y0 * 7 + x0, y1 * 7 + x1)
            turn = 1 - turn
        elif not (i == 0 and all(v < 0 for v in reqs[0])):
            turn = 1 - turn
        seen.add((tuple(cells), turn))
        # our move i (except current round)
        if i < turn_id - 1:
            m = resps[i]
            if m[2] >= 0:
                cells = apply(cells, me, m[1] * 7 + m[0], m[3] * 7 + m[2])
            turn = 1 - turn
            seen.add((tuple(cells), turn))
    assert turn == (0 if me == 1 else 1), (turn, me)

    mv = all_legal(cells, me)
    if not mv:
        print('-1 -1 -1 -1', flush=True)
        return
    eng = Engine(a.exe, eargs)
    eng.setup()
    fen = to_fen(cells, 0 if me == 1 else 1)
    best = eng.query(fen, movetime=None if a.nodes else a.movetime,
                     nodes=a.nodes or None)
    eng.quit()
    pick = None
    if best and best not in ('pass', '0000', 'null'):
        if len(best) == 2:
            to = parse_alg(best)
            cands = [s for (s, t) in mv if t == to]
            if cands:
                frm = cands[0]
                if tuple(apply(cells, me, frm, to)) not in seen:
                    pick = (frm, to)
        elif len(best) == 4:
            frm, to = parse_alg(best[:2]), parse_alg(best[2:4])
            if (frm, to) in mv and tuple(apply(cells, me, frm, to)) not in seen:
                pick = (frm, to)
    if pick is None:
        # repetition guard tripped or parse failure: random legal non-repeat
        fresh = [(s, t) for (s, t) in mv
                 if tuple(apply(cells, me, s, t)) not in seen]
        pool = fresh or mv
        pick = rng.choice(pool)
        log('botzone: fallback move (repeat-guard or parse), pool=%d' % len(pool))
    (fx, fy), (tx, ty) = (pick[0] % 7, pick[0] // 7), (pick[1] % 7, pick[1] // 7)
    print('%d %d %d %d' % (fx, fy, tx, ty), flush=True)


if __name__ == '__main__':
    main()
