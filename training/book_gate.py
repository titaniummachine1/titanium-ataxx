#!/usr/bin/env python3
"""Paired book gate: every book position played twice (both seats).
Kills the deterministic-replay problem (diverse lines) and mirrors
Moonbird's gate method (varied openings instead of startpos-only).

Line format (book): 49 chars top->bottom (.xo#) + ' ' + b/w.
Engines driven via UAI (`serve`), `go nodes N` / `go movetime MS`.

Usage:
  python3 training/book_gate.py --book data/nnue/book100.txt --games 200 \
    --nodes 5000 --titanium <exe> --tuple <tup|none> --moon <exe> \
    --opp-tuple <tup|none> --opp-nodes 5000 --workers 6 \
    --out logs/book_gate.txt
Jobs: book_resume = subset slice --skip/--take for parallel shards.
"""
import argparse
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

def sq_name(sq):
    return chr(ord('a') + sq % 7) + str(7 - sq // 7)


def parse_sq(name):
    f = ord(name[0].lower()) - ord('a')
    r = int(name[1])
    if not (0 <= f < 7 and 1 <= r <= 7):
        raise ValueError('bad sq ' + name)
    return (7 - r) * 7 + f


def parse_line(line):
    board, side = line.split()
    assert len(board) == 49
    cells = [0] * 49
    for i, ch in enumerate(board):
        cells[i] = 0 if ch == '.' else (1 if ch == 'x' else (2 if ch == 'o' else 3))
    return cells, (0 if side == 'b' else 1)


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
            row += 'x' if v == 1 else ('o' if v == 2 else '#')
        if gap:
            row += str(gap)
        rows.append(row)
    return '/'.join(rows) + (' x 0 1' if turn == 0 else ' o 0 1')


def neigh(sq):
    r, f = sq // 7, sq % 7
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            rr, ff = r + dr, f + df
            if 0 <= rr < 7 and 0 <= ff < 7:
                yield rr * 7 + ff


def neigh2(sq):
    r, f = sq // 7, sq % 7
    for dr in (-2, -1, 0, 1, 2):
        for df in (-2, -1, 0, 1, 2):
            if dr == 0 and df == 0:
                continue
            rr, ff = r + dr, f + df
            if 0 <= rr < 7 and 0 <= ff < 7:
                yield rr * 7 + ff


def has_moves(cells, side):
    piece = 1 if side == 0 else 2
    for sq, v in enumerate(cells):
        if v != piece:
            continue
        for t in neigh2(sq):
            if cells[t] == 0:
                return True
    return False


def apply(cells, turn, tok):
    piece = 1 if turn == 0 else 2
    tok = (tok or '').strip()
    if tok in ('', '0000', 'pass'):
        return 'pass'
    if len(tok) == 2:
        to = parse_sq(tok)
        frm = None
    elif len(tok) == 4:
        frm, to = parse_sq(tok[:2]), parse_sq(tok[2:])
    else:
        return 'illegal'
    if cells[to] != 0:
        return 'illegal'
    if frm is None:
        if not any(cells[s] == piece for s in neigh(to)):
            return 'illegal'
    else:
        if cells[frm] != piece:
            return 'illegal'
        d = max(abs(frm // 7 - to // 7), abs(frm % 7 - to % 7))
        if d not in (1, 2):
            return 'illegal'
        if d == 2:
            cells[frm] = 0
    cells[to] = piece
    foe = 2 if turn == 0 else 1
    for s in neigh(to):
        if cells[s] == foe:
            cells[s] = piece
    return 'ok'


def counts(cells):
    return sum(1 for v in cells if v == 1), sum(1 for v in cells if v == 2)


def over(cells):
    b, w = counts(cells)
    return (b == 0 or w == 0 or all(v != 0 for v in cells))


class Uai:
    def __init__(self, exe, args=()):
        self.p = subprocess.Popen([exe] + list(args), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._send('uai')
        self._until('uaiok')
        self._send('isready')
        self._until('readyok')

    def _send(self, s):
        self.p.stdin.write((s + '\n').encode())
        self.p.stdin.flush()

    def _rl(self):
        return self.p.stdout.readline().decode('utf-8', errors='replace').strip()

    def _until(self, tok, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._rl().startswith(tok):
                return
        raise RuntimeError('no ' + tok)

    def newgame(self):
        self._send('uainewgame')

    def go(self, fen, nodes, movetime, timeout=180):
        self._send('position fen ' + fen)
        self._send(('go nodes %d' % nodes) if nodes else ('go movetime %d' % movetime))
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self._rl()
            if line.startswith('bestmove'):
                ps = line.split()
                return ps[1] if len(ps) > 1 else '0000'
        return None

    def quit(self):
        try:
            self._send('quit')
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


WLOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', required=True)
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--nodes', type=int, default=0)
    ap.add_argument('--movetime', type=int, default=0)
    ap.add_argument('--titanium', required=True)
    ap.add_argument('--targs', nargs='*', default=[])
    ap.add_argument('--tup', default='none')
    ap.add_argument('--opp', required=True)
    ap.add_argument('--oargs', nargs='*', default=[])
    ap.add_argument('--opp-tup', default='none')
    ap.add_argument('--opp-nodes', type=int, default=0)
    ap.add_argument('--opp-movetime', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--take', type=int, default=0)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.book) if l.strip()]
    if a.take:
        lines = lines[a.skip:a.skip + a.take]
    rng = random.Random(a.seed)
    # tuple passthrough for titanium-cli serve children
    if a.titanium.endswith('.exe') and a.targs and a.targs[-1] == 'serve' and a.tup != 'none':
        a.targs = a.targs + ['--tuple', a.tup]
    if a.opp.endswith('.exe') and a.oargs and a.oargs[-1] == 'serve' and a.opp_tup != 'none':
        a.oargs = a.oargs + ['--tuple', a.opp_tup]
    # jobs: each position twice (both seats), interleaved, capped at --games
    jobs = []
    order = list(range(len(lines)))
    rng.shuffle(order)
    for i in order:
        jobs.append((lines[i], True, 'pos%d-black' % i))
        jobs.append((lines[i], False, 'pos%d-white' % i))
    jobs = jobs[:a.games]
    # per-side limits: titanium uses --nodes/--movetime, opp uses opp-*
    a.nodes, a.movetime = a.nodes, a.movetime
    with open(a.out, 'w') as f:
        f.write('book-gate %d jobs\n' % len(jobs))
    print('book-gate %d jobs' % len(jobs), flush=True)

    def run(job):
        line, tb, tag = job
        r = play_one_limits(a, line, tb, tag)
        with WLOCK:
            with open(a.out, 'a') as f:
                f.write('game %s [titan=%s]: %s b%d-w%d plies=%d\n' % (
                    r['tag'], r['titanium'], r['winner'], r['black'], r['white'], r['plies']))
        print('game %s [titan=%s]: %s (%d-%d)' % (
            r['tag'], r['titanium'], r['winner'], r['black'], r['white']), flush=True)
        return r

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run, jobs):
            results.append(r)
    w = sum(1 for r in results if r['winner'] == 'titanium')
    l = sum(1 for r in results if r['winner'] == 'moon')
    d = sum(1 for r in results if r['winner'] == 'draw')
    import math
    n = w + l + d
    s = (w + d / 2) / n
    se = math.sqrt(s * (1 - s) / n)
    summary = ('total: titan %d - moon %d draws %d (n=%d) score %.3f CI [%.3f,%.3f]\n'
               % (w, l, d, n, s, s - 1.96 * se, s + 1.96 * se))
    with open(a.out, 'a') as f:
        f.write(summary)
    print(summary, flush=True)


def play_one_limits(a, line, titan_black, tag):
    # same as play_one but opp side uses opp-nodes/opp-movetime
    cells, turn = parse_line(line)
    ta = Uai(a.titanium, a.targs)
    tb = Uai(a.opp, a.oargs)
    ta.newgame()
    tb.newgame()
    ply, passes = 0, 0
    forfeit = None
    while ply < 300 and not over(cells):
        titan_m = (turn == 0) == titan_black
        if not has_moves(cells, turn):
            passes += 1
            if passes >= 2:
                break
            turn = 1 - turn
            ply += 1
            continue
        passes = 0
        if titan_m:
            tok = ta.go(to_fen(cells, turn), a.nodes, a.movetime)
        else:
            on = a.opp_nodes or a.nodes
            om = a.opp_movetime or a.movetime
            tok = tb.go(to_fen(cells, turn), on, om)
        if tok is None:
            forfeit = 'moon' if not titan_m else 'titanium'
            break
        st = apply(cells, turn, tok)
        if st == 'illegal':
            forfeit = 'moon' if not titan_m else 'titanium'
            break
        turn = 1 - turn
        ply += 1
    ta.quit()
    tb.quit()
    b, w = counts(cells)
    if forfeit == 'titanium':
        winner = 'moon'
    elif forfeit == 'moon':
        winner = 'titanium'
    elif b == w:
        winner = 'draw'
    else:
        winner = 'titanium' if ((b > w) == titan_black) else 'moon'
    return {'tag': tag, 'titanium': 'black' if titan_black else 'white',
            'winner': winner, 'black': b, 'white': w, 'plies': ply}


if __name__ == '__main__':
    main()
