#!/usr/bin/env python3
"""Placement benchmark: Titanium vs foreign engines (UAI or UCI dialects).

Paired seats per book position, fixed nodes both sides, Elo readout.
Reuses ataxx rules + FEN from book_gate. Log header carries both exe shas.
Usage:
  python3 training/place_bench.py --book data/nnue/book100a.txt --games 100 \
    --nodes 5000 --a-name titanium --a-exe <exe> --a-args serve --tuple <tup> \
    --b-name fairy14 --b-exe <exe> --b-dialect uci --out logs/place_titan_fairy.txt
"""
import argparse
import hashlib
import math
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, 'training')
import book_gate as bg


class Engine:
    def __init__(self, exe, args, dialect, setup=()):
        self.p = subprocess.Popen([exe] + list(args), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if dialect == 'uci':
            self._send('uci')
            self._until('uciok')
            for s in setup:
                self._send(s)
            self._send('isready')
            self._until('readyok')
        else:
            self._send('uai')
            self._until('uaiok')
            for s in setup:
                self._send(s)
            self._send('isready')
            self._until('readyok')
        self.dialect = dialect

    def _send(self, s):
        self.p.stdin.write((s + '\n').encode())
        self.p.stdin.flush()

    def _rl(self):
        return self.p.stdout.readline().decode('utf-8', errors='replace').strip()

    def _until(self, tok, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._rl().startswith(tok):
                return
        raise RuntimeError('no ' + tok)

    def newgame(self):
        self._send('ucinewgame' if self.dialect == 'uci' else 'uainewgame')

    def go(self, fen, nodes, movetime=0, timeout=180):
        if self.dialect == 'uci':
            # Fairy-ataxx FEN: pieces p/P (black/white), side b/w, empty pocket kills drops
            board, rest = fen.split(' ', 1)
            board = board.replace('x', 'p').replace('o', 'P')
            rest = rest.replace('x 0 1', 'b 0 1').replace('o 0 1', 'w 0 1')
            fen = board + '[] ' + rest
        self._send('position fen ' + fen)
        self._send(('go movetime %d' % movetime) if movetime else ('go nodes %d' % nodes))
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self._rl()
            if line.startswith('bestmove'):
                ps = line.split()
                tok = ps[1] if len(ps) > 1 else '0000'
                return tok.replace('-', '').strip().lower()
        return None

    def go_score(self, fen, nodes, timeout=600):
        """Like go(), but also returns the info-line score (for deep labeling)."""
        if self.dialect == 'uci':
            board, rest = fen.split(' ', 1)
            board = board.replace('x', 'p').replace('o', 'P')
            rest = rest.replace('x 0 1', 'b 0 1').replace('o 0 1', 'w 0 1')
            fen = board + '[] ' + rest
        self._send('position fen ' + fen)
        self._send('go nodes %d' % nodes)
        sc, mv = None, None
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self._rl()
            if line.startswith('info '):
                toks = line.split()
                if 'cp' in toks:
                    try:
                        sc = float(toks[toks.index('cp') + 1])
                    except (ValueError, IndexError):
                        pass
            if line.startswith('bestmove'):
                ps = line.split()
                mv = ps[1] if len(ps) > 1 else '0000'
                break
        return (sc if sc is not None else 0.0), (mv or '0000')

    def quit(self):
        try:
            self._send('quit')
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_one(a_spec, b_spec, line, a_black, tag, nodes, movetime):
    cells, turn = bg.parse_line(line)
    moves_log = []
    ea = Engine(a_spec['exe'], a_spec['args'], a_spec['dialect'], a_spec['setup'])
    eb = Engine(b_spec['exe'], b_spec['args'], b_spec['dialect'], b_spec['setup'])
    ea.newgame()
    eb.newgame()
    ply, passes = 0, 0
    forfeit = None
    while ply < 300 and not bg.over(cells):
        a_move = (turn == 0) == a_black
        if not bg.has_moves(cells, turn):
            passes += 1
            if passes >= 2:
                break
            turn = 1 - turn
            ply += 1
            continue
        passes = 0
        eng = ea if a_move else eb
        fen = bg.to_fen(cells, turn)
        tok = eng.go(fen, nodes, movetime)
        if tok is None:
            forfeit = 'a' if a_move else 'b'
            moves_log.append('%s@%s=None' % ('A' if a_move else 'B', fen))
            break
        moves_log.append('%s@%s=%s' % ('A' if a_move else 'B', fen, tok))
        st = bg.apply(cells, turn, tok)
        if st == 'illegal':
            forfeit = 'a' if a_move else 'b'
            break
        turn = 1 - turn
        ply += 1
    ea.quit()
    eb.quit()
    b, w = bg.counts(cells)
    if forfeit == 'a':
        winner = 'b'
    elif forfeit == 'b':
        winner = 'a'
    elif b == w:
        winner = 'draw'
    else:
        winner = 'a' if ((b > w) == a_black) else 'b'
    return {'tag': tag, 'a_color': 'black' if a_black else 'white',
            'winner': winner, 'black': b, 'white': w, 'plies': ply,
            'forfeit': forfeit, 'moves_log': moves_log, 'start': line}


def sha(p):
    try:
        return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
    except OSError:
        return 'missing'


WLOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', required=True)
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--nodes', type=int, default=5000)
    ap.add_argument('--movetime', type=int, default=0)
    ap.add_argument('--a-name', required=True)
    ap.add_argument('--a-exe', required=True)
    ap.add_argument('--a-args', default='')
    ap.add_argument('--a-dialect', default='uai', choices=['uai', 'uci'])
    ap.add_argument('--a-setup', default=';')
    ap.add_argument('--b-name', required=True)
    ap.add_argument('--b-exe', required=True)
    ap.add_argument('--b-args', default='')
    ap.add_argument('--b-dialect', default='uai', choices=['uai', 'uci'])
    ap.add_argument('--b-setup', default=';')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--take', type=int, default=0)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.book) if l.strip()]
    if a.take:
        lines = lines[a.skip:a.skip + a.take]
    rng = random.Random(a.seed)
    order = list(range(len(lines)))
    rng.shuffle(order)
    jobs = []
    for i in order:
        jobs.append((lines[i], True, 'pos%d-black' % i))
        jobs.append((lines[i], False, 'pos%d-white' % i))
    jobs = jobs[:a.games]
    A = {'exe': a.a_exe, 'args': a.a_args.split(), 'dialect': a.a_dialect,
         'setup': [s for s in a.a_setup.split(';') if s]}
    B = {'exe': a.b_exe, 'args': a.b_args.split(), 'dialect': a.b_dialect,
         'setup': [s for s in a.b_setup.split(';') if s]}
    with open(a.out, 'w') as f:
        f.write('place-bench %d jobs %s\n' % (
            len(jobs), ('movetime=%dms' % a.movetime) if a.movetime else ('nodes=%d' % a.nodes)))
        f.write('A %s: exe=%s sha=%s args=%s\n' % (a.a_name, a.a_exe, sha(a.a_exe), a.a_args))
        f.write('B %s: exe=%s sha=%s args=%s\n' % (a.b_name, a.b_exe, sha(a.b_exe), a.b_args))
    print('place-bench %d jobs: %s vs %s' % (len(jobs), a.a_name, a.b_name), flush=True)

    def run(job):
        line, ab, tag = job
        r = play_one(A, B, line, ab, tag, a.nodes, a.movetime)
        with WLOCK:
            with open(a.out, 'a') as f:
                f.write('game %s [A=%s]: %s b%d-w%d plies=%d forfeit=%s\n' % (
                    r['tag'], r['a_color'], r['winner'], r['black'], r['white'],
                    r['plies'], r['forfeit']))
                if r['forfeit']:
                    f.write('  start %s\n' % r['start'])
                    for ml in r['moves_log'][-6:]:
                        f.write('  %s\n' % ml)
        print('game %s [A=%s]: %s (%d-%d)' % (
            r['tag'], r['a_color'], r['winner'], r['black'], r['white']), flush=True)
        return r

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run, jobs):
            results.append(r)
    w = sum(1 for r in results if r['winner'] == 'a')
    l = sum(1 for r in results if r['winner'] == 'b')
    d = sum(1 for r in results if r['winner'] == 'draw')
    fa = sum(1 for r in results if r.get('forfeit') == 'a')
    fb = sum(1 for r in results if r.get('forfeit') == 'b')
    n = w + l + d
    s = (w + d / 2) / n if n else 0.0
    se = math.sqrt(max(s * (1 - s), 0) / n) if n else 0.0
    elo = -400 * math.log10(1 / max(s, 1e-9) - 1) if 0 < s < 1 else (999 if s == 1 else -999)
    # Miss risk: P(true score > 0.5 | data) — the hidden-gem probability if
    # this result were used to reject. Logged on every screen per user law.
    miss = 1 - 0.5 * (1 + math.erf((0.5 - s) / (max(se, 1e-9) * math.sqrt(2)))) if n else 0.5
    summary = ('total: %s %d - %s %d draws %d (n=%d) forfeits A=%d B=%d score %.3f CI [%.3f,%.3f] Elo %+.0f miss %.3f\n'
               % (a.a_name, w, a.b_name, l, d, n, fa, fb, s, s - 1.96 * se, s + 1.96 * se, elo, miss))
    with open(a.out, 'a') as f:
        f.write(summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
