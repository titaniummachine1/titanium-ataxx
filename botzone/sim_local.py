"""Local Botzone simulator for titanium_botzone.py (Simple IO, long-live, Ataxx).

Referee = the judge's rules: no passes; the game ends the moment EITHER side cannot move, the mobile side
absorbs the empty squares (count = 49 - stuck); 4-corner start (black (0,0),(6,6); white (6,0),(0,6)).
Opponent = greedy (best immediate disc difference, random ties). Measures the driver's wall time per
move from request sent to marker received against the real limits (12 s first move, 4 s later).

  python botzone/sim_local.py --lib <dll/so> --net <tup> --games 2 --scale 0.05
  python botzone/sim_local.py --lib <dll/so> --net <tup> --games 1 --scale 1 --max-driver-moves 3
"""
import argparse
import os
import random
import subprocess
import sys
import time

FULL = (1 << 49) - 1


def bit(x, y):
    return 1 << (y * 7 + x)


def ring(x, y, r):
    m = 0
    for yy in range(max(0, y - r), min(7, y + r + 1)):
        for xx in range(max(0, x - r), min(7, x + r + 1)):
            if (xx, yy) != (x, y):
                m |= bit(xx, yy)
    return m


R1 = [[ring(x, y, 1) for x in range(7)] for y in range(7)]
R2 = [[ring(x, y, 2) for x in range(7)] for y in range(7)]


def bits(v):
    while v:
        b = v & -v
        yield b.bit_length() - 1
        v ^= b


def legal(own, opp):
    occ = own | opp
    empty = ~occ & FULL
    res = []
    for s in bits(own):
        x, y = s % 7, s // 7
        for t in bits(R2[y][x] & empty):
            res.append((x, y, t % 7, t // 7))
    return res


def can_move(own, opp):
    occ = own | opp
    empty = ~occ & FULL
    for s in bits(own):
        if R2[s // 7][s % 7] & empty:
            return True
    return False


def apply(own, opp, m):
    x0, y0, x1, y1 = m
    d = max(abs(x1 - x0), abs(y1 - y0))
    if d == 2:
        own &= ~bit(x0, y0)
    own |= bit(x1, y1)
    flip = R1[y1][x1] & opp
    return own | flip, opp & ~flip


class Game:
    def __init__(self):
        self.occ = [bit(0, 0) | bit(6, 6), bit(6, 0) | bit(0, 6)]
        self.turn = 0

    def moves(self):
        return legal(self.occ[self.turn], self.occ[1 - self.turn])

    def play(self, m):
        if tuple(m) not in set(self.moves()):
            raise ValueError("illegal move %r for side %d" % (m, self.turn))
        o, p = apply(self.occ[self.turn], self.occ[1 - self.turn], m)
        self.occ[self.turn], self.occ[1 - self.turn] = o, p
        self.turn = 1 - self.turn

    def over(self):
        a, b = self.occ
        return a == 0 or b == 0 or (a | b) == FULL or not can_move(a, b) or not can_move(b, a)

    def score(self):
        a, b = self.occ
        ca, cb = bin(a).count("1"), bin(b).count("1")
        ma, mb = can_move(a, b), can_move(b, a)
        if ma and not mb:
            ca = 49 - cb
        elif mb and not ma:
            cb = 49 - ca
        return ca, cb


def greedy(g):
    ms = g.moves()
    best, bm = -99, []
    me = g.turn
    for m in ms:
        o, p = apply(g.occ[me], g.occ[1 - me], m)
        v = bin(o).count("1") - bin(p).count("1")
        if v > best:
            best, bm = v, [m]
        elif v == best:
            bm.append(m)
    return random.choice(bm)


def read_response(proc):
    lines = []
    while True:
        l = proc.stdout.readline()
        if not l:
            raise RuntimeError("driver closed; stderr: " + proc.stderr.read()[-500:])
        l = l.rstrip("\n")
        if l.strip() == ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<":
            return lines
        lines.append(l)


def run_game(args, driver_black):
    env = dict(os.environ, TI_DEBUG="1", TI_LIB=args.lib, TI_NET=args.net, TI_TIME_SCALE=str(args.scale), TI_TT_BITS=str(args.tt_bits))
    proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanium_botzone.py")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    g = Game()
    drv_side = 0 if driver_black else 1
    margins, times, first = [], [], True
    req = (-1, -1, -1, -1)
    dmoves = 0
    try:
        while not g.over():
            if g.turn == drv_side:
                limit = (12000 if first else 4000) * args.scale
                t = time.perf_counter()
                msg = ("1\n%d %d %d %d\n" % req) if first else ("%d %d %d %d\n" % req)
                proc.stdin.write(msg)
                proc.stdin.flush()
                resp = read_response(proc)
                el = (time.perf_counter() - t) * 1000
                mv = tuple(int(v) for v in resp[0].split())
                g.play(mv)                       # raises if the driver sent an illegal move
                margins.append(limit - el)
                times.append(el)
                first = False
                dmoves += 1
                if args.max_driver_moves and dmoves >= args.max_driver_moves:
                    break
            else:
                m = greedy(g)
                req = m
                g.play(m)
        res = g.score()
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    errs = proc.stderr.read().splitlines()
    over = [float(l.split("over=")[1].split()[0]) for l in errs if "over=" in l]
    wd = sum(int(l.split("watchdog=")[1]) for l in errs if "watchdog=" in l)
    return drv_side, res, margins, times, dmoves, g.over(), over, wd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--tt-bits", type=int, default=22)
    ap.add_argument("--max-driver-moves", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    all_margins = []
    all_over = []
    all_wd = 0
    for i in range(args.games):
        black = (i % 2 == 0)
        side, res, margins, times, n, over, ov, wd = run_game(args, black)
        all_margins += margins
        all_over.extend(ov)
        all_wd += wd
        ours, theirs = (res[0], res[1]) if side == 0 else (res[1], res[0])
        result = "game over" if over else "stopped early"
        print("game %d: driver=%s %s, final %d-%d (%s), driver moves %d, move time ms: median %.0f max %.0f, min margin to limit %.1f ms"
              % (i + 1, "black" if black else "white", result, ours, theirs,
                 "WIN" if ours > theirs else "LOSS" if ours < theirs else "DRAW", n,
                 sorted(times)[len(times) // 2], max(times), min(margins)))
    if all_over:
        s = sorted(all_over)
        print("engine search overshoot vs its own budget (ms): median %.1f p90 %.1f max %.1f over %d moves; watchdog fired %d times"
              % (s[len(s) // 2], s[int(len(s) * 0.9)], s[-1], len(s), all_wd))
    print("worst margin over all moves: %.1f ms (positive = finished before the limit; scale %.3f)" % (min(all_margins), args.scale))


if __name__ == "__main__":
    main()
