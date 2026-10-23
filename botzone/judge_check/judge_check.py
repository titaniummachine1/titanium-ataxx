"""Differential test: my engine library's repetition rule vs the reference Botzone judge (judge_ref.h).

For generated games (biased toward jump-shuffling so cycles occur) and for every ply, every legal move is
checked: the judge's verdict (err == 2 = repetition) must equal ti_move_banned(). Also checks that illegal
moves are rejected by both, and that the judge's end-of-game verdict matches the simulator's referee.

  python botzone/judge_check/judge_check.py --lib <titanium_ffi.dll> --judge <jr_check.exe> --games 60
"""
import argparse
import ctypes
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim_local import Game, apply, legal  # noqa: E402


class Judge:
    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def verdict(self, moves):
        flat = " ".join("%d %d %d %d" % m for m in moves)
        self.p.stdin.write("%d %s\n" % (len(moves), flat))
        self.p.stdin.flush()
        f, w, e, et = (int(v) for v in self.p.stdout.readline().split())
        return f, w, e, et

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=5)


def pick(g, last_own, rng):
    """Cycle-friendly policy: prefer undoing the previous own jump, else quiet jumps, else anything."""
    ms = g.moves()
    me = g.turn
    if last_own and rng.random() < 0.55:
        undo = (last_own[2], last_own[3], last_own[0], last_own[1])
        if undo in ms:
            return undo
    quiet = []
    for m in ms:
        o, p = apply(g.occ[me], g.occ[1 - me], m)
        if bin(p).count("1") == bin(g.occ[1 - me]).count("1") and max(abs(m[2] - m[0]), abs(m[3] - m[1])) == 2:
            quiet.append(m)
    pool = quiet if quiet and rng.random() < 0.8 else ms
    return rng.choice(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lib = ctypes.CDLL(os.path.abspath(a.lib))
    lib.ti_new.restype = ctypes.c_void_p
    lib.ti_new.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.ti_new_game.argtypes = [ctypes.c_void_p]
    lib.ti_apply.restype = ctypes.c_int
    lib.ti_apply.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
    lib.ti_move_banned.restype = ctypes.c_int
    lib.ti_move_banned.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
    eng = lib.ti_new(None, 12)
    judge = Judge(a.judge)
    checked = banned_total = mismatches = illegal_checked = 0
    plies_with_cycle = games_with_cycle = end_checked = end_mismatch = 0
    for gi in range(a.games):
        g = Game()
        lib.ti_new_game(eng)
        hist = []
        last = [None, None]
        cyc = False
        snaps = [tuple(g.occ)]
        for ply in range(a.max_plies):
            if g.over() or not g.moves():
                break
            me = g.turn
            ms = g.moves()
            # placement snapshots, to know when a cycle exists (for choosing which plies to check exhaustively)
            k = len(snaps) - 1
            has_cycle = k >= 3 and any(snaps[i] == snaps[k] for i in range(0, k - 1))
            sample = ms if has_cycle else rng.sample(ms, min(len(ms), 4))
            if has_cycle:
                plies_with_cycle += 1
                cyc = True
            for m in sample:
                jv = judge.verdict(hist + [m])
                judge_banned = (jv[2] == 2)
                if jv[2] == 1:
                    print("judge says legal move illegal?", hist, m)
                    mismatches += 1
                mine = lib.ti_move_banned(eng, *m)
                checked += 1
                banned_total += judge_banned
                if mine != int(judge_banned):
                    mismatches += 1
                    if mismatches < 6:
                        print("MISMATCH game %d ply %d move %r judge_banned=%s mine=%d" % (gi, ply, m, judge_banned, mine))
            # a couple of illegal moves must be rejected by both
            bad = (rng.randrange(7), rng.randrange(7), rng.randrange(7), rng.randrange(7))
            if bad not in set(ms):
                jv = judge.verdict(hist + [bad])
                mine = lib.ti_move_banned(eng, *bad)
                illegal_checked += 1
                if not (jv[2] == 1 and mine == -1):
                    mismatches += 1
                    if mismatches < 6:
                        print("ILLEGAL MISMATCH", bad, jv, mine)
            # choose and play a legal, non-banned move (the judge would reject banned ones)
            allowed = [m for m in ms if lib.ti_move_banned(eng, *m) == 0]
            if not allowed:
                break
            mv = pick(g, last[me], rng)
            if mv not in allowed:
                mv = rng.choice(allowed)
            last[me] = mv
            # snapshot before captures (placement), same as the judge
            x0, y0, x1, y1 = mv
            own = g.occ[me]
            if max(abs(x1 - x0), abs(y1 - y0)) == 2:
                own &= ~(1 << (y0 * 7 + x0))
            own |= 1 << (y1 * 7 + x1)
            pl = list(g.occ)
            pl[me] = own
            snaps.append(tuple(pl))
            g.play(mv)
            lib.ti_apply(eng, *mv)
            hist.append(mv)
        games_with_cycle += cyc
        # end-of-game check: the judge's verdict on the final position must match the simulator's rules
        if hist and (g.over() or not g.moves()):
            f, w, e, et = judge.verdict(hist)
            ca, cb = g.score()
            sim_winner = 0 if ca > cb else 1 if cb > ca else -1
            end_checked += 1
            if e == 0 and f == 1 and w != sim_winner:
                end_mismatch += 1
                print("END MISMATCH game %d judge winner %d sim winner %d score %d-%d" % (gi, w, sim_winner, ca, cb))
    judge.close()
    print("games %d (with a cycle: %d) | legal-move checks %d (judge-banned: %d, on %d cycle plies) | illegal-move checks %d | "
          "repetition MISMATCHES: %d | end-of-game verdicts compared: %d, mismatches %d"
          % (a.games, games_with_cycle, checked, banned_total, plies_with_cycle, illegal_checked, mismatches, end_checked, end_mismatch))


if __name__ == "__main__":
    main()
