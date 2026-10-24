"""Minimal gate summary (gate_check.py is missing on this branch).

Parses `titanium-cli match` logs:
  -- game N: book B titanium=x opp=o: titanium wins 28-21 (113 plies)
Counts W/L/D, seat balance (titanium as x vs o), score, Elo, 95% CI.
GATE LAW v5 (user order 2026-10-22): PASS when the 95% CI lower bound on
score clears 0.5 (objectively better, no fixed Elo magnitude). LLR is
printed as data only, never as a pass threshold.

Usage: python training/gate_summary.py logs/gate_A.txt [logs/gate_B.txt ...]
Read-only. Never upgrades a verdict.
"""
import math
import sys


def elo_score(elo):
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def score_elo(s):
    if s <= 0.0:
        return -math.inf
    if s >= 1.0:
        return math.inf
    return 400.0 * math.log10(s / (1.0 - s))


def parse(path):
    W = L = D = 0
    seat_x = seat_o = 0
    games = 0
    for line in open(path, errors='ignore'):
        if not line.startswith('-- game'):
            continue
        games += 1
        if 'titanium=x' in line:
            seat_x += 1
        elif 'titanium=o' in line:
            seat_o += 1
        if 'titanium wins' in line:
            W += 1
        elif 'opponent wins' in line:
            L += 1
        else:
            D += 1
    return W, L, D, seat_x, seat_o, games


def summarize(path):
    W, L, D, sx, so, g = parse(path)
    N = W + L + D
    print('=' * 72)
    print('gate:', path)
    print('games %d  W-L-D %d-%d-%d  seats titanium x=%d o=%d' % (N, W, L, D, sx, so))
    if N == 0:
        print('NO GAMES PARSED')
        return
    s = (W + 0.5 * D) / N
    elo = score_elo(s)
    # normal-approx CI on score, converted to Elo bounds
    var = max((s - s * s) / N, 1e-12)
    se = math.sqrt(var)
    lo = max(min(s - 1.96 * se, 0.999999), 1e-6)
    hi = max(min(s + 1.96 * se, 0.999999), 1e-6)
    lo_elo, hi_elo = score_elo(lo), score_elo(hi)
    print('score %.4f  elo %+.0f  CI [%+.0f, %+.0f]  scoreCI [%.4f, %.4f]'
          % (s, elo, lo_elo, hi_elo, lo, hi))
    # GATE LAW v5: objectively better = 95% CI lower bound clears 0.5.
    if lo > 0.5:
        print('VERDICT: PASS (better than main, 95%% CI lower bound %.4f > 0.5)' % lo)
    elif hi < 0.5:
        print('VERDICT: CONCLUSIVE BELOW (95%% CI upper bound %.4f < 0.5)' % hi)
    else:
        print('VERDICT: INCONCLUSIVE -> EXTEND (CI straddles 0.5; never merge/reject on the bound)')
    if sx != so:
        print('SEAT IMBALANCE: x=%d o=%d (law 11 requires exactly 50/50)' % (sx, so))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for p in sys.argv[1:]:
        summarize(p)
