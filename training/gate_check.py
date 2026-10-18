#!/usr/bin/env python3
"""Fishtest door: recompute SPRT for candidate-vs-baseline from results.db.
Exit 0 + PASS only if pooled LLR >= +2.94 (H0: elo<=0 vs H1: elo>=20).
FAIL (LLR <= -2.94) or UNDECIDED otherwise. Merge scripts must call this
and abort unless it prints PASS - or log a verbatim user override in LEDGER.
Usage: python training/gate_check.py --cand <name-substring> [--elo1 20]
Matches DB rows whose log path contains <cand> (e.g. a net or branch tag).
"""
import argparse
import hashlib
import math
import os
import re
import sqlite3
import sys

DB = 'data/nnue/results.db'
SHA_RE = re.compile(r'^titan: exe=\S+ sha=(\S+) tup=(\S+) sha=(\S+)(?: corr=(\w+))?')


def llr(w, l, d, elo0=0.0, elo1=20.0):
    n = w + l + d
    if n == 0:
        return 0.0
    delta = min(0.98, max(0.001, d / n))
    s0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    s1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
    W0 = max(1e-9, s0 - delta / 2)
    L0 = max(1e-9, 1.0 - s0 - delta / 2)
    W1 = max(1e-9, s1 - delta / 2)
    L1 = max(1e-9, 1.0 - s1 - delta / 2)
    return w * math.log(W1 / W0) + l * math.log(L1 / L0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True)
    ap.add_argument('--elo1', type=float, default=20.0)
    ap.add_argument('--expect', default='',
                    help='optional: artifact file that MUST match the gated sha '
                    '(chain of custody: tested == merged)')
    a = ap.parse_args()
    db = sqlite3.connect(DB)
    rows = db.execute(
        'SELECT games, w, l, d, log FROM gates WHERE log LIKE ?', ('%' + a.cand + '%',)).fetchall()
    W = L = D = G = 0
    arts = set()
    for games, w, l, d, log in rows:
        W += w
        L += l
        D += d
        G += games
        try:
            txt = open(log, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        m = SHA_RE.search(txt, re.M)
        if m:
            arts.add((m.group(2), m.group(3), m.group(4) or 'nocorr'))
    if len(arts) > 1:
        print('cand=%s MIXED ARTIFACTS across %d logs %s -> FAIL '
              '(tested one thing, merging another)' % (a.cand, len(rows), arts))
        sys.exit(1)
    if a.expect:
        try:
            import hashlib as _hl
            cur = _hl.sha256(open(a.expect, 'rb').read()).hexdigest()[:16]
        except OSError:
            cur = 'missing'
        exp = list(arts)[0][1] if arts else None
        if cur != exp:
            print('cand=%s ARTIFACT CHANGED since gating (gated=%s disk=%s) -> FAIL' % (
                a.cand, exp, cur))
            sys.exit(1)
    v = llr(W, L, D, 0.0, a.elo1)
    import math as _m
    def _elo(x):
        x = min(0.999, max(0.001, x))
        return 400.0 * _m.log10(x / (1.0 - x))
    s = (W + D / 2) / max(1, W + L + D)
    n = W + L + D
    se = (s * (1 - s) / max(1, n)) ** 0.5
    elo, elo_lo, elo_hi = _elo(s), _elo(max(0.001, s - 1.96 * se)), _elo(min(0.999, s + 1.96 * se))
    if v >= 2.94:
        verdict = 'PASS (proven better: merge)'
    elif v <= -2.94:
        verdict = 'FAIL (proven worse: reject)'
    elif n < 200:
        verdict = 'TOO-SHORT (under 200 games proves nothing either way: extend)'
    elif W > L:
        verdict = 'MERGE-DEFAULT (ahead %d-%d over %d: merge on user go unless later games prove worse)' % (W, L, n)
    elif W < L:
        verdict = 'EXTEND-OR-REJECT (behind %d-%d over %d: extend to prove better, else reject)' % (W, L, n)
    else:
        verdict = 'EXACT-TIE (user decides; equal merges need explicit approval + not-worse evidence)'
    print('cand=%s games=%d pooled=%d-%d-%d score=%.3f Elo %+.0f [95%% CI %+.0f..%+.0f] LLR=%+.2f -> %s (n=%d logs)' % (
        a.cand, G, W, L, D, s, elo, elo_lo, elo_hi, v, verdict, len(rows)))
    sys.exit(0 if verdict.startswith('PASS') else 1)


if __name__ == '__main__':
    main()
