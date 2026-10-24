"""Ataxx_CPP policy head (one forward pass, top-prior legal move; engine-ctl 'policy') on held-out positions:
is its move a clone or a jump, the best 2-ply move (gain - enemy best recapture, jump vacates its origin),
the max-capture move? Compared with a random legal move.
Usage: python training/probe_cpp_policy.py [n]
"""
import sys

import numpy as np

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import ax_vs_ti as A  # noqa: E402

FULL = (1 << 49) - 1
R1 = [0] * 49
R2 = [0] * 49
for s in range(49):
    r, c = divmod(s, 7)
    for t in range(49):
        rr, cc = divmod(t, 7)
        d = max(abs(rr - r), abs(cc - c))
        if d == 1:
            R1[s] |= 1 << t
        elif d == 2:
            R2[s] |= 1 << t


def best_capture(att, vic, empty):
    land = 0
    x = att
    while x:
        b = x & -x
        k = b.bit_length() - 1
        land |= R1[k] | R2[k]
        x ^= b
    land &= empty
    best = 0
    while land:
        b = land & -land
        best = max(best, bin(vic & R1[b.bit_length() - 1]).count("1"))
        land ^= b
    return best


def moves(us, en):
    """all distinct legal moves as (from, to, is_clone); clones once per destination."""
    empty = ~(us | en) & FULL
    out = []
    for t in range(49):
        if not (empty >> t) & 1:
            continue
        if us & R1[t]:
            out.append((-1, t, True))
        src = us & R2[t]
        while src:
            b = src & -src
            out.append((b.bit_length() - 1, t, False))
            src ^= b
    return out


def score(us, en, mv):
    f, t, clone = mv
    empty = ~(us | en) & FULL
    conv = en & R1[t]
    u2 = (us | (1 << t) | conv) & ~((1 << f) if f >= 0 else 0)
    e2 = en & ~conv
    emp2 = (empty & ~(1 << t)) | ((1 << f) if f >= 0 else 0)
    g = bin(conv).count("1") + (1 if clone else 0)
    return g, g - best_capture(e2, u2, emp2)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    z = np.load("C:/gitProjects/titanium-ataxx/data/nnue/final/final_val_v3.npz")
    o0, o1, st = z["occ0"], z["occ1"], z["stm"]
    rng = np.random.default_rng(4)
    idx = rng.choice(len(st), n, replace=False)
    cpp = A.CppSide()
    stats = dict(n=0, clone=0, best2=0, bestcap=0, rand2=0.0, randcap=0.0, net_pick=[], net_best=[])
    for j in idx:
        a, b, s = int(o0[j]), int(o1[j]), int(st[j])
        us, en = (a, b) if s == 0 else (b, a)
        ms = moves(us, en)
        if len(ms) < 2:
            continue
        cpp.set_state((a, b, 0, s))
        tok = cpp.go_cmd("policy")
        x0, y0, x1, y1 = "abcdefg".index(tok[0]), int(tok[1]) - 1, "abcdefg".index(tok[2]), int(tok[3]) - 1
        fr, to = (6 - y0) * 7 + x0, (6 - y1) * 7 + x1
        dist = max(abs(fr // 7 - to // 7), abs(fr % 7 - to % 7))
        clone = dist <= 1
        sc = [score(us, en, m) for m in ms]
        pick = score(us, en, (-1 if clone else fr, to, clone))
        nets = [x[1] for x in sc]
        caps = [x[0] for x in sc]
        stats["n"] += 1
        stats["clone"] += clone
        stats["best2"] += pick[1] == max(nets)
        stats["bestcap"] += pick[0] == max(caps)
        stats["rand2"] += sum(1 for v in nets if v == max(nets)) / len(nets)
        stats["randcap"] += sum(1 for v in caps if v == max(caps)) / len(caps)
        stats["net_pick"].append(pick[1])
        stats["net_best"].append(max(nets))
    k = stats["n"]
    print("positions %d | Ataxx_CPP policy plays a clone in %.1f%%" % (k, 100 * stats["clone"] / k))
    print("policy move is a best-2-ply move %.1f%% (random legal move %.1f%%)" % (100 * stats["best2"] / k, 100 * stats["rand2"] / k))
    print("policy move is a max-capture move %.1f%% (random %.1f%%)" % (100 * stats["bestcap"] / k, 100 * stats["randcap"] / k))
    d = np.array(stats["net_best"]) - np.array(stats["net_pick"])
    print("2-ply net: chosen %.2f vs best available %.2f; shortfall 0: %.1f%%, 1: %.1f%%, >=2: %.1f%%"
          % (np.mean(stats["net_pick"]), np.mean(stats["net_best"]), 100 * (d == 0).mean(), 100 * (d == 1).mean(), 100 * (d >= 2).mean()))


if __name__ == "__main__":
    main()
