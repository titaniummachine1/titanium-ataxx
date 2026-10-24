"""Where does KataGo's 200-visit search want to play? (raw self-play files, no net evaluation)

KataGo Ataxx splits a move in two stages (cpp/game/gamelogic.cpp, branch Ataxx-onnx): in stage 0 the policy over 49
squares means: EMPTY square = clone there (complete move), OWN stone = pick it for a jump (stage 1 then picks the
destination). We read stage-0 rows only (global input 0 == 0) and the search visit distribution policyTargetsNCMove[:,0].

Measured per position (side to move = "me"):
  * share of search mass on clones vs jumps
  * for clone squares: mass vs what the clone converts, vs the 2-ply net (gain - enemy's best recapture), vs whether
    the square blocks the enemy's best capture square, vs how exposed my stones are after it
  * ANCHOR TESTS: how often Kata's top move / how much of its mass lands on each candidate anchor square:
      A  my max-capture landing square (clone first)          [current nnfeat G2]
      B  my best 2-ply clone square
      C  the enemy's max-capture landing square (my worst hole)  -> blocking
      D  a random empty clone square (baseline)
Usage: python training/probe_kata_policy.py [n_files]
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
pc = np.bitwise_count


def bits_to_u64(b):
    return (b.astype(np.uint64) << np.arange(49, dtype=np.uint64)[None, :]).sum(1).astype(np.uint64) if False else \
        np.bitwise_or.reduce(b.astype(np.uint64) << np.arange(49, dtype=np.uint64)[None, :], axis=1)


def load(n_files, per_file):
    fs = sorted(glob.glob(ROOT + "/data/datasets/KataGo_Ataxx_202604/*/tdata/*.npz"))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(fs), n_files, replace=False)
    U, E, POL = [], [], []
    for i in pick:
        try:
            z = np.load(fs[i])
            G = z["globalInputNC"]
            bits = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
            pol = z["policyTargetsNCMove"][:, 0, :].astype(np.float64)
        except Exception:
            continue
        ok = (G[:, 0] == 0) & (pol.sum(1) > 0) & (bits[:, 3].sum(1) == 0)   # stage 0, has visits, no banned squares
        idx = np.nonzero(ok)[0]
        idx = rng.choice(idx, min(per_file, len(idx)), replace=False)
        U.append(bits_to_u64(bits[idx, 1]))
        E.append(bits_to_u64(bits[idx, 2]))
        POL.append(pol[idx])
    return np.concatenate(U), np.concatenate(E), np.concatenate(POL)


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    us, en, pol = load(n_files, 1500)
    n = len(us)
    pol = pol / pol.sum(1, keepdims=True)
    sq = pol[:, :49]
    occ = us | en
    empty = ~occ & P.FULL
    clone_u = P.spread(us, P.R1) & empty
    land_u = clone_u | (P.spread(us, P.R2) & empty)
    clone_e = P.spread(en, P.R1) & empty
    land_e = clone_e | (P.spread(en, P.R2) & empty)
    B = lambda x: ((x[:, None] >> np.arange(49, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
    Eb, Ub, CU, LE = B(empty), B(us), B(clone_u), B(land_e)
    gain_me = np.stack([pc(en & P.R1[s]) for s in range(49)], 1).astype(np.int64)   # I convert landing on s
    gain_en = np.stack([pc(us & P.R1[s]) for s in range(49)], 1).astype(np.int64)   # enemy converts landing on s
    # 2-ply for clone squares: gain - enemy best capture after my clone
    net2 = np.full((n, 49), -99, np.int64)
    for s in range(49):
        rows = np.nonzero(CU[:, s])[0]
        if not len(rows):
            continue
        conv = en[rows] & P.R1[s]
        u2 = us[rows] | (np.uint64(1) << np.uint64(s)) | conv
        e2 = en[rows] & ~conv
        emp2 = empty[rows] & ~(np.uint64(1) << np.uint64(s))
        le2 = (P.spread(e2, P.R1) | P.spread(e2, P.R2)) & emp2
        best = np.zeros(len(rows), np.int64)
        for t in range(49):
            hit = ((le2 >> np.uint64(t)) & np.uint64(1)) == 1
            best = np.maximum(best, np.where(hit, pc(u2 & P.R1[t]).astype(np.int64), 0))
        net2[rows, s] = 1 + pc(conv).astype(np.int64) - best
    print("positions %d (stage 0, from %d raw files)" % (n, n_files))
    clone_mass = (sq * Eb).sum(1)
    print("search mass on clone moves %.3f, on jumps %.3f, pass %.4f"
          % (clone_mass.mean(), (sq * Ub).sum(1).mean(), pol[:, 49].mean()))
    top = sq.argmax(1)
    top_is_clone = Eb[np.arange(n), top]
    print("top move is a clone in %.1f%% of positions" % (100 * top_is_clone.mean()))

    # --- what attracts mass among clone squares: mean policy mass per clone square, by property
    def table(name, key, lo, hi):
        print("\nclone squares by %s: mean search mass per square (vs uniform over clone squares)" % name)
        ncl = CU.sum(1, keepdims=True).clip(1)
        uni = (CU / ncl)
        for v in range(lo, hi + 1):
            m = CU & (key == v)
            if m.sum() < 500:
                continue
            print("  %+3d: mass %.4f  uniform %.4f  ratio %.2f  (n=%d)" % (v, sq[m].mean(), uni[m].mean(), sq[m].mean() / uni[m].mean(), m.sum()))
    table("stones converted", gain_me, 0, 8)
    table("2-ply net (gain - enemy best recapture)", np.clip(net2, -4, 6), -4, 6)
    table("my stones next to it (what the enemy could convert there)", gain_en, 0, 8)

    # --- anchor tests
    def argmax_key(mask, key, tie):
        k = np.where(mask, key * 4 + tie, -10 ** 6)
        a = k.argmax(1)
        has = mask[np.arange(n), a] & (np.take_along_axis(key, a[:, None], 1)[:, 0] > 0)
        return a, has
    a_A, h_A = argmax_key(land_u.__class__ and B(land_u), gain_me, CU.astype(np.int64))
    a_B, h_B = argmax_key(CU, np.where(CU, net2 + 10, 0), np.zeros_like(gain_me))
    a_C, h_C = argmax_key(LE, gain_en, B(clone_e).astype(np.int64))
    rnd = np.random.default_rng(1)
    a_D = np.array([rnd.choice(np.nonzero(r)[0]) if r.any() else 0 for r in CU])
    h_D = CU.any(1)
    print("\nANCHOR TESTS: does Kata's search play on the candidate anchor square?")
    print("  anchor                                   defined%%  top-move hit%%  mass on it")
    for nm, a_, h_ in (("A my max-capture square (current G2)", a_A, h_A), ("B my best 2-ply clone square", a_B, h_B),
                       ("C enemy's max-capture square (block)", a_C, h_C), ("D random clone square (baseline)", a_D, h_D)):
        m = h_
        hit = (top[m] == a_[m]).mean()
        mass = sq[np.arange(n)[m], a_[m]].mean()
        print("  %-40s %6.1f   %10.1f   %10.3f" % (nm, 100 * m.mean(), 100 * hit, mass))
    # is the top move inside the 3x3 / 5x5 of each anchor? (anchor as a REGION)
    print("\n  top move within Chebyshev distance d of the anchor (region view):")
    cheb = np.array([[max(abs(a // 7 - b // 7), abs(a % 7 - b % 7)) for b in range(49)] for a in range(49)])
    for nm, a_, h_ in (("A", a_A, h_A), ("B", a_B, h_B), ("C", a_C, h_C), ("D", a_D, h_D)):
        d = cheb[a_[h_], top[h_]]
        print("   %s: d=0 %.2f  d<=1 %.2f  d<=2 %.2f" % (nm, (d == 0).mean(), (d <= 1).mean(), (d <= 2).mean()))
    both = h_A & h_C
    print("\n  positions where both sides have a capture: %.1f%%; top move = my best capture %.1f%%, = block of theirs %.1f%%, "
          "both-same-square %.1f%%" % (100 * both.mean(), 100 * (top[both] == a_A[both]).mean(),
                                       100 * (top[both] == a_C[both]).mean(), 100 * (a_A[both] == a_C[both]).mean()))


if __name__ == "__main__":
    main()
