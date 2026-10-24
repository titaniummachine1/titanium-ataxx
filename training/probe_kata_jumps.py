"""Where does KataGo's 200-visit search JUMP to? Stage-1 rows of the raw self-play files (global input 0 == 1):
plane 4 marks the stone already picked, policyTargetsNCMove[:,0] is the search distribution over destinations
(empty squares at Chebyshev distance 2 from the stone).

Per destination: what the jump converts, the 2-ply net (gain - enemy best capture after the jump, which also
VACATES the origin), whether the vacated origin leaves my stones exposed, whether it is the overall best 2-ply move
square, the enemy's best capture square (block), or the max-capture square.
Usage: python training/probe_kata_jumps.py [n_files]
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P  # noqa: E402
from probe_kata_policy import bits_to_u64  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
pc = np.bitwise_count


def best_capture(att, vic, empty):
    """per row: max number of `vic` stones converted by one landing of `att`."""
    land = (P.spread(att, P.R1) | P.spread(att, P.R2)) & empty
    best = np.zeros(len(att), np.int64)
    for t in range(49):
        hit = ((land >> np.uint64(t)) & np.uint64(1)) == 1
        best = np.maximum(best, np.where(hit, pc(vic & P.R1[t]).astype(np.int64), 0))
    return best


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    fs = sorted(glob.glob(ROOT + "/data/datasets/KataGo_Ataxx_202604/*/tdata/*.npz"))
    rng = np.random.default_rng(0)
    U, E, O, POL = [], [], [], []
    for i in rng.choice(len(fs), n_files, replace=False):
        try:
            z = np.load(fs[i])
            G = z["globalInputNC"]
            bits = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
            pol = z["policyTargetsNCMove"][:, 0, :].astype(np.float64)
        except Exception:
            continue
        ok = np.nonzero((G[:, 0] == 1) & (pol[:, :49].sum(1) > 0) & (bits[:, 4].sum(1) == 1) & (bits[:, 3].sum(1) == 0))[0]
        U.append(bits_to_u64(bits[ok, 1])); E.append(bits_to_u64(bits[ok, 2]))
        O.append(bits[ok, 4].argmax(1)); POL.append(pol[ok, :49])
    us, en, org, pol = np.concatenate(U), np.concatenate(E), np.concatenate(O), np.concatenate(POL)
    n = len(us)
    pol = pol / pol.sum(1, keepdims=True)
    empty = ~(us | en) & P.FULL
    print("stage-1 jump decisions: %d (from %d raw files)" % (n, n_files))
    # the picked stone: is it in the 'us' plane? (plane 1 = side to move stones; the picked stone should be mine)
    mine = ((us >> org.astype(np.uint64)) & np.uint64(1)) == 1
    print("picked stone is mine: %.3f" % mine.mean())
    keep = mine
    us, en, org, pol, empty = us[keep], en[keep], org[keep], pol[keep], empty[keep]
    n = len(us)
    R2 = np.array([int(r) for r in P.R2], np.uint64)
    feats = {k: np.full((n, 49), np.nan) for k in ("gain", "net2", "hole_exposed", "en_cap_after")}
    en_best_now = best_capture(en, us, empty)
    for d in range(49):
        rows = np.nonzero(((R2[org] >> np.uint64(d)) & np.uint64(1)) & ((empty >> np.uint64(d)) & np.uint64(1)))[0]
        if not len(rows):
            continue
        conv = en[rows] & P.R1[d]
        u2 = (us[rows] & ~(np.uint64(1) << org[rows].astype(np.uint64))) | (np.uint64(1) << np.uint64(d)) | conv
        e2 = en[rows] & ~conv
        emp2 = (empty[rows] & ~(np.uint64(1) << np.uint64(d))) | (np.uint64(1) << org[rows].astype(np.uint64))
        rec = best_capture(e2, u2, emp2)
        feats["gain"][rows, d] = pc(conv)
        feats["net2"][rows, d] = pc(conv).astype(np.int64) - rec
        feats["en_cap_after"][rows, d] = rec
        # the vacated origin: how many of my stones sit next to it AND can the enemy land there
        ring_o = np.array([int(P.R1[o]) for o in org[rows]], np.uint64)
        en_reach_o = ((P.spread(e2, P.R1) | P.spread(e2, P.R2)) >> org[rows].astype(np.uint64)) & np.uint64(1)
        feats["hole_exposed"][rows, d] = np.where(en_reach_o == 1, pc(u2 & ring_o), 0)
    valid = ~np.isnan(feats["gain"])
    uni = valid / valid.sum(1, keepdims=True)

    def table(name, key, lo, hi):
        print("\njump destinations by %s: mean search mass per destination vs uniform" % name)
        for v in range(lo, hi + 1):
            m = valid & (np.clip(np.nan_to_num(key, nan=-99), lo, hi) == v)
            if m.sum() < 300:
                continue
            print("  %+3d: ratio %.2f  (mass %.4f, n=%d)" % (v, pol[m].mean() / uni[m].mean(), pol[m].mean(), m.sum()))
    table("stones converted", feats["gain"], 0, 7)
    table("2-ply net (gain - enemy best recapture after the jump)", feats["net2"], -6, 5)
    table("my stones next to the vacated origin that the enemy can reach", feats["hole_exposed"], 0, 6)
    top = pol.argmax(1)
    ar = np.arange(n)
    bestnet = np.where(valid, np.nan_to_num(feats["net2"], nan=-99), -99).argmax(1)
    bestgain = np.where(valid, np.nan_to_num(feats["gain"], nan=-99), -99).argmax(1)
    print("\njump top destination = best 2-ply destination %.1f%% | = max-capture destination %.1f%% | random baseline %.1f%%"
          % (100 * (top == bestnet).mean(), 100 * (top == bestgain).mean(), 100 * (1 / valid.sum(1)).mean()))
    print("enemy's best capture BEFORE the jump vs after the chosen jump: mean %.2f -> %.2f"
          % (en_best_now.mean(), feats["en_cap_after"][ar, top].mean()))


if __name__ == "__main__":
    main()
