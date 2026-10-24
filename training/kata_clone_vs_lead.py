"""User hypothesis: when ahead, cloning is good (it fills the board and brings the end closer); when behind, cloning is bad
(jumps keep the game long). READ-ONLY test on KataGo's 200-visit search choices.

Stage-0 rows (first step of a move): search mass on clone squares vs on jump-origin picks, binned by Kata's own value of the
position (TD h1, mover's win prob) and by empties. Also Kata's material lead as a second "ahead" measure.
Usage: python training/kata_clone_vs_lead.py [n_files]
"""
import glob
import sys

import numpy as np

KDIR = "C:/gitProjects/titanium-ataxx/data/datasets/KataGo_Ataxx_202604"


def main():
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(23)
    fs = sorted(glob.glob(KDIR + "/*/tdata/*.npz"))
    V, CM, EMP, MAT = [], [], [], []
    for f in [fs[i] for i in rng.choice(len(fs), nf, replace=False)]:
        z = np.load(f)
        pl = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
        gi = z["globalInputNC"]
        q = z["policyTargetsNCMove"][:, 0, :].astype(np.float64)
        g = z["globalTargetsNC"]
        ok = (gi[:, 0] == 0) & (q.sum(1) > 0) & (pl[:, 3].sum(1) == 0)
        q = q[ok] / q[ok].sum(1, keepdims=True)
        own, opp = pl[ok, 1].astype(bool), pl[ok, 2].astype(bool)
        empty = ~(own | opp)
        CM.append((q[:, :49] * empty).sum(1))                  # mass on empty squares = clones
        V.append(g[ok, 4] + 0.5 * g[ok, 6])
        EMP.append(empty.sum(1))
        MAT.append(own.sum(1).astype(int) - opp.sum(1).astype(int))
    V, CM, EMP, MAT = (np.concatenate(x) for x in (V, CM, EMP, MAT))
    print("stage-0 decisions: %d | overall search mass on clones %.3f" % (len(V), CM.mean()))
    bins = [(0, .1), (.1, .3), (.3, .45), (.45, .55), (.55, .7), (.7, .9), (.9, 1.01)]
    print("\nsearch mass on CLONES by Kata's value of the position (mover's win prob), per empties band:")
    hdr = "win prob    " + "".join("%11s" % ("%d-%d" % b) for b in ((5, 12), (13, 20), (21, 28), (29, 36), (37, 45)))
    print(hdr)
    for lo, hi in bins:
        line = "%.2f-%.2f  " % (lo, min(hi, 1))
        for elo, ehi in ((5, 12), (13, 20), (21, 28), (29, 36), (37, 45)):
            m = (V >= lo) & (V < hi) & (EMP >= elo) & (EMP <= ehi)
            line += "%11s" % ("%.3f(%d)" % (CM[m].mean(), m.sum()) if m.sum() >= 200 else "-")
        print(line)
    print("\nsearch mass on CLONES by material lead (mover stones - enemy stones), empties 13-36:")
    for lo, hi in ((-49, -10), (-9, -5), (-4, -2), (-1, 1), (2, 4), (5, 9), (10, 49)):
        m = (MAT >= lo) & (MAT <= hi) & (EMP >= 13) & (EMP <= 36)
        if m.sum() >= 200:
            print("   lead %+3d..%+3d  clone mass %.3f  (n=%d, mean Kata win prob %.2f)" % (lo, hi, CM[m].mean(), m.sum(), V[m].mean()))
    # partial effect: regress clone mass on value with empties held fixed (empties one-hot)
    X = np.column_stack([V - 0.5] + [(EMP == e).astype(float) for e in range(5, 46)])
    w = np.linalg.lstsq(X, CM, rcond=None)[0]
    print("\nlinear: clone mass change per +0.1 win prob, empties held fixed: %+.4f" % (0.1 * w[0]))


if __name__ == "__main__":
    main()
