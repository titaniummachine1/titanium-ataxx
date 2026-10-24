"""Would relabelling the base data with the FIXED engine (main 653d9a1) give better labels? READ-ONLY measurement.

The datasets' titanium_cp (base-stage target for tuple nets and the NNUE) came from main's 5k search with the TUP7
evaluation bug (NET_ANALYSIS section 27). On Kata-labelled calibration rows we label a sample again with the fixed engine
(`titanium-cli label --nodes N --tuple S57`) and compare both label sets with KataGo's TD value (MSE in win prob after a
logistic scale refit, per empties band), plus how much the labels changed.
Usage: python training/label_quality_probe.py [n_rows] [nodes]
"""
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_features as SF  # noqa: E402
from ataxx_move import board_str  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
EXE = os.environ.get("LABEL_EXE", "C:/gitProjects/main-ataxx/target/release/titanium-cli.exe")
TUP = ROOT + "/data/nnue/final_v2_wdl_ep8.tup"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    fs = sorted(glob.glob(ROOT + "/data/nnue/final/moon1000/calib_c*.npz"))
    row = np.concatenate([np.load(f)["row"] for f in fs])
    row = np.sort(np.random.default_rng(17).choice(row, n, replace=False))
    z = np.load(ROOT + "/data/nnue/final/final_train_v3.npz")
    o0, o1, st, blk = (z[k][row] for k in ("occ0", "occ1", "stm", "blk"))
    keep = blk == 0
    row, o0, o1, st = row[keep], o0[keep], o1[keep], st[keep]
    p = z["kata_td_win"][row, 0].astype(np.float64) + 0.5 * z["kata_td_draw"][row, 0].astype(np.float64)
    old = np.clip(z["titanium_cp"][row].astype(np.float64), -1500, 1500)
    txt = "\n".join(board_str(int(a), int(b), 0, int(s)) for a, b, s in zip(o0, o1, st)) + "\n"
    r = subprocess.run([EXE, "label", "--nodes", str(nodes), "--tuple", TUP], input=txt, capture_output=True, text=True)
    new = np.array([float(l.split()[0]) for l in r.stdout.strip().splitlines()])
    assert len(new) == len(row), (len(new), len(row), r.stderr[-300:])
    new = np.clip(new, -1500, 1500)
    emp = 49 - np.array([bin(int(a) | int(b)).count("1") for a, b in zip(o0, o1)])
    test = np.random.default_rng(3).random(len(row)) < 0.4
    print("rows %d | %d-node labels | mean |new - old| %.0f cp, corr(new, old) %.3f, same sign %.3f" % (
        len(row), nodes, np.abs(new - old).mean(), np.corrcoef(new, old)[0, 1], np.mean(np.sign(new) == np.sign(old))))
    print("MSE vs Kata TD (win prob, scale refit):   old labels (buggy engine)  |  new labels (fixed engine)")
    for lo, hi in ((0, 49),) + tuple(SF.BANDS):
        m = (emp >= lo) & (emp <= hi)
        tr, te = m & ~test, m & test
        if te.sum() < 50:
            continue
        a = SF.scale_fit_mse(old[tr] / 400, p[tr], old[te] / 400, p[te])
        b = SF.scale_fit_mse(new[tr] / 400, p[tr], new[te] / 400, p[te])
        print("   empties %2d-%2d (n=%4d)    %.4f    |    %.4f    (%+.1f%%)" % (lo, hi, te.sum(), a, b, 100 * (b - a) / a))


if __name__ == "__main__":
    main()
