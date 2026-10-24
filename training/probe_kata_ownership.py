"""What does KataGo's net expect to happen to each square? Read its ownership head (no search) on real positions
and break it down by board concepts computed backwards with bit operations.

ownership: per square, the net's prediction of who owns it at game end, +1 = side to move, -1 = opponent.
Stones: grouped by owner x (empty neighbours) x (can the enemy land next to it).
Empty squares: grouped by (my reach, enemy reach) x what I convert there x what the enemy converts there.
Positions: KataGo calibration rows (stratified by empties), all stm==0.
"""
import glob
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
ONNX = ROOT + "/data/engines/kataataxx/ext/KataAtaxx_20260501_other_or_cpu/weights/b12c384h6tfrs2-s3413299200-d1434585433.onnx"


def bits49(b):
    return ((b[:, None] >> np.arange(49, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.float32)


def kata_eval(sess, us, en, bs=64):
    own, win = [], []
    for i in range(0, len(us), bs):
        u, e = bits49(us[i:i + bs]), bits49(en[i:i + bs])
        X = np.zeros((len(u), 22, 49), np.float32)
        X[:, 0], X[:, 1], X[:, 2] = 1, u, e
        g = np.zeros((len(u), 19), np.float32)
        g[:, 9] = 1.0
        out = sess.run(None, {"input_spatial": X.reshape(-1, 22, 7, 7), "input_global": g})
        v = np.exp(out[1] - out[1].max(1, keepdims=True))
        win.append(v[:, 0] / v.sum(1))
        own.append(np.tanh(out[4].reshape(-1, 49)) if np.abs(out[4]).max() > 1.0 else out[4].reshape(-1, 49))
    return np.concatenate(win), np.concatenate(own)


def main():
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    so = ort.SessionOptions()
    so.intra_op_num_threads = 3
    sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    fs = sorted(glob.glob(ROOT + "/data/nnue/final/moon1000/calib_c*.npz"))
    row = np.concatenate([np.load(f)["row"] for f in fs])
    z = np.load(ROOT + "/data/nnue/final/final_train_v3.npz")
    o0, o1 = z["occ0"][row], z["occ1"][row]
    assert (z["stm"][row] == 0).all()
    emp = 49 - P.pc(o0 | o1).astype(np.int64)
    rng = np.random.default_rng(3)
    pick = np.concatenate([rng.choice(np.nonzero((emp >= lo) & (emp <= hi))[0], n_pos // 3, replace=False)
                           for lo, hi in ((9, 15), (16, 24), (25, 35))])
    us, en = o0[pick], o1[pick]
    win, own = kata_eval(sess, us, en)
    occ = us | en
    empty = ~occ & P.FULL
    clone_u, clone_e = P.spread(us, P.R1) & empty, P.spread(en, P.R1) & empty
    land_u = clone_u | (P.spread(us, P.R2) & empty)
    land_e = clone_e | (P.spread(en, P.R2) & empty)
    near_land_e = P.spread(land_e, P.R1)
    near_land_u = P.spread(land_u, P.R1)
    rows = []
    for s in range(49):
        bit = np.uint64(1) << np.uint64(s)
        isu, ise, ise_mt = (us & bit) != 0, (en & bit) != 0, (empty & bit) != 0
        ne = P.pc(empty & P.R1[s]).astype(np.int64)
        gu = P.pc(en & P.R1[s]).astype(np.int64)
        ge = P.pc(us & P.R1[s]).astype(np.int64)
        ru = np.where((clone_u & bit) != 0, 2, np.where((land_u & bit) != 0, 1, 0))
        re = np.where((clone_e & bit) != 0, 2, np.where((land_e & bit) != 0, 1, 0))
        exp_u = (near_land_e & bit) != 0
        exp_e = (near_land_u & bit) != 0
        rows.append((s, isu, ise, ise_mt, ne, gu, ge, ru, re, exp_u, exp_e, own[:, s]))
    print("positions %d | kata win prob mean %.3f | ownership sign check: mean own-stone %.2f, enemy-stone %.2f"
          % (len(pick), win.mean(), np.mean([r[11][r[1]].mean() for r in rows if r[1].any()]),
             np.mean([r[11][r[2]].mean() for r in rows if r[2].any()])))

    def agg(mask_fn, label_fn):
        acc = {}
        for r in rows:
            m = mask_fn(r)
            for lab in np.unique(label_fn(r)[m]) if m.any() else []:
                mm = m & (label_fn(r) == lab)
                a = acc.setdefault(lab, [0.0, 0])
                a[0] += float(r[11][mm].sum())
                a[1] += int(mm.sum())
        return {k: (v[0] / v[1], v[1]) for k, v in sorted(acc.items()) if v[1] >= 200}

    print("\nMY STONES: expected final ownership (+1 = stays mine) by empty neighbours and exposure")
    for ex in (False, True):
        res = agg(lambda r: r[1] & (r[9] == ex), lambda r: r[4])
        print("  %s: " % ("EXPOSED (enemy can land next to it)" if ex else "not exposed                       ")
              + "  ".join("%d empty:%+.2f(n%d)" % (k, v[0], v[1]) for k, v in res.items()))
    print("\nENEMY STONES: expected final ownership (-1 = stays theirs) by empty neighbours and whether I can land next to it")
    for ex in (False, True):
        res = agg(lambda r: r[2] & (r[10] == ex), lambda r: r[4])
        print("  %s: " % ("I CAN ATTACK it" if ex else "safe from me  ")
              + "  ".join("%d empty:%+.2f(n%d)" % (k, v[0], v[1]) for k, v in res.items()))
    print("\nEMPTY SQUARES: expected final ownership by reach (0 none, 1 jump-only, 2 clone) me/enemy")
    res = agg(lambda r: r[3], lambda r: r[7] * 3 + r[8])
    for k, v in res.items():
        print("  my reach %d, enemy reach %d: %+.2f (n%d)" % (k // 3, k % 3, v[0], v[1]))
    print("\nEMPTY SQUARES both sides can reach: ownership by (enemy stones around it = what I convert) - (my stones around it)")
    res = agg(lambda r: r[3] & (r[7] > 0) & (r[8] > 0), lambda r: np.clip(r[5] - r[6], -4, 4))
    print("  " + "  ".join("%+d:%+.2f" % (k, v[0]) for k, v in res.items()))


if __name__ == "__main__":
    main()
