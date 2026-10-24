"""What does KataGo's Ataxx transformer compute internally? (b12c384h6tfrs2, 2026 net, raw net, no search)

1. Attention: for every block, the share of attention mass a square sends to squares at Chebyshev distance 0/1/2/>=3,
   compared with the share a uniform head would send (= how many squares sit at that distance). >1 = the head looks there.
   Also, for EMPTY query squares, which kind of square they attend to at distance 1 and 2 (own / enemy / empty).
2. Linear probes on each block's token states (one 384-d vector per square), ridge regression, train/test split by
   position. A concept that the raw input cannot predict but a deeper layer can = something the net INFERS.
   Concepts (side-to-move relative, computed exactly with bit operations):
     enemy_ring      empty square: enemy stones around it (what I convert by landing)  -- local, 3x3 (baseline)
     jump_only_me    empty square reachable by my jump but not my clone              -- distance 2
     exposed_mine    my stone next to a square the enemy can land on                  -- distance up to 3
     exposed_theirs  enemy stone next to a square I can land on
     clone_net2ply   empty square I can clone onto: (1 + converted) - enemy's best capture count after it  -- 2-ply tactic
Usage: python training/probe_kata_internals.py [n_positions]
"""
import glob
import sys

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
ONNX = ROOT + "/data/engines/kataataxx/ext/KataAtaxx_20260501_other_or_cpu/weights/b12c384h6tfrs2-s3413299200-d1434585433.onnx"
CHEB = np.array([[max(abs(a // 7 - b // 7), abs(a % 7 - b % 7)) for b in range(49)] for a in range(49)])


def build_session():
    m = onnx.load(ONNX)
    g = m.graph
    names = [n.name for n in g.node]
    att, blk = [], []
    for i in range(12):
        att.append(next(n.output[0] for n in g.node if n.name == "/model/blocks.%d/Softmax" % i))
        # block output = what the next block (or the final norm) reads first
        nxt = "/model/blocks.%d/" % (i + 1)
        cand = [n for n in g.node if n.name.startswith(nxt)]
        if cand:
            ins = set(x for n in cand for x in n.input)
            prod = [n for n in g.node if n.name.startswith("/model/blocks.%d/" % i) and any(o in ins for o in n.output)]
            blk.append(prod[-1].output[0])
        else:
            prod = [n for n in g.node if n.name.startswith("/model/blocks.%d/" % i)]
            blk.append(prod[-1].output[0])
    for t in att + blk:
        g.output.append(onnx.helper.make_empty_tensor_value_info(t))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 3
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"]), att, blk


def bits49(b):
    return ((b[:, None] >> np.arange(49, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.float32)


def concepts_per_square(us, en):
    """returns dict name -> (N,49) float with NaN where the concept is undefined"""
    n = len(us)
    occ = us | en
    empty = ~occ & P.FULL
    clone_u, clone_e = P.spread(us, P.R1) & empty, P.spread(en, P.R1) & empty
    land_u = clone_u | (P.spread(us, P.R2) & empty)
    land_e = clone_e | (P.spread(en, P.R2) & empty)
    nl_e, nl_u = P.spread(land_e, P.R1), P.spread(land_u, P.R1)
    B = lambda x: bits49(x).astype(bool)
    E, U, N_ = B(empty), B(us), B(en)
    out = {k: np.full((n, 49), np.nan) for k in ("enemy_ring", "jump_only_me", "exposed_mine", "exposed_theirs", "clone_net2ply")}
    ring_e = np.stack([P.pc(en & P.R1[s]) for s in range(49)], 1).astype(float)
    out["enemy_ring"][E] = ring_e[E]
    jo = B(land_u & ~clone_u)
    out["jump_only_me"][E] = jo[E].astype(float)
    out["exposed_mine"][U] = B(us & nl_e)[U].astype(float)
    out["exposed_theirs"][N_] = B(en & nl_u)[N_].astype(float)
    # 2-ply: clone onto s, then the enemy's best capture (my stones around any square the enemy can land on)
    cu = B(clone_u)
    for i in range(n):
        for s in map(int, np.nonzero(cu[i])[0]):
            conv = int(en[i]) & int(P.R1[s])  # python ints from here on
            u2 = int(us[i]) | (1 << s) | conv
            e2 = int(en[i]) & ~conv
            emp2 = ~(u2 | e2) & int(P.FULL)
            le = 0
            x = e2
            while x:
                b = x & -x
                k = b.bit_length() - 1
                le |= (int(P.R1[k]) | int(P.R2[k]))
                x ^= b
            le &= emp2
            best = 0
            y = le
            while y:
                b = y & -y
                k = b.bit_length() - 1
                best = max(best, bin(u2 & int(P.R1[k])).count("1"))
                y ^= b
            out["clone_net2ply"][i, s] = 1 + bin(conv).count("1") - best
    return out


def ridge_r2(Xtr, ytr, Xte, yte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, Bt = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ym))
    pred = Bt @ w + ym
    return 1 - ((yte - pred) ** 2).mean() / max(((yte - yte.mean()) ** 2).mean(), 1e-9)


def main():
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    sess, att_names, blk_names = build_session()
    fs = sorted(glob.glob(ROOT + "/data/nnue/final/moon1000/calib_c*.npz"))
    row = np.concatenate([np.load(f)["row"] for f in fs])
    z = np.load(ROOT + "/data/nnue/final/final_train_v3.npz")
    o0, o1 = z["occ0"][row], z["occ1"][row]
    emp = 49 - P.pc(o0 | o1).astype(np.int64)
    rng = np.random.default_rng(11)
    pick = rng.choice(np.nonzero((emp >= 12) & (emp <= 35))[0], n_pos, replace=False)
    us, en = o0[pick], o1[pick]
    U, Ebits = bits49(us), bits49(en)
    A_all, H_all = [[] for _ in range(12)], [[] for _ in range(12)]
    for i in range(0, n_pos, 32):
        u, e = U[i:i + 32], Ebits[i:i + 32]
        X = np.zeros((len(u), 22, 49), np.float32)
        X[:, 0], X[:, 1], X[:, 2] = 1, u, e
        gl = np.zeros((len(u), 19), np.float32)
        gl[:, 9] = 1.0
        outs = sess.run(att_names + blk_names, {"input_spatial": X.reshape(-1, 22, 7, 7), "input_global": gl})
        for L in range(12):
            A_all[L].append(outs[L])
            H_all[L].append(outs[12 + L])
        print("forward %d/%d" % (min(i + 32, n_pos), n_pos), flush=True)
    A = [np.concatenate(a) for a in A_all]
    H = [np.concatenate(h) for h in H_all]
    print("attention shape", A[0].shape, "| block output shape", H[0].shape)

    # ---- 1. attention geometry
    ntok = A[0].shape[-1]
    off = ntok - 49            # leading non-square tokens, if any
    uni = np.array([(CHEB == d).sum() for d in range(7)], float) / 49.0 / 49.0   # uniform head: share of keys at d
    print("\nATTENTION by Chebyshev distance (mass share / uniform share; >1 = the layer looks there)")
    print("layer   d0     d1     d2    d3+   | empty query -> d1: own enemy empty | d2: own enemy empty")
    occ_kind = np.where(U.astype(bool), 0, np.where(Ebits.astype(bool), 1, 2))  # 0 own 1 enemy 2 empty
    for L in range(12):
        a = A[L][:, :, off:, off:].mean(1)          # average heads: (N,49,49)
        mass = np.array([a[:, CHEB == d].sum(-1).mean() for d in range(7)]) / 49.0
        ratio = mass / uni
        line = "%2d   %5.2f  %5.2f  %5.2f  %5.2f   |" % (L, ratio[0], ratio[1], ratio[2], mass[3:].sum() / uni[3:].sum())
        for d in (1, 2):
            share = []
            for kind in range(3):
                tot = 0.0
                cnt = 0
                for q in range(49):
                    qm = occ_kind[:, q] == 2
                    if not qm.any():
                        continue
                    keys = CHEB[q] == d
                    w = a[qm][:, q, :][:, keys]
                    kk = occ_kind[qm][:, keys] == kind
                    tot += (w * kk).sum()
                    cnt += w.sum()
                share.append(tot / max(cnt, 1e-9))
            line += "  %.2f %.2f %.2f |" % tuple(share)
        print(line)
    base_kind = np.array([(occ_kind == k).mean() for k in range(3)])
    print("(baseline share of own/enemy/empty squares on the board: %.2f %.2f %.2f)" % tuple(base_kind))

    # ---- 2. probes
    C = concepts_per_square(us, en)
    tr = np.arange(n_pos) < int(n_pos * 0.7)
    print("\nPROBES: test R2 of a linear probe per square (input = linear on the raw 5x5 patch; L0..L11 = block token states)")
    # baseline = raw 5x5 patch around each square (own/enemy/empty/off-board one-hot, 25 cells x 4)
    patch = np.zeros((n_pos, 49, 25 * 4), np.float32)
    for s in range(49):
        r, c = divmod(s, 7)
        k = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 7 and 0 <= cc < 7:
                    t = rr * 7 + cc
                    patch[:, s, 4 * k] = U[:, t]
                    patch[:, s, 4 * k + 1] = Ebits[:, t]
                    patch[:, s, 4 * k + 2] = 1 - U[:, t] - Ebits[:, t]
                else:
                    patch[:, s, 4 * k + 3] = 1
                k += 1
    feats = {"input": patch}
    tok = lambda h: h.reshape(h.shape[0], h.shape[1], 49).transpose(0, 2, 1) if h.ndim == 4 else h[:, off:, :]
    print("%-15s %7s" % ("concept", "input") + "".join("%7s" % ("L%d" % L) for L in range(12)))
    for name, y in C.items():
        m = ~np.isnan(y)
        res = []
        for key in ["input"] + list(range(12)):
            Fm = feats["input"] if key == "input" else tok(H[key])
            Xs = Fm[m]
            ys = y[m]
            rows_tr = np.repeat(tr[:, None], 49, 1)[m]
            res.append(ridge_r2(Xs[rows_tr], ys[rows_tr], Xs[~rows_tr], ys[~rows_tr]))
        print("%-15s" % name + "".join("%7.2f" % r for r in res))


if __name__ == "__main__":
    main()
