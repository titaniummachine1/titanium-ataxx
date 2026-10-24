"""Deeper interpretability of KataAtaxx (b12c384h6tfrs2, raw net, no search). READ-ONLY, no training.

Positions: KataAtaxx's OWN self-play positions (data/datasets/KataGo_Ataxx_202604 raw tdata), fed with the stored
22 spatial + 19 global input planes exactly as Kata saw them. Side-to-move relative (plane 1 = mover, plane 2 = opponent).

Parts (run all, or pass a subset: values probes neurons heads):
  values  OCCLUSION: delete each stone, measure the change of the value head (P(win)-P(loss)). Which stones does Kata
          think matter, and does exposure / empty-neighbour count explain it?
  probes  Linear probes of 14 per-square concepts and 12 position-level concepts on the residual stream after every
          block (L0..L11) vs the raw 5x5 patch. Includes Kata's own future-stone targets (valueTargets ch2/ch3).
  neurons SwiGLU hidden units (1024 / block): single-unit and top-k correlation with each concept, plus the board
          contexts that fire the best unit (ASCII 5x5, M=mover E=enemy .=empty #=off board).
  heads   per (block, head): attention by Chebyshev distance, key kind, mean offset direction.
Usage: python training/probe_kata_deep.py [n_positions] [parts...]
"""
import glob
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P  # noqa: E402
import probe_kata_internals as K  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
KDIR = ROOT + "/data/datasets/KataGo_Ataxx_202604"
CHEB = K.CHEB
SEL = [0, 1, 2, 4, 7, 10, 11]          # blocks whose FFN hidden units are kept


def log(*a):
    print(*a, flush=True)


def load_positions(n, seed=5):
    fs = sorted(glob.glob(KDIR + "/*/tdata/*.npz"))
    rng = np.random.default_rng(seed)
    fs = [fs[i] for i in rng.choice(len(fs), 8, replace=False)]
    per = n // len(fs) + 1
    PL, GL, TD, FIN, F2, F3, ST = [], [], [], [], [], [], []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        pl = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
        occ = pl[:, 1].sum(1) + pl[:, 2].sum(1)
        emp = 49 - occ
        ok = np.nonzero((emp >= 8) & (emp <= 38))[0]
        take = rng.choice(ok, min(per, len(ok)), replace=False)
        g = z["globalTargetsNC"]
        v = z["valueTargetsNCHW"]
        PL.append(pl[take].astype(np.float32))
        GL.append(z["globalInputNC"][take].astype(np.float32))
        TD.append(g[take, 4] - g[take, 5])
        FIN.append(g[take, 0] - g[take, 1])
        F2.append(v[take, 2].reshape(len(take), 49).astype(np.float32))
        F3.append(v[take, 3].reshape(len(take), 49).astype(np.float32))
    cat = np.concatenate
    PL, GL, TD, FIN, F2, F3 = cat(PL)[:n], cat(GL)[:n], cat(TD)[:n], cat(FIN)[:n], cat(F2)[:n], cat(F3)[:n]
    w = (1 << np.arange(49, dtype=np.uint64))
    us = (PL[:, 1].astype(np.uint64) * w).sum(1).astype(np.uint64)
    en = (PL[:, 2].astype(np.uint64) * w).sum(1).astype(np.uint64)
    return PL, GL, TD, FIN, F2, F3, us, en


def build(taps_ffn=SEL, attn=True):
    m = onnx.load(K.ONNX)
    g = m.graph
    names = []
    for L in range(12):
        names.append("/model/blocks.%d/Add_3_output_0" % L)
        if attn:
            names.append("/model/blocks.%d/Softmax_output_0" % L)
    for L in taps_ffn:
        names.append("/model/blocks.%d/Mul_7_output_0" % L)
    for t in names:
        g.output.append(onnx.helper.make_empty_tensor_value_info(t))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    m2 = onnx.load(K.ONNX)
    so2 = ort.SessionOptions()
    so2.intra_op_num_threads = 4
    plain = ort.InferenceSession(m2.SerializeToString(), so2, providers=["CPUExecutionProvider"])
    return sess, plain, names


def value_of(out_value):
    e = np.exp(out_value - out_value.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    return p[:, 0] - p[:, 1]


def forward_all(sess, names, PL, GL, bs=20):
    n = len(PL)
    store = {nm: [] for nm in names}
    vals, own, pol = [], [], []
    outnames = [o.name for o in sess.get_outputs()]
    t0 = time.time()
    for i in range(0, n, bs):
        outs = sess.run(None, {"input_spatial": PL[i:i + bs].reshape(-1, 22, 7, 7), "input_global": GL[i:i + bs]})
        d = dict(zip(outnames, outs))
        for nm in names:
            store[nm].append(d[nm])
        vals.append(d["out_value"])
        own.append(d["out_ownership"])
        pol.append(d["out_policy"])
        log("forward %d/%d (%.0fs)" % (min(i + bs, n), n, time.time() - t0))
    return {k: np.concatenate(v) for k, v in store.items()}, np.concatenate(vals), np.concatenate(own), np.concatenate(pol)


def bits(b):
    return K.bits49(b).astype(bool)


def extra_concepts(us, en):
    n = len(us)
    occ = us | en
    empty = ~occ & P.FULL
    clone_u, clone_e = P.spread(us, P.R1) & empty, P.spread(en, P.R1) & empty
    land_u = clone_u | (P.spread(us, P.R2) & empty)
    land_e = clone_e | (P.spread(en, P.R2) & empty)
    E, U, N_ = bits(empty), bits(us), bits(en)
    nan = lambda: np.full((n, 49), np.nan)
    empt_nb = np.stack([P.pc(empty & P.R1[s]) for s in range(49)], 1).astype(float)
    ring_u = np.stack([P.pc(us & P.R1[s]) for s in range(49)], 1).astype(float)
    ring_e = np.stack([P.pc(en & P.R1[s]) for s in range(49)], 1).astype(float)
    out = {k: nan() for k in ("own_empty_nbrs", "enemy_empty_nbrs", "my_gain", "enemy_gain", "reach_mine", "reach_enemy")}
    out["own_empty_nbrs"][U] = empt_nb[U]
    out["enemy_empty_nbrs"][N_] = empt_nb[N_]
    Lu, Le = bits(land_u), bits(land_e)
    out["my_gain"][Lu] = ring_e[Lu]
    out["enemy_gain"][Le] = ring_u[Le]
    out["reach_mine"][E] = Lu[E].astype(float)
    out["reach_enemy"][E] = Le[E].astype(float)
    return out


def components(emptyrow):
    seen = np.zeros(49, bool)
    sizes = []
    for s in range(49):
        if emptyrow[s] and not seen[s]:
            st, k = [s], 0
            seen[s] = True
            while st:
                a = st.pop()
                k += 1
                r, c = divmod(a, 7)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < 7 and 0 <= cc < 7:
                            t = rr * 7 + cc
                            if emptyrow[t] and not seen[t]:
                                seen[t] = True
                                st.append(t)
            sizes.append(k)
    return sizes


def global_concepts(us, en):
    n = len(us)
    occ = us | en
    empty = ~occ & P.FULL
    land_u = (P.spread(us, P.R1) | P.spread(us, P.R2)) & empty
    land_e = (P.spread(en, P.R1) | P.spread(en, P.R2)) & empty
    ring_e = np.stack([P.pc(en & P.R1[s]) for s in range(49)], 1).astype(float)
    ring_u = np.stack([P.pc(us & P.R1[s]) for s in range(49)], 1).astype(float)
    Lu, Le = bits(land_u), bits(land_e)
    E = bits(empty)
    nl_e, nl_u = P.spread(land_e, P.R1), P.spread(land_u, P.R1)
    safe_u = np.array([sum(1 for s in range(49) if (int(us[i]) >> s) & 1 and not (int(empty[i]) & int(P.R1[s]))) for i in range(n)], float)
    safe_e = np.array([sum(1 for s in range(49) if (int(en[i]) >> s) & 1 and not (int(empty[i]) & int(P.R1[s]))) for i in range(n)], float)
    comp = [components(E[i]) for i in range(n)]
    g = {
        "material_diff": (P.pc(us) - P.pc(en)).astype(float),
        "empties": P.pc(empty).astype(float),
        "my_landing_squares": Lu.sum(1).astype(float),
        "enemy_landing_squares": Le.sum(1).astype(float),
        "my_best_capture": np.where(Lu, ring_e, 0).max(1),
        "enemy_best_capture": np.where(Le, ring_u, 0).max(1),
        "my_exposed_stones": P.pc(us & nl_e).astype(float),
        "enemy_exposed_stones": P.pc(en & nl_u).astype(float),
        "safe_stone_diff": safe_u - safe_e,
        "n_empty_regions": np.array([len(c) for c in comp], float),
        "odd_empty_regions": np.array([sum(1 for x in c if x % 2) for c in comp], float),
        "largest_empty_region": np.array([max(c) if c else 0 for c in comp], float),
    }
    return g


def ridge_fit_r2(Xtr, ytr, Xte, yte, lam):
    return K.ridge_r2(Xtr, ytr, Xte, yte, lam=lam)


def part_values(plain, PL, GL, us, en, nocc=100):
    log("\n=== VALUES: occlusion (delete one stone, change of mover's P(win)-P(loss)); %d positions" % nocc)
    rows = []
    n = min(nocc, len(PL))
    C = K.concepts_per_square(us[:n], en[:n])
    X = extra_concepts(us[:n], en[:n])
    for i in range(n):
        base = value_of(plain.run(["out_value"], {"input_spatial": PL[i:i + 1].reshape(1, 22, 7, 7), "input_global": GL[i:i + 1]})[0])[0]
        stones = [s for s in range(49) if PL[i, 1, s] or PL[i, 2, s]]
        batch = np.repeat(PL[i:i + 1], len(stones), 0)
        for j, s in enumerate(stones):
            batch[j, 1, s] = 0
            batch[j, 2, s] = 0
        v = value_of(plain.run(["out_value"], {"input_spatial": batch.reshape(-1, 22, 7, 7), "input_global": np.repeat(GL[i:i + 1], len(stones), 0)})[0])
        for j, s in enumerate(stones):
            mine = bool(PL[i, 1, s])
            rows.append((i, s, mine, base - v[j], X["own_empty_nbrs" if mine else "enemy_empty_nbrs"][i, s],
                         C["exposed_mine" if mine else "exposed_theirs"][i, s]))
        if (i + 1) % 20 == 0:
            log("occlusion %d/%d" % (i + 1, n))
    R = np.array(rows, float)
    own, d, nb, ex = R[:, 2] == 1, R[:, 3], R[:, 4], R[:, 5]
    log("mean value drop when a stone is DELETED (positive = the stone was good for the mover):")
    log("  my stones   mean %+.4f   (n=%d)" % (d[own].mean(), own.sum()))
    log("  enemy stones mean %+.4f   (n=%d)" % (d[~own].mean(), (~own).sum()))
    log("by number of EMPTY neighbours (stone safety), mean drop | my stones | enemy stones")
    for k in range(9):
        a, b = own & (nb == k), (~own) & (nb == k)
        log("  empty nbrs %d   %+.4f (n=%4d) | %+.4f (n=%4d)" % (k, d[a].mean() if a.any() else np.nan, a.sum(), d[b].mean() if b.any() else np.nan, b.sum()))
    log("by EXPOSED (enemy/mover can land next to it):")
    for flag in (0, 1):
        a, b = own & (ex == flag), (~own) & (ex == flag)
        log("  exposed=%d   my stones %+.4f (n=%d) | enemy stones %+.4f (n=%d)" % (flag, d[a].mean(), a.sum(), d[b].mean(), b.sum()))
    # how much of the per-stone importance do owner / empty-neighbour count / exposure explain
    feats = [own.astype(float), (~own).astype(float)]
    for k in range(1, 9):
        feats += [(own & (nb == k)).astype(float), ((~own) & (nb == k)).astype(float)]
    feats += [(own & (ex == 1)).astype(float), ((~own) & (ex == 1)).astype(float)]
    F = np.stack(feats, 1)
    ntr = int(len(d) * 0.7)
    w = np.linalg.lstsq(F[:ntr], d[:ntr], rcond=None)[0]
    pr = F[ntr:] @ w
    r2 = 1 - ((d[ntr:] - pr) ** 2).mean() / ((d[ntr:] - d[ntr:].mean()) ** 2).mean()
    own_only = np.stack([own.astype(float), (~own).astype(float)], 1)
    w0 = np.linalg.lstsq(own_only[:ntr], d[:ntr], rcond=None)[0]
    pr0 = own_only[ntr:] @ w0
    r20 = 1 - ((d[ntr:] - pr0) ** 2).mean() / ((d[ntr:] - d[ntr:].mean()) ** 2).mean()
    log("R2 of per-stone importance: owner only %.3f | + empty-neighbour count + exposure %.3f" % (r20, r2))


def part_probes(H, PL, TD, FIN, F2, F3, us, en):
    n = len(PL)
    log("\n=== PROBES per-square (test R2; input = raw 5x5 patch, then residual stream after block L0..L11)")
    C = K.concepts_per_square(us, en)
    C.update(extra_concepts(us, en))
    U, Eb = PL[:, 1], PL[:, 2]
    C["future_stone_ch2"] = F2
    C["future_stone_ch3"] = F3
    tr = np.arange(n) < int(n * 0.7)
    patch = np.zeros((n, 49, 100), np.float32)
    for s in range(49):
        r, c = divmod(s, 7)
        k = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 7 and 0 <= cc < 7:
                    t = rr * 7 + cc
                    patch[:, s, 4 * k] = U[:, t]
                    patch[:, s, 4 * k + 1] = Eb[:, t]
                    patch[:, s, 4 * k + 2] = 1 - U[:, t] - Eb[:, t]
                else:
                    patch[:, s, 4 * k + 3] = 1
                k += 1
    log("%-18s %6s" % ("concept", "patch") + "".join("%6s" % ("L%d" % L) for L in range(12)))
    for name, y in C.items():
        m = ~np.isnan(y)
        if m.sum() < 300:
            continue
        rows_tr = np.repeat(tr[:, None], 49, 1)[m]
        res = []
        for key in ["patch"] + list(range(12)):
            Fm = patch if key == "patch" else H[key]
            res.append(ridge_fit_r2(Fm[m][rows_tr], y[m][rows_tr], Fm[m][~rows_tr], y[m][~rows_tr], 10.0))
        log("%-18s" % name + "".join("%6.2f" % r for r in res))
    log("\n=== PROBES position-level (mean-pooled tokens; test R2, ridge lam=100; 'counts' = pooled raw own/enemy counts only)")
    G = global_concepts(us, en)
    G["kata_TD_value"] = TD.astype(float)
    G["game_result"] = FIN.astype(float)
    base = np.stack([U.sum(1), Eb.sum(1)], 1).astype(np.float32)
    pooled = [h.mean(1) for h in H]
    log("%-22s %6s" % ("concept", "counts") + "".join("%6s" % ("L%d" % L) for L in range(12)))
    for name, y in G.items():
        res = []
        for key in ["counts"] + list(range(12)):
            Fm = base if key == "counts" else pooled[key]
            res.append(ridge_fit_r2(Fm[tr], y[tr], Fm[~tr], y[~tr], 100.0))
        log("%-22s" % name + "".join("%6.2f" % r for r in res))


def ascii_ctx(PL, i, s):
    r, c = divmod(s, 7)
    lines = []
    for dr in range(-2, 3):
        row = ""
        for dc in range(-2, 3):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < 7 and 0 <= cc < 7):
                ch = "#"
            else:
                t = rr * 7 + cc
                ch = "M" if PL[i, 1, t] else ("E" if PL[i, 2, t] else ".")
            if dr == 0 and dc == 0:
                ch = ch.lower() if ch != "." else ","
            row += ch
        lines.append(row)
    return lines


def part_neurons(A, PL, us, en, F3):
    n = len(PL)
    C = K.concepts_per_square(us, en)
    C.update(extra_concepts(us, en))
    C["future_stone_ch3"] = F3
    names = ["exposed_mine", "exposed_theirs", "jump_only_me", "clone_net2ply", "my_gain", "enemy_gain",
             "own_empty_nbrs", "reach_enemy", "future_stone_ch3"]
    tr = np.arange(n) < int(n * 0.7)
    log("\n=== NEURONS: SwiGLU hidden units (1024 per block). best |corr| of any single unit, and test R2 with the top-k units")
    log("%-17s %-5s %7s %7s %7s %7s %7s" % ("concept", "block", "|r|max", "R2@1", "R2@5", "R2@20", "R2@100"))
    best = {}
    for name in names:
        y = C[name]
        m = ~np.isnan(y)
        rows_tr = np.repeat(tr[:, None], 49, 1)[m]
        for L in SEL:
            X = A[L][m]
            yy = y[m]
            Xtr, ytr = X[rows_tr], yy[rows_tr]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Z = (Xtr - mu) / sd
            r = (Z * (ytr - ytr.mean())[:, None]).mean(0) / (ytr.std() + 1e-9)
            order = np.argsort(-np.abs(r))
            res = []
            for k in (1, 5, 20, 100):
                cols = order[:k]
                res.append(ridge_fit_r2(Xtr[:, cols], ytr, X[~rows_tr][:, cols], yy[~rows_tr], 1.0))
            best[(name, L)] = (int(order[0]), float(r[order[0]]))
            log("%-17s %-5d %7.2f %7.2f %7.2f %7.2f %7.2f" % (name, L, abs(r[order[0]]), *res))
    log("\n=== NEURONS: what fires the single best unit (5x5 around the square; lower-case/comma = the square itself)")
    for name in ("exposed_mine", "jump_only_me", "my_gain", "own_empty_nbrs"):
        for L in (1, 4, 10):
            u, r = best[(name, L)]
            acts = A[L][:, :, u]
            top = np.dstack(np.unravel_index(np.argsort(-acts.ravel())[:3], acts.shape))[0]
            log("-- %s | block %d unit %d (corr %+.2f) | mean act %.3f, max %.3f" % (name, L, u, r, acts.mean(), acts.max()))
            ctxs = [ascii_ctx(PL, int(i), int(s)) for i, s in top]
            for row in range(5):
                log("   " + "     ".join(c[row] for c in ctxs))


def part_heads(S, PL):
    n = len(PL)
    kind = np.where(PL[:, 1] > 0, 0, np.where(PL[:, 2] > 0, 1, 2))   # 0 mover 1 enemy 2 empty
    uni = np.array([(CHEB == d).sum() for d in range(7)], float) / 49.0 / 49.0
    rr = np.arange(49) // 7
    cc = np.arange(49) % 7
    dR = rr[None, :] - rr[:, None]
    dC = cc[None, :] - cc[:, None]
    log("\n=== HEADS: per (block, head). ratio = attention mass at Chebyshev distance / uniform; keys: share of mass on mover/enemy/empty stones")
    log("blk hd |  d0    d1    d2    d3+  | keys M    E    . (base %.2f %.2f %.2f) | mean offset (rows, cols)" % tuple((kind == k).mean() for k in range(3)))
    rows = []
    for L in range(12):
        A = S["/model/blocks.%d/Softmax_output_0" % L][:, :, -49:, -49:]
        for h in range(6):
            a = A[:, h]                                       # (N,49,49)
            mass = np.array([a[:, CHEB == d].sum(-1).mean() for d in range(7)]) / 49.0
            ratio = mass / uni
            r3 = mass[3:].sum() / uni[3:].sum()
            ks = [(a * (kind[:, None, :] == k)).sum((1, 2)).sum() / a.sum() for k in range(3)]
            off = ((a * dR[None]).sum((1, 2)).mean() / 49.0, (a * dC[None]).sum((1, 2)).mean() / 49.0)
            rows.append((L, h, ratio[0], ratio[1], ratio[2], r3, ks, off))
            log("%2d  %d | %5.2f %5.2f %5.2f %5.2f |      %.2f %.2f %.2f | (%+.2f, %+.2f)" % (L, h, ratio[0], ratio[1], ratio[2], r3, *ks, *off))
    log("-- most self-focused heads (d0), most d1-focused, most d2-focused, most global (d3+):")
    for lab, idx in (("d0", 2), ("d1", 3), ("d2", 4), ("d3+", 5)):
        top = sorted(rows, key=lambda r: -r[idx])[:5]
        log("  %-3s " % lab + "  ".join("L%dh%d=%.1f" % (r[0], r[1], r[idx]) for r in top))


def load_policy_positions(n_each, seed=9):
    """stage-0 rows (global input 0 == 0) and stage-1 rows (== 1, plane 4 = picked stone) with the 200-visit search policy"""
    fs = sorted(glob.glob(KDIR + "/*/tdata/*.npz"))
    rng = np.random.default_rng(seed)
    fs = [fs[i] for i in rng.choice(len(fs), 10, replace=False)]
    out = {0: [], 1: []}
    for f in fs:
        z = np.load(f, allow_pickle=False)
        pl = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49].astype(np.float32)
        gi = z["globalInputNC"].astype(np.float32)
        pol = z["policyTargetsNCMove"][:, 0, :].astype(np.float64)
        emp = 49 - pl[:, 1].sum(1) - pl[:, 2].sum(1)
        for st in (0, 1):
            ok = np.nonzero((gi[:, 0] == st) & (pol.sum(1) > 0) & (pl[:, 3].sum(1) == 0) & (emp >= 8) & (emp <= 38))[0]
            if st == 1:
                ok = ok[pl[ok, 4].sum(1) == 1]
            take = rng.choice(ok, min(n_each // len(fs) + 1, len(ok)), replace=False)
            out[st].append((pl[take], gi[take], pol[take]))
    res = {}
    for st in (0, 1):
        pl = np.concatenate([a[0] for a in out[st]])[:n_each]
        gi = np.concatenate([a[1] for a in out[st]])[:n_each]
        pol = np.concatenate([a[2] for a in out[st]])[:n_each]
        w = 1 << np.arange(49, dtype=np.uint64)
        us = (pl[:, 1].astype(np.uint64) * w).sum(1).astype(np.uint64)
        en = (pl[:, 2].astype(np.uint64) * w).sum(1).astype(np.uint64)
        pk = (pl[:, 4].astype(np.uint64) * w).sum(1).astype(np.uint64)
        res[st] = dict(pl=pl, gi=gi, q=pol / pol.sum(1, keepdims=True), us=us, en=en, pk=pk)
    return res


def softmax_masked(logits, mask):
    z = np.where(mask, logits, -1e9)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z) * mask
    return e / e.sum(1, keepdims=True)


def two_ply_after(us_i, en_i, s, origin=None):
    """mover lands on s (jump from origin if given): stones converted, and the enemy's best recapture afterwards"""
    full = int(P.FULL)
    u = int(us_i)
    if origin is not None:
        u &= ~(1 << int(origin))
    conv = int(en_i) & int(P.R1[s])
    u2 = u | (1 << s) | conv
    e2 = int(en_i) & ~conv
    emp2 = ~(u2 | e2) & full
    le = 0
    x = e2
    while x:
        b = x & -x
        k = b.bit_length() - 1
        le |= int(P.R1[k]) | int(P.R2[k])
        x ^= b
    le &= emp2
    best, y = 0, le
    while y:
        b = y & -y
        k = b.bit_length() - 1
        best = max(best, bin(u2 & int(P.R1[k])).count("1"))
        y ^= b
    return bin(conv).count("1"), best


def part_policy(n_each=500):
    log("\n=== POLICY: Kata's raw net policy vs its 200-visit search policy, how concentrated it is, and what it is made of")
    D = load_policy_positions(n_each)
    sess, _, names = build(taps_ffn=(), attn=False)
    feats = {}
    for st in (0, 1):
        d = D[st]
        S, _, _, pol = forward_all(sess, names, d["pl"], d["gi"], bs=25)
        d["S"] = [S["/model/blocks.%d/Add_3_output_0" % L] for L in range(12)]
        d["logit"] = pol                                                # (N,4,50)
        us, en, pk = d["us"], d["en"], d["pk"]
        empty = ~(us | en) & P.FULL
        if st == 0:
            clone = P.spread(us, P.R1) & empty
            org = np.zeros(len(us), np.uint64)
            for s in range(49):
                has = ((us >> np.uint64(s)) & np.uint64(1)) == 1
                org |= np.where(has & ((P.R2[s] & empty) != 0), np.uint64(1) << np.uint64(s), np.uint64(0))
            legal = bits(clone | org)
        else:
            dest = np.zeros(len(us), np.uint64)
            for s in range(49):
                is_pk = ((pk >> np.uint64(s)) & np.uint64(1)) == 1
                dest |= np.where(is_pk, P.R2[s] & empty, np.uint64(0))
            legal = bits(dest)
        d["legal"], d["clone_mask"] = legal, (bits(clone) if st == 0 else None)
        d["legal"] = np.concatenate([legal, np.zeros((len(us), 1), bool)], 1)       # pass slot never legal here
        log("stage %d: %d positions, mean legal choices %.1f" % (st, len(us), legal.sum(1).mean()))
        q = d["q"]
        qt = q.argmax(1)
        log("  channel check (top-1 agreement of net policy channel c with the search top move):")
        for c in range(4):
            p = softmax_masked(pol[:, c, :], d["legal"])
            log("    out_policy[%d]: top1 agree %.3f" % (c, (p.argmax(1) == qt).mean()))
        p = softmax_masked(pol[:, 0, :], d["legal"])
        d["p"] = p
        ql = q * d["legal"]
        ql = ql / np.maximum(ql.sum(1, keepdims=True), 1e-9)
        kl = (ql * (np.log(np.maximum(ql, 1e-9)) - np.log(np.maximum(p, 1e-9)))).sum(1).mean()
        log("  raw net vs search: top1 agree %.3f | search top-move gets raw prob %.3f (search gives it %.3f) | KL(search||raw) %.3f" % (
            (p.argmax(1) == qt).mean(), p[np.arange(len(p)), qt].mean(), q[np.arange(len(p)), qt].mean(), kl))
        order = np.argsort(-p, 1)
        cum = np.cumsum(np.take_along_axis(p, order, 1), 1)
        nleg = d["legal"].sum(1)
        for cov in (0.5, 0.8, 0.9, 0.95, 0.99):
            k = (cum < cov).sum(1) + 1
            log("  raw policy: %.0f%% of its mass sits in the top %.1f choices (of %.1f legal; median %d)" % (100 * cov, k.mean(), nleg.mean(), np.median(k)))
        rank = (p > p[np.arange(len(p)), qt][:, None]).sum(1) + 1
        log("  rank of the SEARCH top move inside the RAW ordering: top1 %.3f top2 %.3f top3 %.3f top5 %.3f mean rank %.2f" % (
            (rank == 1).mean(), (rank <= 2).mean(), (rank <= 3).mean(), (rank <= 5).mean(), rank.mean()))
    # --- composition of the policy: concepts -> logit
    for st in (0, 1):
        d = D[st]
        us, en = d["us"], d["en"]
        n = len(us)
        legal = d["legal"][:, :49]
        lg = d["logit"][:, 0, :49].astype(np.float64)
        lg = np.where(legal, lg - (lg * legal).sum(1, keepdims=True) / np.maximum(legal.sum(1, keepdims=True), 1), np.nan)
        gain = np.stack([P.pc(en & P.R1[s]) for s in range(49)], 1).astype(float)
        loss = np.stack([P.pc(us & P.R1[s]) for s in range(49)], 1).astype(float)
        empty = ~(us | en) & P.FULL
        empt_nb = np.stack([P.pc(empty & P.R1[s]) for s in range(49)], 1).astype(float)
        own_nb = np.stack([P.pc(us & P.R1[s]) for s in range(49)], 1).astype(float)
        net2 = np.full((n, 49), np.nan)
        origin_of = np.full(n, -1)
        if st == 1:
            for i in range(n):
                origin_of[i] = int(np.nonzero(d["pl"][i, 4])[0][0])
        for i in range(n):
            for s in np.nonzero(legal[i])[0]:
                if st == 0 and ((int(us[i]) >> int(s)) & 1) == 0:                 # clone square
                    c, b = two_ply_after(us[i], en[i], int(s))
                    net2[i, s] = 1 + c - b
                elif st == 1:
                    c, b = two_ply_after(us[i], en[i], int(s), origin_of[i])
                    net2[i, s] = 1 + c - b
        ctr = np.array([abs(s // 7 - 3) + abs(s % 7 - 3) for s in range(49)], float)[None, :].repeat(n, 0)
        edge = np.array([1.0 if (s // 7 in (0, 6) or s % 7 in (0, 6)) else 0.0 for s in range(49)])[None, :].repeat(n, 0)
        if st == 0:
            subsets = {"clone squares": legal & bits(P.spread(us, P.R1) & empty), "jump-origin picks": legal & bits(us)}
        else:
            subsets = {"jump destinations": legal}
        tr = np.arange(n) < int(n * 0.7)
        for nm, m in subsets.items():
            y = lg[m]
            rows_tr = np.repeat(tr[:, None], 49, 1)[m]
            def oh(v, lo, hi):
                return np.stack([(v[m] == k).astype(float) for k in range(lo, hi + 1)], 1)
            blocks = {"stones converted": oh(np.clip(gain, 0, 8), 0, 8), "my stones adjacent": oh(np.clip(loss, 0, 8), 0, 8),
                      "empty neighbours": oh(np.clip(empt_nb, 0, 8), 0, 8), "own neighbours": oh(np.clip(own_nb, 0, 8), 0, 8),
                      "centre/edge": np.stack([ctr[m], edge[m]], 1)}
            if not np.isnan(net2[m]).all():
                v = np.nan_to_num(np.clip(net2, -4, 6), nan=0.0)
                blocks["2-ply net"] = oh(v, -4, 6)
            def fit(keys):
                X = np.concatenate([blocks[k] for k in keys], 1)
                return ridge_fit_r2(X[rows_tr], y[rows_tr], X[~rows_tr], y[~rows_tr], 1.0)
            allk = list(blocks)
            log("\n[stage %d, %s] n=%d  R2 of the raw policy logit (centred per position) explained by cheap board features:" % (st, nm, m.sum()))
            for k in allk:
                log("   %-18s alone R2 %.2f" % (k, fit([k])))
            log("   ALL features together R2 %.2f   (without 2-ply: %.2f)" % (fit(allk), fit([k for k in allk if k != "2-ply net"])))
            res = []
            for key in ["patch"] + list(range(12)):
                if key == "patch":
                    continue
                Fm = d["S"][key]
                res.append(ridge_fit_r2(Fm[m][rows_tr], y[rows_tr], Fm[m][~rows_tr], y[~rows_tr], 10.0))
            log("   logit-lens (R2 of the final logit from the residual stream after block L0..L11): " + " ".join("%.2f" % r for r in res))
    # --- cheap move scorers vs Kata search top move
    d = D[0]
    us, en, q = d["us"], d["en"], d["q"]
    legal = d["legal"][:, :49]
    gain = np.stack([P.pc(en & P.R1[s]) for s in range(49)], 1).astype(float)
    qt = q.argmax(1)
    clone_mask = d["clone_mask"]
    sc_gain = np.where(clone_mask, gain + 0.1, -9)
    top_gain = sc_gain.argmax(1)
    log("\nCheap scorer vs Kata search top move (stage 0): 'max conversions among clone squares' picks the same square in %.3f of positions; "
        "raw Kata policy top1 in %.3f" % ((top_gain == qt).mean(), (d["p"].argmax(1) == qt).mean()))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 800
    parts = [a for a in sys.argv[1:] if not a.isdigit()] or ["values", "probes", "neurons", "heads", "policy"]
    if "policy" in parts:
        part_policy()
        parts = [p for p in parts if p != "policy"]
        if not parts:
            return
    PL, GL, TD, FIN, F2, F3, us, en = load_positions(n)
    log("positions", len(PL), "mean empties", 49 - (PL[:, 1].sum(1) + PL[:, 2].sum(1)).mean())
    log("check: corr(future ch2, current mover-enemy sign) %.3f | ch3 %.3f" % (
        np.corrcoef(F2.ravel(), (PL[:, 1] - PL[:, 2]).ravel())[0, 1], np.corrcoef(F3.ravel(), (PL[:, 1] - PL[:, 2]).ravel())[0, 1]))
    sess, plain, names = build()
    if "values" in parts:
        part_values(plain, PL, GL, us, en)
    if set(parts) & {"probes", "neurons", "heads"}:
        S, vals, own, pol = forward_all(sess, names, PL, GL)
        log("ownership head output: mean %.4f std %.4f (an untrained head sits near 0 with tiny std)" % (own.mean(), own.std()))
        H = [S["/model/blocks.%d/Add_3_output_0" % L] for L in range(12)]
        if "probes" in parts:
            part_probes(H, PL, TD, FIN, F2, F3, us, en)
        if "neurons" in parts:
            A = {L: S["/model/blocks.%d/Mul_7_output_0" % L] for L in SEL}
            part_neurons(A, PL, us, en, F3)
        if "heads" in parts:
            part_heads(S, PL)


if __name__ == "__main__":
    main()
