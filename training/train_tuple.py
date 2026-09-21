"""Train clean-room n-tuple weights (S20) on score RESIDUALS.

Features: 36 anchors x 81 ternary 2x2 patterns (own/enemy-relative, blocker
reads as empty), cell/block order MUST match crates/titanium/src/board.rs
BLOCK2 + block2_index (asserted on the start position below).

Target: residual = dag best_score_cp - exact S13 base eval (material + PST +
tempo + contact + multicap, stm-relative). The tuple table learns what the
hand eval misses. Model = single sparse EmbeddingBag(36*81, 1): 2,916
weights, zero-init (initial pred == base only).

Base eval is ported EXACTLY from search.rs/board.rs (integer bit ops on
uint64, same constants). Start-position assertion guards both ports at once:
base_stm(startpos) == 150 and anchor (5,0) index == 9 (mirrors the Rust tests).

Usage:
  python3 training/train_tuple.py --rows 5000 --epochs 2          # smoke
  python3 training/train_tuple.py                                  # full 400k
"""
import argparse
import struct
import time

import numpy as np

FULL = np.uint64((1 << 49) - 1)


def file_bits(f):
    m = 0
    for r in range(7):
        m |= 1 << (r * 7 + f)
    return np.uint64(m)


FILE_A = file_bits(0)
FILE_G = file_bits(6)


def dist_union(bb, d):
    na = bb & ~FILE_A
    ng = bb & ~FILE_G
    s = 7 * d
    return ((bb >> np.uint64(s)) | (bb << np.uint64(s))
            | (na >> np.uint64(d)) | (ng << np.uint64(d))
            | (na >> np.uint64(8 * d)) | (ng >> np.uint64(6 * d))
            | (na << np.uint64(6 * d)) | (ng << np.uint64(8 * d))) & FULL


def jump_union(bb):
    up = bb >> np.uint64(14)
    dn = (bb << np.uint64(14)) & FULL
    FAB = FILE_A | file_bits(1)
    FFG = file_bits(5) | FILE_G
    lf = (bb & ~FAB) >> np.uint64(2)
    rt = (bb & ~FFG) << np.uint64(2)
    ul = (bb & ~FAB) >> np.uint64(16)
    ur = (bb & ~FFG) >> np.uint64(12)
    dl = (bb & ~FAB) << np.uint64(12)
    dr = (bb & ~FFG) << np.uint64(16)
    v2 = (up | dn) & FULL
    h2 = lf | rt
    n1 = ((v2 & ~FILE_A) >> np.uint64(1)) | ((v2 & ~FILE_G) << np.uint64(1))
    n2 = (h2 >> np.uint64(7)) | ((h2 << np.uint64(7)) & FULL)
    return (up | dn | lf | rt | ul | ur | dl | dr | n1 | n2) & FULL


def build_ring1():
    t = []
    for sq in range(49):
        r, f = divmod(sq, 7)
        m = 0
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr, nf = r + dr, f + df
                if 0 <= nr < 7 and 0 <= nf < 7:
                    m |= 1 << (nr * 7 + nf)
        t.append(np.uint64(m))
    return t


RING1 = build_ring1()
assert RING1[42] == (1 << 35) | (1 << 36) | (1 << 43), hex(int(RING1[42]))

PST = np.array([
    30, 20, 10, 10, 10, 20, 30,
    20, 10, 10, 5, 10, 10, 20,
    10, 10, 5, 0, 5, 10, 10,
    10, 5, 0, 0, 0, 5, 10,
    10, 10, 5, 0, 5, 10, 10,
    20, 10, 10, 5, 10, 10, 20,
    30, 20, 10, 10, 10, 20, 30,
], dtype=np.int32)

MATERIAL, TEMPO, CONTACT_PEN, MULTICAP_PEN = 100, 150, 15, 20


def popcnt(a):
    a = np.ascontiguousarray(a).ravel()
    return np.unpackbits(a.view(np.uint8)).reshape(a.shape[0], 64).sum(1)


def base_eval_stm(occ0, occ1, blk, stm, simple=False):
    """S13 evaluate(), stm-relative cp. simple=True skips the E2/E4 regional
    loops (mat+PST+tempo only) — tuple-native training target (S21d): pairs
    must learn contact-like shapes themselves, and the engine skips the
    dist_union/jump_union scans entirely (~400 ops/node back)."""
    n = len(occ0)
    c0 = popcnt(occ0).astype(np.float64)
    c1 = popcnt(occ1).astype(np.float64)
    mat = MATERIAL * (c0 - c1)
    pst = np.zeros(n)
    for sq in range(49):
        bit = ((occ0 >> np.uint64(sq)) & np.uint64(1)).astype(np.float64) \
            - ((occ1 >> np.uint64(sq)) & np.uint64(1)).astype(np.float64)
        pst += float(PST[sq]) * bit
    black_rel = mat + pst
    if not simple:
        empty = ~(occ0 | occ1 | blk) & FULL
        # black-relative contact/multicap, then flip to stm
        lb = empty & (dist_union(occ1, 1) | jump_union(occ1))  # white landings
        lw = empty & (dist_union(occ0, 1) | jump_union(occ0))  # black landings
        cb = popcnt(occ0 & dist_union(lb, 1)).astype(np.float64)
        cw = popcnt(occ1 & dist_union(lw, 1)).astype(np.float64)
        mb = np.zeros(n)
        mw = np.zeros(n)
        for sq in range(49):
            isb = ((lb >> np.uint64(sq)) & np.uint64(1)).astype(np.float64)
            isw = ((lw >> np.uint64(sq)) & np.uint64(1)).astype(np.float64)
            nb = popcnt(np.uint64(RING1[sq]) & occ0).astype(np.float64)
            nw = popcnt(np.uint64(RING1[sq]) & occ1).astype(np.float64)
            mb += isb * np.clip(nb - 1, 0, None) * MULTICAP_PEN
            mw += isw * np.clip(nw - 1, 0, None) * MULTICAP_PEN
        black_rel = black_rel - CONTACT_PEN * cb + CONTACT_PEN * cw - mb + mw
    flip = np.where(stm == 1, -1.0, 1.0)
    return black_rel * flip + TEMPO


# 36 anchors row-major: cells [(r,c),(r,c+1),(r+1,c),(r+1,c+1)] as bit masks
CELLS = []
for r in range(6):
    for c in range(6):
        CELLS.append([np.uint64(1 << (r * 7 + c)), np.uint64(1 << (r * 7 + c + 1)),
                      np.uint64(1 << ((r + 1) * 7 + c)), np.uint64(1 << ((r + 1) * 7 + c + 1))])

# 80 adjacent block pairs (MUST match board.rs PAIR_BLOCKS): H 24, V 24,
# D1 16, D2 16. Pair index = first + 81 * second.
PAIRS = []
for r in range(6):
    for c in range(4):
        PAIRS.append((0, r * 6 + c, r * 6 + c + 2))          # H
for r in range(4):
    for c in range(6):
        PAIRS.append((1, r * 6 + c, (r + 2) * 6 + c))        # V
for r in range(4):
    for c in range(4):
        PAIRS.append((2, r * 6 + c, (r + 2) * 6 + c + 2))    # D1
for r in range(4):
    for c in range(2, 6):
        PAIRS.append((3, r * 6 + c, (r + 2) * 6 + c - 2))    # D2
assert [sum(1 for p in PAIRS if p[0] == d) for d in range(4)] == [24, 24, 16, 16]
assert (0, 30, 32) in PAIRS  # H ((5,0),(5,2)), mirrors Rust test


def block_indices(own, enm):
    """(n,36) ternary indices. Must give anchor(5,0)==9 on startpos."""
    n = len(own)
    out = np.zeros((n, 36), dtype=np.int64)
    for a, cells in enumerate(CELLS):
        idx = np.zeros(n, dtype=np.int64)
        mult = 1
        for bit in cells:
            d = ((own & bit) != 0).astype(np.int64) + 2 * ((enm & bit) != 0).astype(np.int64)
            idx += d * mult
            mult *= 3
        out[:, a] = idx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache400k.npz')
    ap.add_argument('--rows', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=4.0)
    ap.add_argument('--wd', type=float, default=1e-7)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--dirs', default='0,1,2,3',
                    help='pair directions to train (subset saves lookups, e.g. 0,1)')
    ap.add_argument('--base', default='s13', choices=['s13', 'simple'],
                    help='residual base: s13 (full hand eval) or simple (mat+PST+tempo, tuple-native)')
    ap.add_argument('--target', default='residual', choices=['residual'],
                    help='teacher-score residual (logit/outcome REMOVED S21g: poison)')
    ap.add_argument('--out', default='data/nnue/tupS20.tup')
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    z = np.load(a.cache)
    occ0 = z['occ0'].astype(np.uint64)
    occ1 = z['occ1'].astype(np.uint64)
    blk = z['blk'].astype(np.uint64)
    stm = z['stm'].astype(np.uint8)
    best_cp = np.clip(z['sc'].astype(np.float64) * 2000.0, -1500.0, 1500.0)
    d_out = z['out'].astype(np.float64) if 'out' in z else np.zeros(len(occ0))
    n = len(occ0)
    if a.rows:
        sel = np.random.choice(n, min(a.rows, n), replace=False)
        occ0, occ1, blk, stm, best_cp, d_out = occ0[sel], occ1[sel], blk[sel], stm[sel], best_cp[sel], d_out[sel]
        n = len(occ0)
    print('rows %d' % n, flush=True)

    # stm-relative own/enemy (stm 0 = black=x=occ0 to move)
    is_b = (stm == 0)
    own = np.where(is_b, occ0, occ1)
    enm = np.where(is_b, occ1, occ0)

    t0 = time.time()
    base = base_eval_stm(occ0, occ1, blk, stm, simple=(a.base == 'simple'))
    print('base[%s] eval %.0fs  mean %+.1f  std %.1f'
          % (a.base, time.time() - t0, base.mean(), base.std()), flush=True)
    # startpos guard: black a1 + white g7, black to move -> 150
    assert abs(base_eval_stm(np.array([np.uint64(1 << 42)]), np.array([np.uint64(1 << 6)]),
                             np.array([np.uint64(0)]), np.array([0]),
                             simple=(a.base == 'simple'))[0] - 150.0) < 1e-9
    bi = block_indices(np.array([np.uint64(1 << 42)]), np.array([np.uint64(1 << 6)]))
    assert int(bi[0, 5 * 6 + 0]) == 9 and int(bi[0, 0 * 6 + 5]) == 6, bi[0]
    assert int(bi.sum()) == 15
    print('port guards OK (base 150, anchors 9/6)', flush=True)

    t0 = time.time()
    feats = block_indices(own, enm)
    # pair indices per direction: b1 + 81 * b2
    pidx = []
    for d in range(4):
        pp = [(x, y) for (dd, x, y) in PAIRS if dd == d]
        xa = np.array([x for x, y in pp], dtype=np.int64)
        ya = np.array([y for x, y in pp], dtype=np.int64)
        pidx.append(feats[:, xa] + 81 * feats[:, ya])
    print('features %.0fs' % (time.time() - t0), flush=True)
    off = (np.arange(36, dtype=np.int64)[None, :] * 81 + feats).astype(np.int64)

    # S21g lesson (OUT-COLUMN-POISON, LEDGER): dag sum_outcome is NOT a game
    # result (proof-search visit values, exotic perspective; white rows mean
    # 0.75). Blending it produced a -400-Elo table. Teacher best_score
    # residuals ONLY — never touch `out` again.
    resid = (best_cp - base).astype(np.float32)
    print('residual mean %+.1f std %.1f' % (resid.mean(), resid.std()), flush=True)
    # S21f: trust high-visit rows (wsum weights, mean-normalized). Absent in
    # old caches -> uniform weights (backward compatible).
    if 'wsum' in z:
        sw = z['wsum'].astype(np.float64)
        if a.rows:
            sw = sw[sel]
        sw = np.clip(sw, 0, None)
        sw = sw / max(sw.mean(), 1e-9)
        print('wsum weights: mean 1.0 max %.1f frac>2x %.3f' % (sw.max(), (sw > 2).mean()), flush=True)
        samp_w = torch.from_numpy(sw.astype(np.float32))
    else:
        samp_w = torch.ones(n)

    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n * 0.95)
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr = torch.from_numpy(off[tr]), torch.from_numpy(resid[tr])
    Xva, yva = torch.from_numpy(off[va]), torch.from_numpy(resid[va])

    # 1 single bag + 4 pair bags + scalar bias. Index-0 entries are
    # constrained to exactly 0 after every step (empty patterns contribute
    # nothing, so the engine can SKIP them — bit-identical forward).
    bags = torch.nn.ModuleList(
        [torch.nn.EmbeddingBag(36 * 81, 1, mode='sum', sparse=True)]
        + [torch.nn.EmbeddingBag(81 * 81, 1, mode='sum', sparse=True) for _ in range(4)]
    )
    for bg in bags:
        torch.nn.init.zeros_(bg.weight)
    bias = torch.nn.Parameter(torch.zeros(1))
    params = list(bags.parameters()) + [bias]
    # NOTE: torch SGD weight_decay is dense+sparse incompatible; L2 is added
    # to the loss manually below (works on the dense weight tensors).
    opt = torch.optim.SGD(params, lr=a.lr)
    Xp = [torch.from_numpy(p) for p in pidx]

    def zero_constrained():
        with torch.no_grad():
            for bg in bags:
                bg.weight.data[0].zero_()
    XPtr_tr = [p[tr] for p in Xp]
    XPtr_va = [p[va] for p in Xp]
    wtr, wva = samp_w[tr], samp_w[va]

    active = [int(x) for x in a.dirs.split(',')]
    assert set(active) <= {0, 1, 2, 3} and active, a.dirs

    def forward(Xb, Xpb):
        pred = bags[0](Xb).squeeze(1) + bias
        for d in active:
            pred = pred + bags[d + 1](Xpb[d]).squeeze(1)
        return pred

    bs = 4096
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        for bg in bags:
            bg.train()
        perm = torch.randperm(len(Xtr))
        tot, cnt = 0.0, 0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = forward(Xtr[b], [p[b] for p in XPtr_tr])
            l2 = sum((bg.weight ** 2).sum() for bg in bags)
            wb = wtr[b]
            loss = ((pred - ytr[b]) ** 2 * wb).mean() + a.wd * l2
            loss.backward()
            opt.step()
            zero_constrained()
            tot += loss.item() * len(b)
            cnt += len(b)
        for bg in bags:
            bg.eval()
        with torch.no_grad():
            pv = forward(Xva, XPtr_va)
            vmse = (((pv - yva) ** 2) * wva).mean().item()
        print('epoch %d train_mse %.1f val_mse %.1f bias %+.1f (%.0fs)'
              % (ep, tot / cnt, vmse, bias.item(), time.time() - t0), flush=True)

    ws = bags[0].weight.detach().squeeze(1).numpy().reshape(36, 81)
    wp = [bg.weight.detach().squeeze(1).numpy().reshape(6561) for bg in list(bags)[1:]]
    assert ws[0, 0] == 0 and all(p[0] == 0 for p in wp), 'index-0 constraint broken'
    allw = np.concatenate([ws.reshape(-1)] + [p.reshape(-1) for p in wp])
    print('weight spread: max|w| %.1f mean|w| %.2f frac_nonzero %.3f bias %+.1f'
          % (np.abs(allw).max(), np.abs(allw).mean(), (allw != 0).mean(), float(bias.item())),
          flush=True)
    # TUP4 layout: magic + bias i32 + singles i8 + per-dir ACTIVE counts
    # (LE u32) + active direction tables i8. Values clip to [-127, 127]
    # (S21e: only 4/29k exceed it, zero gate loss).
    counts = [24, 24, 16, 16]
    q = lambda v: max(-127, min(127, int(round(v))))
    payload = struct.pack('<i', int(round(float(bias.item()))))
    payload += bytes(bytearray([(q(v) & 0xFF) for v in ws.reshape(-1)]))
    intc = [counts[d] if d in active else 0 for d in range(4)]
    payload += struct.pack('<4i', *intc)
    for d in range(4):
        seg = wp[d].reshape(-1) if d in active else np.zeros(6561)
        payload += bytes(bytearray([(q(v) & 0xFF) for v in seg]))
    with open(a.out, 'wb') as f:
        f.write(b'TUP4' + payload)
    import os
    print('wrote %s %.1fKB dirs %s' % (a.out, os.path.getsize(a.out) / 1024, a.dirs), flush=True)


if __name__ == '__main__':
    main()
