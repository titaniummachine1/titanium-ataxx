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


def base_eval_stm(occ0, occ1, blk, stm):
    """Exact S13 evaluate(), stm-relative cp. All inputs uint64/u8 arrays."""
    n = len(occ0)
    empty = ~(occ0 | occ1 | blk) & FULL
    c0 = popcnt(occ0).astype(np.float64)
    c1 = popcnt(occ1).astype(np.float64)
    mat = MATERIAL * (c0 - c1)
    pst = np.zeros(n)
    for sq in range(49):
        bit = ((occ0 >> np.uint64(sq)) & np.uint64(1)).astype(np.float64) \
            - ((occ1 >> np.uint64(sq)) & np.uint64(1)).astype(np.float64)
        pst += float(PST[sq]) * bit
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
    black_rel = mat + pst - CONTACT_PEN * cb + CONTACT_PEN * cw - mb + mw
    flip = np.where(stm == 1, -1.0, 1.0)
    return black_rel * flip + TEMPO


# 36 anchors row-major: cells [(r,c),(r,c+1),(r+1,c),(r+1,c+1)] as bit masks
CELLS = []
k = 0
for r in range(6):
    for c in range(6):
        CELLS.append([np.uint64(1 << (r * 7 + c)), np.uint64(1 << (r * 7 + c + 1)),
                      np.uint64(1 << ((r + 1) * 7 + c)), np.uint64(1 << ((r + 1) * 7 + c + 1))])
        k += 1


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
    ap.add_argument('--out', default='data/nnue/tupS20.tup')
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    d = np.load(a.cache)
    occ0 = d['occ0'].astype(np.uint64)
    occ1 = d['occ1'].astype(np.uint64)
    blk = d['blk'].astype(np.uint64)
    stm = d['stm'].astype(np.uint8)
    best_cp = np.clip(d['sc'].astype(np.float64) * 2000.0, -1500.0, 1500.0)
    n = len(occ0)
    if a.rows:
        sel = np.random.choice(n, min(a.rows, n), replace=False)
        occ0, occ1, blk, stm, best_cp = occ0[sel], occ1[sel], blk[sel], stm[sel], best_cp[sel]
        n = len(occ0)
    print('rows %d' % n, flush=True)

    # stm-relative own/enemy (stm 0 = black=x=occ0 to move)
    is_b = (stm == 0)
    own = np.where(is_b, occ0, occ1)
    enm = np.where(is_b, occ1, occ0)

    t0 = time.time()
    base = base_eval_stm(occ0, occ1, blk, stm)
    print('base eval %.0fs  mean %+.1f  std %.1f' % (time.time() - t0, base.mean(), base.std()), flush=True)
    # startpos guard: black a1 + white g7, black to move -> 150
    assert abs(base_eval_stm(np.array([np.uint64(1 << 42)]), np.array([np.uint64(1 << 6)]),
                             np.array([np.uint64(0)]), np.array([0]))[0] - 150.0) < 1e-9
    bi = block_indices(np.array([np.uint64(1 << 42)]), np.array([np.uint64(1 << 6)]))
    assert int(bi[0, 5 * 6 + 0]) == 9 and int(bi[0, 0 * 6 + 5]) == 6, bi[0]
    assert int(bi.sum()) == 15
    print('port guards OK (base 150, anchors 9/6)', flush=True)

    t0 = time.time()
    feats = block_indices(own, enm)
    print('features %.0fs' % (time.time() - t0), flush=True)
    off = (np.arange(36, dtype=np.int64)[None, :] * 81 + feats).astype(np.int64)

    resid = (best_cp - base).astype(np.float32)
    print('residual mean %+.1f std %.1f' % (resid.mean(), resid.std()), flush=True)

    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n * 0.95)
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr = torch.from_numpy(off[tr]), torch.from_numpy(resid[tr])
    Xva, yva = torch.from_numpy(off[va]), torch.from_numpy(resid[va])

    bag = torch.nn.EmbeddingBag(36 * 81, 1, mode='sum', sparse=True)
    torch.nn.init.zeros_(bag.weight)
    # NOTE: torch SGD weight_decay is dense+sparse incompatible; L2 is added
    # to the loss manually below (works on the dense weight tensor).
    opt = torch.optim.SGD(bag.parameters(), lr=a.lr)
    bs = 4096
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        bag.train()
        perm = torch.randperm(len(Xtr))
        tot, cnt = 0.0, 0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = bag(Xtr[b]).squeeze(1)
            loss = torch.nn.functional.mse_loss(pred, ytr[b]) \
                + a.wd * (bag.weight ** 2).sum()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        bag.eval()
        with torch.no_grad():
            pv = bag(Xva).squeeze(1)
            vmse = torch.nn.functional.mse_loss(pv, yva).item()
        print('epoch %d train_mse %.1f val_mse %.1f (%.0fs)' % (ep, tot / cnt, vmse, time.time() - t0), flush=True)

    w = bag.weight.detach().squeeze(1).numpy().reshape(36, 81)
    print('weight spread: max|w| %.1f  mean|w| %.2f  frac_nonzero %.3f'
          % (np.abs(w).max(), np.abs(w).mean(), (w != 0).mean()), flush=True)
    assert np.abs(w).max() < 30000, 'i16 overflow risk'
    payload = struct.pack('<%di' % w.size, *[int(round(v)) for v in w.reshape(-1)])
    with open(a.out, 'wb') as f:
        f.write(b'TUP1' + payload)
    import os
    print('wrote %s %.1fKB' % (a.out, os.path.getsize(a.out) / 1024), flush=True)


if __name__ == '__main__':
    main()
