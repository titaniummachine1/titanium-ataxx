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


def q2_indices(own, enm, blk):
    """(n,36) QUATERNARY 2x2 indices (S36 enriched input = Moonbird level-1
    geometry): digits 0 empty, 1 own, 2 enemy, 3 blocker; same anchors and
    cell order as block_indices (mult x4). Startpos: (5,0)->16, (0,5)->8."""
    n = len(own)
    out = np.zeros((n, 36), dtype=np.uint8)
    for a, cells in enumerate(CELLS):
        idx = np.zeros(n, dtype=np.uint64)
        mult = 1
        for bit in cells:
            o = (own & bit) != 0
            e = (enm & bit) != 0
            bl = (blk & bit) != 0
            d = o.astype(np.uint64) + 2 * e.astype(np.uint64)
            d = np.where(bl, np.uint64(3), d)
            idx += d * mult
            mult *= 4
        out[:, a] = idx.astype(np.uint8)
    return out


RING1A = np.array(build_ring1(), dtype=np.uint64)


def env_tables(own, enm, blk):
    """Per-square ring tallies (S36 enriched input = Moonbird envCounts
    geometry, own-trained weights). Returns (US, THEM) int32 (n,49):
    US[sq] = sq*729 + o*81 + e*9 + bl (own/foe from OUR side);
    THEM swaps the perspective. All-empty rings index sq*729
    (constrained 0, skipped in forward)."""
    n = len(own)
    US = np.zeros((n, 49), dtype=np.int32)
    TH = np.zeros((n, 49), dtype=np.int32)
    for s in range(49):
        ring = RING1A[s]
        o = popcnt(own & ring).astype(np.int64)
        e = popcnt(enm & ring).astype(np.int64)
        bl = popcnt(blk & ring).astype(np.int64)
        US[:, s] = (s * 729 + o * 81 + e * 9 + bl).astype(np.int32)
        TH[:, s] = (s * 729 + e * 81 + o * 9 + bl).astype(np.int32)
    return US, TH


def mob_tables(own, enm, blk):
    """Mobility (S36 enriched input): phase = total occupancy (cap 49),
    count = clone-targets & empty per color. Returns (US, THEM) int64 (n,)
    = phase*50 + count. Count 0 constrained 0, skipped in forward."""
    total = own | enm | blk
    empty = (~total) & FULL
    phase = np.minimum(popcnt(total), 49)
    cu = np.minimum(popcnt(dist_union(own, 1) & empty), 49)
    ct = np.minimum(popcnt(dist_union(enm, 1) & empty), 49)
    return (phase * 50 + cu).astype(np.int64), (phase * 50 + ct).astype(np.int64)


def capn_tables(own, enm, blk):
    """Max conversion available per color (user: network input for how many
    pieces moves convert; STATELESS so TT stays sound — last-move counts
    would corrupt transposition entries). Returns (US, THEM) int64 (n,)
    = max foe-adjacent over landing squares (0-8). US = own landings x foe
    rings, THEM mirrored. 0 constrained 0, skipped in forward."""
    total = own | enm | blk
    empty = (~total) & FULL
    lu = empty & (dist_union(own, 1) | jump_union(own))
    lv = empty & (dist_union(enm, 1) | jump_union(enm))
    mu = np.zeros(len(own), dtype=np.int64)
    mt = np.zeros(len(own), dtype=np.int64)
    for s in range(49):
        bit = np.uint64(1 << s)
        in_u = (lu & bit) != 0
        in_t = (lv & bit) != 0
        cu = popcnt(enm & RING1A[s]).astype(np.int64)
        ct = popcnt(own & RING1A[s]).astype(np.int64)
        mu = np.maximum(mu, np.where(in_u, cu, 0))
        mt = np.maximum(mt, np.where(in_t, ct, 0))
    return np.minimum(mu, 8), np.minimum(mt, 8)


def ring_sorted():
    """Ring squares per square in ASCENDING order (matches NEIGH_POS order
    used by extract8/PEXT: HANDOFFS + extract8_matches_ring1 test)."""
    out = []
    for sq in range(49):
        r, f = sq // 7, sq % 7
        cells = []
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                rr, ff = r + dr, f + df
                if 0 <= rr < 7 and 0 <= ff < 7:
                    cells.append(rr * 7 + ff)
        out.append(sorted(cells))
    return out


RINGSORT = ring_sorted()


def conv_tables(own, enm, blk):
    """Conversion-kernel keys (user: index by the LUT key used for
    conversion). For each square: 8-bit key of FOE occupancy in ascending
    ring order = EXACTLY the key infect_via_lut feeds CONVERT8 (blockers
    read as empty, same conflation). US table sees foe bits, THEM table
    sees own bits. Returns (US, THEM) int32 (n,49), 0-255. Key 0
    constrained 0 (empty surroundings contribute nothing)."""
    n = len(own)
    US = np.zeros((n, 49), dtype=np.int32)
    TH = np.zeros((n, 49), dtype=np.int32)
    for sq in range(49):
        ring = RINGSORT[sq]
        ku = 0
        kt = 0
        for i, s in enumerate(ring):
            bit = np.uint64(1 << s)
            ku |= (((enm & bit) != 0).astype(np.int64) << i)
            kt |= (((own & bit) != 0).astype(np.int64) << i)
        US[:, sq] = ku.astype(np.int32)
        TH[:, sq] = kt.astype(np.int32)
    return US, TH


def capn_bruteforce(cells, color):
    """Independent reference: enumerate moves, max conversions. cells =
    list of 49 (0 empty 1 black 2 white), color 1/2. Cross-check only."""
    best = 0
    for o, v in enumerate(cells):
        if v != color:
            continue
        r, f = o // 7, o % 7
        for dr in (-2, -1, 0, 1, 2):
            for df in (-2, -1, 0, 1, 2):
                if dr == 0 and df == 0:
                    continue
                rr, ff = r + dr, f + df
                if not (0 <= rr < 7 and 0 <= ff < 7):
                    continue
                d = rr * 7 + ff
                if cells[d] != 0:
                    continue
                foe = 3 - color
                conv = 0
                for er in (-1, 0, 1):
                    for ef in (-1, 0, 1):
                        if er == 0 and ef == 0:
                            continue
                        sr, sf = rr + er, ff + ef
                        if 0 <= sr < 7 and 0 <= sf < 7 and cells[sr * 7 + sf] == foe:
                            conv += 1
                best = max(best, conv)
    return best


def load_tup(path):
    """Warm start (PBT exploit step): read TUP4/5/6/7 into bags, missing
    sections stay zero-init. Returns (bias, ws, wps, rest, capconv, paa)
    with rest/capconv/paa None when the file predates them."""
    buf = open(path, 'rb').read()
    magic = buf[:4]
    assert magic in (b'TUP4', b'TUP5', b'TUP6', b'TUP7'), magic
    o = 8
    bias0 = float(struct.unpack('<i', buf[4:8])[0])
    ws = np.frombuffer(buf[o:o + 2916], dtype=np.int8).astype(np.float32)
    o += 2916
    counts = struct.unpack('<4i', buf[o:o + 16])
    o += 16
    wps = []
    for d in range(4):
        wps.append(np.frombuffer(buf[o:o + 6561], dtype=np.int8).astype(np.float32))
        o += 6561

    def r16(n):
        nonlocal o
        v = np.frombuffer(buf[o:o + 2 * n], dtype='<i2').astype(np.float32)
        o += 2 * n
        return v

    rest = capconv = paa = None
    if magic in (b'TUP5', b'TUP6', b'TUP7'):
        wq = r16(36 * 256)
        weu, wet = r16(35721), r16(35721)
        wmu, wmt = r16(2500), r16(2500)
        rest = (wq, weu, wet, wmu, wmt)
    if magic in (b'TUP6', b'TUP7'):
        wcu, wct = r16(9), r16(9)
        wvu, wvt = r16(256), r16(256)
        capconv = (wcu, wct, wvu, wvt)
    if magic == b'TUP7':
        paa = np.frombuffer(buf[o:o + 80 * 6561], dtype=np.int8).astype(np.float32)
    return bias0, ws, wps, rest, capconv, paa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache400k.npz')
    ap.add_argument('--rows', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--max-epochs', type=int, default=60,
                    help='S26 flywheel cap: with --plateau, keep training to this many epochs.')
    ap.add_argument('--plateau', type=float, default=0.0,
                    help='S26 flywheel stop: halt when (prev_val - val)/prev_val < plateau '
                    '(relative val improvement stalls). 0 = off (fixed --epochs). '
                    'Flywheel recipe: --plateau 1e-4 (4 digits, you called it) with '
                    '--max-epochs 60, --ckpt to keep every epoch net.')
    ap.add_argument('--ckpt', default='',
                    help='S26 flywheel checkpoints: path template, e.g. data/nnue/fly12M_ep{ep}.tup. '
                    'Empty = only final --out. Best-val epoch also copied to --out.')
    ap.add_argument('--lr', type=float, default=0.01,
                    help='SGD lr (gated recipe; 4.0-era default exploded weights to rails).')
    ap.add_argument('--opt', default='sgd', choices=['sgd', 'adam', 'adamw', 'radam'],
                    help='optimizer family (dossier order: sgd status quo, adam/WD-off, '
                    'adamw, radam; + --lookahead approximates Ranger). Same init/data '
                    'required for fair comparison; judge exported net by games, not loss.')
    ap.add_argument('--opt-lr', type=float, default=0.0,
                    help='lr for adam-family (SGD --lr ignored when set). 0 = auto: '
                    '0.001 adam/adamw, 0.001 radam. Adaptive methods want ~1e-3, not SGD scale.')
    ap.add_argument('--decay', type=float, default=0.0,
                    help='decoupled weight decay for adamw/radam (dossier late-phase '
                    'suggests 1e-5..1e-4). 0 = off. Ignored by sgd/adam (manual L2 '
                    'in loss covers those via --wd).')
    ap.add_argument('--freq-decay', type=float, default=0.0,
                    help='rarity-weighted decay: per-weight shrink scaled by '
                    '1/sqrt(update count), so rare-pattern weights (overfit-prone) '
                    'decay harder than common ones. 0 = off. Needs no extra data; '
                    'update counts come from the materialized batch indices. '
                    'Prints coverage stats at end.')
    ap.add_argument('--freq-weight', type=float, default=0.0,
                    help='transposition-frequency loss weight: rows whose active '
                    'patterns are common in the dataset get proportionally more '
                    'loss (fit common positions precisely), rare rows keep base '
                    'weight. Linear: mult = 1 + k*relfreq. Sampling unchanged; '
                    'composes with --freq-decay (precision for common, restraint '
                    'for rare). 0 = off.')
    ap.add_argument('--lookahead', action='store_true',
                    help='Lookahead wrapper (k=5, alpha=0.5) around opt.step; '
                    'radam+lookahead ~= Ranger.')
    ap.add_argument('--conv-lr', type=float, default=0.005,
                    help='LR for dense-visit capn/conv families (bags 10-13).')
    ap.add_argument('--noise-w', type=float, default=0.0,
                    help='Explore/exploit slider 1: gaussian noise std added to '
                    'train predictions (target units). >0 explores.')
    ap.add_argument('--smooth', type=float, default=0.0,
                    help='Explore/exploit slider 2: shrink labels toward their '
                    'mean by this fraction (label smoothing).')
    ap.add_argument('--cyclic-T0', type=int, default=0,
                    help='Explore/exploit slider 3 (SGDR): epochs per cycle, '
                    'x --cyclic-mult each restart. 0 = off.')
    ap.add_argument('--cyclic-mult', type=float, default=2.0)
    ap.add_argument('--cyclic-lr-min', type=float, default=1e-5,
                    help='LR floor inside cycles (5-decimal precision).')
    ap.add_argument('--backtrack', action='store_true',
                    help='Backtracking line search: if an epoch overshoots '
                    '(val worse than best), revert weights to best and cut '
                    'LR x --bt-factor. Assesses steps instead of trends.')
    ap.add_argument('--step-lr', action='store_true',
                    help='Per-STEP adaptive LR (not per-epoch): EMA-smoothed '
                    'batch loss with deadband; overshoot -> revert step + '
                    'cut LR, improvement -> raise LR. Reacts in batches.')
    ap.add_argument('--step-up', type=float, default=1.1)
    ap.add_argument('--step-down', type=float, default=0.5)
    ap.add_argument('--step-tol', type=float, default=0.05,
                    help='Deadband: ignore loss moves within this fraction.')
    ap.add_argument('--step-ema', type=float, default=0.01,
                    help='EMA weight for the loss tracker (slow = robust).')
    ap.add_argument('--step-kalman', action='store_true',
                    help='Variance-aware detector (Kalman steady-state family): '
                    'track batch-loss mean AND variance online (self-'
                    'calibrating, no Q/R to tune); overshoot = beyond '
                    'k*sigma, improvement = below -k*sigma. Replaces the '
                    'fixed-deadband EMA test when set.')
    ap.add_argument('--kf-k', type=float, default=2.0,
                    help='Sigma threshold for overshoot/improvement.')
    ap.add_argument('--bt-factor', type=float, default=0.5)
    ap.add_argument('--bt-tol', type=float, default=1e-4,
                    help='Relative worsening that counts as overshoot.')
    ap.add_argument('--wd', type=float, default=1e-7)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--dirs', default='0,1,2,3',
                    help='pair directions to train (subset saves lookups, e.g. 0,1)')
    ap.add_argument('--base', default='s13', choices=['s13', 'simple'],
                    help='residual base: s13 (full hand eval) or simple (mat+PST+tempo, tuple-native)')
    ap.add_argument('--target', default='residual', choices=['residual'],
                    help='teacher-score residual (logit/outcome REMOVED S21g: poison)')
    ap.add_argument('--tcol', default='teacher', choices=['sc', 'teacher'],
                    help='score column: relabeled deep teacher (sane cp scale) or '
                    'dag legacy (sc, x2000 rail-prone). Teacher is the default; '
                    'falls back to sc with a warning if the cache lacks it.')
    ap.add_argument('--blend', type=float, default=1.0,
                    help='teacher weight; remainder = dag legacy (echo regularizer)')
    ap.add_argument('--outcome-weight', type=float, default=0.0,
                    help='S25: weight on game outcome (out column -> cp); '
                    'target = (1-ow)*score_target + ow*outcome_cp. '
                    'outcome_cp = (out*2-1)*OUTCOME_SCALE stm-relative.')
    ap.add_argument('--texel', action='store_true',
                    help='Moonbird-style Texel loss: MSE on win-probability '
                    'sigmoid(base+pred) instead of MSE on cp residuals. '
                    'Targets = sigmoid(best_cp). Same tables, same export.')
    ap.add_argument('--texel-k', type=float, default=567.89,
                    help='Elo-logistic scale for the Texel sigmoid.')
    ap.add_argument('--tree-temp', type=float, default=50.0)
    ap.add_argument('--tree-align', type=float, default=0.1)
    ap.add_argument('--tree-steps', type=int, default=2000)
    ap.add_argument('--tree-backup', default='',
                    help='hive-mind mode: train on a tree-dump npz (with edge_p/edge_c). '
                    'Batches = random subtrees; loss = root MSE vs stored backed-up value '
                    '+ align*(static - soft_backup(children))^2. Backprop flows through '
                    'the backup operator: the whole search learns, not one static eval. '
                    'Needs --opt adam (or sgd), --tree-temp, --tree-align, --tree-steps.')
    ap.add_argument('--val-probe', default='',
                    help='search-backed validation: npz with deep teacher labels '
                    '(same schema as cache). Reports mean|pred+base-teacher| per '
                    'epoch = agreement with DEEP search, not static fit. '
                    'Use for model selection instead of val_mse (which never '
                    'predicted a gate). See --agree-select.')
    ap.add_argument('--agree-select', action='store_true',
                    help='select best checkpoint by agreement (search-backed) '
                    'instead of val_mse (static). Needs --val-probe.')
    ap.add_argument('--out', default='data/nnue/tupS20.tup')
    ap.add_argument('--init', default='',
                    help='Warm start from a .tup (PBT exploit step): loads '
                    'matching sections, missing stay zero-init.')
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    z = np.load(a.cache)
    occ0 = z['occ0'].astype(np.uint64)
    occ1 = z['occ1'].astype(np.uint64)
    blk = z['blk'].astype(np.uint64)
    stm = z['stm'].astype(np.uint8)
    # S26 flywheel: sanitise teacher spikes BEFORE clip/blend. label emits
    # mate scores as +/-99999 and one INT_MIN overflow row; clipping alone
    # leaves 1% rails that dominate sparse-SGD grads. Rows with
    # |teacher| > MATE_CAP are dropped (not clipped): they are search
    # artifacts, and cache12M has 12.4M rows to spare.
    MATE_CAP = 5000.0
    raw_t = z['teacher'].astype(np.float64) if 'teacher' in z else None
    keep = np.ones(len(occ0), dtype=bool)
    if raw_t is not None:
        bad = ~np.isfinite(raw_t) | (np.abs(raw_t) > MATE_CAP)
        keep = ~bad
        if bad.any():
            print('mate-spike filter: dropping %d/%d rows (|teacher|>%.0f)'
                  % (int(bad.sum()), len(raw_t), MATE_CAP), flush=True)
            occ0, occ1, blk, stm = occ0[~bad], occ1[~bad], blk[~bad], stm[~bad]
            z_teacher = raw_t[~bad]
            z_out = (z['out'].astype(np.float64) if 'out' in z
                     else np.zeros(len(raw_t)))[~bad]
            z_sc = (z['sc'].astype(np.float64) if 'sc' in z else np.zeros(len(raw_t)))[~bad]
        else:
            z_teacher = raw_t
            z_out = (z['out'].astype(np.float64) if 'out' in z
                     else np.zeros(len(raw_t)))
            z_sc = (z['sc'].astype(np.float64) if 'sc' in z else np.zeros(len(raw_t)))
    else:
        z_teacher = None
        z_out = (z['out'].astype(np.float64) if 'out' in z
                 else np.zeros(len(occ0)))
        z_sc = (z['sc'].astype(np.float64) if 'sc' in z else np.zeros(len(occ0)))
    n = len(occ0)
    print('rows %d' % n, flush=True)
    # S22: teacher column switch — 'teacher' (our relabeled deep scores) vs
    # 'sc' (dag legacy scores, x2000 rail-prone). Same rows/recipe isolates
    # TEACHER effect. Teacher default; sc only on explicit request or when
    # the cache has no teacher column (warn + fall back, never crash).
    assert a.tcol in ('sc', 'teacher')
    if a.tcol == 'teacher' and z_teacher is None:
        print('tcol=teacher requested but cache has no teacher column: '
              'falling back to sc', flush=True)
        a.tcol = 'sc'
    if a.tcol == 'teacher':
        prime = np.clip(z_teacher * 1.0, -1500.0, 1500.0)
    else:
        prime = np.clip(z['sc'].astype(np.float64) * 2000.0, -1500.0, 1500.0)
        prime = prime[:n] if len(prime) != n else prime
    # S23 blend: target = blend*prime + (1-blend)*dag-legacy. blend=1.0 is
    # pure champ (status quo); lower = retain diverse legacy signal as
    # echo-chamber regularizer. Legacy = sc column (dag scores, stm-rel).
    if abs(a.blend - 1.0) > 1e-9:
        assert z_teacher is not None and a.tcol == 'teacher', 'blend needs teacher tcol + sc legacy'
        legacy = np.clip(z_sc * 2000.0, -1500.0, 1500.0)
        score_cp = (a.blend * prime + (1.0 - a.blend) * legacy).astype(np.float64)
    else:
        score_cp = prime.astype(np.float64)
    print('teacher column: %s blend %.2f' % (a.tcol, a.blend), flush=True)
    d_out = z_out
    # S25 REAL outcome sweep: outcome_cp is stm-relative (out column already
    # stored stm-relative per datagen: 1 = stm won, 0 = stm lost, 0.5 = draw).
    # OUTCOME_SCALE maps [0,1] -> [-S,+S] cp. Sweep --outcome-weight 0..1 with
    # --tcol teacher --blend 1.0 to get pure teacher-vs-outcome (Moonbird recipe).
    OUTCOME_SCALE = 1000.0
    outcome_cp = np.clip((d_out * 2.0 - 1.0) * OUTCOME_SCALE, -1500.0, 1500.0)
    print('outcome stats: mean %+.3f frac0 %.3f frac05 %.3f frac1 %.3f -> cp mean %+.1f std %.1f'
          % (d_out.mean(), (d_out == 0).mean(), (d_out == 0.5).mean(),
             (d_out == 1).mean(), outcome_cp.mean(), outcome_cp.std()), flush=True)
    ow = float(a.outcome_weight)
    assert 0.0 <= ow <= 1.0, 'outcome-weight must be in [0,1]'
    if ow > 0:
        best_cp = ((1.0 - ow) * score_cp + ow * outcome_cp).astype(np.float32)
    else:
        best_cp = score_cp.astype(np.float32)
    print('outcome-weight %.2f' % ow, flush=True)
    if a.rows:
        sel = np.random.choice(n, min(a.rows, n), replace=False)
        occ0, occ1, blk, stm, best_cp = occ0[sel], occ1[sel], blk[sel], stm[sel], best_cp[sel]
        n = len(occ0)
        print('rows subsample %d' % n, flush=True)

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
    t_feat = time.time() - t0
    # S28 STREAMING (12M OOM fix, replaces S27 int32): index arrays are NOT
    # materialised. Block features are int8 (36B/row), index math happens
    # per-batch inside the epoch loop. RAM: feats 0.4GB + cache-mmap 0 +
    # model/tensors small. Same math, same recipe, fits in 16GB.
    feats8 = feats.astype(np.int8)
    t0 = time.time()
    Q = q2_indices(own, enm, blk)
    EUS, ETH = env_tables(own, enm, blk)
    MUS, MTH = mob_tables(own, enm, blk)
    CUS, CTH = capn_tables(own, enm, blk)
    VUS, VTH = conv_tables(own, enm, blk)
    print('enriched features %.0fs' % (time.time() - t0), flush=True)
    # startpos guards (black a1 + white g7, black stm, no blockers):
    # q2 (5,0) = 16 (own 3rd cell x16), (0,5) = 8 (foe 2nd x4)
    o0 = np.array([np.uint64(1 << 42)])
    o1 = np.array([np.uint64(1 << 6)])
    zb = np.array([np.uint64(0)])
    qsp = q2_indices(o0, o1, zb)
    assert int(qsp[0, 5 * 6 + 0]) == 16 and int(qsp[0, 0 * 6 + 5]) == 8, qsp[0]
    esp, thsp = env_tables(o0, o1, zb)
    assert int(esp[0, 42]) == 42 * 729 and int(thsp[0, 6]) == 6 * 729, (esp[0, 42], thsp[0, 6])
    mspu, mspth = mob_tables(o0, o1, zb)
    # capn startpos guard: no landing converts anything (lone stones far apart)
    cspu, cspth = capn_tables(o0, o1, zb)
    assert int(cspu[0]) == 0 and int(cspth[0]) == 0, (cspu[0], cspth[0])
    # brute-force cross-check on 200 sample rows (independent move loop
    # vs vectorized rings — must agree exactly).
    import random as _rng
    _r = _rng.Random(5)
    for _i in _r.sample(range(len(occ0)), min(200, len(occ0))):
        a0, a1, ab = int(occ0[_i]), int(occ1[_i]), int(blk[_i])
        cells = [3 if (ab >> sq) & 1 else 1 if (a0 >> sq) & 1
                 else 2 if (a1 >> sq) & 1 else 0 for sq in range(49)]
        bu = capn_bruteforce(cells, 1 if stm[_i] == 0 else 2)
        bt = capn_bruteforce(cells, 2 if stm[_i] == 0 else 1)
        assert bu == int(CUS[_i]) and bt == int(CTH[_i]), (_i, bu, bt, CUS[_i], CTH[_i])
    print('capn guards OK (startpos 0/0 + 200-row brute-force parity)', flush=True)
    assert int(mspu[0]) == 2 * 50 + 3 and int(mspth[0]) == 2 * 50 + 3, (mspu[0], mspth[0])
    print('enriched guards OK (q2 16/8, env skip, mob 103/103)', flush=True)
    # conv guard on startpos probe (NOT cache row 0): f7(5) ring dr-outer ->
    # e7(4),g7(6),e6(12),f6(13),g6(20); white at g7 = bit1 -> key 2.
    # a1(42): foe far -> key 0. Them-side: own far -> 0.
    vspu, vspth = conv_tables(o0, o1, zb)
    assert int(vspu[0, 5]) == 2 and int(vspu[0, 42]) == 0, (vspu[0, 5], vspu[0, 42])
    assert int(vspth[0, 5]) == 0, vspth[0, 5]
    print('conv guards OK (f7 key 2)', flush=True)
    del feats, own, enm
    import gc as _gc
    _gc.collect()
    print('features %.0fs RAM feats8 %.2fGB'
          % (t_feat, feats8.nbytes / 1e9), flush=True)
    XA = [np.array([x for x, y in pp], dtype=np.int64)
          for pp in [[(x, y) for (dd, x, y) in PAIRS if dd == d] for d in range(4)]]
    YA = [np.array([y for x, y in pp], dtype=np.int64)
          for pp in [[(x, y) for (dd, x, y) in PAIRS if dd == d] for d in range(4)]]
    AR = np.arange(36, dtype=np.int64)[None, :]
    AR2 = np.arange(36, dtype=np.int64)[None, :] * 256
    # Per-anchor pairs (the missing Moonbird feature): all 80 pairs in
    # H/V/D1/D2 order (matches Rust PAIR_BLOCKS), absolute index.
    PAX = np.array([x for (dd, x, y) in PAIRS], dtype=np.int64)
    PAY = np.array([y for (dd, x, y) in PAIRS], dtype=np.int64)
    PAOFF = np.arange(80, dtype=np.int64)[None, :] * 6561
    assert len(PAIRS) == 80 and [sum(1 for p in PAIRS if p[0] == d) for d in range(4)] == [24, 24, 16, 16]
    assert (0, 30, 32) in PAIRS  # H ((5,0),(5,2)) at global position 20

    # S21g lesson (OUT-COLUMN-POISON, LEDGER): dag sum_outcome is NOT a game
    # result (proof-search visit values, exotic perspective; white rows mean
    # 0.75). Blending it produced a -400-Elo table. Teacher best_score
    # residuals ONLY — never touch `out` again.
    LN10K = 2.302585093 / float(a.texel_k)
    if a.texel:
        plab = 1.0 / (1.0 + np.power(10.0, -np.clip(best_cp, -1500.0, 1500.0) / float(a.texel_k)))
        y_full = plab.astype(np.float32)
        print('texel targets: winprob mean %.3f (k=%.2f)' % (y_full.mean(), a.texel_k), flush=True)
        resid = None
    else:
        resid = (best_cp - base).astype(np.float32)
        print('residual mean %+.1f std %.1f' % (resid.mean(), resid.std()), flush=True)
        y_full = None
    if a.smooth > 0:
        if a.texel:
            ym = float(y_full.mean())
            y_full = ((1.0 - a.smooth) * y_full + a.smooth * ym).astype(np.float32)
        else:
            ym = float(resid.mean())
            resid = ((1.0 - a.smooth) * resid + a.smooth * ym).astype(np.float32)
        print('smoothing %.2f toward %.3f' % (a.smooth, ym), flush=True)
    del best_cp
    _gc.collect()
    # S21f: trust high-visit rows (wsum weights, mean-normalized). Absent in
    # old caches -> uniform weights (backward compatible).
    if 'wsum' in z:
        sw = z['wsum'].astype(np.float64)[keep]
        if a.rows:
            sw = sw[sel]
        sw = np.clip(sw, 0, None)
        sw = sw / max(sw.mean(), 1e-9)
        print('wsum weights: mean 1.0 max %.1f frac>2x %.3f' % (sw.max(), (sw > 2).mean()), flush=True)
        samp_w = torch.from_numpy(sw.astype(np.float32))
        del sw
    else:
        samp_w = torch.ones(n)
    _gc.collect()

    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n * 0.95)
    tr, va = idx[:cut], idx[cut:]
    del idx
    if a.texel:
        ytr, yva = torch.from_numpy(y_full[tr]), torch.from_numpy(y_full[va])
        base_tr = torch.from_numpy(base[tr].astype(np.float32))
        base_va = torch.from_numpy(base[va].astype(np.float32))
        del base, y_full
    else:
        ytr, yva = torch.from_numpy(resid[tr]), torch.from_numpy(resid[va])
        del base, resid
    Xtr8, Xva8 = feats8[tr], feats8[va]
    Qtr, Qva = Q[tr], Q[va]
    EUtr, EUva = EUS[tr], EUS[va]
    ETtr, ETva = ETH[tr], ETH[va]
    MUtr, MUva = MUS[tr], MUS[va]
    MTtr, MTva = MTH[tr], MTH[va]
    CUtr, CUva = CUS[tr], CUS[va]
    CTtr, CTva = CTH[tr], CTH[va]
    del CUS, CTH
    VUtr, VUva = VUS[tr], VUS[va]
    VTtr, VTva = VTH[tr], VTH[va]
    del VUS, VTH
    _gc.collect()
    del feats8, Q, EUS, ETH, MUS, MTH
    # Materialized flat indices (S50b speedup): one-time concat, per-batch
    # slicing (views, no recompute). Same values batch_idx built per batch.
    active = [int(x) for x in a.dirs.split(',')]
    assert set(active) <= {0, 1, 2, 3} and active, a.dirs
    SP_OFF = {'s': 0, 'H': 2916, 'V': 9477, 'D1': 16038, 'D2': 22599,
              'q2': 29160, 'envus': 38376, 'envthem': 74097,
              'mobus': 109818, 'mobthem': 112318, 'pa': 114818}
    SP_N = 114818 + 80 * 6561  # 639698
    DE_OFF = {'capnus': 0, 'capnthem': 9, 'convus': 18, 'convthem': 274}
    DE_N = 274 + 256  # 530
    SP_DIR = {0: 'H', 1: 'V', 2: 'D1', 3: 'D2'}
    def materialize(Fb, Qb, EUb, ETb, MUb, MTb, CUb, CTb, VUb, VTb):
        F = Fb.astype(np.int64)
        sp = [(AR * 81 + F)]
        for d in active:
            sp.append((SP_OFF[SP_DIR[d]] + F[:, XA[d]] + 81 * F[:, YA[d]]).reshape(F.shape[0], -1))
        sp.append((AR2 + Qb.astype(np.int64)))
        sp.append((SP_OFF['envus'] + EUb.astype(np.int64)))
        sp.append((SP_OFF['envthem'] + ETb.astype(np.int64)))
        sp.append(((SP_OFF['mobus'] + MUb.astype(np.int64))).reshape(F.shape[0], -1))
        sp.append(((SP_OFF['mobthem'] + MTb.astype(np.int64))).reshape(F.shape[0], -1))
        sp.append((SP_OFF['pa'] + PAOFF + F[:, PAX] + 81 * F[:, PAY]))
        de = [((DE_OFF['capnus'] + CUb.astype(np.int64))).reshape(F.shape[0], -1),
              ((DE_OFF['capnthem'] + CTb.astype(np.int64))).reshape(F.shape[0], -1),
              (DE_OFF['convus'] + VUb.astype(np.int64)),
              (DE_OFF['convthem'] + VTb.astype(np.int64))]
        SP = np.concatenate(sp, axis=1).astype(np.int32)
        DE = np.concatenate(de, axis=1).astype(np.int32)
        return SP, DE

    t0 = time.time()
    SPtr, DEtr = materialize(Xtr8, Qtr, EUtr, ETtr, MUtr, MTtr, CUtr, CTtr, VUtr, VTtr)
    # Transposition-frequency weights (--freq-weight): one counting pass over
    # the materialized sparse indices; per-row multiplier 1 + k*relfreq where
    # relfreq = mean index-count over the row's active patterns, normalized.
    fmult = None
    if a.freq_weight > 0:
        cnt = np.bincount(SPtr.reshape(-1).astype(np.int64), minlength=SP_N).astype(np.float64)
        rf = cnt[SPtr].mean(axis=1)
        rf /= rf.max() if rf.max() > 0 else 1.0
        fmult = (1.0 + a.freq_weight * rf).astype(np.float32)
    SPva, DEva = materialize(Xva8, Qva, EUva, ETva, MUva, MTva, CUva, CTva, VUva, VTva)
    # Search-backed validation probe: fixed positions + deep-teacher labels,
    # materialized once. Per-epoch agreement = mean|pred+base-teacher|.
    # No live search at train time (labels precomputed); the metric still
    # measures search-relevant fit instead of static fit.
    AG = None
    if a.val_probe:
        _pz = np.load(a.val_probe)
        _po = _pz['occ0'].astype(np.uint64)
        _pe = _pz['occ1'].astype(np.uint64)
        _pb = _pz['blk'].astype(np.uint64)
        _ps = _pz['stm'].astype(np.uint8)
        _is_b = (_ps == 0)
        _own = np.where(_is_b, _po, _pe)
        _enm = np.where(_is_b, _pe, _po)
        _F = block_indices(_own, _enm)
        _AG = {'teach': _pz['teacher'].astype(np.float32)}
        _AG['base'] = base_eval_stm(_po, _pe, _pb, _ps).astype(np.float32)
        SPp, DEp = materialize(_F,
                               q2_indices(_own, _enm, _pb),
                               *env_tables(_own, _enm, _pb),
                               *mob_tables(_own, _enm, _pb),
                               *capn_tables(_own, _enm, _pb),
                               *conv_tables(_own, _enm, _pb))
        _AG['SP'], _AG['DE'] = SPp, DEp
        del _F, SPp, DEp
        AG = _AG
        print('val probe: %d rows deep-teacher mean %+.1f' % (
            len(_AG['teach']), float(_AG['teach'].mean())), flush=True)
    print('materialized flat %.0fs RAM %.2fGB' % (time.time() - t0, (SPtr.nbytes + DEtr.nbytes + SPva.nbytes + DEva.nbytes) / 1e9), flush=True)
    del Xtr8, Xva8, Qtr, Qva, EUtr, EUva, ETtr, ETva, MUtr, MUva, MTtr, MTva
    del CUtr, CUva, CTtr, CTva, VUtr, VUva, VTtr, VTva
    _gc.collect()
    wtr, wva = samp_w[tr], samp_w[va]
    del samp_w
    _gc.collect()
    if fmult is not None:
        # Combined focus weight = regret-reward x repetition, bounded by
        # construction (wcap x (1+k)): precision where common AND regretted.
        wq = wtr if isinstance(wtr, np.ndarray) else wtr.numpy()
        eff = wq * fmult
        c = np.corrcoef(wq, fmult)[0, 1]
        print('freq-weight k=%.2f: mult mean %.3f max %.3f | regret-freq corr %+.3f | '
              'effective wmean %.3f wmax %.2f (cap %.1f)' % (
              a.freq_weight, fmult.mean(), fmult.max(), c,
              eff.mean(), eff.max(), 10.0 * (1.0 + a.freq_weight)), flush=True)

    # 1 single bag + 4 pair bags + scalar bias. Index-0 entries are
    # constrained to exactly 0 after every step (empty patterns contribute
    # nothing, so the engine can SKIP them — bit-identical forward).
    # S38 enriched: + quat bag (36x256) + env us/them (49x729 each) + mob
    # us/them (2500 each). Env/mob have separate us/them tables (no sign
    # sharing — SGD learns color asymmetry); empty-ring (sq*729) and
    # zero-count (phase*50) rows constrained 0 and skipped in forward.
    # FUSED bags (S50 speedup: 15 sparse backward ops -> 2). Same math:
    # one bag per visit-density group, per-family base offsets added to the
    # same local indices as before. Sparse group (base lr): singles, pairs,
    # q2, env, mob, pa. Dense group (conv-lr): capn, conv.
    # (Layout consts live above materialize(); bags only need totals here.)
    bags = torch.nn.ModuleList(
        [torch.nn.EmbeddingBag(SP_N, 1, mode='sum', sparse=True),
         torch.nn.EmbeddingBag(DE_N, 1, mode='sum', sparse=True)]
    )
    for bg in bags:
        torch.nn.init.zeros_(bg.weight)
    bias = torch.nn.Parameter(torch.zeros(1))
    params = list(bags.parameters()) + [bias]
    # NOTE: torch SGD weight_decay is dense+sparse incompatible; L2 is added
    # to the loss manually below (works on the dense weight tensors).
    # Dense-visit families (conv 49 lookups/row, capn 1/9) diverge at the
    # sparse-family lr — own param group at --conv-lr.
    def make_opt(a, bags, bias):
        groups = [
            {'params': [bags[0].weight, bias]},
            {'params': [bags[1].weight], 'lr': a.conv_lr},
        ]
        if a.opt == 'sgd':
            return torch.optim.SGD(groups, lr=a.lr)
        lr = a.opt_lr or 0.001
        if a.opt == 'adam':
            return torch.optim.Adam(groups, lr=lr, weight_decay=0.0)
        if a.opt == 'adamw':
            return torch.optim.AdamW(groups, lr=lr, weight_decay=a.decay)
        if a.opt == 'radam':
            return torch.optim.RAdam(groups, lr=lr, weight_decay=a.decay)
        raise ValueError(a.opt)

    opt = make_opt(a, bags, bias)
    # Rarity tracking for --freq-decay: per-weight update counts from the
    # materialized sparse indices (zero extra search/data cost).
    fdec = float(a.freq_decay)
    freq = None
    if fdec > 0:
        freq = [torch.zeros(SP_N), torch.zeros(DE_N)]

    class Lookahead:
        # Slow-weights wrapper: every k steps, slow += alpha*(fast - slow),
        # fast <- slow. Ranger ~= RAdam + this.
        def __init__(self, base, k=5, alpha=0.5):
            self.base = base
            self.k = k
            self.alpha = alpha
            self.step_no = 0
            self.slow = [p.detach().clone() for g in base.param_groups for p in g['params']]

        def zero_grad(self):
            self.base.zero_grad()

        def step(self):
            self.base.step()
            self.step_no += 1
            if self.step_no % self.k:
                return
            idx = 0
            with torch.no_grad():
                for g in self.base.param_groups:
                    for p in g['params']:
                        s = self.slow[idx]
                        idx += 1
                        s.add_(self.alpha * (p.detach() - s))
                        p.copy_(s)

        @property
        def param_groups(self):
            return self.base.param_groups

    if a.lookahead:
        opt = Lookahead(opt)

    def zero_constrained():
        with torch.no_grad():
            sp, de = bags[0].weight.data, bags[1].weight.data
            sp[0].zero_()
            for d in range(4):
                sp[SP_OFF[['H', 'V', 'D1', 'D2'][d]]].zero_()
            sp[SP_OFF['q2']].zero_()
            for sq in range(49):
                sp[SP_OFF['envus'] + sq * 729].zero_()
                sp[SP_OFF['envthem'] + sq * 729].zero_()
            for ph in range(50):
                sp[SP_OFF['mobus'] + ph * 50].zero_()
                sp[SP_OFF['mobthem'] + ph * 50].zero_()
            for pi in range(80):
                sp[SP_OFF['pa'] + pi * 6561].zero_()
            de[DE_OFF['capnus']].zero_()
            de[DE_OFF['capnthem']].zero_()
            de[DE_OFF['convus']].zero_()
            de[DE_OFF['convthem']].zero_()

    if a.init:
        b0, ws0, wps0, rest0, cc0, pa0 = load_tup(a.init)
        with torch.no_grad():
            W, D = bags[0].weight, bags[1].weight
            bias.copy_(torch.tensor([b0]))
            W[SP_OFF['s']:SP_OFF['s'] + 36 * 81].copy_(
                torch.from_numpy(ws0.reshape(-1, 1)))
            for d, nm in enumerate(['H', 'V', 'D1', 'D2']):
                W[SP_OFF[nm]:SP_OFF[nm] + 6561].copy_(
                    torch.from_numpy(wps0[d].reshape(-1, 1)))
            if rest0 is not None:
                wq, weu, wet, wmu, wmt = rest0
                W[SP_OFF['q2']:SP_OFF['q2'] + 36 * 256].copy_(
                    torch.from_numpy(wq.reshape(-1, 1)))
                W[SP_OFF['envus']:SP_OFF['envus'] + 35721].copy_(
                    torch.from_numpy(weu.reshape(-1, 1)))
                W[SP_OFF['envthem']:SP_OFF['envthem'] + 35721].copy_(
                    torch.from_numpy(wet.reshape(-1, 1)))
                W[SP_OFF['mobus']:SP_OFF['mobus'] + 2500].copy_(
                    torch.from_numpy(wmu.reshape(-1, 1)))
                W[SP_OFF['mobthem']:SP_OFF['mobthem'] + 2500].copy_(
                    torch.from_numpy(wmt.reshape(-1, 1)))
            if cc0 is not None:
                wcu, wct, wvu, wvt = cc0
                D[DE_OFF['capnus']:DE_OFF['capnus'] + 9].copy_(
                    torch.from_numpy(wcu.reshape(-1, 1)))
                D[DE_OFF['capnthem']:DE_OFF['capnthem'] + 9].copy_(
                    torch.from_numpy(wct.reshape(-1, 1)))
                D[DE_OFF['convus']:DE_OFF['convus'] + 256].copy_(
                    torch.from_numpy(wvu.reshape(-1, 1)))
                D[DE_OFF['convthem']:DE_OFF['convthem'] + 256].copy_(
                    torch.from_numpy(wvt.reshape(-1, 1)))
            if pa0 is not None:
                W[SP_OFF['pa']:SP_OFF['pa'] + 80 * 6561].copy_(
                    torch.from_numpy(pa0.reshape(-1, 1)))
        zero_constrained()
        print('warm start from %s' % a.init, flush=True)

    # S28 streaming index math: per-batch int8 feats -> int64 EmbeddingBag
    # indices. Same values as the old materialised off/pidx
    # (off = AR*81 + feat, pidx[d] = feat[xa] + 81 * feat[ya]).
    # S37 FIX: S28 wrote AR + feat (dropped the *81), hashing 2916 singles
    # rows into 117 — every training since used broken singles in Rust.
    # S38 enriched: Q (36x256 quat, AR2 stride) + env us/them (49x729,
    # absolute indices) + mob us/them (phase*50+cnt). All plain sums.

    SP_DIR = {0: 'H', 1: 'V', 2: 'D1', 3: 'D2'}

    def batch_idx(SPb, DEb):
        # Slices of materialized int32 flats (views, no compute). 2D
        # fixed-length rows need no offsets (each row = one bag).
        Xsp = torch.from_numpy(SPb.astype(np.int64))
        Xde = torch.from_numpy(DEb.astype(np.int64))
        return Xsp, Xde

    def forward(Xsp, Xde):
        pred = bags[0](Xsp).squeeze(1) + bags[1](Xde).squeeze(1) + bias
        return pred

    def dump_tup(path):
        W = bags[0].weight.detach().squeeze(1).numpy()
        D = bags[1].weight.detach().squeeze(1).numpy()
        ws = W[SP_OFF['s']:SP_OFF['s'] + 36 * 81].reshape(36, 81)
        wp = [W[SP_OFF[nm]:SP_OFF[nm] + 6561].reshape(6561)
              for nm in ['H', 'V', 'D1', 'D2']]
        wq = W[SP_OFF['q2']:SP_OFF['q2'] + 36 * 256].reshape(36 * 256)
        weu = W[SP_OFF['envus']:SP_OFF['envus'] + 35721]
        wet = W[SP_OFF['envthem']:SP_OFF['envthem'] + 35721]
        wmu = W[SP_OFF['mobus']:SP_OFF['mobus'] + 2500]
        wmt = W[SP_OFF['mobthem']:SP_OFF['mobthem'] + 2500]
        assert ws[0, 0] == 0 and all(p[0] == 0 for p in wp) and wq[0] == 0, 'index-0 constraint broken'
        assert all(weu[sq * 729] == 0 and wet[sq * 729] == 0 for sq in range(49)), 'env-0 broken'
        assert all(wmu[ph * 50] == 0 and wmt[ph * 50] == 0 for ph in range(50)), 'mob-0 broken'
        wcu = D[DE_OFF['capnus']:DE_OFF['capnus'] + 9]
        wct = D[DE_OFF['capnthem']:DE_OFF['capnthem'] + 9]
        wvu = D[DE_OFF['convus']:DE_OFF['convus'] + 256]
        wvt = D[DE_OFF['convthem']:DE_OFF['convthem'] + 256]
        assert wcu[0] == 0 and wct[0] == 0 and wvu[0] == 0 and wvt[0] == 0, 'capn/conv-0 broken'
        wpa = W[SP_OFF['pa']:SP_OFF['pa'] + 80 * 6561]
        assert all(wpa[pi * 6561] == 0 for pi in range(80)), 'pa-0 broken'
        allw = np.concatenate([ws.reshape(-1)] + [p.reshape(-1) for p in wp]
                              + [wq, weu, wet, wmu, wmt, wcu, wct, wvu, wvt, wpa])
        print('weight spread: max|w| %.1f mean|w| %.2f frac_nonzero %.3f bias %+.1f'
              % (np.abs(allw).max(), np.abs(allw).mean(), (allw != 0).mean(), float(bias.item())),
              flush=True)
        if freq is not None:
            for bi, (nm, nn) in enumerate((('sparse', SP_N), ('dense', DE_N))):
                fr = freq[bi]
                touched = float((fr > 0).sum()) / nn
                q = torch.quantile(fr[fr > 0] if touched > 0 else fr,
                                   torch.tensor([0.1, 0.5, 0.9])).tolist()
                print('coverage %s: touched %.3f update-count q10/med/q90 %.0f/%.0f/%.0f'
                      % (nm, touched, q[0], q[1], q[2]), flush=True)
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
        # TUP5: TUP4 prefix + i16 LE sections q2 (9216), envus (35721),
        # envthem, mobus (2500), mobthem. i16: enriched families exceed i8.
        # TUP6: TUP5 + capn us/them (9 each) + conv us/them (256 each).
        q16 = lambda arr: struct.pack('<%dh' % len(arr),
                                      *[max(-32767, min(32767, int(round(v)))) for v in arr])
        payload += q16(wq) + q16(weu) + q16(wet) + q16(wmu) + q16(wmt)
        payload += q16(wcu) + q16(wct) + q16(wvu) + q16(wvt)
        # TUP7: TUP6 + per-anchor pairs (80x6561 i8, position order H/V/D1/D2
        # matching Rust PAIR_BLOCKS).
        q8 = lambda arr: bytes(bytearray([(max(-127, min(127, int(round(v)))) & 0xFF) for v in arr]))
        payload += q8(wpa)
        with open(path, 'wb') as f:
            f.write(b'TUP7' + payload)
        import os
        print('wrote %s %.1fKB dirs %s TUP7' % (path, os.path.getsize(path) / 1024, a.dirs), flush=True)


    bs = 4096
    # S26 flywheel: plateau stop on RELATIVE val improvement. Your call of 4
    # digits (1e-4) is right for this data: 12.4M rows x sparse updates =
    # val is glassy-smooth (no minibatch noise at 620k val rows), so a 5th
    # digit is noise-chasing; 4 digits stops at true saturation. Needs
    # --plateau 1e-4 --max-epochs N --ckpt template; plain --epochs is
    # unchanged (plateau 0 = off).
    max_ep = a.max_epochs if a.plateau > 0 else a.epochs
    prev_val, best_val, best_ep, stale = None, float('inf'), 0, 0
    best_agree = [float('inf')]
    step_ema, step_snap, step_reverts, step_bad = None, None, 0, 0
    av_mean, av_var = None, None
    batch_best = float('inf')
    # Cyclic SGDR state (explore/exploit slider 3): cosine lr within a
    # cycle, restart (rewind to best + lr back to max) at cycle end.
    cyc_on = a.cyclic_T0 > 0
    cyc_T, cyc_Ti = 0, max(a.cyclic_T0, 1)
    lr_max0, conv_max0 = float(a.lr), float(a.conv_lr)
    lr_ratio = conv_max0 / max(lr_max0, 1e-12)
    best_state = None
    import shutil
    _gc.collect()
    def run_tree_backup():
        # Hive-mind training: batches are random subtrees; the loss anchors
        # roots at stored deep backed-up values and pulls every node toward
        # its CURRENT-net soft backup (detached fixed point). Gradients flow
        # through the backup operator: blame distributes over the subtree.
        import random as _rng
        z = np.load(a.tree_backup)
        occ0 = z['occ0'].astype(np.uint64); occ1 = z['occ1'].astype(np.uint64)
        blk = z['blk'].astype(np.uint64); stm = z['stm'].astype(np.uint8)
        backed_all = z['teacher'].astype(np.float32)
        w_all = z['wsum'].astype(np.float32) if 'wsum' in z else np.ones(len(occ0), np.float32)
        ep_ = z['edge_p'].astype(np.int64); ec_ = z['edge_c'].astype(np.int64)
        n = len(occ0)
        # Cap memory: random 1.5M-row subset, edges remapped into it.
        if n > 1500000:
            keep = np.sort(np.random.RandomState(a.seed).choice(n, 1500000, replace=False))
            remap = np.full(n, -1, dtype=np.int64)
            remap[keep] = np.arange(len(keep), dtype=np.int64)
            ep2 = remap[ep_]
            ec2 = remap[ec_]
            ok = (ep2 >= 0) & (ec2 >= 0)
            ep_, ec_ = ep2[ok], ec2[ok]
            occ0, occ1, blk, stm = occ0[keep], occ1[keep], blk[keep], stm[keep]
            backed_all, w_all = backed_all[keep], w_all[keep]
            n = len(occ0)
            print('tree subsample: %d rows kept' % n, flush=True)
        kids = {}
        for p, c in zip(ep_.tolist(), ec_.tolist()):
            if 0 <= p < n and 0 <= c < n:
                kids.setdefault(p, []).append(c)
        for k in kids:
            if len(kids[k]) > 64:
                kids[k] = kids[k][:64]
        print('tree: %d nodes, %d internal, %d edges' % (
            n, len(kids), sum(len(v) for v in kids.values())), flush=True)
        is_b = (stm == 0)
        own = np.where(is_b, occ0, occ1); enm = np.where(is_b, occ1, occ0)
        F = block_indices(own, enm); Q = q2_indices(own, enm, blk)
        EUS, ETH = env_tables(own, enm, blk); MUS, MTH = mob_tables(own, enm, blk)
        CUS, CTH = capn_tables(own, enm, blk); VUS, VTH = conv_tables(own, enm, blk)
        SPt, DEt = materialize(F.astype(np.int64), Q, EUS, ETH, MUS, MTH, CUS, CTH, VUS, VTH)
        del F, Q, EUS, ETH, MUS, MTH, CUS, CTH, VUS, VTH
        _gc.collect()
        base_t = base_eval_stm(occ0, occ1, blk, stm).astype(np.float32)
        SPt = torch.from_numpy(SPt); DEt = torch.from_numpy(DEt)
        base_t = torch.from_numpy(base_t)
        backed_t = torch.from_numpy(backed_all)
        w_t = torch.from_numpy(w_all)
        temp = float(a.tree_temp); al = float(a.tree_align)
        roots = np.array(sorted(kids.keys()), dtype=np.int64)
        if len(roots) == 0:
            roots = np.arange(n)
        rrng = _rng.Random(a.seed)
        t00 = time.time()
        tot = 0.0
        for step in range(1, int(a.tree_steps) + 1):
            rbatch = [int(roots[rrng.randrange(len(roots))]) for _ in range(8)]
            seen = {}; order = []
            stack = list(rbatch)
            while stack and len(order) < 2048:
                u = stack.pop()
                if u in seen:
                    continue
                seen[u] = len(order)
                order.append(u)
                stack.extend(kids.get(u, ()))
            idx = torch.tensor(order, dtype=torch.long)
            opt.zero_grad()
            pred = bags[0](SPt[idx]).squeeze(1) + bags[1](DEt[idx]).squeeze(1) + bias
            val = {u: pred[seen[u]] for u in order}
            for u in reversed(order):
                ch = [c for c in kids.get(u, ()) if c in seen]
                if ch:
                    cs = torch.stack([val[c] for c in ch])
                    val[u] = temp * torch.logsumexp(cs / temp, dim=0)
            is_root = torch.zeros(len(order), dtype=torch.bool)
            for r in rbatch:
                if r in seen:
                    is_root[seen[r]] = True
            lroot = (((pred[is_root] + base_t[idx][is_root] - backed_t[idx][is_root]) ** 2)
                     * w_t[idx][is_root]).mean() if is_root.any() else pred.sum() * 0.0
            with torch.no_grad():
                bv = torch.stack([val[u].detach() for u in order])
            lalign = ((pred - bv) ** 2).mean()
            # Manual L2 like the flat path: doubles as the dense-gradient source
            # Adam needs (pure sparse grads crash it).
            l2 = sum((bg.weight ** 2).sum() for bg in bags)
            (lroot + al * lalign + a.wd * l2).backward()
            opt.step()
            zero_constrained()
            tot += float(lroot.detach() + al * lalign.detach())
            if step % 200 == 0:
                print('treestep %d loss %.1f (root %.1f align %.1f)' % (
                    step, tot / 200, float(lroot.detach()), float(lalign.detach())), flush=True)
                tot = 0.0
        print('tree training done %.0fs' % (time.time() - t00), flush=True)

    ntr = len(SPtr)
    if a.tree_backup:
        run_tree_backup()
        dump_tup(a.out)
        return
    for ep in range(1, max_ep + 1):
        t0 = time.time()
        for bg in bags:
            bg.train()
        perm = torch.randperm(ntr)
        tot, cnt = 0.0, 0
        for i in range(0, ntr, bs):
            b = perm[i:i + bs]
            bn = b.numpy()
            Xsp, Xde = batch_idx(SPtr[bn], DEtr[bn])
            if freq is not None:
                with torch.no_grad():
                    freq[0] += torch.bincount(Xsp.flatten(), minlength=SP_N).float()
                    freq[1] += torch.bincount(Xde.flatten(), minlength=DE_N).float()
            opt.zero_grad()
            pred = forward(Xsp, Xde)
            if a.noise_w > 0:
                pred = pred + torch.randn_like(pred) * a.noise_w
            l2 = sum((bg.weight ** 2).sum() for bg in bags)
            wb = wtr[b]
            if fmult is not None:
                wb = wb * torch.from_numpy(fmult[bn]).to(wb.dtype)
            if a.texel:
                pw = torch.sigmoid((base_tr[b] + pred) * LN10K)
                loss = (((pw - ytr[b]) ** 2) * wb).mean() + a.wd * l2
            else:
                loss = ((pred - ytr[b]) ** 2 * wb).mean() + a.wd * l2
            loss.backward()
            opt.step()
            zero_constrained()
            if freq is not None:
                # Rare-pattern weights shrink harder (overfit-prone), common
                # ones barely. Zero-init untouched weights are unaffected.
                with torch.no_grad():
                    lr_now = opt.param_groups[0].get('lr', 0.0)
                    for bi, bg in enumerate(bags):
                        fr = freq[bi]
                        fmax = float(fr.max()) if float(fr.max()) > 0 else 1.0
                        rarity = torch.sqrt(fmax / (fr + 1.0))
                        shrink = (lr_now * fdec * rarity).clamp(max=0.05)
                        bg.weight.mul_(1.0 - shrink.view(-1, 1))
            if a.step_lr:
                # Per-step bold driver: snapshot BEFORE seeing this step's
                # effect is impossible without extra forwards, so assess with
                # 1-step delay — revert restores the pre-step weights.
                # Detector: EMA band by default, or variance-aware band with
                # --step-kalman (online mean+variance, self-calibrating, no
                # fixed deadband — steady-state Kalman family).
                import math as _math2
                cur = loss.item()
                reverted = False
                if step_ema is None:
                    step_ema = cur
                    av_mean, av_var = cur, abs(cur) * 0.1 + 1.0
                    with torch.no_grad():
                        step_snap = ([bg.weight.detach().clone() for bg in bags],
                                     bias.detach().clone())
                else:
                    if a.step_kalman:
                        sigma = _math2.sqrt(max(av_var, 1e-12))
                        over = cur > av_mean + a.kf_k * sigma
                        under = cur < av_mean - a.kf_k * sigma
                    else:
                        over = cur > step_ema * (1.0 + a.step_tol)
                        under = cur < step_ema * (1.0 - a.step_tol)
                    # Absolute guard: 100x above the best batch ever seen is
                    # never noise (EMA would chase the explosion blind).
                    if cur > 100.0 * batch_best:
                        over = True
                        step_bad = 99
                    # Persistence-2: single noisy batches don't trigger;
                    # sustained overshoot does (then revert + cut LR).
                    step_bad = step_bad + 1 if over else 0
                    reverted = False
                    if over and step_bad >= 2 and step_snap is not None:
                        with torch.no_grad():
                            for bg, w in zip(bags, step_snap[0]):
                                bg.weight.copy_(w)
                            bias.copy_(step_snap[1])
                        zero_constrained()
                        for pg in opt.param_groups:
                            pg['lr'] = max(pg['lr'] * a.step_down, a.cyclic_lr_min)
                        step_reverts += 1
                        step_bad = 0
                        reverted = True
                    elif under:
                        for pg in opt.param_groups:
                            pg['lr'] = min(pg['lr'] * a.step_up, lr_max0 * 10.0)
                if cur < batch_best:
                    batch_best = cur
                if not reverted:
                    step_ema = (1.0 - a.step_ema) * step_ema + a.step_ema * cur
                if a.step_kalman:
                    # Frozen on revert batches: don't let explosions poison
                    # the baseline the detector measures against.
                    if not reverted:
                        k = a.step_ema
                        d = cur - av_mean
                        av_mean = av_mean + k * d
                        av_var = (1.0 - k) * av_var + k * d * d
                with torch.no_grad():
                    step_snap = ([bg.weight.detach().clone() for bg in bags],
                                 bias.detach().clone())
            tot += loss.item() * len(b)
            cnt += len(b)
        for bg in bags:
            bg.eval()
        with torch.no_grad():
            # S28: val streamed in 100k blocks (full Xva int64 = 2.5GB).
            totv, cntv = 0.0, 0
            for j in range(0, len(SPva), 100000):
                nv = min(100000, len(SPva) - j)
                Xsp, Xde = batch_idx(SPva[j:j + nv], DEva[j:j + nv])
                pv = forward(Xsp, Xde)
                wv = wva[j:j + nv]
                if a.texel:
                    pwv = torch.sigmoid((base_va[j:j + nv] + pv) * LN10K)
                    totv += ((((pwv - yva[j:j + nv]) ** 2) * wv).sum().item())
                else:
                    totv += ((((pv - yva[j:j + nv]) ** 2) * wv).sum().item())
                cntv += nv
            vmse = totv / max(cntv, 1)
        # Search-backed validation: agreement of CURRENT net with deep search
        # on the fixed probe set (precomputed labels, no live search).
        # This is the number that should predict gates; val_mse never did.
        agree_msg = ''
        if AG is not None:
            with torch.no_grad():
                to = 0.0
                cn = 0
                SPp, DEp = AG['SP'], AG['DE']
                for j in range(0, len(SPp), 2000):
                    e = min(len(SPp), j + 2000)
                    Xsp = torch.from_numpy(SPp[j:e].astype(np.int64))
                    Xde = torch.from_numpy(DEp[j:e].astype(np.int64))
                    pv = forward(Xsp, Xde)
                    to += torch.abs(pv + torch.from_numpy(
                        AG['base'][j:e]) - torch.from_numpy(
                        AG['teach'][j:e])).sum().item()
                    cn += e - j
                agree = to / max(cn, 1)
            agree_msg = ' agree %.0f' % agree
            if agree < best_agree[0]:
                best_agree[0] = agree
                if a.agree_select:
                    best_val, best_ep = vmse, ep
                    with torch.no_grad():
                        best_state = ([bg.weight.detach().clone() for bg in bags],
                                      bias.detach().clone())
                    if a.ckpt:
                        dump_tup(a.ckpt.format(ep=ep))
                        shutil.copy(a.ckpt.format(ep=ep), a.out)
                    agree_msg += ' AGREE-BEST->out'
                else:
                    agree_msg += ' (best %.0f)' % best_agree[0]
        improv = (prev_val - vmse) / max(prev_val, 1e-9) if prev_val else 1.0
        tag = ''
        if vmse < best_val:
            best_val, best_ep = vmse, ep
            with torch.no_grad():
                best_state = ([bg.weight.detach().clone() for bg in bags],
                              bias.detach().clone())
            if a.ckpt:
                dump_tup(a.ckpt.format(ep=ep))
                shutil.copy(a.ckpt.format(ep=ep), a.out)
                tag = ' BEST->out'
        # Backtracking line search: overshoot -> revert to best + cut LR.
        bt_msg = ''
        if a.backtrack and best_state is not None and vmse > best_val * (1.0 + a.bt_tol):
            with torch.no_grad():
                for bg, w in zip(bags, best_state[0]):
                    bg.weight.copy_(w)
                bias.copy_(best_state[1])
            zero_constrained()
            for pg in opt.param_groups:
                pg['lr'] = max(pg['lr'] * a.bt_factor, a.cyclic_lr_min)
            bt_msg = ' BACKTRACK'
        # Cyclic restart: rewind to best, lr back to max, cycle lengthens.
        cyc_msg = ''
        if cyc_on:
            cyc_T += 1
            if cyc_T >= cyc_Ti and best_state is not None:
                with torch.no_grad():
                    for bg, w in zip(bags, best_state[0]):
                        bg.weight.copy_(w)
                    bias.copy_(best_state[1])
                zero_constrained()
                cyc_T, cyc_Ti = 0, int(cyc_Ti * a.cyclic_mult)
                cyc_msg = ' RESTART(T=%d)' % cyc_Ti
            frac = min(cyc_T / max(cyc_Ti, 1), 1.0)
            import math as _math
            lr_ep = a.cyclic_lr_min + 0.5 * (lr_max0 - a.cyclic_lr_min) * (
                1.0 + _math.cos(_math.pi * frac))
            opt.param_groups[0]['lr'] = lr_ep
            opt.param_groups[1]['lr'] = max(lr_ep * lr_ratio, a.cyclic_lr_min)
        if a.plateau > 0 and prev_val is not None and improv < a.plateau:
            stale += 1
        else:
            stale = 0
        print('epoch %d train_mse %.1f val_mse %.1f%s rel_improv %+.2e bias %+.1f lr %.5f reverts %d (%.0fs)%s%s%s'
              % (ep, tot / cnt, vmse, agree_msg, improv, bias.item(), opt.param_groups[0]['lr'], step_reverts,
                 time.time() - t0, tag, cyc_msg, bt_msg), flush=True)
        step_reverts = 0
        prev_val = best_val if bt_msg else vmse
        if a.plateau > 0 and stale >= 2:
            print('plateau: rel improv < %.0e twice in a row -> stop at ep %d (best ep %d val %.1f)'
                  % (a.plateau, ep, best_ep, best_val), flush=True)
            break
    if not a.ckpt:
        dump_tup(a.out)
    else:
        print('flywheel: best ep %d val %.1f -> %s' % (best_ep, best_val, a.out), flush=True)


if __name__ == '__main__':
    main()
