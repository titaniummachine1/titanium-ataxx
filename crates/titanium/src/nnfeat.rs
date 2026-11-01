//! NNUE input features (exp/nnue). Built from docs/NET_ANALYSIS.md: everything the strong nets (KataGo, Ataxx_CPP,
//! Moonbird) look at sits within 2 squares, and the first thing KataGo infers is stone vulnerability.
//! Every concept those nets showed is a separate GROUP that can be switched off, so ablation decides what stays.
//!
//! Perspective P = side whose accumulator this is ("own" = P's stones, "enemy" = the other). All maths is backwards
//! bitboard set operations; G5 is the only group that loops over candidate moves.
//!
//! G1 square state, one feature per square: idx = G1_OFF + sq * SQ_STATES + code
//!    own stone   : exposed * 9 + empty_neighbours                    (0..17)
//!    enemy stone : 18 + exposed * 9 + empty_neighbours               (18..35)
//!       exposed = an empty neighbour of the stone can be landed on (clone or jump) by the stone's opponent
//!    empty       : 36 + ((reach_own * 3 + reach_enemy) * 5 + min(gain_own,4)) * 5 + min(gain_enemy,4)   (36..260)
//!       reach 0 none / 1 jump only / 2 clone; gain_own = enemy stones around it (what P converts landing there)
//!    blocker     : 261
//! G2 anchor (HalfKP): anchor = P's best attack square = empty square P can land on converting the most enemy stones
//!    (ties: clone first, then lowest square). Every stone within Chebyshev 2 of it:
//!    idx = G2_OFF + (anchor * 49 + stone_sq) * 2 + (0 own / 1 enemy); no capture available -> G2_NONE.
//! G3 stone ring (Moonbird env): every stone: idx = G3_OFF + ((sq * 2 + colour) * 9 + own_nb) * 9 + enemy_nb,
//!    own_nb / enemy_nb counted from the STONE's side.
//! G5 2-ply exchange: every empty square P can clone onto: v = 1 + converted - enemy's best capture afterwards,
//!    clamped to -3..5: idx = G5_OFF + sq * 9 + (v + 3).
//! G6 global (per side, from P's view): reach count (empty squares within 2 of the side's stones) x phase, plus
//!    empties parity x phase: idx = G6_OFF + (side_rel * 8 + phase) * 10 + min(reach / 3, 9), side_rel 0 = P, 1 = enemy;
//!    G6_PAR_OFF + phase * 2 + (empties & 1). phase = empties / 7 (0..7).
//! (G4 = 8-cell pattern pairs stays OUTSIDE the accumulator: the existing tuple net as a linear skip path.)

#[cfg(test)]
use crate::board::{dist_union, jump_union};
use crate::board::{Board, FILE_A, FILE_G, FULL, RING1, SQUARES};

pub const G1: u32 = 1;
pub const G2: u32 = 2;
pub const G3: u32 = 4;
pub const G5: u32 = 16;
pub const G6: u32 = 32;
pub const ALL_GROUPS: u32 = G1 | G2 | G3 | G5 | G6;

pub const SQ_STATES: usize = 262;
pub const BLOCKER_CODE: usize = 261;
pub const G1_OFF: usize = 0;
pub const G2_OFF: usize = G1_OFF + SQUARES * SQ_STATES;
pub const G2_NONE: usize = G2_OFF + SQUARES * SQUARES * 2;
pub const G3_OFF: usize = G2_NONE + 1;
pub const G5_OFF: usize = G3_OFF + SQUARES * 2 * 81;
pub const G6_OFF: usize = G5_OFF + SQUARES * 9;
pub const G6_PAR_OFF: usize = G6_OFF + 2 * 8 * 10;
pub const NUM_FEATS: usize = G6_PAR_OFF + 16;
/// G1 49 + G2 <=24 + G3 <=49 + G5 <=49 + G6 3.
pub const MAX_ACTIVE: usize = 49 + 24 + 49 + 49 + 3;

/// Chebyshev-2 box around each square, excluding the square itself.
const fn build_box2() -> [u64; SQUARES] {
    let mut t = [0u64; SQUARES];
    let mut s = 0;
    while s < SQUARES {
        let (r, c) = ((s / 7) as i32, (s % 7) as i32);
        let mut m = 0u64;
        let mut q = 0;
        while q < SQUARES {
            let (rr, cc) = ((q / 7) as i32, (q % 7) as i32);
            let dr = if rr > r { rr - r } else { r - rr };
            let dc = if cc > c { cc - c } else { c - cc };
            let d = if dr > dc { dr } else { dc };
            if d >= 1 && d <= 2 {
                m |= 1u64 << q;
            }
            q += 1;
        }
        t[s] = m;
        s += 1;
    }
    t
}

pub static BOX2: [u64; SQUARES] = build_box2();

/// 3x3 dilation (the stones and their 8 neighbours), separable: 10 ops instead of eight masked shifts.
#[inline(always)]
fn box1(x: u64) -> u64 {
    let h = x | ((x & !FILE_A) >> 1) | ((x & !FILE_G) << 1);
    (h | (h >> 7) | (h << 7)) & FULL
}

/// Empty squares a side with `stones` can land on (clone or jump) = the 5x5 box minus the stones themselves.
#[inline(always)]
fn landing(stones: u64, empty: u64) -> u64 {
    box1(box1(stones)) & empty
}

/// Bit planes (ones, twos, fours, eights) of the number of `v` stones in the 8-neighbourhood of every square,
/// built with a carry-save adder over the 8 shifted masks.
#[inline(always)]
fn victim_planes(v: u64) -> [u64; 4] {
    let s0 = (v >> 1) & !FILE_G;
    let s1 = (v << 1) & !FILE_A;
    let s2 = v >> 7;
    let s3 = (v << 7) & FULL;
    let s4 = (v >> 8) & !FILE_G;
    let s5 = (v << 8) & !FILE_A & FULL;
    let s6 = (v >> 6) & !FILE_A;
    let s7 = (v << 6) & !FILE_G & FULL;
    let t0 = s0 ^ s1 ^ s2;
    let c0 = (s0 & s1) | (s2 & (s0 ^ s1));
    let t1 = s3 ^ s4 ^ s5;
    let c1 = (s3 & s4) | (s5 & (s3 ^ s4));
    let t2 = s6 ^ s7;
    let c2 = s6 & s7;
    let ones = t0 ^ t1 ^ t2;
    let k0 = (t0 & t1) | (t2 & (t0 ^ t1));
    let u0 = c0 ^ c1 ^ c2;
    let uc = (c0 & c1) | (c2 & (c0 ^ c1));
    let twos = u0 ^ k0;
    let tc = u0 & k0;
    [ones, twos, uc ^ tc, uc & tc]
}

/// Add 1 to the count of every square in `m` (a new victim stone adds 1 to each of its 8 neighbours).
#[inline(always)]
fn planes_add(p: &mut [u64; 4], m: u64) {
    let c0 = p[0] & m;
    p[0] ^= m;
    let c1 = p[1] & c0;
    p[1] ^= c0;
    let c2 = p[2] & c1;
    p[2] ^= c1;
    p[3] ^= c2;
}

/// Largest victim count over the candidate squares, read off the bit planes (no per-candidate loop).
#[inline(always)]
fn planes_max(cand: u64, p: &[u64; 4]) -> u32 {
    let mut m = cand;
    let mut best = 0u32;
    for k in (0..4).rev() {
        let x = m & p[k];
        if x != 0 {
            best |= 1 << k;
            m = x;
        }
    }
    best
}

#[cfg(test)]
#[inline]
fn landing_ref(stones: u64, empty: u64) -> u64 {
    (dist_union(stones, 1) | jump_union(stones)) & empty
}

/// Best number of `victims` stones converted by one landing of `attacker` (clone or jump) on `empty`.
#[cfg(test)]
#[inline]
fn best_capture_ref(attacker: u64, victims: u64, empty: u64) -> u32 {
    let mut cand = landing_ref(attacker, empty) & dist_union(victims, 1);
    let mut best = 0;
    while cand != 0 {
        let t = cand.trailing_zeros() as usize;
        best = best.max((RING1[t] & victims).count_ones());
        cand &= cand - 1;
    }
    best
}

/// G5 for one perspective: per clone square the enemy's best reply capture, from base victim planes of `own` plus a
/// bit-sliced increment per added stone (the clone and the converted stones); enemy landing reuses `land_e` when nothing converts.
#[inline(always)]
fn g5_block(own: u64, en: u64, empty: u64, clone_o: u64, land_e: u64, out: &mut [u32; MAX_ACTIVE], n: &mut usize) {
    let base = victim_planes(own);
    let du_own = box1(own);
    let mut c = clone_o;
    while c != 0 {
        let s = c.trailing_zeros() as usize;
        let sbit = 1u64 << s;
        let conv = en & RING1[s];
        let mut p = base;
        planes_add(&mut p, RING1[s]);
        let mut near = du_own | RING1[s];
        let mut cv = conv;
        while cv != 0 {
            let x = cv.trailing_zeros() as usize;
            planes_add(&mut p, RING1[x]);
            near |= RING1[x];
            cv &= cv - 1;
        }
        let land2 = if conv == 0 { land_e & !sbit } else { box1(box1(en & !conv)) & (empty & !sbit) };
        let cand = land2 & near;
        let best_rep = if cand == 0 { 0 } else { planes_max(cand, &p) };
        let v = 1 + conv.count_ones() as i32 - best_rep as i32;
        out[*n] = (G5_OFF + s * 9 + (v.clamp(-3, 5) + 3) as usize) as u32;
        *n += 1;
        c &= c - 1;
    }
}

/// Active feature indices for perspective `side` (0 black / 1 white), restricted to `groups`.
/// Returns the count written into `out`.
pub fn features(b: &Board, side: usize, groups: u32, out: &mut [u32; MAX_ACTIVE]) -> usize {
    let own = b.occ[side];
    let en = b.occ[1 - side];
    let empty = b.empty();
    let clone_o = box1(own) & empty;
    let clone_e = box1(en) & empty;
    let land_o = landing(own, empty);
    let land_e = landing(en, empty);
    let near_land_e = box1(land_e);
    let near_land_o = box1(land_o);

    let mut n = 0;
    let mut anchor = usize::MAX;
    let mut best = (0u32, 0u32); // (gain, clone flag) of the anchor so far
    // codes by square class, branch-free over the class bit sets (the old per-square 3-way branch mispredicted ~half the time)
    let mut codes = [0u16; SQUARES];
    let mut m = own;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        codes[sq] = ((((near_land_e >> sq) & 1) as usize) * 9 + (RING1[sq] & empty).count_ones() as usize) as u16;
        m &= m - 1;
    }
    let mut m = en;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        codes[sq] = (18 + (((near_land_o >> sq) & 1) as usize) * 9 + (RING1[sq] & empty).count_ones() as usize) as u16;
        m &= m - 1;
    }
    let mut m = FULL & !(own | en | empty);
    while m != 0 {
        codes[m.trailing_zeros() as usize] = BLOCKER_CODE as u16;
        m &= m - 1;
    }
    let mut m = empty;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        let ring = RING1[sq];
        let go = (ring & en).count_ones();
        let ge = (ring & own).count_ones();
        let ro = if (clone_o >> sq) & 1 != 0 { 2 } else { ((land_o >> sq) & 1) as usize };
        let re = if (clone_e >> sq) & 1 != 0 { 2 } else { ((land_e >> sq) & 1) as usize };
        if ro > 0 && go > 0 {
            let key = (go, (ro == 2) as u32);
            if key > best {
                best = key;
                anchor = sq;
            }
        }
        codes[sq] = (36 + ((ro * 3 + re) * 5 + go.min(4) as usize) * 5 + ge.min(4) as usize) as u16;
        m &= m - 1;
    }
    if groups & G1 != 0 {
        for sq in 0..SQUARES {
            out[sq] = (G1_OFF + sq * SQ_STATES + codes[sq] as usize) as u32;
        }
        n = SQUARES;
    }
    if groups & G2 != 0 {
        if anchor == usize::MAX {
            out[n] = G2_NONE as u32;
            n += 1;
        } else {
            let mut stones = (own | en) & BOX2[anchor];
            while stones != 0 {
                let t = stones.trailing_zeros() as usize;
                out[n] = (G2_OFF + (anchor * SQUARES + t) * 2 + ((en >> t) & 1) as usize) as u32;
                n += 1;
                stones &= stones - 1;
            }
        }
    }
    if groups & G3 != 0 {
        let mut stones = own | en;
        while stones != 0 {
            let t = stones.trailing_zeros() as usize;
            let colour = ((en >> t) & 1) as usize;
            let (mine, foe) = if colour == 0 { (own, en) } else { (en, own) };
            let on = (RING1[t] & mine).count_ones() as usize;
            let ef = (RING1[t] & foe).count_ones() as usize;
            out[n] = (G3_OFF + ((t * 2 + colour) * 9 + on) * 9 + ef) as u32;
            n += 1;
            stones &= stones - 1;
        }
    }
    if groups & G5 != 0 {
        g5_block(own, en, empty, clone_o, land_e, out, &mut n);
    }
    if groups & G6 != 0 {
        let e = empty.count_ones() as usize;
        let phase = (e / 7).min(7);
        for (rel, land) in [(0usize, land_o), (1usize, land_e)] {
            let bucket = (land.count_ones() as usize / 3).min(9);
            out[n] = (G6_OFF + (rel * 8 + phase) * 10 + bucket) as u32;
            n += 1;
        }
        out[n] = (G6_PAR_OFF + phase * 2 + (e & 1)) as u32;
        n += 1;
    }
    n
}

#[cfg(test)]
/// Straightforward reference implementation (square-by-square); `features` must match it exactly.
pub fn features_ref(b: &Board, side: usize, groups: u32, out: &mut [u32; MAX_ACTIVE]) -> usize {
    let own = b.occ[side];
    let en = b.occ[1 - side];
    let empty = b.empty();
    let clone_o = dist_union(own, 1) & empty;
    let clone_e = dist_union(en, 1) & empty;
    let land_o = landing_ref(own, empty);
    let land_e = landing_ref(en, empty);
    let near_land_e = dist_union(land_e, 1);
    let near_land_o = dist_union(land_o, 1);

    let mut n = 0;
    let mut anchor = usize::MAX;
    let mut best = (0u32, 0u32); // (gain, clone flag) of the anchor so far
    for sq in 0..SQUARES {
        let bit = 1u64 << sq;
        let ring = RING1[sq];
        let code = if own & bit != 0 {
            (((near_land_e & bit) != 0) as usize) * 9 + (ring & empty).count_ones() as usize
        } else if en & bit != 0 {
            18 + (((near_land_o & bit) != 0) as usize) * 9 + (ring & empty).count_ones() as usize
        } else if empty & bit != 0 {
            let go = (ring & en).count_ones();
            let ge = (ring & own).count_ones();
            let ro = if clone_o & bit != 0 { 2 } else if land_o & bit != 0 { 1 } else { 0 };
            let re = if clone_e & bit != 0 { 2 } else if land_e & bit != 0 { 1 } else { 0 };
            if ro > 0 && go > 0 {
                let key = (go, (ro == 2) as u32);
                if key > best {
                    best = key;
                    anchor = sq;
                }
            }
            36 + ((ro * 3 + re) * 5 + go.min(4) as usize) * 5 + ge.min(4) as usize
        } else {
            BLOCKER_CODE
        };
        if groups & G1 != 0 {
            out[n] = (G1_OFF + sq * SQ_STATES + code) as u32;
            n += 1;
        }
    }
    if groups & G2 != 0 {
        if anchor == usize::MAX {
            out[n] = G2_NONE as u32;
            n += 1;
        } else {
            let mut stones = (own | en) & BOX2[anchor];
            while stones != 0 {
                let t = stones.trailing_zeros() as usize;
                out[n] = (G2_OFF + (anchor * SQUARES + t) * 2 + ((en >> t) & 1) as usize) as u32;
                n += 1;
                stones &= stones - 1;
            }
        }
    }
    if groups & G3 != 0 {
        let mut stones = own | en;
        while stones != 0 {
            let t = stones.trailing_zeros() as usize;
            let colour = ((en >> t) & 1) as usize;
            let (mine, foe) = if colour == 0 { (own, en) } else { (en, own) };
            let on = (RING1[t] & mine).count_ones() as usize;
            let ef = (RING1[t] & foe).count_ones() as usize;
            out[n] = (G3_OFF + ((t * 2 + colour) * 9 + on) * 9 + ef) as u32;
            n += 1;
            stones &= stones - 1;
        }
    }
    if groups & G5 != 0 {
        let mut c = clone_o;
        while c != 0 {
            let s = c.trailing_zeros() as usize;
            let conv = en & RING1[s];
            let own2 = own | (1u64 << s) | conv;
            let en2 = en & !conv;
            let empty2 = empty & !(1u64 << s);
            let v = 1 + conv.count_ones() as i32 - best_capture_ref(en2, own2, empty2) as i32;
            out[n] = (G5_OFF + s * 9 + (v.clamp(-3, 5) + 3) as usize) as u32;
            n += 1;
            c &= c - 1;
        }
    }
    if groups & G6 != 0 {
        let e = empty.count_ones() as usize;
        let phase = (e / 7).min(7);
        for (rel, land) in [(0usize, land_o), (1usize, land_e)] {
            let bucket = (land.count_ones() as usize / 3).min(9);
            out[n] = (G6_OFF + (rel * 8 + phase) * 10 + bucket) as u32;
            n += 1;
        }
        out[n] = (G6_PAR_OFF + phase * 2 + (e & 1)) as u32;
        n += 1;
    }
    n
}

/// Both perspectives, side to move first: (us, n_us, them, n_them). One joint pass: the landing sets, the stone
/// codes (they differ by 18 between the perspectives) and the G3 neighbour counts (perspective independent) are
/// computed once; each output equals `features` for that side, in the same order.
pub fn both(b: &Board, groups: u32) -> ([u32; MAX_ACTIVE], usize, [u32; MAX_ACTIVE], usize) {
    let stm = b.turn as usize;
    let pa = b.occ[stm];
    let pb = b.occ[1 - stm];
    let empty = b.empty();
    let clone_a = box1(pa) & empty;
    let clone_b = box1(pb) & empty;
    let land_a = landing(pa, empty);
    let land_b = landing(pb, empty);
    let nl_a = box1(land_a);
    let nl_b = box1(land_b);

    let mut us = [0u32; MAX_ACTIVE];
    let mut them = [0u32; MAX_ACTIVE];
    let (mut nu, mut nt) = (0usize, 0usize);
    let (mut anc_a, mut anc_b) = (usize::MAX, usize::MAX);
    let (mut best_a, mut best_b) = ((0u32, 0u32), (0u32, 0u32));
    let mut ca = [0u16; SQUARES];
    let mut cb = [0u16; SQUARES];
    let mut m = pa;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        let c = (((nl_b >> sq) & 1) as usize) * 9 + (RING1[sq] & empty).count_ones() as usize;
        ca[sq] = c as u16;
        cb[sq] = (c + 18) as u16;
        m &= m - 1;
    }
    let mut m = pb;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        let c = (((nl_a >> sq) & 1) as usize) * 9 + (RING1[sq] & empty).count_ones() as usize;
        ca[sq] = (c + 18) as u16;
        cb[sq] = c as u16;
        m &= m - 1;
    }
    let mut m = FULL & !(pa | pb | empty);
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        ca[sq] = BLOCKER_CODE as u16;
        cb[sq] = BLOCKER_CODE as u16;
        m &= m - 1;
    }
    let mut m = empty;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        let ring = RING1[sq];
        let go = (ring & pb).count_ones();
        let ge = (ring & pa).count_ones();
        let ro = if (clone_a >> sq) & 1 != 0 { 2 } else { ((land_a >> sq) & 1) as usize };
        let re = if (clone_b >> sq) & 1 != 0 { 2 } else { ((land_b >> sq) & 1) as usize };
        if ro > 0 && go > 0 {
            let key = (go, (ro == 2) as u32);
            if key > best_a {
                best_a = key;
                anc_a = sq;
            }
        }
        if re > 0 && ge > 0 {
            let key = (ge, (re == 2) as u32);
            if key > best_b {
                best_b = key;
                anc_b = sq;
            }
        }
        ca[sq] = (36 + ((ro * 3 + re) * 5 + go.min(4) as usize) * 5 + ge.min(4) as usize) as u16;
        cb[sq] = (36 + ((re * 3 + ro) * 5 + ge.min(4) as usize) * 5 + go.min(4) as usize) as u16;
        m &= m - 1;
    }
    if groups & G1 != 0 {
        for sq in 0..SQUARES {
            us[sq] = (G1_OFF + sq * SQ_STATES + ca[sq] as usize) as u32;
            them[sq] = (G1_OFF + sq * SQ_STATES + cb[sq] as usize) as u32;
        }
        nu = SQUARES;
        nt = SQUARES;
    }
    if groups & G2 != 0 {
        for (anchor, en, out, n) in [(anc_a, pb, &mut us, &mut nu), (anc_b, pa, &mut them, &mut nt)] {
            if anchor == usize::MAX {
                out[*n] = G2_NONE as u32;
                *n += 1;
            } else {
                let mut stones = (pa | pb) & BOX2[anchor];
                while stones != 0 {
                    let t = stones.trailing_zeros() as usize;
                    out[*n] = (G2_OFF + (anchor * SQUARES + t) * 2 + ((en >> t) & 1) as usize) as u32;
                    *n += 1;
                    stones &= stones - 1;
                }
            }
        }
    }
    if groups & G3 != 0 {
        let mut stones = pa | pb;
        while stones != 0 {
            let t = stones.trailing_zeros() as usize;
            let col_a = ((pb >> t) & 1) as usize;
            let (same, other) = if col_a == 0 { (pa, pb) } else { (pb, pa) };
            let on = (RING1[t] & same).count_ones() as usize;
            let ef = (RING1[t] & other).count_ones() as usize;
            us[nu] = (G3_OFF + ((t * 2 + col_a) * 9 + on) * 9 + ef) as u32;
            them[nt] = (G3_OFF + ((t * 2 + 1 - col_a) * 9 + on) * 9 + ef) as u32;
            nu += 1;
            nt += 1;
            stones &= stones - 1;
        }
    }
    if groups & G5 != 0 {
        g5_block(pa, pb, empty, clone_a, land_b, &mut us, &mut nu);
        g5_block(pb, pa, empty, clone_b, land_a, &mut them, &mut nt);
    }
    if groups & G6 != 0 {
        let e = empty.count_ones() as usize;
        let phase = (e / 7).min(7);
        let ba = (land_a.count_ones() as usize / 3).min(9);
        let bb = (land_b.count_ones() as usize / 3).min(9);
        us[nu] = (G6_OFF + phase * 10 + ba) as u32;
        us[nu + 1] = (G6_OFF + (8 + phase) * 10 + bb) as u32;
        us[nu + 2] = (G6_PAR_OFF + phase * 2 + (e & 1)) as u32;
        nu += 3;
        them[nt] = (G6_OFF + phase * 10 + bb) as u32;
        them[nt + 1] = (G6_OFF + (8 + phase) * 10 + ba) as u32;
        them[nt + 2] = (G6_PAR_OFF + phase * 2 + (e & 1)) as u32;
        nt += 3;
    }
    (us, nu, them, nt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{infect_direct, FULL, RING2};

    fn rand_board(seed: &mut u64) -> Board {
        let mut next = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        };
        let fill = next() % 49;
        let mut occ = [0u64; 2];
        for sq in 0..49u64 {
            if next() % 49 < fill {
                occ[(next() & 1) as usize] |= 1 << sq;
            }
        }
        let mut b = Board::start();
        b.occ = occ;
        b.blockers = 0;
        b.turn = (next() & 1) as u8;
        b
    }

    /// Brute-force reference for every group, by enumerating moves square by square.
    #[test]
    fn matches_move_enumeration() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for _ in 0..3000 {
            let b = rand_board(&mut seed);
            for side in 0..2 {
                let own = b.occ[side];
                let en = b.occ[1 - side];
                let empty = !(own | en) & FULL;
                let mut out = [0u32; MAX_ACTIVE];
                let n = features(&b, side, ALL_GROUPS, &mut out);
                assert!(n <= MAX_ACTIVE && out[..n].iter().all(|&f| (f as usize) < NUM_FEATS));
                let can_land = |who: u64, t: usize| who & (RING1[t] | RING2[t]) != 0;
                // G1 + anchor reference
                let mut best = (0u32, 0u32, usize::MAX);
                for sq in 0..SQUARES {
                    let bit = 1u64 << sq;
                    let code = out[sq] as usize - G1_OFF - sq * SQ_STATES;
                    if empty & bit != 0 {
                        let ro = if own & RING1[sq] != 0 { 2 } else if own & RING2[sq] != 0 { 1 } else { 0 };
                        let re = if en & RING1[sq] != 0 { 2 } else if en & RING2[sq] != 0 { 1 } else { 0 };
                        let go = infect_direct(en, sq as u8).count_ones();
                        let ge = infect_direct(own, sq as u8).count_ones();
                        assert_eq!(code, 36 + ((ro * 3 + re) * 5 + go.min(4) as usize) * 5 + ge.min(4) as usize);
                        if ro > 0 && go > 0 && (go, (ro == 2) as u32) > (best.0, best.1) {
                            best = (go, (ro == 2) as u32, sq);
                        }
                    } else {
                        let (mine, foe) = if own & bit != 0 { (own, en) } else { (en, own) };
                        let (mut exposed, mut ne) = (false, 0);
                        for t in 0..SQUARES {
                            if RING1[sq] >> t & 1 == 1 && empty >> t & 1 == 1 {
                                ne += 1;
                                exposed |= can_land(foe, t);
                            }
                        }
                        let base = if mine == own { 0 } else { 18 };
                        assert_eq!(code, base + exposed as usize * 9 + ne, "stone sq {sq}");
                    }
                }
                let mut k = 49;
                if best.2 == usize::MAX {
                    assert_eq!(out[k] as usize, G2_NONE);
                    k += 1;
                } else {
                    let stones = ((own | en) & BOX2[best.2]).count_ones() as usize;
                    for &f in &out[k..k + stones] {
                        assert_eq!((f as usize - G2_OFF) / 2 / SQUARES, best.2, "anchor square");
                    }
                    k += stones;
                }
                // G3
                let nst = (own | en).count_ones() as usize;
                for &f in &out[k..k + nst] {
                    let f = f as usize - G3_OFF;
                    let (t, colour) = (f / 81 / 2, f / 81 % 2);
                    let (mine, foe) = if colour == 0 { (own, en) } else { (en, own) };
                    assert_eq!(f % 81, (RING1[t] & mine).count_ones() as usize * 9 + (RING1[t] & foe).count_ones() as usize);
                }
                k += nst;
                // G5: brute force the enemy's best reply capture after each clone
                let ncl = (dist_union(own, 1) & empty).count_ones() as usize;
                for &f in &out[k..k + ncl] {
                    let f = f as usize - G5_OFF;
                    let s = f / 9;
                    let conv = infect_direct(en, s as u8);
                    let (own2, en2, emp2) = (own | (1 << s) | conv, en & !conv, empty & !(1u64 << s));
                    let mut reply = 0;
                    for t in 0..SQUARES {
                        if emp2 >> t & 1 == 1 && can_land(en2, t) {
                            reply = reply.max(infect_direct(own2, t as u8).count_ones());
                        }
                    }
                    let v = (1 + conv.count_ones() as i32 - reply as i32).clamp(-3, 5);
                    assert_eq!(f % 9, (v + 3) as usize, "2-ply at {s}");
                }
                k += ncl;
                assert_eq!(n, k + 3, "G6 adds 3");
                for g in [G1, G2, G3, G5, G6] {
                    let mut o2 = [0u32; MAX_ACTIVE];
                    let m = features(&b, side, ALL_GROUPS & !g, &mut o2);
                    assert!(m < n || (g == G5 && ncl == 0) || (g == G3 && nst == 0), "group mask {g}");
                }
            }
        }
    }

    /// The optimised `features` must equal the square-by-square reference for every group mask.
    #[test]
    fn matches_reference_all_masks() {
        let mut seed = 0x0bad_cafe_1234_5678u64;
        for _ in 0..4000 {
            let b = rand_board(&mut seed);
            for side in 0..2 {
                for g in 0..64u32 {
                    let (mut x, mut y) = ([0u32; MAX_ACTIVE], [0u32; MAX_ACTIVE]);
                    let nx = features(&b, side, g, &mut x);
                    let ny = features_ref(&b, side, g, &mut y);
                    assert_eq!(&x[..nx], &y[..ny], "groups {g}");
                }
            }
        }
    }

    /// The joint `both` equals two independent `features` calls for every board, side to move and group mask.
    #[test]
    fn both_matches_features() {
        let mut seed = 0x5eed_0001_9e37_79b9u64;
        for _ in 0..4000 {
            let mut b = rand_board(&mut seed);
            for turn in 0..2usize {
                b.turn = turn as _;
                for g in 0..64u32 {
                    let (us, nu, them, nt) = both(&b, g);
                    let (mut x, mut y) = ([0u32; MAX_ACTIVE], [0u32; MAX_ACTIVE]);
                    let nx = features(&b, turn, g, &mut x);
                    let ny = features(&b, 1 - turn, g, &mut y);
                    assert_eq!(&us[..nu], &x[..nx], "us groups {g}");
                    assert_eq!(&them[..nt], &y[..ny], "them groups {g}");
                }
            }
        }
    }

    /// Perspective symmetry: features for `side` equal the features for the other side after swapping colours.
    #[test]
    fn perspective_swap() {
        let mut seed = 0xdead_beef_cafe_f00du64;
        for _ in 0..1000 {
            let b = rand_board(&mut seed);
            let mut s = b;
            s.occ = [b.occ[1], b.occ[0]];
            let (mut x, mut y) = ([0u32; MAX_ACTIVE], [0u32; MAX_ACTIVE]);
            let nx = features(&b, 0, ALL_GROUPS, &mut x);
            let ny = features(&s, 1, ALL_GROUPS, &mut y);
            assert_eq!(&x[..nx], &y[..ny]);
        }
    }
}
