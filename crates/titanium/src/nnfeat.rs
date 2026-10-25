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

use crate::board::{dist_union, jump_union, Board, RING1, SQUARES};

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

#[inline]
fn landing(stones: u64, empty: u64) -> u64 {
    (dist_union(stones, 1) | jump_union(stones)) & empty
}

/// Best number of `victims` stones converted by one landing of `attacker` (clone or jump) on `empty`.
#[inline]
fn best_capture(attacker: u64, victims: u64, empty: u64) -> u32 {
    let mut cand = landing(attacker, empty) & dist_union(victims, 1);
    let mut best = 0;
    while cand != 0 {
        let t = cand.trailing_zeros() as usize;
        best = best.max((RING1[t] & victims).count_ones());
        cand &= cand - 1;
    }
    best
}

/// Active feature indices for perspective `side` (0 black / 1 white), restricted to `groups`.
/// Returns the count written into `out`.
pub fn features(b: &Board, side: usize, groups: u32, out: &mut [u32; MAX_ACTIVE]) -> usize {
    let own = b.occ[side];
    let en = b.occ[1 - side];
    let empty = b.empty();
    let clone_o = dist_union(own, 1) & empty;
    let clone_e = dist_union(en, 1) & empty;
    let land_o = landing(own, empty);
    let land_e = landing(en, empty);
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
            let v = 1 + conv.count_ones() as i32 - best_capture(en2, own2, empty2) as i32;
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

/// Both perspectives, side to move first: (us, n_us, them, n_them).
pub fn both(b: &Board, groups: u32) -> ([u32; MAX_ACTIVE], usize, [u32; MAX_ACTIVE], usize) {
    let stm = b.turn as usize;
    let mut us = [0u32; MAX_ACTIVE];
    let mut them = [0u32; MAX_ACTIVE];
    let nu = features(b, stm, groups, &mut us);
    let nt = features(b, 1 - stm, groups, &mut them);
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
