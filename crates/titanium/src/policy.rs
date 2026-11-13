//! Move-ordering policy head (exp/policy, docs/POLICY_PLAN.md). logit(m) = bias[slot] + scalar move features
//! + U[slot] . xin, where xin is the sparse pairwise accumulator product the value net's L1 reads. Fitted by `ti policy`.
//! File: magic "TPOL1\0\0\0", u32 dim, f32 bias[98], f32 g[45], f32 u[98*dim], little endian.
//! TPOL2 (D486): the same, then f32 dest[98][6561] + f32 orig[49][6561]: one lookup per move on the base-3 code of the
//! 8 cells around the destination (and, for jumps, around the origin), mover-relative, as `ti policy --pat k3` fits it.
use crate::board::{Board, Move, RING1};


/// Bundled head (fitted by `ti policy --feat x --out-bin` on the bundled net data/nnue/nnue_main.tnn). Env TI_POLICY=none disables, =path overrides.
static BUNDLED: &[u8] = include_bytes!("../../../data/nnue/policy_main.bin");

pub const SLOTS: usize = 98;
pub const NSCAL: usize = 45;
pub const NSEE_BASE: usize = 18 * 19;
pub const NSEE: usize = 8 * NSEE_BASE;

pub struct Policy {
    pub dim: usize,
    bias: [f32; SLOTS],
    g: [f32; NSCAL],
    see: Vec<f32>,
    u: Vec<f32>,
    /// D486 pattern tables (empty for TPOL1)
    /// D502: stored as i16 at TQ (1/2048 logit steps, |table| <= 16): half the cache footprint of f32 (the nps cost)
    td: Vec<i16>,
    tor: Vec<i16>,
    /// D502 TPOL3: tables shared by the 9 edge classes (row/col in {0, 1..5, 6}) instead of per square
    cls: bool,
}

/// edge class of a square: (row class) * 3 + (col class), class 0 = first line, 1 = inner, 2 = last line
static CLS: [u8; 49] = {
    let mut t = [0u8; 49];
    let mut a = 0;
    while a < 49 {
        let (r, c) = (a / 7, a % 7);
        let fr: u8 = if r == 0 { 0 } else if r == 6 { 2 } else { 1 };
        let fc: u8 = if c == 0 { 0 } else if c == 6 { 2 } else { 1 };
        t[a] = fr * 3 + fc;
        a += 1;
    }
    t
};

const P3: usize = 6561;
const TQ: f32 = 2048.0;

/// D493 speed: per square, the 8 neighbour squares in the fixed order NW N NE W E SW S SE; off-board = bit 63 (never set
/// on a 49-square board, so it reads as empty = digit 0, exactly like the trainer's bounds check).
static NB: [[u8; 8]; 49] = {
    let mut t = [[63u8; 8]; 49];
    let d: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
    let mut a = 0;
    while a < 49 {
        let (r, c) = ((a / 7) as i32, (a % 7) as i32);
        let mut j = 0;
        while j < 8 {
            let (rr, cc) = (r + d[j].0, c + d[j].1);
            if rr >= 0 && rr < 7 && cc >= 0 && cc < 7 {
                t[a][j] = (rr * 7 + cc) as u8;
            }
            j += 1;
        }
        a += 1;
    }
    t
};

/// T3[bits] = base-3 value of 8 binary digits, bit j = digit j, digit 0 most significant (weight 3^7)
static T3: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut c = 0;
    while c < 256 {
        let mut v = 0u16;
        let mut j = 0;
        while j < 8 {
            v = v * 3 + ((c >> j) & 1) as u16;
            j += 1;
        }
        t[c] = v;
        c += 1;
    }
    t
};

/// D502: on-board neighbour mask per square
static NBM: [u64; 49] = {
    let mut t = [0u64; 49];
    let mut a = 0;
    while a < 49 {
        let mut j = 0;
        while j < 8 {
            if NB[a][j] != 63 {
                t[a] |= 1u64 << NB[a][j];
            }
            j += 1;
        }
        a += 1;
    }
    t
};

/// the 8 neighbour bits of `s` (bit j = NB order) with 4 shifts: mask off-board/wrapped cells, shift up by 8 so s-8 >= 0
#[inline(always)]
fn bits8f(x: u64, s: usize) -> usize {
    let y = (x & NBM[s]) << 8;
    (((y >> s) & 7) | (((y >> (s + 7)) & 1) << 3) | (((y >> (s + 9)) & 1) << 4) | (((y >> (s + 14)) & 7) << 5)) as usize
}

#[allow(dead_code)]
#[inline(always)]
fn bits8(x: u64, nb: &[u8; 8]) -> usize {
    let mut v = 0usize;
    let mut j = 0;
    while j < 8 {
        v |= (((x >> nb[j]) & 1) as usize) << j;
        j += 1;
    }
    v
}

/// Base-3 code of the 8 neighbours of `sq` (0 empty/off-board, 1 `me`, 2 `opp`), order NW N NE W E SW S SE, digit 0 most
/// significant; must match titaniumtest src/policy.rs k3code exactly (test k3code_matches_reference).
#[inline]
fn k3code(me: u64, opp: u64, sq: u8) -> usize {
    let s = sq as usize;
    T3[bits8f(me, s)] as usize + 2 * T3[bits8f(opp, s)] as usize
}

#[inline(always)]
pub(crate) fn k3code_pub(me: u64, opp: u64, sq: u8) -> usize {
    k3code(me, opp, sq)
}

/// D502: per-node memo of the 3x3 codes (many moves share a destination or an origin square); new() per node
pub struct KCache {
    valid: u64,
    code: [u16; 49],
}

impl KCache {
    #[inline]
    pub fn new() -> KCache {
        KCache { valid: 0, code: [0; 49] }
    }
    #[inline(always)]
    fn get(&mut self, me: u64, opp: u64, sq: u8) -> usize {
        if self.valid >> sq & 1 == 0 {
            self.code[sq as usize] = k3code(me, opp, sq) as u16;
            self.valid |= 1u64 << sq;
        }
        self.code[sq as usize] as usize
    }
}

impl Default for KCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    fn k3ref(me: u64, opp: u64, sq: u8) -> usize {
        let (r, c) = ((sq / 7) as i32, (sq % 7) as i32);
        let mut code = 0usize;
        for (dr, dc) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
            let (rr, cc) = (r + dr, c + dc);
            let d = if (0..7).contains(&rr) && (0..7).contains(&cc) {
                let bit = 1u64 << (rr * 7 + cc);
                if me & bit != 0 { 1 } else if opp & bit != 0 { 2 } else { 0 }
            } else {
                0
            };
            code = code * 3 + d;
        }
        code
    }
    #[test]
    fn k3code_matches_reference() {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        for _ in 0..20000 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let a = x & ((1u64 << 49) - 1);
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let me = a & x;
            let opp = a & !x;
            for sq in 0..49u8 {
                assert_eq!(super::k3code(me, opp, sq), k3ref(me, opp, sq));
            }
        }
    }

    #[test]
    fn precomputed_logits_match_logit_k() {
        use crate::board::{Board, MoveList, RING1, RING2};
        let mut x: u64 = 0x517CC1B727220A95;
        let next_u64 = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };
        let bundled = super::bundled().expect("bundled policy");
        let mut list = MoveList::new();
        // Synthetic dim>0 TPOL2/TPOL3 policy + bundled TPOL3 dim==0 policy
        let dim = 8usize;
        let mut synth = super::Policy {
            dim,
            bias: [0.0; super::SLOTS],
            g: [0.0; super::NSCAL],
            see: Vec::new(),
            u: vec![0.0; super::SLOTS * dim],
            td: vec![0i16; 18 * super::P3],
            tor: vec![0i16; 9 * super::P3],
            cls: true,
        };
        let mut synth5 = super::Policy {
            dim: 0,
            bias: [0.0; super::SLOTS],
            g: [0.0; super::NSCAL],
            see: (0..super::NSEE).map(|i| ((i as i32 * 29) % 200 - 100) as f32 * 0.015625).collect(),
            u: Vec::new(),
            td: vec![0i16; 18 * super::P3],
            tor: vec![0i16; 9 * super::P3],
            cls: true,
        };
        for (i, v) in synth.bias.iter_mut().enumerate() {
            *v = ((i as i32 - 49) as f32) * 0.03125;
        }
        for (i, v) in synth.g.iter_mut().enumerate() {
            *v = ((i as i32 - 20) as f32) * 0.0625;
        }
        for (i, v) in synth.u.iter_mut().enumerate() {
            *v = (((i % 17) as i32 - 8) as f32) * 0.015625;
        }
        for (i, v) in synth.td.iter_mut().enumerate() {
            *v = ((i as i32 * 37) % 1000 - 500) as i16;
        }
        for (i, v) in synth.tor.iter_mut().enumerate() {
            *v = ((i as i32 * 53) % 1000 - 500) as i16;
        }
        synth5.bias = synth.bias;
        synth5.g = synth.g;
        synth5.td.copy_from_slice(&synth.td);
        synth5.tor.copy_from_slice(&synth.tor);
        // Also test TPOL5 binary serialization -> Policy::from_bytes round-trip
        let mut raw5 = b"TPOL5\0\0\0".to_vec();
        raw5.extend_from_slice(&0u32.to_le_bytes());
        for v in synth5.bias { raw5.extend_from_slice(&v.to_le_bytes()); }
        for v in synth5.g { raw5.extend_from_slice(&v.to_le_bytes()); }
        for &v in &synth5.see { raw5.extend_from_slice(&v.to_le_bytes()); }
        for &v in &synth5.td { raw5.extend_from_slice(&(v as f32 * (1.0 / super::TQ)).to_le_bytes()); }
        for &v in &synth5.tor { raw5.extend_from_slice(&(v as f32 * (1.0 / super::TQ)).to_le_bytes()); }
        let synth5_parsed = super::Policy::from_bytes(&raw5).expect("TPOL5 parse");
        assert!(synth5_parsed.has_see());
        let xin = [(0u16, 0.25f32), (3u16, 0.75f32), (6u16, 0.125f32)];

        for pol in [bundled, &synth, &synth5_parsed] {
            let x_slice: &[(u16, f32)] = if pol.dim > 0 { &xin } else { &[] };
            for n in 0..80 {
                let mut b = Board::start();
                for _ in 0..(n % 24) {
                    b.legal_moves_dedup_into(&mut list);
                    if list.is_empty() || b.game_over() {
                        break;
                    }
                    let idx = (next_u64(&mut x) as usize) % list.len();
                    b = b.make(list.move_at(idx));
                }
                if b.game_over() {
                    continue;
                }
                let mut kc = super::KCache::new();
                let me = b.me();
                let opp = b.opp();
                let empty = b.empty();
                let p_base = crate::nnfeat::victim_planes(me);
                let b1_0 = crate::nnfeat::box1(opp);
                let b2_0 = crate::nnfeat::box1(b1_0);
                let ph_off = super::see_phase_off(empty);
                let mut targets = crate::board::reach_union(me) & empty;
                while targets != 0 {
                    let to = targets.trailing_zeros() as u8;
                    targets &= targets - 1;
                    let r1_to = RING1[to as usize];
                    let flips_m = opp & r1_to;
                    let flips = (flips_m.count_ones() as usize).min(8);
                    let tholes = ((r1_to & empty).count_ones() as usize).min(8);
                    let k3_to = super::k3code_pub(me, opp, to);

                    let (p_to, b1o, b2o) = if pol.has_see() {
                        let (b1o, b2o) = if flips_m == 0 {
                            (b1_0, b2_0)
                        } else {
                            let b1 = crate::nnfeat::box1(opp & !flips_m);
                            (b1, crate::nnfeat::box1(b1))
                        };
                        let mut pt = p_base;
                        crate::nnfeat::planes_add(&mut pt, r1_to);
                        let mut fm = flips_m;
                        while fm != 0 {
                            let u = fm.trailing_zeros() as usize;
                            fm &= fm - 1;
                            crate::nnfeat::planes_add(&mut pt, RING1[u]);
                        }
                        (pt, b1o, b2o)
                    } else {
                        ([0; 4], 0, 0)
                    };

                    let clones = me & r1_to;
                    if clones != 0 {
                        let m = crate::board::Move {
                            from: clones.trailing_zeros() as u8,
                            to,
                        };
                        let sv = if pol.has_see() {
                            let ea = empty & !(1u64 << to);
                            let oi = super::see_opp_state(b1o & ea, b2o & ea, &p_to);
                            pol.see_val(ph_off, 1 + 2 * (flips_m.count_ones() as usize), oi)
                        } else {
                            0.0
                        };
                        let expected = pol.logit_k(&b, m, x_slice, &mut kc);
                        let actual = pol.clone_logit_pre(to, flips, tholes, k3_to, sv, x_slice);
                        assert_eq!(actual.to_bits(), expected.to_bits(), "clone mv {m:?}");
                    }

                    let (jp, jtd) = pol.jump_dest_parts(to, flips, tholes, k3_to);
                    let mut jumps = me & RING2[to as usize];
                    while jumps != 0 {
                        let from = jumps.trailing_zeros() as u8;
                        jumps &= jumps - 1;
                        let m = crate::board::Move { from, to };
                        let fh = ((RING1[from as usize] & empty).count_ones() as usize).min(8);
                        let k3_from = super::k3code_pub(me, opp, from);
                        let (fg, ft) = pol.jump_orig_parts(from, fh, k3_from);
                        let sv = if pol.has_see() {
                            let mut pa = p_to;
                            crate::nnfeat::planes_sub(&mut pa, RING1[from as usize]);
                            let ea = (empty & !(1u64 << to)) | (1u64 << from);
                            let oi = super::see_opp_state(b1o & ea, b2o & ea, &pa);
                            pol.see_val(ph_off, 2 * (flips_m.count_ones() as usize), oi)
                        } else {
                            0.0
                        };
                        let expected = pol.logit_k(&b, m, x_slice, &mut kc);
                        let actual = pol.jump_finish(to, jp, jtd, fg, ft, sv, x_slice);
                        assert_eq!(actual.to_bits(), expected.to_bits(), "jump mv {m:?}");
                    }
                }
            }
        }
    }
}

/// Exact 1-ply opponent reply differential state after our move:
/// 0 = opponent has no legal moves; 1..=18 = 1 + max_opp_diff (0..=17).
#[inline(always)]
pub(crate) fn see_opp_state(b1_empty: u64, b2_empty: u64, p: &[u64; 4]) -> usize {
    if b2_empty == 0 {
        return 0;
    }
    let cl = if b1_empty != 0 {
        1 + 2 * crate::nnfeat::planes_max(b1_empty, p)
    } else {
        0
    };
    let jp = 2 * crate::nnfeat::planes_max(b2_empty, p);
    1 + cl.max(jp) as usize
}

#[inline(always)]
pub(crate) fn see_phase_off(empty: u64) -> usize {
    ((empty.count_ones() as usize).saturating_sub(1) / 6).min(7) * NSEE_BASE
}

impl Policy {
    #[inline(always)]
    pub fn has_see(&self) -> bool {
        !self.see.is_empty()
    }

    #[inline(always)]
    pub(crate) fn see_val(&self, ph_off: usize, my_diff: usize, opp_idx: usize) -> f32 {
        self.see[ph_off + my_diff * 19 + opp_idx]
    }

    pub fn load(path: &str) -> std::io::Result<Policy> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(d: &[u8]) -> std::io::Result<Policy> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let v5 = d.len() >= 12 && &d[..8] == b"TPOL5\0\0\0";
        let v3 = (d.len() >= 12 && &d[..8] == b"TPOL3\0\0\0") || v5;
        let v2 = (d.len() >= 12 && &d[..8] == b"TPOL2\0\0\0") || v3;
        if d.len() < 12 || (&d[..8] != b"TPOL1\0\0\0" && !v2) {
            return Err(bad("bad policy magic"));
        }
        let dim = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        let nsee = if v5 { NSEE } else { 0 };
        let n1 = SLOTS + NSCAL + nsee + SLOTS * dim;
        let (nds, nos) = if v3 { (18, 9) } else { (SLOTS, 49) };
        let n = n1 + if v2 { (nds + nos) * P3 } else { 0 };
        if d.len() != 12 + 4 * n {
            return Err(bad("bad policy size"));
        }
        let f: Vec<f32> = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut bias = [0f32; SLOTS];
        bias.copy_from_slice(&f[..SLOTS]);
        let mut g = [0f32; NSCAL];
        g.copy_from_slice(&f[SLOTS..SLOTS + NSCAL]);
        let see = if v5 { f[SLOTS + NSCAL..SLOTS + NSCAL + NSEE].to_vec() } else { Vec::new() };
        let q = |x: &[f32]| -> Vec<i16> { x.iter().map(|v| (v * TQ).round().clamp(-32767.0, 32767.0) as i16).collect() };
        let (td, tor) = if v2 { (q(&f[n1..n1 + nds * P3]), q(&f[n1 + nds * P3..])) } else { (Vec::new(), Vec::new()) };
        Ok(Policy { dim, bias, g, see, u: f[SLOTS + NSCAL + nsee..n1].to_vec(), td, tor, cls: v3 })
    }

    /// Logit of move `m` in `b` given the sparse accumulator product `xin` (index, value).
    #[inline]
    pub fn logit(&self, b: &Board, m: Move, xin: &[(u16, f32)]) -> f32 {
        self.logit_k(b, m, xin, &mut KCache::new())
    }

    /// `logit` with a per-node code cache (search ordering loop: one KCache per node, same board for every move)
    #[inline]
    pub fn logit_k(&self, b: &Board, m: Move, xin: &[(u16, f32)], kc: &mut KCache) -> f32 {
        let clone = m.is_clone();
        let c = clone as usize;
        let empty = b.empty();
        let flips_raw = b.captures(m.to) as usize;
        let flips = flips_raw.min(8);
        let tholes = ((RING1[m.to as usize] & empty).count_ones() as usize).min(8);
        let slot = m.to as usize + 49 * (!clone) as usize;
        let mut z = self.bias[slot] + self.g[c * 9 + flips] + self.g[18 + c * 9 + tholes];
        if !clone {
            z += self.g[36 + ((RING1[m.from as usize] & empty).count_ones() as usize).min(8)];
        }
        if !self.see.is_empty() {
            let (me, opp) = (b.me(), b.opp());
            let tu = m.to as usize;
            let flips_m = opp & RING1[tu];
            let opp_p = opp & !flips_m;
            let b1o = crate::nnfeat::box1(opp_p);
            let b2o = crate::nnfeat::box1(b1o);
            let mut p = crate::nnfeat::victim_planes(me);
            crate::nnfeat::planes_add(&mut p, RING1[tu]);
            let mut fm = flips_m;
            while fm != 0 {
                let u = fm.trailing_zeros() as usize;
                fm &= fm - 1;
                crate::nnfeat::planes_add(&mut p, RING1[u]);
            }
            let tbit = 1u64 << tu;
            let ea = if clone {
                empty & !tbit
            } else {
                crate::nnfeat::planes_sub(&mut p, RING1[m.from as usize]);
                (empty & !tbit) | (1u64 << m.from)
            };
            let oi = see_opp_state(b1o & ea, b2o & ea, &p);
            let my_diff = clone as usize + 2 * flips_raw;
            z += self.see_val(see_phase_off(empty), my_diff, oi);
        }
        let row = &self.u[slot * self.dim..(slot + 1) * self.dim];
        for &(i, v) in xin {
            z += row[i as usize] * v;
        }
        if !self.td.is_empty() {
            let (me, opp) = (b.me(), b.opp());
            let (ds, os) = if self.cls {
                (CLS[m.to as usize] as usize * 2 + (!clone) as usize, CLS[m.from as usize] as usize)
            } else {
                (slot, m.from as usize)
            };
            let mut t = self.td[ds * P3 + kc.get(me, opp, m.to)] as i32;
            if !clone {
                t += self.tor[os * P3 + kc.get(me, opp, m.from)] as i32;
            }
            z += t as f32 * (1.0 / TQ);
        }
        z
    }

    /// Logit of a clone landing on `to` from precomputed per-destination features.
    /// Bit-identical to `logit_k`.
    #[inline(always)]
    pub(crate) fn clone_logit_pre(
        &self,
        to: u8,
        flips: usize,
        tholes: usize,
        k3_to: usize,
        see_val: f32,
        xin: &[(u16, f32)],
    ) -> f32 {
        let slot = to as usize;
        let mut z = self.bias[slot] + self.g[9 + flips] + self.g[27 + tholes];
        if !self.see.is_empty() {
            z += see_val;
        }
        if self.dim != 0 {
            let row = &self.u[slot * self.dim..(slot + 1) * self.dim];
            for &(i, v) in xin {
                z += row[i as usize] * v;
            }
        }
        if !self.td.is_empty() {
            let ds = if self.cls { CLS[slot] as usize * 2 } else { slot };
            z += self.td[ds * P3 + k3_to] as i32 as f32 * (1.0 / TQ);
        }
        z
    }

    /// Destination-only prefix and quantized pattern term for all jumps landing on `to`.
    #[inline(always)]
    pub(crate) fn jump_dest_parts(
        &self,
        to: u8,
        flips: usize,
        tholes: usize,
        k3_to: usize,
    ) -> (f32, i32) {
        let slot = to as usize + 49;
        let prefix = self.bias[slot] + self.g[flips] + self.g[18 + tholes];
        let td = if !self.td.is_empty() {
            let ds = if self.cls { CLS[to as usize] as usize * 2 + 1 } else { slot };
            self.td[ds * P3 + k3_to] as i32
        } else {
            0
        };
        (prefix, td)
    }

    /// Origin-only scalar term and quantized pattern term for jumps leaving `from`.
    #[inline(always)]
    pub(crate) fn jump_orig_parts(
        &self,
        from: u8,
        from_holes: usize,
        k3_from: usize,
    ) -> (f32, i32) {
        let g = self.g[36 + from_holes];
        let tor = if !self.td.is_empty() {
            let os = if self.cls { CLS[from as usize] as usize } else { from as usize };
            self.tor[os * P3 + k3_from] as i32
        } else {
            0
        };
        (g, tor)
    }

    /// Finish a jump logit from precomputed destination + origin parts.
    /// Exact left-to-right IEEE-754 order matches `logit_k` bit-for-bit.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn jump_finish(
        &self,
        to: u8,
        j_prefix: f32,
        j_td: i32,
        from_g: f32,
        from_tor: i32,
        see_val: f32,
        xin: &[(u16, f32)],
    ) -> f32 {
        let mut z = j_prefix + from_g;
        if !self.see.is_empty() {
            z += see_val;
        }
        if self.dim != 0 {
            let slot = to as usize + 49;
            let row = &self.u[slot * self.dim..(slot + 1) * self.dim];
            for &(i, v) in xin {
                z += row[i as usize] * v;
            }
        }
        if !self.td.is_empty() {
            z += (j_td + from_tor) as f32 * (1.0 / TQ);
        }
        z
    }
}

/// Process-wide policy for experiments: env TI_POLICY=<TPOL1 file>; ordering weight TI_POLW (default 100000 points per logit),
/// minimum depth TI_POLMIN (default 1). None when TI_POLICY is unset or unreadable.
pub fn global() -> Option<&'static Policy> {
    static P: std::sync::OnceLock<Option<Policy>> = std::sync::OnceLock::new();
    P.get_or_init(|| match spec().as_deref() {
        Some("none") => None,
        Some("bundled") | None => Policy::from_bytes(BUNDLED).ok(),
        Some(p) => Some(Policy::load(p).unwrap_or_else(|e| panic!("--policy / TI_POLICY {p}: {e}"))),
    })
    .as_ref()
}

static CONFIGURED: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Command-line `--policy <bundled|FILE|none>` (call once, before the first search); beats env TI_POLICY.
pub fn configure(spec: &str) {
    let _ = CONFIGURED.set(spec.to_string());
}

fn spec() -> Option<String> {
    CONFIGURED.get().cloned().or_else(|| std::env::var("TI_POLICY").ok())
}

/// True when the user chose a policy file explicitly (then it is used with any net); the default bundled head is
/// fitted on the bundled net only and switches itself off for any other net (see Searcher::set_nnue).
pub fn is_forced() -> bool {
    !matches!(spec().as_deref(), None | Some("bundled"))
}

pub fn bundled_bytes() -> &'static [u8] {
    BUNDLED
}

/// The compiled-in head, parsed (None only if the blob is corrupt).
pub fn bundled() -> Option<&'static Policy> {
    static B: std::sync::OnceLock<Option<Policy>> = std::sync::OnceLock::new();
    B.get_or_init(|| Policy::from_bytes(BUNDLED).ok()).as_ref()
}

pub fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
