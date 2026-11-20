//! Move-ordering policy head (docs/POLICY_PLAN.md). logit(m) = bias[slot] + scalar move features + the 3x3 pattern
//! tables around the destination (and, for jumps, around the origin), shared by the 9 edge classes (TPOL3, D502).
//! File: magic "TPOL3\0\0\0", u32 dim (must be 0), f32 bias[98], f32 g[45], f32 dest[18][6561], f32 orig[9][6561].
//! "TPOL5" = the same plus a SEE block after g; play never reads it (D748/D751), so it is skipped on load (D894) until
//! the no-SEE head (D893) replaces the shipped one. TPOL1/TPOL2 (no tables / per-square tables, dim > 0) are gone (LAWS A11).
use crate::board::{Board, Move, RING1};


/// Bundled head (data/nnue/policy_main.bin). Env TI_POLICY=none disables, =path overrides.
static BUNDLED: &[u8] = include_bytes!("../../../data/nnue/policy_main.bin");

pub const SLOTS: usize = 98;
pub const NSCAL: usize = 45;
/// size of the TPOL5 SEE block that is skipped on load
const NSEE: usize = 8 * 18 * 19;

/// One set of policy tables (one phase bucket).
pub struct Head {
    bias: [f32; SLOTS],
    g: [f32; NSCAL],
    inv_qd: f32,
    inv_qo: f32,
    /// D1332/D1333: 3x3 pattern tables quantized to i8 per head (max_abs / 127.0) to halve cache working set (~177 KB/bucket).
    td: Vec<i8>,
    tor: Vec<i8>,
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
const ND: usize = 18;
const NO: usize = 9;

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

    /// synthetic head as a file: TPOL3, or TPOL5 with a SEE block that must be skipped (same logits as the TPOL3)
    pub(crate) fn synth_bytes(tpol5: bool) -> Vec<u8> {
        let mut raw = if tpol5 { b"TPOL5\0\0\0".to_vec() } else { b"TPOL3\0\0\0".to_vec() };
        raw.extend_from_slice(&0u32.to_le_bytes());
        for i in 0..super::SLOTS { raw.extend_from_slice(&(((i as i32 - 49) as f32) * 0.03125).to_le_bytes()); }
        for i in 0..super::NSCAL { raw.extend_from_slice(&(((i as i32 - 20) as f32) * 0.0625).to_le_bytes()); }
        if tpol5 {
            for i in 0..super::NSEE { raw.extend_from_slice(&(((i as i32 * 31) % 200 - 100) as f32 * 0.015625).to_le_bytes()); }
        }
        for i in 0..((super::ND + super::NO) * super::P3) {
            raw.extend_from_slice(&(((i as i32 * 17) % 1000 - 500) as f32 / 2048.0).to_le_bytes());
        }
        raw
    }

    /// D1320: a TPOL6 of nb copies of one head gives that head's logits bit for bit; changing bucket k (13 here) bias changes
    /// exactly the positions with empties * nb / 50 == k (the NNUE's bucket formula).
    #[test]
    fn tpol6_buckets_copy_and_select() {
        use crate::board::{Board, MoveList};
        let one = synth_bytes(false);
        let body = &one[12..];
        let nb = 16usize;
        let mk = |bump: Option<usize>| -> super::Policy {
            let mut raw = b"TPOL6".to_vec();
            raw.extend_from_slice(&[0, 0, 0]);
            raw.extend_from_slice(&0u32.to_le_bytes());
            raw.extend_from_slice(&(nb as u32).to_le_bytes());
            for k in 0..nb {
                let mut blk = body.to_vec();
                if bump == Some(k) {
                    for s in 0..super::SLOTS {
                        let v = f32::from_le_bytes(blk[4 * s..4 * s + 4].try_into().unwrap()) + 1.0;
                        blk[4 * s..4 * s + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                raw.extend_from_slice(&blk);
            }
            super::Policy::from_bytes(&raw).expect("TPOL6")
        };
        let single = super::Policy::from_bytes(&one).expect("TPOL3");
        let (copies, bumped) = (mk(None), mk(Some(13)));
        assert_eq!((single.buckets(), copies.buckets()), (1, nb));
        let mut list = MoveList::new();
        let mut seen = [false; 2];
        let mut x = 0x2545F4914F6CDD1Du64;
        for n in 0..300 {
            let mut b = Board::start();
            for _ in 0..(n % 46) {
                b.legal_moves_dedup_into(&mut list);
                if list.is_empty() || b.game_over() {
                    break;
                }
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                b = b.make(list.move_at(x as usize % list.len()));
            }
            b.legal_moves_dedup_into(&mut list);
            let in5 = (b.empty().count_ones() as usize * nb / 50).min(nb - 1) == 13;
            for i in 0..list.len() {
                let m = list.move_at(i);
                let mut kc = super::KCache::new();
                let (a, c, d) = (single.logit_k(&b, m, &mut kc), copies.logit_k(&b, m, &mut kc), bumped.logit_k(&b, m, &mut kc));
                assert_eq!(a.to_bits(), c.to_bits(), "copies must equal the single head");
                if in5 {
                    assert!((d - a - 1.0).abs() < 1e-5, "bucket 13 position must see the bumped bias");
                    seen[1] = true;
                } else {
                    assert_eq!(a.to_bits(), d.to_bits(), "other buckets untouched");
                    seen[0] = true;
                }
            }
        }
        assert!(seen[0] && seen[1], "both cases exercised");
    }

    #[test]
    fn tpol5_loads_without_its_see_block() {
        let a = super::Head::from_bytes(&synth_bytes(false)).expect("TPOL3");
        let b = super::Head::from_bytes(&synth_bytes(true)).expect("TPOL5");
        assert_eq!(a.bias, b.bias);
        assert_eq!(a.g, b.g);
        assert_eq!(a.td, b.td);
        assert_eq!(a.tor, b.tor);
        let mut bad = synth_bytes(false);
        bad[8..12].copy_from_slice(&4u32.to_le_bytes());
        assert!(super::Policy::from_bytes(&bad).is_err(), "dim > 0 must be refused");
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
        let synth = super::Policy::from_bytes(&synth_bytes(false)).expect("TPOL3");
        let mut list = MoveList::new();
        for pol in [bundled, &synth] {
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
                let pol = pol.head(&b);
                let me = b.me();
                let opp = b.opp();
                let empty = b.empty();
                let mut targets = crate::board::reach_union(me) & empty;
                while targets != 0 {
                    let to = targets.trailing_zeros() as u8;
                    targets &= targets - 1;
                    let r1_to = RING1[to as usize];
                    let flips = ((opp & r1_to).count_ones() as usize).min(8);
                    let tholes = ((r1_to & empty).count_ones() as usize).min(8);
                    let k3_to = super::k3code_pub(me, opp, to);
                    let clones = me & r1_to;
                    if clones != 0 {
                        let m = crate::board::Move { from: clones.trailing_zeros() as u8, to };
                        let expected = pol.logit_k(&b, m, &mut kc);
                        let actual = pol.clone_logit_pre(to, flips, tholes, k3_to);
                        assert_eq!(actual.to_bits(), expected.to_bits(), "clone mv {m:?}");
                    }
                    let j_dest = pol.jump_dest_parts(to, flips, tholes, k3_to);
                    let mut jumps = me & RING2[to as usize];
                    while jumps != 0 {
                        let from = jumps.trailing_zeros() as u8;
                        jumps &= jumps - 1;
                        let m = crate::board::Move { from, to };
                        let fh = ((RING1[from as usize] & empty).count_ones() as usize).min(8);
                        let k3_from = super::k3code_pub(me, opp, from);
                        let j_orig = pol.jump_orig_parts(from, fh, k3_from);
                        let expected = pol.logit_k(&b, m, &mut kc);
                        let actual = pol.jump_finish(j_dest, j_orig);
                        assert_eq!(actual.to_bits(), expected.to_bits(), "jump mv {m:?}");
                    }
                }
            }
        }
    }
    #[test]
    fn tpol6_advance_root_unloads_dead_and_reloads() {
        let one = synth_bytes(false);
        let body = &one[12..];
        let nb = 16usize;
        let mut raw = b"TPOL6\0\0\0".to_vec();
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&(nb as u32).to_le_bytes());
        for _ in 0..nb {
            raw.extend_from_slice(body);
        }
        let mut pol = super::Policy::from_bytes(&raw).expect("TPOL6");
        let ref_head = super::Head::from_bytes(&one).expect("TPOL3");
        // Starts lazy: 0 slots decoded until advance_root
        assert_eq!(pol.loaded_buckets(), 0);
        // Root at 35 empties -> bucket 11, lookahead 4 -> [7..=11] = 5 loaded
        pol.advance_root(35, 4);
        assert_eq!(pol.loaded_buckets(), 5);
        // Root advances to 20 empties -> bucket 6 -> [2..=6], buckets 7..=11 unloaded (still 5 loaded, never 10)
        pol.advance_root(20, 4);
        assert_eq!(pol.loaded_buckets(), 5);
        for b in 7..nb {
            assert!(pol.slots[b].get().is_none(), "dead bucket {b} must be unloaded");
        }
        // New game reset (`uainewgame`): reloads high bucket cleanly and stays bit-identical
        pol.advance_root(45, 3);
        let b = crate::board::Board::start();
        let mut list = crate::board::MoveList::new();
        b.legal_moves_dedup_into(&mut list);
        let mut kc = super::KCache::new();
        for i in 0..list.len() {
            let m = list.move_at(i);
            assert_eq!(pol.logit_k(&b, m, &mut kc).to_bits(), ref_head.logit_k(&b, m, &mut kc).to_bits());
        }
        // Extreme all-clone extension below lookahead lazy-loads on demand
        pol.advance_root(35, 1);
        let b_low = pol.head_bucket(2);
        assert_eq!(b_low.bias, ref_head.bias);
    }
}

impl Head {
    pub fn load(path: &str) -> std::io::Result<Head> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(d: &[u8]) -> std::io::Result<Head> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        if d.len() < 12 {
            return Err(bad("bad policy magic"));
        }
        let nsee = match &d[..8] {
            b"TPOL3\0\0\0" => 0,
            b"TPOL5\0\0\0" => NSEE,
            _ => return Err(bad("bad policy magic (only TPOL3/TPOL5 heads exist)")),
        };
        if u32::from_le_bytes(d[8..12].try_into().unwrap()) != 0 {
            return Err(bad("policy dim must be 0 (no accumulator input)"));
        }
        let n1 = SLOTS + NSCAL + nsee;
        if d.len() != 12 + 4 * (n1 + (ND + NO) * P3) {
            return Err(bad("bad policy size"));
        }
        let f: Vec<f32> = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(Head::from_f32(&f, nsee))
    }

    /// One head from its f32 block: bias[98], g[45], (SEE block of `nsee` values, skipped), dest[18][6561], orig[9][6561].
    pub(crate) fn from_f32(f: &[f32], nsee: usize) -> Head {
        let n1 = SLOTS + NSCAL + nsee;
        let mut bias = [0f32; SLOTS];
        bias.copy_from_slice(&f[..SLOTS]);
        let mut g = [0f32; NSCAL];
        g.copy_from_slice(&f[SLOTS..SLOTS + NSCAL]);
        let quant = |x: &[f32]| -> (Vec<i8>, f32) {
            let max_abs = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_abs <= 0.0 {
                return (vec![0i8; x.len()], 0.0);
            }
            let inv_q = max_abs / 127.0;
            let q = 127.0 / max_abs;
            let v = x.iter().map(|&w| (w * q).round().clamp(-127.0, 127.0) as i8).collect();
            (v, inv_q)
        };
        let (td, inv_qd) = quant(&f[n1..n1 + ND * P3]);
        let (tor, inv_qo) = quant(&f[n1 + ND * P3..n1 + (ND + NO) * P3]);
        Head { bias, g, inv_qd, inv_qo, td, tor }
    }

    /// Decode one head directly from raw little-endian f32 bytes (`per` floats, no SEE block) with zero temp Vec alloc.
    fn from_le_f32_bytes(raw: &[u8]) -> Head {
        #[inline(always)]
        fn f_at(raw: &[u8], i: usize) -> f32 {
            let o = 4 * i;
            f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]])
        }
        let mut bias = [0f32; SLOTS];
        for (i, dst) in bias.iter_mut().enumerate() {
            *dst = f_at(raw, i);
        }
        let mut g = [0f32; NSCAL];
        for (i, dst) in g.iter_mut().enumerate() {
            *dst = f_at(raw, SLOTS + i);
        }
        let quant_slice = |off: usize, len: usize| -> (Vec<i8>, f32) {
            let mut max_abs = 0.0f32;
            for i in 0..len {
                let v = f_at(raw, off + i).abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            if max_abs <= 0.0 {
                return (vec![0i8; len], 0.0);
            }
            let inv_q = max_abs / 127.0;
            let q = 127.0 / max_abs;
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                out.push((f_at(raw, off + i) * q).round().clamp(-127.0, 127.0) as i8);
            }
            (out, inv_q)
        };
        let n1 = SLOTS + NSCAL;
        let (td, inv_qd) = quant_slice(n1, ND * P3);
        let (tor, inv_qo) = quant_slice(n1 + ND * P3, NO * P3);
        Head { bias, g, inv_qd, inv_qo, td, tor }
    }

    /// Logit of move `m` in `b`.
    #[inline]
    pub fn logit(&self, b: &Board, m: Move) -> f32 {
        self.logit_k(b, m, &mut KCache::new())
    }

    /// `logit` with a per-node code cache (one KCache per node, same board for every move). Bit-identical to the
    /// ordering loop's `clone_logit_pre` / `jump_finish` (test precomputed_logits_match_logit_k).
    #[inline]
    pub fn logit_k(&self, b: &Board, m: Move, kc: &mut KCache) -> f32 {
        let clone = m.is_clone();
        let c = clone as usize;
        let empty = b.empty();
        let flips = (b.captures(m.to) as usize).min(8);
        let tholes = ((RING1[m.to as usize] & empty).count_ones() as usize).min(8);
        let slot = m.to as usize + 49 * (!clone) as usize;
        let (me, opp) = (b.me(), b.opp());
        let ds = CLS[m.to as usize] as usize * 2 + (!clone) as usize;
        let mut z = (self.bias[slot] + self.g[c * 9 + flips]) + self.g[18 + c * 9 + tholes];
        z += self.td[ds * P3 + kc.get(me, opp, m.to)] as f32 * self.inv_qd;
        if !clone {
            let orig_z = self.g[36 + ((RING1[m.from as usize] & empty).count_ones() as usize).min(8)]
                + self.tor[CLS[m.from as usize] as usize * P3 + kc.get(me, opp, m.from)] as f32 * self.inv_qo;
            z += orig_z;
        }
        z
    }

    /// Logit of a clone landing on `to` from precomputed per-destination features. Bit-identical to `logit_k`.
    #[inline(always)]
    pub(crate) fn clone_logit_pre(&self, to: u8, flips: usize, tholes: usize, k3_to: usize) -> f32 {
        let slot = to as usize;
        let mut z = (self.bias[slot] + self.g[9 + flips]) + self.g[27 + tholes];
        z += self.td[CLS[slot] as usize * 2 * P3 + k3_to] as f32 * self.inv_qd;
        z
    }

    /// Destination-only total (prefix + quantized dest pattern term) for all jumps landing on `to`.
    #[inline(always)]
    pub(crate) fn jump_dest_parts(&self, to: u8, flips: usize, tholes: usize, k3_to: usize) -> f32 {
        let mut z = (self.bias[to as usize + 49] + self.g[flips]) + self.g[18 + tholes];
        z += self.td[(CLS[to as usize] as usize * 2 + 1) * P3 + k3_to] as f32 * self.inv_qd;
        z
    }

    /// Origin-only total (scalar + quantized origin pattern term) for jumps leaving `from`.
    #[inline(always)]
    pub(crate) fn jump_orig_parts(&self, from: u8, from_holes: usize, k3_from: usize) -> f32 {
        self.g[36 + from_holes] + self.tor[CLS[from as usize] as usize * P3 + k3_from] as f32 * self.inv_qo
    }

    /// Finish a jump logit from precomputed destination + origin totals. Same IEEE-754 order as `logit_k`.
    #[inline(always)]
    pub(crate) fn jump_finish(&self, j_dest: f32, j_orig: f32) -> f32 {
        j_dest + j_orig
    }
}

enum PolicyBytes {
    Static(&'static [u8]),
    Owned(std::sync::Arc<[u8]>),
}

impl PolicyBytes {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            PolicyBytes::Static(s) => s,
            PolicyBytes::Owned(v) => v,
        }
    }
}

/// D1320 / D1330 / D1332: phase-bucketed policy (`TPOL6`, `bucket = empties * nb / 50`) with `i8` quantized pattern tables
/// and lazy per-bucket slots so dead high buckets (`b > b_root`) unload as the game progresses and reload seamlessly on
/// `uainewgame` (or lazy-load if a deep tactical line crosses past `lookahead`).
pub struct Policy {
    nb: usize,
    raw: Option<(PolicyBytes, usize)>,
    slots: Vec<std::sync::OnceLock<Box<Head>>>,
}

impl Policy {
    pub fn load(path: &str) -> std::io::Result<Policy> {
        let bytes = std::fs::read(path)?;
        Self::from_storage(PolicyBytes::Owned(std::sync::Arc::from(bytes)))
    }

    pub fn from_bytes(d: &[u8]) -> std::io::Result<Policy> {
        Self::from_storage(PolicyBytes::Owned(std::sync::Arc::from(d)))
    }

    fn from_static_bytes(d: &'static [u8]) -> std::io::Result<Policy> {
        Self::from_storage(PolicyBytes::Static(d))
    }

    fn from_storage(raw_bytes: PolicyBytes) -> std::io::Result<Policy> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let d = raw_bytes.as_slice();
        if d.len() >= 16 && &d[..8] == b"TPOL6\0\0\0" {
            if u32::from_le_bytes(d[8..12].try_into().unwrap()) != 0 {
                return Err(bad("policy dim must be 0 (no accumulator input)"));
            }
            let nb = u32::from_le_bytes(d[12..16].try_into().unwrap()) as usize;
            let per = SLOTS + NSCAL + (ND + NO) * P3;
            if nb == 0 || nb > 64 || d.len() != 16 + 4 * per * nb {
                return Err(bad("bad TPOL6 size"));
            }
            let slots = (0..nb).map(|_| std::sync::OnceLock::new()).collect();
            return Ok(Policy { nb, raw: Some((raw_bytes, per)), slots });
        }
        let slot = std::sync::OnceLock::new();
        let _ = slot.set(Box::new(Head::from_bytes(d)?));
        Ok(Policy { nb: 1, raw: None, slots: vec![slot] })
    }

    /// Number of phase buckets (1 for TPOL3 / TPOL5).
    pub fn buckets(&self) -> usize {
        self.nb
    }

    /// Number of bucket heads currently resident in heap memory.
    pub fn loaded_buckets(&self) -> usize {
        self.slots.iter().filter(|s| s.get().is_some()).count()
    }

    /// Decode bucket `bk` from raw TPOL6 bytes.
    fn decode_bucket(&self, bk: usize) -> Box<Head> {
        let (raw, per) = self.raw.as_ref().expect("TPOL6 raw bytes");
        let d = raw.as_slice();
        let start = 16 + 4 * per * bk;
        Box::new(Head::from_le_f32_bytes(&d[start..start + 4 * per]))
    }

    #[inline(always)]
    fn head_bucket(&self, bk: usize) -> &Head {
        let slot = &self.slots[bk];
        if let Some(h) = slot.get() {
            return h;
        }
        slot.get_or_init(|| self.decode_bucket(bk))
    }

    /// Unload unreachable buckets `b > b_root` (stones never decrease in Ataxx) and pre-warm `[b_root - lookahead ..= b_root]`.
    pub fn advance_root(&mut self, root_empties: usize, lookahead: usize) {
        if self.nb <= 1 {
            return;
        }
        let b_root = (root_empties * self.nb / 50).min(self.nb - 1);
        for k in (b_root + 1)..self.nb {
            if self.slots[k].get().is_some() {
                self.slots[k] = std::sync::OnceLock::new();
            }
        }
        let lo = b_root.saturating_sub(lookahead);
        for k in lo..=b_root {
            let _ = self.head_bucket(k);
        }
    }

    /// The head of this position's phase bucket (empties * nb / 50, the NNUE's formula).
    #[inline(always)]
    pub fn head(&self, b: &Board) -> &Head {
        if self.nb == 1 {
            return self.head_bucket(0);
        }
        let e = b.empty().count_ones() as usize;
        self.head_bucket((e * self.nb / 50).min(self.nb - 1))
    }

    #[inline]
    pub fn logit(&self, b: &Board, m: Move) -> f32 {
        self.head(b).logit(b, m)
    }

    #[inline]
    pub fn logit_k(&self, b: &Board, m: Move, kc: &mut KCache) -> f32 {
        self.head(b).logit_k(b, m, kc)
    }
}

/// Open a fresh policy instance per current `--policy` / `TI_POLICY` configuration (None = `none` or invalid bundled blob).
pub fn open_configured() -> Option<Policy> {
    match spec().as_deref() {
        Some("none") => None,
        Some("bundled") | None => Policy::from_static_bytes(BUNDLED).ok(),
        Some(p) => Some(Policy::load(p).unwrap_or_else(|e| panic!("--policy / TI_POLICY {p}: {e}"))),
    }
}

/// Process-wide shared policy (for static callers); single-game `Searcher` instances prefer `open_configured()` so dead
/// buckets unload per root move.
pub fn global() -> Option<&'static Policy> {
    static P: std::sync::OnceLock<Option<Policy>> = std::sync::OnceLock::new();
    P.get_or_init(open_configured).as_ref()
}

static CONFIGURED: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Command-line `--policy <bundled|FILE|none>` (call once, before the first search); beats env TI_POLICY.
pub fn configure(spec: &str) {
    let _ = CONFIGURED.set(spec.to_string());
}

fn spec() -> Option<String> {
    CONFIGURED.get().cloned().or_else(|| std::env::var("TI_POLICY").ok())
}

/// True when the user chose a policy file explicitly. Since D821 the bundled head stays on with any big net (each net is
/// overridden on its own: `--policy none` turns it off); kept for callers that report the choice.
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
