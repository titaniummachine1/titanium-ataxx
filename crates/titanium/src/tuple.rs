//! Clean-room n-tuple eval (S20/S21): 36 anchors x 81 ternary 2x2 patterns
//! PLUS 80 adjacent-block pairs (8-cell, 4 directions x 6561, shared tables).
//!
//! Geometry (see `board::BLOCK2`/`PAIR_BLOCKS`): row-major anchors, canonical
//! cell/pair order, digits empty-or-blocker/own/enemy, stm-relative so one
//! table serves both colors. Weights are OUR OWN, trained by
//! training/train_tuple.py on our own data (MSE on score residuals, index-0
//! entries constrained to exactly 0 so empty patterns can be SKIPPED).
//!
//! File format `.tup` v4: magic `b"TUP4"` + bias LE i32 + 36*81 singles i8 +
//! 4 direction counts (LE u32) + concatenated active direction tables
//! (6561 i8 each). i8 because only 4 of 29k trained weights exceed ±127
//! (S21e, zero gate loss): H+V tables = 16KB total, L1-resident — the
//! lookups stop missing L2. Inactive directions cost zero lookups.

use crate::board::{block2_index, dist_union, jump_union, Board, PAIR_BLOCKS, RING1, SQ_BLOCKS};

/// Magic header of `.tup` weight files (v4: bias + i8 singles + sparse dirs).
pub const TUP_MAGIC: [u8; 4] = *b"TUP4";
/// v5: TUP4 prefix + i16 LE sections q2 (36x256) + env us/them (49x729 each)
/// + mob us/them (50x50 each). S38 enriched inputs (quat 2x2, env tallies,
/// mobility), own-trained weights.
/// v6: TUP5 + capn us/them (max conversion 0-8, 9 each) + conv us/them
/// (conversion-kernel LUT keys 0-255 each). User: max-conversion magnitude
/// + PEXT-key patterns as network inputs (stateless — last-move counts
/// would corrupt TT entries).
pub const TUP5_MAGIC: [u8; 4] = *b"TUP5";
pub const TUP6_MAGIC: [u8; 4] = *b"TUP6";
pub const TUP7_MAGIC: [u8; 4] = *b"TUP7";
pub const TUP8_MAGIC: [u8; 4] = *b"TUP8";
/// Piece-count bucket offsets (1..=49 used; index 0 constrained 0).
pub const BUCKETS: usize = 50;
/// 8-cell ternary pair patterns per direction table.
pub const PAIR_PATTERNS: usize = 81 * 81;
/// Geometry pair counts per direction (H, V, D1, D2).
pub const DIR_COUNTS: [usize; 4] = [24, 24, 16, 16];
/// Quaternary 2x2 patterns (empty/own/foe/blocker).
pub const Q2_PATTERNS: usize = 256;
/// Per-square env tallies (own x foe x blocker ring counts, 9^3).
pub const ENV_PATTERNS: usize = 49 * 9 * 9 * 9;
/// Mobility entries (phase x clone-target count).
pub const MOB_PATTERNS: usize = 50 * 50;

/// n-tuple weight table, i8 storage (~29KB max: 2.9KB singles + 26KB
/// pairs — L1-resident). Accumulation in i32 (84 terms x ±127, no overflow).
/// Inactive directions hold zeros and are never looked up.
/// Enriched families (v5) are i16 (~110KB): quat/env/mob own-trained tables.
/// Accumulation stays i32 (25 env terms x ±32767 max -- real weights are
/// small; debug_assert guards in tests, gates guard in production).
/// Per-anchor pairs (v7): 80 anchor-pair tables (the missing Moonbird
/// feature: position-dependent pair weights instead of 4 shared tables).
#[derive(Clone)]
pub struct TupleTable {
    bias: i32,
    single: Box<[[i8; 81]; 36]>,
    pair: Box<[[i8; PAIR_PATTERNS]; 4]>,
    dir_counts: [usize; 4],
    q2: Box<[[i16; Q2_PATTERNS]; 36]>,
    env_us: Box<[i16; ENV_PATTERNS]>,
    env_them: Box<[i16; ENV_PATTERNS]>,
    mob_us: Box<[i16; MOB_PATTERNS]>,
    mob_them: Box<[i16; MOB_PATTERNS]>,
    has_enriched: bool,
    /// Load-time allzero flag: the whole q2 quat family is exactly 0
    /// (champ TUP7: 9216/9216 zero — tree training never touched it).
    /// Skipping its splat + table is bit-identical (adds 0). Same idea as
    /// the idx0 training constraint, but discovered at load, not train.
    has_q2: bool,
    capn_us: Box<[i16; 9]>,
    capn_them: Box<[i16; 9]>,
    conv_us: Box<[i16; 256]>,
    conv_them: Box<[i16; 256]>,
    has_capconv: bool,
    pa: Box<[i8; PA_PATTERNS]>,
    has_pa: bool,
    bucket: Box<[i16; BUCKETS]>,
    has_bucket: bool,
}

/// Per-anchor pair entries (80 pairs x 6561 ternary patterns).
pub const PA_PATTERNS: usize = 80 * 81 * 81;

/// Conversion-kernel key for a square: foe occupancy in dr-outer ring
/// order (matches NEIGH_POS bit order used by extract8/PEXT; off-board
/// absent since we iterate NEIGH_CNT entries only). Bit i = i-th ring
/// square holds a foe stone.
#[inline]
fn conv_key(occ_foe: u64, sq: usize) -> usize {
    let mut key = 0usize;
    for i in 0..crate::board::NEIGH_CNT[sq] as usize {
        let s = crate::board::NEIGH_POS[sq][i] as usize;
        if occ_foe & (1u64 << s) != 0 {
            key |= 1 << i;
        }
    }
    key
}

/// Max conversion available to `own` stones (foe = enemy bits).
fn maxconv(own: u64, foe: u64, empty: u64) -> usize {
    let mut best = 0u32;
    let mut land = empty & (dist_union(own, 1) | crate::board::jump_union(own));
    while land != 0 {
        let sq = land.trailing_zeros() as usize;
        land &= land - 1;
        let c = (foe & RING1[sq]).count_ones();
        if c > best {
            best = c;
        }
    }
    best.min(8) as usize
}

/// Ternary anchor mult (1/3/9/27 from SQ_BLOCKS) -> quaternary mult.
#[inline]
fn qmult(m3: u8) -> u8 {
    match m3 {
        1 => 1,
        3 => 4,
        9 => 16,
        27 => 64,
        _ => 0,
    }
}

/// Square index for env tallies: sq*729 + own*81 + foe*9 + blocker.
#[inline]
fn env_index(sq: usize, o: u32, e: u32, bl: u32) -> usize {
    sq * 729 + (o as usize) * 81 + (e as usize) * 9 + bl as usize
}

impl TupleTable {
    /// All-zero table: forward() == 0 everywhere (integration proof that
    /// wiring the term changes nothing by itself).
    pub fn zeros() -> TupleTable {
        TupleTable {
            bias: 0,
            single: Box::new([[0; 81]; 36]),
            pair: Box::new([[0; PAIR_PATTERNS]; 4]),
            dir_counts: DIR_COUNTS,
            q2: Box::new([[0; Q2_PATTERNS]; 36]),
            env_us: Box::new([0; ENV_PATTERNS]),
            env_them: Box::new([0; ENV_PATTERNS]),
            mob_us: Box::new([0; MOB_PATTERNS]),
            mob_them: Box::new([0; MOB_PATTERNS]),
            has_enriched: false,
            has_q2: false,
            capn_us: Box::new([0; 9]),
            capn_them: Box::new([0; 9]),
            conv_us: Box::new([0; 256]),
            conv_them: Box::new([0; 256]),
            has_capconv: false,
            pa: Box::new([0; PA_PATTERNS]),
            has_pa: false,
            bucket: Box::new([0; BUCKETS]),
            has_bucket: false,
        }
    }

    /// Load `.tup` weights (v4 classic, v5 enriched, v6 +capn/conv, v7
    /// +per-anchor pairs — backward compatible, missing sections read as
    /// zeros).
    pub fn load(path: &str) -> std::io::Result<TupleTable> {
        let buf = std::fs::read(path)?;
        if buf.len() < 8
            || (buf[0..4] != TUP_MAGIC
                && buf[0..4] != TUP5_MAGIC
                && buf[0..4] != TUP6_MAGIC
                && buf[0..4] != TUP7_MAGIC
                && buf[0..4] != TUP8_MAGIC)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad .tup magic (want TUP4/TUP5/TUP6/TUP7/TUP8)",
            ));
        }
        let rd32 = |o: usize| i32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        let rd16 = |o: usize| i16::from_le_bytes([buf[o], buf[o + 1]]);
        let bias = rd32(4);
        let mut o = 8usize;
        let need = |o: usize, n: usize| {
            if o + n > buf.len() {
                Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "truncated .tup",
                ))
            } else {
                Ok(())
            }
        };
        let mut single = Box::new([[0i8; 81]; 36]);
        for a in single.iter_mut() {
            for v in a.iter_mut() {
                need(o, 1)?;
                *v = buf[o] as i8;
                o += 1;
            }
        }
        need(o, 16)?;
        let mut dir_counts = [0usize; 4];
        for d in 0..4 {
            dir_counts[d] = rd32(o) as usize;
            o += 4;
            if dir_counts[d] > DIR_COUNTS[d] {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("dir {d} count {} > geometry {}", dir_counts[d], DIR_COUNTS[d]),
                ));
            }
        }
        let mut pair = Box::new([[0i8; PAIR_PATTERNS]; 4]);
        for d in 0..4 {
            for v in pair[d].iter_mut() {
                need(o, 1)?;
                *v = buf[o] as i8;
                o += 1;
            }
        }
        let mut t = TupleTable {
            bias,
            single,
            pair,
            dir_counts,
            q2: Box::new([[0; Q2_PATTERNS]; 36]),
            env_us: Box::new([0; ENV_PATTERNS]),
            env_them: Box::new([0; ENV_PATTERNS]),
            mob_us: Box::new([0; MOB_PATTERNS]),
            mob_them: Box::new([0; MOB_PATTERNS]),
            has_enriched: false,
            has_q2: false,
            capn_us: Box::new([0; 9]),
            capn_them: Box::new([0; 9]),
            conv_us: Box::new([0; 256]),
            conv_them: Box::new([0; 256]),
            has_capconv: false,
            pa: Box::new([0; PA_PATTERNS]),
            has_pa: false,
            bucket: Box::new([0; BUCKETS]),
            has_bucket: false,
        };
        if buf[0..4] == TUP5_MAGIC || buf[0..4] == TUP6_MAGIC || buf[0..4] == TUP7_MAGIC || buf[0..4] == TUP8_MAGIC {
            for a in t.q2.iter_mut() {
                for v in a.iter_mut() {
                    need(o, 2)?;
                    *v = rd16(o);
                    o += 2;
                }
            }
            for table in [&mut t.env_us, &mut t.env_them].iter_mut() {
                for v in table.iter_mut() {
                    need(o, 2)?;
                    *v = rd16(o);
                    o += 2;
                }
            }
            for table in [&mut t.mob_us, &mut t.mob_them].iter_mut() {
                for v in table.iter_mut() {
                    need(o, 2)?;
                    *v = rd16(o);
                    o += 2;
                }
            }
            t.has_enriched = true;
        }
        // TUP7 files carry capn/conv sections before pa (same layout as
        // TUP6 + pa appended — proven by the trainer's own dump_tup/load_tup
        // on exp/tuple-eval: payload = TUP4 + TUP5 + capn + conv + pa, and
        // every TUP7 blob on disk is exactly 726440 = TUP6-size + pa).
        // The old check read them for TUP6_MAGIC only, so on TUP7 the pa
        // table loaded 1060 bytes shifted (all 80 tables scrambled) and
        // trained capn/conv weights were dropped. Short TUP7 files fail
        // LOUDLY here rather than misalign silently.
        if buf[0..4] == TUP7_MAGIC
            && buf.len() - o != 9 * 2 * 2 + 256 * 2 * 2 + PA_PATTERNS
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "TUP7 size mismatch (want full TUP6 sections + pa)",
            ));
        }
        let has_cc_sections =
            buf[0..4] == TUP6_MAGIC || buf[0..4] == TUP7_MAGIC || buf[0..4] == TUP8_MAGIC;
        if has_cc_sections {
            for table in [&mut t.capn_us, &mut t.capn_them].iter_mut() {
                for v in table.iter_mut() {
                    need(o, 2)?;
                    *v = rd16(o);
                    o += 2;
                }
            }
            for table in [&mut t.conv_us, &mut t.conv_them].iter_mut() {
                for v in table.iter_mut() {
                    need(o, 2)?;
                    *v = rd16(o);
                    o += 2;
                }
            }
            // Content-gated like has_q2 (W4 pattern): all-zero sections add
            // exactly 0, so the tree skips them bit-identically.
            t.has_capconv = t.capn_us.iter().any(|&w| w != 0)
                || t.capn_them.iter().any(|&w| w != 0)
                || t.conv_us.iter().any(|&w| w != 0)
                || t.conv_them.iter().any(|&w| w != 0);
        }
        if buf[0..4] == TUP7_MAGIC || buf[0..4] == TUP8_MAGIC {
            for v in t.pa.iter_mut() {
                need(o, 1)?;
                *v = buf[o] as i8;
                o += 1;
            }
            t.has_pa = true;
        }
        if buf[0..4] == TUP8_MAGIC {
            if buf.len() - o != BUCKETS * 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "TUP8 size mismatch (want TUP7 + piece-count buckets)",
                ));
            }
            for v in t.bucket.iter_mut() {
                *v = rd16(o);
                o += 2;
            }
            t.has_bucket = t.bucket.iter().any(|&w| w != 0);
        }
        // Load-time sparsity flags (winter-clean W4): whole-zero families
        // are skipped bit-identically in the tree (they add 0 by definition).
        // One scan at load; the tree then branches on constants per net.
        t.has_q2 = t.q2.iter().any(|a| a.iter().any(|&w| w != 0));
        Ok(t)
    }

    /// stm-relative tuple score. Splat accumulation (S21b): iterate live
    /// stones (both colors), splat digit*mult into the blocks covering each
    /// stone — no branches, no variable shifts. Index-0 patterns contribute
    /// exactly 0 (training constraint) and are skipped in the sums.
    #[inline]
    pub fn forward(&self, b: &Board) -> i32 {
        // Test/standalone path: compute both sides' landings locally, then
        // share them with capconv. The search tree uses `forward_fast`
        // (evaluate's landings passed in — zero recompute, bit-identical).
        let ou = b.me();
        let ot = b.opp();
        let empty = !ou & !ot & !b.blockers & crate::board::FULL;
        let land_me = empty & (dist_union(ou, 1) | jump_union(ou));
        let land_opp = empty & (dist_union(ot, 1) | jump_union(ot));
        self.forward_fast(b, land_me, land_opp)
    }

    /// Tree path: `land_me`/`land_opp` are evaluate's already-computed
    /// landing sets (same position, same sets — no recompute). Bit-identical
    /// to `forward` (enforced by `forward_fast_matches_forward` test).
    #[inline]
    pub fn forward_fast(&self, b: &Board, land_me: u64, land_opp: u64) -> i32 {
        let (sq_blocks, sq_n) = &SQ_BLOCKS;
        let mut blocks = [0u8; 36];
        // Quat 2x2 indices fused into the same stone passes (was 3 extra
        // traversals in forward_enriched). Guarded: non-enriched tables
        // (v4) skip the quat work entirely; all-zero q2 (champ TUP7) skips
        // it too — bit-identical, the table adds 0 everywhere (has_q2).
        let mut q = [0u8; 36];
        let enr_q = self.has_enriched && self.has_q2;
        let mut stones = b.me();
        while stones != 0 {
            let sq = stones.trailing_zeros() as usize;
            stones &= stones - 1;
            let n = sq_n[sq] as usize;
            let mut k = 0usize;
            while k < n {
                let (a, mult) = sq_blocks[sq][k];
                blocks[a as usize] += mult; // own digit 1
                if enr_q {
                    q[a as usize] += qmult(mult);
                }
                k += 1;
            }
        }
        stones = b.opp();
        while stones != 0 {
            let sq = stones.trailing_zeros() as usize;
            stones &= stones - 1;
            let n = sq_n[sq] as usize;
            let mut k = 0usize;
            while k < n {
                let (a, mult) = sq_blocks[sq][k];
                blocks[a as usize] += 2 * mult; // enemy digit 2
                if enr_q {
                    q[a as usize] += 2 * qmult(mult);
                }
                k += 1;
            }
        }
        if enr_q {
            let mut bb = b.blockers;
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                let n = sq_n[sq] as usize;
                let mut k = 0usize;
                while k < n {
                    let (a, mult) = sq_blocks[sq][k];
                    q[a as usize] += 3 * qmult(mult); // blocker digit 3
                    k += 1;
                }
            }
        }
        let mut s = self.bias;
        for (a, idx) in blocks.iter().enumerate() {
            if *idx != 0 {
                s += self.single[a][*idx as usize] as i32;
            }
        }
        let (pairs, _) = &PAIR_BLOCKS;
        for d in 0..4 {
            let t = &self.pair[d];
            for i in 0..self.dir_counts[d] {
                let (bx, by) = (blocks[pairs[d][i].0 as usize], blocks[pairs[d][i].1 as usize]);
                if bx | by != 0 {
                    s += t[bx as usize + 81 * by as usize] as i32;
                }
            }
        }
        // Per-anchor pairs (S47 the missing feature): same indices, but
        // each of the 80 anchor-pairs has its OWN table (position-dependent
        // weights like Moonbird) instead of 4 shared direction tables.
        if self.has_pa {
            const BASE: [usize; 4] = [0, 24, 48, 64];
            for d in 0..4 {
                for i in 0..self.dir_counts[d] {
                    let (bx, by) =
                        (blocks[pairs[d][i].0 as usize], blocks[pairs[d][i].1 as usize]);
                    if bx | by != 0 {
                        s += self.pa[(BASE[d] + i) * PAIR_PATTERNS + bx as usize
                            + 81 * by as usize] as i32;
                    }
                }
            }
        }
        if self.has_enriched {
            s += self.forward_enriched(b, &q);
        }
        if self.has_capconv {
            s += self.forward_capconv(b, land_me, land_opp);
        }
        if self.has_bucket {
            s += self.bucket[(b.me() | b.opp()).count_ones() as usize] as i32;
        }
        s
    }

    /// S38 enriched families: quat 2x2 table + per-stone env tallies +
    /// mobility. The quat splat now arrives prebuilt (`q`, fused into
    /// forward_fast's stone passes). Indices mirror training/train_tuple.py
    /// exactly (q2_indices, env_tables, mob_tables). Skips mirror the
    /// zero-constraints.
    fn forward_enriched(&self, b: &Board, q: &[u8; 36]) -> i32 {
        let mut s: i32 = 0;
        // All-zero q2 (champ) skips here — bit-identical, adds 0.
        if self.has_q2 {
            for (a, idx) in q.iter().enumerate() {
                if *idx != 0 {
                    s += self.q2[a][*idx as usize] as i32;
                }
            }
        }
        // ENV per-stone tallies (own stones -> env_us, enemy -> env_them).
        let ou = b.me();
        let ot = b.opp();
        let blk = b.blockers;
        let mut stones = ou;
        while stones != 0 {
            let sq = stones.trailing_zeros() as usize;
            stones &= stones - 1;
            let ring = RING1[sq];
            let o = (ring & ou).count_ones();
            let e = (ring & ot).count_ones();
            let bl = (ring & blk).count_ones();
            if o | e | bl != 0 {
                s += self.env_us[env_index(sq, o, e, bl)] as i32;
            }
        }
        stones = ot;
        while stones != 0 {
            let sq = stones.trailing_zeros() as usize;
            stones &= stones - 1;
            let ring = RING1[sq];
            let o = (ring & ot).count_ones();
            let e = (ring & ou).count_ones();
            let bl = (ring & blk).count_ones();
            if o | e | bl != 0 {
                s += self.env_them[env_index(sq, o, e, bl)] as i32;
            }
        }
        // MOB phase x counts.
        let total = ou | ot | blk;
        let empty = !total & crate::board::FULL;
        let phase = total.count_ones().min(49) as usize;
        let cu = (dist_union(ou, 1) & empty).count_ones().min(49) as usize;
        let ct = (dist_union(ot, 1) & empty).count_ones().min(49) as usize;
        if cu != 0 {
            s += self.mob_us[phase * 50 + cu] as i32;
        }
        if ct != 0 {
            s += self.mob_them[phase * 50 + ct] as i32;
        }
        s
    }

    /// S45 capn/conv forward: max-conversion magnitudes (0-8) + per-square
    /// conversion-kernel LUT keys (foe pattern in NEIGH_POS bit order).
    /// Stateless (position-only) so TT stays sound. Skips mirror the
    /// zero-constraints (0 entries).
    fn forward_capconv(&self, b: &Board, land_me: u64, land_opp: u64) -> i32 {
        let ou = b.me();
        let ot = b.opp();
        let mut s: i32 = 0;
        // Max conversion available per side (landing sets shared with the
        // caller — evaluate already computed these exact bitboards).
        let mut mu = 0u32;
        let mut ls = land_me;
        while ls != 0 {
            let sq = ls.trailing_zeros() as usize;
            ls &= ls - 1;
            mu = mu.max((ot & RING1[sq]).count_ones());
        }
        let mut mt = 0u32;
        ls = land_opp;
        while ls != 0 {
            let sq = ls.trailing_zeros() as usize;
            ls &= ls - 1;
            mt = mt.max((ou & RING1[sq]).count_ones());
        }
        if mu != 0 {
            s += self.capn_us[mu.min(8) as usize] as i32;
        }
        if mt != 0 {
            s += self.capn_them[mt.min(8) as usize] as i32;
        }
        // Conversion-kernel keys per square (foe pattern, NEIGH_POS order).
        // NOTE (PARKED, not rejected): BMI2 PEXT tried, hoisted detect —
        // 0.88/0.93 vs 0.98/1.00 scalar on this CPU (latency-bound in the
        // real loop; micro still 702M vs 99M/s). Scalar stays default.
        // Proof `pext_matches_loop` + genbench PEXT micro stay prepared
        // for retest (other CPUs, restructured loops). Rejection is only
        // for what makes the engine worse — this is parked for FOMO.
        let mut sq = 0usize;
        while sq < 49 {
            let (ku, kt) = (conv_key(ot, sq), conv_key(ou, sq));
            if ku != 0 {
                s += self.conv_us[ku] as i32;
            }
            if kt != 0 {
                s += self.conv_them[kt] as i32;
            }
            sq += 1;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_forward_is_zero() {
        let t = TupleTable::zeros();
        let b = Board::start();
        assert_eq!(t.forward(&b), 0);
    }

    #[test]
    fn startpos_block_indices() {
        // Cross-language anchor guard (mirrored in train_tuple.py): black
        // a1 + white g7, black to move. Anchor (5,0) -> 9 (own digit 1 in
        // 3rd cell), anchor (0,5) -> 6 (enemy digit 2 in 2nd cell).
        let b = Board::start();
        for a in 0..36 {
            let idx = block2_index(b.occ[0], b.occ[1], a);
            let (r, c) = (a / 6, a % 6);
            let want = if r == 5 && c == 0 {
                9
            } else if r == 0 && c == 5 {
                6
            } else {
                0
            };
            assert_eq!(idx, want, "anchor {a}");
        }
    }

    #[test]
    fn pair_geometry_counts() {
        // 24 H + 24 V + 16 D1 + 16 D2 = 80, all anchor indices in range,
        // pairs within their direction's shape.
        let (pairs, counts) = &PAIR_BLOCKS;
        assert_eq!(*counts, [24, 24, 16, 16]);
        for d in 0..4 {
            for i in 0..counts[d] {
                let (x, y) = pairs[d][i];
                assert!(x < 36 && y < 36, "dir {d} pair {i}: {x} {y}");
                let (xr, xc) = (x / 6, x % 6);
                let (yr, yc) = (y / 6, y % 6);
                match d {
                    0 => assert!(xr == yr && yc == xc + 2),
                    1 => assert!(xc == yc && yr == xr + 2),
                    2 => assert!(yr == xr + 2 && yc == xc + 2),
                    _ => assert!(yr == xr + 2 && xc == yc + 2),
                }
            }
        }
    }

    #[test]
    fn qmult_maps_ternary_to_quat() {
        assert_eq!((qmult(1), qmult(3), qmult(9), qmult(27)), (1, 4, 16, 64));
    }

    #[test]
    fn env_index_layout() {
        // sq*729 + own*81 + foe*9 + blocker; empty rings land on sq*729.
        assert_eq!(env_index(42, 0, 0, 0), 42 * 729);
        assert_eq!(env_index(6, 1, 2, 3), 6 * 729 + 81 + 18 + 3);
    }

    #[test]
    fn enriched_single_entry_picks() {
        // Black a1 + white b1, black stm (49 contiguous chars + side).
        let b = Board::from_str(
            "..........................................xo..... b",
        )
        .unwrap();
        // Black a1 ring: b1 foe (1), a2/b2 empty
        // -> env_us[42*729+0*81+1*9+0] is the only nonzero env hit.
        let mut t = TupleTable::zeros();
        t.has_enriched = true;
        t.env_us[42 * 729 + 1 * 9] = 5;
        // Everything else reads zero tables (mob fires on real boards but
        // its tables are zero here; white's env_them hit reads 0).
        assert_eq!(t.forward(&b), 5);
        // Q2 spot: anchor 30 cells [35,36,42,43], quaternary mults 1/4/16/64:
        // a1 own 3rd cell -> 16, b1 foe 4th cell -> 2*64=128: idx = 144.
        let mut t2 = TupleTable::zeros();
        t2.has_enriched = true;
        t2.has_q2 = true;
        t2.q2[30][144] = 7;
        assert_eq!(t2.forward(&b), 7);
        // All-zero q2 skips bit-identically (W4 load-time flag).
        let mut t3 = TupleTable::zeros();
        t3.has_enriched = true;
        assert!(!t3.has_q2);
        assert_eq!(t3.forward(&b), 0);
        // The flag is load-bearing, not dead code: same nonzero weights
        // score differently with the flag on vs off (future training that
        // fills q2 automatically takes the full slow path — no silent
        // lobotomy). And zero weights score the same either way.
        let mut t4 = TupleTable::zeros();
        t4.has_enriched = true;
        t4.has_q2 = true; // all-zero table, flag forced on
        assert_eq!(t4.forward(&b), t3.forward(&b));
        t2.has_q2 = false; // nonzero weights, flag forced off
        assert_eq!(t2.forward(&b), 0);
        assert_ne!(t2.forward(&b), 7);
    }

    #[test]
    fn capconv_single_entry_picks() {
        // Black a1, white b1+a2, black stm (35 dots + "o......" + "xo.....").
        // Black maxconv = 2 (b2 takes a2+b1); white maxconv >= 1 but its
        // table is zero here.
        let b = Board::from_str(
            "...................................o......xo..... b",
        )
        .unwrap();
        let mut t = TupleTable::zeros();
        t.has_capconv = true;
        t.capn_us[2] = 7;
        assert_eq!(t.forward(&b), 7);
        // conv key at b2(36): dr-outer ring bits i3 (a2) + i6 (b1) = 72.
        // (No other square on this board has key 72 — verified by enumeration.)
        let mut t2 = TupleTable::zeros();
        t2.has_capconv = true;
        t2.conv_us[72] = 11;
        assert_eq!(t2.forward(&b), 11);
    }

    #[test]
    fn forward_fast_matches_forward() {
        // Shared-landings tree path must score bit-identically to the
        // standalone path (which computes landings locally).
        let mut t = TupleTable::zeros();
        t.has_capconv = true;
        t.capn_us[2] = 7;
        t.capn_them[3] = -5;
        t.conv_us[72] = 11;
        t.conv_them[9] = -3;
        let positions = [
            Board::start(),
            Board::from_str(
                "...................................o......xo..... b",
            )
            .unwrap(),
        ];
        for b in positions {
            let want = t.forward(&b);
            let ou = b.me();
            let ot = b.opp();
            let empty = !ou & !ot & !b.blockers & crate::board::FULL;
            let land_me = empty & (dist_union(ou, 1) | jump_union(ou));
            let land_opp = empty & (dist_union(ot, 1) | jump_union(ot));
            assert_eq!(t.forward_fast(&b, land_me, land_opp), want);
        }
    }

    #[test]
    fn per_anchor_pair_spot() {
        // Startpos: H pair ((5,0),(5,2)) = global pair 20, feats 9+0.
        // Rigged per-anchor entry must be picked exactly there.
        let b = Board::start();
        let mut t = TupleTable::zeros();
        t.has_pa = true;
        t.pa[20 * PAIR_PATTERNS + 9] = 5;
        assert_eq!(t.forward(&b), 5);
    }

    #[test]
    fn startpos_pair_spot() {
        // Startpos: only anchors (5,0) idx 9 and (0,5) idx 6 non-empty.
        // H pair ((5,0),(5,2)) = anchors 30,32: b1=9, b2=0 -> index 9.
        let (pairs, _) = &PAIR_BLOCKS;
        assert_eq!(pairs[0][5 * 4 + 0], (30, 32));
        let b = Board::start();
        let own = b.me();
        let enemy = b.opp();
        assert_eq!(block2_index(own, enemy, 30), 9);
        assert_eq!(block2_index(own, enemy, 32), 0);
    }
}
