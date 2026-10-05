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

use crate::board::{block2_index, dist_union, Board, PAIR_BLOCKS, RING1, SQ_BLOCKS};

/// Magic header of `.tup` weight files (v4: bias + i8 singles + sparse dirs).
pub const TUP_MAGIC: [u8; 4] = *b"TUP4";
/// v5: TUP4 prefix + i16 LE sections q2 (36x256) + env us/them (49x729 each)
/// + mob us/them (50x50 each). S38 enriched inputs (quat 2x2, env tallies,
/// mobility), own-trained weights.
pub const TUP5_MAGIC: [u8; 4] = *b"TUP5";
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
        }
    }

    /// Load `.tup` weights (v4 classic, v5 enriched — backward compatible).
    pub fn load(path: &str) -> std::io::Result<TupleTable> {
        let buf = std::fs::read(path)?;
        if buf.len() < 8 || (buf[0..4] != TUP_MAGIC && buf[0..4] != TUP5_MAGIC) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad .tup magic (want TUP4/TUP5)",
            ));
        }
        let rd32 = |o: usize| i32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
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
        };
        if buf[0..4] == TUP5_MAGIC {
            let rd16 = |o: usize| i16::from_le_bytes([buf[o], buf[o + 1]]);
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
        Ok(t)
    }

    /// stm-relative tuple score. Splat accumulation (S21b): iterate live
    /// stones (both colors), splat digit*mult into the blocks covering each
    /// stone — no branches, no variable shifts. Index-0 patterns contribute
    /// exactly 0 (training constraint) and are skipped in the sums.
    #[inline]
    pub fn forward(&self, b: &Board) -> i32 {
        let (sq_blocks, sq_n) = &SQ_BLOCKS;
        let mut blocks = [0u8; 36];
        let mut stones = b.me();
        while stones != 0 {
            let sq = stones.trailing_zeros() as usize;
            stones &= stones - 1;
            let n = sq_n[sq] as usize;
            let mut k = 0usize;
            while k < n {
                let (a, mult) = sq_blocks[sq][k];
                blocks[a as usize] += mult; // own digit 1
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
                k += 1;
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
        if self.has_enriched {
            s += self.forward_enriched(b, &blocks);
        }
        s
    }

    /// S38 enriched families: quat 2x2 splat + per-stone env tallies +
    /// mobility. Indices mirror training/train_tuple.py exactly (q2_indices,
    /// env_tables, mob_tables). Skips mirror the zero-constraints.
    fn forward_enriched(&self, b: &Board, _blocks: &[u8; 36]) -> i32 {
        let mut s: i32 = 0;
        // Q2 quat splat (base-4 digits; blocker digit 3).
        let (sq_blocks, sq_n) = &SQ_BLOCKS;
        let mut q = [0u8; 36];
        let mut stones = b.me();
        while stones != 0 {
            let sq = stones.trailing_zeros() as usize;
            stones &= stones - 1;
            let n = sq_n[sq] as usize;
            let mut k = 0usize;
            while k < n {
                let (a, m3) = sq_blocks[sq][k];
                q[a as usize] += qmult(m3);
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
                let (a, m3) = sq_blocks[sq][k];
                q[a as usize] += 2 * qmult(m3);
                k += 1;
            }
        }
        stones = b.blockers;
        while stones != 0 {
            let sq = stones.trailing_zeros() as usize;
            stones &= stones - 1;
            let n = sq_n[sq] as usize;
            let mut k = 0usize;
            while k < n {
                let (a, m3) = sq_blocks[sq][k];
                q[a as usize] += 3 * qmult(m3);
                k += 1;
            }
        }
        for (a, idx) in q.iter().enumerate() {
            if *idx != 0 {
                s += self.q2[a][*idx as usize] as i32;
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
        t2.q2[30][144] = 7;
        assert_eq!(t2.forward(&b), 7);
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
