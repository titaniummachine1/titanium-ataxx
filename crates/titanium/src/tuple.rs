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

use crate::board::{block2_index, Board, PAIR_BLOCKS, SQ_BLOCKS};

/// Magic header of `.tup` weight files (v4: bias + i8 singles + sparse dirs).
pub const TUP_MAGIC: [u8; 4] = *b"TUP4";
/// 8-cell ternary pair patterns per direction table.
pub const PAIR_PATTERNS: usize = 81 * 81;
/// Geometry pair counts per direction (H, V, D1, D2).
pub const DIR_COUNTS: [usize; 4] = [24, 24, 16, 16];

/// n-tuple weight table, i8 storage (~29KB max: 2.9KB singles + 26KB
/// pairs — L1-resident). Accumulation in i32 (84 terms x ±127, no overflow).
/// Inactive directions hold zeros and are never looked up.
#[derive(Clone)]
pub struct TupleTable {
    bias: i32,
    single: Box<[[i8; 81]; 36]>,
    pair: Box<[[i8; PAIR_PATTERNS]; 4]>,
    dir_counts: [usize; 4],
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
        }
    }

    /// Load `.tup` v4 weights.
    pub fn load(path: &str) -> std::io::Result<TupleTable> {
        let buf = std::fs::read(path)?;
        if buf.len() < 8 || buf[0..4] != TUP_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad .tup magic (want TUP4)",
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
        Ok(TupleTable {
            bias,
            single,
            pair,
            dir_counts,
        })
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
