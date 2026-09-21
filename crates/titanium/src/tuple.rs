//! Clean-room n-tuple eval (S20): 36 anchors x 81 ternary 2x2 patterns.
//!
//! Geometry (see `board::BLOCK2`/`block2_index`): row-major anchors, cells
//! in canonical order, digits empty-or-blocker/own/enemy, stm-relative so
//! one table serves both colors. Weights are OUR OWN, trained by
//! training/train_tuple.py on our own data (MSE on score residuals).
//!
//! File format `.tup`: magic `b"TUP1"` + 36*81 little-endian i32 (11,664 B
//! payload). Zeros = value-identical to the hand eval (integration proof).

use crate::board::{block2_index, Board};

/// Magic header of `.tup` weight files.
pub const TUP_MAGIC: [u8; 4] = *b"TUP1";
/// Payload: 36 anchors x 81 patterns of i32.
pub const TUP_LEN: usize = 36 * 81;

/// n-tuple weight table (11.7KB, L1-resident).
#[derive(Clone)]
pub struct TupleTable {
    w: Box<[[i32; 81]; 36]>,
}

impl TupleTable {
    /// All-zero table: forward() == 0 everywhere (integration proof that
    /// wiring the term changes nothing by itself).
    pub fn zeros() -> TupleTable {
        TupleTable {
            w: Box::new([[0; 81]; 36]),
        }
    }

    /// Load `.tup` weights (magic + 11,664 B LE i32).
    pub fn load(path: &str) -> std::io::Result<TupleTable> {
        let buf = std::fs::read(path)?;
        if buf.len() != 4 + TUP_LEN * 4 || buf[0..4] != TUP_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad .tup (want magic TUP1 + {} bytes, got {} bytes)",
                    TUP_LEN * 4,
                    buf.len().saturating_sub(4)
                ),
            ));
        }
        let mut w = Box::new([[0i32; 81]; 36]);
        let mut o = 4usize;
        for a in w.iter_mut() {
            for v in a.iter_mut() {
                *v = i32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
                o += 4;
            }
        }
        Ok(TupleTable { w })
    }

    /// stm-relative tuple score: one lookup per anchor. ~36*(4 bit-tests +
    /// pack + lookup + add) per node; skip-empty comes later if gates demand.
    #[inline]
    pub fn forward(&self, b: &Board) -> i32 {
        let own = b.me();
        let enemy = b.opp();
        let mut s = 0i32;
        for a in 0..36 {
            s += self.w[a][block2_index(own, enemy, a)];
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
        // Black a1 (sq 42), white g7 (sq 6), black to move (own=black).
        // Only blocks covering a corner square are non-empty: anchor (5,0)
        // holds a1 as its 3rd cell (own digit 1 * 9 = 9); anchor (0,5)
        // holds g7 as its 2nd cell (ENEMY digit 2 * 3 = 6). The other 34
        // anchors index 0.
        let b = Board::start();
        for a in 0..36 {
            let idx = block2_index(b.occ[0], b.occ[1], a);
            let r = a / 6;
            let c = a % 6;
            let want = if r == 5 && c == 0 {
                9
            } else if r == 0 && c == 5 {
                6
            } else {
                0
            };
            assert_eq!(idx, want, "anchor {a} (r{r} c{c})");
        }
    }

    #[test]
    fn block2_covers_board() {
        // Every square belongs to at least one block; corners to exactly 1.
        let mut cover = [0u32; 49];
        for cells in super::super::board::BLOCK2 {
            for sq in cells {
                cover[sq as usize] += 1;
            }
        }
        for (sq, n) in cover.iter().enumerate() {
            assert!(*n >= 1, "square {sq} uncovered");
        }
        assert_eq!(cover[42], 1); // a1 corner
        assert_eq!(cover[6], 1); // g7 corner
    }
}
