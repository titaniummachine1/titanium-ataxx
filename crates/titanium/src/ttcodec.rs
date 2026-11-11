//! Compact transposition-table field codecs (D648/D650, user idea: a static lookup table of EVERY pseudo-legal move,
//! independent of the board, instead of 16 bits of from/to).
//!
//! Move code (10 bits): 0 = none, 1..=49 = a clone to square (code - 1) (any source: a clone only depends on its
//! destination), 50..=529 = the 480 ordered jump pairs (from, to) at Chebyshev distance 2 on the 7x7 board,
//! 0x3FF = pass. 529 codes do not fit 8 bits (256) but do fit 10.
//!
//! Score code (16 bits, i16): evals are clamped to +-30000 (nnue.rs), so every non-mate score is stored as is;
//! a mate score (|s| >= MATE_BOUND, MATE_BOUND = MATE - 2000) is stored as +-(30001 + distance to MATE), at most 32001.

use crate::board::Move;

const NSQ: usize = 49;
/// pass
pub const PASS_CODE: u16 = 0x3FF;
/// highest real move code (49 clone destinations + 480 jump pairs)
pub const MAX_MOVE_CODE: u16 = 529;

const fn cheb(a: usize, b: usize) -> usize {
    let (ax, ay, bx, by) = (a % 7, a / 7, b % 7, b / 7);
    let dx = if ax > bx { ax - bx } else { bx - ax };
    let dy = if ay > by { ay - by } else { by - ay };
    if dx > dy {
        dx
    } else {
        dy
    }
}

const fn build_codes() -> [u16; NSQ * NSQ] {
    let mut t = [0u16; NSQ * NSQ];
    let mut next = 50u16;
    let mut f = 0;
    while f < NSQ {
        let mut to = 0;
        while to < NSQ {
            let d = cheb(f, to);
            if d == 1 {
                t[f * NSQ + to] = 1 + to as u16;
            } else if d == 2 {
                t[f * NSQ + to] = next;
                next += 1;
            }
            to += 1;
        }
        f += 1;
    }
    t
}

/// code of the move (from, to), index from * 49 + to; 0 for a non-move
static CODE: [u16; NSQ * NSQ] = build_codes();

const fn build_inv_jumps() -> [(u8, u8); 480] {
    let mut inv = [(0u8, 0u8); 480];
    let mut next = 0usize;
    let mut f = 0;
    while f < NSQ {
        let mut to = 0;
        while to < NSQ {
            if cheb(f, to) == 2 {
                inv[next] = (f as u8, to as u8);
                next += 1;
            }
            to += 1;
        }
        f += 1;
    }
    inv
}

static INV_JUMPS: [(u8, u8); 480] = build_inv_jumps();

/// Decode a 10-bit TT move code back to a legal move on `b` without generating the full move list.
/// Returns `None` for empty (`0`), `PASS_CODE`, out-of-range codes, or pseudo-legal moves illegal on `b`.
#[inline]
pub fn decode_move(b: &crate::board::Board, code: u16) -> Option<Move> {
    if code == 0 || code > MAX_MOVE_CODE {
        return None;
    }
    let me = b.me();
    let empty = b.empty();
    if code <= 49 {
        let to = (code - 1) as u8;
        if (empty >> to) & 1 == 0 {
            return None;
        }
        let clones = me & crate::board::RING1[to as usize];
        if clones == 0 {
            return None;
        }
        Some(Move {
            from: clones.trailing_zeros() as u8,
            to,
        })
    } else {
        let (from, to) = INV_JUMPS[(code - 50) as usize];
        if ((me >> from) & (empty >> to) & 1) == 0 {
            return None;
        }
        Some(Move { from, to })
    }
}

/// Code of a move when both squares are known to be on-board (no pass/bounds branch).
#[inline(always)]
pub(crate) fn jump_code(from: u8, to: u8) -> u16 {
    CODE[from as usize * NSQ + to as usize]
}

/// Code of a move (clones collapse to their destination, jumps keep (from, to), pass = PASS_CODE).
#[inline(always)]
pub fn move_code(m: Move) -> u16 {
    if m.from == 255 {
        return PASS_CODE;
    }
    if m.from as usize >= NSQ || m.to as usize >= NSQ {
        return 0;
    }
    CODE[m.from as usize * NSQ + m.to as usize]
}

/// Code of a move packed as `(from << 8) | to` (0xFFFF = pass, 0 = none).
#[inline(always)]
pub fn code_of_u32(v: u32) -> u16 {
    if v == 0xFFFF {
        return PASS_CODE;
    }
    if v == 0 {
        return 0;
    }
    let (f, t) = ((v >> 8) as usize, (v & 255) as usize);
    if f >= NSQ || t >= NSQ {
        return 0;
    }
    CODE[f * NSQ + t]
}

const MATE: i32 = crate::search::MATE;
const MATE_BOUND: i32 = MATE - 2000;
const EVAL_MAX: i32 = 30000;

/// 16-bit score code (see the module doc). Saturates outside +-30000 (non-mate) and the mate window.
#[inline(always)]
pub fn enc_score(s: i32) -> i16 {
    if s >= MATE_BOUND {
        (EVAL_MAX + 1 + (MATE - s).clamp(0, 2000)) as i16
    } else if s <= -MATE_BOUND {
        -((EVAL_MAX + 1 + (MATE + s).clamp(0, 2000)) as i16)
    } else {
        s.clamp(-EVAL_MAX, EVAL_MAX) as i16
    }
}

#[inline(always)]
pub fn dec_score(c: i16) -> i32 {
    let c = c as i32;
    if c > EVAL_MAX {
        MATE - (c - EVAL_MAX - 1)
    } else if c < -EVAL_MAX {
        -MATE + (-c - EVAL_MAX - 1)
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pseudo_legal_move_has_a_unique_code_and_clones_collapse() {
        let mut seen = std::collections::HashMap::new();
        let (mut clones, mut jumps) = (0, 0);
        for f in 0..49u8 {
            for t in 0..49u8 {
                let d = cheb(f as usize, t as usize);
                let c = move_code(Move { from: f, to: t });
                match d {
                    1 => {
                        assert_eq!(c, 1 + t as u16, "clone code depends on the destination only");
                        clones += 1;
                    }
                    2 => {
                        assert!(c >= 50 && c <= MAX_MOVE_CODE, "jump code {c}");
                        assert!(seen.insert(c, (f, t)).is_none(), "duplicate jump code {c}");
                        jumps += 1;
                    }
                    _ => assert_eq!(c, 0),
                }
            }
        }
        assert_eq!(jumps, 480);
        assert!(clones > 0);
        assert_eq!(move_code(Move::PASS), PASS_CODE);
        assert_eq!(code_of_u32(0xFFFF), PASS_CODE);
        assert_eq!(code_of_u32(0), 0);
        assert!(PASS_CODE > MAX_MOVE_CODE && PASS_CODE < 1024);
        assert_eq!(code_of_u32(Move { from: 0, to: 2 }.to_u32()), move_code(Move { from: 0, to: 2 }));
    }

    #[test]
    fn decode_move_matches_legal_moves_dedup() {
        use crate::board::{Board, MoveList};
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut list = MoveList::new();
        for n in 0..400 {
            let mut b = Board::start();
            for _ in 0..(n % 20) {
                b.legal_moves_dedup_into(&mut list);
                if list.is_empty() || b.game_over() {
                    break;
                }
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                b = b.make(list.move_at((x as usize) % list.len()));
            }
            if b.game_over() {
                continue;
            }
            b.legal_moves_dedup_into(&mut list);
            let mut expected = [None; 530];
            for m in list.iter() {
                expected[move_code(m) as usize] = Some(m);
            }
            for code in 0..=MAX_MOVE_CODE {
                assert_eq!(decode_move(&b, code), expected[code as usize], "code {code}");
            }
            assert_eq!(decode_move(&b, PASS_CODE), None);
            assert_eq!(decode_move(&b, MAX_MOVE_CODE + 1), None);
        }
    }

    #[test]
    fn score_codes_round_trip() {
        for s in -30000..=30000 {
            assert_eq!(dec_score(enc_score(s)), s);
        }
        for d in 0..=1999 {
            let s = MATE - d;
            assert_eq!(dec_score(enc_score(s)), s, "winning mate distance {d}");
            assert_eq!(dec_score(enc_score(-s)), -s, "losing mate distance {d}");
        }
        // outside the window: saturates, never wraps
        assert_eq!(dec_score(enc_score(30_500)), 30_000);
        assert_eq!(dec_score(enc_score(-30_500)), -30_000);
        assert!(enc_score(MATE - 1999) as i32 <= 32_001 && enc_score(MATE) as i32 >= 30_001);
    }
}
