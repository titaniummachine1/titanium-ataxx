//! NNUE inference skeleton (E6 roadmap: Net0/Net1).
//!
//! Layout (sanctaphraxx-proven, Reckless-style head):
//! - Sparse incremental FT: 147 binary inputs (49 own + 49 enemy + 49 gap),
//!   DUAL perspective accumulators (one per color, stm-relative `own/enemy`
//!   encoding shared weights). Ataxx has no board orientation (no king, no
//!   promotion rank), so "perspective" is a pure color swap — no geometry.
//! - Dense recomputed block: 8 weakness scalars (stm-relative), concatenated
//!   AFTER the FT like the Quoridor friend net. Weakness is non-local (one
//!   flip changes `conv` on up to 8 neighboring landings), so it must NOT be
//!   incremental — recompute per eval (~100ns, noise vs the forward pass).
//! - Head: (128 x 2 + 8) -> 32 -> 32 -> 1. Scalar first (wasm-safe); AVX2
//!   paths come with the first trained net.
//!
//! Why dual, not single canonical: relative own/enemy encoding shares every
//! pattern across colors (one "own stone on e4" weight, not black-e4 AND
//! white-e4) — ~2x data efficiency, decisive at our ~10M-position scale vs
//! Stockfish billions. Single stm-relative would be worse: every ply flips
//! stm, inverting all 98 features = full refresh per node. Dual costs 2x128
//! i16 adds per move (noise) and picks ours/theirs at eval, with tempo
//! encoded for free. Gaps never change mid-game, so gap columns are
//! refresh-only.

use crate::board::{Board, SQUARES};

/// Sparse input dims: 49 own + 49 enemy + 49 gap.
pub const NNUE_INPUT: usize = 147;
/// Feature-transformer width per perspective.
pub const NNUE_L1: usize = 128;
/// Dense weakness scalars (stm-relative, see `weakness_features` x2).
pub const NNUE_DENSE: usize = 8;
/// Hidden widths of the trained head.
pub const NNUE_H2: usize = 32;
pub const NNUE_H3: usize = 32;
/// ClippedReLU bound (matches sanctaphraxx L1_Q).
pub const NNUE_CLIP: i16 = 255;

/// Feature base offsets inside the shared FT.
const FEAT_OWN: usize = 0;
const FEAT_ENEMY: usize = 49;
const FEAT_GAP: usize = 98;

/// One perspective accumulator.
#[derive(Clone)]
pub struct Accumulator {
    pub vals: [i16; NNUE_L1],
}

impl Accumulator {
    fn zero() -> Accumulator {
        Accumulator { vals: [0; NNUE_L1] }
    }
}

/// Dual-perspective state. `acc[0]` = black-relative, `acc[1]` =
/// white-relative. Eval selects ours/theirs by side to move.
#[derive(Clone)]
pub struct NnueState {
    pub acc: [Accumulator; 2],
}

impl NnueState {
    pub fn new() -> NnueState {
        NnueState {
            acc: [Accumulator::zero(), Accumulator::zero()],
        }
    }

    /// Full refresh from a board (ground truth; incremental updates must
    /// match this bit-exactly, enforced by tests).
    pub fn refresh(&mut self, b: &Board, net: &Network) {
        for p in 0..2 {
            let mut a = net.ft_bias;
            let own = b.occ[p];
            let enemy = b.occ[1 - p];
            let mut bb = own;
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                add_col(&mut a, &net.ft_weights, FEAT_OWN + sq);
            }
            bb = enemy;
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                add_col(&mut a, &net.ft_weights, FEAT_ENEMY + sq);
            }
            bb = b.blockers;
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                add_col(&mut a, &net.ft_weights, FEAT_GAP + sq);
            }
            self.acc[p].vals = a;
        }
    }

    /// Incremental update from a parent state given bitboard diffs.
    /// `add_own/del_own` etc. are square bitboards for perspective `p`'s
    /// own/enemy stones; gaps are refresh-only (never change mid-game).
    #[allow(clippy::too_many_arguments)]
    pub fn update_from(
        &mut self,
        parent: &NnueState,
        net: &Network,
        p: usize,
        add_own: u64,
        del_own: u64,
        add_enemy: u64,
        del_enemy: u64,
    ) {
        let mut a = parent.acc[p].vals;
        let mut bb = add_own;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            add_col(&mut a, &net.ft_weights, FEAT_OWN + sq);
        }
        bb = del_own;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            sub_col(&mut a, &net.ft_weights, FEAT_OWN + sq);
        }
        bb = add_enemy;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            add_col(&mut a, &net.ft_weights, FEAT_ENEMY + sq);
        }
        bb = del_enemy;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            sub_col(&mut a, &net.ft_weights, FEAT_ENEMY + sq);
        }
        self.acc[p].vals = a;
    }

    /// Forward pass from the side-to-move's view. `dense` = 8 weakness
    /// scalars, stm-relative (see datagen record layout).
    pub fn forward(&self, stm: u8, dense: &[i32; NNUE_DENSE], net: &Network) -> i32 {
        let ours = &self.acc[stm as usize].vals;
        let theirs = &self.acc[1 - stm as usize].vals;
        // Layer 1: (crelu(ours) ++ crelu(theirs)) -> 32.
        let mut h2 = net.b2;
        for o in 0..NNUE_L1 {
            let a = crelu(ours[o]) as i32;
            let t = crelu(theirs[o]) as i32;
            if a == 0 && t == 0 {
                continue;
            }
            let w = &net.w2[o];
            for j in 0..NNUE_H2 {
                h2[j] += a * w[j] as i32 + t * w[NNUE_H2 + j] as i32;
            }
        }
        // Dense weakness block concatenated at layer 2 (friend-net pattern).
        for (k, d) in dense.iter().enumerate() {
            let w = &net.wd[k];
            for j in 0..NNUE_H2 {
                h2[j] += *d * w[j] as i32;
            }
        }
        // Layer 2: relu(h2) -> 32. Layer 3: relu -> 1 (scalar, no SIMD yet).
        let mut h3 = net.b3;
        for j in 0..NNUE_H2 {
            let a = h2[j].clamp(0, NNUE_CLIP as i32);
            if a == 0 {
                continue;
            }
            let w = &net.w3[j];
            for k in 0..NNUE_H3 {
                h3[k] += a * w[k] as i32;
            }
        }
        let mut out = net.b4;
        for k in 0..NNUE_H3 {
            let a = h3[k].clamp(0, NNUE_CLIP as i32);
            out += a * net.w4[k] as i32;
        }
        out / net.scale
    }
}

#[inline]
fn crelu(x: i16) -> i16 {
    x.clamp(0, NNUE_CLIP)
}

#[inline]
fn add_col(acc: &mut [i16; NNUE_L1], ft: &[[i16; NNUE_L1]; NNUE_INPUT], feat: usize) {
    debug_assert!(feat < NNUE_INPUT);
    let col = &ft[feat];
    for i in 0..NNUE_L1 {
        acc[i] = acc[i].saturating_add(col[i]);
    }
}

#[inline]
fn sub_col(acc: &mut [i16; NNUE_L1], ft: &[[i16; NNUE_L1]; NNUE_INPUT], feat: usize) {
    debug_assert!(feat < NNUE_INPUT);
    let col = &ft[feat];
    for i in 0..NNUE_L1 {
        acc[i] = acc[i].saturating_sub(col[i]);
    }
}

/// Trainable weights. No embedded net yet — `zeros()` for skeleton tests;
/// the first trained net arrives via `include_bytes!` + loader (Net0).
pub struct Network {
    pub ft_weights: Box<[[i16; NNUE_L1]; NNUE_INPUT]>,
    pub ft_bias: [i16; NNUE_L1],
    /// w2[o] = [ours_x32 ++ theirs_x32] (layer 1 over concatenated 256).
    pub w2: Box<[[i16; NNUE_H2 * 2]; NNUE_L1]>,
    /// Dense -> layer 1 (weakness block).
    pub wd: Box<[[i16; NNUE_H2]; NNUE_DENSE]>,
    pub b2: [i32; NNUE_H2],
    pub w3: Box<[[i16; NNUE_H3]; NNUE_H2]>,
    pub b3: [i32; NNUE_H3],
    pub w4: [i16; NNUE_H3],
    pub b4: i32,
    /// Output divisor (quant scale; sanctaphraxx uses 255*64/400).
    pub scale: i32,
}

impl Network {
    pub fn zeros() -> Network {
        Network {
            ft_weights: Box::new([[0; NNUE_L1]; NNUE_INPUT]),
            ft_bias: [0; NNUE_L1],
            w2: Box::new([[0; NNUE_H2 * 2]; NNUE_L1]),
            wd: Box::new([[0; NNUE_H2]; NNUE_DENSE]),
            b2: [0; NNUE_H2],
            w3: Box::new([[0; NNUE_H3]; NNUE_H2]),
            b3: [0; NNUE_H3],
            w4: [0; NNUE_H3],
            b4: 0,
            scale: 255 * 64 / 4,
        }
    }
}

impl Default for NnueState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Move};

    fn patterned_net() -> Network {
        // Deterministic non-zero weights so refresh-vs-incremental and
        // forward tests measure real arithmetic, not zeros.
        let mut n = Network::zeros();
        for f in 0..NNUE_INPUT {
            for i in 0..NNUE_L1 {
                n.ft_weights[f][i] = ((f * 131 + i * 17 + 7) % 251) as i16 - 125;
            }
        }
        for i in 0..NNUE_L1 {
            n.ft_bias[i] = (i as i16 % 11) - 5;
        }
        for o in 0..NNUE_L1 {
            for j in 0..NNUE_H2 * 2 {
                n.w2[o][j] = ((o * 37 + j * 13 + 3) % 61) as i16 - 30;
            }
        }
        for k in 0..NNUE_DENSE {
            for j in 0..NNUE_H2 {
                n.wd[k][j] = ((k * 41 + j * 11 + 5) % 61) as i16 - 30;
            }
        }
        for j in 0..NNUE_H2 {
            n.b2[j] = (j as i32 % 7) - 3;
            for k in 0..NNUE_H3 {
                n.w3[j][k] = ((j * 29 + k * 19 + 1) % 61) as i16 - 30;
            }
        }
        for k in 0..NNUE_H3 {
            n.b3[k] = (k as i32 % 5) - 2;
            n.w4[k] = ((k * 23 + 9) % 61) as i16 - 30;
        }
        n.b4 = 17;
        n
    }

    /// Diff occ bitboards per color between two positions.
    fn diff(old: u64, new: u64) -> (u64, u64) {
        (new & !old, old & !new)
    }

    #[test]
    fn incremental_matches_refresh() {
        // Random playout: every position's incremental state (from parent
        // diffs, both perspectives) must equal a full refresh bit-exactly.
        let net = patterned_net();
        let mut rng: u64 = 0x243F6A8885A308D3;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut b = Board::start();
        let mut st = NnueState::new();
        st.refresh(&b, &net);
        for _ in 0..60 {
            if b.game_over() {
                break;
            }
            if !b.has_moves(b.turn) {
                b = b.make_pass();
                st.refresh(&b, &net); // pass flips stm: refresh (no diff path)
                continue;
            }
            let moves = b.legal_moves();
            let m: Move = moves.move_at((next() % moves.len() as u64) as usize);
            let nb = b.make(m);
            // Per-perspective diffs from occ changes.
            let mut inc = NnueState::new();
            for p in 0..2 {
                let (add_own, del_own) = diff(b.occ[p], nb.occ[p]);
                let (add_en, del_en) = diff(b.occ[1 - p], nb.occ[1 - p]);
                inc.update_from(&st, &net, p, add_own, del_own, add_en, del_en);
            }
            let mut full = NnueState::new();
            full.refresh(&nb, &net);
            for p in 0..2 {
                assert_eq!(
                    inc.acc[p].vals, full.acc[p].vals,
                    "perspective {p} diverged after move {m:?}"
                );
            }
            // Forward must agree too (cheap determinism check).
            let d = [0i32; NNUE_DENSE];
            assert_eq!(
                inc.forward(nb.turn, &d, &net),
                full.forward(nb.turn, &d, &net)
            );
            b = nb;
            st = full;
        }
    }

    #[test]
    fn zero_net_zero_output() {
        let net = Network::zeros();
        let b = Board::start();
        let mut st = NnueState::new();
        st.refresh(&b, &net);
        let d = [0i32; NNUE_DENSE];
        assert_eq!(st.forward(0, &d, &net), 0);
        assert_eq!(st.forward(1, &d, &net), 0);
    }

    #[test]
    fn feature_count_is_147() {
        // 49 own + 49 enemy + 49 gap; SQUARES must stay 49.
        assert_eq!(SQUARES, 49);
        assert_eq!(FEAT_OWN, 0);
        assert_eq!(FEAT_ENEMY, 49);
        assert_eq!(FEAT_GAP, 98);
        assert_eq!(NNUE_INPUT, 147);
    }
}
