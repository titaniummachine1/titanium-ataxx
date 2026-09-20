//! Sanctaphraxx eval steal (Ciekce, GPL-3.0 research comparison).
//!
//! v3 arch: 147 inputs (own49/enemy49/gap49, per-perspective), FT L1=64
//! Clip(255) INCREMENTAL (unchanged, still cheap), head 128 -> H2=16 ReLU ->
//! 1 (was single 128->1 linear). cp = out * 400/16320 (out_b in head units).
//! Weights: data/nnue/own_v3.s1 (26KB, gitignored, stays local).
//! Legacy 128->1 blobs (sancta_w.s1) load with zero H2 (exact fallback).
//!
//! SCOPE: load + full refresh + SIMD forward ONLY. No per-ply stack, no
//! do/unmake, no S4Undo, no PROF, no Box/Arc in hot path. The lazy
//! incremental path (parent acc + bitboard diffs) lives in search.rs next
//! to copy-make, gated by a single flag, zero-cost when OFF.

use crate::board::Board;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const S1_L1: usize = 64;
pub const S1_INPUT: usize = 147;
pub const S1_CLIP: i16 = 255;
/// Hidden width of the v3 head (128 -> H2 -> 1). Small on purpose: the
/// forward adds 128*16 MAC + 16 MAC (~2k ops vs 128 before); the FT
/// refresh/incremental path is byte-identical and still dominates.
pub const S1_H2: usize = 16;
pub const S1_NUM: i32 = 400;
pub const S1_DIV: i32 = 255 * 64;

/// v3 blob size: FT 147*64 + bias 64 + W1 128*16 + b1 16 + w2 16 + b2.
/// Legacy v1 (128->1) blobs are 19202B: rejected (no exact mapping).
pub const S1N_V3: usize = (147 * 64 + 64 + 128 * S1_H2 + S1_H2 + S1_H2 + 1) * 2;
pub const S1N_V1: usize = (147 * 64 + 64 + 128 + 1) * 2;

pub struct S1Net {
    pub ft_w: Box<[[i16; S1_L1]; S1_INPUT]>,
    pub ft_b: [i16; S1_L1],
    /// v3 head: W1 [128][16] row-major over (crelu(ours)++crelu(theirs)),
    /// b1[16], w2[16], b2 scalar. Legacy v1 blobs: W1/w2 zero, b2 = old
    /// out_b (exact same math as before).
    pub w1: Box<[[i16; S1_H2]; S1_L1 * 2]>,
    pub b1: [i32; S1_H2],
    pub w2: [i16; S1_H2],
    pub b2: i32,
}

pub type S1Acc = [[i16; S1_L1]; 2];

#[inline]
fn acc_add_col(acc: &mut [i16; S1_L1], col: &[i16; S1_L1]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                for k in 0..4 {
                    let o = k * 16;
                    let a = _mm256_loadu_si256(acc.as_ptr().add(o) as *const __m256i);
                    let w = _mm256_loadu_si256(col.as_ptr().add(o) as *const __m256i);
                    _mm256_storeu_si256(
                        acc.as_mut_ptr().add(o) as *mut __m256i,
                        _mm256_add_epi16(a, w),
                    );
                }
            }
            return;
        }
    }
    for i in 0..S1_L1 {
        acc[i] = acc[i].wrapping_add(col[i]);
    }
}

#[inline]
fn acc_sub_col(acc: &mut [i16; S1_L1], col: &[i16; S1_L1]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                for k in 0..4 {
                    let o = k * 16;
                    let a = _mm256_loadu_si256(acc.as_ptr().add(o) as *const __m256i);
                    let w = _mm256_loadu_si256(col.as_ptr().add(o) as *const __m256i);
                    _mm256_storeu_si256(
                        acc.as_mut_ptr().add(o) as *mut __m256i,
                        _mm256_sub_epi16(a, w),
                    );
                }
            }
            return;
        }
    }
    for i in 0..S1_L1 {
        acc[i] = acc[i].wrapping_sub(col[i]);
    }
}

/// v3 head forward: h = relu(b1 + W1^T crelu), out = b2 + w2.h.
/// Scalar: 128*16 + 16 MAC ≈ 2k ops — negligible next to the FT refresh
/// (up to ~40 stones × 64 lanes). FT add/sub stay AVX2 (acc_add_col/sub).
/// cp = out * 400/16320, same scale as v1.
#[inline]
fn forward_head(ours: &[i16; S1_L1], theirs: &[i16; S1_L1], net: &S1Net) -> i32 {
    let mut h = net.b1;
    for i in 0..S1_L1 {
        let a = ours[i].clamp(0, S1_CLIP) as i32;
        if a != 0 {
            let w = &net.w1[i];
            for j in 0..S1_H2 {
                h[j] += a * w[j] as i32;
            }
        }
        let t = theirs[i].clamp(0, S1_CLIP) as i32;
        if t != 0 {
            let w = &net.w1[S1_L1 + i];
            for j in 0..S1_H2 {
                h[j] += t * w[j] as i32;
            }
        }
    }
    let mut out = net.b2;
    for j in 0..S1_H2 {
        let a = h[j].max(0);
        out += a * net.w2[j] as i32;
    }
    out * S1_NUM / S1_DIV
}

impl S1Net {
    pub fn load(path: &str) -> Option<S1Net> {
        let d = std::fs::read(path).ok()?;
        Self::from_bytes(&d)
    }

    /// v3 loader (26KB). v1 19202B blobs: math differs (v1 sums pre-relu
    /// over 128, v3 relus over 16) — no exact mapping. v1 rejected; the v3
    /// trainer distills v1 -> v3 offline (train_own.py --init sancta_w.s1).
    pub fn from_bytes(d: &[u8]) -> Option<S1Net> {
        if d.len() != S1N_V3 {
            return None;
        }
        let n = d.len() / 2;
        let mut w = vec![0i16; n];
        for (i, c) in d.chunks_exact(2).enumerate() {
            w[i] = i16::from_le_bytes([c[0], c[1]]);
        }
        let mut o = 0;
        let mut ft_w = Box::new([[0i16; S1_L1]; S1_INPUT]);
        for f in 0..S1_INPUT {
            ft_w[f].copy_from_slice(&w[o..o + S1_L1]);
            o += S1_L1;
        }
        let mut ft_b = [0i16; S1_L1];
        ft_b.copy_from_slice(&w[o..o + S1_L1]);
        o += S1_L1;
        let mut w1 = Box::new([[0i16; S1_H2]; S1_L1 * 2]);
        for i in 0..S1_L1 * 2 {
            w1[i].copy_from_slice(&w[o..o + S1_H2]);
            o += S1_H2;
        }
        // b1 stored i16 quantized, widen to i32 head units.
        let mut b1 = [0i32; S1_H2];
        for j in 0..S1_H2 {
            b1[j] = w[o + j] as i32 * 256;
        }
        o += S1_H2;
        let mut w2 = [0i16; S1_H2];
        w2.copy_from_slice(&w[o..o + S1_H2]);
        o += S1_H2;
        let b2 = w[o] as i32 * 256;
        Some(S1Net { ft_w, ft_b, w1, b1, w2, b2 })
    }

    /// Feature index: sancta piece_indices(color,sq) = color*49 + sq.
    /// Our sq: rank*7+file (rank 0 = top). Sancta idx flips rows.
    #[inline]
    fn sq_idx(sq: u8) -> usize {
        let row = sq as usize / 7;
        let file = sq as usize % 7;
        (6 - row) * 7 + file
    }

    fn refresh_side(&self, b: &Board, p: usize, acc: &mut [i16; S1_L1]) {
        acc.copy_from_slice(&self.ft_b);
        let mut bb = b.occ[p];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            acc_add_col(acc, &self.ft_w[p * 49 + Self::sq_idx(sq as u8)]);
        }
        bb = b.occ[1 - p];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            acc_add_col(acc, &self.ft_w[(1 - p) * 49 + Self::sq_idx(sq as u8)]);
        }
        bb = b.blockers;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            acc_add_col(acc, &self.ft_w[98 + Self::sq_idx(sq as u8)]);
        }
    }

    /// Root refresh: both perspectives from scratch.
    pub fn refresh_root(&self, b: &Board, acc: &mut S1Acc) {
        self.refresh_side(b, 0, &mut acc[0]);
        self.refresh_side(b, 1, &mut acc[1]);
    }

    /// Lazy incremental child update: copy parent + bitboard diffs.
    /// Call sites have parent+child boards already (copy-make tree).
    pub fn update_child(
        &self,
        parent: &S1Acc,
        child: &mut S1Acc,
        b_parent: &Board,
        b_child: &Board,
    ) {
        *child = *parent;
        let diffs = [
            (b_child.occ[0] & !b_parent.occ[0], 0usize, true),
            (b_parent.occ[0] & !b_child.occ[0], 0usize, false),
            (b_child.occ[1] & !b_parent.occ[1], 1usize, true),
            (b_parent.occ[1] & !b_child.occ[1], 1usize, false),
        ];
        for p in 0..2 {
            let acc = &mut child[p];
            for &(bb, color, is_add) in &diffs {
                let mut x = bb;
                while x != 0 {
                    let sq = x.trailing_zeros() as usize;
                    x &= x - 1;
                    let f = if color == p {
                        p * 49 + Self::sq_idx(sq as u8)
                    } else {
                        (1 - p) * 49 + Self::sq_idx(sq as u8)
                    };
                    if is_add {
                        acc_add_col(acc, &self.ft_w[f]);
                    } else {
                        acc_sub_col(acc, &self.ft_w[f]);
                    }
                }
            }
        }
    }

    /// Forward from a ready accumulator, stm-relative.
    pub fn forward_ready(&self, acc: &S1Acc, stm: u8) -> i32 {
        forward_head(
            &acc[stm as usize],
            &acc[1 - stm as usize],
            self,
        )
    }

    /// Full refresh + forward (ground truth; incremental must match).
    pub fn evaluate(&self, b: &Board, stm: u8, scratch: &mut S1Acc) -> i32 {
        self.refresh_root(b, scratch);
        self.forward_ready(scratch, stm)
    }

    /// Test helper: build a net with explicit weights (no blob needed).
    #[cfg(test)]
    fn from_parts(
        ft_w: Box<[[i16; S1_L1]; S1_INPUT]>,
        ft_b: [i16; S1_L1],
        w1: Box<[[i16; S1_H2]; S1_L1 * 2]>,
        b1: [i32; S1_H2],
        w2: [i16; S1_H2],
        b2: i32,
    ) -> S1Net {
        S1Net { ft_w, ft_b, w1, b1, w2, b2 }
    }

    /// Test helper: v1-equivalent net (linear head through h[0]).
    /// v1 math: out = out_b + sum crelu*out_w. v3: h0 = out_b + sum
    /// (b1[0]=out_b, W1 col0 = out_w), out = 0 + h0*w2[0] with w2[0]=1 —
    /// EXCEPT relu(h0) clips negatives. Tests use positive activations
    /// (bias-dominated), so relu is identity and the mapping is exact.
    #[cfg(test)]
    fn from_v1_parts(
        ft_w: Box<[[i16; S1_L1]; S1_INPUT]>,
        ft_b: [i16; S1_L1],
        out_w: [i16; S1_L1 * 2],
        out_b: i16,
    ) -> S1Net {
        let mut w1 = Box::new([[0i16; S1_H2]; S1_L1 * 2]);
        for i in 0..S1_L1 * 2 {
            w1[i][0] = out_w[i];
        }
        let mut b1 = [0i32; S1_H2];
        b1[0] = out_b as i32;
        let mut w2 = [0i16; S1_H2];
        w2[0] = 1;
        S1Net { ft_w, ft_b, w1, b1, w2, b2: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_net() -> S1Net {
        // Asymmetric FT (feature-dependent) so positions score differently;
        // bias-dominated so relu(h) is identity.
        let mut ft_w = Box::new([[0i16; S1_L1]; S1_INPUT]);
        for f in 0..S1_INPUT {
            for i in 0..S1_L1 {
                ft_w[f][i] = ((f * 31 + i * 7) % 11) as i16 - 5;
            }
        }
        let ft_b = [10i16; S1_L1];
        let mut w1 = Box::new([[0i16; S1_H2]; S1_L1 * 2]);
        for i in 0..S1_L1 * 2 {
            for j in 0..S1_H2 {
                w1[i][j] = ((i + j) % 5) as i16 - 2;
            }
        }
        let b1 = [300i32; S1_H2];
        let mut w2 = [0i16; S1_H2];
        for j in 0..S1_H2 {
            w2[j] = (j as i16 % 3) - 1;
        }
        S1Net::from_parts(ft_w, ft_b, w1, b1, w2, 50)
    }

    #[test]
    fn v3_head_matches_reference() {
        // Independent float reference of h=relu(b1+W1^T crelu), out=b2+w2.h.
        let net = tiny_net();
        let b = Board::start();
        let mut acc = [[0i16; S1_L1]; 2];
        net.refresh_root(&b, &mut acc);
        let got = net.forward_ready(&acc, 0);
        let mut h = [0f64; S1_H2];
        for j in 0..S1_H2 {
            h[j] = net.b1[j] as f64;
        }
        for i in 0..S1_L1 {
            let a = acc[0][i].clamp(0, S1_CLIP) as f64;
            let t = acc[1][i].clamp(0, S1_CLIP) as f64;
            for j in 0..S1_H2 {
                h[j] += a * net.w1[i][j] as f64 + t * net.w1[S1_L1 + i][j] as f64;
            }
        }
        let mut out = net.b2 as f64;
        for j in 0..S1_H2 {
            out += h[j].max(0.0) * net.w2[j] as f64;
        }
        let want = (out * S1_NUM as f64 / S1_DIV as f64) as i32;
        assert_eq!(got, want, "v3 head must match float reference");
    }

    #[test]
    fn s1_incremental_matches_refresh() {
        let net = tiny_net();
        let b0 = Board::start();
        let mut root = [[0i16; S1_L1]; 2];
        net.refresh_root(&b0, &mut root);
        let b1 = b0.make(crate::board::Move { from: 42, to: 33 });
        let mut inc = [[0i16; S1_L1]; 2];
        net.update_child(&root, &mut inc, &b0, &b1);
        let mut full = [[0i16; S1_L1]; 2];
        net.refresh_root(&b1, &mut full);
        assert_eq!(inc, full, "incremental == refresh");
        assert_eq!(
            net.forward_ready(&inc, b1.turn),
            net.forward_ready(&full, b1.turn)
        );
    }

    #[test]
    fn s1_position_sense() {
        let net = tiny_net();
        let b0 = Board::start();
        let mut sc = [[0i16; S1_L1]; 2];
        let v0 = net.evaluate(&b0, 0, &mut sc);
        let b1 = b0.make(crate::board::Move { from: 42, to: 33 });
        let v1 = net.evaluate(&b1, 1, &mut sc);
        assert!(v1 != v0, "position-sensitive: {v0} vs {v1}");
    }

    #[test]
    fn v1_blob_rejected() {
        // v1 math (pre-relu linear sum) has no exact v3 mapping: reject.
        let fake_v1 = vec![0u8; S1N_V1];
        assert!(S1Net::from_bytes(&fake_v1).is_none());
        let fake_v3 = vec![0u8; S1N_V3];
        assert!(S1Net::from_bytes(&fake_v3).is_some());
    }
}
