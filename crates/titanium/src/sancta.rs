//! Sanctaphraxx eval steal (Ciekce, GPL-3.0 research comparison).
//!
//! 147 inputs (own49/enemy49/gap49, per-perspective), L1=64 Clip(255),
//! single output over (crelu(ours) ++ crelu(theirs)), cp=sum*400/(255*64).
//! Weights: data/nnue/sancta_w.s1 (19KB, gitignored, stays local).
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
pub const S1_NUM: i32 = 400;
pub const S1_DIV: i32 = 255 * 64;

pub struct S1Net {
    pub ft_w: Box<[[i16; S1_L1]; S1_INPUT]>,
    pub ft_b: [i16; S1_L1],
    pub out_w: [i16; S1_L1 * 2],
    pub out_b: i16,
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

#[inline]
fn forward_simd(
    ours: &[i16; S1_L1],
    theirs: &[i16; S1_L1],
    out_w: &[i16; S1_L1 * 2],
    out_b: i16,
) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                let zero = _mm256_setzero_si256();
                let cap = _mm256_set1_epi16(S1_CLIP);
                let mut sum = _mm256_setzero_si256();
                for k in 0..4 {
                    let o = k * 16;
                    let av = _mm256_loadu_si256(ours.as_ptr().add(o) as *const __m256i);
                    let ac = _mm256_min_epi16(_mm256_max_epi16(av, zero), cap);
                    let w = _mm256_loadu_si256(out_w.as_ptr().add(o) as *const __m256i);
                    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(ac, w));
                    let tv = _mm256_loadu_si256(theirs.as_ptr().add(o) as *const __m256i);
                    let tc = _mm256_min_epi16(_mm256_max_epi16(tv, zero), cap);
                    let w2 =
                        _mm256_loadu_si256(out_w.as_ptr().add(S1_L1 + o) as *const __m256i);
                    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(tc, w2));
                }
                let lo = _mm256_castsi256_si128(sum);
                let hi = _mm256_extracti128_si256::<1>(sum);
                let s128 = _mm_add_epi32(lo, hi);
                let h64 = _mm_unpackhi_epi64(s128, s128);
                let s64 = _mm_add_epi32(s128, h64);
                let h32 = _mm_shuffle_epi32::<0x4E>(s64);
                let s32 = _mm_add_epi32(s64, h32);
                let total = _mm_cvtsi128_si32(s32);
                return (out_b as i32 + total) * S1_NUM / S1_DIV;
            }
        }
    }
    let mut sum: i32 = out_b as i32;
    for i in 0..S1_L1 {
        let a = ours[i].clamp(0, S1_CLIP) as i32;
        let t = theirs[i].clamp(0, S1_CLIP) as i32;
        sum += a * out_w[i] as i32 + t * out_w[S1_L1 + i] as i32;
    }
    sum * S1_NUM / S1_DIV
}

impl S1Net {
    pub fn load(path: &str) -> Option<S1Net> {
        let d = std::fs::read(path).ok()?;
        Self::from_bytes(&d)
    }

    pub fn from_bytes(d: &[u8]) -> Option<S1Net> {
        if d.len() != (147 * 64 + 64 + 128 + 1) * 2 {
            return None;
        }
        let n = d.len() / 2;
        let mut w = vec![0i16; n];
        for (i, c) in d.chunks_exact(2).enumerate() {
            w[i] = i16::from_le_bytes([c[0], c[1]]);
        }
        let mut ft_w = Box::new([[0i16; S1_L1]; S1_INPUT]);
        for f in 0..S1_INPUT {
            ft_w[f].copy_from_slice(&w[f * S1_L1..(f + 1) * S1_L1]);
        }
        let mut ft_b = [0i16; S1_L1];
        ft_b.copy_from_slice(&w[147 * 64..147 * 64 + 64]);
        let mut out_w = [0i16; S1_L1 * 2];
        out_w.copy_from_slice(&w[147 * 64 + 64..147 * 64 + 64 + 128]);
        let out_b = w[147 * 64 + 64 + 128];
        Some(S1Net { ft_w, ft_b, out_w, out_b })
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
        forward_simd(
            &acc[stm as usize],
            &acc[1 - stm as usize],
            &self.out_w,
            self.out_b,
        )
    }

    /// Full refresh + forward (ground truth; incremental must match).
    pub fn evaluate(&self, b: &Board, stm: u8, scratch: &mut S1Acc) -> i32 {
        self.refresh_root(b, scratch);
        self.forward_ready(scratch, stm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s1_loads_and_scores_start() {
        let bytes = include_bytes!("../../../data/nnue/sancta_w.s1");
        let net = S1Net::from_bytes(bytes).expect("s1 blob");
        let b = Board::start();
        let mut sc = [[0i16; S1_L1]; 2];
        let v_black = net.evaluate(&b, 0, &mut sc);
        let v_white = net.evaluate(&b, 1, &mut sc);
        assert_eq!(v_black, v_white, "start symmetric");
        assert!(v_black.abs() < 200, "start sane: {v_black}");
    }

    #[test]
    fn s1_incremental_matches_refresh() {
        let bytes = include_bytes!("../../../data/nnue/sancta_w.s1");
        let net = S1Net::from_bytes(bytes).expect("s1 blob");
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
        let bytes = include_bytes!("../../../data/nnue/sancta_w.s1");
        let net = S1Net::from_bytes(bytes).expect("s1 blob");
        let b0 = Board::start();
        let mut sc = [[0i16; S1_L1]; 2];
        let v0 = net.evaluate(&b0, 0, &mut sc);
        let b1 = b0.make(crate::board::Move { from: 42, to: 33 });
        let v1 = net.evaluate(&b1, 1, &mut sc);
        assert!(v1 != v0, "position-sensitive: {v0} vs {v1}");
    }
}
