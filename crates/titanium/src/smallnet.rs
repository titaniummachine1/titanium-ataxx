//! exp/fut-sn (D110): the futility small net. A tiny net distilled from the big NNUE's eval of search CHILDREN, trained
//! as a LOWER BOUND (pinball loss at a low quantile, training/smallnet_train.py): the big net's child eval is below it on
//! only ~tau of the calls. The futility step prunes a move without the big net when it is pruned even at this bound.
//! Inputs, side to move relative, two perspectives with shared weights: own / enemy / blocker on each of the 49 squares
//! (147 features) -> W -> CReLU -> concat (stm first) -> 8 empties buckets: L1 (2W -> H) CReLU -> 1, plus per-bucket
//! linear skips of the parent's static eval and the child's hand eval. Layout = training/smallnet_export.py ("TSN1").
//! The weights are compiled into the binary (one exe, nothing extra to ship); they are fitted to ONE big net
//! (data/nnue/nnue_main.tnn e11827d1) and must be retrained with it.

const NB: usize = 8;
/// fixed sizes of the shipped net (compile-time loops vectorise); the loader refuses anything else
const W: usize = 32;
const H: usize = 16;
const D2: usize = 2 * W;

static BLOB: &[u8] = include_bytes!("../../../data/nnue/fut_small.tsn");

pub struct SmallNet {
    /// fused rows [kind][sq][us W | them W]: own stone (us row sq, them row 49+sq), enemy stone (us 49+sq, them sq),
    /// blocker (98+sq both): one 2W add per piece
    rows: Vec<[f32; D2]>,
    bias: [f32; D2],
    /// L1 transposed per bucket: [bucket][input][H]
    l1_t: Vec<[f32; H]>,
    l1_b: [[f32; H]; NB],
    out_w: [[f32; H]; NB],
    out_b: [f32; NB],
    skip: [[f32; NB]; 2],
}

impl SmallNet {
    /// The compiled-in net.
    pub fn builtin() -> SmallNet {
        SmallNet::parse(BLOB).expect("compiled-in fut_small.tsn is malformed")
    }

    pub fn parse(d: &[u8]) -> Result<SmallNet, String> {
        if d.len() < 12 || &d[..4] != b"TSN1" {
            return Err("not a TSN1 file".into());
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        if (u(0), u(1)) != (W, H) {
            return Err(format!("small net is {}x{}, this engine is built for {W}x{H}", u(0), u(1)));
        }
        let mut p = 12;
        let mut take = |n: usize| -> Vec<f32> {
            let v = d[p..p + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            p += 4 * n;
            v
        };
        let (ft, ftb, l1_w, l1_b, out_w, out_b, skip) =
            (take(147 * W), take(W), take(NB * H * D2), take(NB * H), take(NB * H), take(NB), take(2 * NB));
        if p != d.len() {
            return Err("trailing or missing bytes".into());
        }
        let f = |i: usize| &ft[i * W..(i + 1) * W];
        let mut rows = vec![[0f32; D2]; 3 * 49];
        for sq in 0..49 {
            rows[sq][..W].copy_from_slice(f(sq));
            rows[sq][W..].copy_from_slice(f(49 + sq));
            rows[49 + sq][..W].copy_from_slice(f(49 + sq));
            rows[49 + sq][W..].copy_from_slice(f(sq));
            rows[98 + sq][..W].copy_from_slice(f(98 + sq));
            rows[98 + sq][W..].copy_from_slice(f(98 + sq));
        }
        let mut bias = [0f32; D2];
        bias[..W].copy_from_slice(&ftb);
        bias[W..].copy_from_slice(&ftb);
        let mut l1_t = vec![[0f32; H]; NB * D2];
        for bk in 0..NB {
            for i in 0..H {
                for j in 0..D2 {
                    l1_t[bk * D2 + j][i] = l1_w[(bk * H + i) * D2 + j];
                }
            }
        }
        let mut net = SmallNet { rows, bias, l1_t, l1_b: [[0.0; H]; NB], out_w: [[0.0; H]; NB], out_b: [0.0; NB], skip: [[0.0; NB]; 2] };
        for bk in 0..NB {
            net.l1_b[bk].copy_from_slice(&l1_b[bk * H..(bk + 1) * H]);
            net.out_w[bk].copy_from_slice(&out_w[bk * H..(bk + 1) * H]);
            net.out_b[bk] = out_b[bk];
            net.skip[0][bk] = skip[bk];
            net.skip[1][bk] = skip[NB + bk];
        }
        Ok(net)
    }

    #[inline(always)]
    fn add(&self, acc: &mut [f32; D2], mut bb: u64, off: usize) {
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let r = &self.rows[off + sq];
            for j in 0..D2 {
                acc[j] += r[j];
            }
        }
    }

    #[inline(always)]
    fn sub(&self, acc: &mut [f32; D2], mut bb: u64, off: usize) {
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let r = &self.rows[off + sq];
            for j in 0..D2 {
                acc[j] -= r[j];
            }
        }
    }

    /// Full accumulator [us | them] of a position (`own` = the side to move's stones).
    pub fn acc(&self, own: u64, opp: u64, blk: u64) -> [f32; D2] {
        // two independent sums (own + blockers, enemy) so the per-lane add chains are half as long
        let mut a = self.bias;
        let mut b = [0f32; D2];
        self.add(&mut a, own, 0);
        self.add(&mut b, opp, 49);
        self.add(&mut a, blk, 98);
        for j in 0..D2 {
            a[j] += b[j];
        }
        a
    }

    /// Child accumulator from the parent's: the child's side to move is the parent's opponent, so the parent's
    /// accumulator with its halves swapped is the child's BEFORE the move; then only the squares the move changed are
    /// updated (landing square, vacated jump origin, converted stones). Blockers never change. `p_own` / `p_opp` =
    /// parent side to move's stones / the other side's; `c_own` / `c_opp` = the same for the child.
    pub fn child_acc(&self, parent: &[f32; D2], p_own: u64, p_opp: u64, c_own: u64, c_opp: u64) -> [f32; D2] {
        let mut a = [0f32; D2];
        a[..W].copy_from_slice(&parent[W..]);
        a[W..].copy_from_slice(&parent[..W]);
        // before the move the child's own stones were the parent's opponent's, its opponent's the parent's own
        self.sub(&mut a, p_opp & !c_own, 0);
        self.add(&mut a, c_own & !p_opp, 0);
        self.sub(&mut a, p_own & !c_opp, 49);
        self.add(&mut a, c_opp & !p_own, 49);
        a
    }

    /// Head on a ready accumulator: `empties` of the position, `parent_cp` / `hand_cp` as in `bound`.
    pub fn bound_acc(&self, a: &[f32; D2], empties: u32, parent_cp: i32, hand_cp: i32) -> i32 {
        let bk = (empties as usize * NB / 50).min(NB - 1);
        let mut y = self.out_b[bk] + self.l1_out(a, bk);
        y += self.skip[0][bk] * (parent_cp as f32 / 400.0) + self.skip[1][bk] * (hand_cp as f32 / 400.0);
        (400.0 * y).round().clamp(-30000.0, 30000.0) as i32
    }

    /// out_w[bk] . clamp(L1(a)) for bucket `bk`: AVX2 + FMA when the build target has them (x86-64-v3, the project
    /// default), four independent accumulator pairs so the 64-step add chain does not serialise; plain loops otherwise.
    #[inline(always)]
    fn l1_out(&self, a: &[f32; D2], bk: usize) -> f32 {
        let t = &self.l1_t[bk * D2..(bk + 1) * D2];
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
        {
            use std::arch::x86_64::*;
            // SAFETY: compiled only for targets with AVX2 + FMA; every load/store is within fixed-size arrays
            // ([f32; H] rows of 16 = two 8-lane vectors, unaligned loads).
            unsafe {
                let z = _mm256_setzero_ps();
                let one = _mm256_set1_ps(1.0);
                let mut acc = [z; 8];
                for j in (0..D2).step_by(4) {
                    for k in 0..4 {
                        let x = _mm256_min_ps(_mm256_max_ps(_mm256_set1_ps(a[j + k]), z), one);
                        let c = t[j + k].as_ptr();
                        acc[2 * k] = _mm256_fmadd_ps(x, _mm256_loadu_ps(c), acc[2 * k]);
                        acc[2 * k + 1] = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.add(8)), acc[2 * k + 1]);
                    }
                }
                let b = self.l1_b[bk].as_ptr();
                let w = self.out_w[bk].as_ptr();
                let lo = _mm256_add_ps(_mm256_add_ps(acc[0], acc[2]), _mm256_add_ps(acc[4], acc[6]));
                let hi = _mm256_add_ps(_mm256_add_ps(acc[1], acc[3]), _mm256_add_ps(acc[5], acc[7]));
                let lo = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(lo, _mm256_loadu_ps(b)), z), one);
                let hi = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(hi, _mm256_loadu_ps(b.add(8))), z), one);
                let p = _mm256_fmadd_ps(hi, _mm256_loadu_ps(w.add(8)), _mm256_mul_ps(lo, _mm256_loadu_ps(w)));
                let q = _mm_add_ps(_mm256_castps256_ps128(p), _mm256_extractf128_ps::<1>(p));
                let q = _mm_add_ps(q, _mm_movehl_ps(q, q));
                let q = _mm_add_ss(q, _mm_shuffle_ps::<1>(q, q));
                _mm_cvtss_f32(q)
            }
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
        {
            let mut h = self.l1_b[bk];
            for j in 0..D2 {
                let x = a[j].clamp(0.0, 1.0);
                for i in 0..H {
                    h[i] += x * t[j][i];
                }
            }
            (0..H).map(|i| self.out_w[bk][i] * h[i].clamp(0.0, 1.0)).sum()
        }
    }

    /// Bound on the big net's eval of a child, in cp from the child's side to move: `own` / `opp` = the child's side to
    /// move's stones / the other side's, `parent_cp` = the parent's static eval (parent's view), `hand_cp` = the child's
    /// hand eval (child's view).
    pub fn bound(&self, own: u64, opp: u64, blk: u64, parent_cp: i32, hand_cp: i32) -> i32 {
        let a = self.acc(own, opp, blk);
        self.bound_acc(&a, 49 - (own | opp | blk).count_ones(), parent_cp, hand_cp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, MoveList};

    /// The incremental child accumulator equals a full refresh of the child, on every legal move of random games.
    #[test]
    fn child_acc_matches_refresh() {
        let net = SmallNet::builtin();
        let mut b = Board::start();
        let mut seed = 12345u64;
        let mut checked = 0;
        for _ in 0..120 {
            let mut list = MoveList::new();
            b.legal_moves_dedup_into(&mut list);
            if list.is_empty() || b.game_over() {
                b = Board::start();
                continue;
            }
            let (bt, pa) = (b.turn as usize, net.acc(b.occ[b.turn as usize], b.occ[1 - b.turn as usize], b.blockers));
            for i in 0..list.len() {
                let c = b.make(list.move_at(i));
                let ct = c.turn as usize;
                let inc = net.child_acc(&pa, b.occ[bt], b.occ[1 - bt], c.occ[ct], c.occ[1 - ct]);
                let full = net.acc(c.occ[ct], c.occ[1 - ct], c.blockers);
                for j in 0..D2 {
                    assert!((inc[j] - full[j]).abs() < 1e-4, "acc lane {j}: {} vs {}", inc[j], full[j]);
                }
                checked += 1;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            b = b.make(list.move_at((seed >> 33) as usize % list.len()));
        }
        assert!(checked > 1000);
    }
}
