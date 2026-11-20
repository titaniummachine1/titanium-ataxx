//! D1303 (user: "try to gate it"): the futility small net, ZX design. A lower bound on the big NNUE's eval of a search CHILD
//! (pinball loss at a low quantile, hub `ti train snarch` w2+zx, trained from the Zataxx net): the big net's child eval is below
//! it on only ~tau of the calls. The futility step prunes a move without the big net when it is pruned even at this bound.
//!
//! Shape = Zataxx's net: 36 overlapping 2x2 windows over the 7x7 board, each window one of 3^4 = 81 states (empty / side to
//! move / other), 2916 one-hot rows, ONE accumulator of W = 256 (the child's side to move's view), SCReLU, no hidden layer,
//! a per-bucket (32 empties buckets, each NNUE bucket split in two, D1314/D1317) linear output, plus per-bucket skips of the parent's static eval and the child's hand
//! eval. Blockers are ignored (as in the trained model). Layout "TSN2" (see `parse`); the weights are compiled in.
//!
//! Integer inference (QA 255): accumulator = bias + sum of 36 rows (i16), c = clamp(acc, 0, 255), s = (c*c) >> 1 (0..32512),
//! out = sum s * zo_q (zo_q = output weight * 256) / (255 * 255 * 128). The parent's node keeps ONE accumulator, the view
//! of the CHILDREN's side to move (the parent's opponent as "us"); a child's accumulator is that one with the windows the
//! move changed swapped (2 row ops per changed window) -- exactly the incremental form the speed bench measured.

pub const NB: usize = 32;
pub const W: usize = 256;
const NWIN: usize = 36;
const WSTATES: usize = 81;
const NROWS: usize = NWIN * WSTATES;
/// integer scale of the output term: 255 * 255 / 2 (the c*c >> 1) * 256 (zo_q)
const OUT_DIV: f32 = 255.0 * 255.0 * 128.0;

static BLOB: &[u8] = include_bytes!("../../../data/nnue/fut_small.tsn");

/// The compiled-in blob (identity checks in config.rs).
pub fn blob() -> &'static [u8] {
    BLOB
}

/// The accumulator of the children's side to move (one per search node, built lazily).
pub type Acc = [i16; W];

pub struct SmallNet {
    /// [row][W] feature rows (weight * 255, i16)
    ft: Vec<i16>,
    fb: [i16; W],
    /// per bucket output weights * 256
    zo: Vec<[i16; W]>,
    ob: [f32; NB],
    skip: [[f32; NB]; 2],
    /// 4-bit side-to-move pattern << 4 | 4-bit other pattern -> window state code (0..81)
    code: [u16; 256],
    /// the windows (bit w) that contain each square
    sq_win: [u64; 49],
}

#[inline(always)]
fn win_off(w: usize) -> usize {
    7 * (w / 6) + (w % 6)
}

/// The 4 squares of a window (a, a+1, a+7, a+8) of a bitboard as a 4-bit pattern.
#[inline(always)]
fn pat4(bb: u64, off: usize) -> usize {
    let x = bb >> off;
    ((x & 3) | ((x >> 5) & 0b1100)) as usize
}

impl SmallNet {
    /// The compiled-in net.
    pub fn builtin() -> SmallNet {
        SmallNet::parse(BLOB).expect("compiled-in fut_small.tsn is malformed")
    }

    /// "TSN2": magic, NB u32, W u32, ft i16[2916*W], fb i16[W], zo i16[NB*W], ob f32[NB], skip f32[2*NB] (little endian).
    pub fn parse(d: &[u8]) -> Result<SmallNet, String> {
        if d.len() < 12 || &d[..4] != b"TSN2" {
            return Err("not a TSN2 file".into());
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        if (u(0), u(1)) != (NB, W) {
            return Err(format!("small net is {}x{}, this engine is built for {NB}x{W}", u(0), u(1)));
        }
        let want = 12 + 2 * (NROWS * W + W + NB * W) + 4 * (NB + 2 * NB);
        if d.len() != want {
            return Err(format!("small net has {} bytes, expected {want}", d.len()));
        }
        let mut p = 12;
        let mut i16s = |n: usize| -> Vec<i16> {
            let v = d[p..p + 2 * n].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
            p += 2 * n;
            v
        };
        let ft = i16s(NROWS * W);
        let fbv = i16s(W);
        let zov = i16s(NB * W);
        let f32s = |from: usize, n: usize| -> Vec<f32> { d[from..from + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
        let fstart = 12 + 2 * (NROWS * W + W + NB * W);
        let obv = f32s(fstart, NB);
        let skv = f32s(fstart + 4 * NB, 2 * NB);
        let mut fb = [0i16; W];
        fb.copy_from_slice(&fbv);
        let mut zo = vec![[0i16; W]; NB];
        for bk in 0..NB {
            zo[bk].copy_from_slice(&zov[bk * W..(bk + 1) * W]);
        }
        let mut ob = [0f32; NB];
        ob.copy_from_slice(&obv);
        let mut skip = [[0f32; NB]; 2];
        skip[0].copy_from_slice(&skv[..NB]);
        skip[1].copy_from_slice(&skv[NB..]);
        let mut code = [0u16; 256];
        for (idx, c) in code.iter_mut().enumerate() {
            let (us, them) = (idx >> 4, idx & 15);
            let mut v = 0;
            let mut pw = 1;
            for k in 0..4 {
                v += ((us >> k & 1) + 2 * (them >> k & 1)) * pw;
                pw *= 3;
            }
            *c = v as u16;
        }
        let mut sq_win = [0u64; 49];
        for (sq, m) in sq_win.iter_mut().enumerate() {
            let (r, c) = (sq / 7, sq % 7);
            for i in r.saturating_sub(1)..=r.min(5) {
                for j in c.saturating_sub(1)..=c.min(5) {
                    if i <= 5 && j <= 5 && (i == r || i + 1 == r) && (j == c || j + 1 == c) {
                        *m |= 1u64 << (6 * i + j);
                    }
                }
            }
        }
        Ok(SmallNet { ft, fb, zo, ob, skip, code, sq_win })
    }

    #[inline(always)]
    fn row(&self, w: usize, us: u64, them: u64) -> &[i16] {
        let off = win_off(w);
        let idx = w * WSTATES + self.code[pat4(us, off) << 4 | pat4(them, off)] as usize;
        &self.ft[idx * W..(idx + 1) * W]
    }

    /// The accumulator of the parent's CHILDREN: the view of the parent's opponent (= the child's side to move), built
    /// once per node from the parent's stones (`p_own` = the parent's side to move, `p_opp` the other side).
    pub fn parent_acc(&self, p_own: u64, p_opp: u64) -> Acc {
        let mut a = self.fb;
        for w in 0..NWIN {
            let r = self.row(w, p_opp, p_own);
            for j in 0..W {
                a[j] = a[j].wrapping_add(r[j]);
            }
        }
        a
    }

    /// Bound on the big net's eval of a child, in cp from the child's side to move. `parent` = `parent_acc` of the parent,
    /// `p_own`/`p_opp` the parent's stones, `c_own`/`c_opp` the child's side to move's / the other side's stones after the
    /// move, `empties` of the child, `parent_cp` the parent's static eval (parent's view), `hand_cp` the child's hand eval.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn child_bound(&self, parent: &Acc, p_own: u64, p_opp: u64, c_own: u64, c_opp: u64, empties: u32, parent_cp: i32, hand_cp: i32) -> i32 {
        let a = self.child_acc(parent, p_own, p_opp, c_own, c_opp);
        self.bound_acc(&a, empties, parent_cp, hand_cp)
    }

    /// The child's accumulator: the parent's (children's view) with the windows the move touched swapped.
    #[inline]
    pub fn child_acc(&self, parent: &Acc, p_own: u64, p_opp: u64, c_own: u64, c_opp: u64) -> Acc {
        // before the move the child's side to move (= the parent's opponent) owns p_opp, the other side p_own
        let chg = (p_opp ^ c_own) | (p_own ^ c_opp);
        let mut wm = 0u64;
        let mut bb = chg;
        while bb != 0 {
            wm |= self.sq_win[bb.trailing_zeros() as usize];
            bb &= bb - 1;
        }
        let mut a = *parent;
        while wm != 0 {
            let w = wm.trailing_zeros() as usize;
            wm &= wm - 1;
            let (old, new) = (self.row(w, p_opp, p_own), self.row(w, c_own, c_opp));
            for j in 0..W {
                a[j] = a[j].wrapping_sub(old[j]).wrapping_add(new[j]);
            }
        }
        a
    }

    /// Full accumulator of a position from its own side to move's view (`own` = the side to move's stones).
    pub fn acc(&self, own: u64, opp: u64) -> Acc {
        self.parent_acc(opp, own)
    }

    /// Head on a ready accumulator.
    #[inline(always)]
    pub fn bound_acc(&self, a: &Acc, empties: u32, parent_cp: i32, hand_cp: i32) -> i32 {
        let bk = (empties as usize * NB / 50).min(NB - 1);
        let sum = self.out_sum(a, bk);
        let y = self.ob[bk] + sum as f32 / OUT_DIV + self.skip[0][bk] * (parent_cp as f32 / 400.0) + self.skip[1][bk] * (hand_cp as f32 / 400.0);
        (400.0 * y).round().clamp(-30000.0, 30000.0) as i32
    }

    /// sum_j ((clamp(a_j, 0, 255)^2) >> 1) * zo_q[bk][j] in i32 (the reference implementation, always compiled).
    pub fn out_sum_scalar(&self, a: &Acc, bk: usize) -> i32 {
        let zo = &self.zo[bk];
        let mut s = 0i32;
        for j in 0..W {
            let c = a[j].clamp(0, 255) as i32;
            s += ((c * c) >> 1) * zo[j] as i32;
        }
        s
    }

    #[inline(always)]
    fn out_sum(&self, a: &Acc, bk: usize) -> i32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use std::arch::x86_64::*;
            // SAFETY: compiled only with AVX2; all loads are inside the fixed-size arrays (W = 256 = 16 vectors of 16 i16).
            unsafe {
                let (zero, qa) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
                let mut sum = _mm256_setzero_si256();
                let z = self.zo[bk].as_ptr();
                for v in 0..W / 16 {
                    let x = _mm256_loadu_si256(a.as_ptr().add(v * 16) as *const __m256i);
                    let c = _mm256_min_epi16(_mm256_max_epi16(x, zero), qa);
                    let s = _mm256_srli_epi16(_mm256_mullo_epi16(c, c), 1);
                    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(s, _mm256_loadu_si256(z.add(v * 16) as *const __m256i)));
                }
                let lo = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
                let lo = _mm_hadd_epi32(lo, lo);
                let lo = _mm_hadd_epi32(lo, lo);
                _mm_cvtsi128_si32(lo)
            }
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            self.out_sum_scalar(a, bk)
        }
    }

    /// Bound on the big net's eval of a position-as-child (full refresh): `own` / `opp` = the child's side to move's / the
    /// other side's stones.
    pub fn bound(&self, own: u64, opp: u64, _blk: u64, parent_cp: i32, hand_cp: i32) -> i32 {
        let a = self.acc(own, opp);
        self.bound_acc(&a, 49 - (own | opp | _blk).count_ones(), parent_cp, hand_cp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, MoveList};

    /// A deterministic random net (the cost and the arithmetic do not depend on trained values).
    fn random_net() -> SmallNet {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut b = Vec::new();
        b.extend_from_slice(b"TSN2");
        b.extend_from_slice(&(NB as u32).to_le_bytes());
        b.extend_from_slice(&(W as u32).to_le_bytes());
        for _ in 0..NROWS * W + W {
            b.extend_from_slice(&((r() % 41) as i16 - 20).to_le_bytes());
        }
        for _ in 0..NB * W {
            b.extend_from_slice(&((r() % 200) as i16 - 100).to_le_bytes());
        }
        for k in 0..NB {
            b.extend_from_slice(&(0.01f32 * k as f32).to_le_bytes());
        }
        for k in 0..2 * NB {
            b.extend_from_slice(&(0.1f32 * (k % 3) as f32).to_le_bytes());
        }
        SmallNet::parse(&b).unwrap()
    }

    /// The incremental child accumulator equals a full refresh of the child, on every legal move of random games; the AVX2
    /// head equals the scalar reference; the integer head matches an f64 evaluation of the same dequantised net.
    #[test]
    fn child_acc_matches_refresh() {
        let net = random_net();
        let mut b = Board::start();
        let mut seed = 12345u64;
        let mut checked = 0;
        for _ in 0..160 {
            let mut list = MoveList::new();
            b.legal_moves_dedup_into(&mut list);
            if list.is_empty() || b.game_over() {
                b = Board::start();
                continue;
            }
            let bt = b.turn as usize;
            let pa = net.parent_acc(b.occ[bt], b.occ[1 - bt]);
            for i in 0..list.len() {
                let c = b.make(list.move_at(i));
                let ct = c.turn as usize;
                let inc = net.child_acc(&pa, b.occ[bt], b.occ[1 - bt], c.occ[ct], c.occ[1 - ct]);
                let full = net.acc(c.occ[ct], c.occ[1 - ct]);
                assert_eq!(inc, full, "incremental child accumulator != full refresh");
                let e = c.empty().count_ones();
                let bk = (e as usize * NB / 50).min(NB - 1);
                assert_eq!(net.out_sum(&inc, bk), net.out_sum_scalar(&inc, bk));
                let direct = net.child_bound(&pa, b.occ[bt], b.occ[1 - bt], c.occ[ct], c.occ[1 - ct], e, 123, -45);
                assert_eq!(direct, net.bound_acc(&inc, e, 123, -45));
                // f64 reference of the same dequantised weights
                let mut y = net.ob[bk] as f64 + net.skip[0][bk] as f64 * 123.0 / 400.0 + net.skip[1][bk] as f64 * -45.0 / 400.0;
                for j in 0..W {
                    let cc = (inc[j].clamp(0, 255) as f64) / 255.0;
                    y += net.zo[bk][j] as f64 / 256.0 * cc * cc;
                }
                assert!((direct as f64 - 400.0 * y).abs() <= 1.0, "bound {direct} vs f64 {}", 400.0 * y);
                checked += 1;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            b = b.make(list.move_at((seed >> 33) as usize % list.len()));
        }
        assert!(checked > 1000);
    }

    #[test]
    fn window_codes_are_zataxx() {
        let net = random_net();
        // a1 (bit 0) = side to move's stone, b1 (bit 1) = other side's: window (0,0) pattern us 0b0001, them 0b0010 -> 1 + 2*3
        assert_eq!(net.code[(pat4(1, 0) << 4) | pat4(2, 0)], 1 + 2 * 3);
        // the square a2 (bit 7) is the third square of window (0,0): digit weight 9
        assert_eq!(net.code[(pat4(1 << 7, 0) << 4) | pat4(0, 0)], 9);
        // every square sits in 1..=4 windows, the corner in exactly one
        assert_eq!(net.sq_win[0].count_ones(), 1);
        assert_eq!(net.sq_win[24].count_ones(), 4);
    }
}
