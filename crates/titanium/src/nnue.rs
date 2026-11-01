//! exp/nnue: float32 inference of the net trained by training/train_nnue.py (features = crate::nnfeat).
//! Correctness first: plain f32 (auto-vectorised under x86-64-v3); int16/int8 quantisation comes after the net is
//! proven by games. Layout must match training/export_nnue.py:
//!   magic "TNN1", then u32 LE: num_feats, width, l1, l2, buckets, groups mask; then f32 LE arrays in this order:
//!   ft[num_feats][width], ftb[width], psqt[num_feats][buckets], l1_w[buckets*l1][2*width], l1_b[buckets*l1],
//!   l2_w[buckets*l2][l1], l2_b[buckets*l2], out_w[buckets][l2], out_b[buckets].
//!   magic "TNN2" (v2, D65): the same header plus u32 pairwise flag; arrays as above except
//!   l1_w[buckets*l1][din] (din = width if pairwise else 2*width), out_w[buckets][l1 + l2] (the output reads both
//!   hidden layers), then s57w[buckets]: the S57 tuple eval (cp / 400, side to move) times s57w[bucket] is added.
//! Output: cp from the side to move's view = 400 * y (training: win prob = sigmoid(y)).

use crate::board::Board;
use crate::nnfeat::{self, MAX_ACTIVE};

/// The main branch's strongest big NNUE. It is compiled into every native and Botzone build.
pub const BUNDLED_NNUE: &[u8] = include_bytes!("../../../data/nnue/nnue_main.tnn");

/// exp/nnue-fast: the feature transformer runs in int16 with weights and bias scaled by QA (measured on the ep4 net:
/// |w| <= 1.05, |acc| <= 11.4 over 40k val perspectives -> max ~11.7k of i16 32767). Plain integer loops: the compiler
/// vectorises them with AVX2 under x86-64-v3 and with SSE2 on a baseline x86-64 build (no intrinsics, no CPU
/// detection, nothing to fail on older CPUs). Hidden layers stay f32; L1 skips the ~86% zero activations (sparse,
/// transposed weights). `forward_f32` keeps the exact training-parity path as the reference.
pub const QA: i32 = 1024;
const MAX_W: usize = 1024;
const MAX_H: usize = 64;

/// Immutable slice with a 64-byte aligned start (a plain Vec only guarantees the element alignment, so every 512 B
/// feature row and 64 B L1 column straddled one extra cache line). Deref gives the same indexing as a Vec.
struct Aligned<T: Copy> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T: Copy + Send> Send for Aligned<T> {}
unsafe impl<T: Copy + Sync> Sync for Aligned<T> {}
impl<T: Copy> Aligned<T> {
    fn empty() -> Self {
        Aligned { ptr: std::ptr::NonNull::dangling().as_ptr(), len: 0 }
    }
    fn from_vec(v: Vec<T>) -> Self {
        if v.is_empty() {
            return Self::empty();
        }
        let layout = std::alloc::Layout::from_size_align(v.len() * std::mem::size_of::<T>(), 64).unwrap();
        // SAFETY: non-zero size; the copy fills the whole allocation
        unsafe {
            let ptr = std::alloc::alloc(layout) as *mut T;
            assert!(!ptr.is_null(), "out of memory");
            std::ptr::copy_nonoverlapping(v.as_ptr(), ptr, v.len());
            Aligned { ptr, len: v.len() }
        }
    }
}
impl<T: Copy> std::ops::Deref for Aligned<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: ptr/len describe the live allocation (or a dangling empty slice)
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl<T: Copy> Drop for Aligned<T> {
    fn drop(&mut self) {
        if self.len != 0 {
            let layout = std::alloc::Layout::from_size_align(self.len * std::mem::size_of::<T>(), 64).unwrap();
            // SAFETY: allocated in from_vec with this layout
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
        }
    }
}

/// For each byte mask, the positions of its set bits in ascending order (rest 0): lets the activation scan store 8
/// candidate indices at once and advance by the popcount instead of looping over bits with a mispredicted exit.
#[cfg(target_feature = "avx2")]
const fn bit_pos_table() -> [[u8; 8]; 256] {
    let mut t = [[0u8; 8]; 256];
    let mut b = 0;
    while b < 256 {
        let mut n = 0;
        let mut j = 0;
        while j < 8 {
            if (b >> j) & 1 != 0 {
                t[b][n] = j as u8;
                n += 1;
            }
            j += 1;
        }
        b += 1;
    }
    t
}
#[cfg(target_feature = "avx2")]
static BIT_POS: [[u8; 8]; 256] = bit_pos_table();

pub struct Nnue {
    pub num_feats: usize,
    pub width: usize,
    pub l1: usize,
    pub l2: usize,
    pub buckets: usize,
    pub groups: u32,
    /// v2 (TNN2): pairwise product, output over both hidden layers, S57 skip term.
    pub v2: bool,
    pub pairwise: bool,
    /// v3 (TNN3): zero-cost auxiliary Disagreement Head (dis_w, dis_b) over h1 || h2.
    pub has_dis: bool,
    ft: Vec<f32>,
    ftb: Vec<f32>,
    psqt: Vec<f32>,
    l1_w: Vec<f32>,
    l1_b: Vec<f32>,
    l2_w: Vec<f32>,
    l2_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    s57w: Vec<f32>,
    dis_w: Vec<f32>,
    dis_b: Vec<f32>,
    ftq: Aligned<i16>,
    ftbq: Vec<i16>,
    /// l1 weights transposed per bucket: [bucket][din][l1] (one contiguous l1 column per input, for sparse L1)
    l1_wt: Aligned<f32>,
    /// l2 weights transposed per bucket: [bucket][l1 input][l2 output] (so 8 outputs can be summed side by side)
    l2_wt: Aligned<f32>,
    /// psqt transposed: [bucket][feature] (one bucket's column is contiguous: ~105 KB, stays in L2)
    psqt_t: Aligned<f32>,
}

impl Nnue {
    pub fn load(path: &str) -> std::io::Result<Nnue> {
        let d = std::fs::read(path)?;
        Self::parse(&d).map_err(|e| std::io::Error::new(e.kind(), format!("{path}: {e}")))
    }

    /// Load the strongest network tracked by this engine revision.
    pub fn bundled() -> Nnue {
        Self::parse(BUNDLED_NNUE).expect("the bundled main NNUE is malformed")
    }

    /// Shared production instance for one-shot engine APIs.
    pub fn bundled_static() -> &'static Nnue {
        static NET: std::sync::OnceLock<Nnue> = std::sync::OnceLock::new();
        NET.get_or_init(Self::bundled)
    }

    fn parse(d: &[u8]) -> std::io::Result<Nnue> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let has_dis = d.len() >= 4 && &d[..4] == b"TNN3";
        let v2 = d.len() >= 4 && &d[..4] == b"TNN2";
        if d.len() < 28 || (&d[..4] != b"TNN1" && !v2 && !has_dis) {
            return Err(bad("not a TNN1/TNN2 file"));
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        let (num_feats, width, l1, l2, buckets, groups) =
            (u(0), u(1), u(2), u(3), u(4), u(5) as u32);
        let pairwise = (v2 || has_dis) && u(6) != 0;
        if num_feats != nnfeat::NUM_FEATS {
            return Err(bad("feature count differs from this engine's nnfeat"));
        }
        if pairwise && width % 2 != 0 {
            return Err(bad("pairwise needs an even width"));
        }
        let din = if pairwise { width } else { 2 * width };
        let out_in = if v2 { l1 + l2 } else { l2 };
        let mut p = if v2 || has_dis { 32 } else { 28 };
        let mut take = |n: usize| -> Vec<f32> {
            let v = d[p..p + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            p += 4 * n;
            v
        };
        let net = Nnue {
            ft: take(num_feats * width),
            ftb: take(width),
            psqt: take(num_feats * buckets),
            l1_w: take(buckets * l1 * din),
            l1_b: take(buckets * l1),
            l2_w: take(buckets * l2 * l1),
            l2_b: take(buckets * l2),
            out_w: take(buckets * out_in),
            out_b: take(buckets),
            s57w: if v2 { take(buckets) } else { Vec::new() },
            dis_w: if has_dis { take(buckets * (l1 + l2)) } else { Vec::new() },
            dis_b: if has_dis { take(buckets) } else { Vec::new() },
            ftq: Aligned::empty(),
            ftbq: Vec::new(),
            l1_wt: Aligned::empty(),
            l2_wt: Aligned::empty(),
            psqt_t: Aligned::empty(),
            num_feats,
            width,
            l1,
            l2,
            buckets,
            groups,
            v2,
            pairwise,
            has_dis,
        };
        if p != d.len() {
            return Err(bad("trailing or missing bytes"));
        }
        let mut net = net;
        if net.width > MAX_W || net.l1 > MAX_H || net.l2 > MAX_H {
            return Err(bad("width / hidden sizes above the fast path's limits"));
        }
        let q = |x: &f32| -> Option<i16> {
            let v = (x * QA as f32).round();
            if v.abs() < 32767.0 {
                Some(v as i16)
            } else {
                None
            }
        };
        net.ftq = Aligned::from_vec(net
            .ft
            .iter()
            .map(q)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| bad("ft weight overflows int16 at QA"))?);
        net.ftbq = net
            .ftb
            .iter()
            .map(q)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| bad("ft bias overflows int16 at QA"))?;
        let din = if net.pairwise {
            net.width
        } else {
            2 * net.width
        };
        let mut t = vec![0f32; net.buckets * din * net.l1];
        for bk in 0..net.buckets {
            for j in 0..net.l1 {
                for i in 0..din {
                    t[(bk * din + i) * net.l1 + j] = net.l1_w[(bk * net.l1 + j) * din + i];
                }
            }
        }
        net.l1_wt = Aligned::from_vec(t);
        let mut t2 = vec![0f32; net.buckets * net.l1 * net.l2];
        for bk in 0..net.buckets {
            for j in 0..net.l2 {
                for k in 0..net.l1 {
                    t2[(bk * net.l1 + k) * net.l2 + j] = net.l2_w[(bk * net.l2 + j) * net.l1 + k];
                }
            }
        }
        net.l2_wt = Aligned::from_vec(t2);
        let mut pt = vec![0f32; net.buckets * net.num_feats];
        for f in 0..net.num_feats {
            for bk in 0..net.buckets {
                pt[bk * net.num_feats + f] = net.psqt[f * net.buckets + bk];
            }
        }
        net.psqt_t = Aligned::from_vec(pt);
        Ok(net)
    }

    /// Fast path (int16 feature transformer, allocation-free, sparse L1); differs from `forward_f32` only by the
    /// QA rounding of the first layer (measured by `nneval`).
    /// Automatically dispatches to 512-bit AVX-512 (`vpaddw zmm`, 8 ops per 256-wide row) on AVX-512 hosts
    /// (such as Botzone's Xeon/EPYC servers) and 256-bit AVX-2 (`vpaddw ymm`, 16 ops per 256-wide row) on AVX-2 PCs.
    pub fn forward(&self, b: &Board, s57: f32) -> f32 {
        self.forward_with_dis(b, s57).0
    }

    /// Fast path returning both `(eval_logit, dis_cp)` from a single shared accumulator + hidden pass.
    pub fn forward_with_dis(&self, b: &Board, s57: f32) -> (f32, i32) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: AVX-512F/BW/VL and FMA support verified at runtime above.
                return unsafe { self.forward_with_dis_avx512(b, s57) };
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: AVX-2 and FMA support verified at runtime above.
                return unsafe { self.forward_with_dis_avx2(b, s57) };
            }
        }
        self.forward_with_dis_inner(b, s57)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
    unsafe fn forward_with_dis_avx512(&self, b: &Board, s57: f32) -> (f32, i32) {
        self.forward_with_dis_inner(b, s57)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_with_dis_avx2(&self, b: &Board, s57: f32) -> (f32, i32) {
        self.forward_with_dis_inner(b, s57)
    }

    #[inline(always)]
    fn forward_with_dis_inner(&self, b: &Board, s57: f32) -> (f32, i32) {
        let (us, nu, them, nt) = nnfeat::both(b, self.groups);
        let w = self.width;
        let empties = b.empty().count_ones() as usize;
        let bucket = (empties * self.buckets / 50).min(self.buckets - 1);
        let pst = &self.psqt_t[bucket * self.num_feats..(bucket + 1) * self.num_feats];
        // both halves live back to back in one uninitialised scratch (every lane is written by ft_sum before it is read):
        // no 4 KB zero-fill per eval and half the stack lines touched
        let mut acc_buf = std::mem::MaybeUninit::<[i16; 2 * MAX_W]>::uninit();
        let accp = acc_buf.as_mut_ptr() as *mut i16;
        for (half, (idx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            // SAFETY: half * w + w <= 2 * MAX_W; ft_sum initialises all w lanes of this half
            unsafe { self.ft_sum(accp.add(half * w), w, &idx[..n]) };
        }
        // SAFETY: both halves (2 * w lanes) were just fully written
        let acc: &[i16] = unsafe { std::slice::from_raw_parts(accp, 2 * w) };
        // the two sides' position-weight sums are independent float chains: run them side by side (each keeps its order)
        let (iu, it) = (&us[..nu], &them[..nt]);
        let m = nu.min(nt);
        let (mut p0, mut p1) = (0f32, 0f32);
        for k in 0..m {
            p0 += pst[iu[k] as usize];
            p1 += pst[it[k] as usize];
        }
        for &f in &iu[m..] {
            p0 += pst[f as usize];
        }
        for &f in &it[m..] {
            p1 += pst[f as usize];
        }
        let ps = [p0, p1];
        self.head_with_dis(acc, bucket, ps, s57)
    }

    /// a = bias + sum of the active rows (wrapping i16, so any order is bit-identical). The accumulator is kept in
    /// registers one CH-lane block at a time: each row costs one load+add per vector instead of load/add/store.
    #[inline(always)]
    unsafe fn ft_sum(&self, out: *mut i16, w: usize, idx: &[u32]) {
        const CH: usize = 128;
        if w % CH != 0 {
            std::ptr::copy_nonoverlapping(self.ftbq.as_ptr(), out, w);
            let a = std::slice::from_raw_parts_mut(out, w);
            for &f in idx {
                let row = &self.ftq[f as usize * w..(f as usize + 1) * w];
                for (x, r) in a.iter_mut().zip(row) {
                    *x = x.wrapping_add(*r);
                }
            }
            return;
        }
        let n = idx.len();
        for c in (0..w).step_by(CH) {
            let mut r = [0i16; CH];
            r.copy_from_slice(&self.ftbq[c..c + CH]);
            for k in 0..n {
                let f = idx[k] as usize;
                let row = &self.ftq[f * w + c..f * w + c + CH];
                for j in 0..CH {
                    r[j] = r[j].wrapping_add(row[j]);
                }
            }
            std::ptr::copy_nonoverlapping(r.as_ptr(), out.add(c), CH);
        }
    }

    pub fn bench_ft(&self, b: &Board) -> i32 {
        let (us, nu, them, nt) = nnfeat::both(b, self.groups);
        let w = self.width;
        let empties = b.empty().count_ones() as usize;
        let bucket = (empties * self.buckets / 50).min(self.buckets - 1);
        let pst = &self.psqt_t[bucket * self.num_feats..(bucket + 1) * self.num_feats];
        let mut acc = [[0i16; MAX_W]; 2];
        let mut ps = [0f32; 2];
        for (half, (idx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let idx = &idx[..n];
            // SAFETY: w <= MAX_W
            unsafe { self.ft_sum(acc[half].as_mut_ptr(), w, idx) };
            for &f in idx {
                ps[half] += pst[f as usize];
            }
        }
        std::hint::black_box(&acc);
        acc[0][3] as i32 + acc[1][5] as i32 + ps[0] as i32
    }
    /// Everything after the feature transformer: int16 accumulators -> activations -> sparse L1 -> L2 -> output.
    /// Activations are computed on the fly (no zeroed scratch array); arithmetic and order are exactly as before.
    #[inline(always)]
    fn head_with_dis(&self, acc: &[i16], bucket: usize, ps: [f32; 2], s57: f32) -> (f32, i32) {
        let w = self.width;
        let inv = 1.0 / QA as f32;
        let qa = QA as i16;
        let l1 = self.l1;
        let mut h1 = [0f32; MAX_H];
        h1[..l1].copy_from_slice(&self.l1_b[bucket * l1..(bucket + 1) * l1]);
        let din = if self.pairwise { w } else { 2 * w };
        let base = bucket * din * l1;
        // nonzero inputs collected branch-free (no mispredicts on the ~86% zeros), then summed in ascending input order:
        // exactly the terms and order of the plain sparse loop, so the result is bit-identical
        let mut nz_buf = std::mem::MaybeUninit::<[u16; 2 * MAX_W + 8]>::uninit();
        let nzp = nz_buf.as_mut_ptr() as *mut u16;
        let mut cnt = 0usize;
        if self.pairwise {
            let h = w / 2;
            for half in 0..2 {
                let a = &acc[half * w..(half + 1) * w];
                for j in 0..h {
                    unsafe { *nzp.add(cnt) = (half * h + j) as u16 };
                    cnt += (a[j] > 0 && a[h + j] > 0) as usize;
                }
            }
        } else if cfg!(target_feature = "avx2") && w % 32 == 0 {
            #[cfg(target_feature = "avx2")]
            for half in 0..2 {
                use std::arch::x86_64::*;
                let a = &acc[half * w..(half + 1) * w];
                for base in (0..w).step_by(32) {
                    // SAFETY: AVX2 is enabled at compile time; base + 32 <= w <= MAX_W so both 16-lane loads are in bounds.
                    let m = unsafe {
                        let z = _mm256_setzero_si256();
                        let p = a.as_ptr().add(base) as *const __m256i;
                        let c0 = _mm256_cmpgt_epi16(_mm256_loadu_si256(p), z);
                        let c1 = _mm256_cmpgt_epi16(_mm256_loadu_si256(p.add(1)), z);
                        // packs works per 128-bit lane: fix the lane order, then one bit per i16 lane
                        let pk = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(c0, c1));
                        _mm256_movemask_epi8(pk) as u32
                    };
                    // four mask bytes: each byte's set-bit positions come from a table and are stored 8 at a time (entries
                    // past the popcount are overwritten by the next store); no data-dependent branches, same indices, same order
                    for k in 0..4 {
                        let byte = ((m >> (8 * k)) & 0xFF) as usize;
                        // SAFETY: the 8-entry store starts at cnt <= 2 * w - 1 and the buffer has 8 spare entries
                        unsafe {
                            let idx = _mm_cvtepu8_epi16(_mm_loadl_epi64(BIT_POS[byte].as_ptr() as *const __m128i));
                            let off = _mm_set1_epi16((half * w + base + 8 * k) as i16);
                            _mm_storeu_si128(nzp.add(cnt) as *mut __m128i, _mm_add_epi16(idx, off));
                        }
                        cnt += byte.count_ones() as usize;
                    }
                }
            }
        } else if w % 64 == 0 {
            // 64-lane masks (compare + movemask), then walk the set bits ascending: same indices, same order
            for half in 0..2 {
                let a = &acc[half * w..(half + 1) * w];
                let mut base = 0;
                while base < w {
                    let mut m = 0u64;
                    for k in 0..64 {
                        m |= ((a[base + k] > 0) as u64) << k;
                    }
                    while m != 0 {
                        unsafe { *nzp.add(cnt) = (half * w + base + m.trailing_zeros() as usize) as u16 };
                        cnt += 1;
                        m &= m - 1;
                    }
                    base += 64;
                }
            }
        } else {
            for half in 0..2 {
                let a = &acc[half * w..(half + 1) * w];
                for j in 0..w {
                    unsafe { *nzp.add(cnt) = (half * w + j) as u16 };
                    cnt += (a[j] > 0) as usize;
                }
            }
        }
        // SAFETY: the first cnt entries were written above (cnt <= 2 * w <= 2 * MAX_W)
        let nz: &[u16] = unsafe { std::slice::from_raw_parts(nzp, cnt) };
        if !self.pairwise && l1 == 16 {
            // accumulator held in registers; no division per input; per-lane add order = ascending input order, as before
            #[cfg(target_feature = "avx2")]
            {
                use std::arch::x86_64::*;
                // SAFETY: AVX2 is enabled at compile time; h1 has >= 16 lanes and every column slice is 16 floats.
                // mul then add (no fused multiply-add): the per-lane result is the same sequential sum as the scalar loop
                unsafe {
                    let mut h0 = _mm256_loadu_ps(h1.as_ptr());
                    let mut h8 = _mm256_loadu_ps(h1.as_ptr().add(8));
                    if w % 8 == 0 {
                        // every lane's activation once, vectorised (clamp to [0, QA], exact int -> f32, times 1/QA: the same
                        // three operations as the scalar form); the input loop then only broadcasts and multiply-adds
                        let mut vals = std::mem::MaybeUninit::<[f32; 2 * MAX_W]>::uninit();
                        let vp = vals.as_mut_ptr() as *mut f32;
                        let zero = _mm256_setzero_si256();
                        let top = _mm256_set1_epi16(qa);
                        let invv = _mm256_set1_ps(inv);
                        for c in (0..2 * w).step_by(16) {
                            let x = _mm256_loadu_si256(acc.as_ptr().add(c) as *const __m256i);
                            let x = _mm256_min_epi16(_mm256_max_epi16(x, zero), top);
                            let lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x)));
                            let hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(x)));
                            _mm256_storeu_ps(vp.add(c), _mm256_mul_ps(lo, invv));
                            _mm256_storeu_ps(vp.add(c + 8), _mm256_mul_ps(hi, invv));
                        }
                        assert!(base + din * 16 <= self.l1_wt.len());
                        let wt = self.l1_wt.as_ptr().add(base);
                        for &i in nz {
                            // i < 2 * w = din, so the 16-float column and the activation are in bounds
                            let i = i as usize;
                            let v = _mm256_broadcast_ss(&*vp.add(i));
                            let col = wt.add(i * 16);
                            h0 = _mm256_add_ps(h0, _mm256_mul_ps(v, _mm256_loadu_ps(col)));
                            h8 = _mm256_add_ps(h8, _mm256_mul_ps(v, _mm256_loadu_ps(col.add(8))));
                        }
                    } else {
                        for &i in nz {
                            let i = i as usize;
                            let v = _mm256_set1_ps(acc[i].clamp(0, qa) as f32 * inv);
                            let col = self.l1_wt[base + i * 16..base + i * 16 + 16].as_ptr();
                            h0 = _mm256_add_ps(h0, _mm256_mul_ps(v, _mm256_loadu_ps(col)));
                            h8 = _mm256_add_ps(h8, _mm256_mul_ps(v, _mm256_loadu_ps(col.add(8))));
                        }
                    }
                    _mm256_storeu_ps(h1.as_mut_ptr(), h0);
                    _mm256_storeu_ps(h1.as_mut_ptr().add(8), h8);
                }
            }
            #[cfg(not(target_feature = "avx2"))]
            {
                let mut h = [0f32; 16];
                h.copy_from_slice(&h1[..16]);
                for &i in nz {
                    let i = i as usize;
                    let v = acc[i].clamp(0, qa) as f32 * inv;
                    let col: &[f32; 16] = self.l1_wt[base + i * 16..base + i * 16 + 16].try_into().unwrap();
                    for j in 0..16 {
                        h[j] += v * col[j];
                    }
                }
                h1[..16].copy_from_slice(&h);
            }
        } else {
            for &i in nz {
                let i = i as usize;
                let v = if self.pairwise {
                    let h = w / 2;
                    let (half, j) = (i / h, i % h);
                    let a = &acc[half * w..(half + 1) * w];
                    (a[j].clamp(0, qa) as f32 * inv) * (a[h + j].clamp(0, qa) as f32 * inv)
                } else {
                    acc[i].clamp(0, qa) as f32 * inv
                };
                let col = &self.l1_wt[base + i * l1..base + (i + 1) * l1];
                for (h, c) in h1[..l1].iter_mut().zip(col) {
                    *h += v * c;
                }
            }
        }
        for h in h1[..l1].iter_mut() {
            *h = h.clamp(0.0, 1.0);
        }
        let l2 = self.l2;
        let mut h2 = [0f32; MAX_H];
        if l2 % 8 == 0 {
            // 8 outputs at a time: every lane is the same sequential sum over the l1 inputs (mul, then add, in input
            // order) as the plain per-output dot product, so the result is bit-identical; only the latency chains overlap
            let wt = &self.l2_wt[bucket * l1 * l2..(bucket + 1) * l1 * l2];
            for j0 in (0..l2).step_by(8) {
                let mut s = [-0.0f32; 8];
                for k in 0..l1 {
                    let hk = h1[k];
                    let w8: &[f32; 8] = wt[k * l2 + j0..k * l2 + j0 + 8].try_into().unwrap();
                    for l in 0..8 {
                        s[l] += w8[l] * hk;
                    }
                }
                for l in 0..8 {
                    h2[j0 + l] = (s[l] + self.l2_b[bucket * l2 + j0 + l]).clamp(0.0, 1.0);
                }
            }
        } else {
            for (j, h) in h2[..l2].iter_mut().enumerate() {
                let r = bucket * l2 + j;
                let row = &self.l2_w[r * l1..(r + 1) * l1];
                let s: f32 = row.iter().zip(&h1[..l1]).map(|(a, b)| a * b).sum();
                *h = (s + self.l2_b[r]).clamp(0.0, 1.0);
            }
        }
        let o = if self.v2 {
            let k = l1 + l2;
            let row = &self.out_w[bucket * k..(bucket + 1) * k];
            let s1: f32 = row[..l1].iter().zip(&h1[..l1]).map(|(a, b)| a * b).sum();
            let s2: f32 = row[l1..].iter().zip(&h2[..l2]).map(|(a, b)| a * b).sum();
            s1 + s2 + self.out_b[bucket] + self.s57w[bucket] * s57
        } else {
            let row = &self.out_w[bucket * l2..(bucket + 1) * l2];
            row.iter().zip(&h2[..l2]).map(|(a, b)| a * b).sum::<f32>() + self.out_b[bucket]
        };
        let eval_logit = o + 0.5 * (ps[0] - ps[1]);
        let dis_cp = if self.has_dis {
            let (l1, l2) = (self.l1, self.l2);
            let k = l1 + l2;
            let dis_row = &self.dis_w[bucket * k..(bucket + 1) * k];
            let d1: f32 = dis_row[..l1].iter().zip(&h1[..l1]).map(|(a, b)| a * b).sum();
            let d2: f32 = dis_row[l1..].iter().zip(&h2[..l2]).map(|(a, b)| a * b).sum();
            ((400.0f32 * (d1 + d2 + self.dis_b[bucket]).max(0.0f32)).round() as i32).clamp(0, 30000)
        } else {
            0
        };
        (eval_logit, dis_cp)
    }

    /// Exact training-parity path (plain f32, allocating): the reference the fast path is measured against.
    pub fn forward_f32(&self, b: &Board, s57: f32) -> f32 {
        let (us, nu, them, nt) = nnfeat::both(b, self.groups);
        let w = self.width;
        let mut acc = vec![0f32; 2 * w];
        let mut ps = [0f32; 2];
        let empties = b.empty().count_ones() as usize;
        let bucket = (empties * self.buckets / 50).min(self.buckets - 1);
        for (half, (idx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let a = &mut acc[half * w..(half + 1) * w];
            a.copy_from_slice(&self.ftb);
            for &f in &idx[..n] {
                let row = &self.ft[f as usize * w..(f as usize + 1) * w];
                for (x, r) in a.iter_mut().zip(row) {
                    *x += *r;
                }
                ps[half] += self.psqt[f as usize * self.buckets + bucket];
            }
            for x in a.iter_mut() {
                *x = x.clamp(0.0, 1.0);
            }
        }
        // Stockfish pairwise product: each perspective's W clipped values -> W/2 products (first half * second half)
        let x: Vec<f32> = if self.pairwise {
            let h = w / 2;
            let mut v = vec![0f32; w];
            for half in 0..2 {
                let a = &acc[half * w..(half + 1) * w];
                for j in 0..h {
                    v[half * h + j] = a[j] * a[h + j];
                }
            }
            v
        } else {
            acc
        };
        let din = x.len();
        let mut h1 = vec![0f32; self.l1];
        for (j, h) in h1.iter_mut().enumerate() {
            let r = bucket * self.l1 + j;
            let row = &self.l1_w[r * din..(r + 1) * din];
            let s: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            *h = (s + self.l1_b[r]).clamp(0.0, 1.0);
        }
        let mut h2 = vec![0f32; self.l2];
        for (j, h) in h2.iter_mut().enumerate() {
            let r = bucket * self.l2 + j;
            let row = &self.l2_w[r * self.l1..(r + 1) * self.l1];
            let s: f32 = row.iter().zip(&h1).map(|(a, b)| a * b).sum();
            *h = (s + self.l2_b[r]).clamp(0.0, 1.0);
        }
        let o = if self.v2 {
            let k = self.l1 + self.l2;
            let row = &self.out_w[bucket * k..(bucket + 1) * k];
            let s1: f32 = row[..self.l1].iter().zip(&h1).map(|(a, b)| a * b).sum();
            let s2: f32 = row[self.l1..].iter().zip(&h2).map(|(a, b)| a * b).sum();
            s1 + s2 + self.out_b[bucket] + self.s57w[bucket] * s57
        } else {
            let row = &self.out_w[bucket * self.l2..(bucket + 1) * self.l2];
            row.iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>() + self.out_b[bucket]
        };
        o + 0.5 * (ps[0] - ps[1])
    }

    /// Centipawn-like score from the side to move's view (same scale the training target used: 400 per logit).
    /// `s57_cp` = the S57 tuple eval of `b` in cp (side to move); ignored by v1 nets.
    /// Free the float copy of the feature transformer (~27 MB) once the int16 fast path is built. Only the
    /// training-parity reference `forward_f32` needs it; memory-capped hosts (Botzone, 256 MB) drop it.
    pub fn drop_f32_reference(&mut self) {
        self.ft = Vec::new();
    }

    pub fn eval(&self, b: &Board, s57_cp: i32) -> i32 {
        (400.0 * self.forward(b, s57_cp as f32 / 400.0))
            .round()
            .clamp(-30000.0, 30000.0) as i32
    }

    /// Fast path returning both (eval_cp, dis_cp).
    pub fn eval_with_dis(&self, b: &Board, s57_cp: i32) -> (i32, i32) {
        let (logit, dis_cp) = self.forward_with_dis(b, s57_cp as f32 / 400.0);
        let eval_cp = (400.0 * logit).round().clamp(-30000.0, 30000.0) as i32;
        (eval_cp, dis_cp)
    }

    /// Reference eval through `forward_f32` (training parity).
    pub fn eval_f32(&self, b: &Board, s57_cp: i32) -> i32 {
        (400.0 * self.forward_f32(b, s57_cp as f32 / 400.0))
            .round()
            .clamp(-30000.0, 30000.0) as i32
    }
}

#[allow(dead_code)]
const _ASSERT: () = assert!(MAX_ACTIVE < 256);
