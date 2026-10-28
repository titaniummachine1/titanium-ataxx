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
/// prefetch distance (features ahead) for the feature-transformer rows
const PF: usize = 8;

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
    ftq: Vec<i16>,
    ftbq: Vec<i16>,
    /// l1 weights transposed per bucket: [bucket][din][l1] (one contiguous l1 column per input, for sparse L1)
    l1_wt: Vec<f32>,
    /// psqt transposed: [bucket][feature] (one bucket's column is contiguous: ~105 KB, stays in L2)
    psqt_t: Vec<f32>,
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
        let v2 = d.len() >= 4 && &d[..4] == b"TNN2";
        if d.len() < 28 || (&d[..4] != b"TNN1" && !v2) {
            return Err(bad("not a TNN1/TNN2 file"));
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        let (num_feats, width, l1, l2, buckets, groups) =
            (u(0), u(1), u(2), u(3), u(4), u(5) as u32);
        let pairwise = v2 && u(6) != 0;
        if num_feats != nnfeat::NUM_FEATS {
            return Err(bad("feature count differs from this engine's nnfeat"));
        }
        if pairwise && width % 2 != 0 {
            return Err(bad("pairwise needs an even width"));
        }
        let din = if pairwise { width } else { 2 * width };
        let out_in = if v2 { l1 + l2 } else { l2 };
        let mut p = if v2 { 32 } else { 28 };
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
            ftq: Vec::new(),
            ftbq: Vec::new(),
            l1_wt: Vec::new(),
            psqt_t: Vec::new(),
            num_feats,
            width,
            l1,
            l2,
            buckets,
            groups,
            v2,
            pairwise,
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
        net.ftq = net
            .ft
            .iter()
            .map(q)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| bad("ft weight overflows int16 at QA"))?;
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
        net.l1_wt = t;
        let mut pt = vec![0f32; net.buckets * net.num_feats];
        for f in 0..net.num_feats {
            for bk in 0..net.buckets {
                pt[bk * net.num_feats + f] = net.psqt[f * net.buckets + bk];
            }
        }
        net.psqt_t = pt;
        Ok(net)
    }

    /// Fast path (int16 feature transformer, allocation-free, sparse L1); differs from `forward_f32` only by the
    /// QA rounding of the first layer (measured by `nneval`).
    /// Automatically dispatches to 512-bit AVX-512 (`vpaddw zmm`, 8 ops per 256-wide row) on AVX-512 hosts
    /// (such as Botzone's Xeon/EPYC servers) and 256-bit AVX-2 (`vpaddw ymm`, 16 ops per 256-wide row) on AVX-2 PCs.
    pub fn forward(&self, b: &Board, s57: f32) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: AVX-512F/BW/VL and FMA support verified at runtime above.
                return unsafe { self.forward_avx512(b, s57) };
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: AVX-2 and FMA support verified at runtime above.
                return unsafe { self.forward_avx2(b, s57) };
            }
        }
        self.forward_inner(b, s57)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
    unsafe fn forward_avx512(&self, b: &Board, s57: f32) -> f32 {
        self.forward_inner(b, s57)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, b: &Board, s57: f32) -> f32 {
        self.forward_inner(b, s57)
    }

    #[inline(always)]
    fn forward_inner(&self, b: &Board, s57: f32) -> f32 {
        let (us, nu, them, nt) = nnfeat::both(b, self.groups);
        let w = self.width;
        let empties = b.empty().count_ones() as usize;
        let bucket = (empties * self.buckets / 50).min(self.buckets - 1);
        let pst = &self.psqt_t[bucket * self.num_feats..(bucket + 1) * self.num_feats];
        let mut acc = [[0i16; MAX_W]; 2];
        let mut ps = [0f32; 2];
        for (half, (idx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let a = &mut acc[half][..w];
            a.copy_from_slice(&self.ftbq);
            let idx = &idx[..n];
            for k in 0..n {
                // the 13 MB row table misses cache: request the row PF features ahead while adding this one
                if k + PF < n {
                    self.prefetch_row(idx[k + PF] as usize);
                }
                let f = idx[k] as usize;
                let row = &self.ftq[f * w..(f + 1) * w];
                for (x, r) in a.iter_mut().zip(row) {
                    *x = x.wrapping_add(*r);
                }
                ps[half] += pst[f];
            }
        }
        self.head(&acc, bucket, ps, s57)
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
            let a = &mut acc[half][..w];
            a.copy_from_slice(&self.ftbq);
            let idx = &idx[..n];
            for k in 0..n {
                if k + PF < n {
                    self.prefetch_row(idx[k + PF] as usize);
                }
                let f = idx[k] as usize;
                let row = &self.ftq[f * w..(f + 1) * w];
                for (x, r) in a.iter_mut().zip(row) {
                    *x = x.wrapping_add(*r);
                }
                ps[half] += pst[f];
            }
        }
        std::hint::black_box(&acc);
        acc[0][3] as i32 + acc[1][5] as i32 + ps[0] as i32
    }
    #[inline(always)]
    fn prefetch_row(&self, f: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            let base = self.ftq.as_ptr().wrapping_add(f * self.width) as *const i8;
            let bytes = self.width * 2;
            let mut o = 0;
            while o < bytes {
                // SAFETY: prefetch is a hint; it never faults, even on an out-of-range address.
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                        base.wrapping_add(o),
                    )
                };
                o += 64;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = f;
    }

    /// Everything after the feature transformer: int16 accumulators -> activations -> sparse L1 -> L2 -> output.
    /// Activations are computed on the fly (no zeroed scratch array); arithmetic and order are exactly as before.
    #[inline(always)]
    fn head(&self, acc: &[[i16; MAX_W]; 2], bucket: usize, ps: [f32; 2], s57: f32) -> f32 {
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
        let mut nz = [0u16; 2 * MAX_W];
        let mut cnt = 0usize;
        if self.pairwise {
            let h = w / 2;
            for half in 0..2 {
                let a = &acc[half];
                for j in 0..h {
                    nz[cnt] = (half * h + j) as u16;
                    cnt += (a[j] > 0 && a[h + j] > 0) as usize;
                }
            }
        } else {
            for half in 0..2 {
                let a = &acc[half];
                for j in 0..w {
                    nz[cnt] = (half * w + j) as u16;
                    cnt += (a[j] > 0) as usize;
                }
            }
        }
        for &i in &nz[..cnt] {
            let i = i as usize;
            let v = if self.pairwise {
                let h = w / 2;
                let (half, j) = (i / h, i % h);
                let a = &acc[half];
                (a[j].clamp(0, qa) as f32 * inv) * (a[h + j].clamp(0, qa) as f32 * inv)
            } else {
                acc[i / w][i % w].clamp(0, qa) as f32 * inv
            };
            let col = &self.l1_wt[base + i * l1..base + (i + 1) * l1];
            for (h, c) in h1[..l1].iter_mut().zip(col) {
                *h += v * c;
            }
        }
        for h in h1[..l1].iter_mut() {
            *h = h.clamp(0.0, 1.0);
        }
        let l2 = self.l2;
        let mut h2 = [0f32; MAX_H];
        for (j, h) in h2[..l2].iter_mut().enumerate() {
            let r = bucket * l2 + j;
            let row = &self.l2_w[r * l1..(r + 1) * l1];
            let s: f32 = row.iter().zip(&h1[..l1]).map(|(a, b)| a * b).sum();
            *h = (s + self.l2_b[r]).clamp(0.0, 1.0);
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
        o + 0.5 * (ps[0] - ps[1])
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

    /// Reference eval through `forward_f32` (training parity).
    pub fn eval_f32(&self, b: &Board, s57_cp: i32) -> i32 {
        (400.0 * self.forward_f32(b, s57_cp as f32 / 400.0))
            .round()
            .clamp(-30000.0, 30000.0) as i32
    }
}

#[allow(dead_code)]
const _ASSERT: () = assert!(MAX_ACTIVE < 256);
