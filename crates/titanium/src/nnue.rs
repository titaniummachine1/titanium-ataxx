//! exp/nnue: float32 inference of the net trained by training/train_nnue.py (features = crate::nnfeat).
//! Correctness first: plain f32 (auto-vectorised under x86-64-v3); int16/int8 quantisation comes after the net is
//! proven by games. Layout must match training/export_nnue.py:
//!   magic "TNN1", then u32 LE: num_feats, width, l1, l2, buckets, groups mask; then f32 LE arrays in this order:
//!   ft[num_feats][width], ftb[width], psqt[num_feats][buckets], l1_w[buckets*l1][2*width], l1_b[buckets*l1],
//!   l2_w[buckets*l2][l1], l2_b[buckets*l2], out_w[buckets][l2], out_b[buckets].
//! Output: cp from the side to move's view = 400 * y (training: win prob = sigmoid(y)).

use crate::board::Board;
use crate::nnfeat::{self, MAX_ACTIVE};

pub struct Nnue {
    pub num_feats: usize,
    pub width: usize,
    pub l1: usize,
    pub l2: usize,
    pub buckets: usize,
    pub groups: u32,
    ft: Vec<f32>,
    ftb: Vec<f32>,
    psqt: Vec<f32>,
    l1_w: Vec<f32>,
    l1_b: Vec<f32>,
    l2_w: Vec<f32>,
    l2_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

impl Nnue {
    pub fn load(path: &str) -> std::io::Result<Nnue> {
        let d = std::fs::read(path)?;
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{path}: {m}"));
        if d.len() < 28 || &d[..4] != b"TNN1" {
            return Err(bad("not a TNN1 file"));
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        let (num_feats, width, l1, l2, buckets, groups) = (u(0), u(1), u(2), u(3), u(4), u(5) as u32);
        if num_feats != nnfeat::NUM_FEATS {
            return Err(bad("feature count differs from this engine's nnfeat"));
        }
        let mut p = 28;
        let mut take = |n: usize| -> Vec<f32> {
            let v = d[p..p + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            p += 4 * n;
            v
        };
        let net = Nnue {
            ft: take(num_feats * width),
            ftb: take(width),
            psqt: take(num_feats * buckets),
            l1_w: take(buckets * l1 * 2 * width),
            l1_b: take(buckets * l1),
            l2_w: take(buckets * l2 * l1),
            l2_b: take(buckets * l2),
            out_w: take(buckets * l2),
            out_b: take(buckets),
            num_feats,
            width,
            l1,
            l2,
            buckets,
            groups,
        };
        if p != d.len() {
            return Err(bad("trailing or missing bytes"));
        }
        Ok(net)
    }

    /// Raw net output y (win prob for the side to move = sigmoid(y)).
    pub fn forward(&self, b: &Board) -> f32 {
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
        let mut h1 = vec![0f32; self.l1];
        for (j, h) in h1.iter_mut().enumerate() {
            let r = bucket * self.l1 + j;
            let row = &self.l1_w[r * 2 * w..(r + 1) * 2 * w];
            let s: f32 = row.iter().zip(&acc).map(|(a, b)| a * b).sum();
            *h = (s + self.l1_b[r]).clamp(0.0, 1.0);
        }
        let mut h2 = vec![0f32; self.l2];
        for (j, h) in h2.iter_mut().enumerate() {
            let r = bucket * self.l2 + j;
            let row = &self.l2_w[r * self.l1..(r + 1) * self.l1];
            let s: f32 = row.iter().zip(&h1).map(|(a, b)| a * b).sum();
            *h = (s + self.l2_b[r]).clamp(0.0, 1.0);
        }
        let row = &self.out_w[bucket * self.l2..(bucket + 1) * self.l2];
        let o: f32 = row.iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>() + self.out_b[bucket];
        o + 0.5 * (ps[0] - ps[1])
    }

    /// Centipawn-like score from the side to move's view (same scale the training target used: 400 per logit).
    pub fn eval(&self, b: &Board) -> i32 {
        (400.0 * self.forward(b)).round().clamp(-30000.0, 30000.0) as i32
    }
}

#[allow(dead_code)]
const _ASSERT: () = assert!(MAX_ACTIVE < 256);
