//! RL prototype (Jataxx-inspired): 25 overlapping 3x3 windows (3^9 = 19683
//! ternary states each, stm-relative) -> hidden layer (squared ReLU) ->
//! 3 output buckets by piece count. Used as an ADDITIVE RESIDUAL on top of the
//! classical eval: `eval = classical + 100 * net(board)`. A zero net is
//! exactly the classical eval, so it can bootstrap from outcome-labelled
//! self-play (logistic regression with the classical eval as offset).
//!
//! Enabled at runtime by env `TITANIUM_NET3=<file>` (read once). Absent =
//! zero cost (one atomic load + predictable branch).
//!
//! File: b"NET3" | h u32 | w1 f32[25*19683*h] | b1 f32[h] | w2 f32[3*h] | b2 f32[3]

use crate::board::Board;
use std::sync::OnceLock;

pub const WIN: usize = 25;
pub const PAT: usize = 19683;
pub const BUCKETS: usize = 3;
pub const MAGIC: [u8; 4] = *b"NET3";
/// Centipawns per unit of net output.
pub const OUT_CP: f32 = 100.0;
/// Hidden pre-activation clamp (squared ReLU saturates here).
pub const ZMAX: f32 = 6.0;

pub struct Net3 {
    pub h: usize,
    pub w1: Vec<f32>, // [(w*PAT + state) * h + j]
    pub b1: Vec<f32>,
    pub w2: Vec<f32>, // [bucket * h + j]
    pub b2: [f32; BUCKETS],
}

/// Window top-left squares and per-window 9-bit masks (row-major in 7x7).
fn window_masks() -> &'static [u64; WIN] {
    static M: OnceLock<[u64; WIN]> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = [0u64; WIN];
        let mut n = 0;
        for r in 0..5 {
            for c in 0..5 {
                let mut x = 0u64;
                for dr in 0..3 {
                    for dc in 0..3 {
                        x |= 1u64 << ((r + dr) * 7 + c + dc);
                    }
                }
                m[n] = x;
                n += 1;
            }
        }
        m
    })
}

/// base-3 code of (own 9-bit, enemy 9-bit): digit j = 1 own, 2 enemy.
fn base3() -> &'static Vec<u16> {
    static T: OnceLock<Vec<u16>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![0u16; 512 * 512];
        for one in 0..512usize {
            for two in 0..512usize {
                if one & two != 0 {
                    continue;
                }
                let mut code = 0usize;
                let mut p = 1usize;
                for j in 0..9 {
                    if one >> j & 1 == 1 {
                        code += p;
                    } else if two >> j & 1 == 1 {
                        code += 2 * p;
                    }
                    p *= 3;
                }
                t[one * 512 + two] = code as u16;
            }
        }
        t
    })
}

#[inline]
fn pext(x: u64, mask: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        core::arch::x86_64::_pext_u64(x, mask)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut out = 0u64;
        let mut k = 0;
        let mut m = mask;
        while m != 0 {
            let b = m & m.wrapping_neg();
            if x & b != 0 {
                out |= 1 << k;
            }
            k += 1;
            m &= m - 1;
        }
        out
    }
}

/// Active feature rows (window*PAT + state) from the side to move's view.
#[inline]
pub fn features_from(own: u64, foe: u64, out: &mut [u32; WIN]) {
    let masks = window_masks();
    let t = base3();
    for w in 0..WIN {
        let a = pext(own, masks[w]) as usize;
        let e = pext(foe, masks[w]) as usize;
        out[w] = (w * PAT) as u32 + t[a * 512 + e] as u32;
    }
}

#[inline]
pub fn features(b: &Board, out: &mut [u32; WIN]) {
    let t = b.turn as usize;
    features_from(b.occ[t], b.occ[1 - t], out);
}

#[inline]
pub fn bucket_of(pieces: u32) -> usize {
    if pieces < 10 {
        0
    } else if pieces < 20 {
        1
    } else {
        2
    }
}

impl Net3 {
    pub fn zeros(h: usize) -> Net3 {
        Net3 {
            h,
            w1: vec![0.0; WIN * PAT * h],
            b1: vec![0.0; h],
            w2: vec![0.0; BUCKETS * h],
            b2: [0.0; BUCKETS],
        }
    }

    /// Raw output (stone-ish units) for a feature set + bucket.
    #[inline]
    pub fn forward_raw(&self, feats: &[u32; WIN], bucket: usize) -> f32 {
        let h = self.h;
        let mut z = [0f32; 256];
        z[..h].copy_from_slice(&self.b1);
        for &f in feats.iter() {
            let row = &self.w1[f as usize * h..f as usize * h + h];
            for j in 0..h {
                z[j] += row[j];
            }
        }
        let w2 = &self.w2[bucket * h..bucket * h + h];
        let mut o = self.b2[bucket];
        for j in 0..h {
            let x = z[j].clamp(0.0, ZMAX);
            o += w2[j] * x * x;
        }
        o
    }

    /// Centipawn residual, stm-relative.
    #[inline]
    pub fn forward_cp(&self, b: &Board) -> i32 {
        let mut f = [0u32; WIN];
        features(b, &mut f);
        let bk = bucket_of((b.piece_cnt[0] + b.piece_cnt[1]) as u32);
        let o = self.forward_raw(&f, bk);
        (o * OUT_CP).clamp(-1500.0, 1500.0) as i32
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::with_capacity(8 + 4 * (self.w1.len() + self.b1.len() + self.w2.len() + 3));
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&(self.h as u32).to_le_bytes());
        for v in self.w1.iter().chain(self.b1.iter()).chain(self.w2.iter()).chain(self.b2.iter()) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, buf)
    }

    pub fn load(path: &str) -> std::io::Result<Net3> {
        let buf = std::fs::read(path)?;
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        if buf.len() < 8 || buf[0..4] != MAGIC {
            return Err(bad("bad NET3 magic"));
        }
        let h = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        if h == 0 || h > 256 {
            return Err(bad("hidden size out of range (1..=256)"));
        }
        let need = 8 + 4 * (WIN * PAT * h + h + BUCKETS * h + BUCKETS);
        if buf.len() != need {
            return Err(bad("NET3 size mismatch"));
        }
        let mut it = buf[8..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        let mut take = |n: usize| -> Vec<f32> { it.by_ref().take(n).collect() };
        let w1 = take(WIN * PAT * h);
        let b1 = take(h);
        let w2 = take(BUCKETS * h);
        let b2v = take(BUCKETS);
        Ok(Net3 { h, w1, b1, w2, b2: [b2v[0], b2v[1], b2v[2]] })
    }
}

/// Global net from env `TITANIUM_NET3` (None if unset or unreadable).
#[inline]
pub fn global() -> Option<&'static Net3> {
    static G: OnceLock<Option<Net3>> = OnceLock::new();
    G.get_or_init(|| match std::env::var("TITANIUM_NET3") {
        Ok(p) if !p.is_empty() && p != "none" => match Net3::load(&p) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("TITANIUM_NET3 load failed ({p}): {e}");
                None
            }
        },
        _ => None,
    })
    .as_ref()
}
