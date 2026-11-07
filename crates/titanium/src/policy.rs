//! Move-ordering policy head (exp/policy, docs/POLICY_PLAN.md). logit(m) = bias[slot] + scalar move features
//! + U[slot] . xin, where xin is the sparse pairwise accumulator product the value net's L1 reads. Fitted by `ti policy`.
//! File: magic "TPOL1\0\0\0", u32 dim, f32 bias[98], f32 g[45], f32 u[98*dim], little endian.
//! TPOL2 (D486): the same, then f32 dest[98][6561] + f32 orig[49][6561]: one lookup per move on the base-3 code of the
//! 8 cells around the destination (and, for jumps, around the origin), mover-relative, as `ti policy --pat k3` fits it.
use crate::board::{Board, Move, RING1};


/// Bundled head (fitted by `ti policy --feat x --out-bin` on the bundled net data/nnue/nnue_main.tnn). Env TI_POLICY=none disables, =path overrides.
static BUNDLED: &[u8] = include_bytes!("../../../data/nnue/policy_main.bin");

pub const SLOTS: usize = 98;
pub const NSCAL: usize = 45;

pub struct Policy {
    pub dim: usize,
    bias: [f32; SLOTS],
    g: [f32; NSCAL],
    u: Vec<f32>,
    /// D486 pattern tables (empty for TPOL1)
    td: Vec<f32>,
    tor: Vec<f32>,
}

const P3: usize = 6561;

/// D493 speed: per square, the 8 neighbour squares in the fixed order NW N NE W E SW S SE; off-board = bit 63 (never set
/// on a 49-square board, so it reads as empty = digit 0, exactly like the trainer's bounds check).
static NB: [[u8; 8]; 49] = {
    let mut t = [[63u8; 8]; 49];
    let d: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
    let mut a = 0;
    while a < 49 {
        let (r, c) = ((a / 7) as i32, (a % 7) as i32);
        let mut j = 0;
        while j < 8 {
            let (rr, cc) = (r + d[j].0, c + d[j].1);
            if rr >= 0 && rr < 7 && cc >= 0 && cc < 7 {
                t[a][j] = (rr * 7 + cc) as u8;
            }
            j += 1;
        }
        a += 1;
    }
    t
};

/// T3[bits] = base-3 value of 8 binary digits, bit j = digit j, digit 0 most significant (weight 3^7)
static T3: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut c = 0;
    while c < 256 {
        let mut v = 0u16;
        let mut j = 0;
        while j < 8 {
            v = v * 3 + ((c >> j) & 1) as u16;
            j += 1;
        }
        t[c] = v;
        c += 1;
    }
    t
};

#[inline(always)]
fn bits8(x: u64, nb: &[u8; 8]) -> usize {
    let mut v = 0usize;
    let mut j = 0;
    while j < 8 {
        v |= (((x >> nb[j]) & 1) as usize) << j;
        j += 1;
    }
    v
}

/// Base-3 code of the 8 neighbours of `sq` (0 empty/off-board, 1 `me`, 2 `opp`), order NW N NE W E SW S SE, digit 0 most
/// significant; must match titaniumtest src/policy.rs k3code exactly (test k3code_matches_reference).
#[inline]
fn k3code(me: u64, opp: u64, sq: u8) -> usize {
    let nb = &NB[sq as usize];
    T3[bits8(me, nb)] as usize + 2 * T3[bits8(opp, nb)] as usize
}

#[cfg(test)]
mod tests {
    fn k3ref(me: u64, opp: u64, sq: u8) -> usize {
        let (r, c) = ((sq / 7) as i32, (sq % 7) as i32);
        let mut code = 0usize;
        for (dr, dc) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
            let (rr, cc) = (r + dr, c + dc);
            let d = if (0..7).contains(&rr) && (0..7).contains(&cc) {
                let bit = 1u64 << (rr * 7 + cc);
                if me & bit != 0 { 1 } else if opp & bit != 0 { 2 } else { 0 }
            } else {
                0
            };
            code = code * 3 + d;
        }
        code
    }
    #[test]
    fn k3code_matches_reference() {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        for _ in 0..20000 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let a = x & ((1u64 << 49) - 1);
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let me = a & x;
            let opp = a & !x;
            for sq in 0..49u8 {
                assert_eq!(super::k3code(me, opp, sq), k3ref(me, opp, sq));
            }
        }
    }
}

impl Policy {
    pub fn load(path: &str) -> std::io::Result<Policy> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(d: &[u8]) -> std::io::Result<Policy> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let v2 = d.len() >= 12 && &d[..8] == b"TPOL2\0\0\0";
        if d.len() < 12 || (&d[..8] != b"TPOL1\0\0\0" && !v2) {
            return Err(bad("bad policy magic"));
        }
        let dim = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        let n1 = SLOTS + NSCAL + SLOTS * dim;
        let n = n1 + if v2 { (SLOTS + 49) * P3 } else { 0 };
        if d.len() != 12 + 4 * n {
            return Err(bad("bad policy size"));
        }
        let f: Vec<f32> = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut bias = [0f32; SLOTS];
        bias.copy_from_slice(&f[..SLOTS]);
        let mut g = [0f32; NSCAL];
        g.copy_from_slice(&f[SLOTS..SLOTS + NSCAL]);
        let (td, tor) = if v2 { (f[n1..n1 + SLOTS * P3].to_vec(), f[n1 + SLOTS * P3..].to_vec()) } else { (Vec::new(), Vec::new()) };
        Ok(Policy { dim, bias, g, u: f[SLOTS + NSCAL..n1].to_vec(), td, tor })
    }

    /// Logit of move `m` in `b` given the sparse accumulator product `xin` (index, value).
    #[inline]
    pub fn logit(&self, b: &Board, m: Move, xin: &[(u16, f32)]) -> f32 {
        let clone = m.is_clone();
        let c = clone as usize;
        let empty = b.empty();
        let flips = (b.captures(m.to) as usize).min(8);
        let tholes = ((RING1[m.to as usize] & empty).count_ones() as usize).min(8);
        let slot = m.to as usize + 49 * (!clone) as usize;
        let mut z = self.bias[slot] + self.g[c * 9 + flips] + self.g[18 + c * 9 + tholes];
        if !clone {
            z += self.g[36 + ((RING1[m.from as usize] & empty).count_ones() as usize).min(8)];
        }
        let row = &self.u[slot * self.dim..(slot + 1) * self.dim];
        for &(i, v) in xin {
            z += row[i as usize] * v;
        }
        if !self.td.is_empty() {
            let (me, opp) = (b.me(), b.opp());
            z += self.td[slot * P3 + k3code(me, opp, m.to)];
            if !clone {
                z += self.tor[m.from as usize * P3 + k3code(me, opp, m.from)];
            }
        }
        z
    }
}

/// Process-wide policy for experiments: env TI_POLICY=<TPOL1 file>; ordering weight TI_POLW (default 100000 points per logit),
/// minimum depth TI_POLMIN (default 1). None when TI_POLICY is unset or unreadable.
pub fn global() -> Option<&'static Policy> {
    static P: std::sync::OnceLock<Option<Policy>> = std::sync::OnceLock::new();
    P.get_or_init(|| match spec().as_deref() {
        Some("none") => None,
        Some("bundled") | None => Policy::from_bytes(BUNDLED).ok(),
        Some(p) => Some(Policy::load(p).unwrap_or_else(|e| panic!("--policy / TI_POLICY {p}: {e}"))),
    })
    .as_ref()
}

static CONFIGURED: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Command-line `--policy <bundled|FILE|none>` (call once, before the first search); beats env TI_POLICY.
pub fn configure(spec: &str) {
    let _ = CONFIGURED.set(spec.to_string());
}

fn spec() -> Option<String> {
    CONFIGURED.get().cloned().or_else(|| std::env::var("TI_POLICY").ok())
}

/// True when the user chose a policy file explicitly (then it is used with any net); the default bundled head is
/// fitted on the bundled net only and switches itself off for any other net (see Searcher::set_nnue).
pub fn is_forced() -> bool {
    !matches!(spec().as_deref(), None | Some("bundled"))
}

pub fn bundled_bytes() -> &'static [u8] {
    BUNDLED
}

/// The compiled-in head, parsed (None only if the blob is corrupt).
pub fn bundled() -> Option<&'static Policy> {
    static B: std::sync::OnceLock<Option<Policy>> = std::sync::OnceLock::new();
    B.get_or_init(|| Policy::from_bytes(BUNDLED).ok()).as_ref()
}

pub fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
