//! Move-ordering policy head (exp/policy, docs/POLICY_PLAN.md). logit(m) = bias[slot] + scalar move features
//! + U[slot] . xin, where xin is the sparse pairwise accumulator product the value net's L1 reads. Fitted by `ti policy`.
//! File: magic "TPOL1\0\0\0", u32 dim, f32 bias[98], f32 g[45], f32 u[98*dim], little endian.
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
}

impl Policy {
    pub fn load(path: &str) -> std::io::Result<Policy> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(d: &[u8]) -> std::io::Result<Policy> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        if d.len() < 12 || &d[..8] != b"TPOL1\0\0\0" {
            return Err(bad("bad policy magic"));
        }
        let dim = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        let n = SLOTS + NSCAL + SLOTS * dim;
        if d.len() != 12 + 4 * n {
            return Err(bad("bad policy size"));
        }
        let f: Vec<f32> = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut bias = [0f32; SLOTS];
        bias.copy_from_slice(&f[..SLOTS]);
        let mut g = [0f32; NSCAL];
        g.copy_from_slice(&f[SLOTS..SLOTS + NSCAL]);
        Ok(Policy { dim, bias, g, u: f[SLOTS + NSCAL..].to_vec() })
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
