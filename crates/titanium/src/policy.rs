//! Move-ordering policy head (docs/POLICY_PLAN.md). logit(m) = bias[slot] + scalar move features + the 3x3 pattern
//! tables around the destination (and, for jumps, around the origin), shared by the 9 edge classes (TPOL3, D502).
//! File: magic "TPOL3\0\0\0", u32 dim (must be 0), f32 bias[98], f32 g[45], f32 dest[18][6561], f32 orig[9][6561].
//! "TPOL5" = the same plus a SEE block after g; play never reads it (D748/D751), so it is skipped on load (D894) until
//! the no-SEE head (D893) replaces the shipped one. TPOL1/TPOL2 (no tables / per-square tables, dim > 0) are gone (LAWS A11).
use crate::board::{Board, Move, RING1};


/// Bundled head (data/nnue/policy_main.bin). Env TI_POLICY=none disables, =path overrides.
static BUNDLED: &[u8] = include_bytes!("../../../data/nnue/policy_main.bin");

pub const SLOTS: usize = 98;
pub const NSCAL: usize = 45;
/// size of the TPOL5 SEE block that is skipped on load
const NSEE: usize = 8 * 18 * 19;

pub struct Policy {
    bias: [f32; SLOTS],
    g: [f32; NSCAL],
    /// D486/D502 pattern tables, i16 at TQ (1/2048 logit steps), shared by the 9 edge classes
    td: Vec<i16>,
    tor: Vec<i16>,
}

/// edge class of a square: (row class) * 3 + (col class), class 0 = first line, 1 = inner, 2 = last line
static CLS: [u8; 49] = {
    let mut t = [0u8; 49];
    let mut a = 0;
    while a < 49 {
        let (r, c) = (a / 7, a % 7);
        let fr: u8 = if r == 0 { 0 } else if r == 6 { 2 } else { 1 };
        let fc: u8 = if c == 0 { 0 } else if c == 6 { 2 } else { 1 };
        t[a] = fr * 3 + fc;
        a += 1;
    }
    t
};

const P3: usize = 6561;
const TQ: f32 = 2048.0;
const ND: usize = 18;
const NO: usize = 9;

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

/// D502: on-board neighbour mask per square
static NBM: [u64; 49] = {
    let mut t = [0u64; 49];
    let mut a = 0;
    while a < 49 {
        let mut j = 0;
        while j < 8 {
            if NB[a][j] != 63 {
                t[a] |= 1u64 << NB[a][j];
            }
            j += 1;
        }
        a += 1;
    }
    t
};

/// the 8 neighbour bits of `s` (bit j = NB order) with 4 shifts: mask off-board/wrapped cells, shift up by 8 so s-8 >= 0
#[inline(always)]
fn bits8f(x: u64, s: usize) -> usize {
    let y = (x & NBM[s]) << 8;
    (((y >> s) & 7) | (((y >> (s + 7)) & 1) << 3) | (((y >> (s + 9)) & 1) << 4) | (((y >> (s + 14)) & 7) << 5)) as usize
}

/// Base-3 code of the 8 neighbours of `sq` (0 empty/off-board, 1 `me`, 2 `opp`), order NW N NE W E SW S SE, digit 0 most
/// significant; must match titaniumtest src/policy.rs k3code exactly (test k3code_matches_reference).
#[inline]
fn k3code(me: u64, opp: u64, sq: u8) -> usize {
    let s = sq as usize;
    T3[bits8f(me, s)] as usize + 2 * T3[bits8f(opp, s)] as usize
}

#[inline(always)]
pub(crate) fn k3code_pub(me: u64, opp: u64, sq: u8) -> usize {
    k3code(me, opp, sq)
}

/// D502: per-node memo of the 3x3 codes (many moves share a destination or an origin square); new() per node
pub struct KCache {
    valid: u64,
    code: [u16; 49],
}

impl KCache {
    #[inline]
    pub fn new() -> KCache {
        KCache { valid: 0, code: [0; 49] }
    }
    #[inline(always)]
    fn get(&mut self, me: u64, opp: u64, sq: u8) -> usize {
        if self.valid >> sq & 1 == 0 {
            self.code[sq as usize] = k3code(me, opp, sq) as u16;
            self.valid |= 1u64 << sq;
        }
        self.code[sq as usize] as usize
    }
}

impl Default for KCache {
    fn default() -> Self {
        Self::new()
    }
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

    /// synthetic head as a file: TPOL3, or TPOL5 with a SEE block that must be skipped (same logits as the TPOL3)
    pub(crate) fn synth_bytes(tpol5: bool) -> Vec<u8> {
        let mut raw = if tpol5 { b"TPOL5\0\0\0".to_vec() } else { b"TPOL3\0\0\0".to_vec() };
        raw.extend_from_slice(&0u32.to_le_bytes());
        for i in 0..super::SLOTS { raw.extend_from_slice(&(((i as i32 - 49) as f32) * 0.03125).to_le_bytes()); }
        for i in 0..super::NSCAL { raw.extend_from_slice(&(((i as i32 - 20) as f32) * 0.0625).to_le_bytes()); }
        if tpol5 {
            for i in 0..super::NSEE { raw.extend_from_slice(&(((i as i32 * 31) % 200 - 100) as f32 * 0.015625).to_le_bytes()); }
        }
        for i in 0..((super::ND + super::NO) * super::P3) {
            raw.extend_from_slice(&(((i as i32 * 17) % 1000 - 500) as f32 / 2048.0).to_le_bytes());
        }
        raw
    }

    #[test]
    fn tpol5_loads_without_its_see_block() {
        let a = super::Policy::from_bytes(&synth_bytes(false)).expect("TPOL3");
        let b = super::Policy::from_bytes(&synth_bytes(true)).expect("TPOL5");
        assert_eq!(a.bias, b.bias);
        assert_eq!(a.g, b.g);
        assert_eq!(a.td, b.td);
        assert_eq!(a.tor, b.tor);
        let mut bad = synth_bytes(false);
        bad[8..12].copy_from_slice(&4u32.to_le_bytes());
        assert!(super::Policy::from_bytes(&bad).is_err(), "dim > 0 must be refused");
    }

    #[test]
    fn precomputed_logits_match_logit_k() {
        use crate::board::{Board, MoveList, RING1, RING2};
        let mut x: u64 = 0x517CC1B727220A95;
        let next_u64 = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };
        let bundled = super::bundled().expect("bundled policy");
        let synth = super::Policy::from_bytes(&synth_bytes(false)).expect("TPOL3");
        let mut list = MoveList::new();
        for pol in [bundled, &synth] {
            for n in 0..80 {
                let mut b = Board::start();
                for _ in 0..(n % 24) {
                    b.legal_moves_dedup_into(&mut list);
                    if list.is_empty() || b.game_over() {
                        break;
                    }
                    let idx = (next_u64(&mut x) as usize) % list.len();
                    b = b.make(list.move_at(idx));
                }
                if b.game_over() {
                    continue;
                }
                let mut kc = super::KCache::new();
                let me = b.me();
                let opp = b.opp();
                let empty = b.empty();
                let mut targets = crate::board::reach_union(me) & empty;
                while targets != 0 {
                    let to = targets.trailing_zeros() as u8;
                    targets &= targets - 1;
                    let r1_to = RING1[to as usize];
                    let flips = ((opp & r1_to).count_ones() as usize).min(8);
                    let tholes = ((r1_to & empty).count_ones() as usize).min(8);
                    let k3_to = super::k3code_pub(me, opp, to);
                    let clones = me & r1_to;
                    if clones != 0 {
                        let m = crate::board::Move { from: clones.trailing_zeros() as u8, to };
                        let expected = pol.logit_k(&b, m, &mut kc);
                        let actual = pol.clone_logit_pre(to, flips, tholes, k3_to);
                        assert_eq!(actual.to_bits(), expected.to_bits(), "clone mv {m:?}");
                    }
                    let (jp, jtd) = pol.jump_dest_parts(to, flips, tholes, k3_to);
                    let mut jumps = me & RING2[to as usize];
                    while jumps != 0 {
                        let from = jumps.trailing_zeros() as u8;
                        jumps &= jumps - 1;
                        let m = crate::board::Move { from, to };
                        let fh = ((RING1[from as usize] & empty).count_ones() as usize).min(8);
                        let k3_from = super::k3code_pub(me, opp, from);
                        let (fg, ft) = pol.jump_orig_parts(from, fh, k3_from);
                        let expected = pol.logit_k(&b, m, &mut kc);
                        let actual = pol.jump_finish(jp, jtd, fg, ft);
                        assert_eq!(actual.to_bits(), expected.to_bits(), "jump mv {m:?}");
                    }
                }
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
        if d.len() < 12 {
            return Err(bad("bad policy magic"));
        }
        let nsee = match &d[..8] {
            b"TPOL3\0\0\0" => 0,
            b"TPOL5\0\0\0" => NSEE,
            _ => return Err(bad("bad policy magic (only TPOL3/TPOL5 heads exist)")),
        };
        if u32::from_le_bytes(d[8..12].try_into().unwrap()) != 0 {
            return Err(bad("policy dim must be 0 (no accumulator input)"));
        }
        let n1 = SLOTS + NSCAL + nsee;
        if d.len() != 12 + 4 * (n1 + (ND + NO) * P3) {
            return Err(bad("bad policy size"));
        }
        let f: Vec<f32> = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut bias = [0f32; SLOTS];
        bias.copy_from_slice(&f[..SLOTS]);
        let mut g = [0f32; NSCAL];
        g.copy_from_slice(&f[SLOTS..SLOTS + NSCAL]);
        let q = |x: &[f32]| -> Vec<i16> { x.iter().map(|v| (v * TQ).round().clamp(-32767.0, 32767.0) as i16).collect() };
        let (td, tor) = (q(&f[n1..n1 + ND * P3]), q(&f[n1 + ND * P3..]));
        Ok(Policy { bias, g, td, tor })
    }

    /// Logit of move `m` in `b`.
    #[inline]
    pub fn logit(&self, b: &Board, m: Move) -> f32 {
        self.logit_k(b, m, &mut KCache::new())
    }

    /// `logit` with a per-node code cache (one KCache per node, same board for every move). Bit-identical to the
    /// ordering loop's `clone_logit_pre` / `jump_finish` (test precomputed_logits_match_logit_k).
    #[inline]
    pub fn logit_k(&self, b: &Board, m: Move, kc: &mut KCache) -> f32 {
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
        let (me, opp) = (b.me(), b.opp());
        let ds = CLS[m.to as usize] as usize * 2 + (!clone) as usize;
        let mut t = self.td[ds * P3 + kc.get(me, opp, m.to)] as i32;
        if !clone {
            t += self.tor[CLS[m.from as usize] as usize * P3 + kc.get(me, opp, m.from)] as i32;
        }
        z += t as f32 * (1.0 / TQ);
        z
    }

    /// Logit of a clone landing on `to` from precomputed per-destination features. Bit-identical to `logit_k`.
    #[inline(always)]
    pub(crate) fn clone_logit_pre(&self, to: u8, flips: usize, tholes: usize, k3_to: usize) -> f32 {
        let slot = to as usize;
        let mut z = self.bias[slot] + self.g[9 + flips] + self.g[27 + tholes];
        z += self.td[CLS[slot] as usize * 2 * P3 + k3_to] as i32 as f32 * (1.0 / TQ);
        z
    }

    /// Destination-only prefix and quantized pattern term for all jumps landing on `to`.
    #[inline(always)]
    pub(crate) fn jump_dest_parts(&self, to: u8, flips: usize, tholes: usize, k3_to: usize) -> (f32, i32) {
        let prefix = self.bias[to as usize + 49] + self.g[flips] + self.g[18 + tholes];
        (prefix, self.td[(CLS[to as usize] as usize * 2 + 1) * P3 + k3_to] as i32)
    }

    /// Origin-only scalar term and quantized pattern term for jumps leaving `from`.
    #[inline(always)]
    pub(crate) fn jump_orig_parts(&self, from: u8, from_holes: usize, k3_from: usize) -> (f32, i32) {
        (self.g[36 + from_holes], self.tor[CLS[from as usize] as usize * P3 + k3_from] as i32)
    }

    /// Finish a jump logit from precomputed destination + origin parts. Same IEEE-754 order as `logit_k`.
    #[inline(always)]
    pub(crate) fn jump_finish(&self, j_prefix: f32, j_td: i32, from_g: f32, from_tor: i32) -> f32 {
        let mut z = j_prefix + from_g;
        z += (j_td + from_tor) as f32 * (1.0 / TQ);
        z
    }
}

/// Process-wide policy: `--policy` / env TI_POLICY (bundled when unset, `none` = off); ordering weight TI_POLW (default
/// 100000 points per logit), minimum depth TI_POLMIN (default 1).
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

/// True when the user chose a policy file explicitly. Since D821 the bundled head stays on with any big net (each net is
/// overridden on its own: `--policy none` turns it off); kept for callers that report the choice.
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
