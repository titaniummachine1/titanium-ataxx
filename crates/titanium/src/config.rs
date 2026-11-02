//! What this build plays with, and a guard that the defaults stay the strongest settings (user 2026-11-02:
//! "strongest engine by default, cannot accidentally build the wrong thing, still changeable with commands and net files").
//!
//! Defaults compiled in: big NNUE (data/nnue/nnue_main.tnn), futility small net (data/nnue/fut_small.tsn), move-ordering
//! policy head (data/nnue/policy_main.bin, fitted on the bundled big net), futility small net ON, policy weight 1e5 points
//! per logit, child-eval ordering (EVORD) OFF. data/nnue/BUNDLE.txt pins the sha256 of each file; the unit tests below fail
//! when a file is swapped without updating the pin (or when the policy head was not fitted on the bundled net).
//! Overrides, all explicit: `--network <bundled|FILE|none>`, `--policy <bundled|FILE|none>` (env TI_POLICY the same),
//! `--exact-search` (no small net), env TI_POLW / TI_POLMIN. `titanium-cli info` prints the effective configuration.

/// Minimal SHA-256 (lower-case hex) so file ids here match the sha8 ids used in the docs.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let tail_len = data.len() % 64;
    let full = data.len() - tail_len;
    let mut tail = data[full..].to_vec();
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    let mut block = |chunk: &[u8]| {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let mj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(mj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    };
    for c in data[..full].chunks_exact(64) {
        block(c);
    }
    for c in tail.chunks_exact(64) {
        block(c);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// FNV-1a over 8-byte words: a cheap identity of a net file (the loader needs it at every start; sha256 of 29 MB would
/// cost noticeably on a weak core).
pub fn fingerprint(d: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut it = d.chunks_exact(8);
    for c in &mut it {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x100000001b3);
    }
    for &b in it.remainder() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h ^ d.len() as u64
}

/// Pins from data/nnue/BUNDLE.txt: "net" | "policy" | "small" | "policy_fitted_on_net" -> sha256 hex.
pub fn pin(name: &str) -> Option<&'static str> {
    include_str!("../../../data/nnue/BUNDLE.txt")
        .lines()
        .filter_map(|l| l.split_once(' '))
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v.trim())
}

/// Multi-line effective configuration (the `info` command).
pub fn report(net: Option<&crate::nnue::Nnue>) -> String {
    let b = crate::search::build_knobs();
    let sha8 = |d: &[u8]| sha256_hex(d)[..8].to_uppercase();
    let bundled_net = crate::nnue::BUNDLED_NNUE;
    let is_bundled = net.map(|n| n.fingerprint) == Some(crate::nnue::Nnue::bundled_fingerprint());
    let mut s = String::new();
    s.push_str(&format!(
        "titanium {} | compiled with avx2 {} avx512f {}\n",
        env!("CARGO_PKG_VERSION"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512f")
    ));
    s.push_str(&format!("bundled big net    {} ({} B)\n", sha8(bundled_net), bundled_net.len()));
    s.push_str(&format!("bundled policy     {} ({} B)\n", sha8(crate::policy::bundled_bytes()), crate::policy::bundled_bytes().len()));
    s.push_str(&format!("bundled small net  {}\n", sha8(crate::smallnet::blob())));
    s.push_str(&format!(
        "active big net     {}\n",
        match net {
            None => "none (classical eval)".to_string(),
            Some(_) if is_bundled => "bundled".to_string(),
            Some(_) => "CUSTOM (the bundled policy head is off unless --policy FILE is given)".to_string(),
        }
    ));
    let pol_on = crate::policy::global().is_some() && (is_bundled || crate::policy::is_forced());
    s.push_str(&format!(
        "policy head        {} weight {} min depth {}\n",
        if pol_on { "ON" } else { "off" },
        crate::policy::env_f32("TI_POLW", 100000.0),
        crate::policy::env_f32("TI_POLMIN", 1.0)
    ));
    s.push_str(&format!("futility small net {}\n", if crate::search::fut_sn_on() { "ON" } else { "off (exact search)" }));
    s.push_str(&format!(
        "build knobs        FUT_CHEAP_M {} FUT_SN_M {} EVORD_W {} EVORD_MIN_DEPTH {} EVORD_CACHE {}\n",
        b.0, b.1, b.2, b.3, b.4
    ));
    if b != crate::search::DEFAULT_BUILD_KNOBS {
        s.push_str("WARNING: build knobs differ from the shipped defaults (a cargo-build env var was set); this is NOT the strongest configuration\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        let long = vec![b'a'; 1000];
        assert_eq!(sha256_hex(&long), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    }

    /// The compiled-in files are exactly the ones pinned in data/nnue/BUNDLE.txt, and the policy head was fitted on that net.
    #[test]
    fn bundled_files_match_the_pins() {
        assert_eq!(
            Some(sha256_hex(crate::nnue::BUNDLED_NNUE).as_str()),
            pin("net"),
            "data/nnue/nnue_main.tnn changed: update BUNDLE.txt AND refit policy_main.bin (ti policy)"
        );
        assert_eq!(Some(sha256_hex(crate::policy::bundled_bytes()).as_str()), pin("policy"), "data/nnue/policy_main.bin changed: update BUNDLE.txt");
        assert_eq!(Some(sha256_hex(crate::smallnet::blob()).as_str()), pin("small"), "data/nnue/fut_small.tsn changed: update BUNDLE.txt");
        assert_eq!(pin("net"), pin("policy_fitted_on_net"), "the bundled policy head was fitted on a different net: refit (ti policy) and set policy_fitted_on_net");
    }

    /// Shipped defaults are the strongest settings: no cargo-build env var changed a tuned constant.
    #[test]
    fn build_knobs_are_the_shipped_defaults() {
        assert_eq!(
            crate::search::build_knobs(),
            crate::search::DEFAULT_BUILD_KNOBS,
            "FUT_CHEAP_M / FUT_SN_M / EVORD_* env vars were set at build time"
        );
        assert!(crate::policy::bundled().is_some(), "bundled policy head does not parse");
    }

    /// A stale or wrong policy head would raise the node count; the bundled head must beat no head on fixed-depth search.
    #[test]
    fn bundled_policy_reduces_nodes() {
        use crate::board::Board;
        use crate::search::{SearchLimits, Searcher};
        let nn = crate::nnue::Nnue::bundled_static();
        let book = [
            "................x...o.x...o.........oo..xxo....xx w",
            "x.....o....oo....xoo.....oo.........x..x..x....xx w",
            "o.....o.o....................o....x.o...xxo....xx b",
            "xx...ooxx.....oo.....oo......o.........x..o....xx w",
            "x.x...oxx...oox....................oo....xo.o..xx w",
            "x....xx.....x.........oo.....oo...x.o.....o....xx w",
            "x.....ox....oo.x..................xo.o...xoo..xxx w",
            "x.....o.....o...........o.....ooxx..o.oxxxo....xx w",
            "o.....o.o...o.o....o................xx..x.x....xx b",
            "xx..o.o.x...o..x....o............x..o...x.oo..x.x w",
            "xx....ooo......o.......xx.....x.....oo..xxoo....x w",
            "x..x..oxxo.o.....o.................oo..x..oo...xx b",
            "x.....oxx...o.....o.......o.........o...xxooox..x b",
            "x.....o.o.x..x.ox..x.o..............o...x.o.o...x w",
            ".....xxo.o.....o............o.....x.o..x.xo....xx w",
            "xx....o.x.oooo...............oo...x.o...x......xx w",
            "x..x..o.x....o......o.......o.......o...xxoo...xx w",
            "x.....oxx...o...............o.......o...x.oo..xxx w",
            "x.....o....................x.oo...x.oox...o.ox.xx w",
            "o.....o.o....o.........xx....oxx....o...xxo....xx w",
            "xx....oxx...o.oo.....oo.........x...o...x.o.....x w",
            "x...ooo.x...oo...................x..o..xxxoo.x.xx w",
            "x....oo.x....o......o.......x.....xoo...x.oo...xx b",
            "x.....o.x...oo....................x.oooxx.o..ox.x b",
        ];
        let boards: Vec<Board> = book.iter().filter_map(|l| Board::from_str(l)).collect();
        assert!(boards.len() >= 20, "test positions do not parse");
        let (mut with, mut without) = (0u64, 0u64);
        for b in &boards {
            for use_pol in [true, false] {
                let mut s = Searcher::new_unconfigured();
                s.set_nnue(Some(nn));
                s.set_policy(if use_pol { crate::policy::bundled() } else { None }, 100000.0, 1);
                let r = s.search(b, &SearchLimits { time: None, max_nodes: None, max_depth: 8 });
                if use_pol {
                    with += r.nodes
                } else {
                    without += r.nodes
                }
            }
        }
        assert!(with > 0 && with < without, "policy head did not reduce nodes: with {with} without {without}");
    }
}
