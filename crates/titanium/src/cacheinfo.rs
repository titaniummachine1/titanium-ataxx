//! Runtime CPU cache size detection (D646, user: the adaptive table steps must follow the TARGET device's caches, not the
//! build machine's). The Botzone `.so` is built here but runs on the judge's CPU, so this is read at engine start, never
//! at compile time. x86_64: CPUID leaf 4 (Intel, deterministic cache parameters) or 0x8000001D (AMD, same layout).
//! Anything else, or an unreadable CPUID, falls back to typical desktop values.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheSizes {
    /// L1 data cache per core, bytes
    pub l1d: usize,
    /// L2 per core (or per core pair), bytes
    pub l2: usize,
    /// last-level cache (whole, shared), bytes
    pub l3: usize,
    /// false = the fallback constants are in use
    pub detected: bool,
}

pub const FALLBACK: CacheSizes = CacheSizes { l1d: 32 << 10, l2: 256 << 10, l3: 8 << 20, detected: false };

static CACHE: OnceLock<CacheSizes> = OnceLock::new();

/// Detected cache sizes (computed once).
pub fn sizes() -> CacheSizes {
    *CACHE.get_or_init(detect)
}

#[cfg(target_arch = "x86_64")]
fn detect() -> CacheSizes {
    use std::arch::x86_64::{__cpuid, __cpuid_count};
    // SAFETY: CPUID is available on every x86_64 CPU (the intrinsics are safe fns on newer toolchains: unused_unsafe)
    #[allow(unused_unsafe)]
    unsafe {
        let max_basic = __cpuid(0).eax;
        let max_ext = __cpuid(0x8000_0000).eax;
        let v = __cpuid(0);
        let amd = v.ebx == 0x6874_7541 && v.edx == 0x6974_6E65 && v.ecx == 0x444D_4163; // "AuthenticAMD"
        let leaf = if amd && max_ext >= 0x8000_001D {
            0x8000_001D
        } else if max_basic >= 4 {
            4
        } else {
            return FALLBACK;
        };
        let (mut l1d, mut l2, mut l3) = (0usize, 0usize, 0usize);
        for sub in 0..16u32 {
            let r = __cpuid_count(leaf, sub);
            let ty = r.eax & 0x1f; // 0 = no more, 1 data, 2 instruction, 3 unified
            if ty == 0 {
                break;
            }
            let level = (r.eax >> 5) & 7;
            let ways = ((r.ebx >> 22) & 0x3ff) as usize + 1;
            let parts = ((r.ebx >> 12) & 0x3ff) as usize + 1;
            let line = (r.ebx & 0xfff) as usize + 1;
            let sets = r.ecx as usize + 1;
            let size = ways * parts * line * sets;
            match (level, ty) {
                (1, 1) => l1d = size,
                (2, 1) | (2, 3) => l2 = size,
                (3, 1) | (3, 3) => l3 = size,
                _ => {}
            }
        }
        if l1d == 0 || l2 == 0 {
            return FALLBACK;
        }
        // no L3 reported (some cores): treat L2 x 8 as the last level
        CacheSizes { l1d, l2, l3: if l3 == 0 { l2 * 8 } else { l3 }, detected: true }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect() -> CacheSizes {
    FALLBACK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detected_sizes_are_sane() {
        let c = sizes();
        assert!(c.l1d >= 8 << 10 && c.l1d <= 1 << 20, "l1d {}", c.l1d);
        assert!(c.l2 >= c.l1d && c.l2 <= 64 << 20, "l2 {}", c.l2);
        assert!(c.l3 >= c.l2, "l3 {}", c.l3);
    }
}
