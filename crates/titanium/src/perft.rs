//! D1394 (user: "finish movegen"): libataxx-convention perft (a clone counts once per destination square, passes
//! and wipe-outs handled like libataxx / faeries) with bulk counting at depth 1, plus a single-threaded hashed
//! perft (transposition table), the measure bryanQeaton/Ataxx-movegen and kz04px/faeries report.
//!
//! The position is kept side-to-move relative as (me, opp, empty): a perft count depends only on that triple,
//! so the table key is exact (both bitboards are stored, no hash collisions can corrupt a count).

use crate::board::{reach_union, FULL, RING1, RING2};

#[inline(always)]
fn clone_targets(me: u64, empty: u64) -> u64 {
    crate::board::box1(me) & empty
}

/// Legal move count in the libataxx convention (pass = 1 when the mover is stuck and the opponent is not).
#[inline(always)]
pub fn count_moves(me: u64, opp: u64, empty: u64) -> u64 {
    if me == 0 || opp == 0 {
        return 0;
    }
    let mut n = clone_targets(me, empty).count_ones() as u64;
    let mut src = me;
    while src != 0 {
        let s = src.trailing_zeros() as usize;
        src &= src - 1;
        n += (RING2[s] & empty).count_ones() as u64;
    }
    if n == 0 && reach_union(opp) & empty != 0 {
        1
    } else {
        n
    }
}

/// Plain recursive perft, libataxx counts (start position: 16, 256, 6460, 155888, 4752668, 141865520, ...).
pub fn perft_lib(me: u64, opp: u64, empty: u64, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if depth == 1 {
        return count_moves(me, opp, empty);
    }
    if me == 0 || opp == 0 {
        return 0;
    }
    let mut nodes = 0u64;
    let mut moved = false;
    let mut cl = clone_targets(me, empty);
    while cl != 0 {
        let to = cl.trailing_zeros() as usize;
        cl &= cl - 1;
        let tb = 1u64 << to;
        let caps = RING1[to] & opp;
        nodes += perft_lib(opp ^ caps, me | tb | caps, empty ^ tb, depth - 1);
        moved = true;
    }
    let mut src = me;
    while src != 0 {
        let s = src.trailing_zeros() as usize;
        src &= src - 1;
        let sb = 1u64 << s;
        let mut js = RING2[s] & empty;
        while js != 0 {
            let to = js.trailing_zeros() as usize;
            js &= js - 1;
            let tb = 1u64 << to;
            let caps = RING1[to] & opp;
            nodes += perft_lib(opp ^ caps, (me ^ sb) | tb | caps, (empty ^ tb) | sb, depth - 1);
            moved = true;
        }
    }
    if !moved && reach_union(opp) & empty != 0 {
        nodes += perft_lib(opp, me, empty, depth - 1);
    }
    nodes
}

/// One table slot: the exact position, the subtree depth and its count (32 B, two per 64 B cache line).
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct Slot {
    me: u64,
    opp: u64,
    nodes: u64,
    depth: u32,
    _pad: u32,
}

/// Hashed-perft table: 2-slot buckets on one cache line, depth-preferred replacement in the second slot.
pub struct PerftTable {
    slots: Vec<Slot>,
    mask: usize,
}

impl PerftTable {
    /// `mb` megabytes rounded down to a power of two buckets.
    pub fn new(mb: usize) -> Self {
        let bytes = mb.max(1) << 20;
        let mut buckets = 1usize;
        while buckets * 2 * 64 <= bytes {
            buckets *= 2;
        }
        PerftTable { slots: vec![Slot::default(); buckets * 2], mask: buckets - 1 }
    }
    #[inline(always)]
    fn bucket(&self, me: u64, opp: u64) -> usize {
        let h = (me.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ opp.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)).rotate_left(29)
            .wrapping_mul(0x1656_67B1_9E37_79F9);
        ((h >> 32) as usize & self.mask) * 2
    }
    #[inline(always)]
    fn prefetch(&self, me: u64, opp: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let p = self.slots.as_ptr().add(self.bucket(me, opp)) as *const i8;
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p);
        }
    }
    #[inline(always)]
    fn get(&self, me: u64, opp: u64, depth: u32) -> Option<u64> {
        let b = self.bucket(me, opp);
        for s in &self.slots[b..b + 2] {
            if s.depth == depth && s.me == me && s.opp == opp {
                return Some(s.nodes);
            }
        }
        None
    }
    #[inline(always)]
    fn put(&mut self, me: u64, opp: u64, depth: u32, nodes: u64) {
        let b = self.bucket(me, opp);
        // slot 0 = always replace (newest), slot 1 = keep the deeper subtree
        let deep = &self.slots[b + 1];
        let i = if depth >= deep.depth { b + 1 } else { b };
        self.slots[i] = Slot { me, opp, nodes, depth, _pad: 0 };
    }
}

/// Hashed perft, single thread: identical counts to `perft_lib`, subtrees met again by transposition are read back.
pub fn perft_tt(me: u64, opp: u64, empty: u64, depth: u32, tt: &mut PerftTable) -> u64 {
    if depth <= 1 {
        return if depth == 0 { 1 } else { count_moves(me, opp, empty) };
    }
    if me == 0 || opp == 0 {
        return 0;
    }
    if let Some(n) = tt.get(me, opp, depth) {
        return n;
    }
    let mut nodes = 0u64;
    let mut moved = false;
    let mut cl = clone_targets(me, empty);
    while cl != 0 {
        let to = cl.trailing_zeros() as usize;
        cl &= cl - 1;
        let tb = 1u64 << to;
        let caps = RING1[to] & opp;
        let (cm, co) = (opp ^ caps, me | tb | caps);
        if depth > 2 {
            tt.prefetch(cm, co);
        }
        nodes += perft_tt(cm, co, empty ^ tb, depth - 1, tt);
        moved = true;
    }
    let mut src = me;
    while src != 0 {
        let s = src.trailing_zeros() as usize;
        src &= src - 1;
        let sb = 1u64 << s;
        let mut js = RING2[s] & empty;
        while js != 0 {
            let to = js.trailing_zeros() as usize;
            js &= js - 1;
            let tb = 1u64 << to;
            let caps = RING1[to] & opp;
            let (cm, co) = (opp ^ caps, (me ^ sb) | tb | caps);
            if depth > 2 {
                tt.prefetch(cm, co);
            }
            nodes += perft_tt(cm, co, (empty ^ tb) | sb, depth - 1, tt);
            moved = true;
        }
    }
    if !moved && reach_union(opp) & empty != 0 {
        nodes += perft_tt(opp, me, empty, depth - 1, tt);
    }
    tt.put(me, opp, depth, nodes);
    nodes
}

/// (me, opp, empty) of a board from the side to move's view.
pub fn rel(b: &crate::board::Board) -> (u64, u64, u64) {
    let t = b.turn as usize;
    (b.occ[t], b.occ[1 - t], !(b.occ[0] | b.occ[1] | b.blockers) & FULL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    /// libataxx / Autaxx start-position counts (D673).
    const START: [u64; 7] = [1, 16, 256, 6460, 155888, 4752668, 141865520];

    #[test]
    fn perft_lib_matches_libataxx_start_counts() {
        let (me, opp, e) = rel(&Board::start());
        for (d, &want) in START.iter().enumerate() {
            assert_eq!(perft_lib(me, opp, e, d as u32), want, "depth {d}");
        }
    }

    #[test]
    fn perft_tt_equals_plain_perft() {
        let (me, opp, e) = rel(&Board::start());
        let mut tt = PerftTable::new(16);
        for d in 0..=6u32 {
            assert_eq!(perft_tt(me, opp, e, d, &mut tt), START[d as usize], "depth {d}");
        }
        // other positions, incl. blockers and lopsided material
        let mut b = Board::start();
        for i in 0..12 {
            let ml = b.legal_moves();
            if ml.len() == 0 {
                break;
            }
            b = b.make(ml.iter().nth(i * 7 % ml.len()).unwrap());
            let (me, opp, e) = rel(&b);
            let mut tt = PerftTable::new(4);
            for d in 1..=4u32 {
                assert_eq!(perft_tt(me, opp, e, d, &mut tt), perft_lib(me, opp, e, d), "ply {i} depth {d}");
            }
        }
    }

    #[test]
    fn count_moves_matches_engine_dedup_movegen() {
        let mut b = Board::start();
        for i in 0..40 {
            let ml = b.legal_moves();
            if ml.len() == 0 {
                break;
            }
            let mut list = crate::board::MoveList::new();
            b.legal_moves_dedup_into(&mut list);
            let (me, opp, e) = rel(&b);
            if me != 0 && opp != 0 && list.len() > 0 && !list.move_at(0).is_pass() {
                assert_eq!(count_moves(me, opp, e), list.len() as u64, "ply {i}");
            }
            b = b.make(ml.iter().nth(i * 5 % ml.len()).unwrap());
        }
    }
}
