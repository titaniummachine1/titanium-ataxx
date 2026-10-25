//! C ABI for the Titanium engine, Botzone rules. The engine knows NOTHING about the Botzone
//! protocol: the Python driver parses requests, owns the clock, and calls
//!   ti_new -> ti_new_game -> { ti_apply(opponent move) ; ti_search(budget_ms) } *
//! Everything here is single-threaded except `ti_stop`, which only flips an atomic flag and may be
//! called from any thread while `ti_search` runs.
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use titanium::board::{Board, Move};
use titanium::search::{SearchLimits, Searcher, StopReason};
use titanium::nnue::Nnue;
use titanium::tuple::TupleTable;

pub const ABI_VERSION: u32 = 1;
const DEFAULT_TT_BITS: u32 = 22;

pub struct TiEngine {
    board: Board,
    /// Placement snapshots, exactly as the Botzone judge keeps them: index 0 = the start position,
    /// then one entry per move = occupancy right after the piece is placed (before captures).
    snaps: Vec<(u64, u64)>,
    searcher: Searcher<'static>,
    stop: Arc<AtomicBool>,
    stats: TiStats,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TiStats {
    pub nodes: u64,
    pub elapsed_us: u64,
    pub score: i32,
    pub depth: u32,
    /// 0 completed, 1 time, 2 nodes, 3 other
    pub stopped_by: u32,
    /// 1 if the repetition guard replaced the engine's move
    pub repetition_override: u32,
}

#[no_mangle]
pub extern "C" fn ti_abi_version() -> u32 {
    ABI_VERSION
}

/// Occupancy right after `m` is placed (origin cleared for a jump), BEFORE captures: the judge's snapshot.
fn placement(b: &Board, m: Move) -> (u64, u64) {
    let me = b.turn as usize;
    let mut occ = b.occ;
    if !m.is_clone() {
        occ[me] &= !(1u64 << m.from);
    }
    occ[me] |= 1u64 << m.to;
    (occ[0], occ[1])
}

impl TiEngine {
    /// The judge's repetition rule: a move is illegal iff the latest snapshot equals an earlier one
    /// (index i <= k-2) and this move produces the same successor snapshot as the move played then.
    fn banned(&self, m: Move) -> bool {
        let k = self.snaps.len() - 1;
        if k < 3 {
            return false;
        }
        let c = placement(&self.board, m);
        let last = self.snaps[k];
        (0..=k - 2).any(|i| self.snaps[i] == last && self.snaps[i + 1] == c)
    }

    fn push_move(&mut self, m: Move) {
        self.snaps.push(placement(&self.board, m));
        self.board = self.board.make(m);
    }
}

/// Create an engine. `net_path` = path to the big NNUE (`*.tnn`, e.g. nnue_main.tnn: main's default eval) or to a TUP
/// tuple net (S57 fallback); NULL = classical hand eval (weak!).
/// Returns NULL if a net path is given but cannot be loaded (never silently falls back).
/// `tt_bits` = log2 of hash entries (0 -> 22, ~100 MB); each entry is 24 bytes.
#[no_mangle]
pub extern "C" fn ti_new(net_path: *const c_char, tt_bits: u32) -> *mut TiEngine {
    let bits = if tt_bits == 0 { DEFAULT_TT_BITS } else { tt_bits.clamp(12, 26) } as usize;
    // the NNUE is loaded BEFORE the hash table is allocated, so the file buffer and the table never coexist (peak RAM)
    let mut nnue: Option<&'static Nnue> = None;
    if !net_path.is_null() {
        let p = unsafe { CStr::from_ptr(net_path) }.to_string_lossy().into_owned();
        if p.ends_with(".tnn") {
            match Nnue::load(&p) {
                Ok(mut n) => {
                    n.drop_f32_reference();
                    nnue = Some(Box::leak(Box::new(n)));
                }
                Err(_) => return std::ptr::null_mut(),
            }
        }
    }
    let mut searcher: Searcher<'static> = Searcher::with_tt_bits(bits);
    searcher.set_nnue(nnue);
    if !net_path.is_null() && nnue.is_none() {
        let p = unsafe { CStr::from_ptr(net_path) }.to_string_lossy().into_owned();
        match TupleTable::load(&p) {
            Ok(t) => {
                let t: &'static TupleTable = Box::leak(Box::new(t));
                searcher.set_tuple(Some(t));
            }
            Err(_) => return std::ptr::null_mut(),
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    searcher.set_stop_flag(Some(stop.clone()));
    let board = Board::botzone_start();
    Box::into_raw(Box::new(TiEngine { snaps: vec![(board.occ[0], board.occ[1])], board, searcher, stop, stats: TiStats::default() }))
}

#[no_mangle]
pub extern "C" fn ti_free(e: *mut TiEngine) {
    if !e.is_null() {
        unsafe { drop(Box::from_raw(e)) };
    }
}

/// Reset to the Botzone 4-corner start (black a7+g1, white g7+a1, black to move); clears hash
/// table, killers, history and the repetition list.
#[no_mangle]
pub extern "C" fn ti_new_game(e: *mut TiEngine) {
    if e.is_null() {
        return;
    }
    let e = unsafe { &mut *e };
    e.board = Board::botzone_start();
    e.snaps.clear();
    e.snaps.push((e.board.occ[0], e.board.occ[1]));
    e.searcher.reset_for_new_game();
    e.stop.store(false, Ordering::Relaxed);
}

/// Apply a move given in Botzone coordinates (x = file 0..6, y = row 0..6, sq = y*7 + x).
/// Returns 1 = applied, 0 = ignored (the "-1 -1 -1 -1" start marker), -1 = illegal / unparsable
/// (the board is NOT modified, so the caller can resync by replaying the game).
#[no_mangle]
pub extern "C" fn ti_apply(e: *mut TiEngine, x0: i32, y0: i32, x1: i32, y1: i32) -> i32 {
    if e.is_null() {
        return -1;
    }
    let e = unsafe { &mut *e };
    if x0 < 0 && y0 < 0 && x1 < 0 && y1 < 0 {
        return 0;
    }
    if !((0..=6).contains(&x0) && (0..=6).contains(&y0) && (0..=6).contains(&x1) && (0..=6).contains(&y1)) {
        return -1;
    }
    let m = Move { from: (y0 * 7 + x0) as u8, to: (y1 * 7 + x1) as u8 };
    if !e.board.legal_moves().contains(&m) {
        return -1;
    }
    e.push_move(m);
    1
}

/// 1 if the judge would reject this move as a repetition, 0 if it is allowed, -1 if it is not even legal.
#[no_mangle]
pub extern "C" fn ti_move_banned(e: *mut TiEngine, x0: i32, y0: i32, x1: i32, y1: i32) -> i32 {
    if e.is_null() || !((0..=6).contains(&x0) && (0..=6).contains(&y0) && (0..=6).contains(&x1) && (0..=6).contains(&y1)) {
        return -1;
    }
    let e = unsafe { &*e };
    let m = Move { from: (y0 * 7 + x0) as u8, to: (y1 * 7 + x1) as u8 };
    if !e.board.legal_moves().contains(&m) {
        return -1;
    }
    e.banned(m) as i32
}

/// Number of legal moves for the side to move (0 = game over / no move).
#[no_mangle]
pub extern "C" fn ti_legal_count(e: *mut TiEngine) -> i32 {
    if e.is_null() {
        return -1;
    }
    unsafe { &*e }.board.legal_moves().len() as i32
}

/// Ask a running `ti_search` to stop at its next poll (<= 2048 nodes, ~2 ms). Safe from any thread.
#[no_mangle]
pub extern "C" fn ti_stop(e: *mut TiEngine) {
    if !e.is_null() {
        unsafe { &*e }.stop.store(true, Ordering::Relaxed);
    }
}

/// Search the current position for at most `budget_ms` (wall clock, absolute) and write the move
/// to out[0..4] = x0,y0,x1,y1. Plays the move on the internal board.
/// Returns 1 = move written, 0 = no legal move (out = -1s), -1 = bad arguments.
#[no_mangle]
pub extern "C" fn ti_search(e: *mut TiEngine, budget_ms: u64, out: *mut i32) -> i32 {
    if e.is_null() || out.is_null() {
        return -1;
    }
    let e = unsafe { &mut *e };
    let list = e.board.legal_moves();
    if list.is_empty() || e.board.bz_over() {
        unsafe {
            for i in 0..4 {
                *out.add(i) = -1;
            }
        }
        return 0;
    }
    e.stop.store(false, Ordering::Relaxed);
    let limits = SearchLimits { time: Some(Duration::from_millis(budget_ms.max(1))), max_nodes: None, max_depth: u32::MAX };
    let r = e.searcher.search_bz(&e.board, &limits);
    let mut mv = r.best.unwrap_or_else(|| list.move_at(0));
    let mut overridden = 0u32;
    // Judge rule: a move that repeats an earlier transition is rejected as illegal input. If the
    // engine's move is such a move, play the allowed move that wins the most discs instead.
    if e.banned(mv) {
        let mut best: Option<(i32, Move)> = None;
        for i in 0..list.len() {
            let m = list.move_at(i);
            if e.banned(m) {
                continue;
            }
            let c = e.board.make(m);
            let me = e.board.turn as usize;
            let gain = c.piece_cnt[me] as i32 - c.piece_cnt[1 - me] as i32;
            if best.map_or(true, |(g, _)| gain > g) {
                best = Some((gain, m));
            }
        }
        if let Some((_, m)) = best {
            mv = m;
            overridden = 1;
        }
    }
    unsafe {
        *out.add(0) = (mv.from % 7) as i32;
        *out.add(1) = (mv.from / 7) as i32;
        *out.add(2) = (mv.to % 7) as i32;
        *out.add(3) = (mv.to / 7) as i32;
    }
    e.push_move(mv);
    e.stats = TiStats {
        nodes: r.nodes,
        elapsed_us: r.elapsed.as_micros() as u64,
        score: r.score,
        depth: r.depth,
        stopped_by: match r.stopped_by {
            Some(StopReason::Completed) | None => 0,
            Some(StopReason::Time) => 1,
            Some(StopReason::Nodes) => 2,
            #[allow(unreachable_patterns)]
            _ => 3,
        },
        repetition_override: overridden,
    };
    1
}

/// Copy the stats of the last `ti_search` into `out`.
#[no_mangle]
pub extern "C" fn ti_stats(e: *mut TiEngine, out: *mut TiStats) {
    if e.is_null() || out.is_null() {
        return;
    }
    unsafe { *out = (*e).stats };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn play_one(e: *mut TiEngine, budget: u64) -> [i32; 4] {
        let mut out = [0i32; 4];
        assert_eq!(ti_search(e, budget, out.as_mut_ptr()), 1);
        out
    }

    #[test]
    fn abi_and_bad_net_path() {
        assert_eq!(ti_abi_version(), 1);
        let bad = CString::new("definitely/not/a/net.tup").unwrap();
        assert!(ti_new(bad.as_ptr(), 14).is_null(), "a missing net must not silently fall back");
    }

    #[test]
    fn selfplay_is_legal_and_validated() {
        let a = ti_new(std::ptr::null(), 16);
        let b = ti_new(std::ptr::null(), 16);
        ti_new_game(a);
        ti_new_game(b);
        assert_eq!(ti_legal_count(a), 16);
        assert_eq!(ti_apply(a, -1, -1, -1, -1), 0, "start marker ignored");
        for ply in 0..60 {
            let (me, other) = if ply % 2 == 0 { (a, b) } else { (b, a) };
            if ti_legal_count(me) == 0 {
                break;
            }
            let m = play_one(me, 5);
            assert_eq!(ti_apply(other, m[0], m[1], m[2], m[3]), 1, "move {m:?} rejected at ply {ply}");
        }
        // an illegal move is refused and leaves the board untouched
        let n = ti_legal_count(a);
        assert_eq!(ti_apply(a, 0, 0, 6, 6), -1);
        assert_eq!(ti_legal_count(a), n);
        ti_free(a);
        ti_free(b);
    }

    #[test]
    fn repetition_rule_matches_the_judge_example() {
        let e = ti_new(std::ptr::null(), 14);
        ti_new_game(e);
        // black jumps out and back, white too: the start position recurs after 4 plies
        for m in [(0, 0, 2, 0), (6, 0, 4, 0), (2, 0, 0, 0), (4, 0, 6, 0)] {
            assert_eq!(ti_apply(e, m.0, m.1, m.2, m.3), 1);
        }
        // black repeating its very first move now repeats the earlier transition -> banned
        assert_eq!(ti_move_banned(e, 0, 0, 2, 0), 1);
        // any other legal move is fine
        assert_eq!(ti_move_banned(e, 0, 0, 1, 1), 0);
        assert_eq!(ti_move_banned(e, 6, 6, 5, 5), 0);
        assert_eq!(ti_move_banned(e, 3, 3, 3, 4), -1, "not a legal move");
        // the search never plays a banned move
        let mut out = [0i32; 4];
        for _ in 0..1 {
            assert_eq!(ti_search(e, 20, out.as_mut_ptr()), 1);
            assert!(!(out == [0, 0, 2, 0]), "search played the banned repetition");
        }
        ti_free(e);
    }

    #[test]
    fn stop_flag_returns_fast_with_a_move() {
        let e = ti_new(std::ptr::null(), 16);
        ti_new_game(e);
        let t = std::time::Instant::now();
        let mut out = [0i32; 4];
        // budget is huge, but the stop flag is raised from another thread
        let ptr = e as usize;
        let h = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            ti_stop(ptr as *mut TiEngine);
        });
        assert_eq!(ti_search(e, 10_000, out.as_mut_ptr()), 1);
        h.join().unwrap();
        assert!(t.elapsed().as_millis() < 500, "stop took {} ms", t.elapsed().as_millis());
        ti_free(e);
    }
}
