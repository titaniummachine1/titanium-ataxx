//! Evaluation and iterative-deepening alpha-beta (negamax) search.
//!
//! Search order: TT probe -> eval cache (depth 0) -> move scoring
//! (TT move > killers > captures*W + history) -> ordering strategy
//! (lazy selection / insertion sort / LSD radix) -> alpha-beta.

use crate::board::{dist_union, jump_union, Board, Move, MoveList, RING1, SQUARES};
use std::time::Duration;

pub const MATE: i32 = 100_000;
/// Scores at least this close to MATE are mate scores (TT adjustment bound).
const MATE_BOUND: i32 = MATE - 2000;

/// Search stack depth cap (max_depth is clamped well below this).
const MAX_PLY: usize = 40;

/// Per-ply move buffers allocated once per search: nodes only reset `len`,
/// they never re-initialize the arrays. `scratch` backs the radix sort.
struct MoveStack {
    lists: Box<[MoveList; MAX_PLY]>,
    scratch: Box<[MoveList; MAX_PLY]>,
}

impl MoveStack {
    fn new() -> MoveStack {
        MoveStack {
            lists: Box::new([MoveList::new(); MAX_PLY]),
            scratch: Box::new([MoveList::new(); MAX_PLY]),
        }
    }
}

/// Material weight per piece (eval = piece diff * MATERIAL).
const MATERIAL: i32 = 100;
/// Tempo: flat bonus for the side to move. autaxx uses 1.5-2 stones worth.
const TEMPO: i32 = 150;
/// Square PST (autaxx quiet-ordering table): corners/edges good, center bad.
/// Rows top->bottom, symmetric.
const PST: [i32; 49] = [
    30, 20, 10, 10, 10, 20, 30, //
    20, 10, 10, 5, 10, 10, 20, //
    10, 10, 5, 0, 5, 10, 10, //
    10, 5, 0, 0, 0, 5, 10, //
    10, 10, 5, 0, 5, 10, 10, //
    20, 10, 10, 5, 10, 10, 20, //
    30, 20, 10, 10, 10, 20, 30,
];
/// Hole-risk penalties (autaxx): for every EMPTY square adjacent to our
/// stones and reachable by the enemy, penalize by own-stone density around
/// it. Static replacement for qsearch in Ataxx.
const HOLE_PEN: [i32; 9] = [-17, 4, -28, -88, -125, -200, -322, -446, -534];

/// Contact penalty per ENDANGERED stone (E2, "virus strategy"): stones the
/// enemy can infect right now. Avoid contact early, mass-clone first.
const CONTACT_PEN: i32 = 15;
/// Multi-capture exposure (E4, user idea): for every empty landing square
/// the enemy can reach, each of our stones BEYOND THE FIRST that one enemy
/// landing would convert costs this much — a 3-stone cluster next to an
/// enemy-reachable hole is a pending 2-stone loss in a single move.
const MULTICAP_PEN: i32 = 20;

/// Reverse futility pruning margins, index = depth-1 (autaxx, stone = 100).
const RFP_MARGINS: [i32; 4] = [257, 347, 478, 774];

/// E5 proven-win certificate score: above every heuristic eval (±~3000),
/// below MATE_BOUND so TT mate adjustments never touch it.
const CERT_WIN: i32 = 90_000;

// Ordering tiers (titanium-engine pattern: compile-asserted bands so no
// ordering signal can structurally leak into its neighbor).
const ORD_TT: i32 = i32::MAX / 2;
const ORD_KILLER: i32 = 100_000;
const ORD_CAPTURE_UNIT: i32 = 1000; // per conversion, max 8
const ORD_CLONE_BONUS: i32 = 500;
const ORD_JUMP_PEN: i32 = 500;
const ORD_HISTORY_MAX: i32 = 90_000;
const _: () = assert!(ORD_HISTORY_MAX + 8 * ORD_CAPTURE_UNIT + ORD_CLONE_BONUS < ORD_KILLER);
const _: () = assert!(ORD_KILLER < ORD_TT);

/// Static eval from the side-to-move's perspective.
pub fn evaluate(b: &Board) -> i32 {
    let mat = MATERIAL * (b.piece_cnt[0] as i32 - b.piece_cnt[1] as i32);
    let mut pst = 0i32;
    for c in 0..2 {
        let sign = if c == 0 { 1 } else { -1 };
        let mut bb = b.occ[c];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            pst += sign * PST[sq];
        }
    }
    let holes_black = holes_penalty(b, 0);
    let holes_white = holes_penalty(b, 1);
    let contact_black = endangered_count(b, 0);
    let contact_white = endangered_count(b, 1);
    let multicap_black = multicapture_penalty(b, 0);
    let multicap_white = multicapture_penalty(b, 1);
    let mut score = mat + pst - holes_black + holes_white
        - CONTACT_PEN * contact_black as i32
        + CONTACT_PEN * contact_white as i32
        - multicap_black
        + multicap_white;
    if b.turn == 0 {
        score += TEMPO;
    } else {
        score -= TEMPO;
    }
    if b.turn == 0 {
        score
    } else {
        -score
    }
}

/// E2 "virus strategy": stones of `side` the enemy can currently convert —
/// empty landing squares within their reach, ring-1 of a landing touches us.
fn endangered_count(b: &Board, side: u8) -> u32 {
    let them = b.occ[1 - side as usize];
    let us = b.occ[side as usize];
    let landings = b.empty() & (dist_union(them, 1) | jump_union(them));
    (us & dist_union(landings, 1)).count_ones()
}

/// E4 multi-capture exposure: for each enemy-reachable empty landing square,
/// our stones there beyond the first (each extra one = a stone lost in the
/// same single enemy move).
fn multicapture_penalty(b: &Board, side: u8) -> i32 {
    let them = b.occ[1 - side as usize];
    let us = b.occ[side as usize];
    let landings = b.empty() & (dist_union(them, 1) | jump_union(them));
    let mut pen = 0i32;
    let mut ls = landings;
    while ls != 0 {
        let sq = ls.trailing_zeros() as usize;
        ls &= ls - 1;
        let n = (RING1[sq] & us).count_ones() as i32;
        if n >= 2 {
            pen += (n - 1) * MULTICAP_PEN;
        }
    }
    pen
}

/// autaxx hole risk: empty squares adjacent to `side`'s stones that the
/// enemy can land on, weighted by own-stone density around them.
fn holes_penalty(b: &Board, side: u8) -> i32 {
    let us = b.occ[side as usize];
    let them = b.occ[1 - side as usize];
    let holes = b.empty() & dist_union(us, 1) & (dist_union(them, 1) | jump_union(them));
    let mut pen = 0i32;
    let mut hs = holes;
    while hs != 0 {
        let sq = hs.trailing_zeros() as usize;
        hs &= hs - 1;
        let n = (RING1[sq] & us).count_ones() as usize;
        pen += HOLE_PEN[n];
    }
    pen
}

/// Terminal score from the side-to-move's perspective.
fn terminal_score(b: &Board, ply: u32) -> i32 {
    let diff = b.piece_cnt[b.turn as usize] as i32 - b.piece_cnt[1 - b.turn as usize] as i32;
    if diff > 0 {
        MATE - ply as i32
    } else if diff < 0 {
        -MATE + ply as i32
    } else {
        0
    }
}

/// Move ordering strategy. All variants share the same move scoring
/// (TT move > killers > captures*W + history); they differ in HOW the
/// scored list is consumed. Switchable so sperft can A/B them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// Repeated argmax extraction — O(n) per pick, only pays for the moves
    /// actually examined before a cutoff. Zero extra memory.
    Lazy,
    /// Full insertion sort up front — better locality for PV nodes where
    /// every move is searched anyway ("nearly sorted" friendly).
    Insertion,
    /// LSD radix sort (least significant digit first, 3 x 8-bit passes,
    /// descending-stable) — O(n) regardless of presortedness.
    Radix,
}

impl Default for Ordering {
    fn default() -> Self {
        Ordering::Lazy
    }
}

/// Transposition table entry (24 bytes with padding).
#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    mv: u32, // packed Move; 0xFFFF = pass; 0 = none (eval cache entries)
    score: i32,
    depth: u8,
    flag: u8, // 0 empty, 1 exact, 2 lower, 3 upper
}

const FLAG_EMPTY: u8 = 0;
const FLAG_EXACT: u8 = 1;
const FLAG_LOWER: u8 = 2;
const FLAG_UPPER: u8 = 3;

/// Fixed-size transposition table, power-of-two entry count. Depth-0 EXACT
/// entries double as the eval cache (Ataxx transpositions repeat a lot).
pub struct Tt {
    buf: Vec<TtEntry>,
    mask: usize,
}

impl Tt {
    /// `2^bits` entries (bits=20 -> 1M entries, 24 MB).
    pub fn new(bits: usize) -> Tt {
        let size = 1usize << bits.min(24);
        Tt {
            buf: vec![
                TtEntry {
                    key: 0,
                    mv: 0,
                    score: 0,
                    depth: 0,
                    flag: FLAG_EMPTY
                };
                size
            ],
            mask: size - 1,
        }
    }

    pub fn clear(&mut self) {
        for e in self.buf.iter_mut() {
            e.flag = FLAG_EMPTY;
        }
    }

    #[inline]
    fn probe(&self, key: u64) -> Option<&TtEntry> {
        let e = &self.buf[(key as usize) & self.mask];
        if e.flag != FLAG_EMPTY && e.key == key {
            Some(e)
        } else {
            None
        }
    }

    #[inline]
    fn store(&mut self, key: u64, mv: u32, score: i32, depth: u8, flag: u8) {
        let e = &mut self.buf[(key as usize) & self.mask];
        // Depth-preferred replacement; always replace same position so a
        // fresh search overwrites stale bounds.
        if e.flag == FLAG_EMPTY || e.depth <= depth || e.key == key {
            *e = TtEntry {
                key,
                mv,
                score,
                depth,
                flag,
            };
        }
    }
}

/// Mate-score adjustment: store distance-from-node so entries stay correct
/// at any retrieval ply.
#[inline]
fn score_to_tt(score: i32, ply: u32) -> i32 {
    if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

#[inline]
fn score_from_tt(score: i32, ply: u32) -> i32 {
    if score >= MATE_BOUND {
        score - ply as i32
    } else if score <= -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

/// Why a search stopped. Whichever limit trips first wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Ran out of depths to deepen.
    Completed,
    /// Soft time budget exhausted.
    Time,
    /// Node budget exhausted.
    Nodes,
}

#[derive(Clone, Debug)]
pub struct SearchLimits {
    /// Soft time budget per move; `None` = unlimited.
    pub time: Option<Duration>,
    /// Hard node budget; search stops once `nodes` reaches this.
    pub max_nodes: Option<u64>,
    /// Hard depth cap.
    pub max_depth: u32,
}

impl Default for SearchLimits {
    fn default() -> Self {
        SearchLimits {
            time: Some(Duration::from_millis(1000)),
            max_nodes: None,
            max_depth: 20,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub best: Option<Move>,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub stopped_by: Option<StopReason>,
}

pub struct Searcher {
    pub nodes: u64,
    /// Absolute deadline in milliseconds (via `time_src`), if time-limited.
    deadline_ms: Option<f64>,
    start_ms: f64,
    stop: bool,
    stop_reason: StopReason,
    node_limit: Option<u64>,
    /// Monotonic milliseconds. Native builds use `Instant`; wasm builds
    /// inject `performance.now()` (std time does not exist on wasm32).
    time_src: fn() -> f64,
    /// Transposition table (persists across searches in the same Searcher).
    tt: Tt,
    /// Move ordering strategy (benchmarkable switch).
    ordering: Ordering,
    /// Two killer moves per ply (packed u32 moves), refreshed on cutoffs.
    killers: Box<[(u32, u32); MAX_PLY]>,
    /// History heuristic: [mover][to] cutoff counters, aged per search.
    history: Box<[[u32; 49]; 2]>,
}

/// Monotonic milliseconds since process start (native default time source).
fn native_now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static BOOT: OnceLock<Instant> = OnceLock::new();
    BOOT.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new()
    }
}
impl Searcher {
    pub fn new() -> Searcher {
        Searcher::with_tt_bits(20)
    }

    /// Searcher with a `2^bits`-entry transposition table (wasm builds use
    /// smaller tables).
    pub fn with_tt_bits(bits: usize) -> Searcher {
        Searcher {
            nodes: 0,
            deadline_ms: None,
            start_ms: 0.0,
            stop: false,
            stop_reason: StopReason::Completed,
            node_limit: None,
            time_src: native_now_ms,
            tt: Tt::new(bits),
            ordering: Ordering::Lazy,
            killers: Box::new([(0, 0); MAX_PLY]),
            history: Box::new([[0; 49]; 2]),
        }
    }

    /// Select the move ordering strategy (for sperft A/B benchmarks).
    pub fn set_ordering(&mut self, ordering: Ordering) {
        self.ordering = ordering;
    }

    /// Override the clock (wasm builds inject `performance.now()`).
    pub fn set_time_source(&mut self, f: fn() -> f64) {
        self.time_src = f;
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    /// Full state reset between games (TT, killers, history) — prevents
    /// cross-game leakage in long matches (titanium binary_match lesson).
    pub fn reset_for_new_game(&mut self) {
        self.tt.clear();
        self.killers = Box::new([(0, 0); MAX_PLY]);
        self.history = Box::new([[0; 49]; 2]);
    }

    #[inline]
    fn check_stop(&mut self) {
        if let Some(nl) = self.node_limit {
            if self.nodes >= nl {
                self.stop = true;
                self.stop_reason = StopReason::Nodes;
                return;
            }
        }
        if self.nodes & 2047 == 0 {
            if let Some(dl) = self.deadline_ms {
                if (self.time_src)() >= dl {
                    self.stop = true;
                    self.stop_reason = StopReason::Time;
                }
            }
        }
    }

    /// Score moves in place: TT move > killers > captures*W + history.
    /// Clone moves are deduplicated per destination (identical children).
    fn score_into(&self, b: &Board, list: &mut MoveList, tt_mv: u32, ply: u32) {
        b.legal_moves_dedup_into(list);
        let opp = b.opp();
        let (k1, k2) = self.killers[ply as usize];
        let hist = &self.history[b.turn as usize];
        let mut tt_index = usize::MAX;
        for i in 0..list.len() {
            let m = list.move_at(i);
            let packed = m.to_u32();
            if packed == tt_mv {
                tt_index = i;
                list.set_score(i, i32::MAX / 2);
                continue;
            }
            let mut score = (opp & RING1[m.to as usize]).count_ones() as i32 * ORD_CAPTURE_UNIT;
            // Clone preference: a clone nets one more stone than a jump and
            // leaves the origin defended ("duplicate as much as you can").
            if m.is_clone() {
                score += ORD_CLONE_BONUS;
            } else {
                score -= ORD_JUMP_PEN;
            }
            if packed == k1 || packed == k2 {
                score += ORD_KILLER;
            }
            score += (hist[m.to as usize] as i32).min(ORD_HISTORY_MAX);
            list.set_score(i, score);
        }
        if tt_index != usize::MAX && tt_index != 0 {
            list.swap(tt_index, 0);
        }
    }

    fn negamax(
        &mut self,
        b: &Board,
        stack: &mut MoveStack,
        mut alpha: i32,
        mut beta: i32,
        depth: u32,
        ply: u32,
        null_allowed: bool,
    ) -> i32 {
        self.nodes += 1;
        self.check_stop();
        if self.stop {
            return 0;
        }
        debug_assert!((ply as usize) < MAX_PLY);
        if b.game_over() {
            return terminal_score(b, ply);
        }

        // Transposition table probe. Missing entry at depth >= 4 -> IIR
        // (internal iterative reduction, Moonbird): the node has no cached
        // best move to order by, so search one ply less.
        let alpha_orig = alpha;
        let mut tt_mv: u32 = 0;
        let mut depth = depth;
        let mut tt_hit = false;
        if let Some(e) = self.tt.probe(b.hash) {
            tt_hit = true;
            tt_mv = e.mv;
            if e.depth as u32 >= depth {
                let s = score_from_tt(e.score, ply);
                match e.flag {
                    FLAG_EXACT => return s,
                    FLAG_LOWER => {
                        if s > alpha {
                            alpha = s;
                        }
                    }
                    FLAG_UPPER => {
                        if s < beta {
                            beta = s;
                        }
                    }
                    _ => {}
                }
                if alpha >= beta {
                    return s;
                }
            }
        } else if depth >= 4 {
            depth -= 1;
        }
        let _ = tt_hit;

        let static_eval = self.eval_cached(b);

        // Reverse futility pruning (autaxx, exact form): standing pat is so
        // far below alpha that even a generous error margin cannot recover
        // -> fail low. Safe in Ataxx because the eval is capture-risk-aware.
        if null_allowed
            && ply > 0
            && depth <= 4
            && static_eval + RFP_MARGINS[(depth.max(1) - 1) as usize] < alpha
        {
            return alpha;
        }

        // Null-move pruning (autaxx + Moonbird hybrid): pass is a real move
        // in Ataxx and there is no zugzwang to speak of. Gates: not right
        // after another null, depth > 2, enough clone targets (mobility),
        // board not too full (forced-pass danger zone). R = 3.
        if null_allowed && depth > 2 {
            let clone_targets = (dist_union(b.occ[b.turn as usize], 1) & b.empty()).count_ones();
            let fill = (b.piece_cnt[0] + b.piece_cnt[1]) as f64 / SQUARES as f64;
            if clone_targets >= 11 && fill < 0.54 {
                let child = b.make_pass();
                let score = -self.negamax(
                    &child,
                    stack,
                    -beta,
                    -beta + 1,
                    depth - 3,
                    ply + 1,
                    false,
                );
                if self.stop {
                    return 0;
                }
                if score >= beta {
                    return score;
                }
            }
        }

        if depth == 0 {
            return static_eval;
        }

        {
            let list = &mut stack.lists[ply as usize];
            self.score_into(b, list, tt_mv, ply);
        }
        if stack.lists[ply as usize].is_empty() {
            // E5c trap deductions (user insight, free at this node): we are
            // permanently stuck (empties never regrow), opponent is not
            // (game_over was checked). SOUND cases only: while we are
            // stuck, their count never decreases and ours never increases
            // (their landings can convert our stones!).
            let my = b.occ[b.turn as usize].count_ones();
            let opp = b.occ[1 - b.turn as usize].count_ones();
            if opp > my {
                return -CERT_WIN; // they only grow, we only shrink
            }
            if opp == my
                && (dist_union(b.occ[1 - b.turn as usize], 1) & b.empty()) != 0
            {
                return -CERT_WIN; // one clone breaks the tie for good
            }
            // Opponent not stuck is guaranteed: double-stuck ends via passes >= 2.
            return -self.negamax(
                &b.make_pass(),
                stack,
                -beta,
                -alpha,
                depth - 1,
                ply + 1,
                true,
            );
        }

        // Late move pruning (autaxx): once captures exist for us, drop all
        // remaining quiet moves past move 27.
        let has_capture =
            stack.lists[ply as usize].captures_exist(b);
        if has_capture {
            let opp = b.opp();
            stack.lists[ply as usize].lmp_quiets(27, opp);
        }

        let n_moves = stack.lists[ply as usize].len();
        let ordered = !matches!(self.ordering, Ordering::Lazy);
        if ordered {
            if self.ordering == Ordering::Insertion {
                stack.lists[ply as usize].sort_by_score();
            } else {
                stack.lists[ply as usize]
                    .sort_radix_desc(&mut stack.scratch[ply as usize]);
            }
        }

        let mut best = i32::MIN;
        let mut best_mv: u32 = 0xFFFF;
        let mut move_idx = 0usize;
        loop {
            let m = if ordered {
                if move_idx >= n_moves {
                    break;
                }
                stack.lists[ply as usize].move_at(move_idx)
            } else {
                let list = &mut stack.lists[ply as usize];
                if list.is_empty() {
                    break;
                }
                list.pop_best()
            };

            // PVS + universal LMR (autaxx): first move full window full
            // depth; every later move scouts with a null window at a reduced
            // depth (r = 2, or 3 from the 10th move), re-searching full
            // window + full depth only when the scout beats alpha.
            let mut score;
            if move_idx == 0 {
                score = -self.negamax(&b.make(m), stack, -beta, -alpha, depth - 1, ply + 1, true);
                if self.stop {
                    return 0;
                }
            } else {
                let r: u32 = if move_idx < 9 { 2 } else { 3 };
                let reduced = depth.saturating_sub(1 + r);
                score = -self.negamax(
                    &b.make(m),
                    stack,
                    -alpha - 1,
                    -alpha,
                    reduced,
                    ply + 1,
                    true,
                );
                if self.stop {
                    return 0;
                }
                if score > alpha && reduced < depth - 1 {
                    score = -self.negamax(
                        &b.make(m),
                        stack,
                        -beta,
                        -alpha,
                        depth - 1,
                        ply + 1,
                        true,
                    );
                    if self.stop {
                        return 0;
                    }
                }
            }
            move_idx += 1;

            if score > best {
                best = score;
                best_mv = m.to_u32();
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        self.on_cutoff(b, m, depth, ply);
                        break;
                    }
                }
            }
        }
        self.finish_node(b, alpha_orig, beta, best, best_mv, depth, ply);
        best
    }

    /// Eval with the TT as cache (depth-0 EXACT entries).
    #[inline]
    fn eval_cached(&mut self, b: &Board) -> i32 {
        if let Some(e) = self.tt.probe(b.hash) {
            if e.flag == FLAG_EXACT && e.depth == 0 {
                return e.score;
            }
        }
        let ev = evaluate(b);
        self.tt.store(b.hash, 0, ev, 0, FLAG_EXACT);
        ev
    }

    /// Killer/history bookkeeping on a beta cutoff.
    #[inline]
    fn on_cutoff(&mut self, b: &Board, m: Move, depth: u32, ply: u32) {
        let packed = m.to_u32();
        let k = &mut self.killers[ply as usize];
        if k.0 != packed {
            k.1 = k.0;
            k.0 = packed;
        }
        let h = &mut self.history[b.turn as usize][m.to as usize];
        *h = h.saturating_add(1u32 << (depth * 2).min(20));
    }

    /// TT store at the end of a node (skipped when the clock cut it short).
    #[inline]
    fn finish_node(
        &mut self,
        b: &Board,
        alpha_orig: i32,
        beta: i32,
        best: i32,
        best_mv: u32,
        depth: u32,
        ply: u32,
    ) {
        if self.stop {
            return;
        }
        let flag = if best <= alpha_orig {
            FLAG_UPPER
        } else if best >= beta {
            FLAG_LOWER
        } else {
            FLAG_EXACT
        };
        self.tt
            .store(b.hash, best_mv, score_to_tt(best, ply), depth as u8, flag);
    }
    /// Iterative deepening root search. Falls back to any legal move.
    /// Whichever limit hits first (nodes / time / depth) stops the search;
    /// partial iterations are discarded, the last completed depth decides.
    pub fn search(&mut self, b: &Board, limits: &SearchLimits) -> SearchResult {
        self.start_ms = (self.time_src)();
        self.nodes = 0;
        self.stop = false;
        self.stop_reason = StopReason::Completed;
        self.node_limit = limits.max_nodes;
        self.deadline_ms = limits
            .time
            .map(|t| self.start_ms + t.as_secs_f64() * 1000.0);

        let mut stack = MoveStack::new();
        let mut result = SearchResult::default();
        // Age the history table so old cutoffs don't dominate forever.
        for row in self.history.iter_mut() {
            for h in row.iter_mut() {
                *h /= 2;
            }
        }
        {
            let root = &mut stack.lists[0];
            let tt_mv = self.tt.probe(b.hash).map(|e| e.mv).unwrap_or(0);
            self.score_into(b, root, tt_mv, 0);
            root.sort_by_score();
        }
        let n_root = stack.lists[0].len();
        if n_root == 0 {
            // Pass node: give the caller a meaningful score through the pass.
            if !b.game_over() {
                result.score =
                    -self.negamax(&b.make_pass(), &mut stack, -i32::MAX, i32::MAX, 1, 1, true);
            }
            result.elapsed = self.elapsed();
            result.nodes = self.nodes;
            return result; // caller must handle pass / game over
        }
        result.best = Some(stack.lists[0].move_at(0));

        let budget_ms = limits
            .time
            .map(|t| t.as_secs_f64() * 1000.0)
            .unwrap_or(f64::INFINITY);
        // Iteration cost prediction (titanium-engine pattern): project the
        // next iteration as the max of the last two completed iterations —
        // beats growth multipliers on noisy trees.
        let mut it_nodes: [f64; 2] = [0.0, 0.0];
        let mut it_ms: [f64; 2] = [0.0, 0.0];
        for depth in 1..=limits.max_depth.max(1) {
            let nodes_before = self.nodes;
            let iter_start = self.elapsed_ms();
            let mut alpha = i32::MIN + 1;
            let mut best_this = None;
            let mut best_score = i32::MIN;
            for i in 0..n_root {
                let m = stack.lists[0].move_at(i);
                let score = -self.negamax(&b.make(m), &mut stack, -i32::MAX, -alpha, depth - 1, 1, true);
                if self.stop {
                    break;
                }
                if score > best_score {
                    best_score = score;
                    best_this = Some(m);
                    if score > alpha {
                        alpha = score;
                    }
                }
            }
            // Partial-iteration adoption (titanium/Lague): moves recorded in
            // best_this completed their search before the stop, so their
            // scores are valid — adopt instead of discarding.
            if let Some(m) = best_this {
                result.best = Some(m);
                if best_score != i32::MIN {
                    result.score = best_score;
                }
                result.depth = depth;
                // Move best move to the front for the next iteration.
                if !self.stop {
                    for i in 0..n_root {
                        if stack.lists[0].move_at(i) == m {
                            stack.lists[0].swap(i, 0);
                            break;
                        }
                    }
                }
            }
            if self.stop {
                break;
            }
            // Predict the next iteration by TIME only: a node budget should
            // always be fully consumed (aborted-iteration work is already
            // counted and its completed moves adopted).
            it_ms[1] = it_ms[0];
            it_ms[0] = self.elapsed_ms() - iter_start;
            let proj_ms = it_ms[0].max(it_ms[1]);
            if self.elapsed_ms() + proj_ms > budget_ms {
                self.stop_reason = StopReason::Time;
                break;
            }
        }

        if self.stop {
            result.stopped_by = Some(self.stop_reason);
        } else if result.depth >= limits.max_depth {
            result.stopped_by = Some(StopReason::Completed);
        } else {
            result.stopped_by = Some(self.stop_reason);
        }
        result.nodes = self.nodes;
        result.elapsed = self.elapsed();
        result
    }

    #[inline]
    fn elapsed_ms(&self) -> f64 {
        (self.time_src)() - self.start_ms
    }

    #[inline]
    fn elapsed(&self) -> Duration {
        Duration::from_secs_f64((self.elapsed_ms().max(0.0)) / 1000.0)
    }
}

/// One-shot convenience: best move for `board` under `limits`.
pub fn best_move(b: &Board, limits: &SearchLimits) -> SearchResult {
    Searcher::new().search(b, limits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn no_time(depth: u32) -> SearchLimits {
        SearchLimits {
            time: None,
            max_nodes: None,
            max_depth: depth,
        }
    }

    #[test]
    fn finds_capture() {
        // Black g5 clones next to the lone white piece on f4 and converts it.
        let board = ".......
                     .......
                     .......
                     .....o.
                     ......x
                     .......
                     ....... b";
        let b = Board::from_str(board).unwrap();
        let r = best_move(&b, &no_time(4));
        let m = r.best.expect("must find a move");
        let after = b.make(m);
        let (black, white) = after.counts();
        assert_eq!(white, 0, "white piece must be captured, move {:?}", m);
        assert!(black >= 2, "black must not lose pieces");
    }

    #[test]
    fn prefers_big_infection() {
        // Whites on d7 and d5; jumping d4->c6/d6/e6 converts both at once.
        let board = "...o...
                     .......
                     ...o...
                     ...x...
                     .......
                     .......
                     ....... b";
        let b = Board::from_str(board).unwrap();
        let r = best_move(&b, &no_time(3));
        let m = r.best.expect("must find a move");
        let caps = b.captures(m.to);
        assert_eq!(caps, 2, "should convert both whites, got {:?}", m);
        assert!(matches!(m.to as usize, 9 | 10 | 11), "landing {:#?}", m);
    }

    #[test]
    fn handles_pass_position() {
        // Black is fully enclosed (ring1 + ring2 all occupied): no legal move,
        // but white can still move, so search must handle the pass.
        let board = ".......
                     .......
                     .......
                     .......
                     .ooooo.
                     .ooooo.
                     .ooxoo. b";
        let b = Board::from_str(board).unwrap();
        assert!(!b.has_moves(0));
        assert!(b.has_moves(1));
        let r = best_move(&b, &no_time(4));
        assert!(r.best.is_none());
        assert!(r.score < 0); // black is buried
    }

    #[test]
    fn search_terminates() {
        let b = Board::start();
        let r = best_move(&b, &no_time(6));
        assert!(r.best.is_some());
        assert!(r.nodes > 0);
    }

    #[test]
    fn node_budget_stops_first() {
        // A 1000-node budget must stay close to the budget (overshoot
        // bounded by the last node expansion) and still return a move.
        let b = Board::start();
        let limits = SearchLimits {
            time: None,
            max_nodes: Some(1000),
            max_depth: 20,
        };
        let r = best_move(&b, &limits);
        assert!(r.nodes < 3000, "nodes {}", r.nodes);
        assert!(r.best.is_some());
    }

    #[test]
    fn trap_certificate_is_proven_win() {
        // White d4 fully sealed in by a 24-stone black ring (no empty square
        // within white's reach — permanently stuck). Black to move, 24 > 1,
        // black can still expand: forced majority win, search must see it
        // instantly at any depth.
        let board = ".......
                     .xxxxx.
                     .xxxxx.
                     .xxoxx.
                     .xxxxx.
                     .xxxxx.
                     ....... b";
        let b = Board::from_str(board).unwrap();
        assert!(!b.has_moves(1), "white must be trapped");
        let r = best_move(&b, &no_time(2));
        assert!(r.score >= 80_000, "certificate score, got {} depth {} nodes {} best {:?}", r.score, r.depth, r.nodes, r.best.map(|m| m.to_u32()));
    }

    #[test]
    fn all_orderings_find_valid_moves() {
        // Every ordering strategy must return some legal move from the start.
        let b = Board::start();
        for ordering in [Ordering::Lazy, Ordering::Insertion, Ordering::Radix] {
            let mut s = Searcher::with_tt_bits(16);
            s.set_ordering(ordering);
            let r = s.search(&b, &no_time(5));
            let m = match r.best {
                Some(m) => m,
                None => panic!(
                    "{ordering:?} None sb={:?} sc={} d={}",
                    r.stopped_by, r.score, r.depth
                ),
            };
            assert!(b.legal_moves().contains(&m), "{:?} illegal", m);
        }
    }

    fn ordering_debug(o: Ordering) -> &'static str {
        match o {
            Ordering::Lazy => "lazy",
            Ordering::Insertion => "insertion",
            Ordering::Radix => "radix",
        }
    }
}

