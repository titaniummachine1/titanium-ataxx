//! Evaluation and iterative-deepening alpha-beta (negamax) search.
//!
//! Search order: TT probe -> eval cache (depth 0) -> move scoring
//! (TT move > killers > captures*W + history) -> lazy selection -> alpha-beta.

/// exp/nnue-speed: static-eval cache entries (2^16 x 16 B = 1 MiB per searcher).
const ECACHE_BITS: usize = 16;

use crate::board::{dist_union, jump_union, Board, Move, MoveList, RING1, SQUARES};
use crate::nnue::Nnue;
use crate::sancta::{S1Acc, S1Net};
use crate::tuple::TupleTable;
use std::time::Duration;

pub const MATE: i32 = 100_000;
/// Scores at least this close to MATE are mate scores (TT adjustment bound).
const MATE_BOUND: i32 = MATE - 2000;
/// Search window infinity: finite and just above every real score (Stockfish VALUE_INFINITE style). The old
/// i32::MIN+1 / i32::MAX window ends leaked into null-window searches as "scores" and made score_to_tt overflow.
const INF: i32 = MATE + 1000;

/// Search stack supports the full u8 depth range plus the root ply.
const MAX_PLY: usize = u8::MAX as usize + 1;

/// Per-ply move buffers, one alloc per Searcher, reused across searches.
/// `acc` carries the lazy sancta accumulator per ply (refresh_root once at
/// the root, update_child parent->child diffs). OFF path = None check only.
/// MoveList 256x2B moves + 256x4B scores + len = ~1.5KB, acc 2x64x2B =
/// 256B -> ~1.8KB/ply, ~72KB total. No Clone, no Vec, no Drop in the tree.
struct MoveStack {
    lists: Box<[MoveList; MAX_PLY]>,
    acc: Box<[S1Acc; MAX_PLY]>,
    /// exp/search-moonbird: static eval the parent already computed for the child at this ply (futility
    /// reduction), consumed at the child's entry so the eval is not paid twice. i32::MIN = none.
    pre_eval: Box<[i32; MAX_PLY]>,
    /// Zobrist hash at each ply along the active search path for circular-move / repetition pruning.
    path_hash: Box<[u64; MAX_PLY]>,
    /// Lowest ply index since the most recent irreversible (clone) move; positions before this index
    /// have strictly fewer total stones on the board and can never be repeated.
    rep_base: Box<[u32; MAX_PLY]>,
}

impl MoveStack {
    fn new() -> MoveStack {
        MoveStack {
            lists: Box::new([MoveList::new(); MAX_PLY]),
            acc: Box::new([[[0; 64]; 2]; MAX_PLY]),
            pre_eval: Box::new([i32::MIN; MAX_PLY]),
            path_hash: Box::new([0u64; MAX_PLY]),
            rep_base: Box::new([0u32; MAX_PLY]),
        }
    }
}

/// Material weight per piece (eval = piece diff * MATERIAL).
const MATERIAL: i32 = 100;
/// Tempo: flat bonus for the side to move. autaxx uses 1.5-2 stones worth.
const TEMPO: i32 = 150;
/// Reverse futility pruning margins, index = depth-1 (autaxx, stone = 100).
const RFP_MARGINS: [i32; 4] = [257, 347, 478, 774];
const MB_FUT_MIN_IDX: usize = 1;

/// exp/search-spsa-disagree: Selectivity search tuning (TweakSels) modulated by TNN3 Disagreement Head.
/// Values read once from TITANIUM_TUNE="name=val,..."; defaults preserve main bit-identically.
#[derive(Clone, Copy, Debug)]
pub struct Tune {
    pub fut_div: i32,        // child futility divisor: 1 ply per fut_div cp below alpha (default 57)
    pub rfp_scale: i32,      // RFP base margins scale in percent of [257, 347, 478, 774] (default 100)
    pub lmr_half: i32,       // LMR depth * half / (half + n) (default 47)
    pub lmr_min: i32,        // LMR starting move index (default 6)
    pub dis_rfp_num: i32,    // Disagreement head RFP volatility dampening (scaled by 128) (default 32)
    pub dis_fut_num: i32,    // Disagreement head Futility volatility credit (scaled by 128) (default 48)
    pub dis_lmr_num: i32,    // Disagreement head LMR discount on volatile nodes (scaled by 256) (default 0)
    pub dis_vol_thresh: i32, // Disagreement threshold (cp) for tactical +1 extension (default 300)
}

impl Default for Tune {
    fn default() -> Self {
        Tune {
            fut_div: 57,
            rfp_scale: 100,
            lmr_half: 47,
            lmr_min: 6,
            dis_rfp_num: 32,
            dis_fut_num: 48,
            dis_lmr_num: 0,
            dis_vol_thresh: 300,
        }
    }
}

fn tune_from_env() -> Tune {
    let mut t = Tune::default();
    if let Ok(v) = std::env::var("TITANIUM_TUNE") {
        for kv in v.split(',') {
            let mut it = kv.splitn(2, '=');
            let (Some(k), Some(x)) = (it.next(), it.next()) else { continue };
            let Ok(x) = x.trim().parse::<i32>() else { continue };
            match k.trim() {
                "fut_div" => t.fut_div = x.max(1),
                "rfp_scale" => t.rfp_scale = x.max(0),
                "lmr_half" => t.lmr_half = x.max(1),
                "lmr_min" => t.lmr_min = x.max(1),
                "dis_rfp_num" => t.dis_rfp_num = x.max(0),
                "dis_fut_num" => t.dis_fut_num = x.max(0),
                "dis_lmr_num" => t.dis_lmr_num = x.max(0),
                "dis_vol_thresh" => t.dis_vol_thresh = x.max(0),
                _ => {}
            }
        }
    }
    t
}

static TUNE: std::sync::OnceLock<Tune> = std::sync::OnceLock::new();

#[inline]
pub fn tune() -> &'static Tune {
    TUNE.get_or_init(tune_from_env)
}

#[inline(always)]
pub fn pack_eval_dis(eval: i32, dis: i32) -> i32 {
    let e = (eval.clamp(-16384, 16383) + 16384) as u32;
    let d = (dis.clamp(0, 16383) as u32) & 0x3FFF;
    let packed = ((d << 15) | e) as i32;
    if packed == i32::MIN { packed + 1 } else { packed }
}

#[inline(always)]
pub fn unpack_eval_dis(packed: i32) -> (i32, i32) {
    let p = packed as u32;
    let e = ((p & 0x7FFF) as i32) - 16384;
    let d = ((p >> 15) & 0x3FFF) as i32;
    (e, d)
}
/// exp/fut-cheap (D108): before the NNUE child eval of the futility step, a cheap estimate of it
/// pred = (357 * hand(child) - 542 * parent static eval) / 1000 + 248 (least squares on 4.1M recorded futility calls of
/// e11827d1, residual sd 191 cp; the hand eval costs ~72 ns vs ~6.5 us for the NNUE; S57 was a better fit but costs
/// 1.8 us). If the move would be pruned even with the child eval FUT_CHEAP_M cp lower than predicted (better for us), it
/// is pruned without the NNUE eval (offline at M 350: 49% of those NNUE evals saved, 2.8% of the moves the net keeps
/// pruned). BUILD-TIME env FUT_CHEAP_M (0 = off).
const FUT_CHEAP_M: i32 = env_usize(option_env!("FUT_CHEAP_M"), 350) as i32;
/// exp/fut-sn (D110): the cheap estimate above is replaced by the futility small net's LOWER BOUND of the big net's child
/// eval (crate::smallnet, compiled in); FUT_SN_M = extra cp subtracted from that bound (BUILD-TIME env, default 0).
/// FUT_CHEAP_M only switches the pre-filter on (> 0) or off (0) here.
const FUT_SN_M: i32 = env_usize(option_env!("FUT_SN_M"), 0) as i32;

/// Process-wide switch for the futility small net (D112): ON by default (as strong per node as main, 5k +3 [-30,+37],
/// x1.77 faster). `titanium-cli ... --no-fut-sn` turns it off for exact full-strength search (slower). Read once per
/// search into the searcher, never per node.
static FUT_SN_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_fut_sn(on: bool) {
    FUT_SN_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn small_net() -> &'static crate::smallnet::SmallNet {
    static SN: std::sync::OnceLock<crate::smallnet::SmallNet> = std::sync::OnceLock::new();
    SN.get_or_init(crate::smallnet::SmallNet::builtin)
}

// Ordering tiers (titanium-engine pattern: compile-asserted bands so no
// ordering signal can structurally leak into its neighbor).
const ORD_TT: i32 = i32::MAX / 2;

const fn env_usize(v: Option<&str>, default: usize) -> usize {
    match v {
        Some(v) => {
            let b = v.as_bytes();
            let mut i = 0;
            let mut n = 0usize;
            while i < b.len() {
                n = n * 10 + (b[i] - b'0') as usize;
                i += 1;
            }
            n
        }
        None => default,
    }
}
/// exp/eval-order (user idea): moves are also ordered by their child's static eval (our view = minus the child's
/// eval, cp), weight EVORD_W ordering points per cp on top of the usual score (capture 1000/stone, killer 100000,
/// history <= 2000); TT move stays first. Default 100 with EVORD_CACHE=1 (user order 2026-10-25); 0 = off. The evals are cached, so the futility child evals reuse
/// them. BUILD-TIME (env EVORD_W / EVORD_MIN_DEPTH at cargo build).
const EVORD_W: i32 = env_usize(option_env!("EVORD_W"), 100) as i32;
const EVORD_MIN_DEPTH: u32 = env_usize(option_env!("EVORD_MIN_DEPTH"), 1) as u32;
/// 1 = only children whose eval is ALREADY in the eval cache (e.g. from the previous iterative-deepening pass) get the
/// eval term, scored as their gain over the node's own static eval; no new eval is ever computed for ordering.
const EVORD_CACHE: bool = env_usize(option_env!("EVORD_CACHE"), 1) == 1;
const ORD_KILLER: i32 = 100_000;
const ORD_CAPTURE_UNIT: i32 = 1000; // per conversion, max 8
const ORD_CLONE_BONUS: i32 = 500;
const ORD_JUMP_PEN: i32 = 500;
/// exp/ab O1: extra jump penalty per own stone neighboring the origin.
const ORD_JUMP_CONN: i32 = 250;
const ORD_HISTORY_MAX: i32 = 90_000;
/// exp/search-histmb: Moonbird 1.1.0 history (searchUtils.nim) keyed [stm][from][to] (clones keyed [to][to]):
/// +depth^2 for the move that raised alpha, -depth^2/24 for every searched move that did not, |h| capped at 107000
/// (x24 here so the malus stays integer) with the colour's table shrunk /1.25 when the cap is hit. Ordering weight
/// = Moonbird historyMoveOrderingFactor 2.0 stones = 2 conversions at full history.
const HIST_SCALE: i32 = 24;
const HIST_MAX: i32 = 107_000 * HIST_SCALE;
const ORD_HIST_FULL: i32 = 2 * ORD_CAPTURE_UNIT;
type History = [[[i32; 49]; 49]; 2];

#[inline]
fn hist_key(m: Move) -> (usize, usize) {
    if m.is_clone() {
        (m.to as usize, m.to as usize)
    } else {
        (m.from as usize, m.to as usize)
    }
}
const _: () = assert!(ORD_HISTORY_MAX + 8 * ORD_CAPTURE_UNIT + ORD_CLONE_BONUS < ORD_KILLER);
const _: () = assert!(ORD_KILLER < ORD_TT);

/// Contact penalty per ENDANGERED stone (E2, "virus strategy"): stones the
/// enemy can infect right now. Avoid contact early, mass-clone first.
/// Restored S13 (was purged in S1): +190 Elo standalone.
const CONTACT_PEN: i32 = 15;
/// Multi-capture exposure (E4, user idea): for every empty landing square
/// the enemy can reach, each of our stones BEYOND THE FIRST that one enemy
/// landing would convert costs this much. Restored S13: +330 standalone.
const MULTICAP_PEN: i32 = 20;

/// Empty landing squares `them` can reach right now (shared E2+E4 input —
/// computed once per side, not once per term).
#[inline]
fn enemy_landings(b: &Board, them: u8) -> u64 {
    let tp = b.occ[them as usize];
    b.empty() & (dist_union(tp, 1) | jump_union(tp))
}

/// E2: stones of `side` the enemy can currently convert — own stones ring-1
/// of an enemy-reachable landing square.
#[inline]
fn endangered_count(b: &Board, side: u8, landings_them: u64) -> u32 {
    (b.occ[side as usize] & dist_union(landings_them, 1)).count_ones()
}

/// E4: for each enemy-reachable landing square, own stones there beyond the
/// first (each extra one = a stone lost in the same single enemy move).
fn multicapture_penalty(b: &Board, side: u8, landings_them: u64) -> i32 {
    let us = b.occ[side as usize];
    let mut pen = 0i32;
    let mut ls = landings_them;
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

/// Static eval from the side-to-move's perspective. Material + PST + tempo
/// are O(1) incremental; E2/E4 regional terms (contact + multicap exposure)
/// scan enemy reachability per node — the knowledge that won +190/+330.
/// The n-tuple term (`tup`, S20) adds learned local-shape knowledge on top;
/// None = pure hand eval (zero-cost OFF).
pub fn evaluate(b: &Board, tup: Option<&TupleTable>, native_tup: bool) -> i32 {
    let mut score = MATERIAL * (b.piece_cnt[0] as i32 - b.piece_cnt[1] as i32) + b.pst;
    // Tuple-native eval (S21d): mat + PST + tempo + tuples, NO E2/E4
    // regional scans (~400 ops/node back). The pairs must carry contact-like
    // shapes themselves (trained vs the simple base). Falls back to the
    // full hand eval when no table is attached.
    if native_tup {
        if b.turn == 0 {
            score += TEMPO;
        } else {
            score -= TEMPO;
        }
        let out = if b.turn == 0 { score } else { -score };
        if let Some(t) = tup {
            return out + t.forward(b);
        }
        return out;
    }
    // Black-relative balances (negated for white below, like pst).
    let land_b = enemy_landings(b, 1); // squares white threatens to land on
    let land_w = enemy_landings(b, 0);
    score -= CONTACT_PEN * endangered_count(b, 0, land_b) as i32;
    score += CONTACT_PEN * endangered_count(b, 1, land_w) as i32;
    score -= multicapture_penalty(b, 0, land_b);
    score += multicapture_penalty(b, 1, land_w);
    if b.turn == 0 {
        score += TEMPO;
    } else {
        score -= TEMPO;
    }
    let mut out = if b.turn == 0 { score } else { -score };
    // Tuple term is stm-relative by construction: add after the flip.
    // None branch = zero-cost OFF (single predictable branch).
    // forward_fast reuses this eval's landing sets (bit-identical scores,
    // zero recompute — capconv's land_u/land_t ARE land_w/land_b per side).
    if let Some(t) = tup {
        let land_me = if b.turn == 0 { land_w } else { land_b };
        let land_opp = if b.turn == 0 { land_b } else { land_w };
        out += t.forward_fast(b, land_me, land_opp);
    }
    out
}

/// Score moves in place: TT move > killers > captures*W + history.
/// Free function (not a method) so negamax can borrow the move stack
/// mutably while reading killers/history immutably — no split-borrow
/// fight, no unsafe, no clone.
#[inline]
fn score_into(
    b: &Board,
    list: &mut MoveList,
    tt_mv: u32,
    ply: u32,
    killers: &[(u32, u32); MAX_PLY],
    history: &History,
) {
    b.legal_moves_dedup_into(list);
    let opp = b.opp();
    let (k1, k2) = killers[ply as usize];
    let hist = &history[b.turn as usize];
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
        // (exp/ab A tested no-asymmetry: 41-55, keep preference.)
        if m.is_clone() {
            score += ORD_CLONE_BONUS;
        } else {
            // exp/ab O1 (hole-aware jumps): a jump vacates its origin —
            // leaving a connected origin breaks formation, so penalty
            // scales with own stones adjacent to the origin square.
            let conn = (b.occ[b.turn as usize] & RING1[m.from as usize]).count_ones() as i32;
            score -= ORD_JUMP_PEN + conn * ORD_JUMP_CONN;
        }
        if packed == k1 || packed == k2 {
            score += ORD_KILLER;
        }
        let (hf, ht) = hist_key(m);
        score += (hist[hf][ht] as i64 * ORD_HIST_FULL as i64 / HIST_MAX as i64) as i32;
        list.set_score(i, score);
    }
    if tt_index != usize::MAX && tt_index != 0 {
        list.swap(tt_index, 0);
    }
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

/// Botzone-judge terminal score from the side-to-move's perspective: filled
/// counts (the mobile side absorbs empties), so a stalemate of the opponent
/// is a forced win, not a pass-and-continue. Same MATE shape as `terminal_score`.
fn bz_terminal_score(b: &Board, ply: u32) -> i32 {
    let (fb, fw) = b.bz_counts();
    let diff = if b.turn == 0 {
        fb as i32 - fw as i32
    } else {
        fw as i32 - fb as i32
    };
    if diff > 0 {
        MATE - ply as i32
    } else if diff < 0 {
        -MATE + ply as i32
    } else {
        0
    }
}

/// Move scoring is fixed: TT move > killers > captures*W + history, consumed
/// lazily (repeated argmax — only pays for moves examined before a cutoff).

/// Transposition table entry (24 bytes with padding).
#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    mv: u32, // packed Move; 0xFFFF = pass; 0 = none (eval cache entries)
    score: i32,
    depth: u8,
    flag: u8, // 0 empty, 1 exact, 2 lower, 3 upper
    age: u8,  // Generation age for FIFO/LRU eviction without RAM increase (fits in compiler padding)
}

const _: () = assert!(std::mem::size_of::<TtEntry>() == 24);

const FLAG_EMPTY: u8 = 0;
const FLAG_EXACT: u8 = 1;
const FLAG_LOWER: u8 = 2;
const FLAG_UPPER: u8 = 3;

/// Correction history (S54 experiment, OFF unless enabled via set_corr):
/// per-context EMA of (search score - static eval), added to future static
/// evals. Two cheap O(1) keys: stm piece-diff bucket + empties bucket.
/// Bounded, reset per game, never stored in TT (TT keeps search scores).
const CORR_SIZE: usize = 64;
const CORR_SCALE: i32 = 256;
const CORR_WMAX: i32 = 128;
/// Per-entry rails (raw cp): search artifacts beyond this don't teach.
const CORR_TGT_MAX: i32 = 256;
const CORR_ENTRY_MAX: i32 = 512;
/// Applied correction rails (raw cp sum): small nudge, never a verdict.
const CORR_APPLY_MAX: i32 = 64;

pub struct CorrHist {
    pd: Box<[i32; CORR_SIZE]>,
    em: Box<[i32; CORR_SIZE]>,
}

impl CorrHist {
    pub fn new() -> Self {
        CorrHist {
            pd: Box::new([0; CORR_SIZE]),
            em: Box::new([0; CORR_SIZE]),
        }
    }

    #[inline]
    fn idx_pd(b: &Board) -> usize {
        let d = b.piece_cnt[0] as i32 - b.piece_cnt[1] as i32 + 49;
        ((d / 2).clamp(0, 63)) as usize
    }

    #[inline]
    fn idx_em(b: &Board) -> usize {
        let e = b.empty().count_ones() as usize;
        (e * CORR_SIZE / 50).min(CORR_SIZE - 1)
    }

    /// Corrected static eval (stm-relative in, stm-relative out).
    #[inline]
    pub fn apply(&self, b: &Board, raw: i32) -> i32 {
        let c = (self.pd[Self::idx_pd(b)] + self.em[Self::idx_em(b)]) / 4;
        raw + c.clamp(-CORR_APPLY_MAX, CORR_APPLY_MAX)
    }

    /// EMA toward (search - static), depth-weighted. Skips mate scores
    /// (not eval error) and zero diffs (no information).
    pub fn update(&mut self, b: &Board, best: i32, static_eval: i32, depth: u32) {
        if best.abs() > MATE_BOUND || static_eval.abs() > MATE_BOUND {
            return;
        }
        let diff = (best - static_eval).clamp(-CORR_TGT_MAX, CORR_TGT_MAX);
        if diff == 0 {
            return;
        }
        let w = ((depth * depth + 2 * depth + 1) as i32).min(CORR_WMAX);
        for (tab, idx) in [
            (&mut *self.pd, Self::idx_pd(b)),
            (&mut *self.em, Self::idx_em(b)),
        ] {
            let e = tab[idx];
            tab[idx] = (e + w * (diff - e) / CORR_SCALE).clamp(-CORR_ENTRY_MAX, CORR_ENTRY_MAX);
        }
    }
}

/// Fixed-size transposition table, power-of-two entry count. Depth-0 EXACT
/// entries double as the eval cache (Ataxx transpositions repeat a lot).
pub struct Tt {
    buf: Vec<TtEntry>,
    mask: usize,
    pub age: u8,
}

impl Tt {
    /// 2^bits entries (bits=20 -> 1M entries, 24 MB).
    pub fn new(bits: usize) -> Tt {
        let size = 1usize << bits.min(24);
        Tt {
            buf: vec![
                TtEntry {
                    key: 0,
                    mv: 0,
                    score: 0,
                    depth: 0,
                    flag: FLAG_EMPTY,
                    age: 0,
                };
                size
            ],
            mask: size - 1,
            age: 0,
        }
    }

    pub fn clear(&mut self) {
        for e in self.buf.iter_mut() {
            e.flag = FLAG_EMPTY;
        }
        self.age = 0;
    }

    #[inline]
    pub fn step_age(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    #[inline]
    fn probe(&mut self, key: u64) -> Option<TtEntry> {
        let e = &mut self.buf[(key as usize) & self.mask];
        if e.flag != FLAG_EMPTY && e.key == key {
            e.age = self.age;
            Some(*e)
        } else {
            None
        }
    }

    #[inline]
    fn store(&mut self, key: u64, mut mv: u32, score: i32, depth: u8, flag: u8) {
        let e = &mut self.buf[(key as usize) & self.mask];
        let same_pos = e.flag != FLAG_EMPTY && e.key == key;

        // Preserve refutation move on fail-low (FLAG_UPPER):
        if same_pos && flag == FLAG_UPPER && e.mv != 0 && e.mv != 0xFFFF {
            mv = e.mv;
        }

        // Replacement policy:
        // 1. Slot is empty
        // 2. Same position (always update bounds/depth)
        // 3. Stale generation/age (older turns aged out from dead subtrees)
        // 4. Deeper or equal depth
        let age_diff = self.age.wrapping_sub(e.age);
        let replace = e.flag == FLAG_EMPTY
            || same_pos
            || (!same_pos && (age_diff >= 2 || (age_diff == 1 && depth.saturating_add(4) >= e.depth)))
            || e.depth <= depth;

        if replace {
            *e = TtEntry {
                key,
                mv,
                score,
                depth,
                flag,
                age: self.age,
            };
        }
    }
}

/// Mate-score adjustment: store distance-from-node so entries stay correct
/// at any retrieval ply.
#[inline]
fn score_to_tt(score: i32, ply: u32) -> i32 {
    if score >= MATE_BOUND {
        score.saturating_add(ply as i32)
    } else if score <= -MATE_BOUND {
        score.saturating_sub(ply as i32)
    } else {
        score
    }
}

#[inline]
fn score_from_tt(score: i32, ply: u32) -> i32 {
    if score >= MATE_BOUND {
        score.saturating_sub(ply as i32)
    } else if score <= -MATE_BOUND {
        score.saturating_add(ply as i32)
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
    /// Wall-clock budget per search; `None` = unlimited (polled every 64 nodes).
    pub time: Option<Duration>,
    /// Hard node budget; search stops once `nodes` reaches this.
    pub max_nodes: Option<u64>,
    /// Hard depth cap, in plies (0 is treated as 1 for legacy callers).
    pub max_depth: u8,
}

impl Default for SearchLimits {
    fn default() -> Self {
        SearchLimits {
            time: Some(Duration::from_millis(1000)),
            max_nodes: None,
            max_depth: u8::MAX,
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

pub struct Searcher<'a> {
    pub nodes: u64,
    /// Absolute deadline in milliseconds (via `time_src`), if time-limited.
    deadline_ms: Option<f64>,
    start_ms: f64,
    stop: bool,
    stop_reason: StopReason,
    node_limit: Option<u64>,
    /// True if any branch in the current root iteration hit a heuristic leaf, pruning, or unresolved TT cutoff.
    heuristic_used: bool,
    /// futility small-net pre-filter on (FUT_SN_ON at search start)
    fut_sn: bool,
    /// Monotonic milliseconds. Native builds use `Instant`; wasm builds
    /// inject `performance.now()` (std time does not exist on wasm32).
    time_src: fn() -> f64,
    /// Optional external stop request (set from another thread, e.g. a driver
    /// watchdog). Polled on the same 2048-node cadence as the clock.
    ext_stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Transposition table (persists across searches in the same Searcher).
    tt: Tt,
    /// exp/nnue-speed: static-eval cache (position hash -> raw eval). The eval is a pure function of the position, so a
    /// hit returns exactly what a recomputation would: search is bit-identical, only faster (iterative deepening
    /// re-evaluates ~42% of positions). Cleared on new game and whenever the eval source changes.
    ecache: Box<[(u64, i32)]>,
    /// Two killer moves per ply (packed u32 moves), refreshed on cutoffs.
    killers: Box<[(u32, u32); MAX_PLY]>,
    /// History heuristic: [mover][to] cutoff counters, aged per search.
    history: Box<History>,
    /// Optional sancta net (None = classical slim eval). Borrowed weights,
    /// set once via set_sancta, read-only in the tree. Zero refcount,
    /// zero unsafe — the borrow checker proves the net outlives the search.
    sancta: Option<&'a S1Net>,
    /// Optional n-tuple table (None = pure hand eval). Borrowed, read-only
    /// in the tree, same zero-cost-OFF pattern as sancta.
    tuple: Option<&'a TupleTable>,
    /// Optional exp/nnue net (None = unchanged eval path). Takes precedence over sancta/tuple when set.
    nnue: Option<&'a Nnue>,
    /// Tuple-native eval mode: skip E2/E4 scans, mat+PST+tempo+tuples only.
    /// False by default (full hand eval + optional additive tuples).
    tup_native: bool,
    /// Correction history (None = OFF, zero-cost branch). Owned tables,
    /// reset per game; never merged into weights or TT.
    corr: Option<CorrHist>,
    /// Per-ply move/acc buffers, reused across searches (no per-search
    /// alloc: MoveStack is ~72KB of lists + acc, was rebuilt per search).
    stack: MoveStack,
}

/// Monotonic milliseconds since process start (native default time source).
fn native_now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static BOOT: OnceLock<Instant> = OnceLock::new();
    BOOT.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

impl Default for Searcher<'_> {
    fn default() -> Self {
        Self::new()
    }
}
impl<'a> Searcher<'a> {
    /// Standard production searcher: bundled main NNUE, default TT size.
    pub fn new() -> Searcher<'static> {
        Searcher::with_main_network()
    }

    /// Low-level unconfigured searcher for custom evaluators or explicit classical experiments.
    pub fn new_unconfigured() -> Searcher<'a> {
        Searcher::with_tt_bits_unconfigured(20)
    }

    /// Production searcher with the strongest NNUE bundled into this build and a custom TT size.
    pub fn with_tt_bits(bits: usize) -> Searcher<'static> {
        let mut searcher = Searcher::with_tt_bits_unconfigured(bits);
        searcher.set_nnue(Some(Nnue::bundled_static()));
        searcher
    }

    /// Allocate a searcher without choosing an evaluator. This is intended for explicit model replacement.
    pub fn with_tt_bits_unconfigured(bits: usize) -> Searcher<'a> {
        Searcher {
            nodes: 0,
            deadline_ms: None,
            start_ms: 0.0,
            stop: false,
            stop_reason: StopReason::Completed,
            node_limit: None,
            heuristic_used: false,
            fut_sn: true,
            time_src: native_now_ms,
            ext_stop: None,
            tt: Tt::new(bits),
            ecache: vec![(0u64, i32::MIN); 1 << ECACHE_BITS].into_boxed_slice(),
            killers: Box::new([(0, 0); MAX_PLY]),
            history: Box::new([[[0; 49]; 49]; 2]),
            sancta: None,
            tuple: None,
            nnue: None,
            tup_native: false,
            corr: None,
            stack: MoveStack::new(),
        }
    }

    /// A production searcher with the strongest NNUE bundled into this build.
    pub fn with_main_network() -> Searcher<'static> {
        Searcher::with_tt_bits(20)
    }

    /// Attach a sancta net (None = classical). Borrowed for the search
    /// duration; zero-cost OFF (None branch).
    pub fn set_sancta(&mut self, net: Option<&'a S1Net>) {
        self.sancta = net;
        self.clear_ecache();
    }

    /// Attach an n-tuple table (None = pure hand eval). Borrowed for the
    /// search duration; zero-cost OFF (None branch).
    pub fn set_nnue(&mut self, n: Option<&'a Nnue>) {
        self.nnue = n;
        self.clear_ecache();
    }

    pub fn set_tuple(&mut self, t: Option<&'a TupleTable>) {
        self.tuple = t;
        self.clear_ecache();
    }

    /// Tuple-native eval mode (skip E2/E4 scans). Predictable branch,
    /// ~free; the mode it selects is the fast one.
    pub fn set_tuple_native(&mut self, native: bool) {
        self.tup_native = native;
        self.clear_ecache();
    }

    /// Correction-history tables, off by default. Enabled searchers learn
    /// per-context eval biases within a game; disabled = main behavior.
    pub fn set_corr(&mut self, on: bool) {
        self.corr = on.then(CorrHist::new);
    }

    /// Static eval of a non-sancta node: the NNUE when one is loaded (exp/nnue), else the tuple/classical eval.
    /// Every static eval goes through here (node entry AND the futility child eval) so pruning and leaves agree.
    fn clear_ecache(&mut self) {
        self.ecache.fill((0, i32::MIN));
    }

    #[inline]
    fn leaf_eval(&mut self, b: &Board) -> i32 {
        self.leaf_eval_with_dis(b).0
    }

    #[inline]
    fn leaf_eval_with_dis(&mut self, b: &Board) -> (i32, i32) {
        let slot = (b.hash as usize) & ((1 << ECACHE_BITS) - 1);
        let (k, v) = self.ecache[slot];
        if k == b.hash && v != i32::MIN {
            return unpack_eval_dis(v);
        }
        let (eval, dis) = self.leaf_eval_raw_with_dis(b);
        self.ecache[slot] = (b.hash, pack_eval_dis(eval, dis));
        (eval, dis)
    }

    #[inline]
    fn leaf_eval_raw_with_dis(&self, b: &Board) -> (i32, i32) {
        match self.nnue {
            // v2 nets carry an S57 skip input (the tuple eval of the same board); v1 ignores it
            Some(nn) if nn.v2 => nn.eval_with_dis(b, evaluate(b, self.tuple, self.tup_native)),
            Some(nn) => nn.eval_with_dis(b, 0),
            None => (evaluate(b, self.tuple, self.tup_native), 0),
        }
    }

    /// Raw static eval plus correction (if enabled). Single call site shape
    /// so both negamax static-eval points stay consistent.
    #[inline]
    fn corrected(&self, b: &Board, raw: i32) -> i32 {
        match &self.corr {
            Some(c) => c.apply(b, raw),
            None => raw,
        }
    }

    /// Attach an external stop flag: when it reads `true` the running search
    /// stops at the next poll (<= 2048 nodes) and returns its best completed
    /// depth. The flag is never cleared by the engine.
    pub fn set_stop_flag(&mut self, f: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) {
        self.ext_stop = f;
    }

    /// Override the clock (wasm builds inject `performance.now()`).
    pub fn set_time_source(&mut self, f: fn() -> f64) {
        self.time_src = f;
    }

    /// Full state reset between games (TT, killers, history) — prevents
    /// cross-game leakage in long matches (titanium binary_match lesson).
    pub fn reset_for_new_game(&mut self) {
        self.tt.clear();
        self.clear_ecache();
        *self.killers = [(0, 0); MAX_PLY];
        *self.history = [[[0; 49]; 49]; 2];
        if self.corr.is_some() {
            self.corr = Some(CorrHist::new());
        }
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
        if self.nodes & 63 == 0 {
            if let Some(f) = &self.ext_stop {
                if f.load(std::sync::atomic::Ordering::Relaxed) {
                    self.stop = true;
                    self.stop_reason = StopReason::Time;
                    return;
                }
            }
            if let Some(dl) = self.deadline_ms {
                if (self.time_src)() >= dl {
                    self.stop = true;
                    self.stop_reason = StopReason::Time;
                }
            }
        }
    }

    #[inline]
    fn negamax<const BZ: bool>(
        &mut self,
        b: &Board,
        mut alpha: i32,
        mut beta: i32,
        depth: u32,
        ply: u32,
        null_allowed: bool,
    ) -> i32 {
        self.nodes += 1;
        // Hard stack guard (release builds too): never index past the MAX_PLY-sized stacks.
        if ply as usize >= MAX_PLY - 1 {
            self.heuristic_used = true;
            return self.leaf_eval(b);
        }
        // consume a parent-computed static eval first, so a TT return below can never leave it stale
        let pre = std::mem::replace(&mut self.stack.pre_eval[ply as usize], i32::MIN);
        self.check_stop();
        if self.stop {
            return 0;
        }
        debug_assert!((ply as usize) < MAX_PLY);
        // Build-time rules selection (const generic: the dead branch is
        // compiled out, zero hot-path cost either way).
        if BZ {
            // Botzone judge checks both sides after every move (!blackHasMove || !whiteHasMove):
            // if either side has no legal moves, or wipeout/full, the game ends immediately.
            if b.bz_over() {
                return bz_terminal_score(b, ply);
            }
        } else if b.game_over() {
            return terminal_score(b, ply);
        }

        // Mate distance pruning (Stockfish / Reckless):
        // Even if we mate on the next move, our score is at best MATE - (ply + 1).
        // Even if we are mated right here, our score is at worst -MATE + ply.
        alpha = alpha.max(-MATE + ply as i32);
        beta = beta.min(MATE - (ply as i32 + 1));
        if alpha >= beta {
            return alpha;
        }

        // Transposition table probe. Missing entry at depth >= 4 -> IIR
        // (internal iterative reduction, Moonbird): the node has no cached
        // best move to order by, so search one ply less.
        let alpha_orig = alpha;
        let mut tt_mv: u32 = 0;
        let mut depth = depth;
        if let Some(e) = self.tt.probe(b.hash) {
            tt_mv = e.mv;
            if e.depth as u32 >= depth {
                let s = score_from_tt(e.score, ply);
                match e.flag {
                    FLAG_EXACT => {
                        if s.abs() < MATE_BOUND || (MATE - s.abs()) as u32 >= depth {
                            self.heuristic_used = true;
                        }
                        return s;
                    }
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
                    if s.abs() < MATE_BOUND || (MATE - s.abs()) as u32 >= depth {
                        self.heuristic_used = true;
                    }
                    return s;
                }
            }
        } else if depth >= 4 {
            depth -= 1;
        }

        // Lazy incremental sancta: root refreshed once in search(); every
        // child acc = parent acc + bitboard diffs at make() time (below).
        // OFF = single branch to classical, zero-cost.
        // forward_ready(acc, stm) is ALREADY stm-relative (like evaluate) —
        // no negation (that was the 0-50 bug: double-negated white nodes).
        let (raw, node_dis) = match (self.nnue, self.sancta) {
            (None, Some(net)) => (net.forward_ready(&self.stack.acc[ply as usize], b.turn), 0),
            _ if pre != i32::MIN => unpack_eval_dis(pre),
            _ => self.leaf_eval_with_dis(b),
        };
        let static_eval = self.corrected(b, raw);

        // Reverse futility pruning (autaxx, exact form): standing pat is so
        // far below alpha that even a generous error margin cannot recover
        // -> fail low. Safe in Ataxx because the eval is capture-risk-aware.
        // Disabled when alpha/beta are mate scores so centipawn evals never prune mate lines.
        if null_allowed
            && ply > 0
            && depth <= 4
            && alpha.abs() < MATE_BOUND
            && beta.abs() < MATE_BOUND
            && static_eval
                + RFP_MARGINS[(depth.max(1) - 1) as usize] * tune().rfp_scale / 100
                + ((node_dis * tune().dis_rfp_num) >> 7)
                < alpha
        {
            self.heuristic_used = true;
            return alpha;
        }

        // Null-move pruning (autaxx + Moonbird hybrid): pass is a real move
        // in Ataxx and there is no zugzwang to speak of. Gates: not right
        // after another null, depth > 2, enough clone targets (mobility),
        // board not too full (forced-pass danger zone). R = 3.
        // (exp/ab B: R = 2 pooled 108-84-8, straddles, parked for 3rd leg.)
        if null_allowed && !BZ && depth > 2 && beta.abs() < MATE_BOUND {
            let clone_targets = (dist_union(b.occ[b.turn as usize], 1) & b.empty()).count_ones();
            // fill < 0.54 as integer: pieces*50 < 26*49 (no per-node float div).
            if clone_targets >= 11
                && (b.piece_cnt[0] as u32 + b.piece_cnt[1] as u32) * 50 < 26 * SQUARES as u32
            {
                let child = b.make_pass();
                self.stack.path_hash[ply as usize + 1] = child.hash;
                self.stack.rep_base[ply as usize + 1] = self.stack.rep_base[ply as usize];
                // NMP pass flips stm: sancta acc is color-relative so it
                // stays valid — just forward the other perspective.
                // (pass: no stones change, no diff update needed.)
                let score =
                    -self.negamax::<BZ>(&child, -beta, -beta + 1, depth - 3, ply + 1, false);
                if self.stop {
                    return 0;
                }
                if score >= beta {
                    self.heuristic_used = true;
                    return if score >= MATE_BOUND { beta } else { score };
                }
            }
        }

        if depth == 0 {
            self.heuristic_used = true;
            return static_eval;
        }

        {
            let list = &mut self.stack.lists[ply as usize];
            score_into(b, list, tt_mv, ply, &self.killers, &self.history);
            // Fallthrough terminal: legal_moves_dedup_into tracks dedup
            // multiplicities, so len>0 always — but double-pass positions
            // have no children to search: score the pass-through.
            if list.is_empty() {
                self.heuristic_used = true;
                let raw = match (self.nnue, self.sancta) {
                    (None, Some(net)) => net.forward_ready(&self.stack.acc[ply as usize], b.turn),
                    _ => self.leaf_eval(b),
                };
                let static_eval = self.corrected(b, raw);
                self.finish_node(
                    b,
                    alpha_orig,
                    beta,
                    static_eval,
                    0xFFFF,
                    depth,
                    ply,
                    static_eval,
                );
                return static_eval;
            }
        }

        if EVORD_W > 0 && depth >= EVORD_MIN_DEPTH && self.sancta.is_none() {
            let n = self.stack.lists[ply as usize].len();
            if n > 1 {
                for i in 0..n {
                    let (m, sc) = {
                        let l = &self.stack.lists[ply as usize];
                        (l.move_at(i), l.score_at(i))
                    };
                    if sc == ORD_TT {
                        continue;
                    }
                    let child = b.make(m);
                    let add = if EVORD_CACHE {
                        let (k, v) = self.ecache[(child.hash as usize) & ((1 << ECACHE_BITS) - 1)];
                        if k == child.hash && v != i32::MIN {
                            let (ce, _) = unpack_eval_dis(v);
                            (-ce).saturating_sub(static_eval)
                        } else {
                            0
                        }
                    } else {
                        -self.leaf_eval(&child)
                    };
                    self.stack.lists[ply as usize]
                        .set_score(i, sc.saturating_add(add.saturating_mul(EVORD_W)));
                }
            }
        }

        let mut best = i32::MIN;
        let mut best_mv: u32 = 0xFFFF;
        let mut move_idx = 0usize;
        let mut rep_skipped = false;
        // futility small net: the parent's accumulator, built at the first child that needs it (D110)
        let mut sn_parent: Option<[f32; 64]> = None;
        let mut lmr_cnt = 0i32;
        loop {
            let m = {
                let list = &mut self.stack.lists[ply as usize];
                if list.is_empty() {
                    break;
                }
                list.pop_best()
            };

            // Moonbird-style selectivity (exp/search-moonbird A): late moves get a gentle LMR, every move after the
            // first is reduced by how far its child's STATIC eval falls short of alpha, and a move reduced to depth
            // <= 0 is not searched at all. First move: full window, full depth (always searched, so best is set).
            let child = b.make(m);
            if m.is_clone() {
                self.stack.path_hash[ply as usize + 1] = child.hash;
                self.stack.rep_base[ply as usize + 1] = ply + 1;
            } else {
                let base = self.stack.rep_base[ply as usize] as usize;
                let mut is_rep = false;
                if (ply as usize) >= base + 1 {
                    let mut i = ply as usize - 1;
                    loop {
                        if self.stack.path_hash[i] == child.hash {
                            is_rep = true;
                            break;
                        }
                        if i < base + 2 {
                            break;
                        }
                        i -= 2;
                    }
                }
                if is_rep {
                    rep_skipped = true;
                    continue;
                }
                self.stack.path_hash[ply as usize + 1] = child.hash;
                self.stack.rep_base[ply as usize + 1] = base as u32;
            }
            if let Some(net) = self.sancta {
                let (a, z) = self.stack.acc.split_at_mut(ply as usize + 1);
                net.update_child(&a[ply as usize], &mut z[0], b, &child);
            }
            let mut new_depth = depth as i32;
            if move_idx >= tune().lmr_min as usize {
                let base_red = (depth as i32) - (depth as i32 * tune().lmr_half / (tune().lmr_half + lmr_cnt));
                let dis_red_discount = (node_dis * tune().dis_lmr_num) >> 8;
                let red = (base_red - dis_red_discount).max(0);
                new_depth = (depth as i32) - red;
                lmr_cnt += 1;
            }
            let mut child_eval = i32::MIN;
            if move_idx >= MB_FUT_MIN_IDX
                && new_depth > 0
                && self.sancta.is_none()
                && alpha.abs() < MATE_BOUND
            {
                // child eval is from the child's side to move = minus our static value of the move
                if FUT_CHEAP_M > 0 && self.fut_sn && self.nnue.is_some() {
                    let slot = (child.hash as usize) & ((1 << ECACHE_BITS) - 1);
                    if !(self.ecache[slot].0 == child.hash && self.ecache[slot].1 != i32::MIN) {
                        let hand = evaluate(&child, None, false);
                        let (bt, ct) = (b.turn as usize, child.turn as usize);
                        let sn = small_net();
                        let pa = *sn_parent
                            .get_or_insert_with(|| sn.acc(b.occ[bt], b.occ[1 - bt], b.blockers));
                        let ca = sn.child_acc(
                            &pa,
                            b.occ[bt],
                            b.occ[1 - bt],
                            child.occ[ct],
                            child.occ[1 - ct],
                        );
                        let lo = sn.bound_acc(&ca, child.empty().count_ones(), static_eval, hand)
                            - FUT_SN_M;
                        if new_depth - (alpha.saturating_add(lo) / tune().fut_div).clamp(0, 64) <= 0
                            && best > -MATE_BOUND
                        {
                            self.heuristic_used = true;
                            move_idx += 1;
                            continue;
                        }
                    }
                }
                let (ce, child_dis) = self.leaf_eval_with_dis(&child);
                child_eval = pack_eval_dis(ce, child_dis);
                let vol_credit = (child_dis * tune().dis_fut_num) >> 7;
                let shortfall = alpha.saturating_add(ce) - vol_credit;
                new_depth -= (shortfall / tune().fut_div).clamp(0, 64);
                if child_dis > tune().dis_vol_thresh && new_depth < depth as i32 {
                    new_depth += 1;
                }
            }
            // When all moves searched so far at this node lose by forced mate (or when searching
            // for a shorter mate), never skip remaining legal moves without searching them.
            if best <= -MATE_BOUND || alpha >= MATE_BOUND {
                new_depth = new_depth.max(1);
            }
            if new_depth <= 0 {
                self.heuristic_used = true;
                move_idx += 1;
                continue;
            }
            let mut score;
            if move_idx == 0 {
                score = -self.negamax::<BZ>(&child, -beta, -alpha, depth - 1, ply + 1, true);
                if self.stop {
                    return 0;
                }
            } else {
                self.stack.pre_eval[ply as usize + 1] = child_eval;
                score = -self.negamax::<BZ>(
                    &child,
                    -alpha - 1,
                    -alpha,
                    (new_depth - 1) as u32,
                    ply + 1,
                    true,
                );
                if self.stop {
                    return 0;
                }
                if score > alpha {
                    // re-search: full window, full depth
                    self.stack.pre_eval[ply as usize + 1] = child_eval;
                    score = -self.negamax::<BZ>(&child, -beta, -alpha, depth - 1, ply + 1, true);
                    if self.stop {
                        return 0;
                    }
                }
            }
            move_idx += 1;
            if score <= alpha {
                self.hist_add(b.turn as usize, m, -((depth * depth) as i32));
            }

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
        if best == i32::MIN {
            // Every legal move from this node repeated an ancestor position (CIRCULAR_MOVE forfeit).
            return -MATE + ply as i32;
        }
        if best > alpha_orig && best_mv != 0xFFFF {
            self.hist_add(
                b.turn as usize,
                Move::from_u32(best_mv),
                HIST_SCALE * (depth * depth) as i32,
            );
        }
        if !rep_skipped {
            self.finish_node(b, alpha_orig, beta, best, best_mv, depth, ply, raw);
        }
        best
    }

    /// Moonbird history update: clamp to +-HIST_MAX, shrink the colour's table /1.25 once the cap is hit.
    #[inline]
    fn hist_add(&mut self, stm: usize, m: Move, add: i32) {
        let (f, t) = hist_key(m);
        let e = &mut self.history[stm][f][t];
        *e = (*e + add).clamp(-HIST_MAX, HIST_MAX);
        if e.abs() >= HIST_MAX {
            for row in self.history[stm].iter_mut() {
                for h in row.iter_mut() {
                    *h = *h * 4 / 5;
                }
            }
        }
    }

    /// Killer bookkeeping on a beta cutoff (history: hist_add at the node's end, exp/search-histmb).
    #[inline]
    fn on_cutoff(&mut self, _b: &Board, m: Move, _depth: u32, ply: u32) {
        let packed = m.to_u32();
        let k = &mut self.killers[ply as usize];
        if k.0 != packed {
            k.1 = k.0;
            k.0 = packed;
        }
    }

    /// TT store at the end of a node (skipped when the clock cut it short).
    /// Also feeds the correction-history EMA: (search - static) per context.
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
        static_eval: i32,
    ) {
        if self.stop {
            return;
        }
        if let Some(c) = &mut self.corr {
            c.update(b, best, static_eval, depth);
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
        self.search_gen::<false>(b, limits, &[])
    }

    /// Same search under the Botzone judge's rules (no passes; either side
    /// stuck ends the game, mobile side absorbs the empties; NMP off).
    pub fn search_bz(&mut self, b: &Board, limits: &SearchLimits) -> SearchResult {
        self.search_gen::<true>(b, limits, &[])
    }

    /// Botzone search excluding repetition-banned root moves up front so the full
    /// search budget is spent evaluating allowed root moves at full depth.
    pub fn search_bz_excluding(
        &mut self,
        b: &Board,
        limits: &SearchLimits,
        banned_root: &[Move],
    ) -> SearchResult {
        self.search_gen::<true>(b, limits, banned_root)
    }

    fn search_gen<const BZ: bool>(
        &mut self,
        b: &Board,
        limits: &SearchLimits,
        banned_root: &[Move],
    ) -> SearchResult {
        self.start_ms = (self.time_src)();
        self.nodes = 0;
        self.stop = false;
        self.stop_reason = StopReason::Completed;
        self.node_limit = limits.max_nodes;
        self.fut_sn = FUT_SN_ON.load(std::sync::atomic::Ordering::Relaxed);
        let safety_margin_ms = 10.0;
        self.deadline_ms = limits
            .time
            .map(|t| self.start_ms + (t.as_secs_f64() * 1000.0 - safety_margin_ms).max(1.0));

        let mut result = SearchResult::default();
        // Instant terminal fast path (Botzone rules): a full board or a stuck
        // side (including a stuck OPPONENT, which movegen alone would miss)
        // ends the game on the spot: zero nodes searched.
        if BZ && b.bz_over() {
            result.score = bz_terminal_score(b, 0);
            result.elapsed = self.elapsed();
            result.nodes = self.nodes;
            result.stopped_by = Some(StopReason::Completed);
            return result;
        }
        self.tt.step_age();
        // Age the history table so old cutoffs don't dominate forever.
        for side in self.history.iter_mut() {
            for row in side.iter_mut() {
                for h in row.iter_mut() {
                    *h = (*h * 3) / 4;
                }
            }
        }
        let root_tt = self.tt.probe(b.hash);
        let tt_mv = root_tt.map(|e| e.mv).unwrap_or(0);
        {
            // Root ordering: score once, drain by pop_best into a fixed
            // array (best-first). Each iteration reorders the array.
            let (kill, hist) = (&self.killers, &self.history);
            let root = &mut self.stack.lists[0];
            score_into(b, root, tt_mv, 0, kill, hist);
        }
        let n_root = self.stack.lists[0].len();
        if n_root == 0 {
            if BZ {
                // Already game-over on-platform: score the filled-count terminal.
                result.score = bz_terminal_score(b, 0);
            } else if !b.game_over() {
                // Pass node: give the caller a meaningful score through the pass.
                result.score = -self.negamax::<BZ>(&b.make_pass(), -INF, INF, 1, 1, true);
            }
            result.elapsed = self.elapsed();
            result.nodes = self.nodes;
            return result; // caller must handle pass / game over
        }
        // Root best-so-far: drained best-first into a fixed array.
        // Root acc refreshed once (sancta lazy path: children diff from it).
        let mut root_moves = [Move::PASS; 256];
        let n_root = {
            let root = &mut self.stack.lists[0];
            let mut all_moves = [Move::PASS; 256];
            let mut kept = 0usize;
            for slot in all_moves.iter_mut().take(n_root) {
                let m = root.pop_best();
                *slot = m;
                if !banned_root.contains(&m) {
                    root_moves[kept] = m;
                    kept += 1;
                }
            }
            if kept > 0 {
                kept
            } else {
                root_moves[..n_root].copy_from_slice(&all_moves[..n_root]);
                n_root
            }
        };
        if let Some(net) = self.sancta {
            net.refresh_root(b, &mut self.stack.acc[0]);
        }
        // Initial fallback move: prioritize the proven TT move if legal, else first root move.
        let tt_legal_mv = if tt_mv != 0 {
            (0..n_root)
                .find(|&i| root_moves[i].to_u32() == tt_mv)
                .map(|i| root_moves[i])
        } else {
            None
        };
        result.best = tt_legal_mv.or(Some(root_moves[0]));
        if let Some(entry) = root_tt {
            if entry.score.abs() < MATE_BOUND {
                result.score = entry.score;
            }
        }

        let budget_ms = limits
            .time
            .map(|t| t.as_secs_f64() * 1000.0)
            .unwrap_or(f64::INFINITY);
        // Iteration cost prediction: project the next iteration as the max
        // of the last two completed iterations.
        let mut it_ms: [f64; 2] = [0.0, 0.0];
        let mut has_completed_iter = false;

        // Iterative deepening MUST always start at depth 1.
        // Skipping shallow iterations (even if TT has a deep entry) is fatal under hard time
        // limits (e.g. Botzone): if the high-depth iteration is cut short by the deadline before
        // completing, no iteration ever finishes, leaving depth at 0 and blundering with an
        // unverified move. Shallow iterations take <0.1% of search time and guarantee that a
        // verified completed iteration always exists.
        for depth in 1..=limits.max_depth.max(1) {
            let iter_start = self.elapsed_ms();
            self.heuristic_used = false;
            let mut alpha = -INF;
            let mut best_this = None;
            let mut best_score = i32::MIN;
            // Clean-main ID: FULL window every iteration. No aspiration.
            // (Aspiration reverted: narrowed windows changed PVS cutoffs at
            // fixed node budgets -> bench d8 9.4k=>17k, QB5k 45-53-2.-28.)
            self.stack.path_hash[0] = b.hash;
            self.stack.rep_base[0] = 0;
            for i in 0..n_root {
                let m = root_moves[i];
                let child = b.make(m);
                self.stack.path_hash[1] = child.hash;
                self.stack.rep_base[1] = if m.is_clone() { 1 } else { 0 };
                // Root children: acc[1] = acc[0] + diffs (sancta lazy path).
                if let Some(net) = self.sancta {
                    let (a, z) = self.stack.acc.split_at_mut(1);
                    net.update_child(&a[0], &mut z[0], b, &child);
                }
                let score =
                    -self.negamax::<BZ>(&child, -INF, -alpha, u32::from(depth - 1), 1, true);
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
            // Partial-iteration adoption (Stockfish / Reckless):
            // - A completed iteration is adopted unless it forgot a shorter proven winning mate
            //   from a previous completed iteration (`forgotten_mate`).
            // - An aborted partial iteration (`self.stop`) is adopted only if no full iteration
            //   has completed yet, OR if a searched root move strictly beat the previous
            //   completed iteration's score without walking into a forced loss (`best_score > -MATE_BOUND`).
            if let Some(m) = best_this {
                let forgotten_mate = has_completed_iter
                    && result.score >= MATE_BOUND
                    && best_score < result.score;
                let adopt = if !self.stop {
                    !forgotten_mate
                } else {
                    !has_completed_iter
                        || (best_score > result.score
                            && best_score > -MATE_BOUND
                            && !forgotten_mate)
                };
                if adopt {
                    result.best = Some(m);
                    if best_score != i32::MIN {
                        result.score = best_score;
                    }
                    result.depth = u32::from(depth);
                }
                // Move best move to the front for the next iteration.
                if !self.stop {
                    has_completed_iter = true;
                    if !forgotten_mate {
                        if let Some(pos) = (0..n_root).find(|&i| root_moves[i] == m) {
                            root_moves[pos] = root_moves[0];
                            root_moves[0] = m;
                        }
                    }
                }
            }
            if self.stop {
                break;
            }
            // Stop iterative deepening once the shortest forced win is proven within the current
            // depth horizon, or once the entire game tree is resolved to exact terminal states
            // (`!self.heuristic_used`). When losing (`best_score <= -MATE_BOUND`) and heuristic
            // leaves remain, keep searching deeper to find the longest sequence delaying the loss.
            if (best_score >= MATE_BOUND && u32::from(depth) >= (MATE - best_score) as u32)
                || !self.heuristic_used
            {
                self.stop_reason = StopReason::Completed;
                break;
            }
            // exp/full-time: no "next iteration would overrun" early stop any more. A partial iteration is adopted
            // (above), so searching on to the deadline is never wasted; the deadline check stops it.
            it_ms[1] = it_ms[0];
            it_ms[0] = self.elapsed_ms() - iter_start;
            let proj_ms = it_ms[0].max(it_ms[1]);
            // Iteration cost prediction: project if the next iteration would overrun the budget.
            // On Botzone, overrunning by even 1 ms deducts time from future moves and risks time forfeit.
            // We stop cleanly at the completed depth. Only when defending a forced loss do we use all
            // remaining time to find delay paths.
            if result.score > -MATE_BOUND && self.elapsed_ms() + proj_ms > budget_ms {
                self.stop_reason = StopReason::Time;
                break;
            }
        }

        if self.stop {
            result.stopped_by = Some(self.stop_reason);
        } else if result.depth >= u32::from(limits.max_depth) {
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
    Searcher::with_main_network().search(b, limits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn no_time(depth: u8) -> SearchLimits {
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
    fn lazy_ordering_finds_valid_move() {
        // Lazy selection must return some legal move from the start.
        let b = Board::start();
        let mut s = Searcher::with_tt_bits(16);
        let r = s.search(&b, &no_time(5));
        let m = r.best.expect("lazy ordering must find a move");
        assert!(b.legal_moves().contains(&m), "{m:?} illegal");
    }

    // ---- Botzone-rules search (ported from the Botzone bundle) ----

    #[test]
    fn bz_full_board_is_instant_terminal() {
        let rows: Vec<String> = (0..7)
            .map(|r| {
                (0..7)
                    .map(|f| if (r + f) % 2 == 0 { 'x' } else { 'o' })
                    .collect()
            })
            .collect();
        let s = rows.join(
            "
",
        ) + " b";
        let b = Board::from_str(&s).unwrap();
        assert_eq!(b.counts(), (25, 24));
        assert!(b.bz_over());
        assert_eq!(b.bz_counts(), (25, 24)); // both stuck -> raw counts
        let mut se = Searcher::with_tt_bits(16);
        let r = se.search_bz(&b, &no_time(6));
        assert_eq!(r.nodes, 0, "full board must search nothing");
        assert_eq!(r.score, MATE, "black up 25-24");
        assert!(r.best.is_none());
    }

    #[test]
    fn bz_stuck_opponent_is_instant_terminal() {
        // White boxed into the corner (no legal reply), black mobile: game over,
        // black gets every empty square: 48v1 -> MATE, zero nodes.
        let s = "oxx....
xxx....
xxx....
.......
.......
.......
......x b";
        let b = Board::from_str(s).unwrap();
        assert!(b.has_moves(0));
        assert!(!b.has_moves(1));
        assert!(b.bz_over());
        assert_eq!(b.bz_counts(), (48, 1));
        let mut se = Searcher::with_tt_bits(16);
        let r = se.search_bz(&b, &no_time(6));
        assert_eq!(r.nodes, 0, "finished game must search nothing");
        assert_eq!(r.score, MATE);
        assert!(r.best.is_none());
    }

    #[test]
    fn bz_stuck_side_to_move_loses_by_count() {
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
        let mut se = Searcher::with_tt_bits(16);
        let r = se.search_bz(&b, &no_time(4));
        assert!(r.best.is_none());
        assert!(r.score < 0, "buried black loses, score {}", r.score);
    }

    #[test]
    fn bz_rules_still_terminates_from_start() {
        let b = Board::botzone_start();
        assert_eq!(b.counts(), (2, 2));
        assert_eq!(
            b.legal_moves().len(),
            16 * 1 + 0,
            "2 stones x (3 clones + 5 jumps)"
        );
        let mut se = Searcher::with_tt_bits(16);
        let r = se.search_bz(&b, &no_time(4));
        assert!(r.best.is_some());
        assert!(r.nodes > 0);
    }

    #[test]
    fn external_stop_flag_cuts_the_search() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let flag = Arc::new(AtomicBool::new(false));
        let b = Board::botzone_start();
        let mut se = Searcher::with_tt_bits(16);
        se.set_stop_flag(Some(flag.clone()));
        // unflagged: depth-8 search runs well past 2048 nodes
        let free = se.search(&b, &no_time(10));
        // (was depth 8 > 20000: the engine got leaner since; the test only needs a search far longer than one poll)
        assert!(free.nodes > 8_000, "baseline nodes {}", free.nodes);
        // flagged from the start: stops at the first poll (<= 2048 nodes) but still returns a move
        flag.store(true, Ordering::Relaxed);
        se.reset_for_new_game();
        let cut = se.search(&b, &no_time(30));
        assert!(cut.nodes <= 4096, "stopped late: {} nodes", cut.nodes);
        assert!(cut.best.is_some());
    }

    #[test]
    fn winning_mate_stops_at_mate_depth_not_max_depth() {
        // Black can wipe out white in 1 ply with a clone next to the lone white stone.
        // Even with max_depth = u8::MAX (255), iterative deepening must stop at depth 1.
        let board = ".......
                     .......
                     .......
                     .....o.
                     ......x
                     .......
                     ....... b";
        let b = Board::from_str(board).unwrap();
        let mut se = Searcher::with_tt_bits(16);
        let r = se.search_bz(&b, &no_time(u8::MAX));
        assert_eq!(r.score, MATE - 1, "must find mate in 1 ply");
        assert_eq!(r.depth, 1, "must stop at depth 1 instead of looping to 255, got {}", r.depth);
        assert_eq!(r.stopped_by, Some(StopReason::Completed));
    }

    #[test]
    fn losing_mate_picks_longest_delay_and_does_not_spin_to_255() {
        // White dominates the board; only two empty squares remain (a7 and g7).
        // Black has one stone at b7 and one at g5:
        // - If black clones to a7 (filling a7), white immediately fills g7 on ply 2 -> mate in 2 (-MATE + 2).
        // - If black jumps b7->a7 (vacating b7), white still has to fill more squares; or we test
        //   two concrete legal moves with different terminal lengths.
        // Here: 1 empty square at a7 (sq 0), except black can either clone to a7 (fills board in 1 ply -> -MATE + 1)
        // or jump c7->a7 (leaves c7 empty, so white must move on ply 2 -> -MATE + 2).
        let board = ".oxoooo
                     ooooooo
                     ooooooo
                     ooooooo
                     ooooooo
                     ooooooo
                     ooooooo b";
        let b = Board::from_str(board).unwrap();
        let mut se = Searcher::with_tt_bits(16);
        let r = se.search_bz(&b, &no_time(u8::MAX));
        let mv = r.best.expect("must return a delaying move");
        assert!(!mv.is_clone(), "black must jump c7->a7 to delay loss to 2 plies instead of cloning (1 ply), got {mv:?}");
        assert_eq!(r.score, -MATE + 2, "longest delay is 2 plies (-MATE + 2), got {}", r.score);
        assert!(r.depth <= 3, "exhausted terminal tree must stop without spinning to 255, got depth {}", r.depth);
    }

    #[test]
    fn in_tree_circular_repetition_is_pruned() {
        // Two stones separated by blockers so no captures or immediate wipeouts occur.
        // Jump moves can shuffle, but in-tree circular repetitions are pruned so search completes cleanly.
        let board = "x.##.o.
                     .......
                     .......
                     .......
                     .......
                     .......
                     ....... b";
        let b = Board::from_str(board).unwrap();
        let mut se = Searcher::with_tt_bits(16);
        let r = se.search_bz(&b, &no_time(6));
        assert!(r.best.is_some());
        assert_eq!(r.depth, 6);
        assert_eq!(r.stopped_by, Some(StopReason::Completed));
    }

    #[test]
    fn tt_aging_and_fail_low_preservation() {
        let mut tt = Tt::new(10);
        let key = 123456789u64;
        let mv_good = 42u32;

        // 1. Store an exact bound with a good move
        tt.store(key, mv_good, 150, 6, FLAG_EXACT);
        assert_eq!(tt.probe(key).unwrap().mv, mv_good);
        assert_eq!(tt.probe(key).unwrap().score, 150);

        // 2. A deeper search on the same position fails low (FLAG_UPPER)
        tt.store(key, 0xFFFF, 50, 8, FLAG_UPPER);
        // The good move must be preserved!
        assert_eq!(tt.probe(key).unwrap().mv, mv_good, "refutation move must be preserved on fail-low");
        assert_eq!(tt.probe(key).unwrap().score, 50);

        // 3. Test age replacement: position from old turn gets evicted by fresh search
        tt.step_age();
        tt.step_age();
        let other_key = key ^ (1 << 10); // same hash slot, different key
        tt.store(other_key, 99, 200, 3, FLAG_EXACT);
        assert_eq!(tt.probe(other_key).unwrap().key, other_key);
    }

    #[test]
    fn warm_tt_reduces_search_nodes() {
        let b = Board::start();
        let mut se = Searcher::with_tt_bits(16);
        // Initial search to depth 6
        let r1 = se.search(&b, &no_time(6));
        assert_eq!(r1.depth, 6);
        let nodes1 = r1.nodes;

        // In a persistent session, searching the same root position again
        // starts from depth 1 but TT cutoffs massively prune the tree.
        let r2 = se.search(&b, &no_time(6));
        assert_eq!(r2.depth, 6);
        assert!(r2.nodes < nodes1, "warm start nodes ({}) must be fewer than cold start ({})", r2.nodes, nodes1);
    }

    #[test]
    fn pack_and_unpack_eval_dis_roundtrips_without_sentinel_collision() {
        for eval_cp in [-30000, -12345, -1, 0, 1, 420, 30000] {
            for dis_cp in [0, 1, 75, 450, 2500, 30000] {
                let packed = pack_eval_dis(eval_cp, dis_cp);
                assert_ne!(packed, i32::MIN, "packed value must never collide with i32::MIN sentinel");
                let (e, d) = unpack_eval_dis(packed);
                assert_eq!(e, eval_cp.clamp(-16384, 16383));
                assert_eq!(d, dis_cp.clamp(0, 16383));
            }
        }
    }
}
