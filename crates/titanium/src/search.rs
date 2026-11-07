//! Evaluation and iterative-deepening alpha-beta (negamax) search.
//!
//! Search order: TT probe -> eval cache (depth 0) -> move scoring
//! (TT move > killers > captures*W + history) -> lazy selection -> alpha-beta.

/// exp/nnue-speed: static-eval cache entries (2^16 x 16 B = 1 MiB per searcher).
/// eval cache size, log2 entries (16 B each). D575: 2^16 is 100% full and thrashing at 500k nodes per move;
/// `set_ecache_bits` (titanium-cli --ecache-bits N) changes it for searchers created afterwards.
static ECACHE_BITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(16);

/// D585: eval-cache layout. 0 = the old direct-mapped table of (u64 key, i32) entries (default, unchanged);
/// 1 = cache-line buckets: 8 entries of 8 bytes per 64-byte line (32-bit check = 26 hash bits + 6 bits of empties,
/// 32-bit packed eval), one memory fetch per probe, victim = empty > dead (more empties than the search root:
/// fewer stones, unreachable) > least recently used (hits move to the front), plus a prefetch of the child's
/// eval-cache line and TT slot as soon as the move is made.
static ECACHE_MODE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_ecache_mode(m: usize) {
    ECACHE_MODE.store(m.min(1), std::sync::atomic::Ordering::Relaxed);
}

const EC_EMPTY: u64 = 0x8000_0000; // low half = i32::MIN (never a packed eval), check 0

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct EcBucket([u64; 8]);

/// 32-bit check = 25 hash bits | 1 importance bit | 6 bits of empties (the slot index already carries the low hash
/// bits, so these 25 bits are new information: a false match needs 25 equal bits AND equal empties)
#[inline(always)]
fn ec_check(hash: u64, empties: u32, imp: u32) -> u32 {
    (((hash >> 39) as u32) << 7) | ((imp & 1) << 6) | (empties & 63)
}
const EC_IMP_BIT: u32 = 1 << 6;

pub fn set_ecache_bits(bits: usize) {
    ECACHE_BITS.store(bits.clamp(10, 26), std::sync::atomic::Ordering::Relaxed);
}

/// exp/tree-shape: trace hook, expands to nothing unless built with `--features trace`.
macro_rules! trc {
    ($s:ident, $t:ident => $e:expr) => {
        #[cfg(feature = "trace")]
        {
            if let Some($t) = $s.trace.as_mut() {
                $e;
            }
        }
    };
}

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
    /// D590: how the node at this ply was reached (3 first move, 2 top-12, 1 later, 0 null-move probe): TT worth
    imp: Box<[u8; MAX_PLY]>,
}

impl MoveStack {
    fn new() -> MoveStack {
        MoveStack {
            lists: Box::new([MoveList::new(); MAX_PLY]),
            acc: Box::new([[[0; 64]; 2]; MAX_PLY]),
            pre_eval: Box::new([i32::MIN; MAX_PLY]),
            path_hash: Box::new([0u64; MAX_PLY]),
            rep_base: Box::new([0u32; MAX_PLY]),
            imp: Box::new([3u8; MAX_PLY]),
        }
    }
}

/// Material weight per piece (eval = piece diff * MATERIAL).
const MATERIAL: i32 = 100;
/// Tempo: flat bonus for the side to move. autaxx uses 1.5-2 stones worth.
const TEMPO: i32 = 150;
/// Reverse futility pruning margins, index = depth-1 (autaxx, stone = 100).
// (const RFP_MARGINS: [i32; 4] = [257, 347, 478, 774]; -> KNOB_DEFAULT, per NNUE bucket)
/// exp/search-moonbird A: Moonbird 1.1.0 reduction scheme (searchParams.nim) in place of the universal LMR.
/// LMR from the 7th move: new_depth = depth * HALF / (HALF + n_reduced_so_far).
/// (now per phase: Knobs::lmr_min_idx / lmr_half, defaults 6 / 47)
/// Futility reduction from the 2nd move: one ply per FUT_DIV cp the child's static eval falls short of alpha
/// (Moonbird futilityReductionDiv 35 in its units; x1.62 to Titanium cp: 568/ln10 vs 400 per logit). Reduce only.
// (const MB_FUT_MIN_IDX: usize = 1; -> KNOB_DEFAULT, per NNUE bucket)
// (const MB_FUT_DIV: i32 = 57; -> KNOB_DEFAULT, per NNUE bucket)
/// Volatility scaling numerator (scaled by 128) for Reverse Futility Pruning: +25% of predicted disagreement.
// (const DIS_RFP_NUM: i32 = 32; -> KNOB_DEFAULT, per NNUE bucket)
/// Volatility credit numerator (scaled by 128) for Child Futility Pruning: 37.5% credit.
// (const DIS_FUT_NUM: i32 = 24; -> KNOB_DEFAULT, per NNUE bucket)
/// ProbCut (gated +20 Elo @100 ms vs main, user-approved 2026-10-31): min depth, depth reduction, margin cp,
/// disagreement-head credit (/128). A shallow null-window search at pc_beta; if it clears pc_beta the node cuts.
// (const PC_D: u32 = 6; -> KNOB_DEFAULT, per NNUE bucket)
// (const PC_R: u32 = 3; -> KNOB_DEFAULT, per NNUE bucket)
// (const PC_T: i32 = 140; -> KNOB_DEFAULT, per NNUE bucket)
// (const PC_K: i32 = 32; -> KNOB_DEFAULT, per NNUE bucket)

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

pub fn fut_sn_on() -> bool {
    FUT_SN_ON.load(std::sync::atomic::Ordering::Relaxed)
}

/// (FUT_CHEAP_M, FUT_SN_M, EVORD_W, EVORD_MIN_DEPTH, EVORD_CACHE as 0/1): the cargo-build-time knobs, for `info` and the defaults test.
pub fn build_knobs() -> (i32, i32, i32, u32, u32) {
    (FUT_CHEAP_M, FUT_SN_M, EVORD_W, EVORD_MIN_DEPTH, EVORD_CACHE as u32)
}

/// The shipped (strongest, gated) values of the build-time knobs.
pub const DEFAULT_BUILD_KNOBS: (i32, i32, i32, u32, u32) = (350, 0, 0, 1, 1);

/// exp/tree-shape (D557/D558, user order): EVERY search knob is a 16-entry table indexed by the NNUE bucket of the
/// node, bucket = (empties * 16 / 50).min(15) (exactly nnue.rs: 14 at the start position, 0 on a full board), so the
/// search is shaped separately for each of the 16 game phases the net uses. Defaults = the old constants (the default
/// build is bit-identical). `titanium-cli --knobs FILE` overrides: lines `name = v` (all buckets) or
/// `name = v0 v1 ... v15` (bucket 0 = full board ... 15); `#` comments. The searcher copies the table per search.
pub const NK: usize = 25;
pub const K_RFP1: usize = 0; // reverse futility margin depth 1..4 (257 347 478 774)
pub const K_RFP2: usize = 1;
pub const K_RFP3: usize = 2;
pub const K_RFP4: usize = 3;
pub const K_RFP_DIS: usize = 4; // RFP disagreement credit /128 (32)
pub const K_NMP_MARGIN: usize = 5; // null move only when static >= beta + m (-1000000 = no condition)
pub const K_NMP_R: usize = 6; // null move reduction (3)
pub const K_NMP_CLONES: usize = 7; // min clone targets for the null move (11)
pub const K_PC_D: usize = 8; // ProbCut min depth (6)
pub const K_PC_R: usize = 9; // ProbCut reduction (3)
pub const K_PC_T: usize = 10; // ProbCut margin cp (140)
pub const K_PC_K: usize = 11; // ProbCut disagreement credit /128 (32)
pub const K_LMR_MIN: usize = 12; // first move index with LMR (6)
pub const K_LMR_HALF: usize = 13; // LMR half-life (47)
pub const K_FUT_MIN: usize = 14; // first move index with futility reduction (1)
pub const K_FUT_DIV: usize = 15; // cp per ply of futility reduction (57)
pub const K_FUT_DIS: usize = 16; // futility disagreement credit /128 (24)
pub const K_SN_M: usize = 17; // extra cp under the small-net lower bound before a skip (0)
pub const K_IIR: usize = 18; // min depth for IIR without a TT entry (4)
pub const K_KILL_CANON: usize = 19; // 1 = a clone killer matches any clone to the same square (D564; 0 = exact from-to)
pub const K_ORD_AHEAD_CLONE: usize = 20; // ordering points added to clones when static > ord_state_cp (D567; 0)
pub const K_ORD_BEHIND_JUMP: usize = 21; // ordering points added to jumps when static < -ord_state_cp (D567; 0)
pub const K_ORD_STATE_CP: usize = 22; // ahead/behind threshold in cp (100)
pub const K_ORD_KILLER: usize = 24; // D582: ordering points of a killer move (100000 = 1 policy logit)
pub const K_TT_DEAD: usize = 23; // D571 (user): 1 = TT entries with more empties than the root are dead and replaced first; 2 = also no age eviction of live entries (0)
pub const KNOB_NAMES: [&str; NK] = [
    "rfp1", "rfp2", "rfp3", "rfp4", "rfp_dis", "nmp_margin", "nmp_r", "nmp_clones", "pc_d", "pc_r", "pc_t", "pc_k",
    "lmr_min", "lmr_half", "fut_min", "fut_div", "fut_dis", "sn_m", "iir", "kill_canon",
    "ord_ahead_clone", "ord_behind_jump", "ord_state_cp", "tt_dead", "ord_killer",
];
pub const KNOB_DEFAULT: [i32; NK] = [257, 347, 478, 774, 32, -1_000_000, 3, 11, 6, 3, 140, 32, 6, 47, 1, 57, 24, 0, 4, 0, 0, 0, 100, 0, 100_000];
pub const KNOB_RANGE: [(i32, i32); NK] = [
    (0, 5000), (0, 5000), (0, 5000), (0, 5000), (0, 512), (-1_000_000, 1_000_000), (1, 8), (0, 49), (2, 64), (1, 8),
    (0, 2000), (0, 512), (1, 255), (1, 10_000), (1, 255), (1, 10_000), (0, 512), (-2000, 2000), (1, 255), (0, 1),
    (-1_000_000, 1_000_000), (-1_000_000, 1_000_000), (0, 5000), (0, 2), (0, 1_000_000),
];

/// knob table: v[bucket][knob]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knobs {
    pub v: [[i32; NK]; 16],
}

pub const DEFAULT_KNOBS: Knobs = Knobs { v: [KNOB_DEFAULT; 16] };
static KNOBS: std::sync::Mutex<Knobs> = std::sync::Mutex::new(DEFAULT_KNOBS);

/// the NNUE bucket of a position with `empties` empty squares (MUST match nnue.rs forward: empties * 16 / 50)
#[inline(always)]
pub fn knob_bucket(empties: u32) -> usize {
    ((empties as usize) * 16 / 50).min(15)
}

pub fn knobs() -> Knobs {
    *KNOBS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Knobs {
    /// the table as a knob file (only knobs that differ from the default)
    pub fn to_text(&self) -> String {
        let mut o = String::new();
        for k in 0..NK {
            if (0..16).any(|b| self.v[b][k] != KNOB_DEFAULT[k]) {
                let vals: Vec<String> = (0..16).map(|b| self.v[b][k].to_string()).collect();
                o += &format!("{} = {}\n", KNOB_NAMES[k], vals.join(" "));
            }
        }
        o
    }
}

/// parse and install a knob file; returns the installed table
pub fn set_knobs_text(text: &str) -> Result<Knobs, String> {
    let mut k = DEFAULT_KNOBS;
    for (ln, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (name, vals) = line.split_once('=').ok_or(format!("knobs line {}: expected name = values", ln + 1))?;
        let v: Vec<i64> = vals.split_whitespace().map(|x| x.parse::<i64>().map_err(|e| format!("knobs line {}: {e}", ln + 1))).collect::<Result<_, _>>()?;
        if v.len() != 1 && v.len() != 16 {
            return Err(format!("knobs line {}: give 1 value or 16 (one per NNUE bucket 0..15), got {}", ln + 1, v.len()));
        }
        let ki = KNOB_NAMES.iter().position(|n| *n == name.trim()).ok_or(format!("knobs line {}: unknown knob {} (known: {})", ln + 1, name.trim(), KNOB_NAMES.join(" ")))?;
        let (lo, hi) = KNOB_RANGE[ki];
        for b in 0..16 {
            k.v[b][ki] = (if v.len() == 1 { v[0] } else { v[b] }).clamp(lo as i64, hi as i64) as i32;
        }
    }
    *KNOBS.lock().unwrap_or_else(|e| e.into_inner()) = k;
    Ok(k)
}

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
const EVORD_W: i32 = env_usize(option_env!("EVORD_W"), 0) as i32;
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
    kill_canon: bool,
    bias: (i32, i32),
    killer_w: i32,
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
            score += ORD_CLONE_BONUS + bias.0;
        } else {
            score += bias.1;
            // exp/ab O1 (hole-aware jumps): a jump vacates its origin —
            // leaving a connected origin breaks formation, so penalty
            // scales with own stones adjacent to the origin square.
            let conn = (b.occ[b.turn as usize] & RING1[m.from as usize]).count_ones() as i32;
            score -= ORD_JUMP_PEN + conn * ORD_JUMP_CONN;
        }
        // D564: a clone is generated from the LOWEST adjacent own stone, so a killer clone from a sibling may carry
        // another origin; kill_canon matches clones by destination (as hist_key already does)
        let is_killer = packed == k1
            || packed == k2
            || (kill_canon
                && m.is_clone()
                && [k1, k2].iter().any(|&k| {
                    let km = Move::from_u32(k);
                    km.from < 49 && km.to == m.to && km.from != km.to && (RING1[km.from as usize] >> km.to) & 1 != 0
                }));
        if is_killer {
            score += killer_w;
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
    /// D571: empty squares of the stored position (padding byte; empties never increase in Ataxx, so an entry with
    /// more empties than the root can never be reached again). 0 = unknown (never dead).
    empties: u8,
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

/// D590: ranked TT (TT_MODE 1): 16-byte entries, 4 per 64-byte line.
static TT_MODE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_tt_mode(m: usize) {
    TT_MODE.store(m.min(1), std::sync::atomic::Ordering::Relaxed);
}

#[derive(Clone, Copy)]
#[repr(C)]
struct TtE2 {
    chk: u32,
    score: i32,
    mv: u16,
    depth: u8,
    /// flag (2 bits) | importance << 2 (2 bits)
    meta: u8,
    age: u8,
    empties: u8,
    _pad: u16,
}
const _: () = assert!(std::mem::size_of::<TtE2>() == 16);
const TTE2_EMPTY: TtE2 = TtE2 { chk: 0, score: 0, mv: 0, depth: 0, meta: 0, age: 0, empties: 0, _pad: 0 };

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct TtBucket([TtE2; 4]);

/// Fixed-size transposition table, power-of-two entry count. Depth-0 EXACT
/// entries double as the eval cache (Ataxx transpositions repeat a lot).
pub struct Tt {
    buf: Vec<TtEntry>,
    /// D590 ranked buckets (Some = TT_MODE 1; then `buf` is a 1-entry dummy)
    v2: Option<Box<[TtBucket]>>,
    mask: usize,
    pub age: u8,
    /// D571: empties of the current search root (255 = nothing is dead) and the knob mode (0 off, 1, 2)
    pub root_empties: u8,
    pub dead_mode: u8,
}

impl Tt {
    /// 2^bits entries (bits=20 -> 1M entries, 24 MB).
    pub fn new(bits: usize) -> Tt {
        if TT_MODE.load(std::sync::atomic::Ordering::Relaxed) == 1 {
            let nb = (1usize << bits.min(25)) / 4;
            let mut t = Tt::new_v1(0);
            t.v2 = Some(vec![TtBucket([TTE2_EMPTY; 4]); nb].into_boxed_slice());
            t.mask = nb - 1;
            return t;
        }
        Tt::new_v1(bits)
    }

    fn new_v1(bits: usize) -> Tt {
        let size = 1usize << bits.min(24);
        Tt {
            buf: vec![
                TtEntry {
                    key: 0,
                    mv: 0,
                    score: 0,
                    depth: 0,
                    empties: 0,
                    flag: FLAG_EMPTY,
                    age: 0,
                };
                size
            ],
            mask: size - 1,
            v2: None,
            root_empties: 255,
            dead_mode: 0,
            age: 0,
        }
    }

    pub fn clear(&mut self) {
        for e in self.buf.iter_mut() {
            e.flag = FLAG_EMPTY;
        }
        if let Some(v) = self.v2.as_mut() {
            v.fill(TtBucket([TTE2_EMPTY; 4]));
        }
        self.age = 0;
    }

    #[inline(always)]
    fn slot_ptr(&self, key: u64) -> *const i8 {
        if let Some(v) = self.v2.as_ref() {
            return &v[(key as usize) & self.mask] as *const TtBucket as *const i8;
        }
        &self.buf[(key as usize) & self.mask] as *const TtEntry as *const i8
    }

    #[inline]
    pub fn step_age(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    #[inline]
    fn probe(&mut self, key: u64) -> Option<TtEntry> {
        if let Some(v) = self.v2.as_mut() {
            let age = self.age;
            let chk = (key >> 32) as u32;
            for e in v[(key as usize) & self.mask].0.iter_mut() {
                if e.meta & 3 != 0 && e.chk == chk {
                    e.age = age;
                    return Some(TtEntry { key, mv: e.mv as u32, score: e.score, depth: e.depth, flag: e.meta & 3, age, empties: e.empties });
                }
            }
            return None;
        }
        let e = &mut self.buf[(key as usize) & self.mask];
        if e.flag != FLAG_EMPTY && e.key == key {
            e.age = self.age;
            Some(*e)
        } else {
            None
        }
    }

    /// D572: what the table holds relative to a root with `root_empties` empty squares (TT review)
    pub fn census(&self, root_empties: u8) -> String {
        if let Some(v) = self.v2.as_ref() {
            let (mut empty, mut dead, mut live, mut imp) = (0u64, 0u64, 0u64, [0u64; 4]);
            for b in v.iter() {
                for e in b.0.iter() {
                    if e.meta & 3 == 0 {
                        empty += 1;
                    } else if e.empties > root_empties {
                        dead += 1;
                    } else {
                        live += 1;
                        imp[((e.meta >> 2) & 3) as usize] += 1;
                    }
                }
            }
            let n = (v.len() * 4) as f64;
            return format!(
                "tt ranked {} entries (16 B): empty {:.1}% dead {:.1}% live {:.1}% (importance 3:{:.1}% 2:{:.1}% 1:{:.1}% 0:{:.1}%)",
                v.len() * 4, 100.0 * empty as f64 / n, 100.0 * dead as f64 / n, 100.0 * live as f64 / n,
                100.0 * imp[3] as f64 / n, 100.0 * imp[2] as f64 / n, 100.0 * imp[1] as f64 / n, 100.0 * imp[0] as f64 / n
            );
        }
        let (mut empty, mut dead, mut proven) = (0u64, 0u64, 0u64);
        let mut ahead = [0u64; 6]; // live entries by stones above the root: 0, 1, 2, 3-5, 6-10, 11+
        let mut depth = [0u64; 4]; // live entries by depth: 0-2, 3-5, 6-9, 10+
        let mut age = [0u64; 3]; // live entries by age: this search, previous, older
        for e in self.buf.iter() {
            if e.flag == FLAG_EMPTY {
                empty += 1;
                continue;
            }
            if e.empties > root_empties {
                dead += 1;
                continue;
            }
            let d = (root_empties - e.empties) as usize;
            ahead[match d { 0 => 0, 1 => 1, 2 => 2, 3..=5 => 3, 6..=10 => 4, _ => 5 }] += 1;
            depth[match e.depth { 0..=2 => 0, 3..=5 => 1, 6..=9 => 2, _ => 3 }] += 1;
            age[(self.age.wrapping_sub(e.age) as usize).min(2)] += 1;
            if e.score.abs() >= MATE_BOUND {
                proven += 1;
            }
        }
        let n = self.buf.len() as f64;
        let pc = |x: u64| 100.0 * x as f64 / n;
        format!(
            "tt {} entries: empty {:.1}% DEAD(fewer stones than root) {:.1}% live by stones above root 0:{:.1}% 1:{:.1}% 2:{:.1}% 3-5:{:.1}% 6-10:{:.1}% 11+:{:.1}% | live depth 0-2:{:.1}% 3-5:{:.1}% 6-9:{:.1}% 10+:{:.1}% | live age now:{:.1}% prev:{:.1}% older:{:.1}% | proven {:.2}%",
            self.buf.len(), pc(empty), pc(dead), pc(ahead[0]), pc(ahead[1]), pc(ahead[2]), pc(ahead[3]), pc(ahead[4]), pc(ahead[5]),
            pc(depth[0]), pc(depth[1]), pc(depth[2]), pc(depth[3]), pc(age[0]), pc(age[1]), pc(age[2]), pc(proven)
        )
    }

    #[inline]
    fn store(&mut self, key: u64, mv: u32, score: i32, depth: u8, flag: u8) {
        self.store_e(key, mv, score, depth, flag, 0, 1);
    }

    /// D590 ranked store: same position -> update (refutation move kept on fail-low); else the victim is an empty
    /// slot, else a dead entry (more empties than the root), else the lowest worth = depth + 4 x importance
    /// + 16 if proven - 3 x age (searches since last touched, capped at 4)
    #[inline]
    fn store_v2(&mut self, key: u64, mut mv: u32, score: i32, depth: u8, flag: u8, empties: u8, imp: u8) {
        let (age, root_empties, mask) = (self.age, self.root_empties, self.mask);
        let Some(v) = self.v2.as_mut() else { return };
        let bk = &mut v[(key as usize) & mask].0;
        let chk = (key >> 32) as u32;
        let new = |mv: u32| TtE2 { chk, score, mv: mv as u16, depth, meta: (flag & 3) | ((imp & 3) << 2), age, empties, _pad: 0 };
        if let Some(i) = (0..4).find(|&i| bk[i].meta & 3 != 0 && bk[i].chk == chk) {
            if flag == FLAG_UPPER && bk[i].mv != 0 && bk[i].mv != 0xFFFF {
                mv = bk[i].mv as u32;
            }
            bk[i] = new(mv);
            return;
        }
        let worth = |e: &TtE2| -> i32 {
            if e.meta & 3 == 0 {
                return i32::MIN;
            }
            if e.empties > root_empties {
                return i32::MIN + 1;
            }
            let a = age.wrapping_sub(e.age).min(4) as i32;
            e.depth as i32 + 4 * ((e.meta >> 2) & 3) as i32 + if e.score.abs() >= MATE_BOUND { 16 } else { 0 } - 3 * a
        };
        let victim = (0..4).min_by_key(|&i| worth(&bk[i])).unwrap_or(0);
        bk[victim] = new(mv);
    }

    #[inline]
    fn store_e(&mut self, key: u64, mut mv: u32, score: i32, depth: u8, flag: u8, empties: u8, imp: u8) {
        if self.v2.is_some() {
            self.store_v2(key, mv, score, depth, flag, empties, imp);
            return;
        }
        let (root_empties, dead_mode) = (self.root_empties, self.dead_mode);
        let e = &mut self.buf[(key as usize) & self.mask];
        let same_pos = e.flag != FLAG_EMPTY && e.key == key;
        // D571: an entry with more empties than the root is unreachable for the rest of the game
        let dead = dead_mode > 0 && e.flag != FLAG_EMPTY && e.empties > root_empties;

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
        let replace = if dead_mode == 2 {
            e.flag == FLAG_EMPTY || same_pos || dead || e.depth <= depth
        } else {
            e.flag == FLAG_EMPTY
                || same_pos
                || dead
                || (!same_pos && (age_diff >= 2 || (age_diff == 1 && depth.saturating_add(4) >= e.depth)))
                || e.depth <= depth
        };

        if replace {
            *e = TtEntry {
                key,
                mv,
                score,
                depth,
                flag,
                age: self.age,
                empties,
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
    /// D585 bucketed eval cache (Some = ECACHE_MODE 1)
    ec2: Option<Box<[EcBucket]>>,
    /// D586: importance of the next eval-cache store (1 = a searched node or one of the first 3 moves, 0 = a late move)
    ec_imp: u32,
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
    /// exp/policy: move-ordering policy head (None = unchanged), weight (ordering points per logit), min depth, scratch.
    policy: Option<&'a crate::policy::Policy>,
    pol_w: f32,
    pol_min: u32,
    /// set_policy was called explicitly: set_nnue must not drop it
    pol_explicit: bool,
    pol_buf: Vec<(u16, f32)>,
    /// Tuple-native eval mode: skip E2/E4 scans, mat+PST+tempo+tuples only.
    /// False by default (full hand eval + optional additive tuples).
    tup_native: bool,
    /// Correction history (None = OFF, zero-cost branch). Owned tables,
    /// reset per game; never merged into weights or TT.
    corr: Option<CorrHist>,
    /// Per-ply move/acc buffers, reused across searches (no per-search
    /// alloc: MoveStack is ~72KB of lists + acc, was rebuilt per search).
    stack: MoveStack,
    /// exp/tree-shape: search-tree trace (only with --features trace)
    #[cfg(feature = "trace")]
    pub trace: Option<Box<crate::trace::Tracer>>,
    /// per-phase knobs, copied from the global table at every search start (D557)
    knobs: Knobs,
    /// D575: eval-cache hits / misses (trace build only)
    #[cfg(feature = "trace")]
    pub ec_stats: (u64, u64),
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
            ecache: vec![(0u64, i32::MIN); if ECACHE_MODE.load(std::sync::atomic::Ordering::Relaxed) == 1 { 1 } else { 1 << ECACHE_BITS.load(std::sync::atomic::Ordering::Relaxed) }].into_boxed_slice(),
            ec2: if ECACHE_MODE.load(std::sync::atomic::Ordering::Relaxed) == 1 {
                Some(vec![EcBucket([EC_EMPTY; 8]); (1usize << ECACHE_BITS.load(std::sync::atomic::Ordering::Relaxed)) / 8].into_boxed_slice())
            } else {
                None
            },
            ec_imp: 1,
            killers: Box::new([(0, 0); MAX_PLY]),
            history: Box::new([[[0; 49]; 49]; 2]),
            sancta: None,
            tuple: None,
            nnue: None,
            policy: crate::policy::global(),
            pol_w: crate::policy::env_f32("TI_POLW", 100000.0),
            pol_min: crate::policy::env_f32("TI_POLMIN", 1.0) as u32,
            pol_explicit: false,
            pol_buf: Vec::new(),
            tup_native: false,
            corr: None,
            stack: MoveStack::new(),
            #[cfg(feature = "trace")]
            trace: None,
            knobs: DEFAULT_KNOBS,
            #[cfg(feature = "trace")]
            ec_stats: (0, 0),
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
    pub fn set_policy(&mut self, p: Option<&'a crate::policy::Policy>, w: f32, min_depth: u32) {
        self.policy = p;
        self.pol_w = w;
        self.pol_min = min_depth;
        self.pol_explicit = true;
    }

    pub fn set_nnue(&mut self, n: Option<&'a Nnue>) {
        self.nnue = n;
        // The bundled policy head is fitted on the bundled big net (config.rs pins it): with any other net its
        // ordering is meaningless, so it switches itself off unless the user chose a head (--policy FILE / TI_POLICY).
        if !self.pol_explicit && !crate::policy::is_forced() {
            self.policy = match n {
                Some(x) if x.fingerprint != crate::nnue::Nnue::bundled_fingerprint() => None,
                _ => crate::policy::global(),
            };
        }
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
        if let Some(e) = self.ec2.as_mut() {
            e.fill(EcBucket([EC_EMPTY; 8]));
        }
    }

    /// D585: bucketed eval-cache probe; a hit moves the entry to the front of its line (LRU order)
    #[inline]
    fn ec2_get(&mut self, b: &Board) -> Option<i32> {
        let e = self.ec2.as_mut()?;
        let n = e.len();
        let bk = &mut e[(b.hash as usize) & (n - 1)].0;
        let chk = ec_check(b.hash, b.empty().count_ones(), 0);
        for i in 0..8 {
            let x = bk[i];
            if ((x >> 32) as u32 & !EC_IMP_BIT) == chk && x as u32 != 0x8000_0000 {
                if i > 0 {
                    bk.copy_within(0..i, 1);
                    bk[0] = x;
                }
                return Some(x as u32 as i32);
            }
        }
        None
    }

    /// D585/D586 ranked eviction: insert at the front; victim = the first empty slot, else the first dead entry (fewer
    /// stones than the root: unreachable), else the least recently used UNIMPORTANT entry, else the last (LRU)
    #[inline]
    fn ec2_put(&mut self, b: &Board, v: i32) {
        let root_empties = self.tt.root_empties as u32;
        let imp = self.ec_imp;
        let Some(e) = self.ec2.as_mut() else { return };
        let n = e.len();
        let bk = &mut e[(b.hash as usize) & (n - 1)].0;
        let chk = ec_check(b.hash, b.empty().count_ones(), imp);
        let victim = (0..8)
            .find(|&i| bk[i] as u32 == 0x8000_0000)
            .or_else(|| (0..8).find(|&i| ((bk[i] >> 32) as u32 & 63) > root_empties))
            .or_else(|| (0..8).rev().find(|&i| (bk[i] >> 32) as u32 & EC_IMP_BIT == 0))
            .unwrap_or(7);
        bk.copy_within(0..victim, 1);
        bk[0] = ((chk as u64) << 32) | (v as u32 as u64);
    }

    /// D585: start loading the child's eval-cache line and TT slot while the parent still works on the move
    #[inline(always)]
    fn prefetch_child(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        if let Some(e) = self.ec2.as_ref() {
            let pe = &e[(hash as usize) & (e.len() - 1)] as *const EcBucket as *const i8;
            let pt = self.tt.slot_ptr(hash);
            unsafe {
                std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(pe);
                std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(pt);
            }
        }
    }

    #[inline]
    fn leaf_eval(&mut self, b: &Board) -> i32 {
        self.leaf_eval_with_dis(b).0
    }

    #[inline]
    fn leaf_eval_with_dis(&mut self, b: &Board) -> (i32, i32) {
        if self.ec2.is_some() {
            if let Some(v) = self.ec2_get(b) {
                #[cfg(feature = "trace")]
                {
                    self.ec_stats.0 += 1;
                }
                return unpack_eval_dis(v);
            }
            #[cfg(feature = "trace")]
            {
                self.ec_stats.1 += 1;
            }
            let (eval, dis) = self.leaf_eval_raw_with_dis(b);
            self.ec2_put(b, pack_eval_dis(eval, dis));
            return (eval, dis);
        }
        let slot = (b.hash as usize) & (self.ecache.len() - 1);
        let (k, v) = self.ecache[slot];
        if k == b.hash && v != i32::MIN {
            #[cfg(feature = "trace")]
            {
                self.ec_stats.0 += 1;
            }
            return unpack_eval_dis(v);
        }
        #[cfg(feature = "trace")]
        {
            self.ec_stats.1 += 1;
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
    /// D572: TT + eval-cache census for the position `b` (the next search root)
    pub fn cache_census(&self, b: &Board) -> String {
        let filled = match self.ec2.as_ref() {
            Some(e) => e.iter().map(|b| b.0.iter().filter(|&&x| x as u32 != 0x8000_0000).count()).sum::<usize>(),
            None => self.ecache.iter().filter(|e| e.1 != i32::MIN).count(),
        };
        #[cfg(feature = "trace")]
        let hits = format!(" hits {} misses {} hit rate {:.1}%", self.ec_stats.0, self.ec_stats.1, 100.0 * self.ec_stats.0 as f64 / (self.ec_stats.0 + self.ec_stats.1).max(1) as f64);
        #[cfg(not(feature = "trace"))]
        let hits = String::new();
        let cap = self.ec2.as_ref().map(|e| e.len() * 8).unwrap_or(self.ecache.len());
        format!("{} | ecache {} entries ({}) filled {:.1}%{hits}", self.tt.census(b.empty().count_ones() as u8), cap, if self.ec2.is_some() { "8-way 8 B" } else { "direct 16 B" }, 100.0 * filled as f64 / cap as f64)
    }

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

    #[cfg(not(feature = "trace"))]
    #[inline(always)]
    fn negamax<const BZ: bool>(&mut self, b: &Board, alpha: i32, beta: i32, depth: u32, ply: u32, null_allowed: bool) -> i32 {
        self.negamax_body::<BZ>(b, alpha, beta, depth, ply, null_allowed)
    }

    /// trace build: one record per call (crate::trace)
    #[cfg(feature = "trace")]
    fn negamax<const BZ: bool>(&mut self, b: &Board, alpha: i32, beta: i32, depth: u32, ply: u32, null_allowed: bool) -> i32 {
        let e = b.empty().count_ones() as u8;
        trc!(self, t => t.enter(b.hash, e, alpha, beta, depth, ply));
        let s = self.negamax_body::<BZ>(b, alpha, beta, depth, ply, null_allowed);
        trc!(self, t => t.exit(s));
        s
    }

    /// exp/tree-shape: walk the principal variation through the TT (legal, exact-or-bound entries, no repeats)
    pub fn pv_from_tt(&mut self, b: &Board, max: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut cur = *b;
        let mut seen = Vec::new();
        while pv.len() < max {
            let Some(e) = self.tt.probe(cur.hash) else { break };
            if e.mv == 0xFFFF {
                break;
            }
            let m = Move::from_u32(e.mv);
            let mut ml = MoveList::new();
            cur.legal_moves_dedup_into(&mut ml);
            if !(0..ml.len()).any(|i| ml.move_at(i) == m) {
                break;
            }
            cur = cur.make(m);
            if seen.contains(&cur.hash) {
                break;
            }
            seen.push(cur.hash);
            pv.push(m);
        }
        pv
    }

    #[inline]
    fn negamax_body<const BZ: bool>(
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
            trc!(self, t => t.end(crate::trace::kind::STACK, 0));
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
                trc!(self, t => t.end(crate::trace::kind::TERMINAL, 0));
                return bz_terminal_score(b, ply);
            }
        } else if b.game_over() {
            trc!(self, t => t.end(crate::trace::kind::TERMINAL, 0));
            return terminal_score(b, ply);
        }

        // Mate distance pruning (Stockfish / Reckless):
        // Even if we mate on the next move, our score is at best MATE - (ply + 1).
        // Even if we are mated right here, our score is at worst -MATE + ply.
        alpha = alpha.max(-MATE + ply as i32);
        beta = beta.min(MATE - (ply as i32 + 1));
        if alpha >= beta {
            trc!(self, t => t.end(crate::trace::kind::MATE_DIST, 0));
            return alpha;
        }

        // Transposition table probe. Missing entry at depth >= 4 -> IIR
        // (internal iterative reduction, Moonbird): the node has no cached
        // best move to order by, so search one ply less.
        let alpha_orig = alpha;
        let mut tt_mv: u32 = 0;
        let mut depth = depth;
        let bk = knob_bucket(b.empty().count_ones());
        if let Some(e) = self.tt.probe(b.hash) {
            tt_mv = e.mv;
            if e.depth as u32 >= depth {
                let s = score_from_tt(e.score, ply);
                match e.flag {
                    FLAG_EXACT => {
                        if s.abs() < MATE_BOUND || (MATE - s.abs()) as u32 >= depth {
                            self.heuristic_used = true;
                        }
                        trc!(self, t => t.end(crate::trace::kind::TT_CUT, 0));
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
                    trc!(self, t => t.end(crate::trace::kind::TT_CUT, 1));
                    return s;
                }
            }
        } else if depth >= self.knobs.v[bk][K_IIR] as u32 {
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
            _ => {
                self.ec_imp = 1;
                self.leaf_eval_with_dis(b)
            }
        };
        let static_eval = self.corrected(b, raw);
        trc!(self, t => if let Some(r) = t.top() {
            r.sdepth = depth.min(127) as i8;
            r.stat = static_eval;
            r.dis = node_dis;
        });

        // Reverse futility pruning (autaxx, exact form): standing pat is so
        // far below alpha that even a generous error margin cannot recover
        // -> fail low. Safe in Ataxx because the eval is capture-risk-aware.
        // Disabled when alpha/beta are mate scores so centipawn evals never prune mate lines.
        if null_allowed
            && ply > 0
            && depth <= 4
            && alpha.abs() < MATE_BOUND
            && beta.abs() < MATE_BOUND
            && static_eval + self.knobs.v[bk][K_RFP1 + (depth.max(1) - 1) as usize] + ((node_dis * self.knobs.v[bk][K_RFP_DIS]) >> 7) < alpha
        {
            self.heuristic_used = true;
            trc!(self, t => t.end(crate::trace::kind::RFP, alpha - (static_eval + self.knobs.v[bk][K_RFP1 + (depth.max(1) - 1) as usize] + ((node_dis * self.knobs.v[bk][K_RFP_DIS]) >> 7))));
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
            if clone_targets >= self.knobs.v[bk][K_NMP_CLONES] as u32
                && (b.piece_cnt[0] as u32 + b.piece_cnt[1] as u32) * 50 < 26 * SQUARES as u32
                && static_eval >= beta + self.knobs.v[bk][K_NMP_MARGIN]
            {
                let child = b.make_pass();
                self.stack.imp[ply as usize + 1] = 0;
                self.stack.path_hash[ply as usize + 1] = child.hash;
                self.stack.rep_base[ply as usize + 1] = self.stack.rep_base[ply as usize];
                // NMP pass flips stm: sancta acc is color-relative so it
                // stays valid — just forward the other perspective.
                // (pass: no stones change, no diff update needed.)
                trc!(self, t => t.pending(0xFFFF, crate::trace::via::NULL, 255, (depth - 1) as i8));
                let score =
                    -self.negamax::<BZ>(&child, -beta, -beta + 1, depth.saturating_sub(self.knobs.v[bk][K_NMP_R] as u32), ply + 1, false);
                if self.stop {
                    return 0;
                }
                if score >= beta {
                    self.heuristic_used = true;
                    trc!(self, t => t.end(crate::trace::kind::NMP, score - beta));
                    return if score >= MATE_BOUND { beta } else { score };
                }
            }
        }

        // ProbCut: non-PV, not right after a null, margin widens with the net's disagreement output.
        let (pc_d, pc_r, pc_t, pc_k) = (self.knobs.v[bk][K_PC_D] as u32, self.knobs.v[bk][K_PC_R] as u32, self.knobs.v[bk][K_PC_T], self.knobs.v[bk][K_PC_K]);
        if null_allowed && ply > 0 && depth >= pc_d && beta - alpha == 1 && beta.abs() < MATE_BOUND {
            let pc_beta = beta + pc_t + ((node_dis * pc_k) >> 7);
            if pc_beta < MATE_BOUND && static_eval + 50 >= pc_beta - 2 * pc_t {
                trc!(self, t => t.pending(0xFFFF, crate::trace::via::PROBCUT, 255, depth as i8));
                let score = self.negamax::<BZ>(b, pc_beta - 1, pc_beta, depth.saturating_sub(pc_r), ply, false);
                if self.stop {
                    return 0;
                }
                if score >= pc_beta && score < MATE_BOUND {
                    self.heuristic_used = true;
                    trc!(self, t => t.end(crate::trace::kind::PROBCUT, score - pc_beta));
                    return score - (pc_beta - beta);
                }
            }
        }

        if depth == 0 {
            self.heuristic_used = true;
            trc!(self, t => t.end(crate::trace::kind::LEAF, 0));
            return static_eval;
        }

        {
            let list = &mut self.stack.lists[ply as usize];
            let kn = &self.knobs.v[bk];
            // D567: ahead -> clones, behind -> jumps (ordering points, default 0)
            let bias = if static_eval > kn[K_ORD_STATE_CP] {
                (kn[K_ORD_AHEAD_CLONE], 0)
            } else if static_eval < -kn[K_ORD_STATE_CP] {
                (0, kn[K_ORD_BEHIND_JUMP])
            } else {
                (0, 0)
            };
            score_into(b, list, tt_mv, ply, &self.killers, &self.history, kn[K_KILL_CANON] != 0, bias, kn[K_ORD_KILLER]);
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
                trc!(self, t => t.end(crate::trace::kind::NO_MOVES, 0));
                return static_eval;
            }
        }

        trc!(self, t => if let Some(r) = t.top() {
            r.nmoves = self.stack.lists[ply as usize].len().min(255) as u8;
        });
        if let (Some(pol), Some(nn)) = (self.policy, self.nnue) {
            let n = self.stack.lists[ply as usize].len();
            if n > 1 && depth >= self.pol_min {
                let mut buf = std::mem::take(&mut self.pol_buf);
                nn.policy_xin(b, &mut buf);
                let mut kc = crate::policy::KCache::new();
                for i in 0..n {
                    let (m, sc) = {
                        let l = &self.stack.lists[ply as usize];
                        (l.move_at(i), l.score_at(i))
                    };
                    if sc == ORD_TT {
                        continue;
                    }
                    let add = (pol.logit_k(b, m, &buf, &mut kc) * self.pol_w) as i32;
                    self.stack.lists[ply as usize].set_score(i, sc.saturating_add(add));
                }
                self.pol_buf = buf;
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
                        let (k, v) = self.ecache[(child.hash as usize) & (self.ecache.len() - 1)];
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
        let kn = self.knobs.v[bk];
        let (lmr_min, lmr_half) = (kn[K_LMR_MIN] as usize, kn[K_LMR_HALF]);
        let (fut_min, fut_div, fut_dis, sn_m) = (kn[K_FUT_MIN] as usize, kn[K_FUT_DIV], kn[K_FUT_DIS], kn[K_SN_M]);
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
            self.prefetch_child(child.hash);
            // D590 (user): the top ~10-15 moves are real candidates the search often keeps unreduced
            self.ec_imp = (move_idx < 12) as u32;
            self.stack.imp[ply as usize + 1] = if move_idx == 0 { 3 } else if move_idx < 12 { 2 } else { 1 };
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
                    trc!(self, t => t.skip(child.hash, child.empty().count_ones() as u8, m.to_u32() as u16, ply, move_idx, depth as i32 - 1, crate::trace::kind::SKIP_REP, i32::MIN, 0, alpha));
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
            if move_idx >= lmr_min {
                new_depth = depth as i32 * lmr_half / (lmr_half + lmr_cnt);
                lmr_cnt += 1;
            }
            let mut child_eval = i32::MIN;
            if move_idx >= fut_min
                && new_depth > 0
                && self.sancta.is_none()
                && alpha.abs() < MATE_BOUND
            {
                // child eval is from the child's side to move = minus our static value of the move
                if FUT_CHEAP_M > 0 && self.fut_sn && self.nnue.is_some() {
                    let slot = (child.hash as usize) & (self.ecache.len() - 1);
                    let cached = if self.ec2.is_some() {
                        self.ec2_get(&child).is_some()
                    } else {
                        self.ecache[slot].0 == child.hash && self.ecache[slot].1 != i32::MIN
                    };
                    if !cached {
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
                            - FUT_SN_M
                            - sn_m;
                        if new_depth - (alpha.saturating_add(lo) / fut_div).clamp(0, 64) <= 0
                            && best > -MATE_BOUND
                        {
                            self.heuristic_used = true;
                            trc!(self, t => t.skip(child.hash, child.empty().count_ones() as u8, m.to_u32() as u16, ply, move_idx, depth as i32 - 1, crate::trace::kind::SKIP_SN, lo, alpha.saturating_add(lo), alpha));
                            move_idx += 1;
                            continue;
                        }
                    }
                }
                let (ce, child_dis) = self.leaf_eval_with_dis(&child);
                child_eval = pack_eval_dis(ce, child_dis);
                let vol_credit = (child_dis * fut_dis) >> 7;
                let shortfall = alpha.saturating_add(ce) - vol_credit;
                new_depth -= (shortfall / fut_div).clamp(0, 64);
                if child_dis > 300 && new_depth < depth as i32 {
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
                trc!(self, t => t.skip(child.hash, child.empty().count_ones() as u8, m.to_u32() as u16, ply, move_idx, depth as i32 - 1, crate::trace::kind::SKIP_FUT, if child_eval == i32::MIN { i32::MIN } else { unpack_eval_dis(child_eval).0 }, new_depth, alpha));
                move_idx += 1;
                continue;
            }
            let mut score;
            if move_idx == 0 {
                trc!(self, t => t.pending(m.to_u32() as u16, crate::trace::via::FIRST, 0, depth as i8 - 1));
                score = -self.negamax::<BZ>(&child, -beta, -alpha, depth - 1, ply + 1, true);
                if self.stop {
                    return 0;
                }
            } else {
                self.stack.pre_eval[ply as usize + 1] = child_eval;
                trc!(self, t => t.pending(m.to_u32() as u16, crate::trace::via::SCOUT, move_idx.min(255) as u8, depth as i8 - 1));
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
                    trc!(self, t => t.pending(m.to_u32() as u16, crate::trace::via::RESEARCH, move_idx.min(255) as u8, depth as i8 - 1));
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
            trc!(self, t => t.end(crate::trace::kind::CIRCULAR, 0));
            return -MATE + ply as i32;
        }
        trc!(self, t => t.end(crate::trace::kind::FULL, 0));
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
            .store_e(b.hash, best_mv, score_to_tt(best, ply), depth as u8, flag, b.empty().count_ones() as u8, self.stack.imp[ply as usize]);
    }
    /// Iterative deepening root search. Falls back to any legal move.
    /// Whichever limit hits first (nodes / time / depth) stops the search;
    /// an aborted iteration is adopted by the Stockfish rule (see the adoption block in `search_gen`).
    pub fn search(&mut self, b: &Board, limits: &SearchLimits) -> SearchResult {
        self.search_gen::<false>(b, limits, &[])
    }

    /// Same search under the Botzone judge's rules (no passes; either side
    /// stuck ends the game, mobile side absorbs the empties; NMP off).
    pub fn search_bz(&mut self, b: &Board, limits: &SearchLimits) -> SearchResult {
        self.search_gen::<true>(b, limits, &[])
    }

    /// Standard search excluding specified root moves (Multi-PV tree generation).
    pub fn search_excluding(
        &mut self,
        b: &Board,
        limits: &SearchLimits,
        banned_root: &[Move],
    ) -> SearchResult {
        self.search_gen::<false>(b, limits, banned_root)
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
        self.knobs = knobs();
        self.tt.root_empties = b.empty().count_ones() as u8;
        self.tt.dead_mode = self.knobs.v[knob_bucket(b.empty().count_ones())][K_TT_DEAD] as u8;
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
            score_into(b, root, tt_mv, 0, kill, hist, self.knobs.v[knob_bucket(b.empty().count_ones())][K_KILL_CANON] != 0, (0, 0), self.knobs.v[knob_bucket(b.empty().count_ones())][K_ORD_KILLER]);
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
            trc!(self, t => {
                t.clear();
                t.enter(b.hash, b.empty().count_ones() as u8, -INF, INF, u32::from(depth), 0);
            });
            for i in 0..n_root {
                let m = root_moves[i];
                let child = b.make(m);
                self.stack.imp[1] = 3;
                self.stack.path_hash[1] = child.hash;
                self.stack.rep_base[1] = if m.is_clone() { 1 } else { 0 };
                // Root children: acc[1] = acc[0] + diffs (sancta lazy path).
                if let Some(net) = self.sancta {
                    let (a, z) = self.stack.acc.split_at_mut(1);
                    net.update_child(&a[0], &mut z[0], b, &child);
                }
                trc!(self, t => t.pending(m.to_u32() as u16, crate::trace::via::FIRST, i.min(255) as u8, depth as i8 - 1));
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
            trc!(self, t => {
                t.end(crate::trace::kind::FULL, 0);
                t.exit(best_score);
            });
            // Partial-iteration adoption (Stockfish / Reckless):
            // - A completed iteration is adopted unless it forgot a shorter proven winning mate
            //   from a previous completed iteration (`forgotten_mate`).
            // - An aborted partial iteration (`self.stop`) is adopted whenever a root move finished (it
            //   exists as `best_this`), unless it forgot a proven mate: Stockfish semantics.
            if let Some(m) = best_this {
                let forgotten_mate = has_completed_iter
                    && result.score >= MATE_BOUND
                    && best_score < result.score;
                // Stockfish rule: the previous best move is searched first, so once any root move of an aborted
                // iteration has finished, that iteration's best so far is at least as informed as the older depth.
                let adopt = !forgotten_mate;
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

    /// Cutting the search mid-iteration (partial adoption) must always leave a legal move, a real score and a depth.
    #[test]
    fn partial_iteration_adoption_keeps_a_legal_move() {
        let mut seed = 0x51ed_270b_0f13_a7c9u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut list = crate::board::MoveList::new();
        let mut b = Board::start();
        let mut checked = 0;
        for _ in 0..40 {
            b.legal_moves_dedup_into(&mut list);
            if list.len() == 0 {
                break;
            }
            for nodes in [20u64, 60, 150, 400, 1200, 3000] {
                let mut se = Searcher::with_tt_bits(16);
                let lim = SearchLimits { time: None, max_nodes: Some(nodes), max_depth: u8::MAX };
                let r = se.search(&b, &lim);
                let m = r.best.expect("a move must always come back");
                assert!((0..list.len()).any(|i| list.move_at(i) == m), "illegal move at {nodes} nodes");
                assert!(r.depth >= 1 && r.score != i32::MIN);
                checked += 1;
            }
            let i = (next() % list.len() as u64) as usize;
            b = b.make(list.move_at(i));
        }
        assert!(checked >= 60);
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
