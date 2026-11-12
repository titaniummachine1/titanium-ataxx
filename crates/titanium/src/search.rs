//! Evaluation and iterative-deepening alpha-beta (negamax) search.
//!
//! Search order: TT probe -> eval cache (depth 0) -> move scoring
//! (TT move > killers > captures*W + history) -> lazy selection -> alpha-beta.

/// Eval cache size, log2 entries. D606: 2^22 x 8 B = 32 MB in the ranked layout (the old direct-mapped table was
/// 2^16 x 16 B = 1 MiB and thrashed at 500k nodes per move, D575). `set_ecache_bits` (CLI --ecache-bits) applies to
/// searchers created afterwards.
static ECACHE_BITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(22);

pub fn set_ecache_bits(bits: usize) {
    ECACHE_BITS.store(bits.clamp(10, 26), std::sync::atomic::Ordering::Relaxed);
}

/// D585: eval-cache layout. 1 (default) = cache-line buckets: 8 entries of 8 bytes per 64-byte line (32-bit check =
/// 25 hash bits + importance bit + 6 bits of empties, 32-bit packed eval), one memory fetch per probe, victim =
/// empty > dead (fewer stones than the search root: unreachable) > least recently used unimportant entry, plus a
/// prefetch of the child's eval-cache line and TT slot as soon as the move is made. 0 = the old direct-mapped table
/// of (u64 key, i32) entries.
static ECACHE_MODE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

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

/// D590: ranked TT (TT_MODE 1, default): 16-byte entries, 4 per 64-byte line. 0 = the old 24-byte single-slot table.
static TT_MODE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

pub fn set_tt_mode(m: usize) {
    TT_MODE.store(m.min(1), std::sync::atomic::Ordering::Relaxed);
}

/// D643/D652 (user): STAGED SIZE. The ranked tables do not start at their full size: they start at an L1-sized table and
/// JUMP to an L2-sized, then an L3-sized, then the full table (--tt-bits / --ecache-bits, 22 bits by default) when the
/// current stage overflows (more than half of its slots filled). NEVER by doubling or one more bit (user rule D652, it never
/// worked; my doubling version lost at 1000 / 3300 ms, D649). A search that touches only a few thousand positions stays
/// inside the CPU cache; a long search ends at the full size. The stage sizes come from the TARGET CPU's cache sizes read at
/// engine start (crate::cacheinfo, CPUID, never compile-time constants): L1d, L2, L3 / 2 (the L3 is shared), the TT taking
/// 2/3 and the eval cache 1/3 of each budget (the same 2:1 split as the full 64 MB + 32 MB). Growth never rehashes: the index
/// mask widens, the newly exposed buckets are cleared at that moment, entries whose new index bits differ are stranded and
/// evicted like any stale entry. A new game resets to stage 1. The flags below: 0 = automatic stages; N = a 2-stage
/// table starting at 2^N entries (N >= the full size = fixed size, for experiments and parity checks).
static TT_MIN_BITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ECACHE_MIN_BITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_tt_min_bits(b: usize) {
    TT_MIN_BITS.store(if b == 0 { 0 } else { b.clamp(4, 30) }, std::sync::atomic::Ordering::Relaxed);
}

pub fn set_ecache_min_bits(b: usize) {
    ECACHE_MIN_BITS.store(if b == 0 { 0 } else { b.clamp(4, 30) }, std::sync::atomic::Ordering::Relaxed);
}

/// which cache levels are used as stages: bit 0 = L1d, bit 1 = L2, bit 2 = L3 / 2 (default 0 = NO staging: the full size from the first search. The L3-first start gained +17..25% nodes in cold-table tests but nothing in real games, D664; 4 = L3 then full).
/// Fewer stages are allowed (user D652) but a stage is always a cache-level size, never something in between.
static STAGE_MASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// intermediate RAM stages between the L3 stage and the full size (default 0 = jump L3 -> full)
static RAM_STAGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_ram_stages(n: usize) {
    RAM_STAGES.store(n.min(6), std::sync::atomic::Ordering::Relaxed);
}

pub fn set_cache_stages(mask: u32) {
    STAGE_MASK.store(mask & 7, std::sync::atomic::Ordering::Relaxed);
}

fn floor_pow2(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        1usize << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Bucket counts (64-byte buckets, powers of two, strictly increasing, the last = `nb_max`) of the stages.
/// `explicit_bits` != 0: [2^bits entries' worth of buckets, nb_max]; 0: the cache-sized stages, `share` = this table's
/// fraction (numerator, denominator) of each cache budget, `per_bucket` = entries per bucket of the flag's bit convention.
fn stage_list(nb_max: usize, explicit_bits: usize, per_bucket: usize, share: (usize, usize), c: crate::cacheinfo::CacheSizes, mask: u32, ram: usize) -> Vec<usize> {
    if explicit_bits != 0 {
        let nb = (1usize << explicit_bits).div_ceil(per_bucket).clamp(1, nb_max);
        return if nb >= nb_max { vec![nb_max] } else { vec![nb, nb_max] };
    }
    let mut st: Vec<usize> = Vec::new();
    for (i, budget) in [c.l1d, c.l2, c.l3 / 2].into_iter().enumerate() {
        if mask & (1 << i) == 0 {
            continue; // this cache level is not used as a stage (D652: fewer jumps are fine, but only at cache-level sizes)
        }
        let nb = floor_pow2(budget * share.0 / share.1 / 64).clamp(1, nb_max);
        if nb < nb_max && st.last().map_or(true, |&l| nb > l) {
            st.push(nb);
        }
    }
    // extra jumps are allowed beyond the L3 stage, because that part lives in RAM (user D652): `ram` intermediate stages,
    // geometrically spaced between the last cache stage and the full size (cache levels themselves never get in-between sizes)
    if ram > 0 {
        if let Some(&lo) = st.last() {
            let ratio = nb_max as f64 / lo as f64;
            for j in 1..=ram {
                let nb = floor_pow2((lo as f64 * ratio.powf(j as f64 / (ram + 1) as f64)) as usize);
                if nb > *st.last().unwrap() && nb < nb_max {
                    st.push(nb);
                }
            }
        }
    }
    st.push(nb_max);
    st
}

fn tt_stage_list(nb_max: usize) -> Vec<usize> {
    stage_list(nb_max, TT_MIN_BITS.load(std::sync::atomic::Ordering::Relaxed), 4, (2, 3), crate::cacheinfo::sizes(), STAGE_MASK.load(std::sync::atomic::Ordering::Relaxed), RAM_STAGES.load(std::sync::atomic::Ordering::Relaxed))
}

fn ec_stage_list(nb_max: usize) -> Vec<usize> {
    stage_list(nb_max, ECACHE_MIN_BITS.load(std::sync::atomic::Ordering::Relaxed), 8, (1, 3), crate::cacheinfo::sizes(), STAGE_MASK.load(std::sync::atomic::Ordering::Relaxed), RAM_STAGES.load(std::sync::atomic::Ordering::Relaxed))
}
/// D650 (user idea): BIT-PACKED TT entry, 10 bytes = 80 bits, 6 per 64-byte bucket (60 bytes + 4 spare) instead of
/// 4 x 16 bytes: +50% entries in the same memory, still one cache line per probe. Layout (little endian):
///   bytes 0-3 check (high 32 hash bits) | 4-5 score (16-bit code, ttcodec::enc_score) | 6 depth |
///   7 empties (6 bits) + age bits 0-1 | 8 move code bits 0-7 (ttcodec: 10-bit code, any pseudo-legal move) |
///   9 move code bits 8-9, flag (2 bits, 0 = empty slot), importance (2 bits), age bits 2-3.
/// The 4 spare bytes of the bucket (60-63) hold 6 x 5 extra check bits (hash bits 27-31) so every bit of the line is used:
/// the index takes hash bits 0-22 at most, the check bits 32-63, the extension bits 27-31 (59 of the 64 hash bits filter).
/// An all-zero bucket is an empty bucket.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct TtBucket([u8; 64]);

const TT_EBYTES: usize = 10;
/// entries per bucket in the packed layout (--tt-ways N: 4 reproduces the old 16-byte table's replacement behaviour)
const TT_MAX_WAYS: usize = 6;

static TT_WAYS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(TT_MAX_WAYS);

pub fn set_tt_ways(w: usize) {
    TT_WAYS.store(w.clamp(1, TT_MAX_WAYS), std::sync::atomic::Ordering::Relaxed);
}

/// unpacked view of one TT slot
#[derive(Clone, Copy)]
struct TtE2 {
    chk: u32,
    score: i32,
    code: u16,
    depth: u8,
    flag: u8,
    imp: u8,
    /// 4-bit generation
    age: u8,
    empties: u8,
    /// 5 extra check bits (hash bits 27-31)
    ext: u8,
}

impl TtBucket {
    const EMPTY: TtBucket = TtBucket([0u8; 64]);

    #[inline(always)]
    fn ext(&self, i: usize) -> u8 {
        let w = u32::from_le_bytes([self.0[60], self.0[61], self.0[62], self.0[63]]);
        ((w >> (5 * i)) & 31) as u8
    }

    #[inline(always)]
    fn set_ext(&mut self, i: usize, v: u8) {
        let mut w = u32::from_le_bytes([self.0[60], self.0[61], self.0[62], self.0[63]]);
        w = (w & !(31u32 << (5 * i))) | (((v & 31) as u32) << (5 * i));
        self.0[60..64].copy_from_slice(&w.to_le_bytes());
    }

    #[inline(always)]
    fn flag(&self, i: usize) -> u8 {
        (self.0[i * TT_EBYTES + 9] >> 2) & 3
    }

    #[inline(always)]
    fn chk(&self, i: usize) -> u32 {
        let o = i * TT_EBYTES;
        u32::from_le_bytes([self.0[o], self.0[o + 1], self.0[o + 2], self.0[o + 3]])
    }

    #[inline(always)]
    fn empties(&self, i: usize) -> u8 {
        self.0[i * TT_EBYTES + 7] & 63
    }

    #[inline(always)]
    fn age(&self, i: usize) -> u8 {
        let o = i * TT_EBYTES;
        (self.0[o + 7] >> 6) | ((self.0[o + 9] >> 6) << 2)
    }

    #[inline(always)]
    fn set_age(&mut self, i: usize, age: u8) {
        let o = i * TT_EBYTES;
        self.0[o + 7] = (self.0[o + 7] & 63) | ((age & 3) << 6);
        self.0[o + 9] = (self.0[o + 9] & 0x3F) | (((age >> 2) & 3) << 6);
    }

    #[inline(always)]
    fn get(&self, i: usize) -> TtE2 {
        let o = i * TT_EBYTES;
        let b = &self.0;
        TtE2 {
            chk: self.chk(i),
            score: crate::ttcodec::dec_score(i16::from_le_bytes([b[o + 4], b[o + 5]])),
            code: b[o + 8] as u16 | (((b[o + 9] & 3) as u16) << 8),
            depth: b[o + 6],
            flag: (b[o + 9] >> 2) & 3,
            imp: (b[o + 9] >> 4) & 3,
            age: self.age(i),
            empties: b[o + 7] & 63,
            ext: self.ext(i),
        }
    }

    #[inline(always)]
    fn set(&mut self, i: usize, e: &TtE2) {
        let o = i * TT_EBYTES;
        let c = e.chk.to_le_bytes();
        let s = crate::ttcodec::enc_score(e.score).to_le_bytes();
        let b = &mut self.0;
        b[o..o + 4].copy_from_slice(&c);
        b[o + 4] = s[0];
        b[o + 5] = s[1];
        b[o + 6] = e.depth;
        b[o + 7] = (e.empties & 63) | ((e.age & 3) << 6);
        b[o + 8] = e.code as u8;
        b[o + 9] = ((e.code >> 8) & 3) as u8 | ((e.flag & 3) << 2) | ((e.imp & 3) << 4) | (((e.age >> 2) & 3) << 6);
        self.set_ext(i, e.ext);
    }
}

use crate::board::{box1, jump_union, reach_union, Board, Move, MoveList, RING1, RING2, SQUARES};
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
const RFP_MARGINS: [i32; 4] = [257, 347, 478, 774];
/// exp/search-moonbird A: Moonbird 1.1.0 reduction scheme (searchParams.nim) in place of the universal LMR.
/// LMR from the 7th move: new_depth = depth * HALF / (HALF + n_reduced_so_far).
const MB_LMR_MIN_IDX: usize = 6;
const MB_LMR_HALFLIFE: i32 = 47;
/// Futility reduction from the 2nd move: one ply per FUT_DIV cp the child's static eval falls short of alpha
/// (Moonbird futilityReductionDiv 35 in its units; x1.62 to Titanium cp: 568/ln10 vs 400 per logit). Reduce only.
const MB_FUT_MIN_IDX: usize = 1;
const MB_FUT_DIV: i32 = 57;
/// Volatility scaling numerator (scaled by 128) for Reverse Futility Pruning: +25% of predicted disagreement.
const DIS_RFP_NUM: i32 = 32;
/// Volatility credit numerator (scaled by 128) for Child Futility Pruning: 37.5% credit.
const DIS_FUT_NUM: i32 = 24;
/// ProbCut (gated +20 Elo @100 ms vs main, user-approved 2026-10-31): min depth, depth reduction, margin cp,
/// disagreement-head credit (/128). A shallow null-window search at pc_beta; if it clears pc_beta the node cuts.
const PC_D: u32 = 6;
const PC_R: u32 = 3;
const PC_T: i32 = 140;
const PC_K: i32 = 32;

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
const ORD_PLUG_HOLE_UNIT: i32 = env_usize(option_env!("ORD_PLUG_HOLE_UNIT"), 400) as i32;
const ORD_EVAC_UNIT: i32 = env_usize(option_env!("ORD_EVAC_UNIT"), 250) as i32;
const ORD_HISTORY_MAX: i32 = 90_000;
/// exp/search-histmb: Moonbird 1.1.0 history (searchUtils.nim) keyed [stm][from][to] (clones keyed [to][to]):
/// +depth^2 for the move that raised alpha, -depth^2/24 for every searched move that did not, |h| capped at 107000
/// (x24 here so the malus stays integer) with the colour's table shrunk /1.25 when the cap is hit. Ordering weight
/// = Moonbird historyMoveOrderingFactor 2.0 stones = 2 conversions at full history.
const HIST_SCALE: i32 = 24;
const HIST_MAX: i32 = 107_000 * HIST_SCALE;
const ORD_HIST_FULL: i32 = 2 * ORD_CAPTURE_UNIT;
type History = [[[i32; 49]; 49]; 2];

/// Killer key. A clone's child position depends only on its DESTINATION: the `from` that `legal_moves_dedup_into`
/// reports is just the lowest-index neighbouring stone, which differs between sibling positions. A killer clone must
/// therefore be keyed by destination only (same rule as `hist_key`), otherwise it rarely matches (measured on 8 positions to
/// depth 11: 2.7% of clone moves matched the stored killer exactly, 4.0% by destination). Jumps keep their `(from, to)`.
#[inline]
fn kill_key(m: Move) -> u32 {
    if m.is_clone() {
        ((m.to as u32) << 8) | m.to as u32
    } else {
        m.to_u32()
    }
}

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

/// Empty landing squares `them` can reach right now (shared E2+E4 input â€”
/// computed once per side, not once per term).
#[inline]
fn enemy_landings(b: &Board, them: u8) -> u64 {
    b.empty() & reach_union(b.occ[them as usize])
}

/// E2: stones of `side` the enemy can currently convert â€” own stones ring-1
/// of an enemy-reachable landing square.
#[inline]
fn endangered_count(b: &Board, side: u8, landings_them: u64) -> u32 {
    (b.occ[side as usize] & box1(landings_them)).count_ones()
}

/// E4: for each enemy-reachable landing square, own stones there beyond the
/// first (each extra one = a stone lost in the same single enemy move).
/// Computed branch-free via O(1) CSA bit-planes (`nnfeat::victim_planes`).
#[inline]
fn multicapture_penalty(b: &Board, side: u8, landings_them: u64) -> i32 {
    if landings_them == 0 {
        return 0;
    }
    let [p0, p1, p2, p3] = crate::nnfeat::victim_planes(b.occ[side as usize]);
    let active = landings_them & (p1 | p2 | p3);
    if active == 0 {
        return 0;
    }
    let extra = (active & p0).count_ones()
        + 2 * (active & p1).count_ones()
        + 4 * (active & p2).count_ones()
        + 8 * (active & p3).count_ones()
        - active.count_ones();
    extra as i32 * MULTICAP_PEN
}

/// Static eval from the side-to-move's perspective. Material + PST + tempo
/// are O(1) incremental; E2/E4 regional terms (contact + multicap exposure)
/// scan enemy reachability per node â€” the knowledge that won +190/+330.
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
    // zero recompute â€” capconv's land_u/land_t ARE land_w/land_b per side).
    if let Some(t) = tup {
        let land_me = if b.turn == 0 { land_w } else { land_b };
        let land_opp = if b.turn == 0 { land_b } else { land_w };
        out += t.forward_fast(b, land_me, land_opp);
    }
    out
}

/// Score moves in place: TT move > killers > captures*W + history.
/// Free function (not a method) so negamax can borrow the move stack
/// mutably while reading killers/history immutably â€” no split-borrow
/// fight, no unsafe, no clone.
const HIST_DIV: i32 = HIST_MAX / ORD_HIST_FULL;
const _: () = assert!(HIST_MAX % ORD_HIST_FULL == 0);

/// Fused move generation + scoring (+ optional policy head): iterates legal
/// destinations in ascending `to` order (identical tie-break order to
/// `legal_moves_dedup_into`), sharing per-destination (`flips`, `tholes`,
/// `k3_to`) and per-origin (`conn`, `from_holes`, `k3_from`) work across jumps.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn gen_and_score_into<const SKIP_TT: bool, const HAS_POLICY: bool>(
    b: &Board,
    list: &mut MoveList,
    tt_mv: u32,
    ply: u32,
    killers: &[(u32, u32); MAX_PLY],
    history: &History,
    pol: Option<(&crate::policy::Policy, &[(u16, f32)], f32)>,
) {
    use std::mem::MaybeUninit;

    list.clear();
    let me = b.me();
    let opp = b.opp();
    let empty = b.empty();
    let mut targets = reach_union(me) & empty;
    let opp_reach = reach_union(opp) & empty;
    let (k1, k2) = killers[ply as usize];
    let hist = &history[b.turn as usize];
    let mut tt_index = usize::MAX;

    let mut orig_valid: u64 = 0;
    let mut orig_conn = [MaybeUninit::<i32>::uninit(); 49];
    let mut orig_evac = [MaybeUninit::<i32>::uninit(); 49];
    let mut orig_g = [MaybeUninit::<f32>::uninit(); 49];
    let mut orig_tor = [MaybeUninit::<i32>::uninit(); 49];

    while targets != 0 {
        let to = targets.trailing_zeros() as u8;
        targets &= targets - 1;
        let tu = to as usize;
        let r1_to = RING1[tu];
        let caps = (opp & r1_to).count_ones() as i32;
        let cap_sc = caps * ORD_CAPTURE_UNIT;
        let to_is_opp_landing = (opp_reach & (1u64 << tu)) != 0;
        let plug_bonus = if ORD_PLUG_HOLE_UNIT > 0 && to_is_opp_landing {
            let threatened = (me & r1_to).count_ones() as i32;
            if threatened >= 2 {
                (threatened - 1) * ORD_PLUG_HOLE_UNIT
            } else {
                0
            }
        } else {
            0
        };

        let clones = me & r1_to;
        let jumps = me & RING2[tu];

        let (flips, tholes, k3_to) = if HAS_POLICY {
            (
                (caps as usize).min(8),
                ((r1_to & empty).count_ones() as usize).min(8),
                crate::policy::k3code_pub(me, opp, to),
            )
        } else {
            (0, 0, 0)
        };

        if clones != 0 {
            let from = clones.trailing_zeros() as u8;
            let m = Move { from, to };
            if tt_mv != 0 && (1 + to as u32) == tt_mv {
                if !SKIP_TT {
                    tt_index = list.len();
                    list.push_scored(m, ORD_TT);
                }
            } else {
                // Clone preference: a clone nets one more stone than a jump and
                // leaves the origin defended ("duplicate as much as you can").
                // (exp/ab A tested no-asymmetry: 41-55, keep preference.)
                let mut sc = cap_sc + ORD_CLONE_BONUS + plug_bonus;
                let kk = ((to as u32) << 8) | to as u32;
                if kk == k1 || kk == k2 {
                    sc += ORD_KILLER;
                }
                sc += hist[tu][tu] / HIST_DIV;
                if HAS_POLICY {
                    let (p, xin, w) = pol.unwrap();
                    let z = p.clone_logit_pre(to, flips, tholes, k3_to, xin);
                    sc = sc.saturating_add((z * w) as i32);
                }
                list.push_scored(m, sc);
            }
        }

        if jumps != 0 {
            let (j_prefix, j_td) = if HAS_POLICY {
                let (p, _, _) = pol.unwrap();
                p.jump_dest_parts(to, flips, tholes, k3_to)
            } else {
                (0.0, 0)
            };
            let mut js = jumps;
            while js != 0 {
                let from = js.trailing_zeros() as u8;
                js &= js - 1;
                let m = Move { from, to };
                if tt_mv != 0 && crate::ttcodec::jump_code(from, to) as u32 == tt_mv {
                    if !SKIP_TT {
                        tt_index = list.len();
                        list.push_scored(m, ORD_TT);
                    }
                    continue;
                }
                let fu = from as usize;
                let fbit = 1u64 << from;
                if orig_valid & fbit == 0 {
                    orig_valid |= fbit;
                    let r1_from = RING1[fu];
                    orig_conn[fu].write((me & r1_from).count_ones() as i32);
                    let evac_sc = if ORD_EVAC_UNIT > 0 {
                        // Check if from was touching any opponent landing square
                        let danger_holes = opp_reach & r1_from;
                        if danger_holes != 0 {
                            // Sum threatened count on adjacent holes
                            let mut dh = danger_holes;
                            let mut max_threat = 0i32;
                            while dh != 0 {
                                let h = dh.trailing_zeros() as usize;
                                dh &= dh - 1;
                                let t = (me & RING1[h]).count_ones() as i32;
                                if t > max_threat {
                                    max_threat = t;
                                }
                            }
                            if max_threat >= 2 {
                                ORD_EVAC_UNIT * (max_threat - 1)
                            } else {
                                0
                            }
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    orig_evac[fu].write(evac_sc);
                    if HAS_POLICY {
                        let (p, _, _) = pol.unwrap();
                        let fh = ((r1_from & empty).count_ones() as usize).min(8);
                        let k3f = crate::policy::k3code_pub(me, opp, from);
                        let (fg, ft) = p.jump_orig_parts(from, fh, k3f);
                        orig_g[fu].write(fg);
                        orig_tor[fu].write(ft);
                    }
                }
                // SAFETY: `orig_valid & fbit != 0` ensures `orig_conn[fu]`, `orig_evac[fu]` (and `orig_g[fu]`/`orig_tor[fu]` under `HAS_POLICY`) was written.
                let conn = unsafe { orig_conn[fu].assume_init() };
                let evac = unsafe { orig_evac[fu].assume_init() };
                // exp/ab O1 (hole-aware jumps): a jump vacates its origin —
                // leaving a connected origin breaks formation, so penalty
                // scales with own stones adjacent to the origin square.
                let mut sc = cap_sc + plug_bonus + evac - (ORD_JUMP_PEN + conn * ORD_JUMP_CONN);
                let kk = m.to_u32();
                if kk == k1 || kk == k2 {
                    sc += ORD_KILLER;
                }
                sc += hist[fu][tu] / HIST_DIV;
                if HAS_POLICY {
                    let (p, xin, w) = pol.unwrap();
                    let (fg, ft) = unsafe { (orig_g[fu].assume_init(), orig_tor[fu].assume_init()) };
                    let z = p.jump_finish(to, j_prefix, j_td, fg, ft, xin);
                    sc = sc.saturating_add((z * w) as i32);
                }
                list.push_scored(m, sc);
            }
        }
    }

    if !SKIP_TT && tt_index != usize::MAX && tt_index != 0 {
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
#[inline(always)]
fn bz_terminal_score_counts(b: &Board, hm0: bool, hm1: bool, ply: u32) -> i32 {
    let (fb, fw) = b.bz_counts_from(hm0, hm1);
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

fn bz_terminal_score(b: &Board, ply: u32) -> i32 {
    let (_, hm0, hm1) = b.bz_status();
    bz_terminal_score_counts(b, hm0, hm1, ply)
}

#[derive(Clone, Copy, Default)]
struct BzPly {
    rep_base: u32,
    ban_start: u16,
    tainted: bool,
}

/// Pre-capture occupancy `(black, white)` right after `m` is placed, matching
/// the Botzone judge's `savehistory()` snapshot for repetition detection.
#[inline(always)]
fn placement_snap(b: &Board, m: Move) -> (u64, u64) {
    b.placement_snap(m)
}

/// Move scoring is fixed: TT move > killers > captures*W + history, consumed
/// lazily (repeated argmax â€” only pays for moves examined before a cutoff).

/// Transposition table entry (24 bytes with padding).
#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    mv: u32, // packed Move; 0xFFFF = pass; 0 = none (eval cache entries)
    score: i32,
    depth: u8,
    flag: u8, // 0 empty, 1 exact, 2 lower, 3 upper
    age: u8,  // Generation age for FIFO/LRU eviction without RAM increase (fits in compiler padding)
    /// empties of the stored position (padding byte). Empties never increase in Ataxx, so an entry with more
    /// empties than the search root can never be reached again (D571). 0 = unknown.
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

/// Fixed-size transposition table, power-of-two entry count. Depth-0 EXACT
/// entries double as the eval cache (Ataxx transpositions repeat a lot).
pub struct Tt {
    buf: Vec<TtEntry>,
    /// D590 ranked buckets (Some = TT_MODE 1; then `buf` is a 1-entry dummy)
    v2: Option<Box<[TtBucket]>>,
    /// active bucket mask (adaptive size: grows towards nb_max, see TT_MIN_BITS)
    mask: usize,
    /// full bucket count, minimum bucket count, slots filled since the last reset (ranked mode)
    nb_max: usize,
    /// bucket counts of the stages (L1, L2, L3, full: D652) and the current stage
    stages: Vec<usize>,
    stage: usize,
    filled: usize,
    /// entries per bucket in use (packed layout, <= TT_MAX_WAYS)
    ways: usize,
    pub age: u8,
    /// empties of the current search root (255 = nothing is dead): entries/cache lines with more empties are unreachable
    pub root_empties: u8,
}

impl Tt {
    /// 2^bits entries (bits=20 -> 1M entries, 24 MB; ranked mode: 16 B entries, 4 per line).
    pub fn new(bits: usize) -> Tt {
        if TT_MODE.load(std::sync::atomic::Ordering::Relaxed) == 1 {
            let nb = (1usize << bits.min(25)) / 4;
            let mut t = Tt::new_v1(0);
            t.v2 = Some(vec![TtBucket::EMPTY; nb].into_boxed_slice());
            t.nb_max = nb;
            t.stages = tt_stage_list(nb);
            t.stage = 0;
            t.mask = t.stages[0] - 1;
            t.ways = TT_WAYS.load(std::sync::atomic::Ordering::Relaxed);
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
                    flag: FLAG_EMPTY,
                    age: 0,
                    empties: 0,
                };
                size
            ],
            mask: size - 1,
            nb_max: 0,
            stages: Vec::new(),
            stage: 0,
            filled: 0,
            ways: TT_MAX_WAYS,
            v2: None,
            root_empties: 255,
            age: 0,
        }
    }

    pub fn clear(&mut self) {
        for e in self.buf.iter_mut() {
            e.flag = FLAG_EMPTY;
        }
        if let Some(v) = self.v2.as_mut() {
            // back to stage 1; only that part is cleared now, the rest is cleared when it is exposed (grow)
            self.stage = 0;
            let nb0 = self.stages[0];
            self.mask = nb0 - 1;
            self.filled = 0;
            v[..nb0].fill(TtBucket::EMPTY);
        }
        self.age = 0;
    }

    /// Staged size (D652): JUMP to the next stage (L1 -> L2 -> L3 -> full), never by doubling. Never rehash: the mask
    /// widens, the newly exposed buckets are cleared now, stranded entries age out.
    #[cold]
    fn grow(&mut self) {
        if self.stage + 1 < self.stages.len() {
            self.jump_to(self.stage + 1);
        }
    }

    /// Jump to `stage` (never backwards): widen the mask, clear the newly exposed buckets, keep the stranded entries.
    fn jump_to(&mut self, stage: usize) {
        let cur = self.mask + 1;
        if stage <= self.stage || stage >= self.stages.len() || cur >= self.nb_max {
            return;
        }
        self.stage = stage;
        let new = self.stages[stage].max(cur);
        if let Some(v) = self.v2.as_mut() {
            v[cur..new].fill(TtBucket::EMPTY);
        }
        self.mask = new - 1;
        self.filled = self.filled * cur / new;
    }

    /// First search of a game (fresh table): start at the smallest stage whose capacity holds 4x the expected nodes of
    /// the search, so a short search stays in a cache-sized table and a long one starts at the full size and never
    /// jumps in the middle of a search (D657: L3-first +25% nodes at 100 ms, -7% at 1000 ms because of the late jump).
    fn start_stage(&mut self, expected_nodes: usize) {
        if self.v2.is_none() || self.filled != 0 {
            return;
        }
        let mut s = 0;
        while s + 1 < self.stages.len() && self.stages[s] * self.ways < 4usize.saturating_mul(expected_nodes) {
            s += 1;
        }
        self.jump_to(s);
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
        self.probe_e(key, 255)
    }

    /// Probe with the position's empty-square count (D641/D647, user): stones never disappear in Ataxx, so a position
    /// has exactly one stone count, and a hash match whose stored empties differ is a false positive. 255 = unchecked.
    #[inline]
    fn probe_e(&mut self, key: u64, empties: u8) -> Option<TtEntry> {
        // NOTE: the returned `mv` is a ttcodec MOVE CODE (not a packed from/to) in both table modes.
        if let Some(v) = self.v2.as_mut() {
            let age = self.age & 15;
            let chk = (key >> 32) as u32;
            let ext = ((key >> 27) & 31) as u8;
            let ways = self.ways;
            let bk = &mut v[(key as usize) & self.mask];
            for i in 0..ways {
                if bk.flag(i) != 0 && bk.chk(i) == chk && bk.ext(i) == ext && (empties == 255 || bk.empties(i) == empties) {
                    bk.set_age(i, age);
                    let e = bk.get(i);
                    return Some(TtEntry { key, mv: e.code as u32, score: e.score, depth: e.depth, flag: e.flag, age: self.age, empties: e.empties });
                }
            }
            return None;
        }
        let e = &mut self.buf[(key as usize) & self.mask];
        if e.flag != FLAG_EMPTY && e.key == key {
            e.age = self.age;
            let mut r = *e;
            r.mv = crate::ttcodec::code_of_u32(r.mv) as u32;
            Some(r)
        } else {
            None
        }
    }

    #[inline]
    fn store(&mut self, key: u64, mv: u32, score: i32, depth: u8, flag: u8) {
        self.store_e(key, mv, score, depth, flag, 0, 1);
    }

    /// D590 ranked store: same position -> update (refutation move kept on fail-low); else the victim is an empty
    /// slot, else a dead entry (more empties than the root), else the lowest worth = depth + 4 x importance
    /// + 16 if proven - 3 x age (searches since last touched, capped at 4)
    #[inline]
    fn store_v2(&mut self, key: u64, mv: u32, score: i32, depth: u8, flag: u8, empties: u8, imp: u8) {
        let (age, root_empties, mask, ways) = (self.age & 15, self.root_empties, self.mask, self.ways);
        let Some(v) = self.v2.as_mut() else { return };
        let bk = &mut v[(key as usize) & mask];
        let chk = (key >> 32) as u32;
        let ext = ((key >> 27) & 31) as u8;
        let mut code = crate::ttcodec::code_of_u32(mv);
        // same position: update in place (a position has exactly one stone count, so a different count is another position)
        if let Some(i) = (0..ways).find(|&i| bk.flag(i) != 0 && bk.chk(i) == chk && bk.ext(i) == ext && bk.empties(i) == (empties & 63)) {
            if flag == FLAG_UPPER {
                let old = bk.get(i).code;
                if old != 0 && old != crate::ttcodec::PASS_CODE {
                    code = old; // keep the refutation move on a fail-low
                }
            }
            bk.set(i, &TtE2 { chk, score, code, depth, flag, imp, age, empties, ext });
            return;
        }
        let worth = |i: usize| -> i32 {
            if bk.flag(i) == 0 {
                return i32::MIN;
            }
            if bk.empties(i) > root_empties {
                return i32::MIN + 1;
            }
            let e = bk.get(i);
            let a = (age.wrapping_sub(e.age) & 15).min(4) as i32;
            e.depth as i32 + 4 * e.imp as i32 + if e.score.abs() >= MATE_BOUND { 16 } else { 0 } - 3 * a
        };
        let victim = (0..ways).min_by_key(|&i| worth(i)).unwrap_or(0);
        let was_empty = bk.flag(victim) == 0;
        bk.set(victim, &TtE2 { chk, score, code, depth, flag, imp, age, empties, ext });
        if was_empty {
            self.filled += 1;
            if self.filled * 2 > (self.mask + 1) * self.ways && self.mask + 1 < self.nb_max {
                self.grow();
            }
        }
    }

    #[inline]
    fn store_e(&mut self, key: u64, mut mv: u32, score: i32, depth: u8, flag: u8, empties: u8, imp: u8) {
        if self.v2.is_some() {
            self.store_v2(key, mv, score, depth, flag, empties, imp);
            return;
        }
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
    /// Best move so far, published for another thread (the Botzone driver watchdog, D661/D662): 0 = none, else 1 + (from << 8 | to).
    /// Set to a legal fallback at the start of a search and after every adopted iteration.
    best_sink: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>,
    /// Engine-side time safety margin in ms (deadline = budget - this): 10 for the CLI / gates (the harness gets its answer in time), the
    /// Botzone driver sets it to ~0 because the driver owns the clock (ti_stop + hard-deadline backup answer, D662).
    safety_ms: f64,
    /// Transposition table (persists across searches in the same Searcher).
    tt: Tt,
    /// exp/nnue-speed: static-eval cache (position hash -> raw eval). The eval is a pure function of the position, so a
    /// hit returns exactly what a recomputation would: search is bit-identical, only faster (iterative deepening
    /// re-evaluates ~42% of positions). Cleared on new game and whenever the eval source changes.
    ecache: Box<[(u64, i32)]>,
    /// D585 bucketed eval cache (Some = ECACHE_MODE 1; then `ecache` is a 1-entry dummy)
    ec2: Option<Box<[EcBucket]>>,
    /// D586/D590: importance of the next eval-cache store (1 = a searched node or one of the top-12 moves, 0 = a late move)
    ec_imp: u32,
    /// D643 adaptive size of the ranked eval cache: active bucket count (power of two, grows by doubling when more than
    /// half of the active slots were filled), minimum bucket count, slots filled since the last reset
    ec_act: usize,
    /// bucket counts of the stages (L1, L2, L3, full: D652) and the current stage
    ec_stages: Vec<usize>,
    ec_stage: usize,
    ec_filled: usize,
    /// Two killer moves per ply (packed u32 moves), refreshed on cutoffs.
    killers: Box<[(u32, u32); MAX_PLY]>,
    /// History heuristic: [mover][to] cutoff counters, aged per search.
    history: Box<History>,
    /// Optional sancta net (None = classical slim eval). Borrowed weights,
    /// set once via set_sancta, read-only in the tree. Zero refcount,
    /// zero unsafe â€” the borrow checker proves the net outlives the search.
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
    /// Botzone (`BZ = true`) game placement snapshots `(black_occ, white_occ)` covering
    /// the current stone-count epoch from pre-root history plus in-tree plies at `bz_base_offset + ply`.
    bz_snaps: Vec<(u64, u64)>,
    /// Index of the search root (`ply = 0`) within `bz_snaps`.
    bz_base_offset: usize,
    /// Real game ply at search root (`snaps.len().saturating_sub(1)`).
    bz_start_ply: u32,
    /// Relative ply horizon from root where Botzone judge terminates at ply 401
    /// (`401 - bz_start_ply`, or `u32::MAX` when disabled).
    bz_max_plies: u32,
    /// Per-ply Botzone state packed into 8 bytes/ply (`rep_base`, `ban_start`, `tainted`).
    bz_ply: Box<[BzPly; MAX_PLY]>,
    /// Shared scratch stack for Botzone banned repetition snapshots (`[ban_start..ban_start + n_banned]`).
    /// Keeps the 512-byte buffer off every recursive `negamax::<true>` stack frame.
    bz_banned: Vec<(u64, u64)>,
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
            best_sink: None,
            safety_ms: 10.0,
            tt: Tt::new(bits),
            ecache: vec![
                (0u64, i32::MIN);
                if ECACHE_MODE.load(std::sync::atomic::Ordering::Relaxed) == 1 {
                    1
                } else {
                    1 << ECACHE_BITS.load(std::sync::atomic::Ordering::Relaxed)
                }
            ]
            .into_boxed_slice(),
            ec2: if ECACHE_MODE.load(std::sync::atomic::Ordering::Relaxed) == 1 {
                Some(
                    vec![EcBucket([EC_EMPTY; 8]); (1usize << ECACHE_BITS.load(std::sync::atomic::Ordering::Relaxed)) / 8]
                        .into_boxed_slice(),
                )
            } else {
                None
            },
            ec_imp: 1,
            ec_act: ec_stage_list((1usize << ECACHE_BITS.load(std::sync::atomic::Ordering::Relaxed)) / 8)[0],
            ec_stages: ec_stage_list((1usize << ECACHE_BITS.load(std::sync::atomic::Ordering::Relaxed)) / 8),
            ec_stage: 0,
            ec_filled: 0,
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
            bz_snaps: Vec::with_capacity(MAX_PLY + 64),
            bz_base_offset: 0,
            bz_start_ply: 0,
            bz_max_plies: u32::MAX,
            bz_ply: Box::new([BzPly::default(); MAX_PLY]),
            bz_banned: Vec::with_capacity(256),
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
            // back to stage 1; only that part is cleared now, the rest is cleared when it is exposed (ec_grow)
            self.ec_stage = 0;
            self.ec_act = self.ec_stages[0];
            self.ec_filled = 0;
            e[..self.ec_act].fill(EcBucket([EC_EMPTY; 8]));
        }
    }

    /// Staged size (D652): JUMP to the next eval-cache stage (L1 -> L2 -> L3 -> full), never by doubling. Never rehash.
    #[cold]
    fn ec_grow(&mut self) {
        if self.ec_stage + 1 < self.ec_stages.len() {
            self.ec_jump_to(self.ec_stage + 1);
        }
    }

    fn ec_jump_to(&mut self, stage: usize) {
        let cur = self.ec_act;
        if stage <= self.ec_stage || stage >= self.ec_stages.len() {
            return;
        }
        self.ec_stage = stage;
        let new = self.ec_stages[stage].max(cur);
        let Some(e) = self.ec2.as_mut() else { return };
        e[cur..new].fill(EcBucket([EC_EMPTY; 8]));
        self.ec_act = new;
        self.ec_filled = self.ec_filled * cur / new;
    }

    /// Same rule as `Tt::start_stage` for the eval cache (8 entries per bucket).
    fn ec_start_stage(&mut self, expected_nodes: usize) {
        if self.ec2.is_none() || self.ec_filled != 0 {
            return;
        }
        let mut s = 0;
        while s + 1 < self.ec_stages.len() && self.ec_stages[s] * 8 < 4usize.saturating_mul(expected_nodes) {
            s += 1;
        }
        self.ec_jump_to(s);
    }

    /// D585: bucketed eval-cache probe; a hit moves the entry to the front of its line (LRU order)
    #[inline]
    fn ec2_get(&mut self, b: &Board) -> Option<i32> {
        let n = self.ec_act;
        let e = self.ec2.as_mut()?;
        let bk = &mut e[(b.hash as usize) & (n - 1)].0;
        let chk = ec_check(b.hash, b.empty().count_ones(), 0);
        // 2 more check bits (hash bits 37-38) live in the unused top bits 29-30 of the packed value (it needs 29 bits)
        let ext = (((b.hash >> 37) & 3) as u32) << 29;
        for i in 0..8 {
            let x = bk[i];
            if ((x >> 32) as u32 & !EC_IMP_BIT) == chk && x as u32 != 0x8000_0000 && (x as u32 & 0x6000_0000) == ext {
                if i > 0 {
                    bk.copy_within(0..i, 1);
                    bk[0] = x;
                }
                return Some((x as u32 & 0x1FFF_FFFF) as i32);
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
        let n = self.ec_act;
        let Some(e) = self.ec2.as_mut() else { return };
        let bk = &mut e[(b.hash as usize) & (n - 1)].0;
        let chk = ec_check(b.hash, b.empty().count_ones(), imp);
        let victim = (0..8)
            .find(|&i| bk[i] as u32 == 0x8000_0000)
            .or_else(|| (0..8).find(|&i| ((bk[i] >> 32) as u32 & 63) > root_empties))
            .or_else(|| (0..8).rev().find(|&i| (bk[i] >> 32) as u32 & EC_IMP_BIT == 0))
            .unwrap_or(7);
        let was_empty = bk[victim] as u32 == 0x8000_0000;
        bk.copy_within(0..victim, 1);
        let ext = (((b.hash >> 37) & 3) as u64) << 29;
        bk[0] = ((chk as u64) << 32) | (v as u32 as u64 & 0x1FFF_FFFF) | ext;
        if was_empty {
            self.ec_filled += 1;
            if self.ec_filled * 2 > self.ec_act * 8 && self.ec_stage + 1 < self.ec_stages.len() {
                self.ec_grow();
            }
        }
    }

    /// D585: start loading the child's eval-cache line and TT slot while the parent still works on the move
    #[inline(always)]
    fn prefetch_child(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        if let Some(e) = self.ec2.as_ref() {
            let pe = &e[(hash as usize) & (self.ec_act - 1)] as *const EcBucket as *const i8;
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
                return unpack_eval_dis(v);
            }
            let (eval, dis) = self.leaf_eval_raw_with_dis(b);
            self.ec2_put(b, pack_eval_dis(eval, dis));
            return (eval, dis);
        }
        let slot = (b.hash as usize) & (self.ecache.len() - 1);
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
    /// Publish the best move so far into `sink` during searches (see `best_sink`).
    pub fn set_safety_ms(&mut self, ms: f64) {
        self.safety_ms = ms.clamp(0.0, 1000.0);
    }

    pub fn set_best_sink(&mut self, sink: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>) {
        self.best_sink = sink;
    }

    #[inline]
    fn publish_best(&self, m: Move) {
        if let Some(s) = &self.best_sink {
            s.store(1 + m.to_u32(), std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn set_stop_flag(&mut self, f: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) {
        self.ext_stop = f;
    }

    /// Override the clock (wasm builds inject `performance.now()`).
    pub fn set_time_source(&mut self, f: fn() -> f64) {
        self.time_src = f;
    }

    /// Full state reset between games (TT, killers, history) â€” prevents
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

    /// Slow path for Botzone (`BZ = true`) jump-repetition collection.
    /// Populates `self.bz_banned[ban_start..ban_start + n_banned]` and returns `(n_banned, overflow)`.
    #[cold]
    #[inline(never)]
    fn collect_bz_bans_cold(
        &mut self,
        cur_v: usize,
        rbase: usize,
        ban_start: usize,
    ) -> (usize, bool) {
        self.bz_banned.truncate(ban_start);
        let last = self.bz_snaps[cur_v];
        let hist = &self.bz_snaps[rbase..cur_v];
        let mut n = 0usize;
        let mut overflow = false;
        // Distances 1..=3 cannot repeat under alternating non-pass Botzone moves (d=1: placement bit
        // flips; d=2: empty target holds opponent; d=3: P-piece vacating x returns as Q on x pre-captures).
        for i in 0..hist.len().saturating_sub(3) {
            if hist[i] == last {
                let ban = hist[i + 1];
                if !self.bz_banned[ban_start..ban_start + n].contains(&ban) {
                    if n < 32 {
                        self.bz_banned.push(ban);
                        n += 1;
                    } else {
                        overflow = true;
                    }
                }
            }
        }
        (n, overflow)
    }

    /// Cold check whether candidate placement snapshot `snap` repeats an earlier `(last -> snap)` transition.
    #[cold]
    #[inline(never)]
    fn is_bz_snap_banned(
        &self,
        cur_v: usize,
        rbase: usize,
        ban_start: usize,
        n_banned: usize,
        overflow: bool,
        snap: (u64, u64),
    ) -> bool {
        if self.bz_banned[ban_start..ban_start + n_banned].contains(&snap) {
            return true;
        }
        if overflow {
            let last = self.bz_snaps[cur_v];
            return self.bz_snaps[rbase..cur_v.saturating_sub(2)]
                .windows(2)
                .any(|w| w[0] == last && w[1] == snap);
        }
        false
    }

    /// Cold depth-0 check whether any legal jump avoids a Botzone `CIRCULAR_MOVE` forfeit when no clones exist.
    #[cold]
    #[inline(never)]
    fn has_unbanned_bz_jump(
        &self,
        b: &Board,
        cur_v: usize,
        rbase: usize,
        ban_start: usize,
        n_banned: usize,
        overflow: bool,
    ) -> bool {
        let me = b.me();
        let empty = b.empty();
        // Clone moves strictly increase piece count and can never repeat an earlier snapshot.
        if (box1(me) & empty) != 0 {
            return true;
        }
        let mut js_targets = jump_union(me) & empty;
        while js_targets != 0 {
            let to = js_targets.trailing_zeros() as u8;
            js_targets &= js_targets - 1;
            let mut froms = me & RING2[to as usize];
            while froms != 0 {
                let from = froms.trailing_zeros() as u8;
                froms &= froms - 1;
                let snap = placement_snap(b, Move { from, to });
                if !self.is_bz_snap_banned(cur_v, rbase, ban_start, n_banned, overflow, snap) {
                    return true;
                }
            }
        }
        false
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
        if BZ {
            self.bz_ply[ply as usize].tainted = false;
        }
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
            let (over, hm0, hm1) = b.bz_status();
            if over {
                return bz_terminal_score_counts(b, hm0, hm1, ply);
            }
            // Judge also ends the game unconditionally after ply 401 (Black's 201st half-move).
            if ply >= self.bz_max_plies {
                self.bz_ply[ply as usize].tainted = true;
                return bz_terminal_score_counts(b, hm0, hm1, ply);
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

        // Botzone (`BZ = true`) exact placement-snapshot repetition table & horizon check:
        // collect all jump transitions banned by `repeat(nowgame)` at this node before probing TT or pruning.
        let at_bz_horizon = BZ && ply.saturating_add(depth) >= self.bz_max_plies;
        let mut cur_v = 0usize;
        let mut rbase = 0usize;
        let mut ban_start = 0usize;
        let mut n_banned = 0usize;
        let mut banned_overflow = false;
        if BZ {
            let bp = self.bz_ply[ply as usize];
            cur_v = self.bz_base_offset + ply as usize;
            rbase = bp.rep_base as usize;
            ban_start = bp.ban_start as usize;
            if self.bz_start_ply + ply >= 4 && cur_v >= rbase + 4 {
                let last = self.bz_snaps[cur_v];
                if self.bz_snaps[rbase..cur_v - 3].contains(&last) {
                    (n_banned, banned_overflow) = self.collect_bz_bans_cold(cur_v, rbase, ban_start);
                }
            }
        }
        let mut tainted = BZ && at_bz_horizon;

        // Transposition table probe. Missing entry at depth >= 4 -> IIR
        // (internal iterative reduction, Moonbird): the node has no cached
        // best move to order by, so search one ply less.
        let is_pv = beta - alpha > 1;
        let alpha_orig = alpha;
        let mut tt_mv: u32 = 0;
        let mut depth = depth;
        if let Some(e) = self.tt.probe_e(b.hash, b.empty().count_ones() as u8) {
            tt_mv = e.mv;
            if !(BZ && (n_banned > 0 || at_bz_horizon)) && e.depth as u32 >= depth {
                let s = score_from_tt(e.score, ply);
                let cut = if !is_pv {
                    e.flag == FLAG_EXACT
                        || (e.flag == FLAG_LOWER && s >= beta)
                        || (e.flag == FLAG_UPPER && s <= alpha)
                } else {
                    ply == 1
                        && alpha <= -MATE_BOUND
                        && beta < MATE_BOUND
                        && e.flag != FLAG_UPPER
                        && s >= beta
                };
                if cut {
                    if s.abs() < MATE_BOUND || (MATE - s.abs()) as u32 >= depth {
                        self.heuristic_used = true;
                    }
                    return s;
                }
            }
        } else if depth >= 4 && !at_bz_horizon {
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

        // Reverse futility pruning (autaxx, exact form): standing pat is so
        // far below alpha that even a generous error margin cannot recover
        // -> fail low. Safe in Ataxx because the eval is capture-risk-aware.
        // Disabled when alpha/beta are mate scores so centipawn evals never prune mate lines.
        if !is_pv
            && null_allowed
            && !(BZ && (n_banned > 0 || at_bz_horizon))
            && ply > 0
            && depth <= 4
            && alpha.abs() < MATE_BOUND
            && beta.abs() < MATE_BOUND
            && static_eval + RFP_MARGINS[(depth.max(1) - 1) as usize] + ((node_dis * DIS_RFP_NUM) >> 7) < alpha
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
            let clone_targets = (box1(b.occ[b.turn as usize]) & b.empty()).count_ones();
            // fill < 0.54 as integer: pieces*50 < 26*49 (no per-node float div).
            if clone_targets >= 11
                && (b.piece_cnt[0] as u32 + b.piece_cnt[1] as u32) * 50 < 26 * SQUARES as u32
            {
                let child = b.make_pass();
                self.stack.imp[ply as usize + 1] = 0;
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

        // ProbCut: non-PV, not right after a null, margin widens with the net's disagreement output.
        if null_allowed && !(BZ && (n_banned > 0 || at_bz_horizon)) && ply > 0 && depth >= PC_D && beta - alpha == 1 && beta.abs() < MATE_BOUND {
            let pc_beta = beta + PC_T + ((node_dis * PC_K) >> 7);
            if pc_beta < MATE_BOUND && static_eval + 50 >= pc_beta - 2 * PC_T {
                let score = self.negamax::<BZ>(b, pc_beta - 1, pc_beta, depth - PC_R, ply, false);
                if self.stop {
                    return 0;
                }
                let pc_tainted = BZ && self.bz_ply[ply as usize].tainted;
                if BZ {
                    // Failed ProbCut shares `bz_ply[ply]`; clear so its skipped subtree cannot leak taint here.
                    self.bz_ply[ply as usize].tainted = false;
                }
                if score >= pc_beta && score < MATE_BOUND && !pc_tainted && !at_bz_horizon {
                    self.heuristic_used = true;
                    return score - (pc_beta - beta);
                }
            }
        }

        if depth == 0 {
            if BZ && n_banned > 0
                && !self.has_unbanned_bz_jump(b, cur_v, rbase, ban_start, n_banned, banned_overflow)
            {
                self.bz_ply[ply as usize].tainted = true;
                return -MATE + ply as i32;
            }
            // Note: `static_eval` depends only on `b`, so depth-0 leaves with legal unbanned moves remain untainted (`false` from entry).
            self.heuristic_used = true;
            return static_eval;
        }

        let mut best = i32::MIN;
        let mut best_mv: u32 = 0xFFFF;
        let mut move_idx = 0usize;
        let mut rep_skipped = false;
        let mut skip_tt_mv: u32 = 0;

        // Pre-movegen TT move fast path: decode & verify legality in O(1) and search first;
        // on a beta cutoff, return immediately without generating/scoring the full move list.
        if tt_mv != 0 {
            if let Some(m) = crate::ttcodec::decode_move(b, tt_mv as u16) {
                skip_tt_mv = tt_mv;
                let mut is_rep = false;
                let mut c_snap = (0u64, 0u64);
                if BZ && !m.is_clone() && n_banned > 0 {
                    c_snap = placement_snap(b, m);
                    is_rep = self.is_bz_snap_banned(
                        cur_v,
                        rbase,
                        ban_start,
                        n_banned,
                        banned_overflow,
                        c_snap,
                    );
                    if is_rep {
                        rep_skipped = true;
                        tainted = true;
                    }
                }
                if !is_rep {
                    let child = b.make(m);
                    if !BZ && !m.is_clone() {
                        let base = self.stack.rep_base[ply as usize] as usize;
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
                    }
                    if is_rep {
                        rep_skipped = true;
                    } else {
                        self.prefetch_child(child.hash);
                        self.ec_imp = 1;
                        self.stack.imp[ply as usize + 1] = 3;
                        if BZ {
                            if m.is_clone() || n_banned == 0 {
                                c_snap = placement_snap(b, m);
                            }
                            self.bz_snaps[cur_v + 1] = c_snap;
                            self.bz_ply[ply as usize + 1] = BzPly {
                                rep_base: if m.is_clone() {
                                    (cur_v + 1) as u32
                                } else {
                                    rbase as u32
                                },
                                ban_start: (ban_start + n_banned) as u16,
                                tainted: false,
                            };
                        } else {
                            self.stack.path_hash[ply as usize + 1] = child.hash;
                            self.stack.rep_base[ply as usize + 1] = if m.is_clone() {
                                ply + 1
                            } else {
                                self.stack.rep_base[ply as usize]
                            };
                        }
                        if let Some(net) = self.sancta {
                            let (a, z) = self.stack.acc.split_at_mut(ply as usize + 1);
                            net.update_child(&a[ply as usize], &mut z[0], b, &child);
                        }
                        let score =
                            -self.negamax::<BZ>(&child, -beta, -alpha, depth - 1, ply + 1, true);
                        if self.stop {
                            return 0;
                        }
                        if BZ && self.bz_ply[ply as usize + 1].tainted {
                            tainted = true;
                        }
                        move_idx = 1;
                        if score <= alpha {
                            self.hist_add(b.turn as usize, m, -((depth * depth) as i32));
                        }
                        best = score;
                        best_mv = m.to_u32();
                        if score > alpha {
                            alpha = score;
                            if alpha >= beta {
                                self.on_cutoff(b, m, depth, ply);
                                if best > alpha_orig && best_mv != 0xFFFF {
                                    self.hist_add(
                                        b.turn as usize,
                                        m,
                                        HIST_SCALE * (depth * depth) as i32,
                                    );
                                }
                                if BZ {
                                    self.bz_ply[ply as usize].tainted = tainted;
                                }
                                if !(BZ && tainted) {
                                    self.finish_node(b, alpha_orig, beta, best, best_mv, depth, ply, raw);
                                }
                                return best;
                            }
                        }
                    }
                }
            }
        }

        {
            match (self.policy, self.nnue) {
                (Some(pol), Some(nn)) if depth >= self.pol_min => {
                    if pol.dim > 0 && nn.pairwise {
                        let mut buf = std::mem::take(&mut self.pol_buf);
                        nn.policy_xin(b, &mut buf);
                        let list = &mut self.stack.lists[ply as usize];
                        gen_and_score_into::<true, true>(
                            b,
                            list,
                            skip_tt_mv,
                            ply,
                            &self.killers,
                            &self.history,
                            Some((pol, &buf, self.pol_w)),
                        );
                        self.pol_buf = buf;
                    } else {
                        let list = &mut self.stack.lists[ply as usize];
                        gen_and_score_into::<true, true>(
                            b,
                            list,
                            skip_tt_mv,
                            ply,
                            &self.killers,
                            &self.history,
                            Some((pol, &[], self.pol_w)),
                        );
                    }
                }
                _ => {
                    let list = &mut self.stack.lists[ply as usize];
                    gen_and_score_into::<true, false>(
                        b,
                        list,
                        skip_tt_mv,
                        ply,
                        &self.killers,
                        &self.history,
                        None,
                    );
                }
            }
            // Fallthrough terminal: legal_moves_dedup_into tracks dedup
            // multiplicities, so len>0 always â€” but double-pass positions
            // have no children to search: score the pass-through.
            if self.stack.lists[ply as usize].is_empty() && skip_tt_mv == 0 {
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
            let mut c_snap = (0u64, 0u64);
            if BZ && !m.is_clone() && n_banned > 0 {
                c_snap = placement_snap(b, m);
                let is_rep = self.is_bz_snap_banned(
                    cur_v,
                    rbase,
                    ban_start,
                    n_banned,
                    banned_overflow,
                    c_snap,
                );
                if is_rep {
                    rep_skipped = true;
                    tainted = true;
                    continue;
                }
            }
            let child = b.make(m);
            if !BZ && !m.is_clone() {
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
            }
            self.prefetch_child(child.hash);
            // D590 (user): the top ~10-15 moves are real candidates the search often keeps unreduced
            self.ec_imp = (move_idx < 12) as u32;
            self.stack.imp[ply as usize + 1] = if move_idx == 0 { 3 } else if move_idx < 12 { 2 } else { 1 };
            if !BZ {
                self.stack.path_hash[ply as usize + 1] = child.hash;
                self.stack.rep_base[ply as usize + 1] = if m.is_clone() {
                    ply + 1
                } else {
                    self.stack.rep_base[ply as usize]
                };
            }
            if let Some(net) = self.sancta {
                let (a, z) = self.stack.acc.split_at_mut(ply as usize + 1);
                net.update_child(&a[ply as usize], &mut z[0], b, &child);
            }
            let mut new_depth = depth as i32;
            if move_idx >= MB_LMR_MIN_IDX && !at_bz_horizon {
                new_depth = depth as i32 * MB_LMR_HALFLIFE / (MB_LMR_HALFLIFE + lmr_cnt);
                lmr_cnt += 1;
            }
            let mut child_eval = i32::MIN;
            if move_idx >= MB_FUT_MIN_IDX
                && !at_bz_horizon
                && new_depth > 0
                && self.sancta.is_none()
                && alpha.abs() < MATE_BOUND
            {
                // child eval is from the child's side to move = minus our static value of the move
                let mut cached_eval: Option<(i32, i32)> = None;
                let mut hand_cache: Option<i32> = None;
                if FUT_CHEAP_M > 0 && self.fut_sn && self.nnue.is_some() {
                    cached_eval = if self.ec2.is_some() {
                        self.ec2_get(&child).map(unpack_eval_dis)
                    } else {
                        let slot = (child.hash as usize) & (self.ecache.len() - 1);
                        let (k, v) = self.ecache[slot];
                        if k == child.hash && v != i32::MIN {
                            Some(unpack_eval_dis(v))
                        } else {
                            None
                        }
                    };
                    if cached_eval.is_none() {
                        let hand = evaluate(&child, None, false);
                        hand_cache = Some(hand);
                        let (bt, ct) = (b.turn as usize, child.turn as usize);
                        let sn = small_net();
                        let pa = sn_parent
                            .get_or_insert_with(|| sn.acc(b.occ[bt], b.occ[1 - bt], b.blockers));
                        let lo = sn.child_bound(
                            pa,
                            b.occ[bt],
                            b.occ[1 - bt],
                            child.occ[ct],
                            child.occ[1 - ct],
                            child.empty().count_ones(),
                            static_eval,
                            hand,
                        ) - FUT_SN_M;
                        if new_depth - (alpha.saturating_add(lo) / MB_FUT_DIV).clamp(0, 64) <= 0
                            && best > -MATE_BOUND
                        {
                            self.heuristic_used = true;
                            move_idx += 1;
                            continue;
                        }
                    }
                }
                let (ce, child_dis) = if let Some(ed) = cached_eval {
                    ed
                } else if FUT_CHEAP_M > 0 && self.fut_sn && self.nnue.is_some() {
                    let (eval, dis) = if let (Some(nn), Some(h)) = (
                        self.nnue,
                        hand_cache.filter(|_| self.tuple.is_none() && !self.tup_native),
                    ) {
                        if nn.v2 {
                            nn.eval_with_dis(&child, h)
                        } else {
                            nn.eval_with_dis(&child, 0)
                        }
                    } else {
                        self.leaf_eval_raw_with_dis(&child)
                    };
                    let packed = pack_eval_dis(eval, dis);
                    if self.ec2.is_some() {
                        self.ec2_put(&child, packed);
                    } else {
                        let slot = (child.hash as usize) & (self.ecache.len() - 1);
                        self.ecache[slot] = (child.hash, packed);
                    }
                    (eval, dis)
                } else {
                    self.leaf_eval_with_dis(&child)
                };
                child_eval = pack_eval_dis(ce, child_dis);
                let vol_credit = (child_dis * DIS_FUT_NUM) >> 7;
                let shortfall = alpha.saturating_add(ce) - vol_credit;
                new_depth -= (shortfall / MB_FUT_DIV).clamp(0, 64);
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
                move_idx += 1;
                continue;
            }
            if BZ {
                if m.is_clone() || n_banned == 0 {
                    c_snap = placement_snap(b, m);
                }
                self.bz_snaps[cur_v + 1] = c_snap;
                self.bz_ply[ply as usize + 1] = BzPly {
                    rep_base: if m.is_clone() {
                        (cur_v + 1) as u32
                    } else {
                        rbase as u32
                    },
                    ban_start: (ban_start + n_banned) as u16,
                    tainted: false,
                };
            }
            let mut score;
            if move_idx == 0 {
                score = -self.negamax::<BZ>(&child, -beta, -alpha, depth - 1, ply + 1, true);
                if self.stop {
                    return 0;
                }
                if BZ && self.bz_ply[ply as usize + 1].tainted {
                    tainted = true;
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
                if BZ && self.bz_ply[ply as usize + 1].tainted {
                    tainted = true;
                }
                if score > alpha {
                    // re-search: full window, full depth
                    self.stack.pre_eval[ply as usize + 1] = child_eval;
                    score = -self.negamax::<BZ>(&child, -beta, -alpha, depth - 1, ply + 1, true);
                    if self.stop {
                        return 0;
                    }
                    if BZ && self.bz_ply[ply as usize + 1].tainted {
                        tainted = true;
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
                        if BZ && !at_bz_horizon && !self.bz_ply[ply as usize + 1].tainted {
                            tainted = false;
                            rep_skipped = false;
                        }
                        self.on_cutoff(b, m, depth, ply);
                        break;
                    }
                }
            }
        }
        if best == i32::MIN {
            // Every legal move from this node repeated an ancestor position (CIRCULAR_MOVE forfeit).
            if BZ {
                self.bz_ply[ply as usize].tainted = true;
            }
            return -MATE + ply as i32;
        }
        if best > alpha_orig && best_mv != 0xFFFF {
            self.hist_add(
                b.turn as usize,
                Move::from_u32(best_mv),
                HIST_SCALE * (depth * depth) as i32,
            );
        }
        if BZ {
            self.bz_ply[ply as usize].tainted = tainted;
        }
        if !rep_skipped && !(BZ && tainted) {
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
        let packed = kill_key(m);
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
        self.search_gen::<false>(b, limits, &[], &[], false)
    }

    /// Same search under the Botzone judge's rules (no passes; either side
    /// stuck ends the game, mobile side absorbs the empties; NMP off).
    pub fn search_bz(&mut self, b: &Board, limits: &SearchLimits) -> SearchResult {
        self.search_gen::<true>(b, limits, &[], &[], false)
    }

    /// Standard search excluding specified root moves (Multi-PV tree generation).
    pub fn search_excluding(
        &mut self,
        b: &Board,
        limits: &SearchLimits,
        banned_root: &[Move],
    ) -> SearchResult {
        self.search_gen::<false>(b, limits, banned_root, &[], false)
    }

    /// Botzone search excluding repetition-banned root moves up front so the full
    /// search budget is spent evaluating allowed root moves at full depth.
    pub fn search_bz_excluding(
        &mut self,
        b: &Board,
        limits: &SearchLimits,
        banned_root: &[Move],
    ) -> SearchResult {
        self.search_gen::<true>(b, limits, banned_root, &[], false)
    }

    /// Botzone search with full game placement-snapshot history and optional 401-ply
    /// game-end horizon (`input.size() > 800`), matching the Botzone judge exactly.
    pub fn search_bz_game(
        &mut self,
        b: &Board,
        limits: &SearchLimits,
        banned_root: &[Move],
        snaps: &[(u64, u64)],
        respect_401: bool,
    ) -> SearchResult {
        self.search_gen::<true>(b, limits, banned_root, snaps, respect_401)
    }

    fn search_gen<const BZ: bool>(
        &mut self,
        b: &Board,
        limits: &SearchLimits,
        banned_root: &[Move],
        snaps: &[(u64, u64)],
        respect_401: bool,
    ) -> SearchResult {
        self.start_ms = (self.time_src)();
        self.nodes = 0;
        self.stop = false;
        self.stop_reason = StopReason::Completed;
        self.node_limit = limits.max_nodes;
        self.fut_sn = FUT_SN_ON.load(std::sync::atomic::Ordering::Relaxed);
        self.tt.root_empties = b.empty().count_ones() as u8;
        let safety_margin_ms = self.safety_ms;
        self.deadline_ms = limits
            .time
            .map(|t| self.start_ms + (t.as_secs_f64() * 1000.0 - safety_margin_ms).max(1.0));

        if BZ {
            let game_ply = snaps.len().saturating_sub(1);
            self.bz_start_ply = game_ply as u32;
            self.bz_max_plies = if respect_401 {
                401usize.saturating_sub(game_ply) as u32
            } else {
                u32::MAX
            };
            let root_stones = (b.occ[0] | b.occ[1]).count_ones();
            let tail_start = snaps
                .iter()
                .position(|&(x, y)| (x | y).count_ones() == root_stones)
                .unwrap_or(snaps.len());
            let tail = &snaps[tail_start..];
            self.bz_snaps.clear();
            if tail.is_empty() {
                self.bz_snaps.push((b.occ[0], b.occ[1]));
            } else {
                self.bz_snaps.extend_from_slice(tail);
            }
            let base_len = self.bz_snaps.len();
            self.bz_base_offset = base_len - 1;
            self.bz_snaps.resize(base_len + MAX_PLY + 2, (0, 0));
            self.bz_banned.clear();
            self.bz_ply[0] = BzPly::default();
        }

        let mut result = SearchResult::default();
        // Instant terminal fast path (Botzone rules): a full board, a stuck
        // side (including a stuck OPPONENT, which movegen alone would miss),
        // or already past ply 401 ends the game on the spot: zero nodes searched.
        if BZ {
            let (over, hm0, hm1) = b.bz_status();
            if over || self.bz_max_plies == 0 {
                result.score = bz_terminal_score_counts(b, hm0, hm1, 0);
                result.elapsed = self.elapsed();
                result.nodes = self.nodes;
                result.stopped_by = Some(StopReason::Completed);
                return result;
            }
        }
        self.tt.step_age();
        if self.tt.age == 1 {
            // first search of a game (a new game resets the age to 0): pick the starting table stage from the budget
            let expected = match (limits.max_nodes, limits.time) {
                (Some(n), _) => n as usize,
                (None, Some(t)) => (t.as_millis() as usize).saturating_mul(120), // ~120 nodes per ms on a typical core
                _ => usize::MAX, // unknown budget: the full size
            };
            self.tt.start_stage(expected);
            self.ec_start_stage(expected);
        }
        // Age the history table so old cutoffs don't dominate forever.
        for side in self.history.iter_mut() {
            for row in side.iter_mut() {
                for h in row.iter_mut() {
                    *h = (*h * 3) / 4;
                }
            }
        }
        let root_tt = self.tt.probe_e(b.hash, b.empty().count_ones() as u8);
        let tt_mv = root_tt.map(|e| e.mv).unwrap_or(0);
        {
            // Root ordering: score once, drain by pop_best into a fixed
            // array (best-first). Each iteration reorders the array.
            let (kill, hist) = (&self.killers, &self.history);
            let root = &mut self.stack.lists[0];
            gen_and_score_into::<false, false>(b, root, tt_mv, 0, kill, hist, None);
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
        let mut root_banned_snaps = [(0u64, 0u64); 32];
        let mut n_root_banned = 0usize;
        let mut root_banned_overflow = false;
        if BZ && self.bz_start_ply >= 3 && self.bz_base_offset >= 2 {
            let last = self.bz_snaps[self.bz_base_offset];
            for i in 0..=(self.bz_base_offset - 2) {
                if self.bz_snaps[i] == last {
                    let ban = self.bz_snaps[i + 1];
                    if !root_banned_snaps[..n_root_banned].contains(&ban) {
                        if n_root_banned < root_banned_snaps.len() {
                            root_banned_snaps[n_root_banned] = ban;
                            n_root_banned += 1;
                        } else {
                            root_banned_overflow = true;
                        }
                    }
                }
            }
        }
        let n_root = {
            let root = &mut self.stack.lists[0];
            let mut all_moves = [Move::PASS; 256];
            let mut kept = 0usize;
            for slot in all_moves.iter_mut().take(n_root) {
                let m = root.pop_best();
                *slot = m;
                let mut is_banned = banned_root.contains(&m);
                if BZ && !is_banned && !m.is_clone() && n_root_banned > 0 {
                    let c = placement_snap(b, m);
                    is_banned = root_banned_snaps[..n_root_banned].contains(&c)
                        || (root_banned_overflow && {
                            let last = self.bz_snaps[self.bz_base_offset];
                            self.bz_snaps[0..=self.bz_base_offset - 1]
                                .windows(2)
                                .any(|w| w[0] == last && w[1] == c)
                        });
                }
                if !is_banned {
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
                .find(|&i| crate::ttcodec::move_code(root_moves[i]) as u32 == tt_mv)
                .map(|i| root_moves[i])
        } else {
            None
        };
        result.best = tt_legal_mv.or(Some(root_moves[0]));
        if let Some(m) = result.best {
            self.publish_best(m);
        }
        if let Some(entry) = root_tt {
            if entry.score.abs() < MATE_BOUND && (!BZ || (!respect_401 && snaps.is_empty())) {
                result.score = entry.score;
            }
        }

        let mut has_completed_iter = false;

        // Iterative deepening MUST always start at depth 1.
        // Skipping shallow iterations (even if TT has a deep entry) is fatal under hard time
        // limits (e.g. Botzone): if the high-depth iteration is cut short by the deadline before
        // completing, no iteration ever finishes, leaving depth at 0 and blundering with an
        // unverified move. Shallow iterations take <0.1% of search time and guarantee that a
        // verified completed iteration always exists.
        for depth in 1..=limits.max_depth.max(1) {
            self.heuristic_used = false;
            let mut alpha = -INF;
            let mut best_this = None;
            let mut best_score = i32::MIN;
            // Clean-main ID: FULL window every iteration. No aspiration.
            // (Aspiration reverted: narrowed windows changed PVS cutoffs at
            // fixed node budgets -> bench d8 9.4k=>17k, QB5k 45-53-2.-28.)
            self.stack.path_hash[0] = b.hash;
            self.stack.rep_base[0] = 0;
            if BZ {
                self.bz_banned.clear();
                self.bz_ply[0] = BzPly::default();
            }
            for i in 0..n_root {
                let m = root_moves[i];
                let child = b.make(m);
                self.stack.imp[1] = 3;
                self.stack.path_hash[1] = child.hash;
                self.stack.rep_base[1] = if m.is_clone() { 1 } else { 0 };
                if BZ {
                    self.bz_snaps[self.bz_base_offset + 1] = placement_snap(b, m);
                    self.bz_ply[1] = BzPly {
                        rep_base: if m.is_clone() {
                            (self.bz_base_offset + 1) as u32
                        } else {
                            0
                        },
                        ban_start: 0,
                        tainted: false,
                    };
                }
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
                    self.publish_best(m);
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
            // Search on to the deadline (user D662): NO "next iteration would overrun" early stop. A partial iteration is adopted
            // (above, Stockfish rule), so the time up to the deadline is never wasted; check_stop (every 64 nodes) ends the search
            // at deadline_ms = budget - safety_margin_ms.
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

    /// A killer clone depends on its destination only; a jump keeps its origin (the child differs per origin).
    #[test]
    fn killer_key_clone_ignores_source_jump_does_not() {
        use crate::board::Move;
        // destination 16 = (row 2, col 2): sources 8 and 9 are both adjacent (clones), 0 and 2 are two away (jumps)
        let (c1, c2) = (Move { from: 8, to: 16 }, Move { from: 15, to: 16 });
        assert!(c1.is_clone() && c2.is_clone());
        assert_eq!(kill_key(c1), kill_key(c2));
        let (j1, j2) = (Move { from: 0, to: 16 }, Move { from: 2, to: 16 });
        assert!(!j1.is_clone() && !j2.is_clone());
        assert_ne!(kill_key(j1), kill_key(j2));
        assert_ne!(kill_key(c1), kill_key(j1));
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
    fn bz_ply_401_cuts_off_after_black_201st_move() {
        // Black has 4 stones and trails White (5 stones), EXCEPT c5->d6 (clone) captures e6 to lead 5-3.
        // At game_ply == 400 (Black's 201st move, snaps.len() == 401), the judge terminates immediately after
        // ply 401, so that 1-ply capture solves to +MATE - 1 at depth 1. At game_ply == 401 the game is over (0 nodes).
        let board = ".....oo
                     ....o..
                     ..x....
                     .......
                     .......
                     x....o.
                     xx...o. b";
        let b = Board::from_str(board).unwrap();
        assert_eq!(b.counts(), (4, 5));

        let mut se = Searcher::with_tt_bits(16);
        let snaps_400 = vec![(b.occ[0], b.occ[1]); 401];
        let r400 = se.search_bz_game(&b, &no_time(6), &[], &snaps_400, true);
        assert_eq!(r400.score, MATE - 1, "ply-401 winning flip must solve at MATE - 1, got {}", r400.score);
        assert_eq!(r400.depth, 1, "401-ply endgame must stop at depth 1, got {}", r400.depth);
        let mv = r400.best.expect("must return move at game_ply 400");
        let (fb, fw) = b.make(mv).bz_counts();
        assert!(fb > fw, "must play capturing move to win ply 401, got ({fb}, {fw})");

        // Solving near the ply-401 wall must not pollute TT for non-401 searches of the same board.
        let r_unbounded = se.search_bz_game(&b, &no_time(2), &[], &[], false);
        assert!(r_unbounded.score.abs() < MATE_BOUND, "horizon taint leaked into non-401 search: {}", r_unbounded.score);

        // At game_ply == 399 (White's 200th move, bz_max_plies == 2), White must avoid letting Black
        // reach a winning ply-401 board: exact 2-ply horizon finishes at depth 2 with no TT leakage.
        let bw = Board::from_str(
            ".....oo
             ....o..
             ..x....
             .......
             .......
             x....o.
             xx...o. w",
        )
        .unwrap();
        let snaps_399 = vec![(bw.occ[0], bw.occ[1]); 400];
        let mut sew = Searcher::with_tt_bits(16);
        let r399 = sew.search_bz_game(&bw, &no_time(6), &[], &snaps_399, true);
        assert_eq!(r399.depth, 2, "ply-399 exact 401 horizon should complete at depth 2, got {}", r399.depth);
        assert!(r399.score >= MATE_BOUND, "White up 5-4 can preserve ply-401 win in 2 plies, got {}", r399.score);
        let rw_clean = sew.search_bz_game(&bw, &no_time(2), &[], &[], false);
        assert!(rw_clean.score.abs() < MATE_BOUND, "ply-399 horizon leaked into TT: {}", rw_clean.score);

        // At game_ply == 401 (snaps.len() == 402), game already ended on ply 401 -> 0 nodes.
        let snaps_401 = vec![(b.occ[0], b.occ[1]); 402];
        let r401 = se.search_bz_game(&b, &no_time(6), &[], &snaps_401, true);
        assert_eq!(r401.nodes, 0, "past ply 401 must search 0 nodes");
        assert_eq!(r401.score, -MATE, "trailing 4-5 scores as loss at terminal 401");
    }

    #[test]
    fn bz_forces_opponent_into_circular_move_forfeit_trap() {
        // White has 15 stones vs Black's 1 stone, so White wins by massive material EXCEPT its only
        // mobile piece is walled into {a7 (0), c7 (2)}, while Black jumps among {f7 (5), d5 (17), f5 (19)}.
        // After (x@f7,o@a7)->(x@f5,o@a7)->(x@f5,o@c7)->(x@d5,o@c7)->(x@d5,o@a7), Black's jump d5->f5
        // puts White into (x@f5, o@a7) where White's SOLE move a7->c7 repeats (snaps[1], snaps[2])
        // and forfeits White by CIRCULAR_MOVE regardless of stone count.
        let root0 = Board::from_str(
            "o#.##x#
             #######
             ###.#.#
             #######
             #######
             ooooooo
             ooooooo b",
        )
        .unwrap();
        // Moves on root0: f7(5) jumps to d5(17) or f5(19). White a7(0) jumps ONLY to c7(2).
        let j_f7_f5 = Move { from: 5, to: 19 };
        let j_a7_c7 = Move { from: 0, to: 2 };
        let j_f5_d5 = Move { from: 19, to: 17 };
        let j_c7_a7 = Move { from: 2, to: 0 };
        let j_d5_f5 = Move { from: 17, to: 19 };

        let mut cur = root0;
        let mut snaps = vec![(cur.occ[0], cur.occ[1])];
        for mv in [j_f7_f5, j_a7_c7, j_f5_d5, j_c7_a7] {
            assert!(cur.legal_moves().contains(&mv), "illegal setup move {mv:?}");
            snaps.push(placement_snap(&cur, mv));
            cur = cur.make(mv);
        }
        // Now at `cur` (x@d5, o@a7, snaps.len()==5): Black's turn.
        // snaps = [0:(f7,a7), 1:(f5,a7), 2:(f5,c7), 3:(d5,c7), 4:(d5,a7)].
        // If Black plays d5->f5 (cur->(f5,a7) == snaps[1]), White MUST play a7->c7 ((f5,c7) == snaps[2]),
        // which repeats (snaps[1], snaps[2]) and forfeits White by CIRCULAR_MOVE!
        // Note: Black's `d5->f5` is tested both at max_depth=1 (hitting the depth==0 check) and max_depth=3.
        let mut se = Searcher::with_tt_bits(16);
        let r1 = se.search_bz_game(&cur, &no_time(1), &[], &snaps, true);
        assert_eq!(r1.best, Some(j_d5_f5), "depth-1 search must spot forced CIRCULAR_MOVE trap");
        assert_eq!(r1.score, MATE - 1, "depth-1 trap score must be MATE - 1, got {}", r1.score);

        let r3 = se.search_bz_game(&cur, &no_time(3), &[], &snaps, true);
        assert_eq!(r3.best, Some(j_d5_f5), "depth-3 search must spot forced CIRCULAR_MOVE trap");
        assert_eq!(r3.score, MATE - 1, "depth-3 trap score must be MATE - 1, got {}", r3.score);

        // Clean search on the SAME board `cur` without prior trap history must NOT see a cached +MATE!
        let r_clean = se.search_bz_game(&cur, &no_time(2), &[], &[], false);
        assert!(r_clean.score < 0, "TT poisoned by trap search! clean score = {}", r_clean.score);
    }

    #[test]
    fn bz_allows_legal_revisit_when_prior_move_differs() {
        // White (15 stones vs Black's 1) can cycle a7<->c7; Black can jump between f7(5), d5(17), f5(19)
        // with NO clone moves anywhere.
        // At depth 4 from b0 = (x@f7, o@a7):
        //   ply0 B: f7->f5 (s1), ply1 W: a7->c7 (s2), ply2 B: f5->f7 (s3), ply3 W: c7->a7 (s4 == s0).
        // Old path_hash banned ply3 (W c7->a7) 1 ply too early, giving Black a fake +MATE-3!
        // Botzone judge PERMITS s4 because s3=(f7,c7) never appeared before, so White does not forfeit on ply 3.
        let b0 = Board::from_str(
            "o#.##x#
             #######
             ###.#.#
             #######
             #######
             ooooooo
             ooooooo b",
        )
        .unwrap();
        let mut se = Searcher::with_tt_bits(16);
        let r = se.search_bz_game(&b0, &no_time(4), &[], &[], false);
        assert!(
            r.score < 0,
            "legal 4-ply revisit s4 must not trigger fake repetition forfeit mate: score = {}",
            r.score
        );
    }

    #[test]
    fn bz_ban_collector_matches_naive_across_distances_and_overflow() {
        // 1. Random legal jump-heavy BZ sequences from random positions: verify that no alternating
        // non-pass placement-snap sequence ever repeats at distance 1, 2, or 3 even across captures.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_u64 = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut list = MoveList::new();
        for _case in 0..80 {
            let mut b = Board::botzone_start();
            let warmup = (next_u64() % 18) as usize;
            for _ in 0..warmup {
                b.legal_moves_dedup_into(&mut list);
                if list.is_empty() || b.bz_over() {
                    break;
                }
                let idx = (next_u64() as usize) % list.len();
                b = b.make(list.move_at(idx));
            }
            if b.bz_over() {
                continue;
            }
            let mut snaps = vec![(b.occ[0], b.occ[1])];
            let mut rbase = 0usize;
            for _step in 0..40 {
                b.legal_moves_dedup_into(&mut list);
                if list.is_empty() || b.bz_over() {
                    break;
                }
                // Bias heavily toward jumps so same-epoch sequences happen often.
                let mut jumps = Vec::new();
                for i in 0..list.len() {
                    let m = list.move_at(i);
                    if !m.is_clone() {
                        jumps.push(m);
                    }
                }
                let m = if !jumps.is_empty() && (next_u64() % 10 != 0) {
                    jumps[(next_u64() as usize) % jumps.len()]
                } else {
                    list.move_at((next_u64() as usize) % list.len())
                };
                snaps.push(placement_snap(&b, m));
                b = b.make(m);
                let cur = snaps.len() - 1;
                if m.is_clone() {
                    rbase = cur;
                } else if cur >= rbase + 1 {
                    let last = snaps[cur];
                    for prev in rbase..cur {
                        let dist = cur - prev;
                        if dist <= 3 {
                            assert_ne!(
                                snaps[prev], last,
                                "unexpected snapshot repeat at distance {dist}"
                            );
                        }
                    }
                }
            }
        }

        // 2. Directly verify collect_bz_bans_cold + is_bz_snap_banned on >32 banned successors (overflow path)
        // and stack-preserving child truncation.
        let mut se = Searcher::with_tt_bits(12);
        let root_snap = (0x1u64, 0x2u64);
        se.bz_snaps.clear();
        for i in 0..40u64 {
            se.bz_snaps.push(root_snap);
            se.bz_snaps.push((0x100u64 + i, 0x200u64));
        }
        // Final spacing >= 4 before current snapshot.
        se.bz_snaps.push((0x999u64, 0x888u64));
        se.bz_snaps.push((0x99Au64, 0x889u64));
        se.bz_snaps.push((0x99Bu64, 0x88Au64));
        se.bz_snaps.push(root_snap);
        let cur_v = se.bz_snaps.len() - 1;
        // Pre-seed ancestor entry at index 0..1 to ensure child truncation never clobbers ancestors.
        se.bz_banned.clear();
        se.bz_banned.push((0xDEAD, 0xBEEF));
        let (n, overflow) = se.collect_bz_bans_cold(cur_v, 0, 1);
        assert_eq!(n, 32);
        assert!(overflow);
        assert_eq!(se.bz_banned[0], (0xDEAD, 0xBEEF));
        for i in 0..40u64 {
            assert!(
                se.is_bz_snap_banned(cur_v, 0, 1, n, overflow, (0x100u64 + i, 0x200u64)),
                "overflow ban {i} must still be caught"
            );
        }
        assert!(!se.is_bz_snap_banned(cur_v, 0, 1, n, overflow, (0x100u64 + 99, 0x200u64)));
    }

    #[test]
    fn bz_step1c_depth0_unbanned_and_clean_cutoff_stay_untainted() {
        let mut se = Searcher::with_tt_bits(12);
        let b = Board::botzone_start();
        let root_snap = (b.occ[0], b.occ[1]);
        let moves = b.legal_moves();
        let banned_jump = moves
            .iter()
            .find(|m| !m.is_clone())
            .expect("startpos has legal jumps");
        let banned_snap = placement_snap(&b, banned_jump);

        // Set up history at ply=1 (`cur_v = 4`, `rbase = 0`, `bz_start_ply = 4`) where `banned_jump`
        // repeats `bz_snaps[0] -> bz_snaps[1]`, while clones & other jumps remain unbanned.
        se.bz_base_offset = 3;
        se.bz_start_ply = 4;
        se.bz_max_plies = 400;
        se.bz_snaps.clear();
        se.bz_snaps.push(root_snap);
        se.bz_snaps.push(banned_snap);
        se.bz_snaps.push((0xAA, 0xBB));
        se.bz_snaps.push((0xCC, 0xDD));
        se.bz_snaps.push(root_snap); // index 4 == cur_v for ply 1
        se.bz_snaps.resize(4 + MAX_PLY + 2, (0, 0));
        se.bz_ply[1] = BzPly {
            rep_base: 0,
            ban_start: 0,
            tainted: false,
        };

        // 1. Depth-0 leaf with unbanned legal moves must NOT set `tainted = true`.
        let d0_score = se.negamax::<true>(&b, -MATE, MATE, 0, 1, true);
        assert!(d0_score > -MATE_BOUND, "unbanned depth-0 leaf returns static_eval");
        assert!(
            !se.bz_ply[1].tainted,
            "depth-0 static_eval with unbanned moves must not taint parent"
        );

        // 2. Depth-1 search with a clean beta cutoff on an unbanned move must clear taint and store FLAG_LOWER in TT.
        se.tt.clear();
        se.bz_ply[1] = BzPly {
            rep_base: 0,
            ban_start: 0,
            tainted: false,
        };
        let d1_cut = se.negamax::<true>(&b, -25000, -24999, 1, 1, true);
        assert!(d1_cut >= -24999, "must fail high against -24999 beta");
        assert!(
            !se.bz_ply[1].tainted,
            "clean beta cutoff on unbanned move must leave node untainted"
        );
        let tt_hit = se
            .tt
            .probe_e(b.hash, b.empty().count_ones() as u8)
            .expect("clean beta cutoff under n_banned > 0 must store into TT");
        assert_eq!(tt_hit.flag, FLAG_LOWER);
    }

    #[test]
    fn tt_aging_and_fail_low_preservation() {
        let mut tt = Tt::new(10);
        let key = 123456789u64;
        // a real jump (a1 -> c1): the TT hands back its ttcodec move code
        let mv_good = Move { from: 0, to: 2 }.to_u32();
        let code_good = crate::ttcodec::code_of_u32(mv_good) as u32;
        assert!(code_good >= 50);

        // 1. Store an exact bound with a good move
        tt.store(key, mv_good, 150, 6, FLAG_EXACT);
        assert_eq!(tt.probe(key).unwrap().mv, code_good);
        assert_eq!(tt.probe(key).unwrap().score, 150);

        // 2. A deeper search on the same position fails low (FLAG_UPPER)
        tt.store(key, 0xFFFF, 50, 8, FLAG_UPPER);
        // The good move must be preserved!
        assert_eq!(tt.probe(key).unwrap().mv, code_good, "refutation move must be preserved on fail-low");
        assert_eq!(tt.probe(key).unwrap().score, 50);

        // 3. Test age replacement: position from old turn gets evicted by fresh search
        tt.step_age();
        tt.step_age();
        let other_key = key ^ (1 << 10); // same hash slot, different key
        tt.store(other_key, 99, 200, 3, FLAG_EXACT);
        assert_eq!(tt.probe(other_key).unwrap().key, other_key);
    }

    #[test]
    fn size_stages_jump_l1_l2_l3_then_full_never_by_doubling() {
        let c = crate::cacheinfo::FALLBACK; // 32 KB L1d, 256 KB L2, 8 MB L3
        let tt = stage_list(1 << 20, 0, 4, (2, 3), c, 7, 0);
        let ec = stage_list(1 << 19, 0, 8, (1, 3), c, 7, 0);
        assert_eq!(tt, vec![256, 2048, 32768, 1 << 20]);
        assert_eq!(ec, vec![128, 1024, 16384, 1 << 19]);
        for st in [&tt, &ec] {
            for w in st.windows(2) {
                assert!(w[1] >= 8 * w[0], "a stage jump is at least 8x, not a doubling: {st:?}");
            }
            assert!(st.iter().all(|n| n.is_power_of_two()));
        }
        // a big cache never makes a stage larger than the full table; a tiny table collapses the stages
        assert_eq!(stage_list(64, 0, 4, (2, 3), c, 7, 0), vec![64]);
        // explicit start size: two stages, or fixed when it is not smaller than the full size
        assert_eq!(stage_list(1 << 20, 14, 4, (2, 3), c, 7, 0), vec![1 << 12, 1 << 20]);
        assert_eq!(stage_list(1 << 20, 22, 4, (2, 3), c, 7, 0), vec![1 << 20]);
        // fewer stages: only the chosen cache levels, e.g. L2 + L3, L3 only, or none (= the fixed full size)
        assert_eq!(stage_list(1 << 20, 0, 4, (2, 3), c, 6, 0), vec![2048, 32768, 1 << 20]);
        assert_eq!(stage_list(1 << 20, 0, 4, (2, 3), c, 4, 0), vec![32768, 1 << 20]);
        assert_eq!(stage_list(1 << 20, 0, 4, (2, 3), c, 0, 0), vec![1 << 20]);
        // extra jumps are allowed only in RAM (beyond the L3 stage), geometrically spaced
        assert_eq!(stage_list(1 << 20, 0, 4, (2, 3), c, 7, 1), vec![256, 2048, 32768, 131072, 1 << 20]);
        assert_eq!(stage_list(1 << 20, 0, 4, (2, 3), c, 7, 2), vec![256, 2048, 32768, 65536, 262144, 1 << 20]);
    }

    #[test]
    fn start_stage_follows_the_budget_and_never_goes_back() {
        let fresh = || {
            let mut t = Tt::new(22);
            t.stages = vec![256, 2048, 32768, 1 << 20];
            t.stage = 0;
            t.mask = 255;
            t.filled = 0;
            t
        };
        // 6 entries per bucket, the table must hold 4x the expected nodes
        let mut t = fresh();
        t.start_stage(300); // 1,200 <= 256 x 6
        assert_eq!((t.stage, t.mask + 1), (0, 256));
        let mut t = fresh();
        t.start_stage(9_000); // a 100 ms search: 36,000 -> the 32768-bucket (L3-sized) stage
        assert_eq!((t.stage, t.mask + 1), (2, 32768));
        let mut t = fresh();
        t.start_stage(500_000); // the Botzone clock: the full table at once
        assert_eq!((t.stage, t.mask + 1), (3, 1 << 20));
        let mut t = fresh();
        t.start_stage(usize::MAX); // unknown budget: full
        assert_eq!(t.stage, 3);
        // a table that already holds entries keeps its stage; jump_to never goes backwards
        let mut t = fresh();
        t.filled = 1;
        t.start_stage(500_000);
        assert_eq!(t.stage, 0);
        let mut t = fresh();
        t.jump_to(2);
        t.jump_to(1);
        assert_eq!(t.stage, 2);
    }

    #[test]
    fn packed_tt_bucket_round_trips_every_field() {
        let mk = |i: usize| TtE2 {
            chk: 0xDEAD_BEEFu32 ^ (i as u32).wrapping_mul(0x0101_0101),
            score: if i == 3 { MATE - 17 } else if i == 4 { -MATE + 250 } else { -12345 + i as i32 * 7000 },
            code: 529 - i as u16 * 77,
            depth: 250 - i as u8,
            flag: (i % 3) as u8 + 1,
            imp: (i % 4) as u8,
            age: (i * 3 % 16) as u8,
            empties: 49 - i as u8 * 8,
            ext: ((i * 5 + 1) & 31) as u8,
        };
        let mut bk = TtBucket::EMPTY;
        for i in 0..TT_MAX_WAYS {
            assert_eq!(bk.flag(i), 0, "an all-zero slot is empty");
            bk.set(i, &mk(i));
        }
        let check = |bk: &TtBucket| {
            for i in 0..TT_MAX_WAYS {
                let (g, w) = (bk.get(i), mk(i));
                assert_eq!((g.chk, g.score, g.code, g.depth), (w.chk, w.score, w.code, w.depth), "slot {i}");
                assert_eq!((g.flag, g.imp, g.age, g.empties, g.ext), (w.flag, w.imp, w.age, w.empties, w.ext), "slot {i}");
            }
        };
        check(&bk);
        // rewriting the age of one slot and the whole of another leaves the neighbours alone
        bk.set_age(2, 13);
        assert_eq!(bk.age(2), 13);
        let mut e = bk.get(2);
        e.age = mk(2).age;
        bk.set(2, &e);
        check(&bk);
        bk.set(5, &mk(5));
        check(&bk);
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

    #[test]
    fn gen_and_score_matches_reference() {
        let mut x: u64 = 0xD1B5_4A32_D192_ED03;
        let next_u64 = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };
        let mut killers = [(0u32, 0u32); MAX_PLY];
        let mut history = Box::new([[[0i32; 49]; 49]; 2]);
        for s in 0..2 {
            for f in 0..49 {
                for t in 0..49 {
                    history[s][f][t] = ((next_u64(&mut x) % (2 * HIST_MAX as u64)) as i64 - HIST_MAX as i64) as i32;
                }
            }
        }
        let pol = crate::policy::bundled().expect("bundled policy");
        let pol_w = 100_000.0f32;
        let mut ref_list = MoveList::new();
        let mut new_list = MoveList::new();

        for n in 0..200 {
            let mut b = Board::start();
            for _ in 0..(n % 22) {
                b.legal_moves_dedup_into(&mut ref_list);
                if ref_list.is_empty() || b.game_over() {
                    break;
                }
                let idx = (next_u64(&mut x) as usize) % ref_list.len();
                b = b.make(ref_list.move_at(idx));
            }
            if b.game_over() {
                continue;
            }
            let ply = (n % 16) as u32;
            b.legal_moves_dedup_into(&mut ref_list);
            if ref_list.is_empty() {
                continue;
            }
            let tt_mv = if n % 3 == 0 {
                0
            } else {
                crate::ttcodec::move_code(ref_list.move_at((next_u64(&mut x) as usize) % ref_list.len())) as u32
            };
            let k1 = kill_key(ref_list.move_at((next_u64(&mut x) as usize) % ref_list.len()));
            let k2 = kill_key(ref_list.move_at((next_u64(&mut x) as usize) % ref_list.len()));
            killers[ply as usize] = (k1, k2);

            for with_pol in [false, true] {
                let mut kc = crate::policy::KCache::new();
                let mut expected: Vec<(Move, i32)> = Vec::new();
                let opp = b.opp();
                let me = b.occ[b.turn as usize];
                let hist = &history[b.turn as usize];
                let mut tt_pos = None;
                for m in ref_list.iter() {
                    if tt_mv != 0 && crate::ttcodec::move_code(m) as u32 == tt_mv {
                        tt_pos = Some(expected.len());
                        expected.push((m, ORD_TT));
                        continue;
                    }
                    let mut sc = (opp & RING1[m.to as usize]).count_ones() as i32 * ORD_CAPTURE_UNIT;
                    if m.is_clone() {
                        sc += ORD_CLONE_BONUS;
                    } else {
                        let conn = (me & RING1[m.from as usize]).count_ones() as i32;
                        sc -= ORD_JUMP_PEN + conn * ORD_JUMP_CONN;
                    }
                    let empty = b.empty();
                    if ORD_PLUG_HOLE_UNIT > 0 && ((reach_union(opp) & empty) & (1u64 << m.to as usize)) != 0 {
                        let threatened = (me & RING1[m.to as usize]).count_ones() as i32;
                        if threatened >= 2 {
                            sc += (threatened - 1) * ORD_PLUG_HOLE_UNIT;
                        }
                    }
                    if ORD_EVAC_UNIT > 0 && !m.is_clone() {
                        let opp_reach = reach_union(opp) & empty;
                        let r1_from = RING1[m.from as usize];
                        let danger_holes = opp_reach & r1_from;
                        if danger_holes != 0 {
                            let mut dh = danger_holes;
                            let mut max_threat = 0i32;
                            while dh != 0 {
                                let h = dh.trailing_zeros() as usize;
                                dh &= dh - 1;
                                let t = (me & RING1[h]).count_ones() as i32;
                                if t > max_threat {
                                    max_threat = t;
                                }
                            }
                            if max_threat >= 2 {
                                sc += (max_threat - 1) * ORD_EVAC_UNIT;
                            }
                        }
                    }
                    let kk = kill_key(m);
                    if kk == k1 || kk == k2 {
                        sc += ORD_KILLER;
                    }
                    let (hf, ht) = hist_key(m);
                    sc += hist[hf][ht] / HIST_DIV;
                    if with_pol {
                        let add = (pol.logit_k(&b, m, &[], &mut kc) * pol_w) as i32;
                        sc = sc.saturating_add(add);
                    }
                    expected.push((m, sc));
                }

                // Check SKIP_TT = true (all non-TT moves in identical order with identical scores)
                if with_pol {
                    gen_and_score_into::<true, true>(&b, &mut new_list, tt_mv, ply, &killers, &history, Some((pol, &[], pol_w)));
                } else {
                    gen_and_score_into::<true, false>(&b, &mut new_list, tt_mv, ply, &killers, &history, None);
                }
                let expected_skip: Vec<(Move, i32)> = expected.iter().copied().enumerate().filter(|(i, _)| Some(*i) != tt_pos).map(|(_, v)| v).collect();
                let actual_skip: Vec<(Move, i32)> = (0..new_list.len()).map(|i| (new_list.move_at(i), new_list.score_at(i))).collect();
                assert_eq!(actual_skip, expected_skip, "SKIP_TT=true mismatch n={n} with_pol={with_pol}");

                // Check SKIP_TT = false (TT swapped to index 0)
                if !with_pol {
                    gen_and_score_into::<false, false>(&b, &mut new_list, tt_mv, ply, &killers, &history, None);
                    let mut expected_noskip = expected.clone();
                    if let Some(tp) = tt_pos {
                        if tp != 0 {
                            expected_noskip.swap(tp, 0);
                        }
                    }
                    let actual_noskip: Vec<(Move, i32)> = (0..new_list.len()).map(|i| (new_list.move_at(i), new_list.score_at(i))).collect();
                    assert_eq!(actual_noskip, expected_noskip, "SKIP_TT=false mismatch n={n}");
                }
            }
        }
    }
}
