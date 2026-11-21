//! Experimental PUCT + D4-canonicalized Monte Carlo Graph Search (MCGS) using
//! Titanium's full synchronized stack: `nnue_main.tnn` (value + disagreement head),
//! `policy_main.bin` (`TPOL3` priors), and `fut_small.tsn` (`SmallNet` child bounds).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::time::Instant;

use crate::board::{dist_union, jump_union, Board, Move, RING1, RING2};
use crate::nnue::Nnue;
use crate::policy::{k3code_pub, Policy};
use crate::search::{evaluate, SearchLimits, SearchResult, StopReason, MATE};
use crate::smallnet::SmallNet;

fn small_net() -> &'static SmallNet {
    static SN: OnceLock<SmallNet> = OnceLock::new();
    SN.get_or_init(SmallNet::builtin)
}

/// Fast deterministic SplitMix64 / FxHash hasher (zero external dependencies).
#[derive(Default)]
pub struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        let mut z = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks_exact(8) {
            let v = u64::from_le_bytes(chunk.try_into().unwrap());
            self.write_u64(v);
        }
        let rem = bytes.chunks_exact(8).remainder();
        for &b in rem {
            self.0 = (self.0.rotate_left(5) ^ (b as u64)).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.0 = (self.0.rotate_left(5) ^ (i as u64)).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

// ---------------------------------------------------------------- D4 symmetry tables

const fn map_rc(r: usize, c: usize, sym: usize) -> (usize, usize) {
    match sym {
        0 => (r, c),         // id
        1 => (c, 6 - r),     // rot90 CW
        2 => (6 - r, 6 - c), // rot180
        3 => (6 - c, r),     // rot270 CW
        4 => (r, 6 - c),     // flipH
        5 => (6 - r, c),     // flipV
        6 => (c, r),         // diag transpose
        7 => (6 - c, 6 - r), // antidiag transpose
        _ => (r, c),
    }
}

/// Forward square map: `SYM_SQ[sym][sq]` maps a concrete square to symmetry `sym`.
pub static SYM_SQ: [[u8; 49]; 8] = {
    let mut t = [[0u8; 49]; 8];
    let mut s = 0;
    while s < 8 {
        let mut sq = 0;
        while sq < 49 {
            let (r, c) = map_rc(sq / 7, sq % 7, s);
            t[s][sq] = (r * 7 + c) as u8;
            sq += 1;
        }
        s += 1;
    }
    t
};

/// Inverse square map: `INV_SYM_SQ[sym][sym_sq]` maps canonical back to concrete.
pub static INV_SYM_SQ: [[u8; 49]; 8] = {
    let mut inv = [[0u8; 49]; 8];
    let mut s = 0;
    while s < 8 {
        let mut sq = 0;
        while sq < 49 {
            inv[s][SYM_SQ[s][sq] as usize] = sq as u8;
            sq += 1;
        }
        s += 1;
    }
    inv
};

/// Row-chunk bitboard lookup table: 7 shifts/loads transform all 49 bits branch-free.
static SYM_ROW_LUT: [[[u64; 128]; 7]; 8] = {
    let mut lut = [[[0u64; 128]; 7]; 8];
    let mut s = 0;
    while s < 8 {
        let mut r = 0;
        while r < 7 {
            let mut m = 0;
            while m < 128 {
                let mut out = 0u64;
                let mut c = 0;
                while c < 7 {
                    if (m >> c) & 1 != 0 {
                        out |= 1u64 << SYM_SQ[s][r * 7 + c];
                    }
                    c += 1;
                }
                lut[s][r][m] = out;
                m += 1;
            }
            r += 1;
        }
        s += 1;
    }
    lut
};

#[inline(always)]
pub fn transform_bb(bb: u64, sym: usize) -> u64 {
    let l = &SYM_ROW_LUT[sym];
    l[0][(bb & 127) as usize]
        | l[1][((bb >> 7) & 127) as usize]
        | l[2][((bb >> 14) & 127) as usize]
        | l[3][((bb >> 21) & 127) as usize]
        | l[4][((bb >> 28) & 127) as usize]
        | l[5][((bb >> 35) & 127) as usize]
        | l[6][((bb >> 42) & 127) as usize]
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CanonKey {
    pub me: u64,
    pub opp: u64,
    pub blk: u64,
    pub passes: u8,
}

/// Canonicalize `b` (from side-to-move's viewpoint) over all 8 D4 symmetries.
/// Returns `(CanonKey, sym_id)` where `SYM_SQ[sym_id]` maps `b -> key` and
/// `INV_SYM_SQ[sym_id]` maps `key -> b`.
#[inline]
pub fn canonicalize_board(b: &Board, classic_passes: bool) -> (CanonKey, usize) {
    let (me, opp, blk) = (b.me(), b.opp(), b.blockers);
    let passes = if classic_passes { b.passes } else { 0 };
    let mut best_sym = 0usize;
    let mut best = (me, opp, blk);
    for s in 1..8 {
        let t_me = transform_bb(me, s);
        if t_me > best.0 {
            continue;
        }
        let t_opp = transform_bb(opp, s);
        if t_me == best.0 && t_opp > best.1 {
            continue;
        }
        let t_blk = if blk == 0 { 0 } else { transform_bb(blk, s) };
        let cand = (t_me, t_opp, t_blk);
        if cand < best {
            best = cand;
            best_sym = s;
        }
    }
    (
        CanonKey {
            me: best.0,
            opp: best.1,
            blk: best.2,
            passes,
        },
        best_sym,
    )
}

#[inline(always)]
fn map_move(m: Move, sq_map: &[u8; 49]) -> Move {
    if m.is_pass() {
        Move::PASS
    } else {
        Move {
            from: sq_map[m.from as usize],
            to: sq_map[m.to as usize],
        }
    }
}

/// Construct a consistent canonical `Board` (`turn = 0`) WITHOUT paying the
/// 49-stone Zobrist/PST setup loop (neither `Nnue`, `Policy`, nor movegen reads them).
#[inline(always)]
fn light_board(key: &CanonKey) -> Board {
    Board {
        occ: [key.me, key.opp],
        blockers: key.blk,
        turn: 0,
        passes: key.passes,
        ply: 0,
        piece_cnt: [key.me.count_ones() as u8, key.opp.count_ones() as u8],
        blocker_cnt: key.blk.count_ones() as u8,
        hash: 0,
        pst: 0,
    }
}

// ---------------------------------------------------------------- MCGS data structures

#[derive(Clone, Debug)]
pub struct McgsParams {
    pub cpuct_init: f32,
    pub cpuct_factor: f32,
    pub cpuct_base: f32,
    pub fpu_reduction: f32,
    pub root_fpu_reduction: f32,
    pub policy_temp: f32,
    pub dis_cpuct_scale: f32,
    pub dis_leaf_decay: f32,
    pub sn_prior_weight: f32,
    pub lcb_z: f32,
    pub max_nodes: usize,
}

impl Default for McgsParams {
    fn default() -> Self {
        Self {
            cpuct_init: 1.25,
            cpuct_factor: 0.5,
            cpuct_base: 19652.0,
            fpu_reduction: 0.25,
            root_fpu_reduction: 0.10,
            policy_temp: 1.0,
            dis_cpuct_scale: 0.45,
            dis_leaf_decay: 0.50,
            sn_prior_weight: 0.0,
            lcb_z: 0.50,
            max_nodes: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeState {
    Ongoing,
    ProvenWin(u16),
    ProvenLoss(u16),
    ProvenDraw,
}

#[derive(Clone, Copy)]
struct McgsEdge {
    mv: Move, // in this node's canonical coordinates
    prior: f32,
    child: u32, // u32::MAX if unlinked
    edge_visits: u32,
    last_child_q: f32,
}

struct McgsNode {
    key: CanonKey,
    state: NodeState,
    eval_cp: i16,
    dis_cp: u16,
    visits: u32,
    w_total: f64,
    w_sum: f64,
    q: f32,
    edge_start: u32,
    edge_len: u16,
    expanded: bool,
}

#[inline(always)]
fn cp_to_q(cp: i32) -> f32 {
    let p = 1.0 / (1.0 + (-(cp as f32) / 400.0).exp());
    (2.0 * p - 1.0).clamp(-0.999_99, 0.999_99)
}

#[inline(always)]
fn q_to_cp(q: f32) -> i32 {
    let p = (0.5 * (q as f64 + 1.0)).clamp(1e-5, 1.0 - 1e-5);
    (400.0 * (p / (1.0 - p)).ln()).round().clamp(-3000.0, 3000.0) as i32
}

/// Terminal check from `b.turn`'s perspective (works cleanly for both canonical
/// `turn = 0` parents and `turn = 1` children).
#[inline]
fn eval_terminal_board(b: &Board, passes: u8, bz: bool) -> Option<(NodeState, f32)> {
    let stm = b.turn as usize;
    if bz {
        let (over, hm0, hm1) = b.bz_status();
        if over {
            let (c0, c1) = b.bz_counts_from(hm0, hm1);
            let (mc, oc) = if stm == 0 { (c0, c1) } else { (c1, c0) };
            return Some(match mc.cmp(&oc) {
                std::cmp::Ordering::Greater => (NodeState::ProvenWin(0), 1.0),
                std::cmp::Ordering::Less => (NodeState::ProvenLoss(0), -1.0),
                std::cmp::Ordering::Equal => (NodeState::ProvenDraw, 0.0),
            });
        }
        None
    } else if b.is_over() || passes >= 2 || b.stuck_both() {
        let (c0, c1) = b.counts();
        let (mc, oc) = if stm == 0 { (c0, c1) } else { (c1, c0) };
        Some(match mc.cmp(&oc) {
            std::cmp::Ordering::Greater => (NodeState::ProvenWin(0), 1.0),
            std::cmp::Ordering::Less => (NodeState::ProvenLoss(0), -1.0),
            std::cmp::Ordering::Equal => (NodeState::ProvenDraw, 0.0),
        })
    } else {
        None
    }
}

#[inline]
fn has_bz_ban(hist: &[(u64, u64)], snap: (u64, u64)) -> bool {
    let len = hist.len();
    if len < 4 {
        return false;
    }
    let last = hist[len - 1];
    // Distances 1..=3 cannot repeat (matches search.rs::collect_bz_bans_cold).
    for i in 0..len - 3 {
        if hist[i] == last && hist[i + 1] == snap {
            return true;
        }
    }
    false
}

pub struct McgsSearcher {
    pub params: McgsParams,
    nodes: Vec<McgsNode>,
    edges: Vec<McgsEdge>,
    table: FastMap<CanonKey, u32>,
    last_bz: Option<bool>,
    // Reused scratch buffers across playouts (zero heap churn).
    path_edges: Vec<(u32, u32, bool)>, // (node_idx, edge_idx, had_bz_ban)
    path_nodes: Vec<u32>,
    path_snaps: Vec<(u64, u64)>,
    tmp_moves: Vec<Move>,
    logits: Vec<f32>,
}

impl McgsSearcher {
    pub fn new(params: McgsParams) -> Self {
        Self {
            params,
            nodes: Vec::with_capacity(65_536),
            edges: Vec::with_capacity(524_288),
            table: FastMap::default(),
            last_bz: None,
            path_edges: Vec::with_capacity(128),
            path_nodes: Vec::with_capacity(128),
            path_snaps: Vec::with_capacity(512),
            tmp_moves: Vec::with_capacity(256),
            logits: Vec::with_capacity(256),
        }
    }

    pub fn reset_for_new_game(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.table.clear();
    }

    pub fn graph_len(&self) -> usize {
        self.nodes.len()
    }

    fn alloc_or_get(&mut self, key: CanonKey, bz: bool, nn: &Nnue) -> u32 {
        if let Some(&idx) = self.table.get(&key) {
            return idx;
        }
        let b = light_board(&key);
        let (state, v, eval_cp, dis_cp, self_w) =
            if let Some((st, tv)) = eval_terminal_board(&b, key.passes, bz) {
                (st, tv, 0i16, 0u16, 1.0f64)
            } else {
                let (cp, dis) = nn.eval_with_dis(&b, 0);
                let d_u = dis.clamp(0, u16::MAX as i32) as u16;
                let dis_norm = (d_u as f32 / 400.0).clamp(0.0, 2.0);
                // Downweight volatile NNUE leaves when the disagreement head flags high uncertainty.
                let w = (1.0 / (1.0 + self.params.dis_leaf_decay * dis_norm)).clamp(0.35, 1.0) as f64;
                (
                    NodeState::Ongoing,
                    cp_to_q(cp),
                    cp.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    d_u,
                    w,
                )
            };
        let idx = self.nodes.len() as u32;
        self.nodes.push(McgsNode {
            key,
            state,
            eval_cp,
            dis_cp,
            visits: 1,
            w_total: self_w,
            w_sum: self_w * (v as f64),
            q: v,
            edge_start: 0,
            edge_len: 0,
            expanded: false,
        });
        self.table.insert(key, idx);
        idx
    }

    /// Generate deduplicated legal moves and score them with Titanium's fused
    /// `TPOL3` policy loop + optional `SmallNet` child-bound priors & 1-ply terminal checks.
    fn expand_node(&mut self, node_idx: u32, bz: bool, pol: Option<&Policy>) {
        if self.nodes[node_idx as usize].expanded {
            return;
        }
        let (key, node_eval_cp, state) = {
            let n = &self.nodes[node_idx as usize];
            (n.key, n.eval_cp as i32, n.state)
        };
        if state != NodeState::Ongoing {
            self.nodes[node_idx as usize].expanded = true;
            return;
        }

        let use_sn = self.params.sn_prior_weight > 0.0;
        let b = if use_sn {
            Board::from_occ(key.me, key.opp, key.blk, 0)
        } else {
            light_board(&key)
        };
        let me = b.occ[0];
        let opp = b.occ[1];
        let empty = b.empty();
        let mut targets = (dist_union(me, 1) | jump_union(me)) & empty;

        let edge_start = self.edges.len() as u32;
        if targets == 0 {
            if !bz {
                self.edges.push(McgsEdge {
                    mv: Move::PASS,
                    prior: 1.0,
                    child: u32::MAX,
                    edge_visits: 0,
                    last_child_q: 0.0,
                });
                self.nodes[node_idx as usize].edge_start = edge_start;
                self.nodes[node_idx as usize].edge_len = 1;
            }
            self.nodes[node_idx as usize].expanded = true;
            return;
        }

        self.tmp_moves.clear();
        self.logits.clear();

        let sn = use_sn.then(small_net);
        let sn_pa = sn.map(|s| s.acc(me, opp));
        let sn_w = self.params.sn_prior_weight;
        let mut mx = f32::NEG_INFINITY;

        if let Some(pol) = pol {
            // D1494 port: the policy scoring moved to the per-position Head (TPOL6 phase buckets)
            let p = pol.head(&b);
            let inv_t = 1.0 / self.params.policy_temp.max(0.05);
            let mut orig_valid: u64 = 0;
            let mut orig_z = [MaybeUninit::<f32>::uninit(); 49];

            while targets != 0 {
                let to = targets.trailing_zeros() as u8;
                targets &= targets - 1;
                let tu = to as usize;
                let r1_to = RING1[tu];
                let clones = me & r1_to;
                let jumps = me & RING2[tu];

                let flips = ((opp & r1_to).count_ones() as usize).min(8);
                let tholes = ((r1_to & empty).count_ones() as usize).min(8);
                let k3_to = k3code_pub(me, opp, to);

                if clones != 0 {
                    let from = clones.trailing_zeros() as u8;
                    let mv = Move { from, to };
                    let mut z = p.clone_logit_pre(to, flips, tholes, k3_to) * inv_t;
                    if let (Some(sn_ref), Some(pa)) = (sn, sn_pa.as_ref()) {
                        let child = b.make_via_lut(mv);
                        match eval_terminal_board(&child, 0, bz) {
                            Some((NodeState::ProvenLoss(_), _)) => z += 8.0,
                            Some((NodeState::ProvenWin(_), _)) => z -= 8.0,
                            _ => {
                                let hand = evaluate(&child, None, false);
                                let lo = sn_ref.child_bound(
                                    pa,
                                    me,
                                    opp,
                                    child.occ[1],
                                    child.occ[0],
                                    child.empty().count_ones(),
                                    node_eval_cp,
                                    hand,
                                );
                                z += sn_w * ((-lo as f32) / 400.0).clamp(-4.0, 4.0);
                            }
                        }
                    }
                    mx = mx.max(z);
                    self.tmp_moves.push(mv);
                    self.logits.push(z);
                }
                if jumps != 0 {
                    let j_dest = p.jump_dest_parts(to, flips, tholes, k3_to);
                    let mut js = jumps;
                    while js != 0 {
                        let from = js.trailing_zeros() as u8;
                        js &= js - 1;
                        let fu = from as usize;
                        let fbit = 1u64 << from;
                        if orig_valid & fbit == 0 {
                            orig_valid |= fbit;
                            let fh = ((RING1[fu] & empty).count_ones() as usize).min(8);
                            let k3f = k3code_pub(me, opp, from);
                            orig_z[fu].write(p.jump_orig_parts(from, fh, k3f));
                        }
                        // SAFETY: `orig_valid & fbit != 0` ensures `orig_z[fu]` is initialized.
                        let j_orig = unsafe { orig_z[fu].assume_init() };
                        let mv = Move { from, to };
                        let mut z = p.jump_finish(j_dest, j_orig) * inv_t;
                        if let (Some(sn_ref), Some(pa)) = (sn, sn_pa.as_ref()) {
                            let child = b.make_via_lut(mv);
                            match eval_terminal_board(&child, 0, bz) {
                                Some((NodeState::ProvenLoss(_), _)) => z += 8.0,
                                Some((NodeState::ProvenWin(_), _)) => z -= 8.0,
                                _ => {
                                    let hand = evaluate(&child, None, false);
                                    let lo = sn_ref.child_bound(
                                        pa,
                                        me,
                                        opp,
                                        child.occ[1],
                                        child.occ[0],
                                        child.empty().count_ones(),
                                        node_eval_cp,
                                        hand,
                                    );
                                    z += sn_w * ((-lo as f32) / 400.0).clamp(-4.0, 4.0);
                                }
                            }
                        }
                        mx = mx.max(z);
                        self.tmp_moves.push(mv);
                        self.logits.push(z);
                    }
                }
            }
        } else {
            while targets != 0 {
                let to = targets.trailing_zeros() as u8;
                targets &= targets - 1;
                let tu = to as usize;
                let clones = me & RING1[tu];
                if clones != 0 {
                    let mv = Move {
                        from: clones.trailing_zeros() as u8,
                        to,
                    };
                    let mut z = 0.0f32;
                    if let (Some(sn_ref), Some(pa)) = (sn, sn_pa.as_ref()) {
                        let child = b.make_via_lut(mv);
                        match eval_terminal_board(&child, 0, bz) {
                            Some((NodeState::ProvenLoss(_), _)) => z += 8.0,
                            Some((NodeState::ProvenWin(_), _)) => z -= 8.0,
                            _ => {
                                let hand = evaluate(&child, None, false);
                                let lo = sn_ref.child_bound(
                                    pa,
                                    me,
                                    opp,
                                    child.occ[1],
                                    child.occ[0],
                                    child.empty().count_ones(),
                                    node_eval_cp,
                                    hand,
                                );
                                z += sn_w * ((-lo as f32) / 400.0).clamp(-4.0, 4.0);
                            }
                        }
                    }
                    mx = mx.max(z);
                    self.tmp_moves.push(mv);
                    self.logits.push(z);
                }
                let mut js = me & RING2[tu];
                while js != 0 {
                    let from = js.trailing_zeros() as u8;
                    js &= js - 1;
                    let mv = Move { from, to };
                    let mut z = 0.0f32;
                    if let (Some(sn_ref), Some(pa)) = (sn, sn_pa.as_ref()) {
                        let child = b.make_via_lut(mv);
                        match eval_terminal_board(&child, 0, bz) {
                            Some((NodeState::ProvenLoss(_), _)) => z += 8.0,
                            Some((NodeState::ProvenWin(_), _)) => z -= 8.0,
                            _ => {
                                let hand = evaluate(&child, None, false);
                                let lo = sn_ref.child_bound(
                                    pa,
                                    me,
                                    opp,
                                    child.occ[1],
                                    child.occ[0],
                                    child.empty().count_ones(),
                                    node_eval_cp,
                                    hand,
                                );
                                z += sn_w * ((-lo as f32) / 400.0).clamp(-4.0, 4.0);
                            }
                        }
                    }
                    mx = mx.max(z);
                    self.tmp_moves.push(mv);
                    self.logits.push(z);
                }
            }
        }

        let n = self.tmp_moves.len();
        let mut sum = 0.0f32;
        for z in &mut self.logits {
            *z = (*z - mx).exp();
            sum += *z;
        }
        let inv_s = if sum > 1e-12 { 1.0 / sum } else { 1.0 / n as f32 };
        for i in 0..n {
            let prior = if sum > 1e-12 { self.logits[i] * inv_s } else { inv_s };
            self.edges.push(McgsEdge {
                mv: self.tmp_moves[i],
                prior,
                child: u32::MAX,
                edge_visits: 0,
                last_child_q: 0.0,
            });
        }
        self.nodes[node_idx as usize].edge_start = edge_start;
        self.nodes[node_idx as usize].edge_len = n as u16;
        self.nodes[node_idx as usize].expanded = true;
    }

    fn pick_edge(
        &mut self,
        node_idx: u32,
        cur_b: &Board,
        sym: usize,
        bz: bool,
        is_root: bool,
    ) -> (Option<u32>, bool) {
        let (start, len, dis_cp, mut w_sum, w_total, is_ongoing, mut node_q) = {
            let n = &self.nodes[node_idx as usize];
            (
                n.edge_start as usize,
                n.edge_len as usize,
                n.dis_cp,
                n.w_sum,
                n.w_total,
                n.state == NodeState::Ongoing,
                n.q,
            )
        };
        if len == 0 {
            return (None, false);
        }

        // Pass 1: live stale-sibling Q reconciliation + child visit/policy mass tally.
        let mut child_sum = 0u32;
        let mut vis_mass = 0.0f32;
        let mut w_delta = 0.0f64;
        for e in &mut self.edges[start..start + len] {
            if e.edge_visits > 0 {
                if e.child != u32::MAX {
                    let cq = self.nodes[e.child as usize].q;
                    if (cq - e.last_child_q).abs() > 1e-6 {
                        w_delta += (e.edge_visits as f64) * ((e.last_child_q as f64) - (cq as f64));
                        e.last_child_q = cq;
                    }
                }
                child_sum += e.edge_visits;
                vis_mass += e.prior;
            }
        }
        if w_delta != 0.0 {
            w_sum += w_delta;
            self.nodes[node_idx as usize].w_sum = w_sum;
            if is_ongoing && w_total > 0.0 {
                node_q = (w_sum / w_total) as f32;
                self.nodes[node_idx as usize].q = node_q;
            }
        }

        // Disagreement-head adaptive exploration & FPU scaling.
        let dis_norm = (dis_cp as f32 / 400.0).clamp(0.0, 2.0);
        let fpu_base = if is_root {
            self.params.root_fpu_reduction
        } else {
            self.params.fpu_reduction
        };
        let fpu_red = fpu_base / (1.0 + 0.5 * dis_norm);
        let fpu = (node_q - fpu_red * vis_mass.max(0.0).sqrt()).clamp(-1.0, 1.0);
        let cpuct_base_val = self.params.cpuct_init
            + self.params.cpuct_factor
                * ((child_sum as f32 + self.params.cpuct_base + 1.0) / self.params.cpuct_base)
                    .ln();
        let dis_mult = (1.0 + self.params.dis_cpuct_scale * (dis_norm - 0.30)).clamp(0.65, 1.85);
        let cpuct = cpuct_base_val * dis_mult;
        let sqrt_tot = (child_sum.max(1) as f32).sqrt();

        let mut had_ban = false;
        let mut proven_win_ei: Option<u32> = None;
        let mut min_win_plies = u16::MAX;
        let mut best_norm_ei: Option<u32> = None;
        let mut best_norm_score = f32::NEG_INFINITY;
        let mut best_loss_ei: Option<u32> = None;
        let mut max_loss_plies = 0u16;

        for (i, e) in self.edges[start..start + len].iter().enumerate() {
            if bz && !e.mv.is_pass() {
                let cm = map_move(e.mv, &INV_SYM_SQ[sym]);
                let snap = cur_b.placement_snap(cm);
                if has_bz_ban(&self.path_snaps, snap) {
                    had_ban = true;
                    continue;
                }
            }
            let ei = (start + i) as u32;
            if e.child != u32::MAX {
                match self.nodes[e.child as usize].state {
                    NodeState::ProvenLoss(d) => {
                        if d < min_win_plies {
                            min_win_plies = d;
                            proven_win_ei = Some(ei);
                        }
                        if d == 0 {
                            return (Some(ei), had_ban);
                        }
                    }
                    NodeState::ProvenWin(d) => {
                        if best_loss_ei.is_none() || d > max_loss_plies {
                            max_loss_plies = d;
                            best_loss_ei = Some(ei);
                        }
                        continue;
                    }
                    _ => {}
                }
            }
            let q = if e.edge_visits > 0 || (e.child != u32::MAX && self.nodes[e.child as usize].visits > 0) {
                -self.nodes[e.child as usize].q
            } else {
                fpu
            };
            let u = cpuct * e.prior * sqrt_tot / (1.0 + e.edge_visits as f32);
            let s = q + u;
            if s > best_norm_score {
                best_norm_score = s;
                best_norm_ei = Some(ei);
            }
        }
        if proven_win_ei.is_some() {
            return (proven_win_ei, had_ban);
        }
        (best_norm_ei.or(best_loss_ei), had_ban)
    }

    fn try_promote_proof(&mut self, p_idx: u32, e_idx: u32, c_idx: u32, had_ban: bool, bz: bool) {
        let c_state = self.nodes[c_idx as usize].state;
        let mv = self.edges[e_idx as usize].mv;
        match c_state {
            NodeState::ProvenLoss(d) if !bz || mv.is_clone() || d == 0 => {
                let nd = d.saturating_add(1);
                let p = &mut self.nodes[p_idx as usize];
                let improve = match p.state {
                    NodeState::Ongoing | NodeState::ProvenDraw | NodeState::ProvenLoss(_) => true,
                    NodeState::ProvenWin(old) => nd < old,
                };
                if improve {
                    p.state = NodeState::ProvenWin(nd);
                    p.q = 1.0;
                }
            }
            NodeState::ProvenWin(_) if !had_ban && self.nodes[p_idx as usize].expanded => {
                let p = &self.nodes[p_idx as usize];
                if !matches!(p.state, NodeState::Ongoing) {
                    return;
                }
                let start = p.edge_start as usize;
                let len = p.edge_len as usize;
                if len == 0 {
                    return;
                }
                let mut max_dc = 0u16;
                for edge in &self.edges[start..start + len] {
                    if edge.child == u32::MAX {
                        return;
                    }
                    match self.nodes[edge.child as usize].state {
                        NodeState::ProvenWin(dc) if !bz || edge.mv.is_clone() || dc == 0 => {
                            max_dc = max_dc.max(dc);
                        }
                        _ => return,
                    }
                }
                let p_mut = &mut self.nodes[p_idx as usize];
                p_mut.state = NodeState::ProvenLoss(max_dc.saturating_add(1));
                p_mut.q = -1.0;
            }
            _ => {}
        }
    }

    fn backup_normal(&mut self, leaf_idx: u32, bz: bool) {
        let mut prev_child = leaf_idx;
        let mut k = self.path_edges.len();
        while k > 0 {
            k -= 1;
            let (p_idx, e_idx, had_ban) = self.path_edges[k];
            let c_q = self.nodes[prev_child as usize].q;
            let e = &mut self.edges[e_idx as usize];
            debug_assert!(e.child == prev_child || e.child == u32::MAX);
            e.child = prev_child;
            let old_contrib = (e.edge_visits as f64) * (-(e.last_child_q as f64));
            e.edge_visits += 1;
            let new_contrib = (e.edge_visits as f64) * (-(c_q as f64));
            e.last_child_q = c_q;
            let p = &mut self.nodes[p_idx as usize];
            p.w_sum += new_contrib - old_contrib;
            p.w_total += 1.0;
            p.visits += 1;
            if p.state == NodeState::Ongoing {
                p.q = (p.w_sum / p.w_total) as f32;
            }
            self.try_promote_proof(p_idx, e_idx, prev_child, had_ban, bz);
            prev_child = p_idx;
        }
    }

    /// Path-specific leaf (cycle / 401-ply horizon / Botzone all-moves-banned):
    /// updates `parent` directly with `path_v_parent` without altering
    /// `edge_visits` or `last_child_q` on the path-terminal edge, then
    /// propagates exact KataGo delta updates up the ancestors above `parent`.
    fn backup_path_scalar(&mut self, parent_idx: u32, path_v_parent: f32) {
        let p = &mut self.nodes[parent_idx as usize];
        p.w_sum += path_v_parent as f64;
        p.w_total += 1.0;
        p.visits += 1;
        if p.state == NodeState::Ongoing {
            p.q = (p.w_sum / p.w_total) as f32;
        }

        let mut prev_child = parent_idx;
        let mut k = self.path_edges.len();
        while k > 0 {
            k -= 1;
            let (anc_idx, e_idx, _had_ban) = self.path_edges[k];
            let c_q = self.nodes[prev_child as usize].q;
            let e = &mut self.edges[e_idx as usize];
            let old_contrib = (e.edge_visits as f64) * (-(e.last_child_q as f64));
            e.edge_visits += 1;
            let new_contrib = (e.edge_visits as f64) * (-(c_q as f64));
            e.last_child_q = c_q;
            let a = &mut self.nodes[anc_idx as usize];
            a.w_sum += new_contrib - old_contrib;
            a.w_total += 1.0;
            a.visits += 1;
            if a.state == NodeState::Ongoing {
                a.q = (a.w_sum / a.w_total) as f32;
            }
            prev_child = anc_idx;
        }
    }

    fn run_playout(
        &mut self,
        root_b: &Board,
        root_idx: u32,
        root_sym: usize,
        base_snaps: &[(u64, u64)],
        root_ply: usize,
        respect_401: bool,
        bz: bool,
        nn: &Nnue,
        pol: Option<&Policy>,
    ) -> usize {
        self.path_edges.clear();
        self.path_nodes.clear();
        self.path_snaps.clear();
        if bz {
            if base_snaps.is_empty() {
                self.path_snaps.push((root_b.occ[0], root_b.occ[1]));
            } else {
                self.path_snaps.extend_from_slice(base_snaps);
            }
        }

        let mut cur_idx = root_idx;
        let mut cur_b = *root_b;
        let mut cur_sym = root_sym;
        let mut depth = 0usize;

        loop {
            if self.nodes[cur_idx as usize].state != NodeState::Ongoing {
                self.backup_normal(cur_idx, bz);
                return depth;
            }
            if bz && respect_401 && root_ply + depth >= 401 {
                let (b0, b1) = cur_b.counts();
                let v = match b0.cmp(&b1) {
                    std::cmp::Ordering::Greater => 1.0f32,
                    std::cmp::Ordering::Less => -1.0f32,
                    std::cmp::Ordering::Equal => 0.0f32,
                };
                if let Some((parent_idx, _, _)) = self.path_edges.pop() {
                    let cur_turn = cur_b.turn as usize;
                    let v_stm = if cur_turn == 0 { v } else { -v };
                    self.backup_path_scalar(parent_idx, -v_stm);
                }
                return depth;
            }
            if !self.nodes[cur_idx as usize].expanded {
                self.expand_node(cur_idx, bz, pol);
            }
            self.path_nodes.push(cur_idx);

            let (maybe_ei, had_ban) = self.pick_edge(cur_idx, &cur_b, cur_sym, bz, depth == 0);
            let Some(ei) = maybe_ei else {
                if let Some((parent_idx, _, _)) = self.path_edges.pop() {
                    self.backup_path_scalar(parent_idx, 1.0);
                }
                return depth;
            };

            let canon_mv = self.edges[ei as usize].mv;
            let conc_mv = map_move(canon_mv, &INV_SYM_SQ[cur_sym]);
            if bz && !conc_mv.is_pass() {
                self.path_snaps.push(cur_b.placement_snap(conc_mv));
            }
            let next_b = if conc_mv.is_pass() {
                cur_b.make_pass()
            } else {
                cur_b.make_via_lut(conc_mv)
            };
            let (child_key, next_sym) = canonicalize_board(&next_b, !bz);
            let existing = self.edges[ei as usize].child;

            if existing == u32::MAX {
                let child_idx = self.alloc_or_get(child_key, bz, nn);
                if self.path_nodes.contains(&child_idx) {
                    self.backup_path_scalar(cur_idx, 0.0);
                    return depth + 1;
                }
                self.edges[ei as usize].child = child_idx;
                self.path_edges.push((cur_idx, ei, had_ban));
                self.backup_normal(child_idx, bz);
                return depth + 1;
            }

            if self.path_nodes.contains(&existing) {
                self.backup_path_scalar(cur_idx, 0.0);
                return depth + 1;
            }
            self.path_edges.push((cur_idx, ei, had_ban));
            cur_idx = existing;
            cur_b = next_b;
            cur_sym = next_sym;
            depth += 1;
            if depth >= 250 {
                if let Some((parent_idx, _, _)) = self.path_edges.pop() {
                    self.backup_path_scalar(parent_idx, 0.0);
                }
                return depth;
            }
        }
    }

    pub fn search_result(
        &mut self,
        b: &Board,
        nn: &Nnue,
        pol: Option<&Policy>,
        snaps: &[(u64, u64)],
        respect_401: bool,
        bz: bool,
        limits: &SearchLimits,
    ) -> SearchResult {
        let t0 = Instant::now();
        if self.last_bz != Some(bz) || self.nodes.len() > self.params.max_nodes * 9 / 10 {
            self.reset_for_new_game();
        }
        self.last_bz = Some(bz);

        let (root_key, root_sym) = canonicalize_board(b, !bz);
        let root_idx = self.alloc_or_get(root_key, bz, nn);
        self.expand_node(root_idx, bz, pol);

        let root_ply = if bz { snaps.len().saturating_sub(1) } else { b.ply as usize };
        let safety_ms = 2.0f64;
        let time_budget_ms = limits
            .time
            .map(|d| (d.as_secs_f64() * 1000.0 - safety_ms).max(1.0));
        let max_playouts = if let Some(n) = limits.max_nodes {
            n
        } else if time_budget_ms.is_some() {
            u64::MAX
        } else {
            64 * (1u64 << limits.max_depth.min(12))
        };

        let mut playouts = 0u64;
        let mut depth_sum = 0usize;
        let mut stopped_by = Some(StopReason::Completed);

        if self.nodes[root_idx as usize].edge_len > 0
            && self.nodes[root_idx as usize].state == NodeState::Ongoing
        {
            while playouts < max_playouts {
                if self.nodes[root_idx as usize].state != NodeState::Ongoing {
                    break;
                }
                if let Some(limit_ms) = time_budget_ms {
                    let check_mask = if limit_ms < 25.0 { 15 } else { 63 };
                    if (playouts & check_mask) == 0 && playouts > 0 {
                        let el_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        if el_ms >= limit_ms {
                            stopped_by = Some(StopReason::Time);
                            break;
                        }
                    }
                }
                depth_sum += self.run_playout(
                    b,
                    root_idx,
                    root_sym,
                    snaps,
                    root_ply,
                    respect_401,
                    bz,
                    nn,
                    pol,
                );
                playouts += 1;
            }
            if playouts >= max_playouts && limits.max_nodes.is_some() {
                stopped_by = Some(StopReason::Nodes);
            }
        }

        // LCB + proof-aware root selection (skipping Botzone root repetition bans).
        let fallback_snap = [(b.occ[0], b.occ[1])];
        let base_snaps_slice = if snaps.is_empty() {
            &fallback_snap[..]
        } else {
            snaps
        };
        let node = &self.nodes[root_idx as usize];
        let start = node.edge_start as usize;
        let len = node.edge_len as usize;

        let mut best_win: Option<(u16, Move)> = None;
        let mut max_vis = 0u32;
        let mut best_loss: Option<(u16, Move)> = None;
        let mut fallback_mv: Option<Move> = None;

        for e in &self.edges[start..start + len] {
            let conc_mv = map_move(e.mv, &INV_SYM_SQ[root_sym]);
            if bz && !conc_mv.is_pass() && has_bz_ban(base_snaps_slice, b.placement_snap(conc_mv)) {
                continue;
            }
            if fallback_mv.is_none() {
                fallback_mv = Some(conc_mv);
            }
            let c_state = if e.child != u32::MAX {
                self.nodes[e.child as usize].state
            } else {
                NodeState::Ongoing
            };
            match c_state {
                NodeState::ProvenLoss(d) => {
                    if best_win.map_or(true, |(bd, _)| d < bd) {
                        best_win = Some((d, conc_mv));
                    }
                }
                NodeState::ProvenWin(d) => {
                    if best_loss.map_or(true, |(bd, _)| d > bd) {
                        best_loss = Some((d, conc_mv));
                    }
                }
                _ => {
                    max_vis = max_vis.max(e.edge_visits);
                }
            }
        }

        let (best, score) = if let Some((d, mv)) = best_win {
            (Some(mv), MATE - (d as i32 + 1))
        } else {
            let min_vis = (max_vis / 4).max(1);
            let mut best_lcb: Option<(f32, u32, f32, Move)> = None;
            for e in &self.edges[start..start + len] {
                let conc_mv = map_move(e.mv, &INV_SYM_SQ[root_sym]);
                if bz
                    && !conc_mv.is_pass()
                    && has_bz_ban(base_snaps_slice, b.placement_snap(conc_mv))
                {
                    continue;
                }
                if e.child != u32::MAX && matches!(self.nodes[e.child as usize].state, NodeState::ProvenWin(_)) {
                    continue;
                }
                if e.edge_visits < min_vis {
                    continue;
                }
                let q = if e.edge_visits > 0 && e.child != u32::MAX {
                    -self.nodes[e.child as usize].q
                } else {
                    -1.0
                };
                let lcb = q - self.params.lcb_z / (e.edge_visits as f32).sqrt();
                let take = match best_lcb {
                    None => true,
                    Some((bl, bv, bq, _)) => {
                        lcb > bl
                            || ((lcb - bl).abs() <= 1e-6
                                && (e.edge_visits > bv || (e.edge_visits == bv && q > bq)))
                    }
                };
                if take {
                    best_lcb = Some((lcb, e.edge_visits, q, conc_mv));
                }
            }
            if let Some((_, _, q, mv)) = best_lcb {
                (Some(mv), q_to_cp(q))
            } else if let Some((d, mv)) = best_loss {
                (Some(mv), -(MATE - (d as i32 + 1)))
            } else {
                (fallback_mv, q_to_cp(node.q))
            }
        };

        let avg_depth = if playouts > 0 {
            ((depth_sum as f64 / playouts as f64).round() as u32).max(1)
        } else {
            1
        };
        SearchResult {
            best,
            score,
            depth: avg_depth,
            nodes: playouts,
            elapsed: t0.elapsed(),
            stopped_by,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::policy::KCache;
    use crate::search::SearchLimits;
    use std::time::Duration;

    #[test]
    fn d4_canonical_symmetry_invariance() {
        for s in 0..8 {
            for sq in 0..49 {
                let mapped = SYM_SQ[s][sq] as usize;
                assert_eq!(INV_SYM_SQ[s][mapped], sq as u8);
            }
        }
        let b = Board::from_occ(0x0001_0410_2010_0009, 0x0000_8020_4088_0002, 0, 0);
        let (k0, s0) = canonicalize_board(&b, true);
        assert_eq!(transform_bb(k0.me, 0), k0.me);
        let mut inv_me = 0u64;
        for sq in 0..49 {
            if (b.me() >> sq) & 1 != 0 {
                inv_me |= 1u64 << SYM_SQ[s0][sq];
            }
        }
        assert_eq!(inv_me, k0.me);

        for s in 0..8 {
            let sym_b = Board::from_occ(
                transform_bb(b.me(), s),
                transform_bb(b.opp(), s),
                transform_bb(b.blockers, s),
                0,
            );
            let (ks, ss) = canonicalize_board(&sym_b, true);
            assert_eq!(ks, k0, "symmetry {s} produced different canonical key");
            let mut round = 0u64;
            for csq in 0..49 {
                if (ks.me >> csq) & 1 != 0 {
                    round |= 1u64 << INV_SYM_SQ[ss][csq];
                }
            }
            assert_eq!(round, sym_b.me());
        }
    }

    #[test]
    fn fused_policy_priors_match_logit_k() {
        let pol = crate::policy::global().expect("bundled policy present");
        let b = Board::from_occ(0x0001_0410_2010_0009, 0x0000_8020_4088_0002, 0, 0);
        let nn = Nnue::bundled_static();
        let mut params = McgsParams::default();
        params.sn_prior_weight = 0.0;
        let mut mcgs = McgsSearcher::new(params);
        let (key, _) = canonicalize_board(&b, false);
        let root_idx = mcgs.alloc_or_get(key, true, nn);
        mcgs.expand_node(root_idx, true, Some(pol));

        let cb = light_board(&key);
        let mut kc = KCache::new();
        let node = &mcgs.nodes[root_idx as usize];
        let edges = &mcgs.edges[node.edge_start as usize..node.edge_start as usize + node.edge_len as usize];
        let logits: Vec<f32> = edges.iter().map(|e| pol.logit_k(&cb, e.mv, &mut kc)).collect();
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&z| (z - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (e, ref_e) in edges.iter().zip(exps.iter()) {
            let ref_p = ref_e / sum;
            assert!((e.prior - ref_p).abs() < 1e-6, "mv {:?}: {} vs {}", e.mv, e.prior, ref_p);
        }
    }

    #[test]
    fn mcgs_uses_dis_smallnet_and_live_sibling_sync() {
        let b = Board::from_occ(0x0001_0410_2010_0009, 0x0000_8020_4088_0002, 0, 0);
        let nn = Nnue::bundled_static();
        let pol = crate::policy::global();

        let (key, sym) = canonicalize_board(&b, false);
        let mut pure_params = McgsParams::default();
        pure_params.sn_prior_weight = 0.0;
        pure_params.dis_leaf_decay = 0.0;
        let mut pure = McgsSearcher::new(pure_params);
        let rp = pure.alloc_or_get(key, true, nn);
        pure.expand_node(rp, true, pol);

        let mut full_params = McgsParams::default();
        full_params.sn_prior_weight = 0.50;
        let mut full = McgsSearcher::new(full_params);
        let rf = full.alloc_or_get(key, true, nn);
        full.expand_node(rf, true, pol);

        // Verify SmallNet + 1-ply terminal prior shifts probabilities vs pure policy.
        let ep = &pure.edges[pure.nodes[rp as usize].edge_start as usize
            ..pure.nodes[rp as usize].edge_start as usize + pure.nodes[rp as usize].edge_len as usize];
        let ef = &full.edges[full.nodes[rf as usize].edge_start as usize
            ..full.nodes[rf as usize].edge_start as usize + full.nodes[rf as usize].edge_len as usize];
        let max_diff = ep
            .iter()
            .zip(ef.iter())
            .map(|(a, b)| (a.prior - b.prior).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-3, "expected SmallNet prior blend to shift priors, got diff {max_diff}");

        // Verify live sibling Q reconciliation in pick_edge.
        let first_e_idx = full.nodes[rf as usize].edge_start as usize;
        let child_b = b.make_via_lut(map_move(full.edges[first_e_idx].mv, &INV_SYM_SQ[sym]));
        let (ckey, _) = canonicalize_board(&child_b, false);
        let cidx = full.alloc_or_get(ckey, true, nn);
        full.edges[first_e_idx].child = cidx;
        full.edges[first_e_idx].edge_visits = 4;
        full.edges[first_e_idx].last_child_q = 0.25;
        let old_w = full.nodes[rf as usize].w_sum;
        full.nodes[cidx as usize].q = -0.50;
        let _ = full.pick_edge(rf, &b, sym, true, true);
        let new_w = full.nodes[rf as usize].w_sum;
        assert!((full.edges[first_e_idx].last_child_q - (-0.50)).abs() < 1e-6);
        assert!((new_w - old_w - 3.0).abs() < 1e-5, "expected +3.0 live delta, got {}", new_w - old_w);
    }

    #[test]
    fn mcgs_finds_forced_win_in_one() {
        let all48 = ((1u64 << 49) - 1) & !(1u64 << 48);
        let black = 0x0000_0000_FFFF_FFFFu64 | (1u64 << 47);
        let white = all48 & !black;
        let b = Board::from_occ(black, white, 0, 0);
        let nn = Nnue::bundled_static();
        let mut mcgs = McgsSearcher::new(McgsParams::default());
        let lim = SearchLimits {
            time: None,
            max_nodes: Some(64),
            max_depth: 10,
        };
        let r = mcgs.search_result(&b, nn, crate::policy::global(), &[(b.occ[0], b.occ[1])], false, true, &lim);
        assert_eq!(r.best.map(|m| m.to), Some(48));
        assert!(r.score >= MATE - 10, "expected forced mate score, got {}", r.score);
    }

    #[test]
    fn mcgs_respects_botzone_repetition_ban() {
        let b = Board::botzone_start();
        let moves = b.legal_moves();
        let banned_jump = moves.iter().find(|m| !m.is_clone()).expect("jump exists");
        let root_snap = (b.occ[0], b.occ[1]);
        let banned_snap = b.placement_snap(banned_jump);
        let snaps = vec![
            root_snap,
            banned_snap,
            (0x1111, 0x2222),
            (0x3333, 0x4444),
            root_snap,
        ];
        let nn = Nnue::bundled_static();
        let mut mcgs = McgsSearcher::new(McgsParams::default());
        let lim = SearchLimits {
            time: None,
            max_nodes: Some(128),
            max_depth: 10,
        };
        let r = mcgs.search_result(&b, nn, crate::policy::global(), &snaps, true, true, &lim);
        let best = r.best.expect("legal move expected");
        assert_ne!(
            b.placement_snap(best),
            banned_snap,
            "MCGS selected a Botzone circular-banned move"
        );
    }

    #[test]
    fn mcgs_dag_reuses_transpositions_without_double_counting() {
        let b = Board::botzone_start();
        let nn = Nnue::bundled_static();
        let mut mcgs = McgsSearcher::new(McgsParams::default());
        let lim = SearchLimits {
            time: None,
            max_nodes: Some(400),
            max_depth: 20,
        };
        let r = mcgs.search_result(&b, nn, crate::policy::global(), &[(b.occ[0], b.occ[1])], false, true, &lim);
        assert!(r.nodes == 400);
        assert!(
            mcgs.graph_len() < 401,
            "expected D4 transposition sharing (<401 nodes), got {}",
            mcgs.graph_len()
        );
        let root_key = canonicalize_board(&b, false).0;
        let root_node = &mcgs.nodes[mcgs.table[&root_key] as usize];
        let edge_sum: u32 = mcgs.edges[root_node.edge_start as usize
            ..root_node.edge_start as usize + root_node.edge_len as usize]
            .iter()
            .map(|e| e.edge_visits)
            .sum();
        assert_eq!(edge_sum + 1, root_node.visits, "visit accounting invariant violated");
        assert!(root_node.q.abs() <= 1.0 && root_node.q.is_finite());
    }

    #[test]
    fn mcgs_returns_legal_move_and_valid_score() {
        let b = Board::botzone_start();
        let nn = Nnue::bundled_static();
        let mut mcgs = McgsSearcher::new(McgsParams::default());
        let lim = SearchLimits {
            time: Some(Duration::from_millis(25)),
            max_nodes: None,
            max_depth: 30,
        };
        let r = mcgs.search_result(&b, nn, crate::policy::global(), &[(b.occ[0], b.occ[1])], false, true, &lim);
        let mv = r.best.expect("best move missing");
        assert!(b.legal_moves().iter().any(|m| m == mv), "MCGS returned illegal move {mv:?}");
        assert!(r.nodes > 0);
        assert!(r.score.abs() <= MATE);
    }
}
