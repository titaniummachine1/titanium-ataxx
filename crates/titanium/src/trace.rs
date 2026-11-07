//! exp/tree-shape (D547/D549): search-tree trace, compiled ONLY with `--features trace` (never in a normal build).
//! Every negamax call is one record (with its parent, the move that led to it, the window, the static eval and how it
//! ended); every child the move loop skipped without a search is a record of kind SKIP_*. One iteration of iterative
//! deepening is kept at a time (cleared at the iteration start), so after `go depth D` the trace is the whole depth-D
//! tree. Used by `titanium-cli treediff` to find over-pruned (deep PV move skipped / reduced / cut in the shallow tree)
//! and under-pruned nodes (searched in full, result far outside the window) per game phase and knob.

/// How a node ended (or why a child was skipped).
pub mod kind {
    pub const OPEN: u8 = 0; // still on the stack (search stopped)
    pub const TERMINAL: u8 = 1;
    pub const MATE_DIST: u8 = 2;
    pub const TT_CUT: u8 = 3;
    pub const RFP: u8 = 4;
    pub const NMP: u8 = 5;
    pub const PROBCUT: u8 = 6;
    pub const LEAF: u8 = 7; // depth 0: static eval
    pub const NO_MOVES: u8 = 8;
    pub const FULL: u8 = 9; // move loop ran (cutoff or all moves)
    pub const CIRCULAR: u8 = 10;
    pub const STACK: u8 = 11;
    pub const SKIP_SN: u8 = 12; // child skipped by the futility small net (no NNUE eval)
    pub const SKIP_FUT: u8 = 13; // child skipped: futility/LMR reduced it to depth <= 0
    pub const SKIP_REP: u8 = 14; // child skipped: repetition of an ancestor
    pub const NAMES: [&str; 15] = [
        "open", "terminal", "mate_dist", "tt_cut", "rfp", "nmp", "probcut", "leaf", "no_moves", "full", "circular",
        "stack", "skip_sn", "skip_fut", "skip_rep",
    ];
}

/// What kind of call led to a node.
pub mod via {
    pub const ROOT: u8 = 0;
    pub const FIRST: u8 = 1; // first move, full window
    pub const SCOUT: u8 = 2; // later move, null window, reduced depth
    pub const RESEARCH: u8 = 3; // later move re-searched full window, full depth
    pub const NULL: u8 = 4; // null move (pass)
    pub const PROBCUT: u8 = 5; // ProbCut verification at the same ply
    pub const NAMES: [&str; 6] = ["root", "first", "scout", "research", "null", "probcut"];
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rec {
    pub hash: u64,
    pub parent: u32,
    /// move from the parent (Move::to_u32), 0xFFFF = pass / none
    pub mv: u16,
    pub ply: u8,
    pub empties: u8,
    /// depth passed in by the caller
    pub depth: i8,
    /// depth after IIR
    pub sdepth: i8,
    pub kind: u8,
    pub via: u8,
    /// move index in the parent's loop (255 = n/a)
    pub idx: u8,
    /// depth the parent wanted for this move before reductions (parent depth - 1)
    pub full_depth: i8,
    pub alpha: i32,
    pub beta: i32,
    pub stat: i32,
    pub dis: i32,
    pub score: i32,
    /// knob slack: RFP = alpha - (static + margin); SKIP_* = how far below 0 the reduced depth went (plies x 100) or
    /// the fut shortfall in cp; NMP/PROBCUT = score - beta
    pub slack: i32,
}

pub struct Tracer {
    pub recs: Vec<Rec>,
    pub stack: Vec<u32>,
    pub cap: usize,
    /// set by the caller right before a recursive negamax call, consumed at the callee's entry
    pub p_mv: u16,
    pub p_via: u8,
    pub p_idx: u8,
    pub p_full: i8,
    pub overflow: bool,
}

impl Tracer {
    pub fn new(cap: usize) -> Tracer {
        Tracer { recs: Vec::new(), stack: Vec::new(), cap, p_mv: 0xFFFF, p_via: via::ROOT, p_idx: 255, p_full: -1, overflow: false }
    }
    pub fn clear(&mut self) {
        self.recs.clear();
        self.stack.clear();
        self.overflow = false;
        self.pending(0xFFFF, via::ROOT, 255, -1);
    }
    #[inline]
    pub fn pending(&mut self, mv: u16, v: u8, idx: u8, full: i8) {
        self.p_mv = mv;
        self.p_via = v;
        self.p_idx = idx;
        self.p_full = full;
    }
    /// node entry: returns the record index (u32::MAX if the cap is reached)
    #[allow(clippy::too_many_arguments)]
    pub fn enter(&mut self, hash: u64, empties: u8, alpha: i32, beta: i32, depth: u32, ply: u32) -> u32 {
        let parent = self.stack.last().copied().unwrap_or(u32::MAX);
        let (mv, v, idx, full) = (self.p_mv, self.p_via, self.p_idx, self.p_full);
        self.pending(0xFFFF, via::ROOT, 255, -1);
        if self.recs.len() >= self.cap {
            self.overflow = true;
            self.stack.push(u32::MAX);
            return u32::MAX;
        }
        let id = self.recs.len() as u32;
        self.recs.push(Rec {
            hash,
            parent,
            mv,
            ply: ply.min(255) as u8,
            empties,
            depth: depth.min(127) as i8,
            sdepth: depth.min(127) as i8,
            kind: kind::OPEN,
            via: v,
            idx,
            full_depth: full,
            alpha,
            beta,
            stat: i32::MIN,
            dis: 0,
            score: i32::MIN,
            slack: 0,
        });
        self.stack.push(id);
        id
    }
    #[inline]
    pub fn top(&mut self) -> Option<&mut Rec> {
        let id = *self.stack.last()?;
        self.recs.get_mut(id as usize)
    }
    /// mark how the current node ends (the last call before the return wins)
    #[inline]
    pub fn end(&mut self, k: u8, slack: i32) {
        if let Some(r) = self.top() {
            r.kind = k;
            r.slack = slack;
        }
    }
    pub fn exit(&mut self, score: i32) {
        if let Some(id) = self.stack.pop() {
            if let Some(r) = self.recs.get_mut(id as usize) {
                r.score = score;
            }
        }
    }
    /// a child that the move loop skipped without searching it
    #[allow(clippy::too_many_arguments)]
    pub fn skip(&mut self, hash: u64, empties: u8, mv: u16, ply: u32, idx: usize, full: i32, k: u8, stat: i32, slack: i32, alpha: i32) {
        if self.recs.len() >= self.cap {
            self.overflow = true;
            return;
        }
        let parent = self.stack.last().copied().unwrap_or(u32::MAX);
        self.recs.push(Rec {
            hash,
            parent,
            mv,
            ply: (ply + 1).min(255) as u8,
            empties,
            depth: -1,
            sdepth: -1,
            kind: k,
            via: via::SCOUT,
            idx: idx.min(255) as u8,
            full_depth: full.clamp(-128, 127) as i8,
            alpha: -alpha - 1,
            beta: -alpha,
            stat,
            dis: 0,
            score: i32::MIN,
            slack,
        });
    }
}
