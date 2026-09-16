//! Browser-facing API for the Titanium Ataxx engine.
//!
//! Build with:
//! `wasm-pack build crates/titanium-wasm --target web --out-dir ../../web/pkg --out-name titanium`

use std::time::Duration;

use titanium::{Board, Move, SearchLimits, Searcher};
use wasm_bindgen::prelude::*;

/// A game session: keeps the full board history so moves can be undone.
#[wasm_bindgen]
pub struct Game {
    hist: Vec<Board>,
    /// Move that led to `hist[i]` (`Move::PASS` for pass entries).
    moves: Vec<Move>,
    time_ms: [u32; 2],
    max_depth: u32,
    last_depth: u32,
    last_nodes: u32,
    last_ms: u32,
    /// Persistent searcher: TT carries over between moves of the game.
    searcher: Searcher,
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Game {
        let mut searcher = Searcher::with_tt_bits(16);
        searcher.set_time_source(perf::now);
        Game {
            hist: vec![Board::start()],
            moves: vec![Move::PASS],
            time_ms: [600, 600],
            max_depth: 12,
            last_depth: 0,
            last_nodes: 0,
            last_ms: 0,
            searcher,
        }
    }

    /// Per-side think time (ms): 0 = black, 1 = white.
    pub fn set_side_time(&mut self, side: u8, time_ms: u32) {
        self.time_ms[(side as usize) & 1] = time_ms.max(1);
    }

    pub fn set_max_depth(&mut self, max_depth: u32) {
        self.max_depth = max_depth.clamp(1, 30);
    }

    /// Side to move: 0 black, 1 white.
    pub fn turn(&self) -> u8 {
        self.cur().turn
    }

    /// Piece on square: 0 empty, 1 black, 2 white, 3 blocker.
    pub fn piece(&self, sq: u8) -> u8 {
        let b = self.cur();
        let bit = 1u64 << sq;
        if b.occ[0] & bit != 0 {
            1
        } else if b.occ[1] & bit != 0 {
            2
        } else if b.blockers & bit != 0 {
            3
        } else {
            0
        }
    }

    /// Legal moves for the side to move, packed as `(from << 8) | to`.
    pub fn legal_moves(&self) -> Vec<u32> {
        self.cur().legal_moves().iter().map(|m| m.to_u32()).collect()
    }

    pub fn has_moves(&self) -> bool {
        let b = self.cur();
        b.has_moves(b.turn)
    }

    /// Play a move; throws a JS error if it is not legal.
    pub fn make(&mut self, from: u8, to: u8) -> Result<(), JsValue> {
        let b = *self.cur();
        let m = Move { from, to };
        if !b.legal_moves().contains(&m) {
            return Err(JsValue::from_str("illegal move"));
        }
        self.push(b.make(m), m);
        Ok(())
    }

    /// Pass (only allowed when the side to move has no legal move).
    pub fn pass(&mut self) -> Result<(), JsValue> {
        let b = *self.cur();
        if b.has_moves(b.turn) {
            return Err(JsValue::from_str("still has legal moves"));
        }
        self.push(b.make_pass(), Move::PASS);
        Ok(())
    }

    /// Let the engine move for the side to move.
    /// Returns the packed move, or `null` if it passed / game is over.
    pub fn ai_move(&mut self) -> Option<u32> {
        if self.cur().game_over() {
            return None;
        }
        let b = *self.cur();
        if !b.has_moves(b.turn) {
            self.push(b.make_pass(), Move::PASS);
            return None;
        }
        let limits = SearchLimits {
            time: Some(Duration::from_millis(self.time_ms[self.cur().turn as usize] as u64)),
            max_nodes: None,
            max_depth: self.max_depth,
        };
        let started = perf::now();
        let r = self.searcher.search(&b, &limits);
        self.last_depth = r.depth;
        self.last_nodes = r.nodes.min(u32::MAX as u64) as u32;
        self.last_ms = (perf::now() - started).max(0.0) as u32;
        let m = r.best?;
        self.push(b.make(m), m);
        Some(m.to_u32())
    }

    /// Game over (board full / a side wiped / both stuck).
    pub fn over(&self) -> bool {
        self.cur().game_over()
    }

    /// 0 black, 1 white, 2 draw, 3 ongoing.
    pub fn winner(&self) -> u8 {
        if self.over() {
            self.cur().winner()
        } else {
            3
        }
    }

    /// `[black, white]` piece counts.
    pub fn counts(&self) -> Vec<u32> {
        let (b, w) = self.cur().counts();
        vec![b, w]
    }

    /// Take back one ply. Returns false at the start position.
    pub fn undo(&mut self) -> bool {
        if self.hist.len() <= 1 {
            return false;
        }
        self.hist.pop();
        self.moves.pop();
        true
    }

    pub fn history_len(&self) -> usize {
        self.hist.len()
    }

    /// Last played move, packed (`0xFFFF` for a pass).
    pub fn last_move(&self) -> u32 {
        self.moves.last().map(|m| m.to_u32()).unwrap_or(0xFFFF)
    }

    /// `[depth, nodes, ms]` of the most recent engine search.
    pub fn search_info(&self) -> Vec<u32> {
        vec![self.last_depth, self.last_nodes, self.last_ms]
    }

    /// Engine string of the current position (debugging).
    pub fn board_string(&self) -> String {
        self.cur().to_string()
    }
}

impl Game {
    fn cur(&self) -> &Board {
        self.hist.last().expect("history never empty")
    }

    fn push(&mut self, board: Board, m: Move) {
        self.hist.push(board);
        self.moves.push(m);
    }
}

/// `performance.now()` in milliseconds â€” injected into the engine as its
/// monotonic clock (std time is unavailable on wasm32).
#[cfg(target_arch = "wasm32")]
mod perf {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        pub fn now() -> f64;
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod perf {
    pub fn now() -> f64 {
        0.0
    }
}
