//! Titanium - a basic alpha-beta Ataxx engine.
//!
//! Rules implemented (7x7, standard Ataxx):
//! - Move by cloning to any empty square at Chebyshev distance 1, or jumping
//!   to any empty square at distance 2.
//! - After landing, every enemy piece adjacent to the landing square is
//!   converted ("infected") to the mover's color.
//! - A player with no legal move passes. If both players are stuck or the
//!   board is full, the game ends and the majority wins.

pub mod board;
pub mod sancta;
pub mod search;

pub use board::{
    bit_of, dist_union, infect_direct, infect_via_lut, extract8, jump_union, perft_bb, Board, Move,
    MoveList, CONVERT8, NEIGH_CNT, NEIGH_POS, REACH, RING1, RING2, ZOB_BLOCK, ZOB_PIECE, ZOB_SIDE,
    BLACK, BLOCKER, EMPTY, FULL, MAX_MOVES, SIZE, SQUARES, WHITE,
};
#[cfg(target_arch = "x86_64")]
pub use board::extract8_pext;
pub use search::{best_move, SearchLimits, SearchResult, Searcher, StopReason};
pub use sancta::{S1Net, S1Acc, S1_L1, S1_H2, S1N_V3};
