// titanium-ataxx Botzone entry (simple protocol).
// Reads: turnID, then 2*(turnID-1) history lines, then the current opp move.
// Each move line: x0 y0 x1 y1 (Botzone coords, x=file 0..6 L-to-R, y=0 top).
// Replies: one line "x0 y0 x1 y1", or "-1 -1 -1 -1" with no legal move.
// Stateless: full history replayed every turn, then one bounded search.
use std::io::{self, Read};
use std::time::Duration;

use crate::board::{Board, Move};
use crate::search::{Searcher, SearchLimits};

const MOVETIME_MS: u64 = 800;
/// First turn of the game gets the platform's 2x time (2000ms budget).
const MOVETIME_FIRST_MS: u64 = 1500;

fn sq_xy(sq: u8) -> (i32, i32) {
    ((sq % 7) as i32, (sq / 7) as i32)
}

fn main() {
    // Read exactly what the protocol sends (no EOF dependence):
    // turnID, then 2*(turnID-1) history lines, then the current opp line.
    // (The wiki mentions trailing data/globaldata lines; we ignore them.)
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let first: String = lines.next().map_or(String::new(), |l| l.unwrap_or_default());
    let turn_id: usize = first
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let mut nums: Vec<i64> = Vec::with_capacity(4 * (2 * turn_id - 1));
    while nums.len() < 4 * (2 * turn_id - 1) {
        match lines.next() {
            Some(Ok(l)) => {
                for s in l.split_whitespace() {
                    if let Ok(v) = s.parse::<i64>() {
                        nums.push(v);
                    }
                    if nums.len() >= 4 * (2 * turn_id - 1) {
                        break;
                    }
                }
            }
            _ => break,
        }
    }
    while nums.len() < 4 * (2 * turn_id - 1) {
        nums.push(-1);
    }
    let mut idx = 0usize;
    // history: (turn_id-1) pairs of (opp, own), then current opp move.
    let take = |idx: &mut usize| -> (i64, i64, i64, i64) {
        let v = (
            *nums.get(*idx).unwrap_or(&-1),
            *nums.get(*idx + 1).unwrap_or(&-1),
            *nums.get(*idx + 2).unwrap_or(&-1),
            *nums.get(*idx + 3).unwrap_or(&-1),
        );
        *idx += 4;
        v
    };
    let mut reqs: Vec<(i64, i64, i64, i64)> = Vec::new();
    let mut resps: Vec<(i64, i64, i64, i64)> = Vec::new();
    for _ in 0..turn_id.saturating_sub(1) {
        reqs.push(take(&mut idx));
        resps.push(take(&mut idx));
    }
    let cur = take(&mut idx);
    // Reconstruct from the consensus 4-corner start.
    let mut b = Board::start();
    let mut seen: Vec<u64> = vec![b.hash];
    let step = |bd: &Board, x0: i64, y0: i64, x1: i64, y1: i64| -> Option<Board> {
        if x1 < 0 {
            return None;
        }
        let from = (y0 as usize) * 7 + (x0 as usize);
        let to = (y1 as usize) * 7 + (x1 as usize);
        if from > 48 || to > 48 {
            return None;
        }
        let m = Move {
            from: from as u8,
            to: to as u8,
        };
        // Verify legality against our own movegen （ Botzone kills illegal moves).
        let list = bd.legal_moves();
        let mut ok = false;
        for i in 0..list.len() {
            if list.move_at(i).to_u32() == m.to_u32() {
                ok = true;
                break;
            }
        }
        if !ok {
            return None;
        }
        Some(bd.make(m))
    };
    for i in 0..reqs.len() {
        let (x0, y0, x1, y1) = reqs[i];
        if x1 >= 0 {
            if let Some(nb) = step(&b, x0, y0, x1, y1) {
                b = nb;
            }
        }
        seen.push(b.hash);
        if i < resps.len() {
            let (x0, y0, x1, y1) = resps[i];
            if x1 >= 0 {
                if let Some(nb) = step(&b, x0, y0, x1, y1) {
                    b = nb;
                }
            }
            seen.push(b.hash);
        }
    }
    {
        let (x0, y0, x1, y1) = cur;
        if x1 >= 0 {
            if let Some(nb) = step(&b, x0, y0, x1, y1) {
                b = nb;
            }
        }
        seen.push(b.hash);
    }
    // Play.
    let list = b.legal_moves();
    if list.len() == 0 {
        println!("-1 -1 -1 -1");
        return;
    }
    // First turn of the game: platform grants 2x time. Detect: no history
    // was replayed (only the initial position seen).
    let first_turn = seen.len() <= 2;
    let budget = if first_turn {
        MOVETIME_FIRST_MS
    } else {
        MOVETIME_MS
    };
    let mut searcher = Searcher::new();
    let limits = SearchLimits {
        time: Some(Duration::from_millis(budget)),
        max_nodes: None,
        max_depth: 64,
    };
    let r = searcher.search(&b, &limits);
    let mut mv = r.best.unwrap_or_else(|| list.move_at(0));
    // Repetition guard (Botzone kills loopers): never recreate a position.
    let mut child = b.make(mv);
    if seen.contains(&child.hash) {
        let mut alt = None;
        for i in 0..list.len() {
            let m = list.move_at(i);
            if m.to_u32() == mv.to_u32() {
                continue;
            }
            let c = b.make(m);
            if !seen.contains(&c.hash) {
                alt = Some((m, c));
                break;
            }
        }
        if let Some((m2, _)) = alt {
            mv = m2;
            child = b.make(mv);
        }
        let _ = child;
    }
    let (fx, fy) = sq_xy(mv.from);
    let (tx, ty) = sq_xy(mv.to);
    println!("{fx} {fy} {tx} {ty}");
}
