// titanium-ataxx Botzone entry (simple protocol).
// Reads: turnID, then 2*(turnID-1) history lines, then the current opp move.
// Each move line: x0 y0 x1 y1 (Botzone coords, x=file 0..6 L-to-R, y=0 top).
// Replies: one line "x0 y0 x1 y1", or "-1 -1 -1 -1" with no legal move.
// Stateless: full history replayed every turn, then one bounded search.
use std::io::{self, BufRead};
use std::time::Duration;

use crate::board::{Board, Move};
use crate::search::{parallel_search_bz, Searcher, SearchLimits};

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
    let mut fw = first.split_whitespace();
    let turn_id: usize = fw
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let need = 4 * (2 * turn_id - 1);
    // Move tokens may share the turnID line or arrive one line per move —
    // accept both, and never depend on EOF: read exactly `need` numbers.
    let mut nums: Vec<i64> = Vec::with_capacity(need);
    for s in fw {
        if let Ok(v) = s.parse::<i64>() {
            nums.push(v);
        }
    }
    while nums.len() < need {
        match lines.next() {
            Some(Ok(l)) => {
                for s in l.split_whitespace() {
                    if let Ok(v) = s.parse::<i64>() {
                        nums.push(v);
                    }
                    if nums.len() >= need {
                        break;
                    }
                }
            }
            _ => break,
        }
    }
    while nums.len() < need {
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
    // Reconstruct from the consensus 4-corner start. Entries are (opp, own)
    // pairs plus the current opp entry; a "-1 -1 -1 -1" entry is a pass (the
    // turn flips, the board does not change) — except the very first entry
    // of a black-first game, which is a start marker, not a turn.
    let mut opp_seq: Vec<(i64, i64, i64, i64)> = reqs.clone();
    opp_seq.push(cur);
    let marker = opp_seq
        .first()
        .map_or(false, |e| e.0 < 0 && e.1 < 0 && e.2 < 0 && e.3 < 0);
    let mut b = Board::start();
    let mut seen: Vec<u64> = vec![b.hash];
    let apply_entry = |bd: &Board, e: (i64, i64, i64, i64), is_marker: bool| -> Board {
        let (x0, y0, x1, y1) = e;
        if x1 < 0 {
            // Pass (or the start marker, which is not a turn at all).
            return if is_marker { *bd } else { bd.make_pass() };
        }
        if !(0..=6).contains(&x0)
            || !(0..=6).contains(&y0)
            || !(0..=6).contains(&x1)
            || !(0..=6).contains(&y1)
        {
            // Malformed real entry: flip via pass to keep the turn aligned.
            return bd.make_pass();
        }
        let from = (y0 as usize) * 7 + (x0 as usize);
        let to = (y1 as usize) * 7 + (x1 as usize);
        // Apply the log blindly (the platform's history is law); the range
        // check above keeps make() in bounds.
        bd.make(Move {
            from: from as u8,
            to: to as u8,
        })
    };
    for i in 0..opp_seq.len() {
        b = apply_entry(&b, opp_seq[i], i == 0 && marker);
        seen.push(b.hash);
        if i < resps.len() {
            b = apply_entry(&b, resps[i], false);
            seen.push(b.hash);
        }
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
    let mut searcher = Searcher::with_tt_bits(23); // Hash-128 nominal (~192MiB actual)
    // Botzone deployment: the platform judge has no passes — the game ends
    // the moment either side cannot move (mobile side absorbs empties).
    // Rules are a BUILD-TIME selection (_bz monomorphization), so the
    // search knows them with zero hot-path cost.
    // (UAI serve keeps classic pass rules unless setoption BotzoneRules.)
    // Botzone sandbox CPU count is unknown and the 800ms budget leaves no
    // room for a calibration sweep: use up to 4 workers, never oversubscribe.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 4);
    let limits = SearchLimits {
        time: Some(Duration::from_millis(budget)),
        max_nodes: None,
        // No depth cap: the hard deadline is the only bound, so the full
        // 800/1500ms budget is burned (no soft early-exit).
        max_depth: u32::MAX,
    };
    let r = parallel_search_bz(&mut searcher, &b, &limits, threads);
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
