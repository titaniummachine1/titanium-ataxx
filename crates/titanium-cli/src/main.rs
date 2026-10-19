//! Titanium CLI - native testing harness for the engine.
//!
//! Commands (all take --net <file.s1|none> = eval preset, no env vars):
//!   bench [--depth N] [--net F]      search speed from the start position
//!   serve [--net F]                  UAI engine (match children use this)
//!   match --games N [--time MS | --nodes N] [--net F] [--opp CMD] [--opp-net F]
//!                                    [--opp-time M | --opp-nodes N] [--out F]
//!   perft [depth]                    movegen node counts (mirror cross-checked)
//!   selfplay [--games N] [--time MS] engine vs engine, logs to logs/
//!   sperft ...                       speed truth suite
//!
//! Presets: --net none (classical) or path to .s1 (default data/nnue/sancta_w.s1
//! when --net is passed bare). scripts/gate.ps1 wraps the two standard gates.

use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use titanium::{best_move, now_ms, parallel_search, parallel_search_bz, Board, Move, SearchLimits, Searcher, RING1};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "perft" => perft_cmd(first_int(&args).unwrap_or(4) as u32),
        "bench" => bench_cmd(
            flag_int(&args, "--depth").unwrap_or(8) as u32,
            flag_str(&args, "--net"),
            flag_str(&args, "--tuple"),
            args.iter().any(|x| x == "--tuple-native"),
        ),
        "selfplay" => selfplay_cmd(
            flag_int(&args, "--games").unwrap_or(2) as u32,
            flag_int(&args, "--time").unwrap_or(300) as u64,
            flag_str(&args, "--out"),
        ),
        "bestmove" => bestmove_cmd(
            positional_str(&args).or_else(|| Some(Board::start().to_string())),
            flag_int(&args, "--time").map(|ms| ms as u64),
        ),
        "label" => label_cmd(
            flag_int(&args, "--nodes").unwrap_or(20000) as u64,
            flag_str(&args, "--tuple"),
            args.iter().any(|x| x == "--tuple-native"),
        ),
        "play" => play_cmd(flag_int(&args, "--time").unwrap_or(1000) as u64),
        "show" => show_cmd(),
        "genbench" => genbench_cmd(flag_int(&args, "--iters").unwrap_or(200_000) as u64),
        "serve" => serve_cmd(
            flag_int(&args, "--tt-bits").unwrap_or(if args.iter().any(|x| x == "--tournament") {
                23
            } else {
                20
            }) as usize,
            flag_str(&args, "--net"),
            flag_str(&args, "--tuple"),
            args.iter().any(|x| x == "--tuple-native"),
            args.iter().any(|x| x == "--tournament"),
            flag_int(&args, "--threads").map(|n| n as usize),
            args.iter().any(|x| x == "--bz-rules"),
        ),
        "match" => match_cmd(
            flag_int(&args, "--games").unwrap_or(20) as u32,
            flag_int(&args, "--time").unwrap_or(1000) as u64,
            flag_str(&args, "--opp"),
            flag_str(&args, "--net"),
            flag_str(&args, "--opp-net"),
            flag_str(&args, "--tuple"),
            flag_str(&args, "--opp-tuple"),
            args.iter().any(|x| x == "--tuple-native"),
            args.iter().any(|x| x == "--opp-tuple-native"),
            flag_int(&args, "--opp-time").unwrap_or(1000) as u64,
            flag_int(&args, "--opp-depth").map(|d| d as u32),
            flag_int(&args, "--opp-nodes").map(|n| n as u64),
            flag_int(&args, "--nodes").map(|n| n as u64),
            flag_int(&args, "--start-game").unwrap_or(1) as u32,
            flag_str(&args, "--out"),
        ),
        "sperft" => sperft_cmd(
            flag_int(&args, "--depth").unwrap_or(8) as u32,
            flag_int(&args, "--games").unwrap_or(8) as usize,
            flag_str(&args, "--ordering"),
        ),
        _ => {
            print_help();
            ExitCode::SUCCESS
        }
    }
}

fn print_help() {
    println!(
        "titanium-cli - Ataxx engine test harness\n\
         \n\
         USAGE (eval preset = --net <file.s1|none>; NO env vars):\n\
          \x20 titanium-cli bench [--depth N] [--net F]    search speed benchmark\n\
          \x20 titanium-cli serve [--net F] [--tournament] [--threads N]\n\
          \x20             UAI engine (tournament = autotuned SMP + 128MB TT + auto-ponder)\n\
         \x20 titanium-cli match --games N --opp \"CMD\" [--net F] [--opp-net F]\n\
         \x20             [--time MS --opp-time MS | --nodes N --opp-nodes N]\n\
         \x20             [--out F]                       colors alternate\n\
         \x20 titanium-cli selfplay [--games N] [--time MS] engine vs engine into logs/\n\
         \x20 titanium-cli sperft [--depth N] [--games N]   search throughput suite\n\
         \n\
         Scripts: scripts/gate.ps1 <games> <net.s1|none>  (5k + 100ms vs classical)\n\
         \n\
         Board string: 49 chars, rows top->bottom, chars . x o # then side b|w.\n\
         Squares: a1 = bottom-left, g7 = top-right. Moves like e2e3 or e2-e3."
    );
}

// ---------- arg helpers ----------

fn first_int(args: &[String]) -> Option<i64> {
    args.get(1)?.parse().ok()
}

fn flag_int(args: &[String], name: &str) -> Option<i64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn flag_str(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn positional_str(args: &[String]) -> Option<String> {
    args.get(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
}

// ---------- perft ----------

fn mirror(b: &Board) -> Board {
    let flip = |mut bb: u64| -> u64 {
        let mut out = 0u64;
        while bb != 0 {
            let sq = bb.trailing_zeros();
            bb &= bb - 1;
            out |= 1u64 << (sq / 7 * 7 + (6 - sq % 7));
        }
        out
    };
    let mut m = Board {
        occ: [flip(b.occ[1]), flip(b.occ[0])],
        blockers: flip(b.blockers),
        turn: 1 - b.turn,
        passes: b.passes,
        ply: b.ply,
        piece_cnt: [b.piece_cnt[1], b.piece_cnt[0]],
        blocker_cnt: b.blockers.count_ones() as u8,
        hash: 0,
        // PST rows are file-symmetric: color-swap + file-mirror negates.
        pst: -b.pst,
    };
    m.hash = if m.turn == 1 { titanium::ZOB_SIDE } else { 0 };
    for c in 0..2 {
        let mut bb = m.occ[c];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            m.hash ^= titanium::ZOB_PIECE[c][sq];
        }
    }
    m
}

fn perft_cmd(depth: u32) -> ExitCode {
    let start = Board::start();
    let mirrored = mirror(&start);
    println!("Perft from the start position (cross-checked against mirrored board):");
    for d in 1..=depth {
        let t = Instant::now();
        let n = titanium::perft_bb(start.occ[0], start.occ[1], start.blockers, d);
        let m = titanium::perft_bb(mirrored.occ[0], mirrored.occ[1], mirrored.blockers, d);
        let status = if n == m { "ok" } else { "MISMATCH" };
        println!(
            "  depth {d}: {:>10} nodes  ({:>10.1} knps)  mirror={m} [{status}]",
            n,
            n as f64 / t.elapsed().as_secs_f64() / 1000.0
        );
        if n != m {
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

// ---------- bench ----------

fn load_net_preset(spec: &Option<String>) -> Result<Option<titanium::S1Net>, String> {
    // --net none (or flag absent) = classical. --net <path> = that .s1.
    // Owned net, NO Arc: Searcher keeps a raw pointer for the search
    // duration (caller-owned, read-only). Zero refcount in the tree.
    match spec {
        None => Ok(None),
        Some(s) if s == "none" || s.is_empty() => Ok(None),
        Some(s) => {
            let path = if s == "default" { "data/nnue/sancta_w.s1".into() } else { s.clone() };
            match titanium::S1Net::load(&path) {
                Some(n) => Ok(Some(n)),
                None => Err(format!("net file missing: {path}")),
            }
        }
    }
}

fn net_flag_str(spec: &Option<String>) -> String {
    // Re-emit for match children: --net <spec> flows to the opp serve child
    // as --opp-net; same binary, no env inheritance games.
    match spec {
        None => "--net none".to_string(),
        Some(s) if s == "none" || s.is_empty() => "--net none".to_string(),
        Some(s) => format!("--net {s}"),
    }
}

/// --tuple none (or flag absent) = pure hand eval. --tuple <path> = that
/// .tup additive table. Owned table, borrowed by the searcher (same
/// zero-refcount pattern as the net).
fn load_tuple_preset(spec: &Option<String>) -> Result<Option<titanium::TupleTable>, String> {
    match spec {
        None => Ok(None),
        Some(s) if s == "none" || s.is_empty() => Ok(None),
        Some(s) => match titanium::TupleTable::load(s) {
            Ok(t) => Ok(Some(t)),
            Err(e) => Err(format!("tuple file bad ({s}): {e}")),
        },
    }
}

fn tuple_flag_str(spec: &Option<String>) -> String {
    // Re-emit for match children: --tuple <spec> flows to the opp serve
    // child as --opp-tuple.
    match spec {
        None => "--tuple none".to_string(),
        Some(s) if s == "none" || s.is_empty() => "--tuple none".to_string(),
        Some(s) => format!("--tuple {s}"),
    }
}

fn bench_cmd(depth: u32, net_spec: Option<String>, tup_spec: Option<String>, tup_native: bool) -> ExitCode {
    let net = match load_net_preset(&net_spec) {
        Ok(n) => {
            match &net_spec {
                Some(s) if s != "none" && !s.is_empty() => println!("Bench: net ON ({s})"),
                _ => println!("Bench: classical (no net)"),
            }
            n
        }
        Err(e) => {
            eprintln!("Bench: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tup = match load_tuple_preset(&tup_spec) {
        Ok(t) => {
            match &tup_spec {
                Some(s) if s != "none" && !s.is_empty() => println!("Bench: tuple ON ({s})"),
                _ => println!("Bench: hand eval (no tuple)"),
            }
            t
        }
        Err(e) => {
            eprintln!("Bench: {e}");
            return ExitCode::FAILURE;
        }
    };
    let b = Board::start();
    println!("Bench: start position, fixed depth {depth} (no time limit)");
    let limits = SearchLimits {
        time: None,
        max_nodes: None,
        max_depth: depth,
    };
    let mut searcher = Searcher::new();
    searcher.set_sancta(net.as_ref());
    searcher.set_tuple(tup.as_ref());
    searcher.set_tuple_native(tup_native && tup.is_some());
    if tup_native {
        println!("Bench: tuple-NATIVE eval (no E2/E4 scans)");
    }
    let r = searcher.search(&b, &limits);
    let secs = r.elapsed.as_secs_f64();
    println!(
        "  best {}  score {}  depth {}  nodes {}  nps {:.0}",
        move_str(r.best),
        r.score,
        r.depth,
        r.nodes,
        r.nodes as f64 / secs
    );
    ExitCode::SUCCESS
}

// ---------- selfplay ----------

fn move_str(m: Option<Move>) -> String {
    match m {
        Some(m) => format!("{}{}", Board::sq_name(m.from), Board::sq_name(m.to)),
        None => "pass".into(),
    }
}

fn selfplay_cmd(games: u32, time_ms: u64, out: Option<String>) -> ExitCode {
    let mut report = String::new();
    report.push_str(&format!(
        "Titanium selfplay: {games} games, {time_ms} ms per move\n===\n"
    ));
    let mut wins = [0u32; 2];
    let mut draws = 0u32;

    for game in 1..=games {
        let mut b = Board::start();
        let mut searcher = Searcher::new();
        let mut log = String::new();
        let mut ply = 0u32;
        let started = Instant::now();

        while !b.game_over() && ply < 400 {
            let limits = SearchLimits {
                time: Some(Duration::from_millis(time_ms)),
                max_nodes: None,
                max_depth: 24,
            };
            let r = if b.has_moves(b.turn) {
                searcher.search(&b, &limits)
            } else {
                Default::default()
            };
            let mv = match r.best {
                Some(m) => m,
                None => Move::PASS,
            };
            if mv.is_pass() {
                log.push_str(&format!("{:>3}. pass\n", ply + 1));
            } else {
                log.push_str(&format!(
                    "{:>3}. {:<8} depth {:>2}  nodes {:>9}  score {:>6}\n",
                    ply + 1,
                    move_str(Some(mv)),
                    r.depth,
                    r.nodes,
                    r.score
                ));
            }
            b = if mv.is_pass() {
                b.make_pass()
            } else {
                b.make(mv)
            };
            ply += 1;
        }

        let (bl, wh) = b.counts();
        let result = if ply >= 400 && !b.game_over() {
            "draw-move-limit".to_string()
        } else {
            match b.winner() {
                0 => {
                    wins[0] += 1;
                    "black wins".into()
                }
                1 => {
                    wins[1] += 1;
                    "white wins".into()
                }
                _ => {
                    draws += 1;
                    "draw".into()
                }
            }
        };
        report.push_str(&format!(
            "-- game {game}: {result}  (black {bl} - white {wh}, {ply} plies, {:.1}s)\n",
            started.elapsed().as_secs_f64()
        ));
        report.push_str(&log);
        report.push('\n');
        println!("game {game}/{games}: {result} ({bl}-{wh})");
    }

    report.push_str(&format!(
        "===\ntotal: black {0} - white {1}, draws {draws}\n",
        wins[0], wins[1]
    ));

    let path: PathBuf = out.map(PathBuf::from).unwrap_or_else(default_log_path);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::write(&path, &report) {
        Ok(_) => {
            println!("log written to {}", path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to write {}: {e}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn default_log_path() -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    PathBuf::from(format!("logs/selfplay_{ts}.txt"))
}

// ---------- label ----------
// Batch teacher-labeling for bootstrap training (S22): stdin lines of
// "<49-char board> <b|w>" (Board::from_str format), one "<score> <depth>
// <nodes>" per line on stdout, progress on stderr. ONE Searcher, NO reset
// between rows: TT entries are hash-verified so cross-row hits are
// impossible (different positions = different keys); the shared table just
// caches transpositions across rows. Killers/history carry as ordering
// hints only (matched moves are always legal here or ignored).
fn label_cmd(nodes: u64, tup_spec: Option<String>, tup_native: bool) -> ExitCode {
    let tup = match load_tuple_preset(&tup_spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("label: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut searcher = Searcher::new();
    searcher.set_tuple(tup.as_ref());
    searcher.set_tuple_native(tup_native && tup.is_some());
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    let mut n = 0u64;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let Some(b) = Board::from_str(t) else {
            eprintln!("label: bad board line {n}");
            return ExitCode::FAILURE;
        };
        let limits = SearchLimits {
            time: None,
            max_nodes: Some(nodes),
            max_depth: 30,
        };
        let r = searcher.search(&b, &limits);
        use std::io::Write as _;
        let _ = writeln!(out, "{} {} {}", r.score, r.depth, r.nodes);
        n += 1;
        if n % 20000 == 0 {
            eprintln!("label: {n} rows");
        }
    }
    eprintln!("label: done {n} rows");
    ExitCode::SUCCESS
}

// ---------- bestmove ----------

fn bestmove_cmd(board: Option<String>, time_ms: Option<u64>) -> ExitCode {
    let Some(text) = board else {
        eprintln!("bestmove: missing board string");
        return ExitCode::FAILURE;
    };
    let Some(b) = Board::from_str(&text) else {
        eprintln!("bestmove: invalid board string");
        return ExitCode::FAILURE;
    };
    println!("{}", b.pretty());
    let limits = SearchLimits {
        time: time_ms.map(Duration::from_millis),
        max_nodes: None,
        max_depth: 24,
    };
    let r = best_move(&b, &limits);
    if b.game_over() {
        println!("game over: {}", outcome(&b));
        return ExitCode::SUCCESS;
    }
    if r.best.is_none() {
        println!("bestmove pass (no legal moves)");
        return ExitCode::SUCCESS;
    }
    println!(
        "bestmove {}  score {}  depth {}  nodes {}  {:.0} ms",
        move_str(r.best),
        r.score,
        r.depth,
        r.nodes,
        r.elapsed.as_millis()
    );
    ExitCode::SUCCESS
}

fn outcome(b: &Board) -> String {
    let (bl, wh) = b.counts();
    match b.winner() {
        0 => format!("black wins {bl}-{wh}"),
        1 => format!("white wins {wh}-{bl}"),
        _ => format!("draw {bl}-{wh}"),
    }
}

// ---------- play ----------

fn parse_move(text: &str) -> Option<Move> {
    let t: String = text.chars().filter(|&c| c != '-').collect();
    if t.len() != 4 {
        return None;
    }
    let from = Board::parse_sq(&t[0..2])?;
    let to = Board::parse_sq(&t[2..4])?;
    Some(Move { from, to })
}

/// Apply one UAI move token to a board (for `position ... moves ...`).
/// Accepts 4-char jumps ("e2e4"/"e2-e4"), 2-char clones ("e4"),
/// and nulls ("0000"/"null"/"pass"). Matches against our own legal
/// movegen, so only truly legal tokens apply (mirrors libataxx from_uai:
/// from==to 4-char tokens are rejected).
fn apply_uai_token(b: &Board, tok: &str) -> Option<Board> {
    let t: String = tok
        .chars()
        .filter(|&c| c != '-')
        .collect::<String>()
        .to_lowercase();
    if t == "0000" || t == "null" || t == "pass" {
        return Some(b.make_pass());
    }
    let legal = b.legal_moves();
    if t.len() == 2 {
        let to = Board::parse_sq(&t)?;
        legal
            .iter()
            .find(|m| m.to == to && m.is_clone())
            .map(|m| b.make(m))
    } else if t.len() == 4 {
        let from = Board::parse_sq(&t[0..2])?;
        let to = Board::parse_sq(&t[2..4])?;
        if from == to {
            return None;
        }
        legal
            .iter()
            .find(|m| m.from == from && m.to == to)
            .map(|m| b.make(m))
    } else {
        None
    }
}

fn play_cmd(time_ms: u64) -> ExitCode {
    println!("Titanium console game. You are black (x). Moves like e2e3, or: moves, quit");
    let mut b = Board::start();
    let human = 0u8;
    let stdin = io::stdin();
    let mut searcher = Searcher::new();

    loop {
        println!("{}", b.pretty());
        if b.game_over() {
            println!("Game over: {}", outcome(&b));
            return ExitCode::SUCCESS;
        }

        if b.turn == human {
            if !b.has_moves(human) {
                println!("You have no moves - pass.");
                b = b.make_pass();
                continue;
            }
            print!("your move: ");
            let _ = io::stdout().flush();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                return ExitCode::SUCCESS;
            }
            let line = line.trim();
            match line {
                "quit" | "q" => return ExitCode::SUCCESS,
                "moves" => {
                    let mv: Vec<String> = b.legal_moves().iter().map(|m| move_str(Some(m))).collect();
                    println!("{}", mv.join(" "));
                    continue;
                }
                _ => {}
            }
            match parse_move(line).filter(|m| b.legal_moves().contains(m)) {
                Some(m) => b = b.make(m),
                None => {
                    println!("illegal move, try again");
                    continue;
                }
            }
        } else {
            if !b.has_moves(1) {
                println!("engine passes.");
                b = b.make_pass();
                continue;
            }
            let limits = SearchLimits {
                time: Some(Duration::from_millis(time_ms)),
                max_nodes: None,
                max_depth: 24,
            };
            let r = searcher.search(&b, &limits);
            match r.best {
                Some(m) => {
                    println!(
                        "engine plays {} (depth {}, {} nodes, {:.0} ms)",
                        move_str(Some(m)),
                        r.depth,
                        r.nodes,
                        r.elapsed.as_millis()
                    );
                    b = b.make(m);
                }
                None => {
                    println!("engine passes.");
                    b = b.make_pass();
                }
            }
        }
    }
}

// ---------- serve ----------


// ---------- serve (UAI) ----------

/// Minimal UAI engine loop so any titanium build (main/candidate snapshots)
/// can be driven by `match --opp "path\to\titanium-cli.exe serve"`.
fn out_line(s: &str) {
    use std::io::Write;
    let mut o = io::stdout().lock();
    let _ = writeln!(o, "{s}");
    let _ = o.flush();
}

/// Work items for the search worker thread. The protocol thread keeps
/// reading while a search runs (so `stop` / `ponderhit` are honored live);
/// everything touching the Searcher goes through this FIFO channel.
enum Job {
    SetParam(String, i64),
    SetHash(usize),
    SetThreads(usize),
    SetBzRules(bool),
    NewGame,
    Go {
        board: Option<Box<Board>>,
        limits: SearchLimits,
        /// Silent jobs (tournament auto-ponder) warm the TT but never emit
        /// `bestmove` — the GUI never asked for a move on that position.
        silent: bool,
    },
    Quit,
}

/// Same-position identity for FEN continuity: occ + blockers + side only.
/// ply/passes are deliberately excluded — the continuous board made via
/// make() carries the true counters.
fn board_state_eq(a: &Board, b: &Board) -> bool {
    a.occ == b.occ && a.blockers == b.blockers && a.turn == b.turn
}

/// Can `target` be reached from `old` in <= `max_plies` legal plies (real
/// moves, or a pass when side has none)? Returns that child built via
/// make() so ply/passes/TT identity carry over — the "site sent a position
/// instead of the move" continuity rule: never wipe memory when the
/// position is inside our legal-move set. Empty-cell count strictly falls
/// per move, so a match is always the true path (no false positives).
fn reach_board(old: &Board, target: &Board, max_plies: u32) -> Option<Board> {
    if board_state_eq(old, target) {
        return Some(*old);
    }
    let mut frontier: Vec<Board> = vec![*old];
    for _ in 1..=max_plies {
        let mut next: Vec<Board> = Vec::new();
        for b in &frontier {
            let list = b.legal_moves();
            let mut kids: Vec<Board> = Vec::with_capacity(list.len() + 1);
            for m in list.iter() {
                kids.push(b.make(m));
            }
            if kids.is_empty() {
                kids.push(b.make_pass());
            }
            for c in kids {
                if board_state_eq(&c, target) {
                    return Some(c);
                }
                next.push(c);
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    None
}

/// Periodic `info` while searching (fn pointer hook on the Searcher).
fn serve_emit_info(depth: u32, score: i32, nodes: u64, ms: u64) {
    out_line(&format!(
        "info depth {depth} score cp {score} nodes {nodes} time {ms}"
    ));
}

/// Fixed-window throughput probe: total nodes in a 120ms Lazy-SMP search
/// from the start position with a fresh 16MB TT.
fn measure_threads(b: &Board, threads: usize) -> u64 {
    let mut s = Searcher::with_tt_bits(20);
    let limits = SearchLimits {
        time: Some(Duration::from_millis(120)),
        max_nodes: None,
        max_depth: u32::MAX,
    };
    parallel_search(&mut s, b, &limits, threads).nodes
}

/// Tournament startup calibration: try 1, 2, 3, ... threads (then 4, 6, 8,
/// doubling past 8) and stop when adding threads no longer grows total
/// throughput — never oversubscribe into slowdown. Caps at the machine's
/// parallelism (and 32). Each probe is ~120ms; the whole sweep is ~1s once
/// per process start, amortized over the tournament.
fn autotune_threads() -> usize {
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(32);
    if avail <= 1 {
        return 1;
    }
    let b = Board::start();
    let mut cand = vec![1usize, 2, 3];
    let mut t = 4;
    while t <= avail {
        cand.push(t);
        t = if t < 8 { t + 2 } else { t * 2 };
    }
    let mut best_t = 1usize;
    let mut best_nodes = 1u64;
    for &n in &cand {
        if n > avail {
            break;
        }
        let nodes = measure_threads(&b, n).max(1);
        eprintln!("tournament autotune: threads={n} nodes={nodes}");
        // Always accept 1/2/3 (ties keep the smaller count); past that,
        // require a real >5% gain or stop — more threads are oversubscribe.
        let hurdle = if n <= 3 { best_nodes } else { best_nodes * 105 / 100 };
        if nodes >= hurdle {
            best_t = n;
            best_nodes = best_nodes.max(nodes);
        } else {
            break;
        }
    }
    best_t.max(1)
}

fn serve_cmd(
    tt_bits: usize,
    net_spec: Option<String>,
    tup_spec: Option<String>,
    tup_native: bool,
    tournament: bool,
    threads_opt: Option<usize>,
    bz_rules: bool,
) -> ExitCode {
    // Eval presets come from --net / --tuple ONLY. Owned tables move into
    // the worker thread together with the searcher that borrows them.
    let owned = match load_net_preset(&net_spec) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("serve: {e}");
            return ExitCode::FAILURE;
        }
    };
    let owned_tup = match load_tuple_preset(&tup_spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("serve: {e}");
            return ExitCode::FAILURE;
        }
    };
    let stop_flag = Arc::new(AtomicBool::new(false));
    let hit_deadline = Arc::new(AtomicU64::new(0));
    let (tx, rx) = mpsc::channel::<Job>();

    // Thread count: explicit --threads wins; tournament mode autotunes
    // (1,2,3... while total throughput grows); otherwise single-threaded.
    let threads: usize = match threads_opt {
        Some(n) => n.clamp(1, 256),
        None if tournament => autotune_threads(),
        None => 1,
    };

    // The search worker owns the Searcher (nets borrowed from its locals).
    let sf_worker = Arc::clone(&stop_flag);
    let hd_worker = Arc::clone(&hit_deadline);
    let worker = std::thread::spawn(move || {
        let sf_ctrl = Arc::clone(&sf_worker);
        let hd_ctrl = Arc::clone(&hd_worker);
        let mut searcher = Searcher::with_tt_bits(tt_bits);
        searcher.set_sancta(owned.as_ref());
        searcher.set_tuple(owned_tup.as_ref());
        searcher.set_tuple_native(tup_native && owned_tup.is_some());
        searcher.set_external_control(sf_worker, hd_worker);
        searcher.set_iteration_hook(serve_emit_info);
        let mut threads = threads;
        // Rules selection is a BUILD-TIME monomorphization in the engine
        // (`search` vs `search_bz`); this bool only picks which compiled
        // copy runs — never a hot-path branch. Changing it clears the TT:
        // bounds learned under one rule set are unsound under the other.
        let mut bz_rules = bz_rules;
        // Tournament tryhard: after every real move, keep thinking on the
        // opponent's clock (silent ponder on the post-move board). The warm
        // TT carries over, so the site never needs to send `ponderhit` —
        // the next `position` + `go` just converts into a timed search.
        let auto_ponder = tournament;
        while let Ok(job) = rx.recv() {
            match job {
                Job::SetParam(name, v) => {
                    searcher.set_param(&name, v);
                }
                Job::SetHash(bits) => searcher.set_tt_bits(bits),
                Job::SetThreads(n) => {
                    threads = n.clamp(1, 256);
                }
                Job::SetBzRules(on) => {
                    if on != bz_rules {
                        bz_rules = on;
                        searcher.reset_for_new_game();
                    }
                }
                Job::NewGame => searcher.reset_for_new_game(),
                Job::Go {
                    board,
                    limits,
                    silent,
                } => {
                    // Fresh search: clear the external switches. A stale
                    // `stop` aimed at a previous search must not kill this one.
                    sf_ctrl.store(false, Ordering::Relaxed);
                    hd_ctrl.store(0, Ordering::Relaxed);
                    let Some(b) = board else {
                        eprintln!("serve: go without a position — replying 0000 (desync)");
                        if !silent {
                            out_line("bestmove 0000");
                        }
                        continue;
                    };
                    // Classic early-out: no legal reply and no forced terminal
                    // score to report. Under BZ rules the search itself owns
                    // this (root fast path scores the filled-count terminal).
                    if !bz_rules && (b.game_over() || !b.has_moves(b.turn)) {
                        if !silent {
                            out_line("bestmove 0000");
                        }
                        continue;
                    }
                    // Build-time dispatch: exactly one compiled copy runs.
                    let r = if bz_rules {
                        parallel_search_bz(&mut searcher, &b, &limits, threads)
                    } else {
                        parallel_search(&mut searcher, &b, &limits, threads)
                    };
                    if silent {
                        // Auto-ponder: TT warmed, nothing to report.
                        continue;
                    }
                    let mv = match r.best {
                        Some(m) => {
                            if m.is_clone() {
                                Board::sq_name(m.to)
                            } else {
                                format!("{}{}", Board::sq_name(m.from), Board::sq_name(m.to))
                            }
                        }
                        None => "0000".to_string(),
                    };
                    // Final info line (GUIs, logs, search-loss tuning).
                    out_line(&format!(
                        "info depth {} score cp {} nodes {} time {}",
                        r.depth,
                        r.score,
                        r.nodes,
                        r.elapsed.as_millis()
                    ));
                    out_line(&format!("bestmove {mv}"));
                    // Think on the opponent's time: silent infinite search on
                    // the post-move board. Aborted by the next go/stop/quit
                    // (pre-Go stop flag); never emits a move of its own.
                    if auto_ponder {
                        if let Some(m) = &r.best {
                            let post = b.make(*m);
                            if !post.game_over()
                                && post.has_moves(post.turn)
                                && !sf_ctrl.load(Ordering::Relaxed)
                            {
                                hd_ctrl.store(0, Ordering::Relaxed);
                                let pl = SearchLimits {
                                    time: None,
                                    max_nodes: None,
                                    max_depth: u32::MAX,
                                };
                                if bz_rules {
                                    let _ = parallel_search_bz(&mut searcher, &post, &pl, threads);
                                } else {
                                    let _ = parallel_search(&mut searcher, &post, &pl, threads);
                                }
                            }
                        }
                    }
                }
                Job::Quit => break,
            }
        }
    });
    out_line(&format!(
        "info string titanium tournament={tournament} threads={threads} tt_bits={tt_bits}"
    ));

    let stdin = io::stdin();
    let mut current: Option<Board> = None;
    // Allocation computed at `go ponder` time, armed as a deadline by
    // `ponderhit` (pondering runs with no clock until then).
    let mut pending_alloc_ms: Option<f64> = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let t = line.trim().to_string();
        if t.is_empty() {
            continue;
        }
        if t == "quit" {
            stop_flag.store(true, Ordering::Relaxed);
            let _ = tx.send(Job::Quit);
            break;
        }
        if t == "uai" {
            out_line("id name titanium-ataxx");
            out_line("id author titaniummachine1");
            out_line(&format!(
                "option name Hash type spin default {} min 1 max 16384",
                if tournament { 128 } else { 256 }
            ));
            out_line(&format!(
                "option name Threads type spin default {threads} min 1 max 256"
            ));
            out_line("option name BotzoneRules type check default false");
            out_line("option name Tempo type spin default 150 min 0 max 1000");
            out_line("option name ContactPen type spin default 15 min 0 max 200");
            out_line("option name MulticapPen type spin default 20 min 0 max 200");
            out_line("option name RFP0 type spin default 257 min 0 max 2000");
            out_line("option name RFP1 type spin default 347 min 0 max 2000");
            out_line("option name RFP2 type spin default 478 min 0 max 2000");
            out_line("option name RFP3 type spin default 774 min 0 max 2000");
            out_line("option name NullR type spin default 3 min 1 max 6");
            out_line("option name NullMinTargets type spin default 11 min 0 max 40");
            out_line("option name NullMaxFill type spin default 52 min 0 max 100");
            out_line("option name LMRSplit type spin default 9 min 1 max 40");
            out_line("option name LMRR1 type spin default 2 min 0 max 6");
            out_line("option name LMRR2 type spin default 3 min 0 max 6");
            out_line("option name LMRMode type spin default 0 min 0 max 3");
            out_line("option name LMRHalf type spin default 4 min 1 max 16");
            out_line("option name LMRSub type spin default 0 min 0 max 4");
            for i in 0..16 {
                let dflt = if i == 0 { 0 } else if i < 9 { 2 } else { 3 };
                out_line(&format!("option name LMRT{i} type spin default {dflt} min 0 max 8"));
            }
            out_line("option name LMRGap1 type spin default 2000000000 min 0 max 2000000000");
            out_line("option name LMRGap2 type spin default 2000000000 min 0 max 2000000000");
            out_line("option name PCutDepth type spin default 5 min 3 max 12");
            out_line("option name PCutMargin type spin default 2000000000 min 0 max 2000000000");
            out_line("option name PCutReduce type spin default 3 min 1 max 6");
            out_line("option name JumpEscape type spin default 0 min 0 max 10000");
            out_line("option name JumpAttack type spin default 0 min 0 max 10000");
            out_line("option name JumpGamble type spin default 0 min 0 max 10000");
            out_line("option name NullRDiv type spin default 0 min 0 max 16");
            out_line("option name JumpDanger type spin default 0 min 0 max 2000");
            out_line("option name JumpHole type spin default 0 min 0 max 2000");
            out_line("option name JumpSafe type spin default 0 min 0 max 2000");
            out_line("option name JumpCold type spin default 0 min 0 max 10000");
            out_line("option name JumpPrune type spin default 0 min 0 max 1");
            for i in 0..12 {
                out_line(&format!("option name BucketBias{i} type spin default 0 min -2000 max 2000"));
            }
            out_line("option name CurveBase type spin default 2 min 0 max 6");
            out_line("option name CurveSlope type spin default 6 min 0 max 64");
            out_line("option name CurveDepth type spin default 2 min 0 max 64");
            out_line("option name CurveCap type spin default 5 min 1 max 8");
            out_line("uaiok");
        } else if t == "isready" {
            // Answer immediately even mid-search (Stockfish behavior); the
            // job FIFO still orders config before the next `go`.
            out_line("readyok");
        } else if let Some(rest) = t.strip_prefix("setoption name ") {
            // "setoption name <Name> value <N>": Hash resizes the TT,
            // anything else routes to the tunable search params.
            if let Some((name, val)) = rest.split_once(" value ") {
                let name = name.trim();
                if name == "Hash" {
                    if let Ok(mb) = val.trim().parse::<usize>() {
                        // MB -> bits: entries are ~16 bytes; bits = log2(mb*2^20/16).
                        let bits = ((mb as f64 * 65536.0).log2().round() as usize)
                            .clamp(10, 26);
                        let _ = tx.send(Job::SetHash(bits));
                    }
                } else if name == "Threads" {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        let _ = tx.send(Job::SetThreads(n.clamp(1, 256)));
                    }
                } else if name == "BotzoneRules" {
                    // No-pass Botzone judge rules (stalemate = terminal).
                    // Default off = classic Ataxx with passes; the Botzone
                    // bundle hardcodes it on (it has no setoption channel).
                    let on = matches!(val.trim().to_lowercase().as_str(), "1" | "true" | "on");
                    let _ = tx.send(Job::SetBzRules(on));
                } else {
                    match val.trim().parse::<i64>() {
                        Ok(v) => {
                            let _ = tx.send(Job::SetParam(name.to_string(), v));
                        }
                        Err(_) => {
                            eprintln!("serve: bad value for {name}");
                        }
                    }
                }
            }
        } else if t == "uainewgame" {
            let _ = tx.send(Job::NewGame);
        } else if t == "position startpos" || t.starts_with("position startpos ") {
            let mut b = Board::start();
            if let Some(mv) = t.split_once(" moves ") {
                for tok in mv.1.split_whitespace() {
                    match apply_uai_token(&b, tok) {
                        Some(nb) => b = nb,
                        None => {
                            // Desync: never skip silently — log loudly and
                            // keep replaying best-effort on the stale board.
                            eprintln!(
                                "serve: bad move token '{tok}' — position drift, continuing best-effort"
                            );
                        }
                    }
                }
            }
            current = Some(b);
        } else if let Some(fen) = t.strip_prefix("position fen ") {
            // Split off an optional trailing " moves ..." (UAI standard).
            let (fen_part, moves_part) = match fen.split_once(" moves ") {
                Some((f, m)) => (f, Some(m)),
                None => (fen, None),
            };
            match Board::from_ataxx_fen(fen_part.trim()) {
                Some(mut b) => {
                    // Continuity rule: sites that *load a position* instead
                    // of replaying moves must not wipe our memory. If the
                    // fen is reachable from where we stand within 2 legal
                    // plies (our move + the opponent reply), adopt that
                    // child via make() — exact ply/passes, same TT identity.
                    if let Some(cont) =
                        current.as_ref().and_then(|old| reach_board(old, &b, 2))
                    {
                        b = cont;
                    }
                    if let Some(mv) = moves_part {
                        for tok in mv.split_whitespace() {
                            match apply_uai_token(&b, tok) {
                                Some(nb) => b = nb,
                                None => eprintln!(
                                    "serve: bad move token '{tok}' after fen — position drift, continuing best-effort"
                                ),
                            }
                        }
                    }
                    current = Some(b);
                }
                None => {
                    eprintln!("serve: unparseable fen — keeping previous position (desync)");
                }
            }
        } else if t.starts_with("go") {
            let mut nodes: Option<u64> = None;
            let mut movetime: Option<u64> = None;
            let mut depth: Option<u32> = None;
            let mut btime: Option<u64> = None;
            let mut wtime: Option<u64> = None;
            let mut binc: u64 = 0;
            let mut winc: u64 = 0;
            let mut infinite = false;
            let mut ponder = false;
            let toks: Vec<&str> = t.split_whitespace().collect();
            for (i, k) in toks.iter().enumerate() {
                match *k {
                    "nodes" => nodes = toks.get(i + 1).and_then(|v| v.parse().ok()),
                    "movetime" => movetime = toks.get(i + 1).and_then(|v| v.parse().ok()),
                    "depth" => depth = toks.get(i + 1).and_then(|v| v.parse().ok()),
                    "btime" => btime = toks.get(i + 1).and_then(|v| v.parse().ok()),
                    "wtime" => wtime = toks.get(i + 1).and_then(|v| v.parse().ok()),
                    "binc" => binc = toks.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0),
                    "winc" => winc = toks.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0),
                    "infinite" => infinite = true,
                    "ponder" => ponder = true,
                    _ => {}
                }
            }
            // Incremental clock (websites/GUIs): our share = time/30 + inc/2.
            let clock_ms: Option<u64> = match current.as_ref() {
                Some(b) if movetime.is_none() && (btime.is_some() || wtime.is_some()) => {
                    let (tm, inc) = if b.turn == 0 {
                        (btime, binc)
                    } else {
                        (wtime, winc)
                    };
                    tm.map(|ms| (ms / 30 + inc / 2).max(1))
                }
                _ => None,
            };
            // Stockfish semantics: `go infinite` / `go ponder` / bare `go`
            // run until `stop` (or ponderhit); a time allotment burns its
            // FULL budget (hard deadline, no soft early-exit); explicit
            // `depth`/`nodes` bound first when present.
            let alloc_ms: Option<u64> = movetime.filter(|ms| *ms > 0).or(clock_ms);
            let time_ms: Option<u64> = if infinite || ponder {
                None
            } else {
                alloc_ms
            };
            pending_alloc_ms = if ponder {
                alloc_ms.map(|ms| ms as f64)
            } else {
                None
            };
            let limits = SearchLimits {
                time: time_ms.map(Duration::from_millis),
                max_nodes: nodes,
                max_depth: depth.unwrap_or(u32::MAX),
            };
            // Abort any still-running search (double-`go` without `stop`);
            // the worker clears the flag when it starts this job.
            stop_flag.store(true, Ordering::Relaxed);
            let _ = tx.send(Job::Go {
                board: current.map(Box::new),
                limits,
                // GUI-requested searches always report. Silence is only for
                // the worker's own auto-ponder jobs (created internally).
                silent: false,
            });
        } else if t == "stop" {
            stop_flag.store(true, Ordering::Relaxed);
        } else if t == "ponderhit" {
            // Convert the free ponder search into a timed one: deadline =
            // now + the allocation computed at `go` time (our clock was
            // frozen while pondering, so the allotment is still valid).
            if let Some(alloc) = pending_alloc_ms.take() {
                hit_deadline.store((now_ms() + alloc) as u64, Ordering::Relaxed);
            }
        }
    }
    stop_flag.store(true, Ordering::Relaxed);
    let _ = tx.send(Job::Quit);
    let _ = worker.join();
    ExitCode::SUCCESS
}

// ---------- match ----------

struct UaiOpp {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    out: io::BufReader<std::process::ChildStdout>,
    name: String,
}

impl UaiOpp {
    fn start(cmd: &str) -> io::Result<UaiOpp> {
        use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
        let mut parts = cmd.split_whitespace();
        let program = parts
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "empty command"))?;
        let mut command = Command::new(program);
        command.args(parts);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child: Child = command.spawn()?;
        let stdin: ChildStdin = child.stdin.take().expect("stdin");
        let out: BufReader<ChildStdout> = BufReader::new(child.stdout.take().expect("stdout"));
        let mut opp = UaiOpp {
            child,
            stdin,
            out,
            name: program.to_string(),
        };
        opp.send("uai");
        if let Some(id) = opp.line_with("id name") {
            opp.name = id.to_string();
        }
        opp.wait_for("uaiok")?;
        opp.send("isready");
        opp.wait_for("readyok")?;
        Ok(opp)
    }

    fn send(&mut self, line: &str) {
        use std::io::Write;
        let _ = writeln!(self.stdin, "{line}");
        let _ = self.stdin.flush();
    }

    fn read_line(&mut self) -> Option<String> {
        let mut s = String::new();
        match self.out.read_line(&mut s) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(s.trim().to_string()),
        }
    }

    fn line_with(&mut self, prefix: &str) -> Option<String> {
        loop {
            let l = self.read_line()?;
            if l.starts_with(prefix) {
                return Some(l[prefix.len()..].trim().to_string());
            }
        }
    }

    fn wait_for(&mut self, token: &str) -> io::Result<()> {
        loop {
            match self.read_line() {
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("{token} never arrived"),
                    ))
                }
                Some(l) if l.starts_with(token) => return Ok(()),
                Some(_) => {}
            }
        }
    }

    /// Position + go, returns the bestmove token.
    fn best_move(&mut self, fen: &str, go: &str) -> Option<String> {
        self.send(&format!("position fen {fen}"));
        self.send(go);
        loop {
            let l = self.read_line()?;
            if let Some(rest) = l.strip_prefix("bestmove ") {
                return Some(rest.split_whitespace().next().unwrap_or("").to_string());
            }
        }
    }

    fn quit(&mut self) {
        self.send("quit");
        let _ = self.child.wait();
    }
}

/// Convert a UAI bestmove token to a titanium move in the context of `board`.
/// 2 chars = clone to that square (any adjacent source is equivalent).
/// 4 chars = jump from->to.
fn parse_opp_move(board: &Board, token: &str) -> Option<Move> {
    let token = token.trim();
    if token.len() == 2 {
        let to = Board::parse_sq(token)?;
        let clones = board.me() & RING1[to as usize];
        if clones == 0 {
            return None;
        }
        return Some(Move {
            from: clones.trailing_zeros() as u8,
            to,
        });
    }
    if token.len() == 4 {
        let from = Board::parse_sq(&token[..2])?;
        let to = Board::parse_sq(&token[2..])?;
        return Some(Move { from, to });
    }
    None
}

fn match_cmd(
    games: u32,
    time_ms: u64,
    opp_cmd: Option<String>,
    net_spec: Option<String>,
    opp_net_spec: Option<String>,
    tup_spec: Option<String>,
    opp_tup_spec: Option<String>,
    tup_native: bool,
    opp_tup_native: bool,
    opp_time: u64,
    opp_depth: Option<u32>,
    opp_nodes: Option<u64>,
    max_nodes: Option<u64>,
    start_game: u32,
    out: Option<String>,
) -> ExitCode {
    // Eval presets, explicit per side — no env vars anywhere:
    // --net = OUR side (in-process), --opp-net = child side (appended to the
    // opp serve command). --opp self = both in-process (opp side gets opp-net).
    let net = match load_net_preset(&net_spec) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("match: {e}");
            return ExitCode::FAILURE;
        }
    };
    let opp_net = match load_net_preset(&opp_net_spec) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("match: opp {e}");
            return ExitCode::FAILURE;
        }
    };
    let tup = match load_tuple_preset(&tup_spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("match: {e}");
            return ExitCode::FAILURE;
        }
    };
    let opp_tup = match load_tuple_preset(&opp_tup_spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("match: opp {e}");
            return ExitCode::FAILURE;
        }
    };
    // "self" = titanium vs titanium in-process (no external process).
    // External opp: auto-append the opp preset to "<exe> serve" commands so
    // one flag controls the child, no 50-line powershell, no env games.
    let mut opp = if opp_cmd.as_deref() == Some("self") {
        None
    } else {
        let cmd = match opp_cmd {
            Some(c) => {
                // Auto-append opp presets to "<exe> serve" child commands.
                let mut with_flags = c;
                if with_flags.trim_end().ends_with("serve") {
                    if !with_flags.contains("--net") {
                        with_flags = format!("{with_flags} {}", net_flag_str(&opp_net_spec));
                    }
                    if !with_flags.contains("--tuple") {
                        with_flags = format!("{with_flags} {}", tuple_flag_str(&opp_tup_spec));
                    }
                    if opp_tup_native && !with_flags.contains("--tuple-native") {
                        with_flags = format!("{with_flags} --tuple-native");
                    }
                }
                with_flags
            }
            None => {
                eprintln!("match: --opp \"<UAI engine command>\" or \"self\" is required");
                return ExitCode::FAILURE;
            }
        };
        match UaiOpp::start(&cmd) {
            Ok(o) => Some(o),
            Err(e) => {
                eprintln!("match: failed to start opponent '{cmd}': {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    let opp_name = match &opp {
        Some(o) => o.name.clone(),
        None => "titanium".to_string(),
    };
    println!("titanium vs {opp_name} — {games} games");

    let mut tally = (0u32, 0u32, 0u32); // titanium wins, opp wins, draws
    let limits_desc = format!(
        "titanium limits: {}, net: {} vs {}",
        match max_nodes {
            Some(n) => format!("{n} nodes"),
            None => format!("{time_ms} ms"),
        },
        net_spec.as_deref().unwrap_or("none"),
        opp_net_spec.as_deref().unwrap_or("none"),
    );
    let mut report = String::new();
    report.push_str(&format!(
        "Match: titanium vs {opp_name} — games {start_game}..{} — {limits_desc}, opp {}\n===\n",
        start_game + games - 1,
        opp_depth
            .map(|d| format!("depth {d}"))
            .unwrap_or_else(|| format!("movetime {opp_time} ms"))
    ));

    // Incremental log: header first, then append per game — played games
    // are never lost, even if the run is killed.
    let path: PathBuf = out.map(PathBuf::from).unwrap_or_else(default_log_path);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&path, &report);

    for game in start_game..start_game + games {
        let mut b = Board::start();
        let mut searcher = Searcher::new(); // fresh TT per game
        searcher.set_sancta(net.as_ref());
        searcher.set_tuple(tup.as_ref());
        searcher.set_tuple_native(tup_native && tup.is_some());
        let mut opp_searcher = Searcher::new(); // for --opp self
        opp_searcher.set_sancta(opp_net.as_ref());
        opp_searcher.set_tuple(opp_tup.as_ref());
        opp_searcher.set_tuple_native(opp_tup_native && opp_tup.is_some());
        let titanium_is_black = game % 2 == 1;
        let mut log = String::new();
        let mut result = None;

        let node_limits = |nodes: Option<u64>| SearchLimits {
            time: if time_ms == 0 { None } else { Some(Duration::from_millis(time_ms)) },
            max_nodes: nodes,
            max_depth: 24,
        };
        let titanium_limits = node_limits(max_nodes);
        let self_opp_limits = node_limits(max_nodes);

        while !b.game_over() && b.ply < 300 {
            let side = b.turn;
            let titanium_moves = (side == 0) == titanium_is_black;
            if !b.has_moves(side) {
                log.push_str("pass ");
                b = b.make_pass();
                continue;
            }
            let mv = if titanium_moves {
                searcher.search(&b, &titanium_limits).best
            } else {
                match &mut opp {
                    Some(o) => {
                        let go = if let Some(n) = opp_nodes {
                            format!("go nodes {n}")
                        } else if let Some(d) = opp_depth {
                            format!("go depth {d}")
                        } else {
                            format!("go movetime {opp_time}")
                        };
                        let reply = o.best_move(&b.to_ataxx_fen(), &go);
                        reply.as_deref().and_then(|t| parse_opp_move(&b, t))
                    }
                    None => opp_searcher.search(&b, &self_opp_limits).best,
                }
            };
            match mv {
                Some(m) => {
                    log.push_str(&format!("{} ", move_str(Some(m))));
                    b = b.make(m);
                }
                None => {
                    if titanium_moves {
                        log.push_str("pass ");
                        b = b.make_pass();
                    } else {
                        log.push_str("<forfeit> ");
                        result = Some(if side == 0 { 1 } else { 0 });
                        break;
                    }
                }
            }
        }

        let outcome = result.unwrap_or_else(|| {
            if b.ply >= 300 && !b.game_over() {
                2
            } else {
                b.winner()
            }
        });
        let (bl, wh) = b.counts();
        let (titanium_color, opp_color) = if titanium_is_black { ("x", "o") } else { ("o", "x") };
        let titanium_won =
            (outcome == 0 && titanium_is_black) || (outcome == 1 && !titanium_is_black);
        let line = if outcome == 2 {
            tally.2 += 1;
            format!("draw {bl}-{wh}")
        } else if titanium_won {
            tally.0 += 1;
            format!("titanium wins {bl}-{wh}")
        } else {
            tally.1 += 1;
            format!("opponent wins {bl}-{wh}")
        };
        let game_line = format!(
            "-- game {game}: titanium={titanium_color} opp={opp_color}: {line} ({} plies)\n{}\n",
            b.ply, log
        );
        report.push_str(&game_line);
        if let Ok(mut f) = fs::OpenOptions::new().append(true).open(&path) {
            use std::io::Write;
            let _ = writeln!(f, "{game_line}");
        }
        println!("game {game}: {line}");
    }

    if let Some(o) = &mut opp {
        o.quit();
    }
    report.push_str(&format!(
        "===\ntotal: titanium {} - {opp_name} {}, draws {}\n",
        tally.0, tally.1, tally.2
    ));
    let _ = fs::write(&path, &report);
    println!(
        "total: titanium {} - {opp_name} {}, draws {} — log: {}",
        tally.0,
        tally.1,
        tally.2,
        path.display()
    );
    ExitCode::SUCCESS
}

// ---------- show ----------

fn show_cmd() -> ExitCode {
    let b = Board::start();
    println!("{}", b.pretty());
    println!("engine string: \"{}\"", b.to_string());
    println!("ataxx fen:     \"{}\"", b.to_ataxx_fen());
    ExitCode::SUCCESS
}

// ---------- sperft ----------

/// Fixed deterministic benchmark suite: start position + mid-game playouts.
fn sperft_positions(n: usize) -> Vec<Board> {
    let mut rng = Rng(0x5DEECE66D);
    let mut suite = vec![Board::start()];
    for plies in [8u32, 14, 20, 26, 30, 34, 38] {
        suite.push(random_position(&mut rng, plies));
    }
    suite.truncate(n.max(1));
    suite
}

/// Search-throughput benchmark ("perft for the search"): fixed depth, no
/// time limit, deterministic position suite. Node counts are reproducible,
/// nps tracks search optimizations (TT, ordering, eval).
fn sperft_cmd(depth: u32, positions: usize, _ordering: Option<String>) -> ExitCode {
    let suite = sperft_positions(positions);
    println!(
        "Search benchmark: fixed depth {depth}, {} positions, 1M-entry TT\n",
        suite.len()
    );

    {
        let mut total_nodes = 0u64;
        let mut total_time = 0.0f64;
        for (i, b) in suite.iter().enumerate() {
            let mut searcher = Searcher::with_tt_bits(20);
            let limits = SearchLimits {
                time: None,
                max_nodes: None,
                max_depth: depth,
            };
            let t = Instant::now();
            let r = searcher.search(b, &limits);
            let dt = t.elapsed().as_secs_f64();
            total_nodes += r.nodes;
            total_time += dt;
            let _ = i;
        }
        println!(
            "  {:>12} nodes  {:>8.2} s  {:>8.2} Mnps",
            total_nodes,
            total_time,
            total_nodes as f64 / total_time / 1_000_000.0
        );
    }
    ExitCode::SUCCESS
}

// ---------- genbench ----------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Walk random playouts to reach varied mid-game positions.
fn random_position(rng: &mut Rng, plies: u32) -> Board {
    let mut b = Board::start();
    for _ in 0..plies {
        if b.game_over() {
            break;
        }
        let moves = b.legal_moves();
        if moves.is_empty() {
            b = b.make_pass();
            continue;
        }
        b = b.make(moves.move_at((rng.next() % moves.len() as u64) as usize));
    }
    b
}

fn genbench_cmd(iters: u64) -> ExitCode {
    let mut rng = Rng(0x243F6A8885A308D3);
    println!("Movegen benchmark, ~{iters} iterations per item\n");

    // Pre-generate a varied position pool (not part of the timed loops).
    let pool: Vec<Board> = (0..256).map(|_| random_position(&mut rng, 30)).collect();
    let pick = |i: u64| &pool[(i % 256) as usize];

    // 1) bulk target generation (the core bitboard op)
    let t = Instant::now();
    let mut acc = 0u64;
    for i in 0..iters {
        let b = pick(i);
        acc |= b.targets(0) | b.targets(1);
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "  targets x2          {:>10.0} /s   (acc {acc:x})",
        iters as f64 / dt
    );

    // 2) full legal move list fill
    let t = Instant::now();
    let mut total = 0u64;
    for i in 0..iters {
        total += pick(i).legal_moves().len() as u64;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "  legal_moves fill    {:>10.0} /s   (avg {:.1} moves)",
        iters as f64 / dt,
        total as f64 / iters as f64
    );

    // 3) make(): direct AND vs INFECT_LUT
    let b = pick(7);
    let moves = b.legal_moves();
    let n = iters.min(4_000_000);
    let t = Instant::now();
    let mut sink = 0u64;
    for i in 0..n {
        let m = moves.move_at((i % moves.len() as u64) as usize);
        let next = b.make(m);
        sink ^= next.occ[0] ^ next.occ[1];
    }
    let dt = t.elapsed().as_secs_f64();
    println!("  make (AND)          {:>10.0} /s   (sink {sink:x})", n as f64 / dt);

    let t = Instant::now();
    let mut sink2 = 0u64;
    for i in 0..n {
        let m = moves.move_at((i % moves.len() as u64) as usize);
        let next = b.make_via_lut(m);
        sink2 ^= next.occ[0] ^ next.occ[1];
    }
    let dt = t.elapsed().as_secs_f64();
    println!("  make_via_lut        {:>10.0} /s   (sink {sink2:x})", n as f64 / dt);

    // 4) extraction: portable loop vs BMI2 PEXT
    let opp = b.opp();
    let t = Instant::now();
    let mut k = 0u64;
    for i in 0..2_000_000u64 {
        k += titanium::extract8(opp, (i % 49) as u8) as u64;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("  extract8 (loop)     {:>10.0} /s   (k {k})", 2_000_000.0 / dt);

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("bmi2") {
            let t = Instant::now();
            let mut k = 0u64;
            for i in 0..2_000_000u64 {
                k += unsafe { titanium::extract8_pext(opp, (i % 49) as u8) } as u64;
            }
            let dt = t.elapsed().as_secs_f64();
            println!("  extract8 (PEXT)     {:>10.0} /s   (k {k})", 2_000_000.0 / dt);
        } else {
            println!("  extract8 (PEXT)     not supported on this CPU");
        }
    }

    // 5) perft throughput for reference (raw bitboard movegen)
    let b0 = Board::start();
    let t = Instant::now();
    let nodes = titanium::perft_bb(b0.occ[0], b0.occ[1], b0.blockers, 4);
    let dt = t.elapsed().as_secs_f64();
    println!(
        "  perft(4)            {:>10.0} knps  ({nodes} nodes)",
        nodes as f64 / dt / 1000.0
    );

    ExitCode::SUCCESS
}

#[cfg(test)]
mod serve_tests {
    use super::*;

    #[test]
    fn continuity_adoption_keeps_counters() {
        // A site that *loads a position* (FEN) instead of replaying moves
        // must not wipe memory: reach_board() adopts the make() child, so
        // ply/passes stay exact even when the FEN string flattened them.
        let b0 = Board::start();
        let b1 = b0.make(b0.legal_moves().move_at(0));
        let b2 = b1.make(b1.legal_moves().move_at(0));
        let target = Board::from_ataxx_fen(&b2.to_ataxx_fen()).expect("fen");
        assert_eq!(target.ply, 2, "fen must restore ply");
        // From two plies back: the "our move + opponent reply" case.
        let got = reach_board(&b0, &target, 2).expect("reachable in 2 plies");
        assert_eq!(got.ply, 2);
        assert_eq!(got.passes, 0);
        assert!(board_state_eq(&got, &target));
        // From one ply back: the plain opponent-reply case.
        let got1 = reach_board(&b1, &target, 2).expect("reachable in 1 ply");
        assert_eq!(got1.ply, 2);
        // Three plies ahead: outside the legal-move set -> caller rebuilds.
        let b3 = b2.make(b2.legal_moves().move_at(0));
        let t3 = Board::from_ataxx_fen(&b3.to_ataxx_fen()).expect("fen");
        assert!(reach_board(&b0, &t3, 2).is_none());
        // Identity: re-sending our own position adopts itself.
        assert!(reach_board(&b2, &target, 2).is_some());
    }

    #[test]
    fn continuity_pass_keeps_counters() {
        // Full board: side to move has no moves, so the only legal child is
        // a pass. The site sending the post-pass FEN must adopt make_pass()
        // and keep ply/passes from the chain (pass counts as a ply).
        let full = Board::from_ataxx_fen("xoxoxox/oxoxoxo/xoxoxox/oxoxoxo/xoxoxox/oxoxoxo/oxoxoxo x 0 1")
            .expect("fen");
        assert!(!full.has_moves(full.turn));
        let passed = full.make_pass();
        let target = Board::from_ataxx_fen(&passed.to_ataxx_fen()).expect("fen");
        let got = reach_board(&full, &target, 1).expect("reachable via pass");
        assert_eq!(got.ply, 1);
        assert_eq!(got.passes, 1);
        assert!(board_state_eq(&got, &target));
    }
}
