//! Titanium CLI - native testing harness for the engine.
//!
//! Commands:
//!   perft [depth]                      movegen node counts (mirror cross-checked)
//!   bench [--depth N]                  search speed from the start position
//!   selfplay [--games N] [--time MS]   engine vs engine, logs to logs/
//!   bestmove ["<49 chars> b"] [--time MS]
//!   play [--time MS]                   console game vs the engine
//!   show                               print the start position

use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use titanium::{best_move, Board, Move, SearchLimits, Searcher, RING1, StopReason};
use titanium::{MASK_ALL, MASK_CONTACT, MASK_HOLES, MASK_MATERIAL, MASK_MULTICAP, MASK_PST, MASK_TEMPO};

/// Parse an eval ablation mask: "all" (default) or comma list of inputs to
/// REMOVE, e.g. `--opp-mask noholes` / `--opp-mask nopst,notempo`.
fn parse_mask(s: Option<String>) -> u32 {
    let Some(s) = s else { return MASK_ALL };
    if s.trim().eq_ignore_ascii_case("all") {
        return MASK_ALL;
    }
    let mut mask = MASK_ALL;
    for tok in s.split(',') {
        match tok.trim().to_ascii_lowercase().as_str() {
            "nomaterial" | "nomat" => mask &= !MASK_MATERIAL,
            "nopst" => mask &= !MASK_PST,
            "noholes" | "nohole" => mask &= !MASK_HOLES,
            "nocontact" => mask &= !MASK_CONTACT,
            "nomulticap" | "nomulti" => mask &= !MASK_MULTICAP,
            "notempo" => mask &= !MASK_TEMPO,
            other => eprintln!("match: unknown mask token '{other}' (use nomaterial,nopst,noholes,nocontact,nomulticap,notempo)"),
        }
    }
    mask
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "perft" => perft_cmd(first_int(&args).unwrap_or(4) as u32),
        "bench" => bench_cmd(flag_int(&args, "--depth").unwrap_or(8) as u32),
        "selfplay" => selfplay_cmd(
            flag_int(&args, "--games").unwrap_or(2) as u32,
            flag_int(&args, "--time").unwrap_or(300) as u64,
            flag_str(&args, "--out"),
        ),
        "bestmove" => bestmove_cmd(
            positional_str(&args).or_else(|| Some(Board::start().to_string())),
            flag_int(&args, "--time").map(|ms| ms as u64),
        ),
        "play" => play_cmd(flag_int(&args, "--time").unwrap_or(1000) as u64),
        "show" => show_cmd(),
        "genbench" => genbench_cmd(flag_int(&args, "--iters").unwrap_or(200_000) as u64),
        "datagen" => datagen_cmd(
            flag_int(&args, "--games").unwrap_or(100) as u32,
            flag_int(&args, "--nodes").unwrap_or(5000) as u64,
            flag_int(&args, "--seed").unwrap_or(1) as u64,
            flag_int(&args, "--opening-plies").unwrap_or(8) as u32,
            flag_str(&args, "--out").unwrap_or_else(|| "data/nnue/datagen.txt".into()),
        ),
        "serve" => serve_cmd(flag_int(&args, "--tt-bits").unwrap_or(20) as usize),
        "match" => match_cmd(
            flag_int(&args, "--games").unwrap_or(20) as u32,
            flag_int(&args, "--time").unwrap_or(1000) as u64,
            flag_str(&args, "--opp"),
            flag_int(&args, "--opp-time").unwrap_or(1000) as u64,
            flag_int(&args, "--opp-depth").map(|d| d as u32),
            flag_int(&args, "--opp-nodes").map(|n| n as u64),
            flag_int(&args, "--nodes").map(|n| n as u64),
            flag_int(&args, "--start-game").unwrap_or(1) as u32,
            flag_str(&args, "--out"),
            parse_mask(flag_str(&args, "--mask")),
            parse_mask(flag_str(&args, "--opp-mask")),
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
         USAGE:\n\
         \x20 titanium-cli perft [depth]                    movegen node counts\n\
         \x20 titanium-cli bench [--depth N]                search speed benchmark\n\
         \x20 titanium-cli selfplay [--games N] [--time MS] engine vs engine into logs/\n\
         \x20 titanium-cli bestmove [\"<49 chars> b\"] [--time MS]\n\
         \x20 titanium-cli play [--time MS]                 console game vs engine\n\
         \x20 titanium-cli serve [--time MS]                  line-protocol engine (one board per line)\n\
         \x20 titanium-cli match --games N --time MS --opp \"CMD\"\n\
         \x20                                                 UAI opponent, colors alternate\n\
         \x20                                 [--nodes N] [--opp-time MS] [--opp-depth N]\n\
         \x20 titanium-cli sperft [--depth N] [--games N]     search throughput suite\n\
         \x20 titanium-cli show                             show start position\n\
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

fn perft(b: &Board, depth: u32, stack: &mut [titanium::MoveList], ply: usize) -> u64 {
    if depth == 0 {
        return 1;
    }
    b.legal_moves_into(&mut stack[ply]);
    let n = stack[ply].len();
    if depth == 1 {
        return n as u64;
    }
    let mut total = 0u64;
    for i in 0..n {
        let m = stack[ply].move_at(i);
        total += perft(&b.make(m), depth - 1, stack, ply + 1);
    }
    total
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

fn bench_cmd(depth: u32) -> ExitCode {
    let b = Board::start();
    println!("Bench: start position, fixed depth {depth} (no time limit)");
    let limits = SearchLimits {
        time: None,
        max_nodes: None,
        max_depth: depth,
    };
    let r = best_move(&b, &limits);
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

fn serve_cmd(tt_bits: usize) -> ExitCode {
    let stdin = io::stdin();
    let mut searcher = Searcher::with_tt_bits(tt_bits);
    let mut current: Option<Board> = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let t = line.trim().to_string();
        if t.is_empty() {
            continue;
        }
        if t == "quit" {
            break;
        }
        if t == "uai" {
            out_line("id name Titanium Ataxx");
            out_line("id author titaniummachine1");
            out_line("uaiok");
        } else if t == "isready" {
            out_line("readyok");
        } else if t == "uainewgame" {
            searcher.reset_for_new_game();
        } else if t == "position startpos" {
            current = Some(Board::start());
        } else if let Some(fen) = t.strip_prefix("position fen ") {
            current = Board::from_ataxx_fen(fen.trim());
        } else if t.starts_with("go") {
            let mut nodes = None;
            let mut movetime = None;
            let mut depth = None;
            let toks: Vec<&str> = t.split_whitespace().collect();
            for (i, k) in toks.iter().enumerate() {
                match *k {
                    "nodes" => nodes = toks.get(i + 1).and_then(|v| v.parse().ok()),
                    "movetime" => movetime = toks.get(i + 1).and_then(|v| v.parse().ok()),
                    "depth" => depth = toks.get(i + 1).and_then(|v| v.parse().ok()),
                    _ => {}
                }
            }
            let reply = current
                .as_ref()
                .map(|b| {
                    if b.game_over() || !b.has_moves(b.turn) {
                        return "0000".to_string();
                    }
                    let limits = SearchLimits {
                        time: movetime.filter(|ms| *ms > 0).map(Duration::from_millis),
                        max_nodes: nodes,
                        max_depth: depth.unwrap_or(24),
                    };
                    match searcher.search(b, &limits).best {
                        Some(m) => {
                            if m.is_clone() {
                                Board::sq_name(m.to)
                            } else {
                                format!("{}{}", Board::sq_name(m.from), Board::sq_name(m.to))
                            }
                        }
                        None => "0000".to_string(),
                    }
                })
                .unwrap_or_else(|| "0000".to_string());
            out_line(&format!("bestmove {reply}"));
        }
        // "stop" ignored: searches are synchronous.
    }
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
    opp_time: u64,
    opp_depth: Option<u32>,
    opp_nodes: Option<u64>,
    max_nodes: Option<u64>,
    start_game: u32,
    out: Option<String>,
    mask: u32,
    opp_mask: u32,
) -> ExitCode {
    // "self" = titanium vs titanium in-process (no external process).
    let mut opp = if opp_cmd.as_deref() == Some("self") {
        None
    } else {
        let cmd = match opp_cmd {
            Some(c) => c,
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
    println!("titanium vs {opp_name} â€” {games} games");

    let mut tally = (0u32, 0u32, 0u32); // titanium wins, opp wins, draws
    let limits_desc = format!(
        "titanium limits: {}",
        match max_nodes {
            Some(n) => format!("{n} nodes"),
            None => format!("{time_ms} ms"),
        }
    );
    let mut report = String::new();
    report.push_str(&format!(
        "Match: titanium vs {opp_name} â€” games {start_game}..{} â€” {limits_desc}, opp {}\n===\n",
        start_game + games - 1,
        opp_depth
            .map(|d| format!("depth {d}"))
            .unwrap_or_else(|| format!("movetime {opp_time} ms"))
    ));

    // Incremental log: header first, then append per game â€” played games
    // are never lost, even if the run is killed.
    let path: PathBuf = out.map(PathBuf::from).unwrap_or_else(default_log_path);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&path, &report);

    for game in start_game..start_game + games {
        let mut b = Board::start();
        // Seed-varied LEGAL book: game number picks which outward clone each
        // side plays (3x3 corner options), so node-limited deterministic
        // search yields distinct games without early wipes.
        {
            let mut seed = game as u64 * 0x9E3779B97F4A7C15 + 0x243F6A8885A308D3;
            let black_opts = ["b2", "a2", "b1"];
            let white_opts = ["f6", "g6", "f7"];
            for i in 0..10 {
                let opts = if b.turn == 0 { black_opts } else { white_opts };
                let sq = opts[(rng_next(&mut seed) % 3) as usize];
                let to = Board::parse_sq(sq).unwrap();
                let clones = b.occ[b.turn as usize] & RING1[to as usize];
                if clones == 0 || !b.has_moves(b.turn) || b.game_over() {
                    break;
                }
                b = b.make(Move { from: clones.trailing_zeros() as u8, to });
                let _ = i;
            }
        }
        let mut searcher = Searcher::new(); // fresh TT per game
        searcher.set_eval_mask(mask);
        let mut opp_searcher = Searcher::new(); // for --opp self
        opp_searcher.set_eval_mask(opp_mask);
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
        "total: titanium {} - {opp_name} {}, draws {} â€” log: {}",
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
fn sperft_cmd(depth: u32, positions: usize, ordering: Option<String>) -> ExitCode {
    let orderings: Vec<(titanium::Ordering, &str)> = match ordering.as_deref() {
        Some("lazy") => vec![(titanium::Ordering::Lazy, "lazy")],
        Some("insertion") => vec![(titanium::Ordering::Insertion, "insertion")],
        Some("radix") => vec![(titanium::Ordering::Radix, "radix")],
        _ => vec![
            (titanium::Ordering::Lazy, "lazy"),
            (titanium::Ordering::Insertion, "insertion"),
            (titanium::Ordering::Radix, "radix"),
        ],
    };
    let suite = sperft_positions(positions);
    println!(
        "Search benchmark: fixed depth {depth}, {} positions, 1M-entry TT\n",
        suite.len()
    );

    for (ord, name) in &orderings {
        let mut total_nodes = 0u64;
        let mut total_time = 0.0f64;
        for (i, b) in suite.iter().enumerate() {
            let mut searcher = Searcher::with_tt_bits(20);
            searcher.set_ordering(*ord);
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
            "  {name:<10} {:>12} nodes  {:>8.2} s  {:>8.2} Mnps",
            total_nodes,
            total_time,
            total_nodes as f64 / total_time / 1_000_000.0
        );
    }
    ExitCode::SUCCESS
}

// ---------- genbench ----------

struct Rng(u64);

#[allow(dead_code)]
fn rng_next(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

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

fn datagen_cmd(
    games: u32,
    nodes_per_move: u64,
    seed: u64,
    opening_plies: u32,
    out_path: String,
) -> ExitCode {
    use std::io::BufWriter;

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .expect("open datagen output");
    let mut out = BufWriter::new(file);

    let mut rng: u64 = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut next = move || -> u64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let limits = SearchLimits {
        time: None,
        max_nodes: Some(nodes_per_move),
        max_depth: 20,
    };

    let mut total_positions = 0u64;
    let mut wins = [0u32; 2];
    let mut draws = 0u32;
    let started = Instant::now();

    for game in 1..=games {
        let mut b = Board::start();
        let mut searcher = Searcher::with_tt_bits(16);
        let mut opening = 0u32;
        while opening < opening_plies && !b.game_over() {
            let mut list = b.legal_moves();
            if list.is_empty() {
                b = b.make_pass();
            } else {
                let idx = (next() % list.len() as u64) as usize;
                let mv = list.move_at(idx);
                b = b.make(mv);
            }
            opening += 1;
        }
        if b.game_over() {
            continue;
        }

        let mut records: Vec<(String, i32, u8, [i32; 8])> = Vec::with_capacity(96);
        let mut hot_plies = 0u32;
        let mut quiet_plies = 0u32;
        let mut adjudicated: Option<u8> = None;
        let mut ply = 0u32;

        loop {
            if b.game_over() {
                break;
            }
            if ply >= 300 {
                adjudicated = Some(2);
                break;
            }
            let r = searcher.search(&b, &limits);
            // Dense NNUE features, stm-relative: weakness of side to move,
            // then weakness of enemy. Net learns the weights.
            let (sc, sca, mc, en) = titanium::weakness_features(&b, b.turn);
            let (oc, oca, omc, oen) = titanium::weakness_features(&b, 1 - b.turn);
            records.push((
                b.to_ataxx_fen(),
                r.score.clamp(-2000, 2000),
                b.turn,
                [sc, sca, mc, en as i32, oc, oca, omc, oen as i32],
            ));
            let s = r.score;
            if s.abs() > 2500 {
                quiet_plies = 0;
                hot_plies += 1;
                if hot_plies >= 5 {
                    adjudicated = Some(if s > 0 { b.turn } else { 1 - b.turn });
                    break;
                }
            } else {
                hot_plies = 0;
                quiet_plies += 1;
                if quiet_plies >= 10 && ply >= 60 {
                    adjudicated = Some(2);
                    break;
                }
            }
            let mv = match r.best {
                Some(m) => m,
                None => Move::PASS,
            };
            b = if mv.is_pass() { b.make_pass() } else { b.make(mv) };
            ply += 1;
        }

        let result = match adjudicated {
            Some(0) => 0,
            Some(1) => 1,
            _ => b.winner(),
        };
        match result {
            0 => wins[0] += 1,
            1 => wins[1] += 1,
            _ => draws += 1,
        }
        for (fen, score, turn, w) in &records {
            let res = match (result, *turn) {
                (2, _) => 0.5,
                (w, side) => {
                    if w == side {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            let _ = writeln!(
                out,
                "{fen}|{score}|{res:.1}|{},{},{},{},{},{},{},{}",
                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]
            );
            total_positions += 1;
        }
        out.flush().ok();

        if game % 10 == 0 {
            println!(
                "datagen {seed}: game {game}/{games} pos {total_positions} W{}/L{}/D{} ({:.0}s)",
                wins[0],
                wins[1],
                draws,
                started.elapsed().as_secs_f32()
            );
            out.flush().ok();
        }
    }
    let _ = out.flush();
    println!(
        "datagen {seed} done: {games} games, {total_positions} positions -> {out_path}"
    );
    ExitCode::SUCCESS
}
