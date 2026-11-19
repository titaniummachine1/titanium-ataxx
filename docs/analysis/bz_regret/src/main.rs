use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::thread;

#[derive(Clone)]
struct Pos {
    idx: usize,
    gid: usize,
    ply: usize,
    fen: String,
    moves: String,
}

#[derive(Clone)]
struct SearchOut {
    mv: String,
    cp: i32,
    is_mate: bool,
    mate_val: i32,
    depth: u32,
    nodes: u64,
    time_ms: u64,
}

#[derive(Clone)]
struct RefRow {
    pos: Pos,
    ref_out: SearchOut,
}

fn flag_str(args: &[String], k: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == k).map(|w| w[1].clone())
}

fn flag_usize(args: &[String], k: &str) -> Option<usize> {
    flag_str(args, k).and_then(|v| v.parse().ok())
}

fn has_flag(args: &[String], k: &str) -> bool {
    args.iter().any(|x| x == k)
}

fn load_positions(path: &str) -> Vec<Pos> {
    let f = File::open(path).expect("open pos file");
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(f).lines().enumerate() {
        let l = line.unwrap();
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = l.split('\t').collect();
        out.push(Pos {
            idx,
            gid: parts[0].parse().unwrap(),
            ply: parts[1].parse().unwrap(),
            fen: parts[2].to_string(),
            moves: parts.get(3).copied().unwrap_or("").to_string(),
        });
    }
    out
}

fn parse_info_line(line: &str, cur: &mut SearchOut) {
    let tok: Vec<&str> = line.split_whitespace().collect();
    let mut i = 1;
    while i < tok.len() {
        match tok[i] {
            "depth" if i + 1 < tok.len() => {
                cur.depth = tok[i + 1].parse().unwrap_or(cur.depth);
                i += 2;
            }
            "score" if i + 2 < tok.len() => {
                if tok[i + 1] == "cp" {
                    cur.is_mate = false;
                    cur.cp = tok[i + 2].parse().unwrap_or(cur.cp);
                } else if tok[i + 1] == "mate" {
                    cur.is_mate = true;
                    cur.mate_val = tok[i + 2].parse().unwrap_or(0);
                    cur.cp = if cur.mate_val > 0 {
                        10000 - cur.mate_val
                    } else {
                        -10000 - cur.mate_val
                    };
                }
                i += 3;
            }
            "nodes" if i + 1 < tok.len() => {
                cur.nodes = tok[i + 1].parse().unwrap_or(cur.nodes);
                i += 2;
            }
            "time" if i + 1 < tok.len() => {
                cur.time_ms = tok[i + 1].parse().unwrap_or(cur.time_ms);
                i += 2;
            }
            _ => i += 1,
        }
    }
}

fn run_uai_batch(
    exe: &str,
    serve_args: &[String],
    cmds: &[String],
) -> Vec<SearchOut> {
    let mut cmd = Command::new(exe);
    cmd.arg("serve");
    for a in serve_args {
        cmd.arg(a);
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().expect("spawn engine");
    {
        let mut stdin = child.stdin.take().unwrap();
        for c in cmds {
            writeln!(stdin, "{c}").unwrap();
        }
        writeln!(stdin, "quit").unwrap();
    }
    let stdout = child.stdout.take().unwrap();
    let mut results = Vec::new();
    let mut cur = SearchOut {
        mv: String::new(),
        cp: 0,
        is_mate: false,
        mate_val: 0,
        depth: 0,
        nodes: 0,
        time_ms: 0,
    };
    for line in BufReader::new(stdout).lines() {
        let l = line.unwrap();
        if l.starts_with("info ") {
            parse_info_line(&l, &mut cur);
        } else if let Some(rest) = l.strip_prefix("bestmove ") {
            cur.mv = rest.split_whitespace().next().unwrap_or("").to_string();
            results.push(cur.clone());
            cur = SearchOut {
                mv: String::new(),
                cp: 0,
                is_mate: false,
                mate_val: 0,
                depth: 0,
                nodes: 0,
                time_ms: 0,
            };
        }
    }
    let _ = child.wait();
    results
}

fn pos_cmd(p: &Pos, extra_mv: Option<&str>) -> String {
    match (p.moves.is_empty(), extra_mv) {
        (true, None) => format!("position fen {}", p.fen),
        (false, None) => format!("position fen {} moves {}", p.fen, p.moves),
        (true, Some(m)) => format!("position fen {} moves {m}", p.fen),
        (false, Some(m)) => format!("position fen {} moves {} {m}", p.fen, p.moves),
    }
}

fn eval_positions_parallel(
    exe: &str,
    serve_args: &[String],
    positions: &[Pos],
    go_cmd: &str,
    threads: usize,
    warm: bool,
) -> Vec<SearchOut> {
    let mut by_game: Vec<Vec<usize>> = Vec::new();
    let mut gmap: HashMap<usize, usize> = HashMap::new();
    for (i, p) in positions.iter().enumerate() {
        let g = *gmap.entry(p.gid).or_insert_with(|| {
            by_game.push(Vec::new());
            by_game.len() - 1
        });
        by_game[g].push(i);
    }
    let mut chunks: Vec<Vec<usize>> = vec![Vec::new(); threads];
    for (gi, g) in by_game.iter().enumerate() {
        chunks[gi % threads].extend(g.iter().copied());
    }
    let out_vec: Mutex<Vec<Option<SearchOut>>> = Mutex::new(vec![None; positions.len()]);
    thread::scope(|s| {
        for chunk in chunks {
            let out_ref = &out_vec;
            s.spawn(move || {
                if chunk.is_empty() {
                    return;
                }
                let mut cmds = Vec::new();
                let mut last_gid = usize::MAX;
                for &i in &chunk {
                    let p = &positions[i];
                    if !warm || p.gid != last_gid {
                        cmds.push("uainewgame".to_string());
                        last_gid = p.gid;
                    }
                    cmds.push(pos_cmd(p, None));
                    cmds.push(go_cmd.to_string());
                }
                let res = run_uai_batch(exe, serve_args, &cmds);
                assert_eq!(
                    res.len(),
                    chunk.len(),
                    "engine returned {} results for {} positions",
                    res.len(),
                    chunk.len()
                );
                let mut guard = out_ref.lock().unwrap();
                for (k, &i) in chunk.iter().enumerate() {
                    guard[i] = Some(res[k].clone());
                }
            });
        }
    });
    out_vec.into_inner().unwrap().into_iter().map(|x| x.unwrap()).collect()
}

fn load_ref(path: &str) -> Vec<RefRow> {
    let f = File::open(path).expect("open ref");
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let l = line.unwrap();
        if l.trim().is_empty() || l.starts_with('#') {
            continue;
        }
        let p: Vec<&str> = l.split('\t').collect();
        out.push(RefRow {
            pos: Pos {
                idx: p[0].parse().unwrap(),
                gid: p[1].parse().unwrap(),
                ply: p[2].parse().unwrap(),
                fen: p[3].to_string(),
                moves: p[4].to_string(),
            },
            ref_out: SearchOut {
                mv: p[5].to_string(),
                cp: p[6].parse().unwrap(),
                is_mate: p[7] == "1",
                mate_val: p[8].parse().unwrap(),
                depth: p[9].parse().unwrap(),
                nodes: p[10].parse().unwrap(),
                time_ms: p[11].parse().unwrap(),
            },
        });
    }
    out
}

fn load_score_cache(path: &str) -> HashMap<(usize, String), (i32, bool, i32)> {
    let mut map = HashMap::new();
    if let Ok(f) = File::open(path) {
        for line in BufReader::new(f).lines() {
            if let Ok(l) = line {
                let p: Vec<&str> = l.split('\t').collect();
                if p.len() >= 5 {
                    if let (Ok(idx), Ok(cp), Ok(mv)) = (
                        p[0].parse::<usize>(),
                        p[2].parse::<i32>(),
                        p[4].parse::<i32>(),
                    ) {
                        map.insert((idx, p[1].to_string()), (cp, p[3] == "1", mv));
                    }
                }
            }
        }
    }
    map
}

fn append_score_cache(path: &str, entries: &[(usize, String, i32, bool, i32)]) {
    if entries.is_empty() {
        return;
    }
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open cache append");
    for (idx, mv, cp, is_m, mval) in entries {
        writeln!(f, "{idx}\t{mv}\t{cp}\t{}\t{mval}", if *is_m { 1 } else { 0 }).unwrap();
    }
}

fn child_to_parent_cp(child: &(i32, bool, i32)) -> i32 {
    // Child score is from opponent perspective; negate for parent STM perspective.
    if child.1 {
        let p_cp = if child.0 > 0 { -2000 } else { 2000 };
        p_cp
    } else {
        (-child.0).clamp(-2000, 2000)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let serve_args: Vec<String> = flag_str(&args, "--args")
        .unwrap_or_else(|| "--bz --tt-bits 22".to_string())
        .split_whitespace()
        .map(String::from)
        .collect();
    if cmd == "gen-ref" {
        let pos_path = flag_str(&args, "--pos").unwrap();
        let exe = flag_str(&args, "--exe").unwrap();
        let nodes = flag_usize(&args, "--nodes").unwrap_or(150_000);
        let threads = flag_usize(&args, "--threads").unwrap_or(4);
        let out_path = flag_str(&args, "--out").unwrap();
        let positions = load_positions(&pos_path);
        let go_cmd = format!("go nodes {nodes}");
        let res = eval_positions_parallel(&exe, &serve_args, &positions, &go_cmd, threads, true);
        let mut f = File::create(&out_path).unwrap();
        writeln!(f, "# ref exe={exe} nodes={nodes} args={}", serve_args.join(" ")).unwrap();
        for (p, r) in positions.iter().zip(res.iter()) {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                p.idx,
                p.gid,
                p.ply,
                p.fen,
                p.moves,
                r.mv,
                r.cp,
                if r.is_mate { 1 } else { 0 },
                r.mate_val,
                r.depth,
                r.nodes,
                r.time_ms
            )
            .unwrap();
        }
        println!("Generated ref ({} pos) -> {out_path}", positions.len());
    } else if cmd == "eval" {
        let ref_path = flag_str(&args, "--ref").unwrap();
        let ref_exe = flag_str(&args, "--ref-exe").unwrap();
        let ref_nodes = flag_usize(&args, "--ref-nodes").unwrap_or(80_000);
        let cache_path = flag_str(&args, "--cache").unwrap();
        let cand_exe = flag_str(&args, "--exe").unwrap();
        let label = flag_str(&args, "--label").unwrap_or_else(|| cand_exe.clone());
        let threads = flag_usize(&args, "--threads").unwrap_or(1);
        let warm = !has_flag(&args, "--cold");
        let go_cmd = if let Some(ms) = flag_usize(&args, "--movetime") {
            format!("go movetime {ms}")
        } else if let Some(d) = flag_usize(&args, "--depth") {
            format!("go depth {d}")
        } else {
            let n = flag_usize(&args, "--nodes").unwrap_or(15_000);
            format!("go nodes {n}")
        };

        let refs = load_ref(&ref_path);
        let positions: Vec<Pos> = refs.iter().map(|r| r.pos.clone()).collect();
        let outs = eval_positions_parallel(&cand_exe, &serve_args, &positions, &go_cmd, threads, warm);

        // Identify missing (pos_idx, move) child scores for differences.
        let mut cache = load_score_cache(&cache_path);
        let mut todo: Vec<(usize, String, Pos)> = Vec::new();
        for (r, o) in refs.iter().zip(outs.iter()) {
            if o.mv != r.ref_out.mv {
                for mv in [&r.ref_out.mv, &o.mv] {
                    if !mv.is_empty() && !cache.contains_key(&(r.pos.idx, mv.clone())) {
                        todo.push((r.pos.idx, mv.clone(), r.pos.clone()));
                    }
                }
            }
        }
        // Dedup todo.
        todo.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        todo.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        if !todo.is_empty() {
            let score_threads = 4usize;
            let mut chunks: Vec<Vec<(usize, String, Pos)>> = vec![Vec::new(); score_threads];
            for (k, item) in todo.into_iter().enumerate() {
                chunks[k % score_threads].push(item);
            }
            let scored: Mutex<Vec<(usize, String, i32, bool, i32)>> = Mutex::new(Vec::new());
            let score_go = format!("go nodes {ref_nodes}");
            let ref_args = vec!["--bz".to_string(), "--tt-bits".to_string(), "22".to_string()];
            let rex = ref_exe.as_str();
            thread::scope(|s| {
                for ch in chunks {
                    let scored_ref = &scored;
                    let sg = &score_go;
                    let ra = &ref_args;
                    s.spawn(move || {
                        if ch.is_empty() {
                            return;
                        }
                        let mut cmds = Vec::new();
                        for (_, mv, p) in &ch {
                            cmds.push("uainewgame".to_string());
                            cmds.push(pos_cmd(p, Some(mv)));
                            cmds.push(sg.clone());
                        }
                        let res = run_uai_batch(rex, ra, &cmds);
                        let mut g = scored_ref.lock().unwrap();
                        for ((idx, mv, _), r) in ch.into_iter().zip(res) {
                            g.push((idx, mv, r.cp, r.is_mate, r.mate_val));
                        }
                    });
                }
            });
            let new_entries = scored.into_inner().unwrap();
            append_score_cache(&cache_path, &new_entries);
            for (idx, mv, cp, is_m, mval) in new_entries {
                cache.insert((idx, mv), (cp, is_m, mval));
            }
        }

        // Summarize overall and by phase: open <= 26, mid 36..56, end >= 66
        let mut total_n = 0u64;
        let mut total_ms = 0u64;
        let mut sum_depth = 0.0f64;
        let mut log_nodes = 0.0f64;
        let mut agree = 0usize;
        let mut diffs = 0usize;
        let mut sum_regret = 0.0f64;
        let mut sum_signed = 0.0f64;
        let mut blunders = 0usize;
        let mut p_reg = [0.0f64; 3];
        let mut p_cnt = [0usize; 3];
        let mut p_agr = [0usize; 3];
        let mut p_dep = [0.0f64; 3];
        let mut p_nod = [0.0f64; 3];
        for (r, o) in refs.iter().zip(outs.iter()) {
            total_n += o.nodes;
            total_ms += o.time_ms.max(1);
            sum_depth += o.depth as f64;
            log_nodes += (o.nodes.max(1) as f64).ln();
            let ph = if r.pos.ply <= 26 { 0 } else if r.pos.ply <= 56 { 1 } else { 2 };
            p_cnt[ph] += 1;
            p_dep[ph] += o.depth as f64;
            p_nod[ph] += (o.nodes.max(1) as f64).ln();
            if o.mv == r.ref_out.mv {
                agree += 1;
                p_agr[ph] += 1;
            } else {
                diffs += 1;
                let s_ref = child_to_parent_cp(&cache[&(r.pos.idx, r.ref_out.mv.clone())]);
                let s_cand = child_to_parent_cp(&cache[&(r.pos.idx, o.mv.clone())]);
                let loss = (s_ref - s_cand).max(0) as f64;
                let signed = (s_ref - s_cand) as f64;
                sum_regret += loss;
                sum_signed += signed;
                if loss >= 150.0 {
                    blunders += 1;
                }
                p_reg[ph] += loss;
            }
        }
        let n = refs.len() as f64;
        let knps = total_n as f64 / total_ms as f64;
        println!(
            "{:<18} | go={:<15} | agree {:>5.1}% ({}/{}) | regret {:>5.2} cp (signed {:>+5.2}, diff_only {:>5.1}) | blnd>=150 {:>2} | depth {:>5.2} | geoN {:>7.0} | agg {:>5.1} knps",
            label,
            go_cmd,
            100.0 * agree as f64 / n,
            agree,
            refs.len(),
            sum_regret / n,
            sum_signed / n,
            if diffs > 0 { sum_regret / diffs as f64 } else { 0.0 },
            blunders,
            sum_depth / n,
            (log_nodes / n).exp(),
            knps
        );
        let pnames = ["open(<=26)", "mid(36-56)", "end(>=66) "];
        for ph in 0..3 {
            let c = p_cnt[ph] as f64;
            println!(
                "   {}: agr {:>5.1}% reg {:>5.2} cp depth {:>5.2} geoN {:>7.0} (n={})",
                pnames[ph],
                100.0 * p_agr[ph] as f64 / c,
                p_reg[ph] / c,
                p_dep[ph] / c,
                (p_nod[ph] / c).exp(),
                p_cnt[ph]
            );
        }
    } else {
        eprintln!("usage: bz_regret gen-ref | eval ...");
        std::process::exit(1);
    }
}
