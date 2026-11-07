//! exp/tree-shape (D547/D549/D552): `titanium-cli treediff` (build with `--features trace`).
//! For positions from gate games (every --every plies), per game phase (16 phases by empties):
//!  1. deep search (--dnodes, default 500k): best move, score, PV (walked through the TT) = the reference;
//!  2. shallow search to the deepest depth that completes within --snodes (default 50k), rerun with that depth limit
//!     and the trace on, so the whole final iteration is in memory;
//!  3. the deep PV is walked through the shallow tree: at every ply, what the shallow tree did to the deep PV move
//!     (searched at full depth / reduced by r plies / skipped by the small net or futility / never reached because a
//!     sibling cut first / the node itself was cut by RFP, null move, ProbCut, TT) = OVER-pruning evidence;
//!  4. whole-tree statistics: per phase and node kind, how many nodes and how many subtree nodes each kind costs; failed
//!     null-move and ProbCut searches (nodes spent without a cut); RFP near misses = non-PV nodes at depth <= 4 that
//!     failed low anyway, by how far RFP missed = UNDER-pruning evidence.
//! Output: --out PREFIX writes PREFIX.pos.tsv (one row per position), PREFIX.pv.tsv (one row per deep-PV ply),
//! PREFIX.nodes.tsv (aggregate per phase x kind); --dump N also writes the first N shallow trees as PREFIX.treeK.tsv.
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::process::ExitCode;
use titanium::trace::{kind, via, Rec, Tracer};
use titanium::{Board, Move, SearchLimits, Searcher};

pub fn phase_of(empties: u32) -> usize {
    (45i32 - empties as i32).clamp(0, 47) as usize / 3
}

fn json_str<'a>(s: &'a str, key: &str, from: usize) -> Option<(&'a str, usize)> {
    let pat = format!("\"{key}\":\"");
    let i = s[from..].find(&pat)? + from + pat.len();
    let j = s[i..].find('"')? + i;
    Some((&s[i..j], j))
}

/// (start fen, moves) of every game line
fn read_games(path: &str, max: usize) -> Vec<(String, Vec<String>)> {
    let Ok(text) = fs::read_to_string(path) else { return Vec::new() };
    let mut out = Vec::new();
    for line in text.lines().take(max) {
        let Some((start, _)) = json_str(line, "start", 0) else { continue };
        let mut mv = Vec::new();
        let mut at = line.find("\"moves\"").unwrap_or(0);
        while let Some((m, j)) = json_str(line, "move", at) {
            mv.push(m.to_string());
            at = j;
        }
        out.push((start.to_string(), mv));
    }
    out
}

fn mname(m: Move) -> String {
    if m.is_pass() {
        "pass".into()
    } else if m.is_clone() {
        Board::sq_name(m.to)
    } else {
        format!("{}{}", Board::sq_name(m.from), Board::sq_name(m.to))
    }
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

const RFP_MARGINS: [i32; 4] = [257, 347, 478, 774]; // MUST match search.rs
const DIS_RFP_NUM: i32 = 32;

#[derive(Default, Clone)]
struct Agg {
    n: u64,
    sub: u64,
    /// knob buckets: cuts (nmp/pc), fail-lows (rfp gap), alpha raised (move index)
    hit: u64,
}

fn sbucket(d: i32) -> &'static str {
    if d < -200 {
        "a<-200"
    } else if d < -100 {
        "b-200..-100"
    } else if d < 0 {
        "c-100..0"
    } else if d < 100 {
        "d0..100"
    } else if d < 200 {
        "e100..200"
    } else if d < 300 {
        "f200..300"
    } else {
        "g>=300"
    }
}

fn ibucket(i: u8) -> &'static str {
    match i {
        0 => "00",
        1 => "01",
        2 => "02",
        3 => "03",
        4..=5 => "04-05",
        6..=8 => "06-08",
        9..=12 => "09-12",
        13..=20 => "13-20",
        _ => "21+",
    }
}

pub fn run(args: &[String]) -> ExitCode {
    if let Some(p) = arg(args, "--report") {
        return report(&p);
    }
    let Some(games) = arg(args, "--games") else {
        eprintln!("treediff --games G.jsonl [--max-games 40] [--every 4] [--snodes 50000] [--dnodes 500000] [--out PREFIX] [--dump N] [--per-phase 12]");
        return ExitCode::from(2);
    };
    let max_games: usize = arg(args, "--max-games").and_then(|v| v.parse().ok()).unwrap_or(40);
    let every: usize = arg(args, "--every").and_then(|v| v.parse().ok()).unwrap_or(4);
    let snodes: u64 = arg(args, "--snodes").and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let dnodes: u64 = arg(args, "--dnodes").and_then(|v| v.parse().ok()).unwrap_or(500_000);
    let per_phase: usize = arg(args, "--per-phase").and_then(|v| v.parse().ok()).unwrap_or(12);
    let dump: usize = arg(args, "--dump").and_then(|v| v.parse().ok()).unwrap_or(0);
    if let Some(p) = arg(args, "--report") {
        return report(&p);
    }
    // --shard K/N: this process takes positions with index % N == K (run N processes in parallel), files PREFIX.sK.*
    let (shard_k, shard_n) = arg(args, "--shard")
        .and_then(|v| v.split_once('/').and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))))
        .unwrap_or((0, 1));
    let mut prefix = arg(args, "--out").unwrap_or_else(|| "treediff".into());
    if shard_n > 1 {
        prefix = format!("{prefix}.s{shard_k}");
    }
    let nn = Some(titanium::nnue::Nnue::bundled());

    // positions, at most per_phase per phase, spread over the games
    let mut pos: Vec<(Board, usize)> = Vec::new();
    let mut cnt = [0usize; 16];
    for (start, mv) in read_games(&games, max_games) {
        let Some(mut b) = Board::from_ataxx_fen(&start) else { continue };
        let mut taken = [false; 16]; // at most one position per phase per game: spread over games
        for (k, m) in mv.iter().enumerate() {
            if k > 0 && k % every == 0 && !b.game_over() {
                let ph = phase_of(b.empty().count_ones());
                if cnt[ph] < per_phase && !taken[ph] {
                    taken[ph] = true;
                    cnt[ph] += 1;
                    pos.push((b, k));
                }
            }
            let Some(mm) = b.parse_uai_move(m) else { break };
            b = if mm.is_pass() { b.make_pass() } else { b.make(mm) };
        }
    }
    eprintln!("treediff: {} positions, per phase {:?}", pos.len(), cnt);

    let mut fpos = fs::File::create(format!("{prefix}.pos.tsv")).unwrap();
    let mut fpv = fs::File::create(format!("{prefix}.pv.tsv")).unwrap();
    writeln!(fpos, "pos\tphase\tempties\tply\tsdepth\tsnodes\tddepth\tdnodes\tshallow\tdeep\tagree\tshallow_score\tdeep_score\tpv_len\tfirst_event\tfirst_event_ply").unwrap();
    writeln!(fpv, "pos\tphase\tagree\tpv_ply\tmove\tevent\tdepth\tfull_depth\treduction\tidx\tslack\tstat\tsubtree").unwrap();
    // phase x kind aggregates
    let mut agg: HashMap<(usize, String), Agg> = HashMap::new();

    for (pi, (b, gply)) in pos.iter().enumerate() {
        if pi % shard_n != shard_k {
            continue;
        }
        let ph = phase_of(b.empty().count_ones());
        // deep reference
        let mut sd = Searcher::new_unconfigured();
        sd.set_nnue(nn.as_ref());
        let rd = sd.search(b, &SearchLimits { time: None, max_nodes: Some(dnodes), max_depth: u8::MAX });
        let Some(deep) = rd.best else { continue };
        // the root is not stored in the TT (search_gen root loop): PV = deep best + the TT walk from its child
        let mut pv = vec![deep];
        pv.extend(sd.pv_from_tt(&b.make(deep), 39));
        // shallow depth that completes within snodes
        let mut s0 = Searcher::new_unconfigured();
        s0.set_nnue(nn.as_ref());
        let r0 = s0.search(b, &SearchLimits { time: None, max_nodes: Some(snodes), max_depth: u8::MAX });
        let ds = if r0.stopped_by == Some(titanium::StopReason::Nodes) { r0.depth.saturating_sub(1).max(1) } else { r0.depth.max(1) };
        drop(s0);
        let mut ss = Searcher::new_unconfigured();
        ss.set_nnue(nn.as_ref());
        ss.trace = Some(Box::new(Tracer::new(20_000_000)));
        let rs = ss.search(b, &SearchLimits { time: None, max_nodes: None, max_depth: ds as u8 });
        let t = ss.trace.take().unwrap();
        let recs: &[Rec] = &t.recs;
        if recs.is_empty() {
            continue;
        }
        // subtree sizes (children come after parents)
        let mut sub = vec![1u64; recs.len()];
        let mut children: Vec<Vec<u32>> = vec![Vec::new(); recs.len()];
        for i in (1..recs.len()).rev() {
            let p = recs[i].parent;
            if p != u32::MAX && (p as usize) < recs.len() {
                if recs[i].depth >= 0 {
                    sub[p as usize] += sub[i];
                }
                children[p as usize].push(i as u32);
            }
        }
        for c in children.iter_mut() {
            c.reverse();
        }
        // whole-tree aggregates
        for (i, r) in recs.iter().enumerate() {
            let ph_n = phase_of(r.empties as u32);
            let mut key = kind::NAMES[r.kind as usize].to_string();
            if r.via == via::NULL {
                key = if recs.get(r.parent as usize).map(|p| p.kind) == Some(kind::NMP) { "null_search_cut".into() } else { "null_search_failed".into() };
            } else if r.via == via::PROBCUT {
                key = if recs.get(r.parent as usize).map(|p| p.kind) == Some(kind::PROBCUT) { "probcut_search_cut".into() } else { "probcut_search_failed".into() };
            }
            let a = agg.entry((ph_n, key)).or_default();
            a.n += 1;
            a.sub += if r.depth >= 0 { sub[i] } else { 0 };
            // RFP near misses: non-PV, depth 1..4, failed low, not a null/probcut search itself
            if r.kind == kind::FULL && r.ply > 0 && r.sdepth >= 1 && r.sdepth <= 4 && r.beta - r.alpha == 1 && r.score != i32::MIN && r.score <= r.alpha && r.alpha.abs() < 90_000 {
                let gap = r.alpha - (r.stat + RFP_MARGINS[(r.sdepth - 1) as usize] + ((r.dis * DIS_RFP_NUM) >> 7));
                let bucket = if gap > -50 { "rfp_miss_0_50" } else if gap > -100 { "rfp_miss_50_100" } else if gap > -200 { "rfp_miss_100_200" } else { "rfp_miss_200plus" };
                let a = agg.entry((ph_n, format!("{bucket}_d{}", r.sdepth))).or_default();
                a.n += 1;
                a.sub += sub[i];
            }
            let par = recs.get(r.parent as usize);
            // KNOB BUCKETS (D556): null move by parent static - beta (hit = cut)
            if r.via == via::NULL {
                if let Some(p) = par {
                    let a = agg.entry((phase_of(p.empties as u32), format!("knob_nmp_try {}", sbucket(p.stat - p.beta)))).or_default();
                    a.n += 1;
                    a.sub += sub[i];
                    a.hit += (p.kind == kind::NMP) as u64;
                }
            }
            // ProbCut by parent static - pc_beta (the probcut search window is pc_beta-1, pc_beta)
            if r.via == via::PROBCUT {
                if let Some(p) = par {
                    let a = agg.entry((phase_of(p.empties as u32), format!("knob_pc_try {}", sbucket(p.stat - r.beta)))).or_default();
                    a.n += 1;
                    a.sub += sub[i];
                    a.hit += (p.kind == kind::PROBCUT) as u64;
                }
            }
            // RFP gap (alpha - (static + margin)) for non-PV depth 1..4 nodes that ran the move loop or were RFP-cut
            // (hit = failed low anyway; gap > 0 = RFP fired)
            if (r.kind == kind::FULL || r.kind == kind::RFP)
                && r.ply > 0
                && r.sdepth >= 1
                && r.sdepth <= 4
                && r.beta - r.alpha == 1
                && r.score != i32::MIN
                && r.alpha.abs() < 90_000
                && r.stat != i32::MIN
            {
                let gap = r.alpha - (r.stat + RFP_MARGINS[(r.sdepth - 1) as usize] + ((r.dis * DIS_RFP_NUM) >> 7));
                let a = agg.entry((ph_n, format!("knob_rfp_d{} gap {}", r.sdepth, sbucket(gap)))).or_default();
                a.n += 1;
                a.sub += sub[i];
                a.hit += (r.kind == kind::RFP || r.score <= r.alpha) as u64;
            }
            // move index (LMR start, D556): scouts at index k (hit = the scout failed high = the move raised alpha), by
            // the PARENT's phase; reduced ones separately; skipped children counted
            if let Some(p) = par {
                let pph = phase_of(p.empties as u32);
                if r.via == via::SCOUT && r.depth >= 0 && r.score != i32::MIN {
                    let a = agg.entry((pph, format!("knob_idx {} scout", ibucket(r.idx)))).or_default();
                    a.n += 1;
                    a.sub += sub[i];
                    a.hit += (r.score <= r.alpha) as u64;
                    if r.full_depth > r.depth {
                        let a = agg.entry((pph, format!("knob_idx {} reduced", ibucket(r.idx)))).or_default();
                        a.n += 1;
                        a.sub += sub[i];
                        a.hit += (r.score <= r.alpha) as u64;
                    }
                }
                if r.kind == kind::SKIP_FUT || r.kind == kind::SKIP_SN {
                    let a = agg.entry((pph, format!("knob_idx {} skipped", ibucket(r.idx)))).or_default();
                    a.n += 1;
                }
            }
        }
        // deep PV through the shallow tree
        let mut cur = 0usize;
        let mut bb = *b;
        let mut first: Option<(String, usize)> = None;
        let agree = rs.best == Some(deep);
        for (k, &m) in pv.iter().enumerate() {
            let child = bb.make(m);
            let cand: Vec<u32> = children[cur]
                .iter()
                .copied()
                .filter(|&c| {
                    let r = &recs[c as usize];
                    r.hash == child.hash && r.via != via::NULL && r.via != via::PROBCUT
                })
                .collect();
            let node = &recs[cur];
            let (ev, rec): (String, Option<&Rec>) = if let Some(&last) = cand.last() {
                let r = &recs[last as usize];
                match r.kind {
                    kind::SKIP_SN | kind::SKIP_FUT | kind::SKIP_REP => (kind::NAMES[r.kind as usize].to_string(), Some(r)),
                    _ => {
                        let red = r.full_depth as i32 - r.depth as i32;
                        if r.via == via::SCOUT && red > 0 {
                            (format!("reduced{}", red.min(9)), Some(r))
                        } else {
                            ("full".to_string(), Some(r))
                        }
                    }
                }
            } else if node.kind == kind::FULL {
                ("cut_by_sibling".to_string(), None)
            } else if node.kind == kind::LEAF {
                ("horizon".to_string(), None)
            } else {
                (format!("node_{}", kind::NAMES[node.kind as usize]), None)
            };
            let r = rec.copied().unwrap_or_default();
            let red = if rec.is_some() && r.depth >= 0 { r.full_depth as i32 - r.depth as i32 } else { 0 };
            writeln!(
                fpv,
                "{pi}\t{ph}\t{}\t{k}\t{}\t{ev}\t{}\t{}\t{red}\t{}\t{}\t{}\t{}",
                agree as u8,
                mname(m),
                if rec.is_some() { r.depth as i32 } else { node.sdepth as i32 },
                if rec.is_some() { r.full_depth as i32 } else { node.sdepth as i32 - 1 },
                if rec.is_some() { r.idx as i32 } else { -1 },
                if rec.is_some() { r.slack } else { node.slack },
                if rec.is_some() { r.stat } else { node.stat },
                cand.last().map(|&c| sub[c as usize]).unwrap_or(0)
            )
            .unwrap();
            if first.is_none() && ev != "full" && ev != "horizon" {
                first = Some((ev.clone(), k));
            }
            match cand.last() {
                Some(&c) if recs[c as usize].depth >= 0 && ev != "horizon" => {
                    cur = c as usize;
                    bb = child;
                }
                _ => break,
            }
        }
        let (fe, fk) = first.unwrap_or(("none".into(), 0));
        writeln!(
            fpos,
            "{pi}\t{ph}\t{}\t{gply}\t{ds}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{fe}\t{fk}",
            b.empty().count_ones(),
            rs.nodes,
            rd.depth,
            rd.nodes,
            rs.best.map(mname).unwrap_or_default(),
            mname(deep),
            agree as u8,
            rs.score,
            rd.score,
            pv.len()
        )
        .unwrap();
        if pi < dump {
            let mut f = fs::File::create(format!("{prefix}.tree{pi}.tsv")).unwrap();
            writeln!(f, "id\tparent\tmove\tply\tempties\tdepth\tsdepth\tfull_depth\tkind\tvia\tidx\talpha\tbeta\tstat\tdis\tscore\tslack\tsubtree").unwrap();
            for (i, r) in recs.iter().enumerate() {
                let m = if r.mv == 0xFFFF { "-".to_string() } else { mname(Move::from_u32(r.mv as u32)) };
                writeln!(
                    f,
                    "{i}\t{}\t{m}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    r.parent as i64,
                    r.ply,
                    r.empties,
                    r.depth,
                    r.sdepth,
                    r.full_depth,
                    kind::NAMES[r.kind as usize],
                    via::NAMES[r.via as usize],
                    r.idx,
                    r.alpha,
                    r.beta,
                    r.stat,
                    r.dis,
                    r.score,
                    r.slack,
                    if r.depth >= 0 { sub[i] } else { 0 }
                )
                .unwrap();
            }
        }
        eprintln!(
            "[{}/{}] phase {ph:2} empties {:2} shallow d{ds} {}n {} | deep d{} {}n {} | agree {} | first PV event {fe} @ply {fk}",
            pi + 1,
            pos.len(),
            b.empty().count_ones(),
            rs.nodes,
            rs.best.map(mname).unwrap_or_default(),
            rd.depth,
            rd.nodes,
            mname(deep),
            agree as u8
        );
    }
    let mut keys: Vec<_> = agg.keys().cloned().collect();
    keys.sort();
    let mut fn_ = fs::File::create(format!("{prefix}.nodes.tsv")).unwrap();
    writeln!(fn_, "phase\tkind\tcount\tsubtree_nodes\thit").unwrap();
    for k in keys {
        let a = &agg[&k];
        writeln!(fn_, "{}\t{}\t{}\t{}\t{}", k.0, k.1, a.n, a.sub, a.hit).unwrap();
    }
    eprintln!("wrote {prefix}.pos.tsv, {prefix}.pv.tsv, {prefix}.nodes.tsv");
    ExitCode::SUCCESS
}

fn read_tsv(path: &std::path::Path) -> Vec<HashMap<String, String>> {
    let Ok(t) = fs::read_to_string(path) else { return Vec::new() };
    let mut it = t.lines();
    let Some(h) = it.next() else { return Vec::new() };
    let hs: Vec<&str> = h.split('\t').collect();
    it.map(|l| hs.iter().zip(l.split('\t')).map(|(a, b)| (a.to_string(), b.to_string())).collect()).collect()
}

fn files(prefix: &str, suffix: &str) -> Vec<std::path::PathBuf> {
    let p = std::path::Path::new(prefix);
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    let stem = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let mut v: Vec<_> = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| {
                    let n = f.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                    n.starts_with(&stem) && n.ends_with(suffix) && (n == format!("{stem}{suffix}") || n[stem.len()..].starts_with(".s"))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// `treediff --report PREFIX`: summary over PREFIX[.sK].{pos,pv,nodes}.tsv, printed and written to PREFIX.report.txt
fn report(prefix: &str) -> ExitCode {
    let mut out = String::new();
    let pos: Vec<_> = files(prefix, ".pos.tsv").iter().flat_map(|f| read_tsv(f)).collect();
    let pv: Vec<_> = files(prefix, ".pv.tsv").iter().flat_map(|f| read_tsv(f)).collect();
    let nodes: Vec<_> = files(prefix, ".nodes.tsv").iter().flat_map(|f| read_tsv(f)).collect();
    let g = |r: &HashMap<String, String>, k: &str| r.get(k).cloned().unwrap_or_default();
    let gi = |r: &HashMap<String, String>, k: &str| r.get(k).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
    out += &format!("treediff report {prefix}: {} positions, {} PV rows\n\n", pos.len(), pv.len());
    out += "phase empties  n  agree%  sdepth ddepth  |deep-shallow score|  first event when they disagree\n";
    for ph in 0..16 {
        let rs: Vec<_> = pos.iter().filter(|r| gi(r, "phase") == ph).collect();
        if rs.is_empty() {
            continue;
        }
        let n = rs.len() as f64;
        let ag = rs.iter().filter(|r| g(r, "agree") == "1").count() as f64;
        let sd = rs.iter().map(|r| gi(r, "sdepth")).sum::<i64>() as f64 / n;
        let dd = rs.iter().map(|r| gi(r, "ddepth")).sum::<i64>() as f64 / n;
        let ds = rs.iter().map(|r| (gi(r, "deep_score") - gi(r, "shallow_score")).abs().min(2000)).sum::<i64>() as f64 / n;
        let mut fe: HashMap<String, usize> = HashMap::new();
        for r in rs.iter().filter(|r| g(r, "agree") == "0") {
            *fe.entry(g(r, "first_event")).or_default() += 1;
        }
        let mut fev: Vec<_> = fe.into_iter().collect();
        fev.sort_by(|a, b| b.1.cmp(&a.1));
        let fes: Vec<String> = fev.iter().map(|(k, v)| format!("{k} {v}")).collect();
        out += &format!(
            "{ph:5} {:>2}-{:<2} {:3} {:6.1}  {sd:6.1} {dd:6.1}  {ds:8.0}              {}\n",
            45 - 3 * ph as i64 - 2,
            45 - 3 * ph as i64,
            rs.len(),
            100.0 * ag / n,
            fes.join(", ")
        );
    }
    // PV events: what the shallow tree did to the deep PV move, by side (even ply = the root side) and agreement
    out += "\nPV events (deep PV move in the shallow tree), share of PV plies: root side (even ply) | other side (odd ply)\n";
    for agree in ["0", "1"] {
        let mut c: HashMap<(String, bool), usize> = HashMap::new();
        let mut tot = [0usize; 2];
        for r in pv.iter().filter(|r| g(r, "agree") == agree) {
            let odd = gi(r, "pv_ply") % 2 == 1;
            *c.entry((g(r, "event"), odd)).or_default() += 1;
            tot[odd as usize] += 1;
        }
        let mut evs: Vec<String> = c.keys().map(|k| k.0.clone()).collect();
        evs.sort();
        evs.dedup();
        out += &format!("  {} positions ({} / {} plies):\n", if agree == "1" { "AGREE" } else { "DISAGREE" }, tot[0], tot[1]);
        for e in evs {
            let a = *c.get(&(e.clone(), false)).unwrap_or(&0);
            let b = *c.get(&(e.clone(), true)).unwrap_or(&0);
            out += &format!("    {e:16} {:5.1}% | {:5.1}%\n", 100.0 * a as f64 / tot[0].max(1) as f64, 100.0 * b as f64 / tot[1].max(1) as f64);
        }
    }
    // tree composition per phase group (4 phases each)
    out += "\nTree composition (shallow trees): count and share of tree nodes per node kind, by phase group\n";
    for grp in 0..4 {
        let rs: Vec<_> = nodes.iter().filter(|r| gi(r, "phase") / 4 == grp).collect();
        if rs.is_empty() {
            continue;
        }
        let mut c: HashMap<String, (i64, i64)> = HashMap::new();
        for r in &rs {
            let e = c.entry(g(r, "kind")).or_default();
            e.0 += gi(r, "count");
            e.1 += gi(r, "subtree_nodes");
        }
        let total: i64 = c.iter().filter(|(k, _)| !k.starts_with("rfp_miss") && !k.starts_with("skip") && !k.contains("_search_")).map(|(_, v)| v.0).sum();
        out += &format!("  phases {}-{} (empties {}..{}), {} nodes:\n", grp * 4, grp * 4 + 3, 45 - 12 * grp as i64 - 11, 45 - 12 * grp as i64, total);
        let mut ks: Vec<_> = c.into_iter().collect();
        ks.sort();
        for (k, (n, sub)) in ks {
            out += &format!("    {k:24} {n:9} ({:5.1}% of nodes)  subtree nodes {sub:10} ({:5.1}%)\n", 100.0 * n as f64 / total.max(1) as f64, 100.0 * sub as f64 / total.max(1) as f64);
        }
    }
    // knob tables: every knob_* key, per phase -> count, hit %, subtree nodes
    let mut kk: Vec<String> = nodes.iter().map(|r| g(r, "kind")).filter(|k| k.starts_with("knob_")).collect();
    kk.sort();
    kk.dedup();
    out += "\nKnob buckets per phase: count / hit% (nmp, pc: cut; rfp: failed low; idx: raised alpha) / subtree nodes\n";
    for k in kk {
        out += &format!("  {k}\n");
        for ph in 0..16 {
            let rs: Vec<_> = nodes.iter().filter(|r| g(r, "kind") == k && gi(r, "phase") == ph).collect();
            if rs.is_empty() {
                continue;
            }
            let (n, h, sb) = rs.iter().fold((0i64, 0i64, 0i64), |a, r| (a.0 + gi(r, "count"), a.1 + gi(r, "hit"), a.2 + gi(r, "subtree_nodes")));
            out += &format!("    phase {ph:2}: {n:8}  {:5.1}%  {sb:9}\n", 100.0 * h as f64 / n.max(1) as f64);
        }
    }
    print!("{out}");
    let _ = fs::write(format!("{prefix}.report.txt"), &out);
    ExitCode::SUCCESS
}