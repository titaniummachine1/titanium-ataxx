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
}

pub fn run(args: &[String]) -> ExitCode {
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
    let prefix = arg(args, "--out").unwrap_or_else(|| "treediff".into());
    let nn = Some(titanium::nnue::Nnue::bundled());

    // positions, at most per_phase per phase, spread over the games
    let mut pos: Vec<(Board, usize)> = Vec::new();
    let mut cnt = [0usize; 16];
    for (start, mv) in read_games(&games, max_games) {
        let Some(mut b) = Board::from_ataxx_fen(&start) else { continue };
        for (k, m) in mv.iter().enumerate() {
            if k > 0 && k % every == 0 && !b.game_over() {
                let ph = phase_of(b.empty().count_ones());
                if cnt[ph] < per_phase {
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
    writeln!(fn_, "phase\tkind\tcount\tsubtree_nodes").unwrap();
    for k in keys {
        let a = &agg[&k];
        writeln!(fn_, "{}\t{}\t{}\t{}", k.0, k.1, a.n, a.sub).unwrap();
    }
    eprintln!("wrote {prefix}.pos.tsv, {prefix}.pv.tsv, {prefix}.nodes.tsv");
    ExitCode::SUCCESS
}
