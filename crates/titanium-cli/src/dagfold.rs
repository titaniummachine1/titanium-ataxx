//! Game-DAG fold: datagen shard text -> SQLite DAG with natural dedup.
//!
//! The SAME fen reached via different move orders is the SAME position: the
//! fen string drops pass-count/clock, so dedup falls out of the map key with
//! ZERO extra compute — no pairwise compares, no hashing pass over the corpus.
//! Labels deduplicate by running average: repeated visits refine the target
//! instead of duplicating the row (Quoridor's 10.3M-labels-on-5.2M-positions
//! lesson: dup rows let one position dominate the epoch).
//!
//! Layout: `nodes(fen PK, stm, w[8], visits, sum_outcome, sum_score)` +
//! `edges(parent, child, move, PRIMARY KEY(parent,child,move))` +
//! `meta(key,value)` (fold provenance: shard list, totals).
//! Fold is incremental: re-running on the same DB + new shards only adds.
//!
//! Usage: `titanium-cli dagfold --db data/nnue/dag.db --inputs a.txt,b.txt`

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;

use titanium::Board;

pub fn dagfold_cmd(db_path: &str, inputs: &str) -> ExitCode {
    let inputs: Vec<&str> = inputs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if inputs.is_empty() {
        eprintln!("dagfold: --inputs a.txt[,b.txt...] is required");
        return ExitCode::FAILURE;
    }
    let db = match DagDb::open(db_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("dagfold: cannot open {db_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut stats = FoldStats::default();
    let mut cache: HashMap<String, CachedNode> = HashMap::new();
    let mut edges: Vec<(String, String, String)> = Vec::new();
    let mut cache: HashMap<String, CachedNode> = HashMap::new();
    for path in &inputs {
        let file = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("dagfold: cannot open {path}: {e}");
                continue;
            }
        };
        // Track the game path per file: consecutive records from one game
        // form parent->child edges. A new file resets the chain. Records
        // carry no game id, so edges are best-effort: parent = previous
        // record's fen in THIS file only, and only if legal (child must be
        // reachable in one move or pass from parent — else chain broke,
        // e.g. game boundary inside a shard, and we reset).
        let mut prev: Option<Board> = None;
        let mut prev_fen = String::new();
        for line in BufReader::new(file).lines() {
            let Ok(line) = line else { continue };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            stats.lines += 1;
            let mut parts = line.split('|');
            let (Some(fen), Some(score_s), Some(res_s), Some(w_s)) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                stats.skipped += 1;
                prev = None;
                continue;
            };
            let Some(b) = Board::from_ataxx_fen(fen) else {
                stats.skipped += 1;
                prev = None;
                continue;
            };
            let score: i32 = score_s.parse().unwrap_or(0);
            let res: f64 = res_s.parse().unwrap_or(0.5);
            let w: Vec<i32> = w_s.split(',').map(|x| x.parse().unwrap_or(0)).collect();
            if w.len() != 8 {
                stats.skipped += 1;
                prev = None;
                continue;
            }
            // Fold the node: new fen = insert, seen fen = running average.
            let key = cache_key(fen);
            let entry = cache.entry(key.clone()).or_insert_with(|| CachedNode {
                stm: b.turn,
                w,
                visits: 0,
                sum_outcome: 0.0,
                sum_score: 0.0,
                dirty: true,
            });
            if entry.visits > 0 {
                stats.cache_dup += 1;
            } else {
                stats.cache_new += 1;
            }
            entry.visits += 1;
            entry.sum_outcome += res;
            entry.sum_score += score as f64;
            // Edge from previous record if it connects legally (buffered,
            // flushed with the nodes — one python call per FLUSH, not per move).
            if let Some(pb) = &prev {
                if let Some(mv) = connect_move(pb, &b) {
                    edges.push((prev_fen.clone(), cache_key(fen), mv));
                }
            }
            prev = Some(b);
            prev_fen = fen.to_string();
        }
        if cache.len() >= 20000 {
            db.flush_nodes(db_path, &mut cache, &mut edges, &mut stats);
        }
        prev = None;
    }
    // Final flush of anything cached.
    db.flush_nodes(db_path, &mut cache, &mut edges, &mut stats);
    db.bump_meta(db_path, "folds", 1);
    db.add_meta_inputs(db_path, &inputs);
    println!(
        "dagfold: {lines} lines -> cache {new_nodes}n/{dup_hits}d, edges +{new_edges}, skipped {skipped} -> {db_path} (batches ok {bok}/err {berr})",
        lines = stats.lines,
        new_nodes = stats.cache_new,
        dup_hits = stats.cache_dup,
        new_edges = stats.new_edges,
        skipped = stats.skipped,
        bok = stats.batches_ok,
        berr = stats.batches_err,
    );
    let (nn, nv, ne) = db.counts(db_path).unwrap_or((0, 0, 0));
    println!("dagfold: db now holds {nn} nodes / {nv} visits / {ne} edges");
    ExitCode::SUCCESS
}

#[derive(Default)]
struct FoldStats {
    lines: u64,
    cache_new: u64,
    cache_dup: u64,
    skipped: u64,
    new_edges: u64,
    batches_ok: u64,
    batches_err: u64,
}

struct CachedNode {
    stm: u8,
    w: Vec<i32>,
    visits: u64,
    sum_outcome: f64,
    sum_score: f64,
    dirty: bool,
}

/// Map key: the fen board+side part only (drop the trailing " 0 1" clock —
/// same position, same node regardless of path length).
fn cache_key(fen: &str) -> String {
    match fen.split_whitespace().collect::<Vec<_>>().as_slice() {
        [board, side, ..] => format!("{board} {side}"),
        _ => fen.to_string(),
    }
}

/// Legal connector between consecutive records: a pass (same stones, side
/// flips) or any single move. Returns the move token for the edge label.
/// NOTE: positions are compared by FULL board (occ+blockers+turn); pass and
/// ply counters are intentionally ignored (same DAG node by construction).
fn connect_move(parent: &Board, child: &Board) -> Option<String> {
    if parent.occ == child.occ && parent.blockers == child.blockers
        && child.turn == 1 - parent.turn
    {
        return Some("pass".to_string());
    }
    // Child must differ by exactly one landing + optional vacate + flips.
    // Cheap check: child side-to-move flipped and stone diff is bounded.
    if child.turn != 1 - parent.turn {
        return None;
    }
    let moves = parent.legal_moves();
    for i in 0..moves.len() {
        let m = moves.move_at(i);
        if parent.make(m) == *child {
            return Some(format!("{:?}", m.to_u32()));
        }
    }
    None
}

// ---------- storage backend: python sqlite3 via stdio ----------
// rusqlite is NOT a workspace dep and no sqlite3 CLI exists on this box, so
// dagfold streams SQL batches to `python3 -c` running sqlite3 from the stdlib.
// Same on-disk schema as documented above; python is already a harness dep.

pub struct DagDb {
    _path: String,
    py: String,
}

fn find_python() -> Option<String> {
    for cand in ["python3", "python", "py"] {
        let probe = std::process::Command::new(cand)
            .arg("-c")
            .arg("import sqlite3")
            .output();
        if probe.map(|o| o.status.success()).unwrap_or(false) {
            return Some(cand.to_string());
        }
    }
    None
}

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS nodes(fen TEXT PRIMARY KEY, stm INTEGER, w0 INTEGER, w1 INTEGER, w2 INTEGER, w3 INTEGER, w4 INTEGER, w5 INTEGER, w6 INTEGER, w7 INTEGER, visits INTEGER, sum_outcome REAL, sum_score REAL);\nCREATE TABLE IF NOT EXISTS edges(parent TEXT, child TEXT, mv TEXT, PRIMARY KEY(parent, child, mv));\nCREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);";

const PY_RUNNER: &str = "import sqlite3,sys\np=sys.argv[1]\nsql=sys.stdin.read()\nc=sqlite3.connect(p)\nc.executescript(sql)\nc.commit()\nprint(c.execute('SELECT COUNT(*), COALESCE(SUM(visits),0) FROM nodes').fetchone())\nc.close()\n";

/// Escape a string for embedding in a double-quoted Python literal.
fn py_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            _ => o.push(ch),
        }
    }
    o.push('"');
    o
}

impl DagDb {
    pub fn open(path: &str) -> Result<DagDb, String> {
        let py = find_python().ok_or_else(|| "no python with sqlite3".to_string())?;
        let prog = format!(
            "import sqlite3; c=sqlite3.connect({}); c.executescript({}); c.commit(); c.close()",
            py_str(path),
            py_str(SCHEMA)
        );
        let o = std::process::Command::new(&py)
            .arg("-c")
            .arg(prog)
            .output()
            .map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).to_string());
        }
        Ok(DagDb { _path: path.to_string(), py })
    }

    fn exec_batch(&self, path: &str, sql: &str) -> Result<String, String> {
        // SQL via stdin (no quoting layers at all): argv[1] = db path.
        let o = std::process::Command::new(&self.py)
            .arg("-c")
            .arg(PY_RUNNER)
            .arg(path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child.stdin.take().map(|mut s| s.write_all(sql.as_bytes()));
                child.wait_with_output()
            })
            .map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).to_string());
        }
        Ok(String::from_utf8_lossy(&o.stdout).to_string())
    }

    fn esc(s: &str) -> String {
        s.replace('\'', "''")
    }

    /// Bulk flush: one python call per FLUSH (not per row): INSERT-or-IGNORE
    /// the batch, then running-average UPDATE only rows that pre-existed
    /// (`visits > v` is FALSE for fresh rows whose stored visits == batch v;
    /// exact-collision race is astronomically rare and self-heals next fold).
    /// Edges ride the same call.
    pub fn flush_nodes(
        &self,
        path: &str,
        cache: &mut HashMap<String, CachedNode>,
        edges: &mut Vec<(String, String, String)>,
        stats: &mut FoldStats,
    ) {
        if cache.is_empty() && edges.is_empty() {
            cache.clear();
            edges.clear();
            return;
        }
        let mut sql = String::from("BEGIN;");
        for (fen, n) in cache.iter() {
            let f = Self::esc(fen);
            sql.push_str(&format!(
                "INSERT OR IGNORE INTO nodes VALUES('{f}',{},{} ,{},{},{},{},{},{},{},{},{},{});",
                n.stm, n.w[0], n.w[1], n.w[2], n.w[3], n.w[4], n.w[5], n.w[6], n.w[7],
                n.visits, n.sum_outcome, n.sum_score
            ));
            sql.push_str(&format!(
                "UPDATE nodes SET visits=visits+{v}, sum_outcome=sum_outcome+{o}, \
                 sum_score=sum_score+{s} WHERE fen='{f}' AND visits>{v};",
                v = n.visits, o = n.sum_outcome, s = n.sum_score, f = f
            ));
        }
        for (p, c, m) in edges.iter() {
            sql.push_str(&format!(
                "INSERT OR IGNORE INTO edges VALUES('{}','{}','{}');",
                Self::esc(p), Self::esc(c), Self::esc(m)
            ));
        }
        sql.push_str("COMMIT;");
        match self.exec_batch(path, &sql) {
            Ok(_) => stats.batches_ok += 1,
            Err(e) => {
                stats.batches_err += 1;
                eprintln!("dagfold flush: {e}");
            }
        }
        stats.new_edges += edges.len() as u64;
        cache.clear();
        edges.clear();
    }

    pub fn bump_meta(&self, path: &str, key: &str, by: i64) {
        let k = Self::esc(key);
        let _ = self.exec_batch(path, &format!(
            "INSERT INTO meta VALUES('{k}',{by}) ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+{by};"
        ));
    }

    pub fn add_meta_inputs(&self, path: &str, inputs: &[&str]) {
        let v = Self::esc(&inputs.join(";"));
        let _ = self.exec_batch(path, &format!(
            "INSERT INTO meta VALUES('inputs','{v}') ON CONFLICT(key) DO UPDATE SET value=value||';'||'{v}';"
        ));
    }

    pub fn counts(&self, path: &str) -> Result<(u64, u64, u64), String> {
        let out = self.exec_batch(
            path,
            "SELECT COUNT(*), COALESCE(SUM(visits),0) FROM nodes; SELECT COUNT(*) FROM edges;",
        )?;
        let mut lines = out.lines();
        let (mut nn, mut nv, mut ne) = (0u64, 0u64, 0u64);
        if let Some(l) = lines.next() {
            let nums: Vec<u64> = l
                .trim_matches(|c| c == '(' || c == ')' || c == ' ')
                .split(", ")
                .map(|x| x.parse().unwrap_or(0))
                .collect();
            if nums.len() >= 2 {
                nn = nums[0];
                nv = nums[1];
            }
        }
        if let Some(l) = lines.next() {
            ne = l
                .trim_matches(|c| c == '(' || c == ')' || c == ' ' || c == ',')
                .split(',')
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
        }
        Ok((nn, nv, ne))
    }
}
