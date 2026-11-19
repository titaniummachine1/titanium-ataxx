//! RL prototype commands (Jataxx-style outcome-grounded self-play):
//!   rlgen   --games N --nodes N --threads T --out F.rec [--seed S] [--tuple P]
//!   rltrain --data a.rec,b.rec --out net.n3 [--init net.n3] [--hidden 64]
//!           [--epochs 4] [--lr 0.02] [--val 0.05] [--blend 0.0]
//!
//! Record (24 bytes LE): occ0 u64 | occ1 u64 | turn u8 | result u8 (stm: 0 loss,
//! 1 draw, 2 win) | base_cp i16 (classical+tuple eval, NO net, stm-relative) |
//! margin i8 (stm stones - foe stones at game end) | pad 3.
//! Training = logistic regression of the game outcome on
//! (base_cp + 100*net(board)) / S, so the net only learns what the classical
//! eval misses and a zero net is exactly the classical eval.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use titanium::net3::{self, Net3, BUCKETS, WIN, ZMAX, OUT_CP};
use titanium::search::evaluate_base;
use titanium::{Board, SearchLimits, Searcher, TupleTable};

const REC: usize = 24;
const DEFAULT_TUP: &str = "data/nnue/tupS55tree.tup";

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}
fn flag_num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    flag(args, name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

// ---------------------------------------------------------------- rlgen

pub fn rlgen(args: &[String]) -> std::process::ExitCode {
    let games: usize = flag_num(args, "--games", 100);
    let nodes: u64 = flag_num(args, "--nodes", 5000);
    let threads: usize = flag_num(args, "--threads", 6);
    let seed: u64 = flag_num(args, "--seed", 1);
    let open_min: u64 = flag_num(args, "--open-min", 2);
    let open_max: u64 = flag_num(args, "--open-max", 8);
    let out = flag(args, "--out").unwrap_or("rl.rec").to_string();
    let tup_path = flag(args, "--tuple").unwrap_or(DEFAULT_TUP);
    let tup = if tup_path == "none" {
        None
    } else {
        match TupleTable::load(tup_path) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("rlgen: tuple {tup_path}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    };
    let net_on = net3::global().is_some();
    println!("rlgen: {games} games, {nodes} nodes, {threads} threads, net={net_on}, out={out}");

    let next = AtomicUsize::new(0);
    let sink: Mutex<Vec<u8>> = Mutex::new(Vec::new());
    let stats: Mutex<[u64; 4]> = Mutex::new([0; 4]); // black, white, draw, positions
    let t0 = Instant::now();

    std::thread::scope(|s| {
        for tid in 0..threads {
            let next = &next;
            let sink = &sink;
            let stats = &stats;
            let tup = tup.as_ref();
            s.spawn(move || {
                let mut searcher = Searcher::new();
                searcher.set_tuple(tup);
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break;
                    }
                    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ ((g as u64 + 1) << 20) ^ (tid as u64 + 77));
                    for _ in 0..8 {
                        rng.next();
                    }
                    searcher.reset_for_new_game();
                    let mut b = Board::start();
                    // random opening for diversity
                    let open = open_min + rng.below(open_max - open_min + 1);
                    for _ in 0..open {
                        if b.game_over() {
                            break;
                        }
                        let ms = b.legal_moves();
                        if ms.is_empty() {
                            b = b.make_pass();
                        } else {
                            b = b.make(ms.move_at(rng.below(ms.len() as u64) as usize));
                        }
                    }
                    let mut recs: Vec<(u64, u64, u8, i16)> = Vec::new();
                    let mut ply = 0;
                    while !b.game_over() && ply < 300 {
                        ply += 1;
                        if !b.has_moves(b.turn) {
                            b = b.make_pass();
                            continue;
                        }
                        // +-20% node jitter keeps deterministic engines from replaying games
                        let jn = (nodes as f32 * (0.8 + 0.4 * rng.unit())) as u64;
                        let limits = SearchLimits { time: None, max_nodes: Some(jn), max_depth: 24 };
                        let r = searcher.search(&b, &limits);
                        let Some(mv) = r.best else {
                            b = b.make_pass();
                            continue;
                        };
                        if r.score.abs() < 90_000 {
                            let base = evaluate_base(&b, tup, false).clamp(-30000, 30000) as i16;
                            recs.push((b.occ[0], b.occ[1], b.turn, base));
                        }
                        b = if mv.is_pass() { b.make_pass() } else { b.make(mv) };
                    }
                    let (bl, wh) = b.counts();
                    let winner = if ply >= 300 && !b.game_over() { 2 } else { b.winner() };
                    let mut buf = Vec::with_capacity(recs.len() * REC);
                    for (o0, o1, turn, base) in &recs {
                        let t = *turn as usize;
                        let res: u8 = if winner == 2 { 1 } else if winner as usize == t { 2 } else { 0 };
                        let margin = if t == 0 { bl as i32 - wh as i32 } else { wh as i32 - bl as i32 };
                        buf.extend_from_slice(&o0.to_le_bytes());
                        buf.extend_from_slice(&o1.to_le_bytes());
                        buf.push(*turn);
                        buf.push(res);
                        buf.extend_from_slice(&base.to_le_bytes());
                        buf.push(margin.clamp(-49, 49) as i8 as u8);
                        buf.extend_from_slice(&(g as u32).to_le_bytes()[..3]); // game id (val split)
                    }
                    sink.lock().unwrap().extend_from_slice(&buf);
                    let mut st = stats.lock().unwrap();
                    st[winner.min(2) as usize] += 1;
                    st[3] += recs.len() as u64;
                    let done = st[0] + st[1] + st[2];
                    if done % 20 == 0 {
                        println!("  {done}/{games} games, {} positions, {:.0}s", st[3], t0.elapsed().as_secs_f32());
                    }
                }
            });
        }
    });

    let data = sink.into_inner().unwrap();
    if let Some(p) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(p);
    }
    std::fs::File::create(&out).and_then(|mut f| f.write_all(&data)).expect("write rec");
    let st = stats.into_inner().unwrap();
    println!(
        "rlgen done: black {} white {} draw {} | {} positions | {:.1}s | {}",
        st[0], st[1], st[2], st[3], t0.elapsed().as_secs_f32(), out
    );
    std::process::ExitCode::SUCCESS
}

// -------------------------------------------------------------- rltrain

#[derive(Clone, Copy)]
struct Sample {
    own: u64,
    foe: u64,
    base: f32,
    y: f32,
    pieces: u32,
    game: u64,
    out: f32, // game outcome (stm-relative 0..1) for reporting
    cp: f32,  // champion-labelled 5k search score (clamped +-1500)
    stm: u8,
}

/// v9 `.tdat` record (hub `ti data pack`): 48 B = occ0 u64 | occ1 u64 | outcome f32 | cp f32 |
/// wsum f32 | p3 f32 | cp5k f32 | zx i32 | stm u8 | ...
const TDAT: usize = 48;

fn tdat_rows(path: &str) -> usize {
    std::fs::metadata(path).map(|m| m.len() as usize / TDAT).unwrap_or(0)
}

/// Orientation check on a sample of rows: returns (flip_stm, flip_outcome).
/// flip_stm: own = occ[1-stm] fits cp better than own = occ[stm].
fn detect_orientation(path: &str, n: usize) -> (bool, bool) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut buf = vec![0u8; TDAT * n.min(tdat_rows(path))];
    f.read_exact(&mut buf).expect("read sample");
    let corr = |xs: &[f64], ys: &[f64]| -> f64 {
        let n = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (x, y) in xs.iter().zip(ys) {
            sxy += (x - mx) * (y - my);
            sxx += (x - mx) * (x - mx);
            syy += (y - my) * (y - my);
        }
        sxy / (sxx.sqrt() * syy.sqrt() + 1e-12)
    };
    let (mut d_a, mut d_b, mut cps, mut outs) = (vec![], vec![], vec![], vec![]);
    for c in buf.chunks_exact(TDAT) {
        let o0 = u64::from_le_bytes(c[0..8].try_into().unwrap());
        let o1 = u64::from_le_bytes(c[8..16].try_into().unwrap());
        let out = f32::from_le_bytes(c[16..20].try_into().unwrap()) as f64;
        let cp = f32::from_le_bytes(c[20..24].try_into().unwrap()).clamp(-1500.0, 1500.0) as f64;
        let stm = c[40] as usize;
        let (own_a, foe_a) = if stm == 0 { (o0, o1) } else { (o1, o0) };
        d_a.push(own_a.count_ones() as f64 - foe_a.count_ones() as f64);
        d_b.push(foe_a.count_ones() as f64 - own_a.count_ones() as f64);
        cps.push(cp);
        outs.push(out);
    }
    let (ca, cb) = (corr(&d_a, &cps), corr(&d_b, &cps));
    let co = corr(&cps, &outs);
    println!("orientation check on {} rows: corr(cp, stones own=occ[stm]) {ca:.3} vs own=occ[1-stm] {cb:.3}; corr(cp, outcome) {co:.3}", cps.len());
    (cb > ca, co < 0.0)
}

fn load_tdat(path: &str, ymode: &str, base_none: bool, flip_stm: bool, flip_out: bool, max_rows: usize) -> Vec<Sample> {
    use std::io::Read;
    let total = tdat_rows(path).min(max_rows);
    let mut f = std::io::BufReader::with_capacity(1 << 22, std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}")));
    let mut v: Vec<Sample> = Vec::with_capacity(total);
    let mut chunk = vec![0u8; TDAT * (1 << 16)];
    let mut left = total;
    while left > 0 {
        let n = left.min(1 << 16);
        f.read_exact(&mut chunk[..n * TDAT]).expect("read tdat");
        for c in chunk[..n * TDAT].chunks_exact(TDAT) {
            let o0 = u64::from_le_bytes(c[0..8].try_into().unwrap());
            let o1 = u64::from_le_bytes(c[8..16].try_into().unwrap());
            let mut out = f32::from_le_bytes(c[16..20].try_into().unwrap());
            if flip_out {
                out = 1.0 - out;
            }
            let cp = f32::from_le_bytes(c[20..24].try_into().unwrap()).clamp(-1500.0, 1500.0);
            let stm = (c[40] as usize) ^ (flip_stm as usize);
            let (own, foe) = if stm == 0 { (o0, o1) } else { (o1, o0) };
            let sg = 1.0 / (1.0 + (-cp / 400.0).exp());
            let y = match ymode {
                "cp" => sg,
                "blend" => 0.75 * sg + 0.25 * out,
                _ => out,
            };
            v.push(Sample { own, foe, base: if base_none { 0.0 } else { cp }, y, pieces: (own | foe).count_ones(), game: 0, out, cp, stm: c[40] });
        }
        left -= n;
    }
    v
}

/// tstg_v10_31m.bin record (40 B, repr(C)): occ_us u64 (side to move) | occ_them u64 | blockers u64 |
/// turn u8 | from u8 | to u8 | flags u8 | score_cp i16 | outcome_wdl u8 (0 loss,1 draw,2 win) | depth u8 | pad[6]
const BIN: usize = 40;

fn load_bin(path: &str, ymode: &str, base_none: bool, max_rows: usize) -> Vec<Sample> {
    use std::io::Read;
    let total = (std::fs::metadata(path).expect("bin metadata").len() as usize / BIN).min(max_rows);
    let mut f = std::io::BufReader::with_capacity(1 << 22, std::fs::File::open(path).expect("open bin"));
    let mut v: Vec<Sample> = Vec::with_capacity(total);
    let mut chunk = vec![0u8; BIN * (1 << 16)];
    let mut left = total;
    let mut wdl_hist = [0u64; 4];
    let mut depth_sum = 0u64;
    let mut adj_ham = 0u64;
    let mut prev = 0u64;
    while left > 0 {
        let n = left.min(1 << 16);
        f.read_exact(&mut chunk[..n * BIN]).expect("read bin");
        for c in chunk[..n * BIN].chunks_exact(BIN) {
            let own = u64::from_le_bytes(c[0..8].try_into().unwrap());
            let foe = u64::from_le_bytes(c[8..16].try_into().unwrap());
            let cp = (i16::from_le_bytes([c[28], c[29]]) as f32).clamp(-1500.0, 1500.0);
            let wdl = c[30];
            wdl_hist[(wdl as usize).min(3)] += 1;
            depth_sum += c[31] as u64;
            adj_ham += (own ^ prev).count_ones() as u64;
            prev = own;
            let out = wdl.min(2) as f32 * 0.5;
            let sg = 1.0 / (1.0 + (-cp / 400.0).exp());
            let y = match ymode {
                "cp" => sg,
                "blend" => 0.75 * sg + 0.25 * out,
                _ => out,
            };
            v.push(Sample { own, foe, base: if base_none { 0.0 } else { cp }, y, pieces: (own | foe).count_ones(), game: 0, out, cp, stm: c[24] });
        }
        left -= n;
    }
    // Quick facts: outcome populated? mean depth; adjacent-row similarity (ordered by game?) and correlations.
    let n = v.len() as f64;
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    let (mut tx, mut txx, mut txy) = (0.0f64, 0.0, 0.0);
    for s in &v {
        let x = s.base as f64;
        let y = s.out as f64;
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
        let d = s.own.count_ones() as f64 - s.foe.count_ones() as f64;
        tx += d; txx += d * d; txy += d * x;
    }
    let corr = |sxy: f64, sx: f64, sy: f64, sxx: f64, syy: f64| (sxy - sx * sy / n) / (((sxx - sx * sx / n) * (syy - sy * sy / n)).sqrt() + 1e-12);
    println!(
        "bin facts ({} rows): outcome_wdl hist loss/draw/win/other = {:?}; mean depth {:.1}; mean adjacent-row hamming(own) {:.1} (random pairs ~ 12+); corr(cp,outcome) {:.3}; corr(cp, stones us-them) {:.3}",
        v.len(), wdl_hist, depth_sum as f64 / n, adj_ham as f64 / n,
        corr(sxy, sx, sy, sxx, syy),
        corr(txy, tx, sx, txx, sxx)
    );
    v
}

fn load_samples(paths: &str, blend: f32) -> Vec<Sample> {
    let mut v = Vec::new();
    for (fi, p) in paths.split(',').filter(|s| !s.is_empty()).enumerate() {
        let buf = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
        for c in buf.chunks_exact(REC) {
            let o0 = u64::from_le_bytes(c[0..8].try_into().unwrap());
            let o1 = u64::from_le_bytes(c[8..16].try_into().unwrap());
            let turn = c[16] as usize;
            let res = c[17];
            let base = i16::from_le_bytes([c[18], c[19]]) as f32;
            let margin = c[20] as i8 as f32;
            let (own, foe) = if turn == 0 { (o0, o1) } else { (o1, o0) };
            let y_out = res as f32 * 0.5;
            let y_mar = 1.0 / (1.0 + (-margin / 4.0).exp());
            v.push(Sample {
                own,
                foe,
                base,
                y: (1.0 - blend) * y_out + blend * y_mar,
                pieces: (own | foe).count_ones(),
                game: ((fi as u64) << 24) | u32::from_le_bytes([c[21], c[22], c[23], 0]) as u64,
                out: y_out,
                cp: base,
                stm: turn as u8,
            });
        }
    }
    v
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn bce(p: f32, y: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}


#[derive(Clone, Copy)]
struct Ptrs {
    w1: *mut f32,
    b1: *mut f32,
    w2: *mut f32,
    b2: *mut f32,
    aw1: *mut f32,
    ab1: *mut f32,
    aw2: *mut f32,
    ab2: *mut f32,
}
unsafe impl Send for Ptrs {}
unsafe impl Sync for Ptrs {}

/// One thread's share of an epoch. Sparse first-layer rows are updated immediately (lock-free, Hogwild);
/// the dense parts (b1, w2, b2) accumulate gradients locally and flush every 16 samples (less cache-line ping-pong).
fn train_chunk(p: Ptrs, h: usize, chunk: &[u32], train: &[Sample], s_cp: f32, k: f32, lr: f32, done: &std::sync::atomic::AtomicUsize, ep: usize, t0: Instant) -> f64 {
    let eps = 1e-8f32;
    let mut z = [0f32; 256];
    let mut a = [0f32; 256];
    let mut dz = [0f32; 256];
    let mut gw2 = vec![0f32; BUCKETS * h];
    let mut gb2 = [0f32; BUCKETS];
    let mut gb1 = vec![0f32; h];
    let mut f = [0u32; WIN];
    let mut loss = 0.0f64;
    let mut cnt = 0usize;
    unsafe {
        for (n, &idx) in chunk.iter().enumerate() {
            let x = &train[idx as usize];
            net3::features_from(x.own, x.foe, &mut f);
            let bk = net3::bucket_of(x.pieces);
            for j in 0..h {
                z[j] = *p.b1.add(j);
            }
            for &fi in f.iter() {
                let row = p.w1.add(fi as usize * h);
                for j in 0..h {
                    z[j] += *row.add(j);
                }
            }
            let mut o = *p.b2.add(bk);
            for j in 0..h {
                let c = z[j].clamp(0.0, ZMAX);
                a[j] = c * c;
                o += *p.w2.add(bk * h + j) * a[j];
            }
            let pr = sigmoid((x.base + OUT_CP * o) / s_cp);
            loss += bce(pr, x.y) as f64;
            let d_o = (pr - x.y) * k;
            gb2[bk] += d_o;
            for j in 0..h {
                let w = *p.w2.add(bk * h + j);
                dz[j] = if z[j] > 0.0 && z[j] < ZMAX { d_o * w * 2.0 * z[j] } else { 0.0 };
                gw2[bk * h + j] += d_o * a[j];
                gb1[j] += dz[j];
            }
            for &fi in f.iter() {
                let row = fi as usize * h;
                for j in 0..h {
                    let g = dz[j];
                    if g != 0.0 {
                        let ix = row + j;
                        let ac = p.aw1.add(ix);
                        *ac += g * g;
                        *p.w1.add(ix) -= lr * g / ((*ac).sqrt() + eps);
                    }
                }
            }
            cnt += 1;
            if cnt == 16 || n + 1 == chunk.len() {
                for b in 0..BUCKETS {
                    let g = gb2[b];
                    if g != 0.0 {
                        let ac = p.ab2.add(b);
                        *ac += g * g;
                        *p.b2.add(b) -= lr * g / ((*ac).sqrt() + eps);
                        gb2[b] = 0.0;
                    }
                }
                for ix in 0..BUCKETS * h {
                    let g = gw2[ix];
                    if g != 0.0 {
                        let ac = p.aw2.add(ix);
                        *ac += g * g;
                        *p.w2.add(ix) -= lr * g / ((*ac).sqrt() + eps);
                        gw2[ix] = 0.0;
                    }
                }
                for j in 0..h {
                    let g = gb1[j];
                    if g != 0.0 {
                        let ac = p.ab1.add(j);
                        *ac += g * g;
                        *p.b1.add(j) -= lr * g / ((*ac).sqrt() + eps);
                        gb1[j] = 0.0;
                    }
                }
                cnt = 0;
            }
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if d % 4_000_000 == 0 {
                println!("    ep {ep} {}M samples, {:.0}s", d / 1_000_000, t0.elapsed().as_secs_f32());
            }
        }
    }
    loss
}
pub fn rltrain(args: &[String]) -> std::process::ExitCode {
    let data = flag(args, "--data").unwrap_or("");
    let out = flag(args, "--out").unwrap_or("net.n3").to_string();
    let h: usize = flag_num(args, "--hidden", 64);
    let epochs: usize = flag_num(args, "--epochs", 4);
    let lr: f32 = flag_num(args, "--lr", 0.02);
    let val_frac: f32 = flag_num(args, "--val", 0.05);
    let blend: f32 = flag_num(args, "--blend", 0.0);
    let seed: u64 = flag_num(args, "--seed", 7);
    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let base_none = flag(args, "--base") == Some("none");
    let (train_v, val_v): (Vec<Sample>, Vec<Sample>) = if let Some(tp) = flag(args, "--tdat") {
        // Real dataset: v9 .tdat (hub `ti data pack`); val = the provided val_v9.tdat (separate file).
        let vp = flag(args, "--val-tdat").expect("--val-tdat required with --tdat");
        let ymode = flag(args, "--y").unwrap_or("outcome");
        let max_rows: usize = flag_num(args, "--max-rows", usize::MAX);
        let (flip_stm, flip_out) = detect_orientation(vp, 200_000);
        println!("orientation: flip_stm={flip_stm} flip_outcome={flip_out}; y={ymode} base={}", if base_none { "none" } else { "cp" });
        let t0 = Instant::now();
        let val_v = load_tdat(vp, ymode, base_none, flip_stm, flip_out, usize::MAX);
        let train_v = load_tdat(tp, ymode, base_none, flip_stm, flip_out, max_rows);
        println!("loaded {} train / {} val rows in {:.0}s", train_v.len(), val_v.len(), t0.elapsed().as_secs_f32());
        (train_v, val_v)
    } else if let Some(bp) = flag(args, "--bin") {
        // tstg_v10_31m.bin: one file, no val split provided -> contiguous tail block (+ gap) as validation.
        let ymode = flag(args, "--y").unwrap_or("outcome");
        let max_rows: usize = flag_num(args, "--max-rows", usize::MAX);
        let tail: f32 = flag_num(args, "--val-tail", 0.05);
        let t0 = Instant::now();
        let all = load_bin(bp, ymode, base_none, max_rows);
        let nval = ((all.len() as f32 * tail) as usize).max(1000).min(all.len() / 2);
        let cut = all.len() - nval;
        let gap = 5000.min(cut);
        let train_v = all[..cut - gap].to_vec();
        let val_v = all[cut..].to_vec();
        println!("loaded {} train / {} val rows (tail block, {} gap) in {:.0}s; y={ymode} base={}", train_v.len(), val_v.len(), gap, t0.elapsed().as_secs_f32(), if base_none { "none" } else { "cp" });
        (train_v, val_v)
    } else {
        let mut samples = load_samples(data, blend);
        if samples.is_empty() {
            eprintln!("rltrain: no samples (--data a.rec,b.rec)");
            return std::process::ExitCode::FAILURE;
        }
        for i in (1..samples.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            samples.swap(i, j);
        }
        // Split BY GAME (positions of one game share an outcome -> no leakage).
        let thresh = (val_frac * 1000.0) as u64;
        let mix = |g: u64| {
            let mut x = g.wrapping_add(0x9E3779B97F4A7C15);
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            (x ^ (x >> 31)) % 1000
        };
        let (val_v, train_v): (Vec<Sample>, Vec<Sample>) = samples.iter().partition(|s| mix(s.game) < thresh);
        (train_v, val_v)
    };
    let (val, train) = (&val_v[..], &train_v[..]);
    if val.is_empty() || train.is_empty() {
        eprintln!("rltrain: need more games for a game-level val split");
        return std::process::ExitCode::FAILURE;
    }
    println!("rltrain: {} train / {} val samples, hidden {h}, lr {lr}", train.len(), val.len());

    // Calibrate S (cp per logit) on base alone.
    let mut best_s = 300.0f32;
    let mut best_l = f32::MAX;
    // (calibrate on at most 2M rows; with --base none S is fixed at 400)
    let cal: &[Sample] = &train[..train.len().min(2_000_000)];
    let grid: &[f32] = if base_none { &[400.0] } else { &[100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 650.0, 800.0, 1200.0] };
    for &s in grid {
        let l: f32 = cal.iter().map(|x| bce(sigmoid(x.base / s), x.y)).sum::<f32>() / cal.len() as f32;
        if l < best_l {
            best_l = l;
            best_s = s;
        }
    }
    println!("calibrated S = {best_s} cp/logit, base-only train BCE {best_l:.4}");
    let s_cp = best_s;
    let k = OUT_CP / s_cp; // d logit / d o

    let mut net = match flag(args, "--init") {
        Some(p) => Net3::load(p).unwrap_or_else(|e| panic!("init {p}: {e}")),
        None => {
            let mut n = Net3::zeros(h);
            for v in n.w1.iter_mut() {
                *v = (rng.unit() - 0.5) * 0.04;
            }
            for v in n.b1.iter_mut() {
                *v = 0.5;
            }
            for v in n.w2.iter_mut() {
                *v = (rng.unit() - 0.5) * 0.1;
            }
            n
        }
    };
    let h = net.h;
    let mut a_w1 = vec![0f32; net.w1.len()];
    let mut a_b1 = vec![0f32; h];
    let mut a_w2 = vec![0f32; BUCKETS * h];
    let mut a_b2 = [0f32; BUCKETS];
    let eps = 1e-8f32;

    let eval = |net: &Net3, set: &[Sample]| -> (f32, f32, f32, f32) {
        let (mut lb, mut ln, mut ab, mut an) = (0.0, 0.0, 0.0, 0.0);
        let mut f = [0u32; WIN];
        for x in set {
            net3::features_from(x.own, x.foe, &mut f);
            let o = net.forward_raw(&f, net3::bucket_of(x.pieces));
            let pb = sigmoid(x.base / s_cp);
            let pn = sigmoid((x.base + OUT_CP * o) / s_cp);
            lb += bce(pb, x.y);
            ln += bce(pn, x.y);
            if x.y != 0.5 {
                ab += ((pb > 0.5) == (x.y > 0.5)) as u32 as f32;
                an += ((pn > 0.5) == (x.y > 0.5)) as u32 as f32;
            }
        }
        let n = set.len() as f32;
        (lb / n, ln / n, ab / n, an / n)
    };
    let (vb, vn, ab, an) = eval(&net, val);
    println!("epoch 0: val BCE base {vb:.4} net {vn:.4} | acc base {ab:.4} net {an:.4}");
    let mut best = vn;
    let _ = net.save(&out);

    let mut order: Vec<u32> = (0..train.len() as u32).collect();
    let nthreads: usize = flag_num(args, "--threads", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    println!("training with {nthreads} threads (Hogwild, dense layers micro-batched x16)");
    let t0 = Instant::now();
    for ep in 1..=epochs {
        for i in (1..order.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            order.swap(i, j);
        }
        let ptrs = Ptrs { w1: net.w1.as_mut_ptr(), b1: net.b1.as_mut_ptr(), w2: net.w2.as_mut_ptr(), b2: net.b2.as_mut_ptr(), aw1: a_w1.as_mut_ptr(), ab1: a_b1.as_mut_ptr(), aw2: a_w2.as_mut_ptr(), ab2: a_b2.as_mut_ptr() };
        let done = std::sync::atomic::AtomicUsize::new(0);
        let per = (order.len() + nthreads - 1) / nthreads;
        let mut loss = 0.0f64;
        let train_ref: &[Sample] = train;
        std::thread::scope(|sc| {
            let hs: Vec<_> = order
                .chunks(per)
                .map(|chunk| {
                    let done = &done;
                    sc.spawn(move || train_chunk(ptrs, h, chunk, train_ref, s_cp, k, lr, done, ep, t0))
                })
                .collect();
            for hnd in hs {
                loss += hnd.join().unwrap();
            }
        });
        let (vb, vn, ab, an) = eval(&net, val);
        println!(
            "epoch {ep}: train BCE {:.4} | val BCE base {vb:.4} net {vn:.4} | acc base {ab:.4} net {an:.4} | {:.0}s",
            loss / order.len() as f64,
            t0.elapsed().as_secs_f32()
        );
        if vn < best {
            best = vn;
            let _ = net.save(&out);
            println!("  saved {out} (val BCE {vn:.4})");
        }
    }
    report(&net, val, s_cp);
    println!("rltrain done: best val BCE {best:.4} (base {vb:.4})");
    std::process::ExitCode::SUCCESS
}

// ------------------------------------------------------- fidelity report

fn corr_of(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 { return 0.0; }
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    sxy / ((sxx * syy).sqrt() + 1e-12)
}

/// Distillation fidelity of the net (plus base) on a validation set, all rows and side-to-move==0 rows:
/// corr(p_net, target y), corr(p_net, teacher p=sigmoid(cp/400)), corr(100*o, cp), plus the teacher ceiling corr(p_teacher, y).
fn report(net: &Net3, set: &[Sample], s_cp: f32) {
    let mut f = [0u32; WIN];
    let mut cols: [Vec<f64>; 6] = Default::default(); // p_net, y, p_teacher, o*100, cp, stm0 flag
    for x in set {
        net3::features_from(x.own, x.foe, &mut f);
        let o = net.forward_raw(&f, net3::bucket_of(x.pieces));
        let pn = sigmoid((x.base + OUT_CP * o) / s_cp) as f64;
        cols[0].push(pn);
        cols[1].push(x.y as f64);
        cols[2].push(sigmoid(x.cp / 400.0) as f64);
        cols[3].push((x.base + OUT_CP * o) as f64);
        cols[4].push(x.cp as f64);
        cols[5].push(if x.stm == 0 { 1.0 } else { 0.0 });
    }
    for (name, only0) in [("all rows", false), ("stm=0 rows", true)] {
        let idx: Vec<usize> = (0..set.len()).filter(|&i| !only0 || cols[5][i] == 1.0).collect();
        let pick = |c: usize| -> Vec<f64> { idx.iter().map(|&i| cols[c][i]).collect() };
        println!(
            "FIDELITY [{name}, n={}]: corr(net, target) {:.4} | corr(net, teacher p) {:.4} | corr(net cp, cp) {:.4} | ceiling corr(teacher p, target) {:.4}",
            idx.len(),
            corr_of(&pick(0), &pick(1)),
            corr_of(&pick(0), &pick(2)),
            corr_of(&pick(3), &pick(4)),
            corr_of(&pick(2), &pick(1))
        );
    }
}

// ------------------------------------------------------------ rlbench

#[cfg(target_arch = "x86_64")]
mod bench {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fwd_i32<const N: usize>(w1: *const i16, f: &[u32; super::WIN], w2: *const f32, scale: f32) -> f32 {
        let mut acc = [_mm256_setzero_si256(); N];
        for &fi in f.iter() {
            let row = w1.add(fi as usize * N * 8);
            for j in 0..N {
                let v = _mm_loadu_si128(row.add(j * 8) as *const __m128i);
                acc[j] = _mm256_add_epi32(acc[j], _mm256_cvtepi16_epi32(v));
            }
        }
        let sc = _mm256_set1_ps(scale);
        let mut sum = _mm256_setzero_ps();
        for j in 0..N {
            let x = _mm256_max_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc[j]), sc), _mm256_setzero_ps());
            sum = _mm256_fmadd_ps(_mm256_mul_ps(x, x), _mm256_loadu_ps(w2.add(j * 8)), sum);
        }
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let s4 = _mm_add_ps(hi, lo);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
    }

    /// int16 accumulators (16 lanes per ymm): needs sum of 25 rows to fit i16 (true for small trained weights).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fwd_i16<const N: usize>(w1: *const i16, f: &[u32; super::WIN], w2: *const f32, scale: f32) -> f32 {
        // N = hidden/16
        let mut acc = [_mm256_setzero_si256(); N];
        for &fi in f.iter() {
            let row = w1.add(fi as usize * N * 16);
            for j in 0..N {
                acc[j] = _mm256_add_epi16(acc[j], _mm256_loadu_si256(row.add(j * 16) as *const __m256i));
            }
        }
        let sc = _mm256_set1_ps(scale);
        let mut sum = _mm256_setzero_ps();
        for j in 0..N {
            let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[j]));
            let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[j], 1));
            let xl = _mm256_max_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo), sc), _mm256_setzero_ps());
            let xh = _mm256_max_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi), sc), _mm256_setzero_ps());
            sum = _mm256_fmadd_ps(_mm256_mul_ps(xl, xl), _mm256_loadu_ps(w2.add(j * 16)), sum);
            sum = _mm256_fmadd_ps(_mm256_mul_ps(xh, xh), _mm256_loadu_ps(w2.add(j * 16 + 8)), sum);
        }
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let s4 = _mm_add_ps(hi, lo);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
    }
}

/// rlbench --tdat <tdat> --hidden 128 [--positions 300000]: ns per evaluation of the Jataxx-style net
/// (25 windows, int16 weights, AVX2) on REAL positions: feature extraction, int32-acc forward, int16-acc forward.
#[cfg(target_arch = "x86_64")]
pub fn rlbench(args: &[String]) -> std::process::ExitCode {
    let h: usize = flag_num(args, "--hidden", 128);
    let npos: usize = flag_num(args, "--positions", 300_000);
    let tp = flag(args, "--tdat").expect("--tdat required");
    let s = load_tdat(tp, "outcome", true, false, false, npos);
    let siblings = args.iter().any(|a| a == "--siblings");
    let t0 = Instant::now();
    let mut feats: Vec<[u32; WIN]> = Vec::new();
    let mut sink = 0u64;
    if siblings {
        // Search-like order: all children of one parent back-to-back (rows shared between siblings stay cache-hot).
        for x in s.iter().take(60_000) {
            let stm = x.stm as usize;
            let mut occ = [0u64; 2];
            occ[stm] = x.own;
            occ[1 - stm] = x.foe;
            let b = titanium::Board { occ, blockers: 0, turn: stm as u8, passes: 0, ply: 0, piece_cnt: [occ[0].count_ones() as u8, occ[1].count_ones() as u8], blocker_cnt: 0, hash: 0, pst: 0 };
            if b.game_over() { continue; }
            let ml = b.legal_moves();
            for i in 0..ml.len() {
                let c = b.make(ml.move_at(i));
                let t = c.turn as usize;
                let mut f = [0u32; WIN];
                net3::features_from(c.occ[t], c.occ[1 - t], &mut f);
                feats.push(f);
            }
        }
        println!("sibling mode: {} children of the first 60000 parents ({:.1} children/parent)", feats.len(), feats.len() as f64 / 60000.0);
        let t_f = Instant::now();
        for x in s.iter().take(5000) {
            let mut f = [0u32; WIN];
            net3::features_from(x.own, x.foe, &mut f);
            sink = sink.wrapping_add(f[0] as u64);
        }
        let _ = t_f;
    } else {
        feats = vec![[0u32; WIN]; s.len()];
        for _rep in 0..5 {
            for (i, x) in s.iter().enumerate() {
                net3::features_from(x.own, x.foe, &mut feats[i]);
                sink = sink.wrapping_add(feats[i][0] as u64);
            }
        }
    }
    let t_feat = if siblings { 49.0 } else { t0.elapsed().as_nanos() as f64 / (5.0 * s.len() as f64) };
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let w1: Vec<i16> = (0..WIN * net3::PAT * h).map(|_| ((rng.next() % 401) as i32 - 200) as i16).collect();
    let w2: Vec<f32> = (0..h).map(|_| rng.unit() - 0.5).collect();
    let reps = 5;
    let mut acc = 0f32;
    let t1 = Instant::now();
    for _ in 0..reps {
        for f in feats.iter() {
            acc += unsafe {
                match h {
                    64 => bench::fwd_i32::<8>(w1.as_ptr(), f, w2.as_ptr(), 1.0 / 8192.0),
                    128 => bench::fwd_i32::<16>(w1.as_ptr(), f, w2.as_ptr(), 1.0 / 8192.0),
                    192 => bench::fwd_i32::<24>(w1.as_ptr(), f, w2.as_ptr(), 1.0 / 8192.0),
                    256 => bench::fwd_i32::<32>(w1.as_ptr(), f, w2.as_ptr(), 1.0 / 8192.0),
                    _ => panic!("--hidden must be 64,128,192 or 256"),
                }
            };
        }
    }
    let t_i32 = t1.elapsed().as_nanos() as f64 / (reps as f64 * feats.len() as f64);
    let t2 = Instant::now();
    for _ in 0..reps {
        for f in feats.iter() {
            acc += unsafe {
                match h {
                    64 => bench::fwd_i16::<4>(w1.as_ptr(), f, w2.as_ptr(), 1.0 / 8192.0),
                    128 => bench::fwd_i16::<8>(w1.as_ptr(), f, w2.as_ptr(), 1.0 / 8192.0),
                    192 => bench::fwd_i16::<12>(w1.as_ptr(), f, w2.as_ptr(), 1.0 / 8192.0),
                    _ => bench::fwd_i16::<16>(w1.as_ptr(), f, w2.as_ptr(), 1.0 / 8192.0),
                }
            };
        }
    }
    let t_i16 = t2.elapsed().as_nanos() as f64 / (reps as f64 * feats.len() as f64);
    println!("rlbench: hidden {h}, {} real positions, weights table {:.0} MB (random int16, not trained)", feats.len(), (w1.len() * 2) as f64 / 1e6);
    println!("  feature extraction (25 windows, pext+table): {t_feat:.0} ns/eval");
    println!("  forward int32 accumulators (Jataxx style):  {t_i32:.0} ns/eval   => total {:.0} ns", t_feat + t_i32);
    println!("  forward int16 accumulators:                 {t_i16:.0} ns/eval   => total {:.0} ns", t_feat + t_i16);
    println!("  champion reference (KEY_FACTS D426): 5000 ns/eval (208 rows x 256 wide, 0.53 us extraction)");
    println!("  (checksum {acc} {sink})");
    std::process::ExitCode::SUCCESS
}
// ----------------------------------------------------- sibling agreement

/// rlsib gen --tdat F --parents N --out boards.txt --groups groups.txt
/// rlsib cmp --net n.n3 --boards boards.txt --groups groups.txt --champ bits.txt
/// Champion values come from `titanium-cli nnevaldump --nnue nnue_main.tnn --boards boards.txt` (f32 bits, side-to-move relative).
pub fn rlsib(args: &[String]) -> std::process::ExitCode {
    let sub = args.get(1).map(String::as_str).unwrap_or("");
    if sub == "gen" {
        let tp = flag(args, "--tdat").expect("--tdat");
        let np: usize = flag_num(args, "--parents", 3000);
        let every: usize = flag_num(args, "--every", 40);
        let s = load_tdat(tp, "outcome", true, false, false, np * every + 10);
        let mut boards = String::new();
        let mut groups = String::new();
        let mut used = 0usize;
        for x in s.iter().step_by(every) {
            if used >= np { break; }
            let stm = x.stm as usize;
            let mut occ = [0u64; 2];
            occ[stm] = x.own;
            occ[1 - stm] = x.foe;
            let b = titanium::Board { occ, blockers: 0, turn: stm as u8, passes: 0, ply: 0, piece_cnt: [occ[0].count_ones() as u8, occ[1].count_ones() as u8], blocker_cnt: 0, hash: 0, pst: 0 };
            if b.game_over() { continue; }
            let ml = b.legal_moves();
            if ml.len() < 2 { continue; }
            // dedupe identical children (clones to the same square give the same board)
            let mut seen: Vec<String> = Vec::new();
            for i in 0..ml.len() {
                let c = b.make(ml.move_at(i)).to_string();
                if !seen.contains(&c) { seen.push(c); }
            }
            if seen.len() < 2 { continue; }
            for c in &seen { boards.push_str(c); boards.push('\n'); }
            groups.push_str(&format!("{}\n", seen.len()));
            used += 1;
        }
        std::fs::write(flag(args, "--out").unwrap_or("sib_boards.txt"), boards).unwrap();
        std::fs::write(flag(args, "--groups").unwrap_or("sib_groups.txt"), groups).unwrap();
        println!("rlsib gen: {used} parents written");
        return std::process::ExitCode::SUCCESS;
    }
    // cmp
    let net = Net3::load(flag(args, "--net").expect("--net")).expect("net");
    let boards = std::fs::read_to_string(flag(args, "--boards").expect("--boards")).unwrap();
    let groups: Vec<usize> = std::fs::read_to_string(flag(args, "--groups").expect("--groups")).unwrap().lines().filter_map(|l| l.trim().parse().ok()).collect();
    let champ: Vec<f32> = std::fs::read_to_string(flag(args, "--champ").expect("--champ")).unwrap().lines().filter_map(|l| l.trim().parse::<u32>().ok()).map(f32::from_bits).collect();
    let bl: Vec<titanium::Board> = boards.lines().filter_map(|l| titanium::Board::from_str(l.trim())).collect();
    assert_eq!(bl.len(), champ.len(), "boards {} vs champion values {}", bl.len(), champ.len());
    let mut f = [0u32; WIN];
    let net_v: Vec<f32> = bl.iter().map(|c| {
        let t = c.turn as usize;
        net3::features_from(c.occ[t], c.occ[1 - t], &mut f);
        net.forward_raw(&f, net3::bucket_of((c.occ[0] | c.occ[1]).count_ones()))
    }).collect();
    let (mut n, mut top1, mut top3, mut rnd, mut regret) = (0usize, 0usize, 0usize, 0.0f64, 0.0f64);
    let mut at = 0usize;
    for g in groups {
        // mover picks the child with the LOWEST side-to-move value at the child
        let cs = &champ[at..at + g];
        let ns = &net_v[at..at + g];
        at += g;
        let best = (0..g).min_by(|&a, &b| cs[a].partial_cmp(&cs[b]).unwrap()).unwrap();
        let pick = (0..g).min_by(|&a, &b| ns[a].partial_cmp(&ns[b]).unwrap()).unwrap();
        let mut order: Vec<usize> = (0..g).collect();
        order.sort_by(|&a, &b| ns[a].partial_cmp(&ns[b]).unwrap());
        n += 1;
        if pick == best { top1 += 1; }
        if order.iter().take(3).any(|&i| i == best) { top3 += 1; }
        rnd += 1.0 / g as f64;
        regret += 400.0 * (cs[pick] - cs[best]) as f64; // cp approximation: champion 'y' units x 400
    }
    let xs: Vec<f64> = champ.iter().map(|&v| v as f64).collect();
    let ys: Vec<f64> = net_v.iter().map(|&v| v as f64).collect();
    println!("SIBLING [{} parents, {} children]: top1 agreement {:.1}% (random {:.1}%) | top3 contains champion best {:.1}% | mean regret {:.1} cp | corr(net, champion over children) {:.3}",
        n, bl.len(), 100.0 * top1 as f64 / n as f64, 100.0 * rnd / n as f64, 100.0 * top3 as f64 / n as f64, regret / n as f64, corr_of(&xs, &ys));
    std::process::ExitCode::SUCCESS
}