use std::time::Instant;
use titanium::board::{Board, MoveList};
use titanium::nnfeat;
use titanium::nnue::Nnue;

fn main() {
    let net = Nnue::bundled_static();
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut boards = Vec::new();
    let mut list = MoveList::new();
    while boards.len() < 20000 {
        let mut b = Board::start();
        for _ in 0..60 {
            b.legal_moves_dedup_into(&mut list);
            if list.len() == 0 {
                break;
            }
            let i = (next() % list.len() as u64) as usize;
            b = b.make(list.move_at(i));
            if b.ply > 6 {
                boards.push(b);
            }
        }
    }
    boards.truncate(20000);
    // parent -> child feature diff size (same perspective), over every legal move of 2000 parents
    let (mut tot_n, mut tot_d, mut cnt) = (0usize, 0usize, 0usize);
    for b in boards.iter().take(2000) {
        b.legal_moves_dedup_into(&mut list);
        for i in 0..list.len() {
            let c = b.make(list.move_at(i));
            for side in 0..2 {
                let mut pa = [0u32; nnfeat::MAX_ACTIVE];
                let mut ca = [0u32; nnfeat::MAX_ACTIVE];
                let pn = nnfeat::features(b, side, net.groups, &mut pa);
                let cn = nnfeat::features(&c, side, net.groups, &mut ca);
                let ps: std::collections::HashSet<u32> = pa[..pn].iter().copied().collect();
                let cs: std::collections::HashSet<u32> = ca[..cn].iter().copied().collect();
                tot_d += ps.symmetric_difference(&cs).count();
                tot_n += cn;
                cnt += 1;
            }
        }
    }
    println!("diff: avg active {:.1}, avg add+remove rows {:.1} ({:.0}% of a full rebuild's {:.1} adds)", tot_n as f64 / cnt as f64, tot_d as f64 / cnt as f64, 100.0 * (tot_d as f64 / cnt as f64) / (tot_n as f64 / cnt as f64), tot_n as f64 / cnt as f64);
    let reps = 20;
    for (name, g) in [("none", 0u32), ("G1", nnfeat::G1), ("G2", nnfeat::G2), ("G3", nnfeat::G3), ("G5", nnfeat::G5), ("G6", nnfeat::G6)] {
        let t = Instant::now();
        let mut s = 0usize;
        for _ in 0..reps {
            for b in &boards {
                let (_, nu, _, nt) = nnfeat::both(b, g);
                s += nu + nt;
            }
        }
        println!("  group {name}: {:.2} us/eval ({s})", t.elapsed().as_secs_f64() / (reps * boards.len()) as f64 * 1e6);
    }
    let t = Instant::now();
    let mut s = 0usize;
    for _ in 0..reps {
        for b in &boards {
            let (_, nu, _, nt) = nnfeat::both(b, net.groups);
            s += nu + nt;
        }
    }
    let n = (reps * boards.len()) as f64;
    let f = t.elapsed().as_secs_f64() / n * 1e6;
    println!("features(both)   {:.2} us/eval  (active sum {})", f, s);
    let t = Instant::now();
    let mut s2 = 0i64;
    for _ in 0..reps {
        for b in &boards {
            s2 += net.bench_ft(b) as i64;
        }
    }
    let ft = t.elapsed().as_secs_f64() / n * 1e6;
    println!("features+FT sum  {:.2} us/eval  ({})", ft, s2);
    let t = Instant::now();
    let mut s3 = 0i64;
    for _ in 0..reps {
        for b in &boards {
            let (e, d) = net.eval_with_dis(b, 0);
            s3 += (e + d) as i64;
        }
    }
    let full = t.elapsed().as_secs_f64() / n * 1e6;
    println!("full eval        {:.2} us/eval  ({})", full, s3);
    println!("=> features {:.2} | FT sum {:.2} | head {:.2} us", f, ft - f, full - ft);
}
