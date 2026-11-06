//! D425 cost split of the champion eval per position (read-only microbench, single thread, black_box'd).
//! Times on val_v9_clean positions: legal move generation (reference), nnfeat extraction per input group, full extraction,
//! the full eval (extraction + accumulator refresh + L1/L2), and the active-feature counts that drive the accumulator cost.
//!
//!   nnue_cost_split [n_positions] [repeats]
use std::hint::black_box;
use std::io::Read;
use std::time::Instant;
use titanium::board::Board;
use titanium::nnfeat::{self, ALL_GROUPS, G1, G2, G3, G5, G6};

fn main() {
    if std::env::args().any(|a| a == "--ft-i8") { titanium::nnue::set_ft_i8(true); }
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let mut f = std::fs::File::open("C:/gitProjects/titanium-ataxx/data/nnue/packed/val_v9_clean.tdat").expect("val");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let stride = (buf.len() / 48 / n).max(1);
    let boards: Vec<Board> = buf
        .chunks_exact(48)
        .step_by(stride)
        .map(|b| Board::from_occ(u64::from_le_bytes(b[0..8].try_into().unwrap()), u64::from_le_bytes(b[8..16].try_into().unwrap()), 0, b[40]))
        .collect();
    // optional 3rd arg: a net file (D431 pruned-width speed check); default = bundled champion
    let owned = std::env::args().nth(3).filter(|a| !a.starts_with("--")).map(|p| titanium::nnue::Nnue::load(&p).expect("load net"));
    let nn: &titanium::nnue::Nnue = match &owned { Some(n) => n, None => titanium::nnue::Nnue::bundled_static() };
    let per = |label: &str, f: &dyn Fn(&Board) -> u64| {
        let mut best = f64::MAX;
        let mut sink = 0u64;
        for _ in 0..reps {
            let t = Instant::now();
            for b in &boards {
                sink = sink.wrapping_add(f(black_box(b)));
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        black_box(sink);
        let ns = best * 1e9 / boards.len() as f64;
        println!("{label:<52} {ns:>8.0} ns/position");
        ns
    };
    println!("{} positions, best of {reps} repeats, width {} l1 {} l2 {} groups {:#x}", boards.len(), nn.width, nn.l1, nn.l2, nn.groups);
    let mg = per("legal move generation (dedup), reference", &|b| {
        let mut l = titanium::board::MoveList::new();
        b.legal_moves_dedup_into(&mut l);
        l.iter().count() as u64
    });
    for (name, g) in [("G1 square state", G1), ("G2 anchor", G2), ("G3 stone ring", G3), ("G5 2-ply exchange", G5), ("G6 global", G6)] {
        per(&format!("extract {name}"), &|b| {
            let (_, nu, _, nt) = nnfeat::both(b, g);
            (nu + nt) as u64
        });
    }
    let ex = per("extract ALL groups (both perspectives)", &|b| {
        let (_, nu, _, nt) = nnfeat::both(b, ALL_GROUPS);
        (nu + nt) as u64
    });
    let ev = per("FULL eval (extract + accumulator refresh + L1/L2)", &|b| nn.eval(b, 0) as u64);
    let mut act = 0usize;
    for b in &boards {
        let (_, nu, _, nt) = nnfeat::both(b, nn.groups);
        act += nu + nt;
    }
    let act = act as f64 / boards.len() as f64;
    println!("active features per position (both perspectives): {act:.1} -> accumulator refresh = {act:.0} rows x {} i16 lanes", nn.width);
    println!("split: extraction {:.0}% | accumulate + layers {:.0}% of the eval; eval = {:.1}x a move generation", 100.0 * ex / ev, 100.0 * (ev - ex) / ev, ev / mg);
    // incremental potential: per colour perspective, how many features differ between a position and each child
    // (parent "us" = child "them", same colour), split by group
    if std::env::args().nth(4).as_deref() != Some("inc") { return; }
    let groups = [("G1", G1), ("G2", G2), ("G3", G3), ("G5", G5), ("G6", G6)];
    let mut tot = [0f64; 6];
    let mut kids = 0f64;
    for b in boards.iter().take(3000) {
        let mut l = titanium::board::MoveList::new();
        b.legal_moves_dedup_into(&mut l);
        for m in l.iter() {
            let c = b.make(m);
            kids += 1.0;
            for (gi, (_, g)) in groups.iter().enumerate() {
                let (pu, npu, pt, npt) = nnfeat::both(b, *g);
                let (cu, ncu, ct, nct) = nnfeat::both(&c, *g);
                let diff = |a: &[u32], b2: &[u32]| {
                    let sa: std::collections::HashSet<u32> = a.iter().copied().collect();
                    let sb: std::collections::HashSet<u32> = b2.iter().copied().collect();
                    (sa.symmetric_difference(&sb).count()) as f64
                };
                let d = diff(&pu[..npu], &ct[..nct]) + diff(&pt[..npt], &cu[..ncu]);
                tot[gi] += d;
                tot[5] += d;
            }
        }
    }
    println!("\nINCREMENTAL potential over {} parent->child moves (rows to add+remove per eval, both perspectives):", kids as u64);
    for (gi, (name, _)) in groups.iter().enumerate() {
        println!("  {name}: {:.1}", tot[gi] / kids);
    }
    println!("  ALL: {:.1} changed rows vs {act:.1} rows for a full refresh", tot[5] / kids);
}
