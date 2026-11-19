//! Standalone reader for a Titanium game file (`.tgam`, version 1). Standard library only: no engine, no hub.
//! It proves that docs/DATASET_GUIDE.md is enough to parse the format.
//!
//!   rustc -O --edition 2021 read_tgam.rs      (on this PC add: --target x86_64-pc-windows-gnu, llvm-mingw on PATH)
//!   read_tgam C:\gitProjects\titanium-ataxx\data\datasets\v10\gate_games.tgam [games to print]
//!
//! Board: 49 squares, square = row * 7 + column, row 0 = rank 7, so square 0 = a7 and square 48 = g1.
//! A position is two 64-bit sets (stones of side 0 = x, stones of side 1 = o) plus the blocked squares.
use std::io::Read;

fn u16le(b: &[u8]) -> u16 { u16::from_le_bytes([b[0], b[1]]) }
fn u32le(b: &[u8]) -> u32 { u32::from_le_bytes(b[0..4].try_into().unwrap()) }
fn u64le(b: &[u8]) -> u64 { u64::from_le_bytes(b[0..8].try_into().unwrap()) }

/// Squares at distance `d` (king-move distance) from `sq`.
fn ring(sq: usize, d: i32) -> u64 {
    let (x, y) = ((sq % 7) as i32, (sq / 7) as i32);
    let mut m = 0u64;
    for t in 0..49i32 {
        if ((t % 7) - x).abs().max(((t / 7) - y).abs()) == d {
            m |= 1 << t;
        }
    }
    m
}

fn name(sq: usize) -> String { format!("{}{}", (b'a' + (sq % 7) as u8) as char, 7 - sq / 7) }

/// 10-bit move code -> (from, to). Codes 1..=49: clone to square code-1 (from = lowest own neighbour).
/// Codes 50..=529: the jumps, numbered in order of (from, to) over all pairs at distance 2. 1023 = pass.
fn decode(code: u16, jumps: &[(usize, usize)], me: u64, empty: u64) -> Option<(usize, usize)> {
    match code {
        1..=49 => {
            let to = (code - 1) as usize;
            let src = me & ring(to, 1);
            (empty >> to & 1 == 1 && src != 0).then(|| (src.trailing_zeros() as usize, to))
        }
        50..=529 => {
            let (f, t) = jumps[(code - 50) as usize];
            (me >> f & 1 == 1 && empty >> t & 1 == 1).then_some((f, t))
        }
        _ => None,
    }
}

/// 16-bit score code -> text. Scores are from the view of the side that moves.
fn score(c: i16) -> String {
    match c as i32 {
        -32768 => "no score".into(),
        c if c > 30000 => format!("win in {} plies", c - 30001),
        c if c < -30000 => format!("loss in {} plies", -c - 30001),
        c => format!("{c} cp"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let show: u64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut f = std::io::BufReader::new(std::fs::File::open(&a[1]).expect("open"));
    let mut jumps = Vec::new();
    for from in 0..49 {
        for to in 0..49 {
            if ring(from, 2) >> to & 1 == 1 {
                jumps.push((from, to));
            }
        }
    }
    assert_eq!(jumps.len(), 480);

    // File header, 32 bytes.
    let mut h = [0u8; 32];
    f.read_exact(&mut h).unwrap();
    assert_eq!(&h[0..4], b"TGAM");
    assert_eq!(u32le(&h[4..]), 1, "version");
    let (games, plies, sources) = (u64le(&h[8..]), u64le(&h[16..]), u32le(&h[24..]));
    println!("{games} games, {plies} plies, {sources} sources");

    let (mut seen_plies, mut illegal, mut passes, mut wins0, mut draws, mut unknown) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    for g in 0..games {
        // Game header, 40 bytes.
        let mut gh = [0u8; 40];
        f.read_exact(&mut gh).unwrap();
        let mut occ = [u64le(&gh[0..]), u64le(&gh[8..])];
        let blockers = u64le(&gh[16..]);
        let n = u16le(&gh[24..]) as usize;
        let source = u16le(&gh[26..]);
        let mut turn = (gh[28] & 1) as usize; // side to move in the start position
        let cand_seat = gh[28] >> 1 & 3; // 0 / 1 = seat of the tested engine in a gate game, 2 = unknown
        let forfeit = gh[28] >> 3 & 1 == 1;
        let (final0, final1, result) = (gh[29], gh[30], gh[31]); // result for side 0: 2 won, 1 draw, 0 lost, 3 unknown
        match result { 2 => wins0 += 1, 1 => draws += 1, 3 => unknown += 1, _ => {} }
        if g < show {
            println!("game {g}: source {source}, book line {}, pair {}, candidate seat {cand_seat}, forfeit {forfeit}, final {final0}-{final1}, result {result}",
                u32le(&gh[32..]), u32le(&gh[36..]));
        }
        // Then 4 bytes per ply: u16 (move code in the low 10 bits, search depth in the high 6), i16 score code.
        let mut pl = vec![0u8; n * 4];
        f.read_exact(&mut pl).unwrap();
        for p in 0..n {
            let w = u16le(&pl[p * 4..]);
            let (code, depth) = (w & 0x3FF, w >> 10);
            let sc = i16::from_le_bytes([pl[p * 4 + 2], pl[p * 4 + 3]]);
            seen_plies += 1;
            if code == 0x3FF {
                passes += 1;
                turn ^= 1;
                continue;
            }
            let empty = !(occ[0] | occ[1] | blockers) & ((1u64 << 49) - 1);
            let Some((from, to)) = decode(code, &jumps, occ[turn], empty) else {
                illegal += 1;
                break;
            };
            if g < show && p < 6 {
                println!("  ply {p}: side {turn} plays {}{} (depth {depth}, {})", name(from), name(to), score(sc));
            }
            // Apply: a jump leaves its square, a clone does not; every enemy stone next to `to` changes side.
            if ring(from, 2) >> to & 1 == 1 {
                occ[turn] &= !(1 << from);
            }
            occ[turn] |= 1 << to;
            let flip = occ[turn ^ 1] & ring(to, 1);
            occ[turn] |= flip;
            occ[turn ^ 1] &= !flip;
            turn ^= 1;
        }
    }
    println!("read {seen_plies} plies: illegal moves {illegal}, passes {passes}; side 0 won {wins0}, draws {draws}, result unknown {unknown}");
    assert_eq!(seen_plies + if illegal > 0 { u64::MAX / 2 } else { 0 }, plies, "ply count must match the header");
}
