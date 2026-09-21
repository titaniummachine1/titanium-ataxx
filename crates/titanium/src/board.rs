//! Bitboard board representation and move generation for 7x7 Ataxx.
//!
//! The 49 squares fit in a single u64: bit `i` is square `rank * 7 + file`,
//! rank 0 is the top row, file 0 is the left column.
//!
//! Move geometry: 8 compass directions only. Distance 1 = clone, distance 2 =
//! jump. Knight-like offsets (dr,df) = (1,2) are NOT legal moves.

pub const SIZE: usize = 7;
pub const SQUARES: usize = 49;
pub const FULL: u64 = (1u64 << SQUARES) - 1;

pub const EMPTY: u8 = 0;
pub const BLACK: u8 = 1;
pub const WHITE: u8 = 2;
pub const BLOCKER: u8 = 3;

/// Upper bound on legal moves in a 7x7 position. Real positions peak around
/// ~160; padded for safety (guarded by debug_assert in `push`).
pub const MAX_MOVES: usize = 256;

/// Bit for a square index.
#[inline]
pub const fn bit_of(sq: u8) -> u64 {
    1u64 << sq
}

const fn file_bits(f: usize) -> u64 {
    let mut m = 0u64;
    let mut r = 0usize;
    while r < SIZE {
        m |= 1u64 << (r * SIZE + f);
        r += 1;
    }
    m
}

/// File masks for shift wrap protection.
pub const FILE_A: u64 = file_bits(0);
pub const FILE_B: u64 = file_bits(1);
pub const FILE_F: u64 = file_bits(5);
pub const FILE_G: u64 = file_bits(6);

/// Deterministic const-time xorshift for zobrist key generation.
const fn zobrist_next(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

const ZOB_SEED: u64 = 0x9E3779B97F4A7C15;

/// `[color][sq]` zobrist keys, generated at compile time (deterministic).
const fn build_zob_piece() -> [[u64; SQUARES]; 2] {
    let mut t = [[0u64; SQUARES]; 2];
    let mut s = ZOB_SEED;
    let mut c = 0usize;
    while c < 2 {
        let mut sq = 0usize;
        while sq < SQUARES {
            s = zobrist_next(s);
            t[c][sq] = s;
            sq += 1;
        }
        c += 1;
    }
    t
}

pub static ZOB_PIECE: [[u64; SQUARES]; 2] = build_zob_piece();

/// Side-to-move key (black to move = no key; white to move XORed in).
pub static ZOB_SIDE: u64 = {
    let mut s = ZOB_SEED;
    let mut i = 0;
    while i < 2 * SQUARES {
        s = zobrist_next(s);
        i += 1;
    }
    zobrist_next(s)
};

/// Per-square blocker keys (blockers never change mid-game; folded in once).
pub static ZOB_BLOCK: [u64; SQUARES] = {
    let mut t = [0u64; SQUARES];
    let mut s = ZOB_SEED;
    let mut i = 0;
    while i < 2 * SQUARES + 1 {
        s = zobrist_next(s);
        if i < SQUARES {
            t[i] = s;
        }
        i += 1;
    }
    t
};

/// Union of all squares reached by shifting `bb` by exactly `d` steps along
/// the 8 compass directions, with file-wrap masked out. For d=1 this is the
/// full Chebyshev-1 ring; for d=2 it is only the straight/diagonal subset.
#[inline]
pub fn dist_union(bb: u64, d: u32) -> u64 {
    let na = bb & !FILE_A;
    let ng = bb & !FILE_G;
    ((bb >> (7 * d))      // up
        | (bb << (7 * d))      // down
        | (na >> d)            // left
        | (ng << d)            // right
        | (na >> (8 * d))      // up-left
        | (ng >> (6 * d))      // up-right
        | (na << (6 * d))      // down-left
        | (ng << (8 * d)))     // down-right
        & FULL // down shifts spill past bit 48; stays 49-bit clean
}

/// Full Chebyshev-2 ring (16 cells): straight + diagonal 2-steps plus the
/// knight offsets. Ataxx jumps reach every square of the 5x5 neighborhood
/// outside the 3x3 center. Off-board spill is removed here: left-shifted
/// bits (>= 49) would otherwise wrap back onto the board through the
/// knight shifts (e.g. 49 >> 1 = 48).
#[inline]
pub fn jump_union(bb: u64) -> u64 {
    let up = bb >> 14;
    let dn = (bb << 14) & FULL;
    let lf = (bb & !(FILE_A | FILE_B)) >> 2;
    let rt = (bb & !(FILE_F | FILE_G)) << 2;
    let ul = (bb & !(FILE_A | FILE_B)) >> 16;
    let ur = (bb & !(FILE_F | FILE_G)) >> 12;
    let dl = (bb & !(FILE_A | FILE_B)) << 12;
    let drr = (bb & !(FILE_F | FILE_G)) << 16;
    let v2 = (up | dn) & FULL;
    let h2 = lf | rt;
    let n1 = ((v2 & !FILE_A) >> 1) | ((v2 & !FILE_G) << 1); // (±2,±1)
    let n2 = (h2 >> 7) | ((h2 << 7) & FULL);                // (±1,±2)
    (up | dn | lf | rt | ul | ur | dl | drr | n1 | n2) & FULL
}

/// Exactly-Chebyshev-distance-d neighbor masks, computed at compile time.
/// d=1: clone ring / infection zone (8 cells), d=2: jump ring (16 cells,
/// knight-like offsets ARE legal Ataxx jumps).
const fn build_ring(dist: u32) -> [u64; SQUARES] {
    let mut table = [0u64; SQUARES];
    let mut sq = 0usize;
    while sq < SQUARES {
        let r = (sq / SIZE) as i32;
        let f = (sq % SIZE) as i32;
        let mut dr = -(dist as i32);
        while dr <= dist as i32 {
            let mut df = -(dist as i32);
            while df <= dist as i32 {
                // Ring cells only: max(|dr|, |df|) == dist.
                if dr.abs() == dist as i32 || df.abs() == dist as i32 {
                    let nr = r + dr;
                    let nf = f + df;
                    if nr >= 0 && nr < SIZE as i32 && nf >= 0 && nf < SIZE as i32 {
                        table[sq] |= 1u64 << ((nr as usize) * SIZE + nf as usize);
                    }
                }
                df += 1;
            }
            dr += 1;
        }
        sq += 1;
    }
    table
}

/// Squares at Chebyshev distance 1 (clone targets / infection zone).
pub static RING1: [u64; SQUARES] = build_ring(1);
/// Squares at Chebyshev distance 2 (jump targets, 5x5 ring).
pub static RING2: [u64; SQUARES] = build_ring(2);
/// `RING1 | RING2` per square: everywhere a piece can move from/to.
pub static REACH: [u64; SQUARES] = {
    let mut t = [0u64; SQUARES];
    let mut i = 0;
    while i < SQUARES {
        t[i] = RING1[i] | RING2[i];
        i += 1;
    }
    t
};

/// Ring-1 neighbor square indices per square (0xFF = absent), plus the count.
/// This is the extraction layout for the 8-bit infection LUT key: neighbor
/// `i` of a square corresponds to bit `i` of the extracted pattern.
const NEIGH_TABLES: ([[u8; 8]; SQUARES], [u8; SQUARES]) = {
    let mut pos = [[0xFFu8; 8]; SQUARES];
    let mut cnt = [0u8; SQUARES];
    let mut sq = 0usize;
    while sq < SQUARES {
        let r = (sq / SIZE) as i32;
        let f = (sq % SIZE) as i32;
        let mut dr = -1i32;
        while dr <= 1 {
            let mut df = -1i32;
            while df <= 1 {
                if !(dr == 0 && df == 0) {
                    let nr = r + dr;
                    let nf = f + df;
                    if nr >= 0 && nr < SIZE as i32 && nf >= 0 && nf < SIZE as i32 {
                        pos[sq][cnt[sq] as usize] = ((nr as usize) * SIZE + nf as usize) as u8;
                        cnt[sq] += 1;
                    }
                }
                df += 1;
            }
            dr += 1;
        }
        sq += 1;
    }
    (pos, cnt)
};

pub const NEIGH_POS: [[u8; 8]; SQUARES] = NEIGH_TABLES.0;
pub const NEIGH_CNT: [u8; SQUARES] = NEIGH_TABLES.1;

/// Single universal conversion table: input is the 8-bit enemy pattern
/// around a landing square (bit i = NEIGH_POS[sq][i] holds an enemy), output
/// is which neighbor slots convert. Square-independent — infection depends
/// only on the local pattern, so ONE table serves all 49 landing squares.
/// Under standard rules every adjacent enemy converts, which makes this the
/// identity map; it stays a table so rule variants (support-required
/// infection, shielding blockers, ...) change data, not the make() path.
/// bits below `NEIGH_CNT[sq]` so off-board is a permanently-empty padding
/// slot, and materialization stops at `NEIGH_CNT` — edge squares need no
/// special casing anywhere.
pub static CONVERT8: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut pat = 0usize;
    while pat < 256 {
        t[pat] = pat as u8;
        pat += 1;
    }
    t
};

/// Enemy pieces converted when landing on `sq`: compact-key LUT + local
/// materialization through `NEIGH_POS`. Bit-identical to `infect_direct`.
#[inline]
pub fn infect_via_lut(opp: u64, sq: u8) -> u64 {
    let s = sq as usize;
    let flips = CONVERT8[extract8(opp, sq) as usize];
    let mut mask = 0u64;
    let mut i = 0usize;
    while i < NEIGH_CNT[s] as usize {
        if flips & (1 << i) != 0 {
            mask |= 1u64 << NEIGH_POS[s][i];
        }
        i += 1;
    }
    mask
}

/// Direct bitboard computation (reference / fast path): one AND.
#[inline]
pub fn infect_direct(opp: u64, sq: u8) -> u64 {
    opp & RING1[sq as usize]
}

/// Sum over `bb`'s set bits of `popcount(REACH[sq] & empty)` — the total
/// (from, to) move-pair count for the pieces `bb` on the empty squares
/// `empty`. One table load + AND + POPCNT per piece.
#[inline]
fn count_pairs(bb: u64, empty: u64) -> u64 {
    let mut n = 0u64;
    let mut ps = bb;
    while ps != 0 {
        let sq = ps.trailing_zeros() as usize;
        ps &= ps - 1;
        n += (REACH[sq] & empty).count_ones() as u64;
    }
    n
}

/// Raw bitboard perft — the movement-generator stress test.
///
/// No `Board` copies, no move lists:
/// - depth-1 nodes are bulk-counted with `popcount` (no per-move stores),
/// - clone moves to the same destination are folded into ONE recursion with a
///   multiplicity (all ring-1 sources yield the identical child position),
/// - jumps recurse on raw u64s with only the vacated source differing.
///
/// Counts every `(from, to)` move pair — exactly what the `Board`-based
/// perft counts (enforced by a unit test), just without materializing moves.
pub fn perft_bb(me: u64, opp: u64, block: u64, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let empty = !(me | opp | block) & FULL;
    let targets = (dist_union(me, 1) | jump_union(me)) & empty;

    if depth == 1 {
        // Bulk-count (from, to) pairs with no move materialization. The
        // bipartite (piece, target) relation is symmetric, so iterate
        // whichever side has fewer set bits (pieces early game, targets late).
        let mut n = 0u64;
        if me.count_ones() <= targets.count_ones() {
            n += count_pairs(me, empty);
        } else {
            let mut ts = targets;
            while ts != 0 {
                let to = ts.trailing_zeros() as usize;
                ts &= ts - 1;
                n += (me & REACH[to]).count_ones() as u64;
            }
        }
        return n;
    }

    if depth == 2 {
        // Bulk-count grandchildren directly: skips the per-child call plus
        // the child's own targets recomputation (jump_union/dist_union).
        let mut total = 0u64;
        let mut ts = targets;
        while ts != 0 {
            let to = ts.trailing_zeros() as usize;
            ts &= ts - 1;
            let land = 1u64 << to;
            let caps = opp & RING1[to];
            let child_opp = opp & !caps;

            let clones = me & RING1[to];
            if clones != 0 {
                let child_me = me | caps | land;
                let cempty = !(child_me | child_opp | block) & FULL;
                total += clones.count_ones() as u64 * count_pairs(child_opp, cempty);
            }
            let mut js = me & RING2[to];
            while js != 0 {
                let from = js.trailing_zeros();
                js &= js - 1;
                let child_me = (me & !(1u64 << from)) | caps | land;
                let cempty = !(child_me | child_opp | block) & FULL;
                total += count_pairs(child_opp, cempty);
            }
        }
        return total;
    }

    let mut total = 0u64;
    let mut ts = targets;
    while ts != 0 {
        let to = ts.trailing_zeros() as usize;
        ts &= ts - 1;
        let land = 1u64 << to;
        let caps = opp & RING1[to];
        let child_opp = opp & !caps;

        // Clones: every ring-1 source -> the SAME child position.
        let clones = me & RING1[to];
        if clones != 0 {
            let child_me = me | caps | land;
            total += clones.count_ones() as u64
                * perft_bb(child_opp, child_me, block, depth - 1);
        }

        // Jumps: each vacated source gives a distinct child.
        let mut jumps = me & RING2[to];
        while jumps != 0 {
            let from = jumps.trailing_zeros();
            jumps &= jumps - 1;
            let child_me = (me & !(1u64 << from)) | caps | land;
            total += perft_bb(child_opp, child_me, block, depth - 1);
        }
    }
    total
}

/// Extract the ring-1 occupancy pattern around `sq` as a compact 8-bit key
/// (bit i = NEIGH_POS[sq][i] occupied). Portable PEXT-equivalent.
#[inline]
pub fn extract8(occ: u64, sq: u8) -> u8 {
    let s = sq as usize;
    let mut key = 0u8;
    let mut i = 0usize;
    while i < NEIGH_CNT[s] as usize {
        if occ & (1u64 << NEIGH_POS[s][i]) != 0 {
            key |= 1u8 << i;
        }
        i += 1;
    }
    key
}

/// BMI2 hardware PEXT variant of `extract8` (requires BMI2 support).
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn extract8_pext(occ: u64, sq: u8) -> u8 {
    std::arch::x86_64::_pext_u64(occ, RING1[sq as usize]) as u8
}

/// A move: `from` -> `to`. `from == to == 255` encodes a pass.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Move {
    pub from: u8,
    pub to: u8,
}

impl Move {
    pub const PASS: Move = Move { from: 255, to: 255 };

    #[inline]
    pub const fn is_pass(self) -> bool {
        self.from == 255
    }

    /// True if this move is a clone (distance 1) rather than a jump (distance 2).
    /// Table test: `to` inside `from`'s ring-1. No div/mod (was 2x /7+%7).
    #[inline]
    pub fn is_clone(self) -> bool {
        debug_assert!(self.from < 49 && self.to < 49);
        (RING1[self.from as usize] >> self.to) & 1 != 0
    }

    /// Pack into `u32` as `(from << 8) | to` (pass = 0xFFFF).
    #[inline]
    pub const fn to_u32(self) -> u32 {
        ((self.from as u32) << 8) | self.to as u32
    }

    #[inline]
    pub const fn from_u32(v: u32) -> Move {
        Move {
            from: (v >> 8) as u8,
            to: v as u8,
        }
    }
}

/// Fixed-capacity move list: no heap allocation, safe to use per search node.
#[derive(Clone, Copy)]
pub struct MoveList {
    moves: [Move; MAX_MOVES],
    scores: [i32; MAX_MOVES],
    len: usize,
}

impl MoveList {
    pub const fn new() -> Self {
        MoveList {
            moves: [Move::PASS; MAX_MOVES],
            scores: [0; MAX_MOVES],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES, "move list overflow");
        if self.len < MAX_MOVES {
            self.moves[self.len] = m;
            self.len += 1;
        }
    }

    #[inline]
    pub fn set_score(&mut self, i: usize, s: i32) {
        self.scores[i] = s;
    }

    #[inline]
    pub fn move_at(&self, i: usize) -> Move {
        self.moves[i]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reset for reuse without re-initializing the backing arrays.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.len].iter().copied()
    }

    pub fn contains(&self, m: &Move) -> bool {
        self.moves[..self.len].contains(m)
    }

    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        self.moves.swap(i, j);
        self.scores.swap(i, j);
    }

    /// Lazily extract the highest-scored remaining move (swap-with-last).
    /// Cheaper than a full sort at cut-nodes: alpha-beta usually refutes in
    /// the first few picks, and this only ever pays for what it uses.
    #[inline]
    pub fn pop_best(&mut self) -> Move {
        debug_assert!(self.len > 0);
        let mut best_i = 0;
        let mut best_s = self.scores[0];
        for i in 1..self.len {
            if self.scores[i] > best_s {
                best_s = self.scores[i];
                best_i = i;
            }
        }
        let m = self.moves[best_i];
        self.len -= 1;
        self.moves[best_i] = self.moves[self.len];
        self.scores[best_i] = self.scores[self.len];
        m
    }
}

/// Square PST (autaxx quiet-ordering table): corners/edges good, center bad.
/// Rows top->bottom, horizontally symmetric (mirror helpers rely on this).
/// Maintained INCREMENTALLY on `Board::pst` (updated in make()) — the static
/// eval never loops over stones.
pub const PST: [i32; 49] = [
    30, 20, 10, 10, 10, 20, 30, //
    20, 10, 10, 5, 10, 10, 20, //
    10, 10, 5, 0, 5, 10, 10, //
    10, 5, 0, 0, 0, 5, 10, //
    10, 10, 5, 0, 5, 10, 10, //
    20, 10, 10, 5, 10, 10, 20, //
    30, 20, 10, 10, 10, 20, 30,
];

/// Immutable board; `make` is copy-make (cheap, ~48 bytes).
#[derive(Clone, Copy, Debug)]
pub struct Board {
    /// `[black, white]` occupancy bitboards.
    pub occ: [u64; 2],
    pub blockers: u64,
    /// Side to move: 0 = black, 1 = white.
    pub turn: u8,
    /// Consecutive passes (game over at 2).
    pub passes: u8,
    /// Plies played since the start position (for mate-distance scores).
    pub ply: u32,
    /// Incremental piece counters (eval / terminal checks never popcount).
    pub piece_cnt: [u8; 2],
    pub blocker_cnt: u8,
    /// Incremental zobrist hash (side to move + pieces + blockers).
    pub hash: u64,
    /// Incremental PST balance (black sum minus white sum), updated in make().
    pub pst: i32,
}

impl Board {
    /// Standard start: black bottom-left, white top-right, black to move.
    pub fn start() -> Board {
        Board {
            occ: [bit_of(6 * 7), bit_of(6)],
            blockers: 0,
            turn: 0,
            passes: 0,
            ply: 0,
            piece_cnt: [1, 1],
            blocker_cnt: 0,
            hash: ZOB_PIECE[0][42] ^ ZOB_PIECE[1][6],
            pst: PST[6 * 7] - PST[6],
        }
    }

    #[inline]
    pub fn occupied(&self) -> u64 {
        self.occ[0] | self.occ[1] | self.blockers
    }

    #[inline]
    pub fn empty(&self) -> u64 {
        !self.occupied() & FULL
    }

    #[inline]
    pub fn me(&self) -> u64 {
        self.occ[self.turn as usize]
    }

    #[inline]
    pub fn opp(&self) -> u64 {
        self.occ[1 - self.turn as usize]
    }

    /// Squares this side could land on right now (branch-free bitboard math:
    /// Chebyshev-1 ring + full Chebyshev-2 ring, masked to empty squares).
    pub fn targets(&self, side: u8) -> u64 {
        let pieces = self.occ[side as usize];
        (dist_union(pieces, 1) | jump_union(pieces)) & self.empty()
    }

    /// Quick "does this side have any legal move" check.
    pub fn has_moves(&self, side: u8) -> bool {
        self.targets(side) != 0
    }

    /// All legal moves for the side to move (passes are NOT included).
    pub fn legal_moves(&self) -> MoveList {
        let mut list = MoveList::new();
        self.legal_moves_into(&mut list);
        list
    }

    /// Fill a caller-provided list (reusable buffers avoid per-node init).
    pub fn legal_moves_into(&self, list: &mut MoveList) {
        list.clear();
        let me = self.me();
        let mut targets = self.targets(self.turn);
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            let mut froms = me & REACH[to as usize];
            while froms != 0 {
                let from = froms.trailing_zeros() as u8;
                froms &= froms - 1;
                list.push(Move { from, to });
            }
        }
    }

    /// Like `legal_moves_into`, but clone moves to the same destination are
    /// deduplicated to one representative source: every clone to a given
    /// square yields the identical child position, so search never needs the
    /// extra copies. Jump moves stay one-per-source.
    pub fn legal_moves_dedup_into(&self, list: &mut MoveList) {
        list.clear();
        let me = self.me();
        let mut targets = self.targets(self.turn);
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            let clones = me & RING1[to as usize];
            if clones != 0 {
                let from = clones.trailing_zeros() as u8;
                list.push(Move { from, to });
            }
            let mut jumps = me & RING2[to as usize];
            while jumps != 0 {
                let from = jumps.trailing_zeros() as u8;
                jumps &= jumps - 1;
                list.push(Move { from, to });
            }
        }
    }

    /// Number of enemy pieces adjacent to `to` that a move landing there would convert.
    #[inline]
    pub fn captures(&self, to: u8) -> u32 {
        (self.opp() & RING1[to as usize]).count_ones()
    }

    /// Apply a move (copy-make). Does not validate legality.
    /// Clone: origin stays occupied. Jump: origin is vacated.
    pub fn make(&self, m: Move) -> Board {
        debug_assert!(!m.is_pass());
        let mut next = *self;
        let me_idx = self.turn as usize;
        let opp_idx = 1 - me_idx;
        // PST sign: pst is black-relative (black sum minus white sum).
        let sign: i32 = if me_idx == 0 { 1 } else { -1 };
        next.hash = self.hash ^ ZOB_SIDE;
        if !m.is_clone() {
            next.occ[me_idx] &= !bit_of(m.from);
            next.piece_cnt[me_idx] -= 1;
            next.hash ^= ZOB_PIECE[me_idx][m.from as usize];
            next.pst -= sign * PST[m.from as usize];
        }
        next.occ[me_idx] |= bit_of(m.to);
        next.hash ^= ZOB_PIECE[me_idx][m.to as usize];
        next.pst += sign * PST[m.to as usize];
        let captured = next.occ[opp_idx] & RING1[m.to as usize];
        next.occ[opp_idx] &= !captured;
        next.occ[me_idx] |= captured;
        let mut caps_bb = captured;
        while caps_bb != 0 {
            let sq = caps_bb.trailing_zeros() as usize;
            caps_bb &= caps_bb - 1;
            next.hash ^= ZOB_PIECE[opp_idx][sq] ^ ZOB_PIECE[me_idx][sq];
            // Captured stone flips sides: swing is twice the square value.
            next.pst += 2 * sign * PST[sq];
        }
        let caps = captured.count_ones() as u8;
        next.piece_cnt[me_idx] += 1 + caps;
        next.piece_cnt[opp_idx] -= caps;
        next.turn = opp_idx as u8;
        next.passes = 0;
        next.ply += 1;
        next
    }

    /// LUT-driven variant of `make`: the conversion mask comes from the
    /// compact-key `CONVERT8` table instead of a mask AND. Must be
    /// bit-identical to `make` (enforced by tests).
    pub fn make_via_lut(&self, m: Move) -> Board {
        debug_assert!(!m.is_pass());
        let mut next = *self;
        let me_idx = self.turn as usize;
        let opp_idx = 1 - me_idx;
        let sign: i32 = if me_idx == 0 { 1 } else { -1 };
        next.hash = self.hash ^ ZOB_SIDE;
        if !m.is_clone() {
            next.occ[me_idx] &= !bit_of(m.from);
            next.piece_cnt[me_idx] -= 1;
            next.hash ^= ZOB_PIECE[me_idx][m.from as usize];
            next.pst -= sign * PST[m.from as usize];
        }
        next.occ[me_idx] |= bit_of(m.to);
        next.hash ^= ZOB_PIECE[me_idx][m.to as usize];
        next.pst += sign * PST[m.to as usize];
        let flip = infect_via_lut(next.occ[opp_idx], m.to);
        next.occ[opp_idx] &= !flip;
        next.occ[me_idx] |= flip;
        let mut caps_bb = flip;
        while caps_bb != 0 {
            let sq = caps_bb.trailing_zeros() as usize;
            caps_bb &= caps_bb - 1;
            next.hash ^= ZOB_PIECE[opp_idx][sq] ^ ZOB_PIECE[me_idx][sq];
            next.pst += 2 * sign * PST[sq];
        }
        let caps = flip.count_ones() as u8;
        next.piece_cnt[me_idx] += 1 + caps;
        next.piece_cnt[opp_idx] -= caps;
        next.turn = opp_idx as u8;
        next.passes = 0;
        next.ply += 1;
        next
    }

    /// Side to move passes (only legal when it has no moves).
    pub fn make_pass(&self) -> Board {
        let mut next = *self;
        next.turn = 1 - self.turn as usize as u8;
        next.passes += 1;
        next.ply += 1;
        next.hash = self.hash ^ ZOB_SIDE;
        next
    }

    /// Hard game end: board full or a side wiped out (O(1) via counters).
    pub fn is_over(&self) -> bool {
        self.piece_cnt[0] == 0
            || self.piece_cnt[1] == 0
            || (self.piece_cnt[0] + self.piece_cnt[1] + self.blocker_cnt) as usize == SQUARES
    }

    /// Both players stuck (with passes this also ends the game).
    pub fn stuck_both(&self) -> bool {
        !self.has_moves(0) && !self.has_moves(1)
    }

    /// Game over including the double-pass condition. O(1): the "both stuck"
    /// case terminates through consecutive passes (`passes >= 2`), so the
    /// per-node mobility scan `stuck_both()` is NOT part of this hot check.
    pub fn game_over(&self) -> bool {
        self.is_over() || self.passes >= 2
    }

    /// `(black_count, white_count)` — O(1) incremental counters.
    pub fn counts(&self) -> (u32, u32) {
        (self.piece_cnt[0] as u32, self.piece_cnt[1] as u32)
    }

    /// 0 black wins, 1 white wins, 2 draw, 3 ongoing. Call only when `game_over()`.
    pub fn winner(&self) -> u8 {
        let (b, w) = self.counts();
        match b.cmp(&w) {
            std::cmp::Ordering::Greater => 0,
            std::cmp::Ordering::Less => 1,
            std::cmp::Ordering::Equal => 2,
        }
    }

    /// Parse a 49-char board string (rows top->bottom, chars `.xo#`) plus a
    /// side char `b`/`w`.
    pub fn from_str(s: &str) -> Option<Board> {
        let mut b = Board {
            occ: [0; 2],
            blockers: 0,
            turn: 0,
            passes: 0,
            ply: 0,
            piece_cnt: [0; 2],
            blocker_cnt: 0,
            hash: 0,
            pst: 0,
        };
        let mut cells = 0usize;
        let mut side = 'b';
        for c in s.chars() {
            match c {
                '.' => cells += 1,
                'x' => {
                    b.occ[0] |= bit_of(cells as u8);
                    cells += 1;
                }
                'o' => {
                    b.occ[1] |= bit_of(cells as u8);
                    cells += 1;
                }
                '#' => {
                    b.blockers |= bit_of(cells as u8);
                    cells += 1;
                }
                'b' | 'w' => side = c,
                c if c.is_whitespace() => {}
                _ => return None,
            }
        }
        if cells != SQUARES {
            return None;
        }
        b.turn = if side == 'b' { 0 } else { 1 };
        b.piece_cnt[0] = b.occ[0].count_ones() as u8;
        b.piece_cnt[1] = b.occ[1].count_ones() as u8;
        b.blocker_cnt = b.blockers.count_ones() as u8;
        b.hash = if b.turn == 1 { ZOB_SIDE } else { 0 };
        for c in 0..2 {
            let mut bb = b.occ[c];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                b.hash ^= ZOB_PIECE[c][sq];
                b.pst += if c == 0 { PST[sq] } else { -PST[sq] };
            }
        }
        let mut blk = b.blockers;
        while blk != 0 {
            let sq = blk.trailing_zeros() as usize;
            blk &= blk - 1;
            b.hash ^= ZOB_BLOCK[sq];
        }
        Some(b)
    }

    /// Serialize as `"<49 chars> <b|w>"`.
    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity(SQUARES + 2);
        for sq in 0..SQUARES as u8 {
            let bit = bit_of(sq);
            let c = if self.blockers & bit != 0 {
                '#'
            } else if self.occ[0] & bit != 0 {
                'x'
            } else if self.occ[1] & bit != 0 {
                'o'
            } else {
                '.'
            };
            s.push(c);
        }
        s.push(' ');
        s.push(if self.turn == 0 { 'b' } else { 'w' });
        s
    }

    /// libataxx-style FEN (rows top->bottom, digit runs, then side and two
    /// counters) for talking UAI to other engines: `6o/7/7/7/7/7/x6 x 0 1`.
    pub fn to_ataxx_fen(&self) -> String {
        let mut rows = Vec::with_capacity(7);
        for r in 0..SIZE {
            let mut row = String::with_capacity(SIZE);
            let mut gap = 0u32;
            for f in 0..SIZE {
                let bit = bit_of((r * SIZE + f) as u8);
                let c = if self.blockers & bit != 0 {
                    if gap > 0 {
                        row.push_str(&gap.to_string());
                        gap = 0;
                    }
                    '#'
                } else if self.occ[0] & bit != 0 {
                    if gap > 0 {
                        row.push_str(&gap.to_string());
                        gap = 0;
                    }
                    'x'
                } else if self.occ[1] & bit != 0 {
                    if gap > 0 {
                        row.push_str(&gap.to_string());
                        gap = 0;
                    }
                    'o'
                } else {
                    gap += 1;
                    continue;
                };
                row.push(c);
            }
            if gap > 0 {
                row.push_str(&gap.to_string());
            }
            rows.push(row);
        }
        format!(
            "{} {} 0 1",
            rows.join("/"),
            if self.turn == 0 { 'x' } else { 'o' }
        )
    }

    /// libataxx-style FEN parser (inverse of `to_ataxx_fen`): rows
    /// top->bottom separated by `/`, digit runs, `x o #`, then side and two
    /// counters (ignored).
    pub fn from_ataxx_fen(s: &str) -> Option<Board> {
        let mut b = Board {
            occ: [0; 2],
            blockers: 0,
            turn: 0,
            passes: 0,
            ply: 0,
            piece_cnt: [0; 2],
            blocker_cnt: 0,
            hash: 0,
            pst: 0,
        };
        let mut parts = s.split_whitespace();
        let board_part = parts.next()?;
        let side = parts.next()?;
        let mut row = 0usize;
        let mut col = 0usize;
        for ch in board_part.chars() {
            match ch {
                '/' => {
                    if col != SIZE || row >= SIZE - 1 {
                        return None;
                    }
                    row += 1;
                    col = 0;
                }
                '1'..='7' => {
                    col += ch as usize - '0' as usize;
                    if col > SIZE {
                        return None;
                    }
                }
                'x' | 'o' | '#' => {
                    if col >= SIZE {
                        return None;
                    }
                    let bit = bit_of((row * SIZE + col) as u8);
                    match ch {
                        'x' => b.occ[0] |= bit,
                        'o' => b.occ[1] |= bit,
                        _ => b.blockers |= bit,
                    }
                    col += 1;
                }
                _ => return None,
            }
        }
        if row != SIZE - 1 || col != SIZE {
            return None;
        }
        b.turn = match side {
            "x" => 0,
            "o" => 1,
            _ => return None,
        };
        b.piece_cnt[0] = b.occ[0].count_ones() as u8;
        b.piece_cnt[1] = b.occ[1].count_ones() as u8;
        b.blocker_cnt = b.blockers.count_ones() as u8;
        b.hash = if b.turn == 1 { ZOB_SIDE } else { 0 };
        for c in 0..2 {
            let mut bb = b.occ[c];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                b.hash ^= ZOB_PIECE[c][sq];
                b.pst += if c == 0 { PST[sq] } else { -PST[sq] };
            }
        }
        let mut blk = b.blockers;
        while blk != 0 {
            let sq = blk.trailing_zeros() as usize;
            blk &= blk - 1;
            b.hash ^= ZOB_BLOCK[sq];
        }
        Some(b)
    }

    /// File-and-rank square name, chess style: a1 = bottom-left, g7 = top-right.
    pub fn sq_name(sq: u8) -> String {
        let file = (b'a' + sq % 7) as char;
        let rank = 7 - sq / 7;
        format!("{file}{rank}")
    }

    /// Parse a square name like "d4".
    pub fn parse_sq(name: &str) -> Option<u8> {
        let bytes = name.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].to_ascii_lowercase().wrapping_sub(b'a') as i32;
        let rank = (bytes[1] as char).to_digit(10)? as i32;
        if !(0..7).contains(&file) || !(1..=7).contains(&rank) {
            return None;
        }
        Some(((7 - rank) * 7 + file) as u8)
    }

    /// Human-readable ASCII board for the CLI.
    pub fn pretty(&self) -> String {
        let mut s = String::with_capacity(120);
        s.push_str("  a b c d e f g\n");
        for r in 0..SIZE {
            s.push((b'0' + (7 - r) as u8) as char);
            s.push(' ');
            for f in 0..SIZE {
                let bit = bit_of((r * 7 + f) as u8);
                let c = if self.blockers & bit != 0 {
                    '#'
                } else if self.occ[0] & bit != 0 {
                    'x'
                } else if self.occ[1] & bit != 0 {
                    'o'
                } else {
                    '.'
                };
                s.push(c);
                s.push(' ');
            }
            s.push('\n');
        }
        s.push_str(&format!(
            "to move: {}, black: {}, white: {}\n",
            if self.turn == 0 { "black" } else { "white" },
            self.occ[0].count_ones(),
            self.occ[1].count_ones()
        ));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mirror(b: &Board) -> Board {
        // Swap colors and mirror files so perft counts must match.
        let mut m = Board {
            occ: [0; 2],
            blockers: 0,
            turn: 1 - b.turn,
            passes: b.passes,
            ply: b.ply,
            piece_cnt: [b.piece_cnt[1], b.piece_cnt[0]],
            blocker_cnt: b.blocker_cnt,
            hash: 0,
            // PST rows are file-symmetric, so color-swap + file-mirror just
            // negates the balance.
            pst: -b.pst,
        };
        let flip = |mut bb: u64| -> u64 {
            let mut out = 0u64;
            while bb != 0 {
                let sq = bb.trailing_zeros();
                bb &= bb - 1;
                out |= 1u64 << ((sq / 7) * 7 + (6 - sq % 7));
            }
            out
        };
        m.occ[0] = flip(b.occ[1]);
        m.occ[1] = flip(b.occ[0]);
        m.blockers = flip(b.blockers);
        m.hash = if m.turn == 1 { ZOB_SIDE } else { 0 };
        for c in 0..2 {
            let mut bb = m.occ[c];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                m.hash ^= ZOB_PIECE[c][sq];
            }
        }
        let mut blk = m.blockers;
        while blk != 0 {
            let sq = blk.trailing_zeros() as usize;
            blk &= blk - 1;
            m.hash ^= ZOB_BLOCK[sq];
        }
        m
    }

    fn perft(b: &Board, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = b.legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut n = 0;
        for i in 0..moves.len() {
            n += perft(&b.make(moves.move_at(i)), depth - 1);
        }
        n
    }

    #[test]
    fn corner_geometry() {
        // From a1: clones a2 b2 b1; jumps a3 b3 c3 c2 c1 (full 5x5 ring,
        // knight offsets included). 8 moves total.
        let b = Board::start();
        let targets = b.targets(0);
        let expect = bit_of(35)  // a2
            | bit_of(36)         // b2
            | bit_of(43)         // b1
            | bit_of(28)         // a3
            | bit_of(29)         // b3
            | bit_of(30)         // c3
            | bit_of(37)         // c2
            | bit_of(44);        // c1
        assert_eq!(targets, expect, "a1 targets must be 3 clones + 5 jumps");
    }

    #[test]
    fn start_has_8_moves() {
        // One corner piece each: 3 clones + 5 jumps.
        assert_eq!(Board::start().legal_moves().len(), 8);
    }

    #[test]
    fn perft_symmetry() {
        let b = Board::start();
        assert_eq!(perft(&b, 2), perft(&mirror(&b), 2));
        assert_eq!(perft(&b, 3), perft(&mirror(&b), 3));
        assert_eq!(perft(&b, 4), perft(&mirror(&b), 4));
    }

    #[test]
    fn perft_growth() {
        // Sanity: strictly increasing node counts from the start position.
        let b = Board::start();
        let p1 = perft(&b, 1) as i64;
        let p2 = perft(&b, 2) as i64;
        let p3 = perft(&b, 3) as i64;
        assert!(p1 < p2 && p2 < p3, "perft not increasing: {p1} {p2} {p3}");
    }

    #[test]
    fn shift_tables_match_ring_tables() {
        // The shift-based generators must reproduce the per-square ring
        // tables exactly for every square. Off-board spill bits are masked
        // by every real caller (`& FULL` via `empty()`), so compare masked.
        for sq in 0..SQUARES as u8 {
            let bb = bit_of(sq);
            assert_eq!(
                dist_union(bb, 1) & FULL,
                RING1[sq as usize],
                "RING1 mismatch at {sq}"
            );
            assert_eq!(
                jump_union(bb) & FULL,
                RING2[sq as usize],
                "RING2 mismatch at {sq}"
            );
        }
    }

    #[test]
    fn perft_bb_differential() {
        // Walk depth-2 positions with both counters; find the first child
        // where the leaf counts diverge and dump the position.
        let walk = |b: &Board, path: &mut String| -> bool {
            let moves = b.legal_moves();
            for i in 0..moves.len() {
                let m = moves.move_at(i);
                let child = b.make(m);
                // Side to move in the child is child.turn — feed perft_bb
                // the bitboards in (mover, opponent) order.
                let a = perft_bb(
                    child.occ[child.turn as usize],
                    child.occ[1 - child.turn as usize],
                    child.blockers,
                    1,
                );
                let board_n = child.legal_moves().len() as u64;
                if a != board_n {
                    path.push_str(&format!(
                        "diverge after move {} ({}): bb={} board={} pos={}",
                        path.len(),
                        Move { from: m.from, to: m.to }.to_u32(),
                        a,
                        board_n,
                        child.to_string()
                    ));
                    return true;
                }
            }
            false
        };
        let mut path = String::new();
        let b = Board::start();
        assert!(!walk(&b, &mut path), "{path}");
        // Depth 2: check all grandchildren too.
        let moves = b.legal_moves();
        for i in 0..moves.len() {
            let c1 = b.make(moves.move_at(i));
            let mut p2 = format!("{i}");
            if walk(&c1, &mut p2) {
                panic!("depth-3 divergence: root move {i}: {p2}");
            }
        }
    }

    #[test]
    fn perft_bb_matches_board_perft() {
        // The raw-bitboard perft (bulk leaves + clone folding) must produce
        // exactly the same pair counts as the Board-based reference.
        for d in 1..=5u32 {
            let b = Board::start();
            assert_eq!(
                perft_bb(b.occ[0], b.occ[1], b.blockers, d),
                perft(&b, d),
                "perft_bb diverged at depth {d}"
            );
        }
    }

    #[test]
    fn incremental_hash_matches_scratch() {
        // After every playout move, the incremental hash must equal a
        // from-scratch recompute (and differ across sides to move).
        let recompute = |b: &Board| -> u64 {
            let mut h = if b.turn == 1 { ZOB_SIDE } else { 0 };
            for c in 0..2 {
                let mut bb = b.occ[c];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    h ^= ZOB_PIECE[c][sq];
                }
            }
            let mut blk = b.blockers;
            while blk != 0 {
                let sq = blk.trailing_zeros() as usize;
                blk &= blk - 1;
                h ^= ZOB_BLOCK[sq];
            }
            h
        };
        let mut state: u64 = 0x0123456789ABCDEF;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..15 {
            let mut b = Board::start();
            assert_eq!(b.hash, recompute(&b));
            for _ in 0..40 {
                if b.game_over() {
                    break;
                }
                let moves = b.legal_moves();
                if moves.is_empty() {
                    b = b.make_pass();
                } else {
                    b = b.make(moves.move_at((rng() % moves.len() as u64) as usize));
                }
                assert_eq!(b.hash, recompute(&b), "hash diverged");
            }
        }
    }

    #[test]
    fn dedup_moves_cover_same_children() {
        // Dedup list: one clone per destination + one jump per source. The
        // set of resulting child positions must equal the full list's set.
        let mut state: u64 = 0xABCDEF0123456789;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let child_set = |b: &Board, list: &MoveList| -> std::collections::HashSet<[u64; 2]> {
            (0..list.len())
                .map(|i| b.make(list.move_at(i)).occ)
                .collect()
        };
        for _ in 0..25 {
            let mut b = Board::start();
            for _ in 0..25 {
                if b.game_over() {
                    break;
                }
                let full = b.legal_moves();
                let mut dedup = MoveList::new();
                b.legal_moves_dedup_into(&mut dedup);

                for i in 0..dedup.len() {
                    assert!(full.contains(&dedup.move_at(i)), "dedup move illegal");
                }
                // Same child positions, strictly fewer-or-equal moves.
                assert_eq!(child_set(&b, &full), child_set(&b, &dedup));
                assert!(dedup.len() <= full.len());

                let moves = b.legal_moves();
                if moves.is_empty() {
                    b = b.make_pass();
                    continue;
                }
                b = b.make(moves.move_at((rng() % moves.len() as u64) as usize));
            }
        }
    }

    #[test]
    fn clone_infects_neighbors() {
        // Black jumps a4->c4; c4 is adjacent to the white piece on d4, so it is converted.
        let board = "..x.... \
                     ....... \
                     ....... \
                     x..o..x \
                     ....... \
                     ....... \
                     .....x. b";
        let b = Board::from_str(board).unwrap();
        let m = Move {
            from: Board::parse_sq("a4").unwrap(),
            to: Board::parse_sq("c4").unwrap(),
        };
        assert!(b.legal_moves().contains(&m));
        let after = b.make(m);
        let (black, white) = after.counts();
        assert_eq!(white, 0);
        assert_eq!(black, 5); // 4 originals - vacated a4 + landed c4 + converted d4
        assert_eq!(after.winner(), 0);
    }

    #[test]
    fn string_roundtrip() {
        let b = Board::start();
        let parsed = Board::from_str(&b.to_string()).unwrap();
        assert_eq!(parsed.occ, b.occ);
        assert_eq!(parsed.turn, b.turn);
    }

    #[test]
    fn sq_names() {
        assert_eq!(Board::parse_sq("a1"), Some(42));
        assert_eq!(Board::parse_sq("g7"), Some(6));
        assert_eq!(Board::sq_name(42), "a1");
        assert_eq!(Board::sq_name(6), "g7");
    }

    #[test]
    fn infect_lut_exhaustive() {
        // For every square and every possible enemy pattern, the LUT must
        // agree with the direct bitboard computation and with NEIGH_POS.
        for sq in 0..SQUARES as u8 {
            for pat in 0..256usize {
                // Rebuild the occupancy the pattern encodes.
                let mut occ = 0u64;
                let s = sq as usize;
                let cnt = NEIGH_CNT[s] as usize;
                for i in 0..cnt {
                    if pat & (1 << i) != 0 {
                        occ |= 1u64 << NEIGH_POS[s][i];
                    }
                }
                // Bits outside the neighborhood must be ignored by the LUT.
                let polluted = occ | (FULL & !RING1[s]);
                let via_lut = infect_via_lut(polluted, sq);
                let direct = infect_direct(occ, sq);
                assert_eq!(via_lut, direct, "LUT mismatch at sq {sq} pat {pat}");
                // Stray high bits of the pattern (beyond cnt) must not matter.
                if (pat >> cnt) != 0 {
                    assert_eq!(infect_via_lut(occ, sq), direct);
                }
            }
        }
    }

    #[test]
    fn extract8_matches_ring1() {
        // Round-trip: extraction key must describe exactly RING1 occupancy.
        for sq in 0..SQUARES as u8 {
            let mut occ = 0u64;
            for i in 0..NEIGH_CNT[sq as usize] {
                if i % 2 == 0 {
                    occ |= 1u64 << NEIGH_POS[sq as usize][i as usize];
                }
            }
            let key = extract8(occ, sq);
            for i in 0..NEIGH_CNT[sq as usize] as usize {
                assert_eq!(
                    key & (1 << i) != 0,
                    occ & (1u64 << NEIGH_POS[sq as usize][i]) != 0
                );
            }
        }
    }

    #[test]
    fn make_equals_make_via_lut() {
        // Deterministic fuzz: random playouts, every move applied both ways.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..40 {
            let mut b = Board::start();
            for _ in 0..60 {
                if b.game_over() {
                    break;
                }
                let moves = b.legal_moves();
                if moves.is_empty() {
                    b = b.make_pass();
                    continue;
                }
                let m = moves.move_at((rng() % moves.len() as u64) as usize);
                let a = b.make(m);
                let c = b.make_via_lut(m);
                assert_eq!(a.occ, c.occ, "make/make_via_lut diverged on {:?}", m);
                assert_eq!(a.piece_cnt, c.piece_cnt);
                b = a;
            }
        }
    }

    #[test]
    fn incremental_counts_match_popcount() {
        let mut state: u64 = 0xDEADBEEFCAFEBABE;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20 {
            let mut b = Board::start();
            for _ in 0..50 {
                if b.game_over() {
                    break;
                }
                assert_eq!(b.piece_cnt[0] as u32, b.occ[0].count_ones());
                assert_eq!(b.piece_cnt[1] as u32, b.occ[1].count_ones());
                // Incremental PST must equal the from-scratch balance.
                let mut want = 0i32;
                for c in 0..2 {
                    let sign = if c == 0 { 1 } else { -1 };
                    let mut bb = b.occ[c];
                    while bb != 0 {
                        let sq = bb.trailing_zeros() as usize;
                        bb &= bb - 1;
                        want += sign * PST[sq];
                    }
                }
                assert_eq!(b.pst, want, "incremental pst diverged");
                let moves = b.legal_moves();
                if moves.is_empty() {
                    b = b.make_pass();
                    continue;
                }
                b = b.make(moves.move_at((rng() % moves.len() as u64) as usize));
            }
        }
    }
}
