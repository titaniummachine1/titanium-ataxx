# TITANIUM — Ataxx engine

A basic **alpha-beta Ataxx engine** in Rust with two faces:

- **Browser** — engine compiled to WebAssembly, GUI in plain JS via Vite (HMR in dev)
- **Native binary** — `titanium-cli` for testing: perft, bench, selfplay, console play

## Rules (implemented)

7x7 board, black (bottom-left) vs white (top-right), black moves first.

- **Clone**: move to any empty square at Chebyshev distance 1 (piece stays).
- **Jump**: move to any empty square at Chebyshev distance 2 (piece moves).
- **Infection**: after *any* landing, every enemy piece adjacent to the landing
  square flips to your color.
- No legal move = forced pass. Board full, a side wiped out, or both players
  stuck = game over. Majority wins.

## Layout

```
crates/titanium        engine core: bitboard rules, movegen, alpha-beta search
crates/titanium-cli    native test binary (perft / bench / selfplay / play / bestmove)
crates/titanium-wasm   wasm-bindgen bindings for the browser
web/                   Vite GUI (vanilla JS + canvas)
data/ logs/ scripts/   local scratch space — git-ignored, never pushed
```

## Develop (browser GUI, hot reload)

Prereqs: Rust, `rustup target add wasm32-unknown-unknown`, `cargo install wasm-pack`, Node 20+.

```sh
cd web
npm install
npm run wasm   # build engine -> web/pkg (only needed after Rust changes)
npm run dev    # vite dev server at http://localhost:5173
```

JS/CSS/HTML edits hot-reload; after Rust edits re-run `npm run wasm`.

Production bundle: `npm run build` (outputs `web/dist`, relative paths, works on any static host).

## Test the engine natively

```sh
cargo test -p titanium                       # movegen/eval/search unit tests
cargo run -p titanium-cli --release -- perft 4
cargo run -p titanium-cli --release -- bench --depth 8
cargo run -p titanium-cli --release -- selfplay --games 4 --time 300   # -> logs/
cargo run -p titanium-cli --release -- play --time 1000                # console vs engine
cargo run -p titanium-cli --release -- bestmove "<49 chars> b" --time 500
```

Board string: 49 chars, rows top→bottom, `.` empty `x` black `o` white `#` blocker,
then side `b`/`w`. Squares are chess-style: `a1` bottom-left, `g7` top-right.

```
titanium start:  "......x ....... ....... ....... ....... ....... o...... b"
```

## Search (v0.1)

Negamax alpha-beta, copy-make on u64 bitboards (49 squares), precomputed
distance-1/distance-2 ring tables, iterative deepening with soft time limit,
move ordering by infection count, material + mobility eval.
No transposition table, no killers/history yet — see roadmap below.

## Roadmap

- [ ] transposition table + zobrist
- [ ] killer moves / history heuristic
- [ ] blocker ("wall") positions
- [ ] board sizes (5x5 / 6x6 / 8x8)
- [ ] opening book from selfplay data (`data/`)
