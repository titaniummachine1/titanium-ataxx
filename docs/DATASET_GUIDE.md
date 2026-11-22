# Titanium dataset guide (start here)

Written 2026-11-13 (D760-D764). For someone who has never seen this project. Every command below was run on
this PC on that day; the outputs quoted are real. Design history and audits: `docs/research/data-training/DATASET_V10.md`.

## 1. What the dataset is

Ataxx games played by the Titanium engine, kept in three linked forms. Each form has one job.

| Form | File | One record is | Job |
|---|---|---|---|
| **Games** | `*.tgam` (+ `.tgam.json`) | one whole game: start position + 4 bytes per move | the source of truth; everything else is rebuilt from it |
| **Rows** | `*.tdat` (+ `.chains`, `.manifest.json`) | one position, 48 bytes | what the trainer reads |
| **Game DAG** | `data/dag/games_dag.db` (Turso) | one position / one move / one game | which positions were already played (opening novelty), catalogue of games |

Why games and not loose positions: a position's training targets need its future (what the search said one,
two, ten plies later, and who won). Loose positions lose that. A game keeps it and costs 4.4 bytes per
position instead of 48.

### Files on this PC (all under `C:\gitProjects\titanium-ataxx\`)

| Path | Content |
|---|---|
| `data/datasets/v10/gate_games.tgam` | 57,244 gate games, 5,557,890 plies, 177 sources, 24.5 MB |
| `data/datasets/v10/e2e/` | the worked example of section 2 (48 self-play games and everything made from them) |
| `data/nnue/packed/train_v9.tdat` | 28,990,531 rows, the set the champion net was trained on (older, position form) |
| `data/nnue/packed/train_v11.tdat` (+ .kata .aux .chains .manifest.json) | 53,004,489 rows = train_v10 + 15,280,905 self-play rows of 2026-11-17 (D1049) |
| `data/nnue/packed/val_v9.tdat`, `val_v9_clean.tdat` | validation rows |
| `data/nnue/packed/train_v9.chains` | index of the row runs of `train_v9` that are still in game order |
| `data/nnue/packed/train_v9_lam09.tdat`, `_lam07.tdat` | `train_v9` with lambda-return targets (section 5) |
| `data/dag/games_dag.backup-20261114.db` | ORIGINAL game DAG (2026-10-24, sha256 1d6ce733...), frozen, never written |
| `data/dag/games_dag.db` | WORKING game DAG (Turso, MVCC journal; this name since 2026-11-14, D766). Check it with `ti dag stat` before relying on its content |

The tool is `ti` (the hub, Rust): `C:\gitProjects\titaniumtest\target\x86_64-pc-windows-gnu\release\ti.exe`.
Source: `C:\gitProjects\titaniumtest\src\` (`gen.rs` generator, `gamedata.rs` game files and rows, `dag.rs`
game DAG, `data.rs` row format, `targets.rs` training target).

## 2. Quick start: from nothing to trainer rows (measured run)

**CURRENT PIPELINE (D925/D926, 2026-11-16).** `ti gen` writes BINARY `STEM.tgam` (played move + its score per ply)
and `STEM.tbst` (the engine's BEST move + its score per ply) and writes every finished game into the sequence DAG.
There is no JSON and no conversion step:

```
ti gen  --out data/gen/run/batch_0001 --games 2000 --nodes 5000 --threads 4 --seq-dag data/dag/seq_dag.db
ti data games-tdat --tgam data/gen/run/batch_0001.tgam,data/gen/run/batch_0002.tgam --out data/datasets/vN/rows   # rows straight from the game files
ti data pack-games --out data/datasets/vN/all --inputs a.tgam,b.tgam,old_gate.games.jsonl   # OPTIONAL: merge files into one .tgam + .tbst
ti data lambda     --tdat data/datasets/vN/rows.tdat --chains data/datasets/vN/rows.chains --out data/datasets/vN/rows_lam09 --lambda 0.9
```

Label rule (user D925): a row's cp is ALWAYS the score of the engine's best move in that position (the value of the
position); the played move (it differs on ~3.6% of plies because of the novelty rule) only leads to the next row and
to the game result, so outcome / lambda / WDL follow the game that was actually played. A self-play `.tgam` without its
`.tbst` is refused. `pack-games` reads `.tgam` (+ `.tbst`, labels kept with each game) and old gate `*.games.jsonl`
alike; it never overwrites an existing output. Before 2026-11-16 `pack-games` read JSON only and turned binary input
into an EMPTY file ("stream did not contain valid UTF-8"), and `games-tdat` used the played-move score unless
`--best-scores` was given: both fixed.

The section below is the HISTORY of the first JSON pipeline (2026-11-13); its commands no longer apply to `ti gen`
output.

```
ti gen  --out data/datasets/v10/e2e/gen_e2e_a.jsonl --games 24 --time-ms 50 --threads 4 --novelty 8 --dag data/dag/games_dag.db
ti data pack-games --out data/datasets/v10/e2e/gen_e2e --inputs data/datasets/v10/e2e/gen_e2e_a.jsonl,data/datasets/v10/e2e/gen_e2e_b.jsonl
ti data games-stat --tgam data/datasets/v10/e2e/gen_e2e.tgam --by-source
ti data games-tdat --tgam data/datasets/v10/e2e/gen_e2e.tgam --out data/datasets/v10/e2e/gen_e2e_rows
ti data lambda     --tdat data/datasets/v10/e2e/gen_e2e_rows.tdat --chains data/datasets/v10/e2e/gen_e2e_rows.chains --out data/datasets/v10/e2e/gen_e2e_lam09 --lambda 0.9
ti dag stat --db data/dag/games_dag.db
```

What each step did and what it printed (history):

1. **`ti gen`** plays self-play games in-process under Botzone rules, 4 games at a time. In the first 8 plies
   it plays the least-played sound move (section 6). It writes one JSON line per game and writes every
   finished game into the DAG. Printed: `24 games ... in 26s on 4 thread(s)`, `novelty plies 192: best move
   was new 161, switched to a less played move 31`, `DAG write-back: 24 games stored, 0 conflict retries`.
2. **`pack-games`** turns game logs (`ti gen` output or gate `*.games.jsonl`) into one `.tgam`. It replays
   every game through the engine and drops exact duplicates. Printed: `48 games, 4834 plies, 21288 B (4.40 B
   per position)`, `exact duplicates dropped: 0`.
3. **`games-stat`** is the health check of a game file. Printed: `different positions: 4665 of 4834 (96.5%)`,
   from ply 20 on 100% new.
4. **`games-tdat`** writes one trainer row per ply plus a `.chains` index (which rows belong together).
   Printed: `4834 rows from 48 games`.
5. **`lambda`** (optional) writes a copy of the rows whose result field holds the lambda return (section 5).
6. **`dag stat`** checks the DAG identities (section 4). Printed: `identities OK`.

   Printed: `rows whose outcome now carries the future: 4393 (90.9%)`.

To train the value net on rows: `ti train run`. To train the policy head on game logs: `ti policy2` (section 5).
To test the result: `ti gate` only (`docs/GATE_RULE.md`).

## 3. Formats

All integers little-endian. Board: 49 squares, `square = row * 7 + column`, row 0 = rank 7, so square 0 = `a7`
and square 48 = `g1`. A position is two 64-bit sets (bit s = a stone on square s): side 0 (`x`) and side 1
(`o`), plus a set of blocked squares.

### 3.1 Game file `.tgam` (version 1)

File header, 32 bytes:

| Offset | Type | Meaning |
|---|---|---|
| 0 | 4 bytes | `TGAM` |
| 4 | u32 | version = 1 |
| 8 | u64 | number of games |
| 16 | u64 | number of plies in the whole file |
| 24 | u32 | number of sources (listed in the `.tgam.json` next to the file) |
| 28 | u32 | 0 |

Then per game a 40-byte header:

| Offset | Type | Meaning |
|---|---|---|
| 0 | u64 | start position: stones of side 0 |
| 8 | u64 | start position: stones of side 1 |
| 16 | u64 | blocked squares |
| 24 | u16 | number of plies N |
| 26 | u16 | source id (index into the `.tgam.json` list) |
| 28 | u8 | flags: bit 0 = side to move at the start; bits 1-2 = seat of the tested engine in a gate game (0, 1; 2 = unknown); bit 3 = game ended by forfeit |
| 29 | u8 | final stones of side 0 as the log recorded them (255 = not recorded) |
| 30 | u8 | final stones of side 1 |
| 31 | u8 | result for side 0: 2 = won, 1 = draw, 0 = lost, 3 = unknown |
| 32 | u32 | opening-book line |
| 36 | u32 | pair / game number inside its source |

Then N plies of 4 bytes:

| Bytes | Meaning |
|---|---|
| u16 | bits 0-9 = move code, bits 10-15 = search depth of that move (0-63) |
| i16 | score code of the search, from the view of the side that moves |

**Move code** (10 bits): `1..=49` = clone to square `code - 1` (the source stone is not stored: any own
neighbour gives the same position; the decoder reports the lowest one). `50..=529` = a jump: number all
(from, to) pairs at distance 2 in order of `from`, then `to`; the first is 50. `1023` = pass. `0` = none.

**Score code** (16 bits): `-32768` = no score logged. `-30000..=30000` = centipawns. `30001 + k` = win in k
plies, `-(30001 + k)` = loss in k plies. Engine helpers: `titanium::ttcodec::{move_code, decode_move,
enc_score, dec_score}`.

**To apply a move:** a jump empties its from-square, a clone does not; the to-square gets a stone; every
enemy stone next to the to-square changes side; the turn passes.

### 3.2 Trainer row `.tdat` (48 bytes, no file header: rows = file size / 48)

| Offset | Type | Field | Meaning |
|---|---|---|---|
| 0 | u64 | occ0 | stones of side 0 |
| 8 | u64 | occ1 | stones of side 1 |
| 16 | f32 | outcome | game result for the side to move: 1 won, 0.5 draw, 0 lost (or a lambda return, section 5) |
| 20 | f32 | cp | search score in centipawns for the side to move; `|cp| >= 98000` = proven win / loss |
| 24 | f32 | wsum | sample weight (1.0) |
| 28 | f32 | p3 | teacher win probability (0 when unused) |
| 32 | f32 | cp5k | the 5,000-node score (= cp unless the row was relabelled deeper) |
| 36 | i32 | zx | reserved |
| 40 | u8 | stm | side to move (0 or 1) |
| 41 | u8 | teacher_kind | 1 = old game-order rows, 2 = rows made from a game file or shuffled rows |
| 42 | u8 | proof_kind | 0 = none |
| 43 | u8 | relabel_src | which relabel produced cp: 0/1/2 = relabel passes, 200 = in-game eval (no relabel; `ti data games-tdat` default) |
| 44 | u8 | ti_clash | flag from the v9 build |
| 45 | 3 bytes | | padding |

Rows carry no blocked squares: they assume the open board. A row is a PROOF when |cp| >= 98000 (outcome then exact 1 / 0); how proofs are found and added (Botzone-rules Zataxx, `ti proof-lines`): [runbooks/PROOF_HUNT.md](./runbooks/PROOF_HUNT.md). `X.manifest.json` says how the file was made.

**`.chains`** (which consecutive rows are one game run): header `TCHN`, u32 version 1, u64 rows, u64 runs;
then 8 bytes per run: u32 first row, u16 length, u8 flag, u8 0. Inside one run, row i+1 is exactly one ply
after row i with the other side to move.

### 3.3 Game DAG (`games_dag.db`, Turso NATIVE storage format — not SQLite, engine-only access)

| Table | Key | Other columns |
|---|---|---|
| `nodes` | `o0, o1, blk, stm` = stones side 0, stones side 1, blocked squares, side to move | `n` visits; `w, d, l` results **for the side to move in that row**; `min_ply` earliest ply (0-based) the position appeared at |
| `edges` | parent key + `mv` (text: from-square + to-square like `a1a2`, clones too, or `pass`) | child key `c0, c1, cblk, cstm`; `n, w, d, l` for the parent's mover |
| `games` | `src, idx` | `header`, `book`, `titanium_is` (`x`, `o`, `both`), `result` (`x`, `o`, `draw`), `plies` |

Identities that always hold (`ti dag stat` checks them): `SUM(nodes.n) = SUM(games.plies) + COUNT(games)`
(every game adds its start position and one position per ply), `SUM(edges.n) = SUM(games.plies)`,
`n = w + d + l` in every row.

Proven on the original file by `ti dag verify`: 200,000 edges replayed through the engine, 0 wrong children;
on 27,044 results at full-board positions (winner known from the stones) 0 contradict "w = the side to move
won", 13,479 contradict "w = side 0 won".

Clone edges: rows imported before 2026-11-13 spell the source square as the log did; rows written by the hub
use the lowest neighbouring stone. To count per move, group clones by the child key, not by `mv`.

## 4. How to read it

- **Game file without any project code:** `docs/analysis/read_tgam.rs` (110 lines, standard library only).
  It reads the headers, decodes and applies every move with its own board. Run on `gate_games.tgam`:
  `57244 games, 5557890 plies ... illegal moves 0, passes 214676`.
- **Game file in Rust with the hub:** `gamedata::GameReader::open(path)` yields `PackedGame`;
  `game.replay()` gives every ply with the board before it, the move, score and depth;
  `game.values_side0(..)`, `game.z_side0()`, `lambda_returns(..)` give the value series and returns.
- **Rows:** read 48 bytes at a time (in the hub: `data::Rec::from_bytes`).
- **DAG by hand:** there is no hand tool. The file is Turso-native (no sqlite3/rusqlite/`tursodb.exe`
  access — all report "file is not a database"). Read-only inspection is `ti dag stat --db FILE`
  (counts + identity checks). Per-position lookups go through the engine (`conn.visits(&board)`).
- **DAG in Rust:** `dag::Dag::open(path)` (read + write) or `Dag::open_read(path)`; one `connect()` per
  thread; `conn.visits(&board)`; `conn.record_game(&GameRec { .. })`.

## 5. Training targets

The trainer computes the value target of a row itself (`targets.rs`):

`target = (1 - w) * sigmoid(cp / 400) + w * outcome`, with `w` = 0.55 at 0-12 empty squares and 0.25 above;
`cp` is clamped to +-1500 first; a proven row (`|cp| >= 98000`) is exactly 1 or 0.

**Lambda return** (uses the game link): `ti data lambda --tdat X --chains X.chains --out STEM --lambda L`
writes a copy of the rows in which `outcome` is replaced by the lambda-weighted mix of the later scores of
the same game and the final result. `--lambda 1` gives a byte-identical file. The trainer needs no change:
it just reads the new file. State: built and checked; the first training run with it did not gain
(reported -12.2 +-26.6 Elo, a single run).

**Policy head, advantage-weighted labels** (D765/D767): `ti policy2 ... --adv-lambda 0.9 [--adv-scale 0.10]
[--adv-clip 1]`. The policy head learns from the moves played in game logs. With this flag a move counts
more when the game then went better than the mover's own score promised, and less when it went worse:
weight = exp(clip(advantage / scale)), advantage = lambda return after the move minus the mover's score, in
win-probability units; weights are scaled to mean 1. It reads the game logs (`.jsonl`) directly; no extra
file is written. Every run prints, on held-out games, how often the head picks the played move among moves
the game proved right and among moves it proved wrong. First measurement (3,200 gate games): the gap between
the two grows from +0.47 to +1.00 points, overall top-1 falls 0.3 points. With the full production recipe (Kata rows included) the weighted head is identical to the plain one on every validation line (D774), so it was not gated: a null result. Rule (D773): a move is never labelled best because an old game played it.

**Not built yet:** targets for the disagreement head, values for the moves that were NOT played.

## 6. Opening novelty and the DAG

`ti gen --novelty P --dag DB`: in the first P plies the engine searches its best move, then its best move
without that one, and so on (`--novelty-tries`, default 4). It stops at the first candidate whose resulting
position was never played, or when a candidate is more than `--novelty-max-loss` centipawns (default 60)
worse than the best move. It plays the candidate with the fewest visits; ties go to the better move. Every
finished game is written to the DAG in one transaction, so the next game, the other threads and every later
run see it. No opening book is needed.

Rules:

- **One process writes a DAG file at a time.** Parallel writing is between the threads of one process
  (`--threads`). Never run two `ti gen --dag` or `ti dag add` on the same file together.
- **A game enters the DAG once.** Either `ti gen --dag` writes it, or `ti dag add --games F.tgam` imports it,
  never both. Within one route a repeat is refused (same `src` + index), across routes it is not detected.
- **Games made elsewhere** (cloud): give the workers a copy of the DAG with `--dag-readonly`, then at home
  `ti data pack-games` and `ti dag add`.
- Never write `games_dag.backup-20261114.db` (the original). `ti dag init --db NEW --from OLD` makes a working copy of any DAG file.

## 7. Why Turso, and what it is not used for

Turso is an in-process database (a Rust crate, no server) with its own NATIVE storage format —
not SQLite. Stock `sqlite3`, `rusqlite` and the old `tursodb.exe` all report "file is not a database"
(verified 2026-11-14). The ONLY way to read or write a DAG file is the engine itself (`ti dag stat`,
`ti dag verify`, `ti dag add`, `ti gen --dag`); never point any other tool at the file.
Measured on this PC, 4 threads, 2.6 M position rows (D762):

| Operation | Rate |
|---|---|
| scan | 2.3 M rows/s |
| lookup of a position | 57,000/s on one thread, 163,000/s on four |
| writing games, plain journal | 2,300 row writes/s (writers wait for each other) |
| writing games, MVCC journal (`BEGIN CONCURRENT`) | 16,000 row writes/s = 178 games/s |
| 64 KiB blobs | read 1,052 MB/s; four parallel writers fall to 7 MB/s |

So the DAG lives in Turso in MVCC mode. The positions themselves do NOT live in SQL rows: a billion rows
would be 75 GB or more and a shuffled training pass would be lookups at 57,000/s per thread. A billion
positions as game files are about 4.4 GB, written as one file per generation worker (no contention at all)
and read sequentially.

## 8. Health checks before trusting a file

| Command | Passes when |
|---|---|
| `ti data games-stat --tgam F.tgam --by-source` | no source is lopsided; the share of new positions is high |
| `ti dag stat --db X.db` | prints `identities OK` |
| `ti dag verify --db X.db` | `wrong child 0`, and 0 results contradict the side-to-move view |
| file size of `X.tdat` / 48 | equals `rows` in `X.manifest.json` |

## 9. Not done, not checked

- No train / validation split tool for game files. Rule when it is built: split by game (better: by start
  position), never by row.
- No values for moves that were not played; no policy or disagreement targets (section 5).
- `gate_games.tgam` has only 8,837 start positions for 57,244 games; self-play with novelty is the fix.
- Whether the gate games imported into the working DAG overlap the games already in the original DAG was
  not checked (different log names; an overlap would only inflate visit counts).
- Two processes writing one DAG file: not supported, not tested.
- The cloud build of the hub with the Turso crate (Modal image) was not rebuilt or tested.
- Building on this PC needs `C:\gitProjects\toolchains\llvm-mingw-20261006-ucrt-x86_64\bin` on PATH.

## train_v14 (2026-11-18, D1253)
- = train_v13 + hunt proofs (47-48 stones) + proof back-propagation along game chains (outcome = 0.9^n x proof + (1-0.9^n) x pooled WDL of repeated positions) ; held-out tail unchanged (val_v14tail == val_v13tail).
- Files: data/nnue/packed/train_v14.{tdat,chains,kata,aux,manifest.json}; train_v14h.tdat (training head, fnv 5E692E0EA0DB92E4); val_v14tail.tdat. Original labels: train_v13 (same row order, read-only).
- Build: ti proof-hunt --apply-tdat (head-only queue data/proofs/v13_apply_head) -> cat tail -> ti data proof-wdl --decay 0.9 --preserve-tail 1528091 -> ti data split. Numbers: DECISIONS D1253.
