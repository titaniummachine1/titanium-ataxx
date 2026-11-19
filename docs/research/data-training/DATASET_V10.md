# Dataset v10: games, not positions (D750, D753, D756, D757; 2026-11-13)

> **Newcomers: read `docs/DATASET_GUIDE.md` first** (formats, parsing, commands). The novelty / DAG part of section 7 below (hex `--seen` file, rusqlite) is superseded by the guide's sections 6-7 (D763).

User order: the dataset must be linked to its games (game id, ply, each position's future), packed with the engine's
own bit codes, and it must be able to carry per-move root values for the policy. This file is the spec and the
first results. Code: hub `titaniumtest/src/gamedata.rs` (uncommitted in the hub repo, like the other hub changes).

## Why (measured on train_v9, 28,990,531 rows)
- A v9 row (48 B + 16 B .kata + 12 B .aux = 76 B) has stones, side to move, result, scores, teacher value, flags.
  It has NO game id, ply, final stone counts, move played or per-move values.
- The score is an integer in 100% of rows (non-proven max 2877): it fits the engine's 16-bit score code. `cp5k`
  equals `cp` in 93.21% of rows. The five flag bytes take 50 combinations.
- v9 was de-duplicated by position (914 distinct `wsum` values = sqrt of the merge count; 25% of rows have a
  fractional outcome). That merge is what broke the game links.

## What exists now

### 1. Run index for an old `.tdat`: `ti data chains --tdat X.tdat [--out X.chains] [--min-len 2]`
Finds every run of rows where each row is exactly one legal move after the previous one. Nothing is rewritten.
File `.chains`: "TCHN", u32 version 1, u64 rows, u64 runs, then 8 B per run: start row u32, length u16,
convention u8 (0 = occ0 is the side to move, 1 = occ0 is colour 0 and `stm` names the mover), 0.

Result on train_v9 (0.6 s): 17,119,714 rows (59.1%) are one move after the previous row.

| run length | runs | rows | share of file |
|---|---|---|---|
| 1 | 10,165,022 | 10,165,022 | 35.1% |
| 2-9 | 1,053,978 | 4,478,596 | 15.4% |
| 10-29 | 515,895 | 8,697,660 | 30.0% |
| 30-59 | 125,224 | 4,907,541 | 16.9% |
| 60-99 | 10,547 | 725,458 | 2.5% |
| 100+ | 151 | 16,254 | 0.1% |

Index written: `data/nnue/packed/train_v9.chains`, 1,705,795 runs of >= 2 rows covering 18,825,509 rows (64.9%),
13.6 MB. The linked part is teacher_kind 1 (first 70% of the file, 83% of its rows linked); teacher_kind 2
(last 30%, 8.5M rows) is shuffled. val_v9 is shuffled (1.3% linked; `val_v9.chains` written, nearly empty).
Outcome flips 0/1 between linked rows in 94.6%.

### 2. Game file: `ti data pack-games --out STEM [--roots DIR,..] [--inputs a.games.jsonl,..]`
`.tgam`, little endian: header 32 B ("TGAM", version 1, games u64, plies u64, sources u32, 0), then per game
a 40 B header and 4 B per ply.
- Game header: occ[0], occ[1], blockers (3 x u64 = start position), n_plies u16, source index u16, flags u8
  (bit 0 start side to move, bits 1-2 candidate seat, bit 3 forfeit), final stone counts 2 x u8, result u8
  (side 0: 2 won / 1 draw / 0 lost), book u32, pair u32.
- Ply: u16 = 10-bit move code (`titanium::ttcodec::move_code`, the transposition-table code: clone = destination,
  480 jump pairs, pass 0x3FF) + 6 bits search depth; i16 = 16-bit score code (`enc_score`, mover's view, mates
  included; -32768 = no score logged).
- `STEM.tgam.json`: every source file with its gate header (both engines, exe hashes, clock, book), so games can be
  filtered by engine or clock.
- Game id = index in the file; ply = index in the game; the future of a position = the rest of its game.
- Every game is replayed before writing, and each move code is decoded back on its position and must give the
  same next position.

Result: 177 gate files (699 MB of JSON) -> `data/datasets/v10/gate_games.tgam`: 57,244 games (6,042 exact
duplicates dropped, 0 skipped), 5,557,890 positions, 24.5 MB = 4.41 B per position (v9: 76 B).

### 3. Check: `ti data games-stat --tgam F.tgam [--lambda 0.7]`
On `gate_games.tgam` (0.3 s): all 57,244 games replay through the engine's decoder; mean 97.1 plies per game;
side 0 won 27,856 / draw 2,343 / lost 27,045; a score on 93.8% of plies, mean depth 12.1; recorded final counts equal
the replayed final board in 56,201 games (the rest were scored by the gate's end-of-game rule).

Mean squared error against the final result, 5,212,039 scored plies:

| ply | mover's own value | lambda 0.7 return |
|---|---|---|
| 0-19 | 0.2368 | 0.2353 |
| 20-39 | 0.2279 | 0.2248 |
| 40-59 | 0.2003 | 0.1857 |
| 60-79 | 0.1106 | 0.0851 |
| 80+ | 0.0462 | 0.0348 |
| all | 0.1715 | 0.1606 |

The lambda return contains the result itself, so being closer to it is expected; this table is a sanity check of
the pipeline, NOT evidence that training on it gains strength.

### 4. Library (same file)
`GameReader` (streaming), `PackedGame::replay()` (positions + move + score + depth per ply),
`values_side0()`, `lambda_returns(values, z, lambda)`, `read_chains()`.

## 5. Trainer wiring (D758/D759): the future goes into the row
The trainer's mapper sees one row at a time (`gpu.rs:146`, `target_train_rw(row)`), target =
`(1 - w) * sigmoid(cp) + w * outcome` (champion: w = 0.55 / 0.55 / 0.25 by empties). So no trainer change:

`ti data lambda --tdat X.tdat --out STEM --lambda L [--chains X.chains] [--horizon 64]` writes a copy of the file
whose `outcome` holds the truncated lambda return of the row's own run: weight `(1 - L) * L^(n-1)` on the search
value `n` plies later (flipped to the row's side for odd `n`), the remaining weight stays on the row's own game
result; a proven later row is exact truth and takes all the remaining weight. Proven rows, unlinked rows, positions,
`cp` and everything else are copied unchanged. Any `ti train` run (local or Modal) trains on it as on any `.tdat`;
validation on val_v9_clean stays comparable.

Checks: lambda 1 reproduces train_v9.tdat BYTE FOR BYTE (sha256 C99081FD6C87A4AF... on both); unit tests for the
weights, proofs, lambda 0 and the unlinked row. Written:

| file | rows changed | mean abs change of `outcome` | mean weight left on the game result |
|---|---|---|---|
| data/nnue/packed/train_v9_lam07.tdat | 15,509,003 (53.5%) | 0.351 | 0.145 |
| data/nnue/packed/train_v9_lam09.tdat | 15,509,003 (53.5%) | 0.244 | 0.358 |

## 6. Quality audit (D759)

### train_v9 runs: `ti data chain-audit --tdat X.tdat [--deep D.tdat]`
One-ply consistency of the 5k labels (own win probability vs the next row's, both unproven):

| empties | pairs | mean abs diff | diff > 0.10 | diff > 0.25 | side ahead flips |
|---|---|---|---|---|---|
| 0-9 | 2,887,178 | 0.0644 | 19.2% | 3.1% | 7.9% |
| 10-19 | 5,194,111 | 0.0396 | 6.1% | 2.3% | 3.1% |
| 20-29 | 4,581,361 | 0.0387 | 6.5% | 2.2% | 3.2% |
| 30+ | 2,588,960 | 0.0342 | 6.2% | 2.0% | 2.9% |

The last 10 empties are the noisy part of v9.

Against the 40k-node relabel (logs/sbx_late1/late_40k.tdat): 60,751 of its rows sit inside runs. Mean squared
error of each candidate target against sigmoid(40k score):

| target | mse | vs own 5k label |
|---|---|---|
| own 5k label | 0.00531 | 0 |
| next ply's label | 0.00935 | +76% |
| 0.5 own + 0.5 next | 0.00499 | -6% |
| 0.5 own + 0.5 later labels (lambda 0.7) | 0.00340 | -36% |
| game result alone | 0.19450 | +3564% |
| champion target (own + result) | 0.02748 | +418% |
| own + lambda 0.5 return | 0.00383 | -28% |
| own + lambda 0.7 return | 0.00428 | -19% |
| own + lambda 0.9 return | 0.00855 | +61% |

Reading: the later labels of the same game carry real information about the deeper search (own + later is a third
closer to the 40k label than the 5k label alone). LIMIT of this test: a deeper search cannot credit what the game
result knows, and gates showed that result weight matters (0.55 beat 0.25 by +23 Elo, 0.7 and 0.85 lost, less
result weight in the middlegame lost 32). With lambda 0.7 only 0.145 of the result slot stays on the true result.
So this table says "cleaner by the search yardstick", NOT "stronger". Only a fine-tune + gate decides.

### Gate games: `ti data games-stat --tgam F --by-source`
| clock | games | mean depth | mse vs result | one-ply abs diff |
|---|---|---|---|---|
| 100 ms | 31,800 | 11.6 | 0.1735 | 0.0246 |
| 5000 nodes | 18,752 | 12.0 | 0.1697 | 0.0295 |
| 1000 ms | 2,506 | 16.3 | 0.1688 | 0.0225 |
| 200 / 300 ms | 2,430 | 13.8 / 15.0 | 0.1665 / 0.1601 | 0.0286 / 0.0302 |
| other (50 ms, 2k-50k nodes) | 1,756 | 8.1-17.5 | 0.153-0.178 | 0.017-0.029 |

- Lopsided sources (candidate below 35% or above 65%): 19 sources, 2,162 games (3.8%). The rest is the champion
  line against a near-equal candidate.
- Ply-to-ply the gate scores are MORE consistent than the v9 5k labels (0.017-0.030 against 0.034-0.064).
- Only 8,837 different start positions for 57,244 games (6.5 games per start); 77.3% of positions are distinct:
  ply 0-19 55.0% new, 20-39 89.7%, 40-59 94.5%, 60-79 95.1%, 80+ 51.9%.

### Verdict
- Wiring: safe. It cannot damage the existing pipeline (old file untouched, lambda 1 is bit-identical, nothing in
  the trainer changed).
- train_v9 + lambda: worth one experiment, outcome open. Start with lambda 0.9, then 0.7. Fine-tune from the
  champion (C14), gate at 100 ms and 1000 ms. A wart to know: 53.5% of rows get the new target, 46.5% keep the
  raw result (unlinked rows), so one file holds two target definitions.

#### Correction (2026-11-14): measured tails, and lambda is NOT the result weight
- Two different knobs were being confused. **lambda** is the TD(lambda) bootstrap coefficient that generates this
  file (`G_t = (1-l) * sum_n l^(n-1) v_(t+n) + l^N z`). The **result weight** is `--result-weight-phase`
  (`target = (1-w) * sigmoid(cp) + w * outcome`), champion `w = 0.55 / 0.55 / 0.25` by empties. They multiply:
  training on a lambda file *without* `--result-weight-phase 0.55,0.55,0.25` silently uses the hard-coded 0.25.
- Measured mean tail = weight left on the true result, over the 18.83M rows inside `train_v9.chains`
  (1,705,795 runs, mean run 11.0 plies), recomputed from the chain index: **lambda 0.9 -> 0.484**,
  **lambda 0.7 -> 0.249**, lambda 0.55 -> 0.184. The earlier "0.36 / 0.145" figures in this doc were wrong
  (they ignored that most rows sit 1-3 plies from the run end). Including the 10.2M unlinked rows (tail 1.0)
  the file-wide means are 0.665 (0.9) and 0.511 (0.7).
- Consequence: lambda 0.9 keeps ~48% of the result slot real, which is *closer* to the champion's 0.55 result
  weight than 0.7 is. 0.9 is the conservative first run for that reason, not because of the old 0.36 claim.
- **Effective result weight (the number that actually matters):** with the champion weights, a linked row trains on
  `(1-w) * sigmoid(cp) + w * (mix + tail * z)`, so the true-result share is `w * tail`:
  lambda 0.9 + w 0.55 -> **0.266**; lambda 0.9 + w 0.25 -> 0.121; lambda 0.7 + w 0.55 -> **0.137**.
  The shipped no-lambda target is 0.25. So **lambda 0.9 with `--result-weight-phase 0.55,0.55,0.25` lands on
  essentially the result weight the engine already runs (0.266 vs 0.25) while adding the future labels**;
  lambda 0.7 halves it (0.137) and is the risky arm, consistent with our own result-weight history
  (0.55 beat 0.25 by +23 Elo, and cutting result weight in the middlegame lost 32).
- **External corroboration that val convergence is not strength:** an Amethyst (chess) SPRT of lambda 0.1 vs 0.25
  finished at Elo -102.3 +- 34.3 (276 games, LLR -2.27, H0 accepted) even though low lambda converged fast. Their
  lambda scale is not ours (theirs favours the search label), but the failure mode transfers: a target that learns
  faster can be a weaker target. No lambda value gets promoted on val loss here; only gates decide.
- Gate games: good labels, but a small add-on (5.2M scored positions), repeated openings, two engines per game.
  They need a validation split by book line, never by row. Not converted to trainer rows yet.
- What v9 really lacks stays the same: late-game label depth (last 10 empties) and per-move values.

## 7. The data flywheel (2026-11-13): games -> training rows in one command

The dataset is dominated by games played by an engine ~1,000 Elo weaker than today's. Positions come from that
play, so the *state distribution* is stale even though every row was relabelled. Fresh games could be packed
(`ti data pack-games` -> `.tgam`) but never became trainer rows, which is the link that was missing.

    ti gen --out fresh/shardN.jsonl --games 12 --time-ms 100 [--novelty 8] [--dag FILE]
    ti data pack-games --roots fresh --out fresh                 # .tgam (4 B/ply, game level)
    ti data games-tdat  --tgam fresh.tgam --out packed/fresh \
            [--scores search] [--limit-games N] [--relabel-src 1]  # .tdat + .chains + manifest

`ti gen` (hub, `src/gen.rs`) plays the games in-process under Botzone rules — the Modal image builds only `ti`,
and `ti sprt` needs an external engine binary, so this is what makes generation possible on the cloud at all.
The engine CLI has the same thing as `selfplay --jsonl --book --book0`.

### Labels come from the game, not a second search

`games-tdat` **defaults to the eval each engine already produced on every ply** of the game (`--scores search`
opts back into re-labelling). Measured on 686 games / 61,036 plies:

| path | wall | note |
|---|---|---|
| re-search every position at 5k | ~23 min | redundant: the game is already a labelled run |
| in-game evals (default) | **3.0 s** | ~460x faster, and the labels are the deeper in-game values (depth 8-16) |

Result: `fresh698.tdat` = 61,036 rows, 686 chains, 4,894 proven (8.0%), `teacher_kind=2`, `relabel_src=1`.
`train_v9.*` is untouched (sha256 `C99081FD...`, 28,990,531 rows).

### Cost (measured, not estimated)

8.2 core-s per 100 ms game; Modal Function CPU is $0.0000131/core-s ⇒ **~$0.000107/game, ~$1.1 per million
labelled positions**, labels included because they are free.

| spend | games | positions | wall at 100 x 1-core containers |
|---|---|---|---|
| $2 | ~18,600 | ~1.8 M | ~25 min |
| $5 | ~36,000 | ~3.6 M | ~50 min |
| $10 | ~73,000 | ~7.2 M | ~1.6 h |

Containers are **1 core**: the generator is single-threaded, so a 2-core container bills for an idle core.

### Opening coverage without a book: the DAG

`ti gen --novelty P` replaces the engine's move with the next-best root move whose child is unseen, for the
first `P` plies. Verified: four games from the same start produce four different openings (`a7b6` / `g1f2` /
`g1f1` / `a7a6`). The backing store today is a hex text file (`--seen` / `--seen-out`); the real target is
`data/dag/games_dag.db` — see its manifest (`data/dag/games_dag.manifest.json`) for the schema, the proven
`n == w+d+l` identity, and the four facts that are still UNVERIFIED (creator, date, w/d/l perspective,
`min_ply`).

Rules that keep the mix honest:
- Fresh rows are stamped. Never mix generations without being able to filter by `relabel_src`.
- Replay only games with a decided result (`forfeit`/`unknown` are skipped, counted in the summary).
- A fresh run is a *distribution* fix, not a volume fix: the net saturated on 29 M rows of 5k labels, so judge a
  new generation by a gate, never by row count.
- Never re-search a game to label it. The only reason to spend a second search is `--scores search` explicitly.

## NOT done / not checked
- No net was trained on the lambda files, no gate. `.tgam` games are not converted to trainer rows.
- Root-move value vectors (16 bit per legal move, mean 45.6 moves per position = 91 B, or top-10 = 33 B): the
  format has no slot for them yet and none were generated (about 97 core-hours per 1M positions at 1,000 nodes per
  move; the 1-ply version is nearly free).
- The gate games mix many engine versions and clocks (weak experimental candidates included). Use the manifest to
  filter before training; no filter was applied.
- Kata arena games, Botzone games and data/dag/games_dag.db were not packed (other formats).
- No DAG index over the shuffled teacher_kind 2 rows; not tested whether those rows have successors in the file.
- A `.tdat` run is a fragment where a de-duplicated position was dropped; whether a run starts at a game start is
  not recorded.
