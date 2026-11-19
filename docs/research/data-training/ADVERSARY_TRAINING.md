# ADVERSARY TRAINING RESEARCH

Date: 2026-10-21
Purpose: accumulate what the strong Ataxx engines actually do, so Titanium can
steal the useful parts instead of rediscovering them.

Status: measured results are 4-game smokes unless stated otherwise. They prove
direction/plumbing, not Elo. Gates still decide merges.

## TL;DR

- **Ataxx_CPP** (EmptyBlueBox/Ataxx, commit `d412e90d`) is the strongest
  local/open adversary measured so far: 4-0 vs Titanium (1s), 4-0 vs KataAtaxx
  b18 (50 visits), 3-1 vs Moonbird (1s). Its author reports 16/20 vs `atx` at
  3.94s with the endgame phase nets.
- **Ataxx_CPP is a distilled small transformer + graph/PUCT MCTS**, not
  alpha-beta: 6 transformer blocks, 64 channels, 4 heads, 14 input planes,
  INT8/VNNI inference, plus four phase-specialized nets by empty-square band.
- **Moonbird** is alpha-beta + large 4x2/2x4 tuple tables tuned by SGD
  (25 MB params), not NNUE. It is the closest existing engine to Titanium's
  tuple-eval line.
- **KataAtaxx** is the teacher used by Ataxx_CPP: b12c384h6tfrs2 transformer,
  one net, KataGo selfplay data (~91.5 GB), 200 visits in the release selfplay
  config. It is still the best open **single-net** reference.
- **atx** is #1 on Botzone but closed server-side only. Its descriptions say
  NNUE + persistent tree search, VNNI-accelerated, author-trained NNUE; no
  public source, no public weights.
- **Stockfish/Lc0 recipe that transfers**: first train on engine selfplay data,
  then retrain/fine-tune on stronger-teacher-rescored data; teacher-only data
  from scratch is worse. Quality/learnability beats raw eval. Use
  quantization-aware training and judge by games, not validation loss.
- **For Titanium**: use Ataxx_CPP's MCTS as the "Lc0" teacher, not its raw
  transformer alone. Two stages: Titanium selfplay/base labels, then
  Ataxx_CPP-rescored data. Add phase buckets only after the data supports them.

## 1. Adversary matrix

| Engine | Access | Eval | Search | Training | Buckets | Measured vs Titanium |
|---|---|---|---|---|---|---|
| atx | Botzone only, `opensource:false` | NNUE (spatial + relational, VNNI) | persistent tree search / graph MCTS variants | author data; versions mention V9/V28 checkpoints, step 2048 | not public | not runnable locally |
| KataAtaxx 2026 | open release + data | b12c384h6tfrs2 transformer, ONNX | KataGo PUCT graph search | KataGo selfplay, 91.5 GB, Muon, 2 GPUs | one net | 0-4 at 20 visits/6s Titanium |
| Ataxx_CPP | open repo + weights | 6x64x4 transformer, INT8 | graph/PUCT MCTS, Gumbel root, persistent tree | distill KataAtaxx + selfplay, 4 phase nets | 4 empty-square bands + champion | 4-0 at 1s |
| Moonbird | open binary/source | 4x2/2x4 tuple tables, SGD | alpha-beta + TT/NMP/LMR/aspiration | selfplay 6k nodes, 50M samples, SGD | no phase nets | 1-3 at 1s |
| AutaxxNNUE | open engine + weights | NNUE | alpha-beta | external NNUE weights | no | 1-3 at 1s |
| Fairy-Stockfish | open | HalfKAv2 NNUE, Ataxx net 245 KB | alpha-beta | variant-nnue-pytorch | Stockfish output buckets | not tested |
| ataxx-zero-ai | open HF checkpoints | 6x128x8 transformer | MCTS, 420 sims | 40 iters, weak curriculum | no | not competitive (0/12 vs hard) |

## 2. Ataxx_CPP: the most important open reference

Source: `data/engines/ataxx_cpp`, commit
`d412e90d549ec1fb10cd96f998dc1db2064e740c`.

### 2.1 Network

`ataxx/model.py`:

- Input: 14 planes on 7x7. Own, opponent, empty, ones, near/far own/opponent
  floods, local counts, ply/401, own fraction, opponent fraction, empty
  fraction.
- Trunk: 3x3 conv stem to 64 channels, then **6 transformer blocks**.
- Block: LayerNorm -> QKV (4 heads) -> learned 49x49 attention bias -> residual
  projection; LayerNorm -> 2x MLP (GELU) -> residual.
- Policy head: 1x1 conv to 32, plus pooled 2*C global projection, then 1x1 conv
  to **17 move kinds x 49 destinations**.
- Value head: 1x1 conv to 32, pooled 64, MLP 64 -> 4 outputs
  (win/loss/no-result/score), plus an ownership auxiliary head.
- Symmetry: 8-fold (x/y/transpose) augmentation and symmetry tables.
- Export/int8: `Dense8` stores int8 weights and uint8 activations; AVX2 and
  AVX-512/VNNI kernels. `champion.bin` and phase nets are ~553 KB each.
- Phase nets: same architecture, retrained on **empty-square** bands:
  `p0-8`, `p8-15`, `p15-30`, `p30-50`; `champion.bin` is the all-phase net.
  At inference `Engine` picks the phase by `empty` count; if no phase matches,
  it uses the champion.

### 2.2 Search

`src/search.h`, `src/engine.cpp`:

- AlphaZero-style **graph/PUCT MCTS**, not alpha-beta.
- Root uses Gumbel selection phases (`gumbelStart`, `gumbelPick`, completed-Q
  re-ranking) for low-visit strength.
- Persistent subtree: `advance(move)` rewrites the tree to the chosen child;
  tree is reused across moves.
- Transposition cache (`cache`, limit 200000) stores policy/value per position.
- Terminal proving (`proven`) and exact PV backup; virtual loss/busy nodes.
- Defaults around `cpuct=1.0`, `cpuctLog=0.45`, `fpu=0.2`.
- Selfplay uses the same search; data collection records MCTS-improved policy,
  q value, outcome, final board, score.

### 2.3 Training pipeline (exact shape from `ataxx/train.py`)

1. Build `build/libataxx.so`.
2. `selfplay.run` generates raw rows with the current student:
   - 2048 games, 8 threads, 50k-row shards.
   - 25% of games use `full=400` visits, otherwise `fast=64`.
   - temperature 1.0 for first 20 plies, then 0.25; 15% of games start from a
     random opening; 8-fold symmetry.
   - Record `bb`, final `bb`, MCTS policy visit counts, side, `z` outcome,
     `score`, and `q` (MCTS improved value).
3. `kata.label` relabels every raw position with the **KataAtaxx ONNX teacher**:
   - teacher policy top-32 counts, quantized to uint16;
   - teacher q to int16;
   - rows are consumed from `run/raw` and written to `run/kd`.
   This is the "Lc0 rescore" equivalent: student selfplay positions, stronger
   teacher labels.
4. `fit` base:
   - batch 2048; AdamW; 420k steps; lr 1e-3; EMA 0.004; grad clip 1.0;
   - data window starts at 2M rows, grows to 120M; `ratio=4`, `start=200k`;
   - loss: policy CE + 1.5 * value CE + 0.5 * score MSE + 0.5 * ownership MSE;
   - value target blends teacher q with real outcome using `q_mix=0.75`;
   - `score` and `ownership` are auxiliary targets, masked when the final board
     is empty/unknown.
5. QAT: 100k steps, lr 1e-4, quantized net, 30M-row window; EMA; export.
6. Phase specialization:
   - first train `p0-15` (100k), `p15-30` (800k), `p30-50` (800k);
   - initialize `p0-8` and `p8-15` from `p0-15`, 800k steps each;
   - export `champion.bin` from QAT and `p0-8/p8-15/p15-30/p30-50.bin`.
7. Package: `bot.py`/`bot.cpp` pass phase specs as `file:lo:hi`; C++ engine
   loads all and routes by empty count. `ATAXX_NOVNNI=1` disables VNNI path.

Note: base training is done with `quant=False`; QAT then phase nets are
quantized. The engine's int8 path is what makes the 6x64 net CPU-viable.

### 2.4 Why it is strong

- Teacher labels from a 12-block/384-channel transformer, distilled into a
  6-block/64-channel student. This is a large compression ratio; the student is
  small enough for CPU MCTS.
- Teacher q + real outcome mix, not pure imitation.
- MCTS policy labels (visit distribution), not just the best move.
- Phase nets specialize without making each net bigger. Four nets + champion
  total ~2.8 MB, still small.
- Persistent graph MCTS reuses the tree across moves and caches transpositions;
  low-visit play is much stronger than stateless MCTS.
- Quantization-aware training preserves the int8 path.

### 2.5 Caveats

- Our 4-0/3-1 results are 4-game smokes. The author's 16/20 vs atx is their
  claim, not independently reproduced here.
- Ataxx_CPP default rules use `LOOPDRAW_PASSSCORING`; Titanium's classic gates
  allow a forced pass. Teacher labels must use the same rule as the student or
  outcomes near repetition/pass differ.
- Its raw transformer is not a drop-in Titanium eval: input planes, move
  encoding, and search targets differ.

## 3. KataAtaxx: the teacher and single-net reference

Source: `hzyhhzy/KataGomo` release `Ataxx_20260501`, branch `Ataxx-onnx`;
training scripts `scripts/ataxx7xtf1`.

- Model: `b12c384h6tfrs2-bng-silu`: 12 transformer blocks, 384 channels,
  6 heads, SwiGLU/FFN 1024, RMSNorm, RoPE; KataGo-style 22 spatial + 19 global
  inputs; 7x7 board; ONNX export.
- Search: KataGo MCTS/PUCT graph search, root noise, LCB selection, subtree
  reuse, NN cache; release selfplay `maxVisits=200`, 4000 game threads,
  `reducedVisitsMin=40`, random initial stones 20%, banned locations 10%,
  root Dirichlet noise, temperature schedule; 2 GPUs.
- Training (`run.sh`): loop 10 x (60k selfplay games -> shuffle -> train
  `b12c384h6tfrs2` batch 2048, lr-base 6e-6, lr-scale 0.25, 10M samples/epoch,
  Muon, 2 GPUs), then lr-scale decays 0.177/0.125/0.088/0.0625/0.03 and export
  ONNX. Selfplay data is ~91.5 GB (13 tar shards).
- No phase buckets: one net at a time; `switchNetsMidGame=false`.
- Strength: +200 Elo vs the 2025 b18 model; needs 1/4 the time per move.
- Role for us: **teacher** (Ataxx_CPP already uses it this way), not a local
  gate adversary unless we want a single-net baseline. It is stronger than
  Titanium but weaker than Ataxx_CPP in our smoke.

## 4. Moonbird: the tuple-eval reference

Source: `data/engines/moonbird_src` (Nim), binary
`scripts/moonbird/Moonbird-1.1.0-windows-amd64.exe`.

- Alpha-beta search: TT, nullmove, PVS, LMR, futility, aspiration, move
  ordering/history.
- Eval: large **4x2 and 2x4 tuple tables** plus 2x2 structure, mobility by
  occupancy phase, environment counts, turn bonus. Tuned by SGD.
- Params: `res/params/default.bin` 25,766,556 bytes (25 MB) — far larger than
  Titanium's 726 KB tuple net.
- Training: `nim genData --run Moonbird 6_000 50_000_000 false` (6k-node
  sample games, 50M target samples), then `tuneEvalParams` + weather factory
  for search params.
- No NNUE, no phase networks; one eval table with phase-indexed mobility.
- Measured: lost 1-3 to Ataxx_CPP at 1s. Still much stronger than Titanium;
  its tuple-pattern lessons are directly relevant.

## 5. atx: the closed champion

Botzone bot `6aa07f142418e3081fb60ede`, owner `sqz`, `opensource:false`.
Version descriptions are the only public architecture clues:

- v2: "NNUE spatial evaluator + tree search; persistent model and subtree
  reuse; 3.8s move budget."
- v3: "Checkpoint 12288; spatial and relational NNUE; VNNI; persistent tree."
- v4: "V9 step 2048; fixed-rule NNUE; VNNI-512 acceleration; persistent tree."
- v5: "V28 Control step 2048; author-data NNUE; VNNI acceleration; persistent
  tree."
- v6: "V28 NNUE with graph MCTS and visit catch-up; persistent CPU search."
- v9: "Neural evaluation with persistent tree search."

Interpretation, not fact: custom NNUE with spatial/relational features, trained
on author data at checkpoints/steps (12288, 2048), VNNI int8 inference, and a
persistent graph/tree search with visit catch-up; likely a Python front-end
around a compiled core. No public source/weights; cannot use as teacher.
Botzone login `Terminator_PL` works, but source is owner-only.

## 6. Fairy-Stockfish and ataxx-zero-ai (open but not leading)

- Fairy-Stockfish: alpha-beta + HalfKAv2 NNUE; Ataxx variant net
  `ataxx-e631fe1b1b6d.nnue` 245,728 bytes; trainer
  `fairy-stockfish/variant-nnue-pytorch`; Stockfish-style output buckets.
  It is a general variant engine, not Ataxx-specialized; useful as a data/training
  reference, unlikely to be the strongest local opponent.
- ataxx-zero-ai: transformer 6 layers, d_model 128, 8 heads, FFN 512; MCTS
  420 sims; 40 iterations, 70 episodes/iter, 50k replay, 85% self / 12%
  heuristic / 3% random. At iteration 39 it scored 0/12 vs the hard heuristic
  — a clear negative example: selfplay volume and curriculum matter more than
  architecture labels.

## 7. Stockfish/Lc0 recipe, and how to map it to Ataxx

Official nnue-pytorch wiki (`Training-datasets`):

- Stockfish selfplay generator: depth 9, 18B positions, 250 threads, hash
  10240, `random_multi_pv 4`, `random_multi_pv_diff 100`, `noob_3moves.epd`
  opening book. A fixed-node variant uses 5000 nodes/move.
- Lc0 data rescorer: run Lc0 on positions, convert/rescore to NNUE plain
  format. Lc0-derived data is higher quality and generally better, but
  **not from scratch**.
- Practical order: train first on Stockfish depth-9/5k-node data, then retrain
  an already-good net on Lc0-derived data. Lc0-only training is worse.
- Data quality vs learnability is empirical; validation loss is not the goal.
- Mixed/ordered datasets matter (e.g. T60T70 + SF + Leela).
- Use multiple seeds, quantization-aware clipping, and judge by games.

### 7.1 Ataxx translation

- **Student**: Titanium tuple/NNUE net (726 KB tuple currently).
- **Teacher**: Ataxx_CPP MCTS at high visits (400-4000+), because it is the
  strongest open engine measured and already distills KataAtaxx. Use its
  **search policy visits**, q, score, and ownership — not only raw net eval.
- **Data**:
  1. Titanium selfplay / book / existing cache12M positions (5k labels) as
     the broad, learnable base.
  2. Ataxx_CPP selfplay (or Ataxx_CPP on Titanium positions) for coverage,
     especially opening/endgame.
  3. Relabel with Ataxx_CPP high-visit search; keep top-k policy, q, score,
     ownership; mix real outcome (0.5-0.75 q mix, as Ataxx_CPP does).
- **Rule alignment**: use Ataxx_CPP's `LOOPDRAW_PASSSCORING` for teacher games,
  or convert labels to Titanium classic semantics; near repetition/pass the
  labels otherwise disagree.
- **Two-stage**:
  1. base student on 5k/selfplay labels + outcome anchor;
  2. fine-tune on Ataxx_CPP-rescored data (not teacher-only from scratch);
  3. optional selfplay RL / teacher refresh;
  4. phase-bucket specialization;
  5. QAT if the loader quantizes.
- **MultiPV/soft policy**: use Ataxx_CPP top-k visit distribution. Single
  best-move labels were already shown weak/overfit in Titanium history.
- **Validation**: gate by games (5k nodes first for quality, 100ms/200ms for
  speed), not val loss. Keep outcome anchoring; pure teacher imitation has
  failed here repeatedly.

## 8. Buckets: what they did, what to copy

### 8.1 Ataxx_CPP (separate phase nets)

- Four nets by **empty squares**: 0-8, 8-15, 15-30, 30-50; plus champion for
  fallback. This is piece-count/phase specialization.
- Each net starts from the QAT champion, then trains on data filtered to its
  band. The phase labels come from `Data(phase)`:
  `empty = 49 - popcount(bb[0]|bb[1])`.
- Engine routes at the root by empty count; a phase net can be selected per
  move. Total model size ~5x553 KB, still small.

### 8.2 Stockfish (output buckets)

- One network, shared trunk, **output buckets** (8 piece-count buckets); not
  separate full nets. Cheaper, but less specialization.

### 8.3 Titanium options

1. **Four separate tuple nets** (Ataxx_CPP style), initialize from champion,
   train on phase-filtered data; route by empty count at root. Pros: simple
   training, proven by Ataxx_CPP. Cons: 4x file/loading, boundary moves,
   sparse phase data.
2. **One tuple net with output buckets** (Stockfish style): shared features,
   bucket-specific final weights. Pros: compact, smooth. Cons: requires eval
   loader/architecture change; tuple net is not a normal dense net.
3. **Two buckets first** (opening/endgame) if data coverage is thin; add 4
   only after opening/endgame data exists. Current Titanium cache is ~84%
   midgame, 5% opening, 11% endgame; phase nets would starve without new data.

Recommendation: do **not** jump to 4 buckets on the existing 5k cache. First
build an opening/endgame corpus with the Ataxx_CPP teacher; then test 2 vs 4
buckets as an experiment, with speed parity and a 5k-node gate. Buckets are a
capacity/specialization trick, not a free Elo button.

## 9. Proposed Titanium experiment plan

### Stage A - teacher/data plumbing

- Add a teacher-label path around Ataxx_CPP command mode:
  `new` -> `play ...` -> `go evals N` / `go time S`; parse move, visits, q.
- Mine positions from Titanium cache12M, book50k, and Ataxx_CPP selfplay.
- For each position store: FEN/bb, side, teacher top-k policy, q, score,
  ownership if available, search visits, rule tag, empty count.
- Deduplicate by position/hash; cap quiet repetition; balance phases.

### Stage B - small-net distillation

- Train Titanium's tuple net on the base corpus with outcome anchor and
  teacher soft policy; do not train on teacher-only labels.
- Add phase bucket heads/nets only after Stage A gives enough opening/endgame
  rows.
- Quantization-aware if the target path is int8.

### Stage C - selfplay/RL

- Use the strongest current Titanium + Ataxx_CPP teacher for exploration.
- Keep the FOMO rule: no net rejected because val loss is worse; gate by games.

### Stage D - gates

- 5k-node quality leg, then 100ms/200ms time legs; 100+ games; paired book;
  user approval before merge.
- Track exe hash + net hash + refs; no dirty-tree gates.

## 10. Anti-lessons from Titanium history (do not repeat)

- Validation loss lies: retrains repeatedly improved val and lost games.
- Value-only golden doses (3/6/10%) did not convert; 3% parity, 6/10% worse.
- Pure outcome/regret/online loops were parity or worse; no free Elo.
- Self-distillation/echo chambers fossilize at scale.
- Smaller net + more buckets can lose to a bigger/simpler net if data is
  sparse; test, don't assume.
- 4-game results are smoke; only 100+ game paired-book gates prove direction.
- Teacher and student must share rules and move encoding; LOOPDRAW vs classic
  pass changes labels and outcomes.

## 11. Source index

- Ataxx_CPP: `github.com/EmptyBlueBox/Ataxx`, commit `d412e90d...`; local
  `data/engines/ataxx_cpp`; training `ataxx/{model,train,selfplay,kata,package}.py`;
  engine `src/{engine.cpp,tf.h,search.h}`.
- KataAtaxx: `github.com/hzyhhzy/KataGomo` releases `Ataxx_20260501`,
  `Ataxx_20250131`; branch `Ataxx-onnx`, `scripts/ataxx7xtf1`.
- KataGo_Transformer: `github.com/hzyhhzy/KataGo_Transformer` (training/ONNX
  tooling; Ataxx v206 profiles).
- Moonbird: `github.com/tsoj/Moonbird`; local `data/engines/moonbird_src`.
- atx: Botzone ranklist/detail, owner `sqz`.
- Fairy-Stockfish: `github.com/fairy-stockfish/Fairy-Stockfish`;
  `variant-nnue-pytorch`; Ataxx net `ataxx-e631fe1b1b6d.nnue`.
- ataxx-zero-ai: `github.com/Dieg0Code/ataxx-zero-ai`; HF
  `dieg0code/ataxx-zero`.
- Stockfish data recipe:
  `github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets`.
- Measured logs: `logs/ataxx_cpp_4g_1s.txt`,
  `logs/ataxx_cpp_vs_kata_4g.txt`, `logs/ataxx_cpp_vs_moon_4g.txt`.
