# What the strong Ataxx engines look at (2026-10-24) — measured, not guessed

Read-only analysis, no training. Scripts: `training/analyze_moonbird.py`, `training/probe_concepts.py`,
`training/probe_kata_internals.py` (+ `probe_kata_ownership.py`, result discarded, see 6). Positions: the 100k
stratified KataGo calibration rows (or subsets), side-to-move relative.

## 0. Answer: what to look for

1. **Everything that matters is within 2 squares.** Kata's attention concentrates on distance 0-1 (3-4.7x a uniform head)
   and distance 2 in specific layers (L3, L6, L8, L11: 1.7-2.3x); distance >= 3 is suppressed (0.2-0.6x).
   Ataxx_CPP hard-wires the same: some heads biased to the 3x3 ring (+1.6..+2.5), others to the distance-2 jump ring
   (+2.0..+2.8), distance >= 3 negative. The board's important geometry is the **3x3 capture ring and the 5x5 jump reach**.
2. **The first thing Kata's net computes is stone vulnerability.** Linear probes on its token states:

   | concept (per square) | linear on raw 5x5 patch | Kata layer 0 | layer 1 | layer 11 |
   |---|---|---|---|---|
   | my stone exposed (enemy can land next to it) | 0.47 | 0.91 | 0.92 | 0.80 |
   | enemy stone exposed (I can land next to it) | 0.45 | 0.85 | 0.90 | 0.80 |
   | empty square reachable only by my jump | 0.40 | 0.67 | 0.91 | 0.81 |
   | 2-ply: clone gain minus enemy's best recapture | 0.57 | 0.64 | 0.79 | 0.53 |
   | enemy stones around an empty square (local count) | 1.00 | 1.00 | 1.00 | 0.82 |

   Exposure and jump reach cannot be read linearly from the raw neighbourhood, and Kata builds them in its first
   1-2 layers. The local exchange value (what I win by landing minus what the enemy wins back) also peaks at layer 1.
3. **The strong engines' values depend on safety and exposure; Titanium's do not.** R2 gained by adding one concept
   to material (empties 25-35; similar in 16-24):

   | concept | Kata 200v | Ataxx_CPP net | Moonbird eval | Titanium 5k |
   |---|---|---|---|---|
   | safe stones (no empty neighbour) | +0.072 | +0.136 | +0.135 | **+0.000** |
   | exposed stones | +0.069 | +0.114 | +0.098 | **+0.008** |
   | attack potential (sum of captures over my landing squares) | +0.044 | +0.072 | +0.050 | **+0.002** |
   | best single capture | +0.022 | +0.047 | +0.041 | +0.065 |
   | material alone (R2) | 0.22 | 0.29 | 0.32 | **0.67** |

   Titanium's search values are mostly material + best single capture. They are blind to stone safety, the concept
   every stronger engine leans on. This is also why Titanium-labelled base data cannot teach it.
4. **Tempo is large:** Moonbird's side-to-move bonus is +251 (k = 568 per factor 10 in odds) ~ 2.8x odds.
5. **Mobility matters late, not in the midgame:** Moonbird's reach-count table: with 46 squares filled, 0 reachable
   squares = -565 (lost), with 36 filled 0 = -388; in the opening/midgame the table is small. Once safety and pairs are
   known, dropping mobility costs almost nothing (ablation below).

## 1. Moonbird 1.1.0 (verified exact)

- The binary we play is tag **1.1.0**, eval = ternary 2x2-pair tables 4 x 49 x 6561 + stone-neighbourhood
  env[49][9][9] + mobility[50][50] + tempo, **2.58 MB** (`data/engines/moonbird_1.1.0/`, fetched from GitHub; its
  params are embedded byte-for-byte in the exe). `data/engines/moonbird_src` is a NEWER unreleased checkout with
  4-state cells (25.7 MB); docs/ADVERSARY_TRAINING.md's "25 MB, 35x our params" is wrong for the engine we test.
- Re-implementation reproduces Moonbird's own depth-1 search score exactly (300/300 positions, pruning off).
- Training (source): own 6k-node (and 1k) games, one sample every 20-40 plies (heavily decorrelated, many games),
  label = 0.5 * result + 0.5 * search win-prob, MSE on sigmoid(eval / 568 * ln10), per-sample SGD, 2 epochs.
- Ablation (MSE vs Kata TD h1 after refitting the scale; lower = better):

  | empties | full | drop pairs | drop env | drop mobility | material+tempo |
  |---|---|---|---|---|---|
  | 16-24 | 0.0427 | 0.0511 | 0.0580 | 0.0435 | 0.0546 |
  | 25-35 | 0.0336 | 0.0492 | 0.0552 | 0.0338 | 0.0447 |
  | 36-49 | 0.0300 | 0.0469 | 0.0434 | 0.0302 | 0.0327 |

- What env learned (stone value by own/enemy neighbours): stones with many empty neighbours are weak, stones whose
  ring is filled are strong (inner own5/enemy0 +56, own8 +40; corner stone with all 3 neighbours filled +123..+126
  even when they are enemy stones); inner stone with 3 enemy neighbours and empties around it -37. **Env = stone safety.**
- Our tuple net has the same families (pairs `pa`, `env`, `mob`) but our `mob` port counts clone targets only
  (distance 1); Moonbird counts all squares within distance 2. Port bug.
- So Moonbird's edge at equal nodes is not table size: same families, ~3.5x our bytes. Prior ledger (S39) also points
  at search + training data.

## 2. Ataxx_CPP

- Author hand-feeds exactly the vulnerability geometry as input planes (`src/board.h encode`): own/opp/empty, clone
  reach per side, jump-only reach per side, per empty square enemy-neighbour count and own-neighbour count, ply and
  stone counts. Learned attention biases: heads specialised on the 3x3 ring or the distance-2 ring (table in 0.1).

## 3. KataGo 2026 (b12c384h6tfrs2)

- Inputs are raw (source `cpp/neuralnet/nninputs.cpp`, branch Ataxx-onnx): on-board, own, opp, banned, a two-stage
  move flag, rule flags, komi. No hand features: everything in section 0 items 1-2 is inferred.
- Raw net speed here: ~194 ms/position (4 threads). Our input encoding reproduces outputs on raw self-play inputs
  (max win-prob diff 0.016; value head order win/loss/no-result confirmed).

## 4. Consequences for our NNUE

- Per-square inputs within distance 2: stone exposure + empty-neighbour count (safety), clone/jump reach for both sides,
  capture counts for both sides, and the 2-ply local exchange value; plus tempo and phase buckets (mobility/stuck late).
  This is what both strong nets compute first; giving it as input lets a small fast net skip those layers.
- A hidden layer is needed: safety, exposure and attack interact (a +4 capture next to my exposed stones is not +4).
- Data: Titanium 5k labels do not contain the safety signal; the KataGo teacher labels do.

## 5. Caveats

- Probes: 400 positions (~280 train / 120 test), ridge on 384-d states; numbers are directional, not exact.
- Concept probe judges value models by linear R2 on hand-built concepts; it shows what values depend on, not causation.
- Kata's labels come from mostly Botzone end rules; Titanium/Moonbird use classic rules (endgame mismatch).

## 6. Discarded

- Kata ownership head: ~0.02 for own and enemy stones alike (untrained in the Ataxx build or misread). Not used.

# Part 2 (2026-10-24, afternoon): policies, jumps, autaxx, current Stockfish — and the anchor decision

Part 1 above analysed VALUES only and picked the anchor without testing it. This part closes that gap. Scripts:
`training/probe_kata_policy.py`, `probe_kata_jumps.py`, `probe_cpp_policy.py` (all read-only).

## 7. KataGo's 200-visit SEARCH policy (raw self-play visit counts; 45k stage-0 + 553k stage-1 decisions)
- KataGo splits a move: stage 0 = click an empty square (= clone) or an own stone (= pick it for a jump), stage 1 =
  jump destination (cpp/game/gamelogic.cpp, branch Ataxx-onnx).
- **Jumps dominate strong play:** 60% of search mass is on jumps; the top move is a jump in 59% of positions.
- **The 2-ply exchange decides, not the raw capture count.** Clone squares, search mass vs uniform:
  2-ply net -3 0.11 | -2 0.18 | -1 0.46 | 0 0.92 | +1 1.08 | +2 1.03, while by raw conversions it only climbs to 0.87.
  Jump destinations: net 0/+1 1.93/1.96x uniform, net <= -3 ~0.3x; top jump destination = best 2-ply destination
  70.4% vs max-capture destination 67.4% vs random 37.3%.
- **Kata trades, it does not play safe:** the opponent's best capture rises after its chosen jump (3.29 -> 4.08).
  Jumps that leave an exposed hole next to own stones are disfavoured mildly (0 exposed 1.31x, 1 exposed 0.69x).
- **Anchor test** (how often the top move lands on the candidate anchor; mass on it):
  B my best 2-ply clone square 19.8% / 0.186 > A my max-capture square (current G2) 12.3% / 0.114 >
  C enemy's best capture square (block) 9.1% / 0.087 > D random clone square 6.1% / 0.060.
  -> The current anchor (A) is not the best definition; B wins clearly. Hit rates are capped because most top moves
  are jumps (stage 0 cannot show their destination).

## 8. Ataxx_CPP policy head (one forward pass, 3000 held-out positions)
- Plays a clone 50.7%; its move is a best-2-ply move 52.4% (random 7.5%), a max-capture move 45.9% (random 8.1%).
- In the other ~48% it gives up 2-ply value on purpose (chosen -0.30 vs best 0.45): positional knowledge beyond the
  local exchange (what search + a good eval must supply).

## 9. autaxx (alpha-beta + NNUE; beat our tuple net 2:1 at equal nodes on this CPU)
- Inputs = raw board only, 98 (own/opp per square, perspective flipped), **32 neurons per perspective** with separate
  weights per side, ReLU -> 64 -> 32 -> 32 ReLU, output from concat(layer1, layer2) (skip), eval = 600 * y, float,
  38 KB. Move ordering: TT, killer, captures by 100*captures + 100 if clone, then quiets.
- Lesson: a nonlinear hidden layer on the plain board already beats our 726 KB linear tuple net per node. Input
  richness is not what made the difference there.

## 10. Stockfish NNUE as of 2026 (official-stockfish/nnue-pytorch + Stockfish src/nnue/features, read from source)
- Default inputs `Full_Threats + PP_3Wide + K32Q2^`:
  - Full_Threats (59,808 inputs, <=256 active): one input per edge of the ATTACK GRAPH (attacker piece+square ->
    attacked piece+square), oriented by perspective and king side.
  - PP_3Wide: pawn pairs inside a 3-file band (local pair structure, like Moonbird's pattern pairs).
  - K32Q2: piece-square relative to the own king bucketed into 32 coarse mirrored regions x opponent-queen flag
    (the anchor is COARSE, not the exact square).
- Trunk: L1 = 1024 feature transformer, pairwise product of accumulator halves, layer stacks (buckets) 32 -> 32,
  output fed by both hidden layers (skip). Loss: symmetric sigmoid of scaled score, |p - q| ** pow_exp, lambda
  schedule start -> end with jitter, sample weights emphasising decisive positions. Optimizers: ranger_lite / AdamW /
  schedule-free.
- (Fairy-Stockfish variant trainer: per-variant compile-time config; its Ataxx build was NOT inspected.)

## 11. Decisions for NNUE v2 (each one is an ablation arm; games decide)
1. **Anchor = landing square of the best 2-ply move over ALL moves (clones and jumps)**, not the max-capture square.
   Arms: exact square vs coarse symmetry class (Stockfish K32 style); 1 anchor vs top-2 anchors.
2. **Threat edges (Stockfish Full_Threats for Ataxx):** one input per (side, clone|jump-only reach, landing square,
   direction of an adjacent enemy stone) = 2*2*49*8 = 1,568 inputs: WHICH stones are hit from WHERE, not counts.
3. **2-ply exchange for jumps too** (G5 today covers clones only): per empty square the best jump net over origins,
   with the vacated origin included.
4. Keep G1 square state (reach/capture/exposure: what Kata computes in layers 0-1 and Ataxx_CPP hand-feeds).
5. Trunk: add skip connections from both hidden layers to the output (Stockfish + autaxx both do); width 128-256
   (Optuna: width importance 0.0 at 2.5M rows); pairwise product = ablation arm.
6. Loss/labels: lambda schedule start -> end instead of a fixed 0.75; keep Kata TD as the teacher stage.
7. Search lever (outside the net): order captures by 2-ply net instead of raw capture count.
