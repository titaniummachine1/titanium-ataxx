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

# Part 3 (2026-10-24, evening): architecture research close-out, KataAtaxx first

User order: finish the research on the top engines to find the best NNUE architecture; "especially KataAtaxx".
Read-only. Scripts: scratchpad `kata_npz_layout.py`, `kata_npz_targets.py`, `kata_npz_sample.py` (read the Kata dataset npz).

## 12. KataAtaxx: what its TRAINING DATA actually contains (8 of 834 raw files sampled, plus one 989-row file)
- `valueTargetsNCHW` (5 x 7x7): channels 0, 1, 4 are ALL ZERO. Channels 2 and 3 are populated (65% / 74% of squares
  nonzero, values -1..1) = KataGo's FUTURE-POSITION targets (which stone sits on each square some turns ahead).
  => the net's only spatial auxiliary task is future stone ownership of each square. There is NO final-ownership target.
  This explains why the ownership head probe read ~0.02 (section 6): that head was never trained in the Ataxx build.
- `scoreDistrN` (218 bins) is all zero: NO final stone-margin target. KataAtaxx learns win/loss/draw only (5 time
  horizons: final result + TD at 4 horizons, globalTargetsNC cols 0/4/8/12/16 ...) plus the 2-stage policy.
- `policyTargetsNCMove` stage 1 (jump destination) present for 100% of rows; mean root visits 197.
- Consequence for the NNUE: Kata's strength comes from win/loss value + policy + a spatial future-stone aux task, not
  from a score head. An NNUE aux head "future stone per square" is the Kata analogue (training-only, free at inference);
  the data is in the raw tdata (channels 2/3) but NOT in our sidecars (`kata_labels_*_nopol.npz` keep TD/visits/draw only).
  Extracting it is a new dataset job (not started).
- Ataxx_CPP does use ownership+score aux (ownership MSE 0.5, score MSE 0.5) trained from its own selfplay: a second
  independent engine with a spatial aux head.

## 13. KataAtaxx net recap (facts only)
- b12c384h6tfrs2: 12 transformer blocks, 384 ch, 6 heads, RoPE, RMSNorm, SwiGLU; ONNX 86 MB; trained with Muon.
  Raw 22 binary + 19 global inputs (no hand features). Release selfplay config: 200 visits, 70% PASSSCORING.
- Cost here 0.45 s/visit on CPU (ignores maxTime). Not usable as an eval; it is the TEACHER only.
- Its first two layers compute stone exposure and jump reach (section 0): this is the part an NNUE can take as input.
  What layers 3-11 add (long-range tempo, endgame counting) is not understood. NOT analysed: deeper-layer concepts
  (region/parity of empties, stuck/zugzwang, chain threats), whether the policy trunk shares them with the value head.

## 14. Other engines closed out this evening
- Sanctaphraxx 0.1.11 (read from source): 147 inputs (own/opp/gap per square), ONE feature transformer of 64 per
  perspective, CReLU(255), then a single dot product (no hidden layers), 19 KB, datagen at 5000 nodes with win/draw
  adjudication. It loses 104-0 to us at 5k, but its search is a ~300-line PVS+TT: net and search are confounded, so this
  cannot be read as "no hidden layer is bad".
- autaxx: 98 -> 32 -> 64 -> 32 -> 32 with skip (section 9); +54 Elo over our tuple net at 5k nodes, about level at 100 ms.
- Moonbird src (newer checkout) eval = same families as 1.1.0 (2x2-block pairs in 4 offsets 2/14/16/12 => 2x4, 4x2 and
  two diagonal windows, 4^8 states per cell because blocked squares are a 4th state; env[sq][own][enemy][blockers];
  mobility[phase][reach]; turn bonus; int16 params, SGD). Not a different architecture, only blocked-square support.
- Moonbird search (read this session): PVS + TT, null move (needs >= 11 clone targets), LMR from move 7, futility
  reduction by eval, IIR, aspiration (start +-8, x1.9), history ordering by pieceDelta + history, clone preferred over
  double (jump bonus -2.2). Nothing exotic; the 1-99 result at equal 5k nodes (S39, pre-S55 nets) may still be mostly
  search: the S39 "same-eval graft test" is the only evidence and is 3 weeks old, NOT re-run with S57.
- Fairy-Stockfish: the ADVERSARY matrix row "Ataxx net 245 KB, HalfKAv2" is NOT verifiable here: the local checkout has
  no [ataxx] variant, only quadwrangle/crusade variants with the ataxx enclosing rule. Treated as unverified.
- atx (Botzone #1, 2670): closed. Descriptions only (spatial + relational NNUE, VNNI, persistent tree). Nothing to read.
- tiktaxx eval, AtaxxZero (ResNet 12x128, MCTS), loltaxx: not studied further, not competitive.
- Web search found no public Ataxx NNUE rating list or paper; the only open Ataxx NNUE sources are autaxx and Sanctaphraxx.

## 15. Architecture recommendation (a proposal built from the evidence above; every piece is an ablation arm, games decide)
1. Inputs, sparse and mostly locally updatable: (a) own/enemy/blocked per square (147; autaxx, Sanctaphraxx);
   (b) Moonbird-style env per stone (square x own neighbours x enemy neighbours) = stone safety, the strongest single
   signal (+.135 R2 in Moonbird, +.07 in Kata); (c) reach/threat edges (D61 item 2); (d) 2-ply exchange landing features
   (D61 items 1 and 3).
2. NEW vs D61: a linear tuple-table term added at the output (Stockfish PSQT idea; Moonbird's pair tables are worth
   .012-.016 MSE and cannot be NNUE inputs: 6561 states per cell). Our existing tuple net could be that term (residual).
3. Trunk: feature transformer 128 (width importance 0.0 in Optuna), then 32 -> 32, skip from all layers to output.
4. NEW vs D61: OUTPUT BUCKETS by filled-square count (Stockfish layer stacks; Moonbird mobility[phase]; Ataxx_CPP's four
   empties-band nets 0-8/8-15/15-30/30-50). Supported by three engines, absent from D61.
5. Training: base stage Titanium 5k labels, teacher stage Kata TD h1 (v3), lambda schedule, Moonbird-like 0.5/0.5 result mix
   as comparison. NEW: Kata futurepos aux head (training-only), 8-fold board symmetry augmentation (Kata and Ataxx_CPP
   both train with it; whether our trainer does is NOT verified).
6. Risks to state plainly: (i) anchor/2-ply features are global per node (no accumulator update) and cost about a search
   node each: 5k-node gates ignore that, 100 ms gates will not; (ii) eval gains may be capped by search: S39 says the
   Moonbird gap is mostly search; (iii) NNUE v1 already has better base-stage MSE than S57 (.01717 vs .0217) but worse
   Kata-TD MSE (.0687 vs .0663) and has never played, so no architecture claim has a game result yet.

## 16. NOT analysed (do not call this done)
- Deeper Kata layers (3-11) concepts; what the Kata policy trunk knows that its value head does not.
- Whether the S39 search-vs-eval conclusion still holds with S57 (re-run the Moonbird-eval-in-Titanium-search graft).
- Moonbird's data-gen details beyond the source (50M samples, 6k nodes) and whether its advantage is data volume.
- Any NNUE trained on Kata TD at all: v1 was trained on the base-stage labels; the teacher stage has not run.
- Speed of the D61 features inside the search (no prototype).

## 17. KataAtaxx POLICY (measured 2026-10-24, `training/probe_kata_deep.py policy`, logs/probe_kata_deep_policy.txt)
User view: the policy holds most of Kata's/Ataxx_CPP's strength (Ataxx_CPP policy-only == Titanium S57 at ~30k nodes; their evals are
not better than Titanium's). Kata raw net (no search), 500 stage-0 + 500 stage-1 positions from its own self-play, exact stored inputs:
- Channel check: out_policy[0] is the main policy (top-1 agreement with the 200-visit search 0.648 stage 0 / 0.910 stage 1);
  channel 2 agrees almost as well (0.644 / 0.908), channels 1 and 3 do not (0.10-0.19: opponent/aux policies).
- Stage 0 (choose a clone square or a jump origin; 22.2 legal choices): the raw policy puts 50% of its mass on 1.6 choices, 80% on 3.0,
  90% on 4.1, 99% on 7.7. The SEARCH top move is raw rank 1 in 64.8%, top 2 84.6%, top 3 93.6%, top 5 98.4% (mean rank 1.69).
  Search raises the top move's probability from 0.485 (raw) to 0.711.
- Stage 1 (jump destination; 5.7 legal): 90% of mass on 1.7 choices; search top move is raw rank 1 in 91.0%, top 2 in 99.0%.
- Cheap scorer: "max stones converted among clone squares" picks Kata's search top square in only 16.4% (raw Kata policy 64.8%).
- What the raw policy logit is made of (R2 of the centred logit, cheap board features): stones converted .25 alone, 2-ply net .26 alone,
  all together .34 (clone squares) / .32 (jump-origin picks) / .32 (jump destinations). Two thirds of the policy is NOT explained by
  conversions, 2-ply exchange, empty/own neighbours or centre/edge.
- Logit lens (final logit from the residual stream after block 0..11, stage 0 clones): .39 .50 .60 .65 .69 .72 .74 .78 .83 .85 .90 .97;
  jump destinations .37 .52 .57 .62 .67 .69 .69 .71 .72 .76 .83 .92. The move preference forms gradually over all 12 blocks.
- Reading: a policy that puts the search's best move in its top 3 of ~22 (94%) is a very strong move ordering. Titanium's capture-count
  ordering finds it far less often. Whether a small feature-based scorer can approach this is untested; the unexplained two thirds
  is the "positional knowledge" a big net supplies.
- Ataxx_CPP's policy was measured earlier (section 8), not re-run here.

## 18. What KataAtaxx computes INSIDE (logs/probe_kata_deep_rest.txt; 800 of its own self-play positions, exact inputs)
Per-square linear probes on the residual stream after each block (test R2; "patch" = linear on the raw 5x5 neighbourhood):
| concept | patch | L0 | L1 | L3 | L7 | L11 |
|---|---|---|---|---|---|---|
| my stone exposed (enemy can land next to it) | .50 | .86 | .87 | .83 | .80 | .74 |
| enemy stone exposed | .48 | .85 | .87 | .84 | .79 | .73 |
| empty square in my reach (clone or jump) | .31 | .40 | **.83** | .77 | .75 | .62 |
| empty square in enemy reach | .26 | .24 | **.77** | .73 | .72 | .45 |
| empty square reachable only by my jump | .39 | .58 | **.87** | .80 | .70 | .56 |
| clone 2-ply net (gain - enemy best recapture) | .56 | .53 | .69 | .65 | .62 | .47 |
| future stone (Kata's own aux target, ch2) | .22 | .30 | .31 | .33 | .34 | .39 |
- Order of inference: block 0 computes EXPOSURE (local counts); block 1 computes REACH (distance-2 jump geometry needs a second
  attention hop). The 2-ply exchange is never represented better than .69: Kata does not compute "my capture minus the recapture"
  per square; it computes exposure and reach and lets later blocks combine them.
- Future stones: barely above the raw patch (.39 at the end); long horizon (ch3) ~.10. The trunk does not "see the future board".
Position-level probes (mean-pooled tokens, test R2):
| concept | raw counts | L0 | L1 | L3 | L7 | L9 | L11 |
|---|---|---|---|---|---|---|---|
| safe-stone difference (stones with no empty neighbour) | .02 | **.81** | .84 | .82 | .69 | .64 | .43 |
| my exposed stones | .36 | .90 | .88 | .83 | .82 | .81 | .66 |
| number of empty regions | .26 | .73 | .77 | .70 | .63 | .64 | .50 |
| odd-sized empty regions (parity) | .08 | .28 | .34 | .23 | .21 | .18 | .08 |
| my best capture | .25 | .61 | .67 | .60 | .62 | .57 | .39 |
| **Kata's own TD value** | .14 | .39 | .48 | **.67** | .72 | .80 | .82 |
- The value is assembled gradually: +.19 at block 3, then steady to .82 at block 11. Safety and exposure are available from block 0;
  what blocks 3-11 add to the value is NOT a single cheap concept from our list (parity fades away; regions fade).
- Parity of empty regions is weakly represented and dropped: not something Kata uses.
Neurons (SwiGLU, 1024 per block):
- Early blocks have single units for the basic concepts: block 0 unit for "stones I convert by landing here" (|r| .96, R2 .92 from
  ONE unit), block 1 unit 900 = "my stone is exposed" (|r| .86, R2 .72 from one unit). Later blocks spread concepts over many units.
- Block 10 unit 559 (max activation 20.5, far above any other) fires on MY stone with no empty neighbour sitting inside enemy
  stones (contexts: m surrounded by E). Candidate concept "trapped/locked stone"; NOT verified beyond its top contexts.
Attention heads (72 = 12 x 6):
- Block 0 heads are almost uniform (ratio ~1): block 0 is mostly the per-square MLP.
- From block 1 the heads specialise into three kinds: 3x3-ring heads (d0/d1 up to 9.8x uniform), distance-2 jump-ring heads
  (L1h3 3.8x, L2h1/h2 3.2x, L3h4 4.1x, L6h4 4.4x, L8h3 4.6x) and a few "look far" heads (L1h0, L2h5, L4h3, L4h4: zero mass at d0-d1).
- Key-kind heads: L1h5, L5h0, L7h1, L10h2 look at EMPTY neighbours (60-76% of mass vs 44% base) = liberties; L4h1, L7h4, L10h3 look at
  ENEMY neighbours (44-55% vs 31%). No directional preference (mean offsets ~0): the geometry is symmetric.
- Everything Kata attends to is inside distance 2, plus a few global pooling heads. Confirms section 0.
Ownership head: output mean .013 std .026 = untrained (consistent with section 12: no ownership target in its data).
Occlusion by DELETING a stone (100 positions): confounded (deleting creates a hole the mover can use), see section 20 for the
colour-flip version. Raw: deleting an enemy fully-enclosed stone = +.57 for the mover (a hole inside the enemy mass), own enclosed
stone only -.12.

## 19. Why Kata picks its move: choice model (training/kata_choice_model.py, logs/kata_choice_model.txt)
3000 decisive Kata decisions (search top move > 50% of visits; 1200 clones, 1800 jumps), 54.9 legal moves on average.
Top-1 agreement on held-out decisions: random .021 | max conversions .173 | max material after the enemy's greedy reply (2-ply) .248 |
**12-feature conditional logit .299** (train .286, so not overfit). For scale only: the raw Kata net matches its search on the
FIRST STEP (square / origin choice, ~22 options) in .648 (section 17); that is an easier task than the full move (~55 options), so
the two numbers are not directly comparable.
Exchange rates (logit weight per unit, divided by the weight of one converted stone):
| feature | value in converted stones |
|---|---|
| one converted stone | 1.00 |
| making the move a jump (vacating the origin) | -2.21 |
| ... but the origin stone was attackable anyway | +1.16 (so jumping a doomed stone costs ~1 stone) |
| one more SAFE stone than the enemy after the move (no empty neighbour) | **+1.06** |
| material after the enemy's greedy reply, per stone | +0.50 |
| one more of my stones exposed after the move | -0.36 |
| landing square has a filled neighbour (filling a hole) | -0.29 per neighbour |
| clone reach difference | +0.15 |
| enemy's best recapture | -0.15 |
| territory (empties only I reach) | +0.06 |
- Kata pays real material for SAFETY: one safe stone is worth about one converted stone; exposure costs a third of a stone each.
  Safety weight is highest with 10-18 empties (+1.70) and lowest with 28-36 (+0.95).
- 92% of Kata's jump origins are attackable stones (72% of all own jump-capable stones): it jumps with stones it would lose anyway.
  Jump destinations convert 3.18 on average and allow a 4.04 recapture: Kata trades, the 2-ply material count says it loses.
- Against the 2-ply material move (when they differ, 81% of cases), after the enemy's greedy reply Kata's move has 2.47 FEWER stones
  in material but 1.29 fewer exposed stones and +0.34 more safe stones than the enemy (training/kata_why_moves.py).
- **v2 (4000 decisions, 20 features incl. 8 permanence features after the flip test; logs/kata_choice_model_v2.txt):**
  top-1 .309 (v1 12 features .310), top-3 .604, top-5 .745; tree ranker on the same features top-1 .315 / top-3 .594.
  2-ply material with conversions as tie-break: top-1 **.300**, top-3 .529. So for the FIRST choice, cheap features add almost
  nothing over 2-ply material; safety/permanence lift top-3 (.53 -> .60). Weights are collinear across the safety/permanence
  features (v1 safe_a 1.06 splits into safe_a .67 + perm terms .2-.3 each in v2): read the GROUP effect, not single exchange rates.
  The cheap-feature ceiling for Kata's full move is ~31% top-1 / ~60% top-3 / ~75% top-5 of ~55 legal moves.
- The cheap features still leave most of the policy unexplained (30% vs Kata's 65%): what is missing is NOT more local tactics
  (2-ply, reach and territory add almost nothing in drop-one). Inference, not proven: the rest is whatever blocks 3-11 compute.

## 20. What makes a stone valuable to Kata's VALUE head: colour-flip test (training/probe_kata_flip.py, logs/probe_kata_flip.txt)
150 of Kata's own positions, every stone flipped to the other colour (occupancy unchanged, material moves by 2), 4,254 stones.
Number = drop of the owner's P(win)-P(loss) when its stone changes colour (bigger = Kata values that stone more).
| stone | value |
|---|---|
| fully enclosed (0 empty neighbours) | **0.81** mover / 0.77 enemy |
| 1-3 empty neighbours | 0.32-0.46 |
| 4-7 empty neighbours | 0.22-0.38 |
| NOT attackable (other side cannot land next to it) | **0.79** / 0.75 |
| attackable | 0.32 / 0.37 |
| corner / edge / interior | 0.78 / 0.66 / 0.39 |
| own-colour neighbours 0-1 / 2-3 / 4-5 / 6-8 | 0.52 / 0.51 / 0.53 / 0.59 (support hardly matters) |
| empties 8-15 / 16-24 / 25-38 | 0.63 / 0.47 / 0.37 |
- **STONE PERMANENCE is what Kata's value counts.** A stone that cannot be converted next move is worth 2.5-3x a stone that can.
  Support by own stones barely matters; walls (edge/corner) matter because they remove landing squares.
- This is exactly Moonbird's env table (own / enemy neighbours per square => empty neighbours and walls), its strongest family,
  and exactly the safe/exposed concept Titanium's 5k labels ignore (+.000 in section 0).
- Owner, empty neighbours, attackable, support, edge, corner and phase explain R2 .34 of the per-stone value linearly: the
  rest depends on the wider position (to be checked: whether the enemy's landing square converts it with many others).
- Caveat: one flip moves material by 2 in positions that are often close, so values saturate; read them as RELATIVE.

## 21. VALUE feature screen: how far cheap features + knobs get toward Kata's value (training/screen_features.py, logs/screen_features.txt)
100k stratified calibration rows (no blocked squares), 36 features in 6 groups, 75/25 split, MSE vs Kata win prob (lower = better).
| model | 9-15 empties | 16-24 | 25-35 | 36-49 |
|---|---|---|---|---|
| constant | .0756 | .0585 | .0586 | .0513 |
| material only | .0717 | .0566 | .0501 | .0343 |
| Titanium 5k SEARCH value (scale refit) | .0648 | .0500 | .0431 | .0303 |
| Moonbird static eval (scale refit) | .0577 | .0411 | .0349 | .0308 |
| hand features, linear | .0645 | .0469 | .0356 | .0293 |
| **hand + Moonbird families, linear** | **.0556** | **.0396** | **.0314** | **.0287** |
| hand features, per-value knob tables (pure lookup HCE) | .0660 | .0473 | .0354 | .0295 |
| hand + Moonbird, knob tables | .0586 | .0423 | .0325 | .0294 |
| hand + Moonbird, gradient-boosted trees | .0558 | .0384 | .0310 | .0277 |
| raw 98 squares, trees | .0693 | .0486 | .0359 | .0317 |
| raw squares + all features, trees | .0558 | .0384 | .0307 | .0275 |
| Ataxx_CPP raw value (a deep net distilled from Kata) | **.0440** | **.0295** | **.0260** | **.0244** |
Findings:
- A linear HCE (hand features + Moonbird's tables) closes 57-78% of the gap between material and Ataxx_CPP's deep net.
  Trees and per-value knob tables add almost nothing on top: the effects are near-LINEAR in these counts, and the remaining gap
  is information these features do not contain (trees on raw squares cannot find it either; a deep net can).
- Greedy order of what reduces error after material: **Moonbird families first** (3 of 4 bands), then **safety**; reach, attack
  (incl. 2-ply) and structure (regions, parity) add <=.0005. Only in the opening (36-49) attack/safety come first.
- Drop-one from the full tree model: Moonbird families are the only group with a large unique contribution (+.008/.006/.003 in
  the 9-35 bands); safety second (+.0005-.0018). Moonbird's 2x4 pattern pairs carry something our hand concepts do not.
- **Titanium's 5k SEARCH values are further from Kata than Moonbird's STATIC eval** in every band except the opening.
- Single-feature correlation with Kata's value: Moonbird env .41, material .36, fully-enclosed-stone difference .28, safe .28,
  edge .23, Moonbird pairs .22, moves .18.
- Caveat: closeness to Kata's value is not strength. Ataxx_CPP's value is the closest here and lost 4-108 when grafted into our
  alpha-beta search (AXNET_PROBE). It measures what the eval should KNOW, not how it should be scaled for search.

## 22. Root-move ordering by a policy net: would it help Titanium? (training/root_order_probe.py, logs/root_order_probe.txt)
300 val positions (10-36 empties). Truth = Titanium main S57 best move at 1,000,000 nodes; uainewgame before every search.
| picker | agrees with Titanium's 1M-node move |
|---|---|
| Titanium @ 1k nodes | .373 |
| Titanium @ 5k nodes | .467 |
| Titanium @ 50k nodes | .560 |
| **Ataxx_CPP policy, ONE forward pass (~0.7 ms)** | **.443** |
- The policy's single pass beats Titanium's own 1k-node search and nearly matches 5k nodes. In **17.3%** of positions Titanium @5k
  misses the deep move while the policy has it (20.0% at 1k): the two are complementary, not redundant.
- By phase (10-18 / 19-27 / 28-36 empties): policy .46/.38/.49 vs Ti@5k .52/.37/.52 vs Ti@50k .54/.59/.56.
- Truth is Titanium's own deep choice, which biases the test toward Titanium's shallow searches; the policy's real accuracy (vs a
  stronger judge) is likely understated.
- Reading for root ordering: useful exactly where the root search is shallow (5k-node gates, fast time controls, first iterations);
  at 50k+ nodes the search already beats the policy, so expect small gains there. Costs one 0.7 ms pass per move, once.

## 23. What does S57's STATIC eval miss vs Kata? (training/s57_residual.py, logs/s57_residual.txt)
Same 100k rows as section 21; S57 static = the trainer's exact forward (tuple TUP7 + hand base eval, final_v2_wdl_ep8.tup).
| MSE vs Kata | 9-15 empties | 16-24 | 25-35 | 36-49 |
|---|---|---|---|---|
| **S57 static eval** (scale refit) | **.0572** | **.0397** | **.0314** | **.0283** |
| Moonbird static eval | .0577 | .0411 | .0349 | .0308 |
| Titanium 5k SEARCH value | .0648 | .0500 | .0431 | .0303 |
| Ataxx_CPP raw value (deep net) | .0440 | .0295 | .0260 | .0244 |
| gap S57 -> deep net | .0132 | .0102 | .0054 | .0039 |
| removed by S57 + ALL cheap groups (linear) | .0026 | .0020 | .0021 | .0013 |
| removed by S57 + safety only | .0008 | .0006 | .0013 | .0006 |
| removed by S57 + Moonbird families only | .0016 | .0018 | .0012 | .0008 |
- **S57's static eval is already the best cheap eval we have**: at least as close to Kata as Moonbird's, and clearly closer than
  Titanium's own 5k SEARCH values (the search moves it AWAY from Kata; the labels it learned from carry that bias).
- **S57 already contains almost everything the cheap features know.** All six groups together close only 13-33% of the gap to
  the deep net (trees on top: same). Permanence alone: 5-24% of the gap. An HCE term on S57 would be a small gain at best.
- **The missing knowledge is deep, and it is concentrated in the endgame** (gap .013 at 9-15 empties vs .004 in the opening).
  Caveat: Kata's data is ~70% PASSSCORING (Botzone-like end rule) vs classic rules in Titanium; part of the endgame gap can be
  the RULE difference, not knowledge. Not separated yet.

## 24. Consequences (agent proposal, D63; games decide)
1. The NNUE's job is the DEEP part: S57 already holds the cheap pattern/safety knowledge. Feed S57's tuple output into the NNUE
   output as a skip term (Stockfish PSQT idea) so the hidden layers learn only the residual, and train it on the Kata TEACHER
   stage (TD values), which no NNUE has been trained on yet.
2. Revise D61: the 2-ply anchor, jump 2-ply and threat-edge inputs were chosen from POLICY hit rates; for VALUE they add ~0 on
   top of S57 (reach/attack groups remove <=.0005). Build them only as later ablation arms, not in the first v2.
3. Keep cheap permanence inputs (safe / attackable per stone): they are what Kata computes in block 0, cost a few bit
   operations, and help the first layer; expected gain on top of the S57 skip is small.
4. Phase/output buckets with the most capacity where the gap is largest (<= 24 empties). Check the rule mismatch first.
5. Search lever, independent of the net: Ataxx_CPP policy for ROOT move ordering (section 22), one pass per move.

## 25. Rule mismatch vs knowledge (training/rule_gap_probe.py, logs/rule_gap_probe.txt; 40k Kata raw stage-0 rows)
- Kata global input column 2 is on for 29% of rows (candidate PASSCONTINUE flag). Toggling ONLY it changes Kata's value by a FLAT
  .07-.08 in every phase (0-8 empties .082 ... 36-49 .075): it is not an endgame-specific end-rule effect (meaning of the flag
  unverified; KataGo's nninputs.cpp for the Ataxx branch not available locally).
- S57 static vs Kata TD, flag off / on: 9-15 .0545/.0515, 16-24 .0359/.0321, 25-35 .0239/.0224, 36-49 .0168/.0217. S57 is 5-10%
  closer on flag rows in the middle phases, evenly: the rule (if that is what it is) explains a small part of the gap.
- **0-8 empties: S57 static is nearly useless vs Kata (.121-.138 vs constant .144-.162).** Alpha-beta reaches the end of the game
  there, so this gap costs little in play. The gap that matters for the NNUE is 16-35 empties, where the search cannot see the end.

## 26. "Ahead -> clone (shorten the game), behind -> jump" (user hypothesis; training/kata_clone_vs_lead.py, logs/kata_clone_vs_lead.txt)
1,979,555 Kata stage-0 decisions; share of 200-visit search mass on clones.
- By material lead (13-36 empties): -10 or worse .418 | -9..-5 .419 | -4..-2 .421 | -1..+1 .443 | +2..+4 .446 | +5..+9 .469 |
  +10 or better .490. Monotone: further ahead => more cloning.
- By Kata's win prob (5-12 empties): <.10 .372 | .45-.55 .355 | .70-.90 .384 | >.90 .471. Clearly winning endgames clone most.
- NOT confirmed: "behind => cloning is bad". Moderately behind looks like even (~.40); hopeless positions clone MORE (<.10 win:
  .44-.73 in the 13-45 bands). U-shape: decided positions clone, close positions jump.
- Size: +.0075 clone share per +.1 win prob with empties held fixed. Real but minor; a phase-bucketed net can express it through
  material lead x empties. Possible search use (untested): prefer clones in move ordering when far ahead.

## 27. ENGINE BUG: main evaluates TUP7 tuple nets differently from how they were trained (found 2026-10-24)
Found while building the v2 S57 skip term (training/tuple_parity.py: single-family .tup copies, engine `nneval` vs the
trainer's own forward train_stream.features_for + materialize32 + base_eval_stm). Hand eval: 0 cp. Tuple, S57 on 400 val
positions: mean |engine - trainer| 68 cp, max 565 cp; engine S57 vs trainer S57 over all val rows: 66 cp mean, corr .963.
Causes (crates/titanium/src/tuple.rs on main 298b7ae; TUP7 shipped since S53 1903dca, so S53..S57 all affected):
1. **Loader**: capn/conv were read only for TUP6. TUP7 = TUP6 + per-anchor pairs and CARRIES capn/conv before `pa`
   (train_full/train_stream dump order ... capn, conv, pa; the Python loader reads them for TUP6 and TUP7). So for every
   TUP7 net the engine (a) never used capn/conv (trainer: 32 + 53 cp mean magnitude) and (b) read the 524,880-byte `pa`
   table 1,060 bytes too early = every per-anchor pair weight wrong (corr -.008 with the trained term). No trailing-bytes check.
2. **Zero constraints**: train_stream.zero_constrained zeroes only sp[0] (block 0, state 0) and q2 block 0; the engine skips
   state 0 of EVERY 2x2 block. S57: 32/36 single[a][0] and 27/36 q2[a][0] weights are nonzero and were dropped (16.6 / 2.7 cp).
3. **Env definition**: the trainer's env_tables tallies the ring of EVERY square (env_us[sq][own][foe][blk] +
   env_them[sq][foe][own][blk]); the engine tallied only own stones (env_us) and enemy stones (env_them) (20.1 / 12.9 cp).
Fix (branch exp/tup7-fix 653d9a1, also in exp/nnue): read capn/conv for TUP7, reject trailing bytes, sum state 0 of every
block, env on every square. After the fix every family and the full S57 = **0 cp** (tuple_parity.py). Cargo tests pass.
Effect: bench d8 from the start position a1b2 score 90 / 10,915 nodes (main) -> a1a2 score 13 / 10,844 nodes (fixed);
the fixed eval is ~2.4x slower per node (316k vs 753k nps, measured under training load): it now computes the capconv and
49-square env terms main skipped. Gate: 5k nodes first (direction); time controls need a faster implementation.
Gates (vs 298b7ae, same S57): 5k +77 (400g), 100 ms +45 (400g), 50 ms +26 (800g), all PASS; vs Moonbird at node odds +53.
MERGED + PUSHED: main 653d9a1 (2026-10-24, user order).
Consequences: every TUP7 gate (S53-S57) compared nets through this mismatched forward; the S-series gains are real (they won
games) but the engine never played the function the trainer optimised. The NNUE v2 S57 skip must come from the ENGINE
(`titanium-cli s57dump`), never from the Python forward of an unfixed engine.

## 28. Base-label quality: relabelling with the current engine (training/label_quality_probe.py, logs/label_quality_probe*.txt)
3,000 Kata-labelled calibration rows (1,241 test rows after a 60/40 split), 5k-node `label`, MSE vs Kata TD after a scale refit.
| labeller | all | 9-15 | 16-24 | 25-35 | 36-49 |
|---|---|---|---|---|---|
| stored titanium_cp (main ea0bdf5 + S55tree, buggy TUP7 eval) | .0652 | .0757 | .0596 | .0420 | .0341 |
| main 298b7ae + S57 (buggy eval) | .0537 | .0670 | .0461 | .0324 | .0321 |
| main 653d9a1 + S57 (fixed eval) | **.0524** | **.0621** | .0471 | **.0294** | .0322 |
- The base data was labelled by an engine two nets old. Relabelling with current main makes the base-stage targets ~20% closer
  to Kata (30% at 25-35 empties). Most of it is the better net (S55tree -> S57); the eval fix adds ~2% overall, up to 7-9% in
  the 9-15 and 25-35 bands (noisy: ~240 test rows per band).
- This is the Stockfish loop (relabel with the stronger engine, retrain). Cost estimate: the fixed eval runs ~400k nps
  single-threaded, so 32.75M rows x 5k nodes ~ 110 core-hours (~14 h on 8 workers). Not started (one job at a time).
- Old buggy main kept for comparisons in worktree C:/gitProjects/main298-ataxx (detached 298b7ae).

## 29. NNUE v1 after ONE Kata-teacher epoch vs S57 vs Ataxx_CPP (training/nets_residual.py, logs/nets_residual_ep1.txt)
60k held-out val Kata rows, 30% test split, MSE vs Kata TD (nets: scale refit; Ataxx_CPP raw value as is).
| empties | S57 (fixed engine) | NNUE v1 teacher ep1 (EMA) | Ataxx_CPP raw value (deep transformer) |
|---|---|---|---|
| 0-8 | .1327 | .1192 | **.0936** |
| 9-15 | .0615 | .0533 | **.0457** |
| 16-24 | .0406 | .0325 | **.0305** |
| 25-35 | .0323 | **.0264** | .0267 |
| 36-49 | .0267 | **.0222** | .0228 |
- After one epoch the 256-wide NNUE is 18-20% closer to Kata than S57 in the middle game and already level with Ataxx_CPP's
  distilled transformer from 25 empties up. The remaining gap is the ENDGAME (<= 24 empties) -> where v2 capacity/buckets go.
- Residual by concept group on top of each net: the NNUE has absorbed all cheap knowledge (any group removes <= .0002) except a
  little Moonbird-pattern information in the endgame (.0006-.0009 at <= 15 empties). S57 still misses safety (.0009-.0032) and
  patterns (.0011-.0058). S57 + NNUE together are no better than the NNUE alone (linear combination): the S57 skip in v2 is
  not expected to add much; v2's gain must come from capacity (width, pairwise, endgame buckets) and more teacher epochs.
- Not yet: games. Closeness to Kata is not strength (Ataxx_CPP's value lost when grafted into alpha-beta).

## 30. SEARCH competence gap vs Moonbird (2026-10-24, training/search_depth_probe.py)
40 mid-game val positions (14-34 empties), fresh process / uainewgame per position, S57 for Titanium. (Measured while the
NNUE teacher run used 4 threads: absolute nps low for both, the comparison is under the same load.)
| | Titanium main 653d9a1 | Titanium exp/search-moonbird A | Moonbird 1.1.0 |
|---|---|---|---|
| depth @100 ms | 6.50 (33,855 nodes) | 7.08 (7,189 nodes) | **11.4** (6-8k counted nodes) |
| depth @5k nodes | 4.10 | 6.92 | **9.05** |
- Moonbird searches ~5 plies deeper at equal time. Its node count is not comparable: it statically evaluates every searched
  child for futility and does not count those evals as nodes.
- Search differences (both sources read): Moonbird has futility REDUCTION by the child's static eval (1 ply per 35 of its
  units below alpha, moves reduced to <= 0 skipped), LMR depth*47/(47+n) from move 7, null move R = 4 + depth/2 (ours 3),
  aspiration +-8 x1.9 (ours none, reverted earlier), history with malus (ours bonus only), no killers; ours has RFP margins
  257-774 at depth <= 4, universal LMR -2/-3 from move 2, clone/jump ordering terms. Neither has quiescence.
- Variant A (Moonbird LMR + futility reduction, rest unchanged): 3.2x fewer nodes for bench d8 (3,410 vs 10,844) and +2.8
  plies at 5k nodes, but at equal time only +0.6 ply: our static eval costs microseconds (TUP7 fix: ~2.4x slower), Moonbird's
  is a few table lookups, so per-child evals collapse our node rate (~72k vs ~340k nps). => SEARCH SELECTIVITY AND EVAL SPEED
  ARE ONE LEVER for us: a fast (incremental) eval is what makes Moonbird-style pruning pay at real time controls.
- Not yet tested: Moonbird null move R, aspiration with the fixed eval, history malus, retuning RFP margins for the fixed eval.
