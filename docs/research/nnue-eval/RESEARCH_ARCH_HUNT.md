# Architecture hunt: maximum knowledge per inference (user order D1110, 2026-11-17)

Goal: the big-net architecture that gives the most Elo per CPU nanosecond on one weak core (B13). Every candidate is
judged by four measurements, in this order, and nothing is called better before its gate:
1. LOCALITY: rows changed per move (lab arch-locality / group-diff, offline counting, valid on a busy PC).
2. SPEED: ns per eval in an engine prototype, QUIET PC only (D1105: busy-PC timings were wrong).
3. INFORMATION: held-out val at saturation: first a residual on the frozen champion (lab hce-train --on-nnue, D1053
   protocol), then a net with the inputs, grafted from the champion (C14, never random init).
3b. DIRECT STRENGTH PER NODE (user D1113): lab eval-gate = plain alpha-beta, no TT / pruning / policy / SmallNet, both sides the same search, fixed depth (only the eval decides) or --nodes; takes nets AND residual models (parity: lab residual-check). 400 book pairs. Compare at EQUAL NODES (--nodes 1000 / 5000) and fixed depth (user D1116); scale-match nets before any engine gate (D1115).
4. ELO: ti gate at equal time (100 ms + 1000 ms), insight + analyze.

## What is already measured (do not redo)
- Today's eval, 3.34 us per child (quiet PC, D1106): extraction 590 ns (18%) | rows 2 x 845 ns (51%) | psqt + head
  ~1,060 ns (32%). Head = f32, non-pairwise, 512 inputs. Big-net evals = 66% of node time, child futility evals 47%
  (D1091).
- Partial incremental on TODAY's inputs: exact, but -6% in search, ceiling ~4% per eval (D1088, D1106). Reason:
  today's groups change 66 of 104 rows per move and only half the eval is rows. CLOSED for today's inputs.
- What the champion lacks (D1053/D1059/D1061, frozen net + one linear group, 1.53M held-out rows): pair patterns
  -2.65%; all weakness terms -0.81%; PST / regions / gataxx terms -0.45..-0.53%. OPEN control: a same-size table
  without pair meaning (66,420 weights vs 16).
- The champion is not D4-symmetric: mean 43 cp, max 1,038 cp over 140,000 symmetry pairs (d1053 logs).
- Weakness planes for all squares at once: 41 ns (D1062): computing knowledge is cheap, turning it into rows is not.
- Stockfish (D1070, source): incremental base inputs (2-4 changes per move) + threat and pawn-pair inputs, i16 FT,
  pairwise product, int8 sparse L1, 192 layer stacks picked by a handcrafted class.

## Locality of candidate input families (D1111, logs/arch/d1110_arch_locality.txt)
3,000 val_v11_clean positions, 123,799 children, one accumulator, absolute colours. A move changes 3.20 squares on
average (p50 3, p90 5, p99 8, max 10): attacks ARE local.

| family | active | diff/move | ratio | p90 | max |
|---|---|---|---|---|---|
| stone-square (2x49) | 27.3 | 4.6 | 0.169 | 8 | 18 |
| 2x2 windows, all 36 | 36.0 | 15.6 | 0.434 | 24 | 38 |
| 2x2 windows, non-empty | 30.4 | 14.5 | 0.477 | 24 | 38 |
| 3x3 windows (25), non-empty | 24.3 | 20.1 | 0.828 | 33 | 50 |
| 3x3 around each stone | 27.3 | 17.6 | 0.645 | 30 | 58 |
| stone ring counts (= G3) | 27.3 | 16.9 | 0.619 | 30 | 58 |
| stone pairs, distance 1 | 66.7 | 17.3 | 0.260 | 38 | 102 |
| stone pairs, distance 2 | 90.9 | 28.8 | 0.317 | 67 | 238 |
| stone pairs, distance <= 2 | 157.6 | 46.1 | 0.293 | 105 | 332 |
| stone pairs, distance 3 | 93.6 | 33.1 | 0.354 | 76 | 256 |
| stone pairs, all distances | 408.1 | 126.5 | 0.310 | 284 | 774 |
| threatened stones | 16.8 | 5.7 | 0.341 | 11 | 24 |
| threats (landing sq x colour x clone/jump x gain) | 20.3 | 13.2 | 0.653 | 23 | 48 |
| ENGINE today (G1-G6) | 104.2 | 66.1 | 0.634 | 105 | 158 |

Reading: a Stockfish-style set "stone-square + adjacent stone pairs + threatened stones" = 110.8 active rows (about
today's 104) but only ~27.6 row ops per move incrementally (-58% vs today's diff, -74% vs today's full refresh).
Pairs are incremental-friendly (ratio 0.26-0.35 at every distance) because a pair changes only when one of its two
stones changes. NOT known yet: whether these inputs carry the champion's information (step 3), and the real ns
(step 2: per-row cost was 9-16 ns, sibling-cached rows are cheaper, D1106).

## Candidate architectures (status)
| id | inputs | head | layer stacks | status |
|---|---|---|---|---|
| A0 | today G1-G6 | f32, 512 in | 16 by empties | champion |
| A1 | today + pair rows (zero-init graft) | as A0 | as A0 | info -2.65% (linear, frozen); size control open |
| A2 | stone-square + pairs d1 (+ d2) + threatened stones, incremental stack | int8 pairwise | class: empties x material balance | locality measured; info + speed open |
| A3 | 2x2 windows + pairs + threats (Zataxx-like base) | int8 pairwise | class | locality measured |
| A4 | A2 with narrower W (128/192) paid back by better inputs | int8 pairwise | class | open |
| A5 | Jataxx-like: 25 3x3 windows (D4-tied, ~62k classes) [+ 2x2], W 64-256 SCReLU (D1118) | f32 / int8 | 16 by empties | rows 24 vs 104; trainer lab win-net to build; init = regression onto the champion's first layer |

## Queue (one long job per machine; PC busy with the D1108 gate until it ends)
1. [lab, after the gate] info screens on the frozen champion, D1053 protocol, same rows: size control for pairs;
   --spairs bands 1 / 2 / 3 / 4 (ready, D1072); stone-square; threatened stones; class stacks (empties x material).
2. [lab] information of a STANDALONE input set: train a small net with only A2's inputs (zattrain-style) vs the same
   net with today's inputs, equal width, saturation, held out by game. Decides A2 vs A0 per input row.
3. [engine] A2 prototype on the D1106 accumulator stack (exp/incr-recheck has it): ns/eval on a quiet PC.
4. [hub trainer + Modal] feature-set extension, graft from the champion, train A2 / A3, export, gate.
5. KATA ABLATION (user D1110; D1121: moved up, runs right after the D1119 queue; 'kataxx' / 'Katax' = KataAtaxx, whose
   net is built from 3x3 convolutions, while Jataxx = 3x3 tuples, source only, no weights): what KataAtaxx looks at. Cheap first: the D1053 residual protocol with Kata's VALUE as
   the target on Kata rows (train_v9 teacher value column; caveat D101: TD values carry the game result) = which
   groups explain Kata beyond our champion. Expensive later: occlusion of squares on the Kata net itself (needs the Kata
   engine; 300-600 ms per CPU eval, so only a few thousand positions). Earlier Kata internals: docs/archive NET_ANALYSIS
   Part 3.
6. Head: int8 + pairwise (32% of the eval) on any winner; D4-consistency loss in training (43 cp asymmetry).

## 2026-11-18 03:53 A5 RESULTS: 3x3 tuple NETWORK (Jataxx input, Zataxx architecture) - D1135-D1165
All numbers: lab standard (whole dataset, champion target, D4 augmentation, held-out = last 1,528,091 rows = whole self-play games of main), held-out weighted |sigmoid - t|^2.5 loss; champion 812C2118 = 0.012118 on the same rows.
| model | data | best held-out | vs champion | file |
|---|---|---|---|---|
| linear 3x3 tuple SUM (no hidden layer, D4-tied, 63,369 w) = lower bound only | train_v11 | 0.019011 | +57% | logs/hce/d1130_t3.log |
| 3x3 network 256 SCReLU (126M w), exact Zataxx graft, 6 epochs | train_v11 | 0.011844 | -2.3% | logs/hce/d1138_jat3.log |
| same, continued 1 epoch (then overfit: epoch 2 worse) | train_v12 | 0.011816 | -2.5% | logs/hce/d1148_jat3.log |
| downscaled 256 -> 128 (shrink --refit: unexplained 21.25%, refit RMSE 145.7 cp), SATURATED epoch 60 (D1166) | train_v12 | 0.011191 | -7.65% | logs/hce/d1152_jat3_128.log |
- 128 net by empties (D1167, logs/arch/d1156_jat3_128_compare.txt): 0-6 -16.50%; 7-12 +3.87%; 13-24 +6.51%; 25-36 -0.33%; 37-49 +0.61%; blend with the champion -12.9% (a=0.60); best move differs in 54.09%. Gain only where the search usually reaches the game end anyway (<= 6 empties): play value unproven.
- Where the 256 net wins (D1145, logs/arch/d1144_jat3_compare.txt): 0-6 empties -6.07%; 7-12 +1.12%; 13-24 +3.99%; 25-36 +0.29%; 37-49 +4.30%; blend with the champion -8.0% overall (complementary, error correlation 0.86-0.97), best move differs in 52% of positions.
- Speed (D1147, lab loops vs engine champion, quiet PC): champion 4,949 ns; 3x3 untied 256 i16 4,111 (bimodal 1.8-5.0k); untied 128 i16 1,280; D4-tied 256 i16 1,378; D4-tied 128 i16 601.
- NOT done: games (plain alpha-beta eval-gate at equal nodes), engine port + incremental, hybrid with the champion inputs.

## 2026-11-18 07:20 WHAT THE 3x3 NET SEES THAT THE CHAMPION MISSES (D1169; lab jat3-explain, logs/arch/d1169_jat3_explain.txt, d1170_jat3_explain_e12.txt, d1170_jat3_explain_e49.txt)
- Gain by empties: 2-5 empties -13..-20% (all of the gain); 1 empty +21.7%; 8-24 empties +4..+10% worse; 33-41 slightly better.
- Most-biased champion patterns (z 10-28, n 5k-90k): FULL SOLID 3x3 blocks of one colour at corner / edge windows. Champion too pessimistic for the owner, too optimistic for the other side by 1-3 win-probability points beyond its mean bias at that empties count; the 3x3 net corrects about half (full corner block = +1,069 cp row).
- Window occlusion: corner windows 2-4x the centre in every band.
- One hand-made feature explains it best: SAFE STONES (no empty square among the 8 neighbours, so no move can convert them until a jump empties a neighbour). Weighted correlation with the champion's error: +0.127 (0-6 empties), +0.104 (7-12), +0.057 (7-49); plain stone difference +0.108 / +0.041 / -0.011; reach-territory, mobility, parity ~0. ataxx.cpp reported 'unattackable stones +140 Elo' (D1013).
- CORRECTION D1170: the champion already sees safe stones per square (G1 stone code includes empty_neighbours; 0 = safe), so it has the information and under-values it. Likely missing: a GLOBAL safe-stone COUNT x phase (G6 has reach count and parity, not this). Next: residual with that one global group on the frozen champion (D1053 protocol), then add to G6 with zero-init rows, saturate, gate (BACKLOG top).

- 2026-11-18 12:2x S1 RESULT (D1183, standard): safe-stone COUNT one-hot on the frozen champion 0.011764 vs calibration 0.011786 (-0.19%), gataxx terms 0.011761 (-0.21%): a global safe count is not what the champion misses; the D1169 bias is multi-square shape. Calibration alone -2.74%.

## OBJECTIVE (user principle D1189, 2026-11-18): inputs are levers, hidden layers leverage them
Maximise fitness (held-out loss, then Elo at equal time) per nanosecond of inference. Every input group is judged by two
numbers: INFORMATION = paired held-out gain +- SE (lab residual-pair, D1188; frozen-champion linear test = lower bound)
and COST = ns per eval on a quiet PC (extraction + accumulator rows added / changed per move, D1091 ~16 ns per row).
A lever's value THROUGH the hidden layers (interactions) needs the joint test: champion + new rows (zero-init graft),
saturated, paired against the champion retrained the same way without them. Width / depth / buckets are chosen by the
same gain-per-ns ratio. Small gains are never dropped on size (D1188).

## LEVER TABLE (D1190/D1201, lab standard, frozen champion + ONE linear group, paired vs calibration gB 0.0117859, block SE)
| lever | loss | gain vs gB | z | status |
|---|---|---|---|---|
| w3 3x3 window codes (D4-tied) | 0.011621 | ~-1.40% | pending | biggest so far |
| g0 gataxx terms (material, linear safe, edge, reach, cohesion) | 0.0117607 | -0.214% +- 0.069% | 3.1 | significant |
| s9 safe-stone counts one-hot | 0.0117635 | -0.190% +- 0.022% | 8.5 | significant |
| s10 inside pieces (erosion d1+d2, user D1185) | 0.0117724 | -0.115% +- 0.024% | 4.9 | significant |
Linear frozen test = LOWER BOUND; joint graft test next for every significant lever (D1189). 11 more groups in the D1193 screen.

## LEVER ATTENTION design v1 (D1199/D1200, user: 'attention is all you need'; attention BETWEEN inputs)
- Tokens = the champion's input groups per perspective: G1 square state, G2 anchor, G3 stone ring, G5 2-ply exchange,
  G6 global (+ later the new levers: safe counts, erosion, 3x3 windows) -> 5 (later ~9) group accumulators a_{p,g}
  (256 wide each, = the champion's FT rows summed per group; their sum IS today's accumulator, FT bias on the G1 token).
- Cheap attention: key/query per token = 256 -> 8 projection of the token (10 tokens x 2k MACs), 10x10 scores, softmax;
  output = context-dependent group gates: acc'_p = sum_g (1 + gamma_{p,g}) a_{p,g}, gamma = attention-mixed scalars
  (or per-channel later). Zero-init gamma = EXACT champion at step 0 (C14). Head = the champion's l1 / l2 / out.
- Cost estimate (to measure, quiet PC): FT rows unchanged (incremental per group as today); extra ~25k MACs = ~1 us
  scalar, less with SIMD; lab tf-bench 'LEVER attention' times the head part.
- Training: whole standard data, champion target, D4, saturation; paired held-out vs the champion AND vs the champion
  retrained identically without attention (so the data/recipe gain is not credited to attention); then eval-gate at
  5,000 nodes (B11b); then engine port + ti gate at equal time.
- Where to build: the hub trainer (TrainableNet backprop exists there; GPU path needs a bullet graph op) - CPU first.

## 2026-11-18 13:55 UPDATE (D1205)
- Lever table add: l1 Funes regions (material, corners, edges, corner pockets, centre arms, graded holes, singles) -0.245% +- 0.032%, z 7.7.
- FITNESS -> ELO (eval-gate 5,000 nodes, the B11b standard): nn_v13sat (held-out -8.55% +- 0.36%) vs champion = -0.9 [-24.6, +22.8] over 400 pairs: no strength per node from a large held-out gain. nn_v13anneal running. Open: why game-position accuracy does not become strength per node.
- Lever attention v1 attention arm running since 13:48 (step-0 parity OK: 47.7 cp max = quantisation, val 0.012115 vs 0.012118).

## 2026-11-18 14:13 Depth at 1 s, Titanium vs Zataxx (D1208/D1209)
Quiet PC, 60 positions (12 per empties band), movetime 1000, single thread, default hash. Median completed depth Titanium / Zataxx: 40-49 empties 12/17, 30-39 12.5/17, 20-29 12.5/17.5, 10-19 15/20, 2-9 17.5/22, all 13/19. nps 203k / 1,514k (7.5x), nodes per move 188k / 1.09M. Same best move 49/60. Raw: logs/depth1s/d1208_results.tsv. Titanium is ~6 plies shallower because a node costs ~7x more; this is the fitness-per-ns gap the lever table must close (cheaper eval, fewer active rows, incremental updates), alongside more data.
