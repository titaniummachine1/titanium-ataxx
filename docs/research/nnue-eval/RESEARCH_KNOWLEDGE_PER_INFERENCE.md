# Maximum knowledge per inference cost: HCE knowledge + how Stockfish does it (D1063, user 2026-11-17)

User subject: "squeeze as much knowledge as possible with the least possible inference, using the knowledge from the
HCE; and learn how Stockfish did it." Companion plan: docs/research/nnue-eval/NNUE_EFFICIENCY_PLAN.md (D1055).

## 1. What our own measurements say (2026-11-17)
| Finding | Number | Source |
|---|---|---|
| Node time of main 0120fd12 | 6.0 us/node; SmallNet halves it (11.6 us without); fused policy ordering ~0 ns and -29% nodes | D1051 |
| Big net full forward (lab) | 5.1 us; the ENGINE also does a full refresh every eval (nnue.rs:365-411, no incremental accumulator; corrected D1072) | D1051, D1072 |
| Rows changed by one move | 135.8 of 208 active rows (G2/G5/G6 + G3 ring change globally): full incremental saved ~35%, measured -12% nps | D188, D426 |
| What the frozen champion LACKS (held-out games, linear correction, beyond a calibration baseline) | pair patterns -2.65%, all HCE -3.05%; dynamic PST -0.65%, soft weaknesses -0.24%, counts / regions / raw PST -0.3..-0.5% | D1059, D1061 |
| Weakness information cost | all squares as bitboard planes 40.9 ns; per-stone table extraction 1,175 ns (lab) / ~243 ns (optimised HCE) | D1062 |
| Incremental footprint of a move | 3.18 changed squares; touches 31.9 of 80 pair placements (2x2 windows: ~10) | D1056 |
| Champion symmetry error | 40 cp mean / 858 cp max across the 8 board symmetries | D1035 |
| Labels | today's main labels differ systematically from v9 labels: a val set must come from the same games (held out by game AND position) | D1053, D1060 |

## 2. How Stockfish does it TODAY (read from the master SOURCE, 2026-11-17, D1070)
The first version of this section (from one nnue-pytorch doc, partly summarised) was out of date; this replaces it.
Read: src/evaluate.cpp/.h, nnue/nnue_architecture.h, nnue_feature_transformer.h, nnue_accumulator.h/.cpp, nnue_common.h,
network.cpp, features/{k32h3,full_threats,pp_3wide}.{h,cpp}, bitboard.h. NOT read: position code that builds the dirty
threat list, layers/*.h, nnz_helper.h, the search call site of optimism.

**The answer to "which eval compute does Stockfish replace with handcrafted heuristics": none.** The handcrafted eval is
gone; there is no small net any more (one net, `nn-bddd9d57eb87.nnue`) and no PSQT output channel. The only handcrafted
code left is output SCALING: material-only `simple_eval` as a correlation reference, `v += material*v/90649`, rule50
damping. Handcrafted KNOWLEDGE survives in three cheap places instead:
1. **As sparse INPUT features** (cost = active count x incremental update):
   - K32H3 piece-square: 32 king buckets x 11x64 x 3 material blocks (enemy queen? / rook-or-queen?) = 67,584 rows;
   - **FullThreats**: one row per directed attack (attacker piece, from, to, attacked piece), not king-bucketed,
     redundant pairs removed, 59,808 rows, up to 256 active;
   - **PP_3Wide pawn pairs**: an unordered pair of pawns (either colour) on the same or adjacent files, 4,560 rows.
   All three add into ONE 1024-wide i16 accumulator; threat + pair rows share one table and use no block.
2. **As a cheap handcrafted CLASSIFIER that picks one of 192 layer stacks** (only one runs, so capacity at ~0 cost):
   piece-count band (8) x queens (4) x pawn layout (3: both halves / one half / none) x 'many officers' (2).
3. **As output scaling** (material, rule50), described above.
Cost engineering: L1 1024 (512 per perspective after the pairwise product) -> 32 -> 32 -> 1 with SqrCReLU+CReLU concat
and a skip term; sparse first layer (non-zero list); i8 weights for PSQ AND threats/pairs, i16 accumulators, u8 activations
(OutputScale 16, WeightScaleBits 6, HiddenOneVal 128); Finny caches [block][king sq][colour] for refreshes (threats and
pairs are re-added from scratch on a refresh); lazy accumulators updated forward or backward from the last usable state;
a 'hybrid' update for king moves inside the same half.

| Stockfish (source) | Titanium today | Ataxx analogue / action |
|---|---|---|
| No HCE in the eval, no small net, one net | big net + futility SmallNet (halves our node time, D1051) + policy head | keep the SmallNet (our search needs it); do NOT replace net compute with an HCE |
| Pawn PAIRS on 3 adjacent files as inputs | none | **pair patterns** = the largest thing our net lacks (-2.65% held out, D1059) |
| THREAT inputs (attacker, from, to, victim) | none | **weak-stone / capture rows**: "stone S flips if the enemy moves to T" (weakness planes 40.9 ns for all squares, D1062) |
| 192 layer stacks chosen by a handcrafted class (count x queens x pawn layout x officers) | 16 stacks by piece count | **class stacks**: empties band x mobility class (both clone? jump-only side?) x material sign x frontier/region class; zero inference cost |
| Material blocks change the PSQ table (3 slabs) | one PSQ table | one slab per material-lead class (cheap, refresh only when the class changes) |
| i8 weights everywhere in the FT, i16 accumulators | FT i16 weights | i8 FT was lossless +12% nps (D440/D465): needed once inputs grow |
| L1 1024 wide, 32-32 tail, sparse L1 | W256, L1 skips ~86% zeros | lean = narrower FT (W128 2.8 vs W256 5.1 us full forward, D432) paid back by the new inputs + stacks |
| Finny caches + lazy forward/backward updates; PSQ incremental, threats/pairs re-added on refresh | FULL REFRESH every eval (corrected D1072) | **split by group**: local groups incremental on the search stack, global groups (G2/G5/G6, G3 ring) re-added per eval; measure the per-group row-change share first (D1072 B/C) |
| Output scaled by material and rule50 | none | optional: scale by empties / mobility (cheap) |

## 3. The Titanium version of the Stockfish recipe (ranked experiments)
1. **Pattern inputs (PP_3Wide analogue):** the top D4 orbits of the pair placements (24 of 80) or 2x2 window codes
   as extra sparse FT rows, coalesced from factorised parts during training; graft (C14): new rows start at zero =
   exact champion. Measure: held-out val on val_v11_clean, active rows, ns per eval, then gates.
2. **Threat inputs:** sparse "weak stone" rows (square x weakness level, only stones with level >= k), updated only
   around the changed squares (40.9 ns for all planes). Same measurement.
3. **Width trade:** with 1 + 2 in, train W128 (and W192) and compare against the W256 champion at equal ns per eval
   (the D1054 curve): the goal is champion quality at half the first-layer cost.
4. **Symmetry:** a D4-consistency term in the loss (no inference cost) first; weight tying by symmetry class only if
   a cheap canonical index exists.
5. **Bigger teacher, same student:** train a larger net offline and distil it into the W256 student (the ceiling comes
   from training, the cost stays the same).
5b. **Class layer stacks (Stockfish's 192 stacks):** pick the stack by empties band x mobility class x material
   sign (zero inference cost; init every new stack from the matching current bucket = exact champion, C14).
6. **Helpers:** one child model (SmallNet leader + policy logit, D1051) instead of two; HCE tables only where they win.

## 3b. User idea D1072 (2026-11-17): pairs decay with distance; lazy updates per element
Stone relations in Ataxx are local but leak wider with fast-decaying strength (like pawn pairs across ranks). Tests:
(A) residual study of stone pairs by Chebyshev distance band 1 / 2 / 3 / 4+ x colour combo (own-own, own-opp, opp-opp),
D4-tied, one group per band, held out, saturation: shows how fast the missing pair information decays and sets the pair
radius (incremental cost: a changed square touches 8 pairs at r1, 24 at r2, 48 at r3). Today's measured pair group is
gataxx's 80 placements of two adjacent 2x2 blocks (= distance 1-3 inside a 4x2 / 2x4 / corner box), not distance bands.
(B) per-group row-change share on real search moves; (C) incremental accumulator for local groups + per-eval re-add of
global groups in the engine, bench at equal nodes, then time legs.

**(B) MEASURED (D1073, lab group-diff, 123,799 children of 3,000 val_v11_clean positions, per accumulator):**
| Group | active rows | rows changed per move | ratio | incremental cheaper |
|---|---|---|---|---|
| G1 square state | 49.00 | 26.86 | 0.548 | 97.2% |
| G2 anchor | 12.48 | 12.63 | 1.012 | 52.1% |
| G3 stone ring | 27.27 | 16.90 | 0.619 | 85.3% |
| G5 2-ply exchange | 12.03 | 9.80 | 0.814 | 56.9% |
| G6 global (pre-summed) | 1.00 | 1.03 | 1.028 | 48.6% |
Row ops per accumulator: full refresh 101.78 (today); G1+G3+G5 incremental + G2/G6 re-added 67.03 (-34.1%); adaptive
per node and group 58.49 (-42.5%). G1 and G3 are local; G2 (anchor moves) and G6 are global. Extraction, accumulator
copies and cache-hit parents are not counted; the FT-sum share of the current eval is still unmeasured.

## 4. What is NOT known yet
- Whether any of 1-3 gains Elo (all numbers above are val / cost proxies; D1026: val alone does not rank evals).
- The engine-path cost split of one big-net eval (FT update vs dense layers) and big-net evals per node.
- The hub trainer and the engine have a FIXED feature set (crate::nnfeat): experiments 1-3 need a feature-set
  extension in both (with a parity test, A11 / trainer-input-matches-engine rule).

**(C) MEASURED (D1088): partial incremental does NOT pay.** Built (G1 + G3 incremental, G2/G5/G6 re-added, bit-identical,
node-identical) and timed: full eval 4.8 us/child vs local 4.9 us/child + 3-5 us per-node build; bench depth 12 -10% nps.
Row sums are memory-bound at 12-14 ns per 512-B row (95.7 rows per accumulator = 1.15 us; the diff list 67.1 rows = 0.9 us),
so the best case is ~0.5 us of 4.8 us per child, eaten by the diff bookkeeping; the SmallNet prunes most children before a
big-net eval, so the per-node build is not repaid. The (B) estimate (-23..-29%) was based on a wrong FT share (69%; real
~48%). Levers that attack the real cost (bytes per row): i8 FT rows (half the bytes), narrower W, fewer active rows.
Patch of the tried code: docs/analysis/d1088_local_acc_tried.patch.
