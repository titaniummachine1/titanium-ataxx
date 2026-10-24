# What matters in Ataxx evaluation and move choice (ranked, measured) — living summary

Purpose: the short list to design the NNUE (and any HCE / move ordering) from. Every line points at the measurement.
Details: docs/NET_ANALYSIS.md (sections 0-21). Started 2026-10-24. Update this file whenever a new measurement lands.

## A. VALUE: what decides how good a position is (Kata 200-visit TD value = reference)
Ranked by error removed on held-out positions after material (section 21) and by direct tests on Kata's net (sections 18, 20).
1. **Stone permanence / safety.** A stone that cannot be converted next move (no empty neighbour, or no enemy landing square next
   to it) is worth 2.5-3x a convertible stone to Kata's value head (flip test: enclosed .81 vs .22-.37; not attackable .79 vs .32).
   Kata computes "exposed" in block 0 and "safe-stone difference" from block 0 (R2 .81 from 0.02 raw). Titanium's 5k labels are
   blind to it (+.000). Walls (edge .66, corner .78 vs interior .39) count because they remove landing squares.
2. **Local patterns (Moonbird's 2x4 / 4x2 / diagonal pattern pairs + per-square stone environment).** The biggest single group
   after material in 3 of 4 phases and the only group with a large unique contribution in a tree model. Carries something the
   hand concepts do not (likely shape: which stones protect which landing squares).
3. **Material** (the base; R2 .22-.36 alone vs Kata).
4. **Phase.** Everything changes with the number of empties: stone value .63 (8-15 empties) vs .37 (25-38); safety weight in move
   choice 1.70 (10-18) vs .95 (28-36). Use phase buckets / output buckets.
5. Small or nothing: reach / mobility differences, best captures, 2-ply exchange, empty-region count and parity, territory
   (each <=.0005 MSE after the above). Parity is represented weakly by Kata and dropped by its last layers.
- Ceiling: linear hand + Moonbird features close 57-78% of the gap from material to Ataxx_CPP's deep net; trees/knob tables add
  ~nothing => the rest needs information only a deeper net extracts (value R2 in Kata's trunk rises to .82 only by block 11).

## B. POLICY: what decides the move (Kata 200-visit search choice = reference)
1. **Jumps with doomed stones.** 92% of Kata's jump origins are attackable stones (72% base rate). Jumping costs ~2.2 converted
   stones in the choice model, ~1 stone if the origin was going to be lost anyway.
2. **Safety after the move.** One more safe stone than the enemy after the move ~ +1.06 converted stones; each own stone left exposed
   -0.36. Against the 2-ply material move, Kata's choice gives up 2.47 stones of material for 1.29 fewer exposed stones.
3. **Conversions** (1.00 = unit) and **material after the enemy's reply** (+0.50 per stone): matter, but Kata trades: its jumps
   convert 3.2 and allow a 4.0 recapture.
3b. Lead -> clone (section 26): further ahead, Kata clones more (clone share .42 at lead <= -10 to .49 at >= +10; .47 vs .36 in
   won vs even endgames). Hopeless positions also clone more (U-shape). Small effect overall.
4. Small: landing next to filled squares (-0.29 per neighbour), clone reach (+0.15), recapture (-0.15), territory (+0.06).
- Concentration: the raw Kata policy puts 90% of its mass on ~4 of 22 first-step choices; the search's move is in its raw top 3
  in 94%. Two thirds of the policy logit is not explained by cheap features; it forms gradually over all 12 blocks.
- **Cheap-feature ceiling for Kata's full move (~55 legal): top-1 ~31%, top-3 ~60%, top-5 ~75%** (20-feature choice model and a
  tree ranker agree). 2-ply material + conversions alone: top-1 30%, top-3 53%. Safety/permanence features improve the top-3,
  not the top-1. Exchange rates in 1-3 are GROUP effects (the safety features are collinear).
- => A hand-written move scorer cannot reproduce Kata's choices; a policy NET can (root ordering by Ataxx_CPP's policy is the
  realistic way to use it; see E).

## C. What the strong nets look at (geometry)
- Everything within Chebyshev distance 2: 3x3 ring heads and distance-2 jump-ring heads; a few global pooling heads. No direction
  preference. Kata block 0 = exposure and own-gain units (single neurons, |r| .86-.96); block 1 = reach (needs 2 attention hops).
- Kata's training data: win/loss at 5 horizons, 2-stage policy, future-stone aux; NO score, NO ownership target.

## A2. What S57 (our champion eval) is missing (section 23)
- S57's STATIC eval is already the closest cheap eval to Kata (closer than Moonbird's, much closer than Titanium's 5k search).
- It already contains the cheap knowledge: all hand groups + Moonbird tables on top of S57 close only 13-33% of its gap to a deep
  net; permanence alone 5-24%. => an HCE add-on gives little; the missing part is DEEP and sits mostly in the endgame
  (gap .013 at 9-15 empties vs .004 in the opening). Rule check (section 25): a rule flag explains only 5-10% evenly; at 0-8 empties
  S57 is nearly blind but the search reaches the end there. The gap that matters for play is 16-35 empties.

## D. Consequences for the NNUE (proposal D63, games decide)
- **S57 tuple output as a skip term; the NNUE hidden layers learn only the residual; train the teacher stage on Kata TD.**
- 2-ply anchor / jump 2-ply / threat edges: demoted to later ablation arms (≈0 value gain on top of S57).
- Inputs must make permanence cheap: per stone (square, own neighbours, enemy neighbours, walls) — Moonbird env as sparse inputs —
  plus "attackable" per stone. Piece-square inputs + a hidden layer for the pattern part (Moonbird pairs are too big as inputs).
- Phase buckets (output buckets / layer stacks) by empties.
- Reach / 2-ply / region features: low priority for VALUE (screen says ~0); keep 2-ply and safety-after-move for MOVE ORDERING work.
- The 2-ply anchor (D61) was chosen from policy hit rates; for value it adds ~nothing here: re-check before building it.

## E. Not yet measured / open
- Whether any of this converts to Elo in alpha-beta (no game played on any of it).
- What Kata blocks 3-11 compute for value and policy (not a cheap concept from our list).
- Block 10 unit 559 ("my stone locked inside enemy stones"?) unverified.
- Root ordering (measured, section 22): Ataxx_CPP's one-pass policy finds Titanium's 1M-node move 44% (Ti@1k 37%, Ti@5k 47%,
  Ti@50k 56%); in 17% the policy has it while Ti@5k does not. Promising for shallow roots (5k gates, fast TC); no game played yet.
