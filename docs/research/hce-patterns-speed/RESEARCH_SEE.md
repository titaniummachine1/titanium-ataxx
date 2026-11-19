# RESEARCH_SEE: cheap approximations of the 2-ply exchange (D944, 2026-11-16)

User order: the hand-crafted exchange heuristic may be under-finished; make it as fast as possible; the network already understands exchanges, so a more precise exchange INPUT could gain Elo; exact 2-ply is thought too expensive, so look for better cheap approximations.

## What the exact value is
For a move (clone or jump to `to`): my_diff = clone + 2*flips (popcount, free); opp state = 0 if the opponent has no move, else 1 + the opponent's best reply gain (clone 1+2v, jump 2v, v = my stones around the landing). Engine / policy2 path: victim planes (bit-sliced counts of my stones around every square) of the parent position, +1 ring of the destination, +1 ring of every flipped stone, -ring of the jump origin, then planes_max over the clone / jump landing squares. TPOL5 `--see 2d8` indexes (phase, my_diff, opp state).

## Tool and data
`ti see-screen --tdat X.tdat --rows N --seed S [--bench]` (hub `src/seescreen.rs`, hub commit after a6c8aa2). 29,969 random positions of train_v10.tdat, 1,365,640 moves. Ground truth = brute-force 2-ply (make the move, enumerate the opponent's moves).

## Results (run 2026-11-16, PC loaded by a 30M-node search, ns numbers indicative)
| approximation | exact match | coarse-bucket match | mean abs error | H(exact given approx) | top-1 net move differs | cost |
|---|---|---|---|---|---|---|
| exact (planes) | 100.00% | 100% | 0 | 0 bits | 0% | 28.1 ns/move (5.7 of it movegen + node setup) |
| noflip (planes + destination ring only) | 42.69% | 67.6% | 1.54 | 2.52 | 44.1% | 23.0 |
| parent (as noflip, landings from the parent) | 42.14% | 69.9% | 1.46 | 2.53 | 43.5% | 15.7 |
| levels (top-2 count levels, mask tests only) | 42.06% | 70.1% | 1.45 | 2.54 | 44.2% | 13.4 |
| levels3 (levels + exact bonus from flips in the 5x5 window) | 51.25% | 77.6% | 1.09 | 2.21 | 39.2% | 39.6 |
| node (one constant per node: the opponent's best reply if we passed) | 33.15% | 59.8% | 1.85 | 2.72 | 49.4% | 10.0 |
The exact planes path equals the brute-force 2-ply on all 1.37M moves (0 differences), so the reference is trusted.

## Findings
1. Every cheap approximation loses most of the information: the best one keeps 51% exact matches and removes only 0.5 of the ~2.7 bits the node constant leaves open. The information is in the FLIPPED stones (noflip and levels are barely better than the node constant; adding flips lifts 42% -> 51%) and in the changed landing sets, which cannot be approximated by a few mask tests.
2. The exact path is already cheap: about 22 ns per move above movegen (bitboard carry-save adders, no per-move search). The flip-aware approximation costs MORE (39.6 ns). There is no cheap shortcut at the per-move level.
3. So "2-ply is too expensive" is not the real problem: exact costs ~-6.8% nps when run at every node (D744) and still did not pay in play (D748-D750). What can still be won is WHERE it runs, not a worse formula:
   - two-stage ordering: rank by the cheap score (policy without SEE), compute the exact state only for the top K (3-4) moves, ~10x less work per node (22 ns x 4 vs x 45);
   - exact state only at nodes where ordering decides (depth >= 2, or when the best cheap scores are close);
   - the SmallNet exchange-margin input (D941/H21) pays the cost once per child that is evaluated anyway.
## NOT checked
Information value inside the TRAINED policy (retrain TPOL with each approximation as input, needs hours); play strength of any variant; node counts. This screen only says how close each approximation is to the exact value.
