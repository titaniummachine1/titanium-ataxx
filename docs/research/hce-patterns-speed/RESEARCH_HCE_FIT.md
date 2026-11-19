# RESEARCH_HCE_FIT: what the champion NNUE looks at, as hand-crafted terms and piece-square tables (D948/D949, 2026-11-16)

> **IN-SEARCH TL;DR 2026-11-17 ~03:15 (D1018-D1032): val loss does NOT rank evals by playing strength; gate them inside a search.**
> Same search, two evals (ti sprt, libataxx rules, 4 parallel games): our D991 HCE vs gataxx 2.0.0's eval = **+109 +-54 per node (5k nodes)**
> in Zataxx's search and **+129 +-63** in gataxx's own search (our HCE called from Java through titanium_lab.dll, parity 200/200).
> At 100 ms it lost only on speed (-124 / -138, 2.6-5x fewer nodes); the **fast HCE** (direct base-3 pair table 80 x 6561 + fused terms,
> 437 vs 2,063 ns/eval, parity 0.003 cp) turns the Zataxx-search 100 ms result into **+67.1 +-40.1** (278 g). Versus Zataxx's own net
> (2x2-window NNUE) our HCE is -209 per node and -196 at 100 ms although its val loss is better (0.010904 vs 0.011393): game-position
> val loss misses what a search needs. Zataxx-architecture nets retrained on our data: -54.6 per node, -16 at 100 ms: closed (same speed,
> no eval gain). Pair-table ablation (lab hce-ablate): the tables encode edge structure (edge orbit a1-c1 = 1/4 of their value; centre
> diagonals ~0; gataxx's mobility table +0.01% = useless). Export lesson: Zataxx's AVX2 flatten keeps 16 bits of clamp(h)*w, |w| must be <= 128.
>
> **TL;DR 2026-11-17 01:38 (D986-D1012), all on the same 564,834 val_v9_clean rows, lab hce-train val metric** (external engines: one fitted scale k):
> NNUE 812C2118 **0.008978** | Zataxx net (36 2x2 windows -> 256 SCReLU) trained like the NNUE **with D4** **0.009928** (agreement 53.43%) |
> without D4 0.010283 (overfit, 49.27%) | **our HCE 32b + gataxx-style pair tables + mobility 0.010904** (48.72%; restart D1007 = no gain) |
> Zataxx shipped 0.011393 (50.68%) | gataxx shipped 0.012431 | our HCE 45b / 32b / 16b without pair tables 0.012496 / 0.012548 / 0.012666 |
> gataxx base terms 0.017180. Lessons: measure the external baseline first (D993); D4 always on (D1001); the pair tables were the missing
> HCE piece (D991). Tools: lab hce-train --pairs 1, val-fen / val-score, zataxx-val, zataxx-train (titaniumtest lab, commit cdeb49d).

User order: find the hand-crafted engine branch and build the best possible hand-crafted eval to understand what the net sees; then: PST for captures, PST for jumps, and a DYNAMIC PST (stone value by its 3x3 area, own and enemy pieces separately).

## Where the old HCE is
`C:/gitProjects/wt-classical` (2026-09-21, S13 contact-eval, merged then): material 100 + PST + tempo 150 + E2 contact penalty 15 per endangered stone + E4 multi-capture exposure 20 per extra stone. The same hand eval still exists on main as `search::evaluate` (used only as an input of tools). Speed branches of that era: exp/slim-eval-speed (2026-09-20), exp/speed-eval .. exp/speed4 (2026-11-01, NNUE speed, bit-identical).

## Method
Hub `ti hce-fit` (titaniumtest src/hcefit.rs). Target = bundled champion 812C2118 static eval (cp, side to move, clamp +-2000). Data = 20,000 random train_v10 positions + all 1,003,854 children (fit) and 2,000 other positions + all their children (validation). Closed-form ridge regression (exact optimum, so "saturated" by construction, LAWS C15-safe), 4 phases by empties (0-12, 13-24, 25-36, 37-49), squares folded to the 10 D4 symmetry classes. Metrics on validation: R2, MAE, SIBLING BEST-MOVE AGREEMENT with the NNUE (which child the eval prefers, the metric that exposed TSTG, D439) and regret in NNUE cp. Run: 33 s on the PC. Raw report: logs/hce/d949_hcefit_20k.txt (all tables).

## A) Position model
| model | R2 | MAE cp | best-move agreement | regret cp |
|---|---|---|---|---|
| bias + material | 0.345 | 194.6 | 25.9% | 164.1 |
| + static PST | 0.519 | 163.3 | 25.6% | 177.0 |
| + dynamic PST (3x3 counts) | 0.682 | 127.4 | 35.9% | 103.3 |
| + hand terms (no PST) | 0.676 | 137.8 | 41.6% | 62.9 |
| ALL minus dynamic PST | 0.735 | 120.1 | 42.2% | 62.3 |
| ALL | 0.779 | 103.5 | 45.6% | 55.5 |
Drop-one ablation from ALL (dR2 / d agreement): dynamic PST 0.044 / 3.45 pts, see2_me (best 2-ply exchange of the side to move) 0.016 / 2.45, holes_shared 0.003 / 0.75, frontier 0.002, move_count 0.001 / 0.5; every other term < 0.001 (material and static PST are fully contained in the dynamic PST).
Weights that stand out (cp per unit): see2_me 41 (empties 0-12), 13.5 (13-24), 12.3 (25-36), 65 (37-49 = opening); material 48-73; frontier -21..-33 (stones next to an empty square cost); holes (empties nobody else can reach) +24..+61 in the middlegame; opponent without a move -345 / side to move without a move +123..+297 (Botzone pass rules, endgame).

## Dynamic PST (the user's idea) reads like this
Value of a stone depends mainly on how FILLED its 3x3 is, not on the square: e.g. empties 0-12, an own centre stone with 0 enemy neighbours +159 cp, with 8 enemy neighbours -201; an own corner stone gains value with neighbours (+3 own +44, +3 enemy +140): a corner whose neighbours are filled can no longer be flipped. Own stones with many own neighbours are worth more on edges (+54..+131 with 3-5 own neighbours) and less in the centre. Static PST alone explains far less (R2 0.52 vs 0.68): the net's "PST" is dynamic.

## B) Move model (what a move is worth to the mover: -eval(child) - eval(parent))
R2 0.360, best-move agreement 31.6%. Ablation: captures -16.0 pts agreement (R2 -0.125), opponent best reply (exact 2-ply state) -6.4 pts (R2 -0.074), square tables -0.05 pts (R2 -0.025). Greedy capture count alone: 25.5%.
Midgame (empties 25-36) tables: each converted stone ~ +45..+90 cp up to 5, flat after 6; opponent's best reply: 0 move available +199, gain 1 +188, gain 3 +14, gain 9 -27, gain 15 -241 (about -35..-45 cp per 2 stones of reply); JUMP ORIGIN PST: leaving a corner / edge square costs -79..-83 cp, leaving a centre square +31..+35 (the hole you leave behind is what the net prices); jump destination: centre -28..-30, corner +6; clone destination: -24..+9 (small).

## Findings
1. The net's position value is mostly material + a DYNAMIC, neighbourhood-conditioned PST + the 2-ply exchange; static PST is a poor proxy (user's hunch confirmed).
2. The 2-ply exchange (D944) is the 2nd strongest hand term in positions and the 2nd strongest in moves: the net does price "what the opponent takes back".
3. Jumps are judged by the HOLE they leave (origin table), not by where they land.
4. A linear HCE with all of this reaches R2 0.78 and 45.6% sibling best-move agreement with the net; the remaining gap is nonlinear (consistent with D459, where the net's own inputs used linearly reached 54%).
## NOT done / next
Full 3x3 PATTERN tables (3^8 configurations per class, not counts) as the dynamic PST; interaction terms (dynamic PST x 2-ply exchange); an HCE engine build + equal-search gate vs the net; KataAtaxx comparison (D945).

## C) WEAKNESSES (user idea D951, run 2026-11-16, report logs/hce/d951_hcefit_weak_20k.txt)
Weakness of a stone = an empty square of its 3x3 the other side can land on (clone or jump); mask = box2(enemy) & empty, so unreachable empties drop out (O(1) with bit masks; engine form = victim planes of the landing mask).
| model | R2 | best-move agreement | regret cp |
|---|---|---|---|
| material + static PST | 0.519 | 25.6% | 177.0 |
| material + weakness COUNT only (18 numbers per phase, no square) | 0.574 | 35.3% | 95.5 |
| material + static PST + weakness count | 0.685 | 37.2% | 88.7 |
| material + dynamic PST (3x3 neighbour counts) | 0.682 | 35.9% | 103.3 |
| material + weakness PST (square class x weakness count) | 0.712 | 39.6% | 82.7 |
| material + dynamic + weakness PST | 0.749 | 42.2% | 77.0 |
| ALL (everything incl. hand terms) | 0.792 | 48.25% (was 45.6% before weaknesses) | 50.3 |
Ablation from ALL: weakness PST dR2 0.010 / 2.15 pts agreement (largest with see2_me and dynamic PST).
Value per stone by its weakness count (material + weakness count model, cp, on top of material), empties 25-36: own stone 0 weaknesses +60.7, 1 +14.7, 2 +6.7, 3 -5.0, 4..7 +0..+13; enemy stone 0 weaknesses -51.0, 1 +15.1, then ~0..+21. Same shape in every phase: the big step is 0 -> 1 weakness (~45-55 cp in the middlegame); more weaknesses cost little more. In the opening (37-49 empties) the cost grows to 3-4 weaknesses (-31..-42).
Edge question (does the edge bonus come from fewer weaknesses?): own-stone static PST in empties 25-36 without -> with the weakness count: edge centre 17.5 -> 2.8, edge (0,2) 16.3 -> 7.5, next-to-corner 50.2 -> 39.4, corner 94.7 -> 85.4. So the EDGE bonus is mostly weaknesses (user right); the CORNER bonus is not: a corner has only 3 neighbours, so it can never get many weaknesses later and converts few stones when it is lost; a term for future exposure (empty neighbours regardless of reach) is the next test.
Reading: the net prices "safe stone" (0 weaknesses) vs "attackable stone" almost as a binary, plus corners; weaknesses explain more of the eval than any square table.

## D) Two-pass eval, network occlusion, piece mobility (D953-D956, run 2026-11-16, report logs/hce/d956_lab_20k.txt)
Tool moved out of the hub into `titaniumtest/lab` (`lab hce-fit`, dev profile: edit rebuild 7.9 s, engine compiled once optimised). This run took 447 s (pass 2 = 2 x ~45 pass-1 evals per position over 1M positions dominates).
Two-pass (user D953): PASS 1 = material + PST + weakness PST + pressure (1-0.5^attackers, weighted by stones converted): R2 0.728 / 42.1% agreement (without pressure 0.712 / 39.6%); PASS 1 + PASS 2 (best pass-1 gain over my moves / jumps / opponent moves, improving-move count) 0.758 / 46.0%; ALL 0.801 / 48.7% (ALL minus pass 2: 0.793 / 47.6%).
NETWORK OCCLUSION (user D954; 58,493 stones of 2,000 positions; value of one stone to its owner = eval change when it is flipped):
- by weakness count, empties 25-36: own 0 weaknesses 219 cp, 1: 90, 2: 92, 3: 76, 4..7: 72-84; enemy 231, 106, 98, 82, ... The net itself shows the same binary step as the HCE: a stone with no weakness is worth ~2.4x one with any.
- by square class (flip): corner 258, next-to-corner 151, edge 127/104, (1,1) 90, inner 54-73: corners and edges are worth the most, partly from weaknesses.
- PIECE MOBILITY (user D955/D956): the net values a stone LESS the more legal moves it has: own stone m=0 348 cp, 3: 183, 6: 106, 9: 81, 12: 76, then flat 75-93 up to 17. The user's increasing curve u(m) does not fit (fit slope negative, R2 0.47 on the means): a stone's free squares around it are potential weaknesses, not value. Corner stones: 0 moves 468 cp, 4 moves 216.
- LOCALITY (held-out R2 of the per-stone flip value): side x phase 0.232, + class 0.331, + weakness x class 0.470, + 3x3 neighbour counts 0.492, + 5x5 ring counts 0.495, + piece mobility 0.494, + best capture 0.503, + best safe capture 0.493. Half of a stone's value is local (3x3 + weaknesses); the 5x5 ring, mobility and capture terms add nothing; the other half is non-local / nonlinear.

## E) What SEARCH sees that the static eval does not (D958, `lab deep-see`, logs/hce/d958_deepsee.txt, 382 s)
3,000 random train_v10 positions, main search at 5k and 30k nodes (7 threads) vs the static eval; 4,615 stones of 150 positions flipped and searched at 5k.
| phase | mean |search - static| 5k / 30k | corr static vs 30k | sign flips (|score| > 50) 30k | 5k -> 30k change |
|---|---|---|---|---|
| empties 0-12 | 386 / 492 cp | 0.765 | 17.0% | 162 cp |
| empties 13-24 | 64 / 65 | 0.935 | 4.8% | 30 |
| empties 25-36 | 53 / 56 | 0.949 | 2.2% | 21 |
| empties 37-49 | 57 / 57 | 0.932 | 1.4% | 14 |
- The static eval agrees with search in the opening and middlegame (corr 0.93-0.95, 1-5% sign flips); the search's big corrections are in the ENDGAME (0-12 empties: ~400-500 cp, 17% sign flips): counting, parity, forced sequences.
- The correction is NOT a static pattern: all compact hand terms together explain 1.4% of it (held-out R2 0.014; best single terms press_conv2, weakness count, components, holes_shared ~0.3% each).
- Stone level: search values a stone MORE than the static eval (318 vs 227 cp), most for safe stones (0 weaknesses 509 vs 336). The per-stone gap is not local (class / weaknesses / 3x3 / mobility / safe capture explain 0-4%). The biggest gaps are all near-full boards where flipping one stone turns a decided game (search +-2000 = proven) around: the stone value there is the game result, not a pattern.
- NOT done: the stone-level study restricted to undecided middlegame positions (|score| < 300), where local patterns could show; deeper than 30k nodes.

## F) SOFT weaknesses (user D959, `lab hce-fit --no-pass2`, logs/hce/d959_soft_20k.txt, 120 s)
Soft weakness of a stone = an ENEMY neighbour that can jump out (empty square at distance exactly 2) while another enemy stone can land on the vacated square afterwards.
Network occlusion, mean value of one stone (flip, cp) by hard x soft weaknesses, empties 13-24 (own stones; enemy stones the same shape):
| hard \ soft | 0 | 1 | 2 | 3+ |
|---|---|---|---|---|
| 0 | 258 | 224 | 213 | 145 |
| 1 | 125 | 80 | 77 | 59 |
| 2+ | 98 | 70 | 66 | 62 |
- The net does price soft weaknesses: each costs ~15-45 cp, about a quarter to a third of a hard weakness; 3+ soft weaknesses take a safe stone from 258 to 145 (about half way to one hard weakness, 125). Diminishing like the hard ones (user's pressure idea holds for both).
- In the linear HCE the soft_weak term adds dR2 0.003 / +0.2 pts agreement (ALL 0.796 / 47.8% without pass 2).

## G) Area of effect and shallow search (D961/D962, 2026-11-16; logs/hce/d961_radius.txt, logs/hce/d962_depths.txt)
AREA OF EFFECT (`lab radius`, 400 undecided positions, every stone pair, interaction I = f(AB) - f(A) - f(B) + f(none) of the static eval): mean |I| by Chebyshev distance, empties 25-36 same / opposite colour: d1 48.7 / 32.0, d2 32.0 / 25.3, d3 21.6 / 20.0, d4 19.8 / 20.8, d5 20.7 / 24.3; empties 13-24 the same shape (44.5, 27.9, 21.5, 22.1, 22.9). So a stone's value depends strongly on stones that TOUCH it (3x3), about half as much on stones at jump distance 2 (5x5), and beyond that only on a flat ~20 cp floor that does not fall with distance (the global nonlinearity of the win-probability output, not a pattern). Area of effect = the 5x5 (radius 2 = the jump range), strongest in the 3x3. Opening and endgame interactions are larger and not distance-limited (piece-count / decided games).
SHALLOW SEARCH (`lab deep-see`, 3,000 positions): mean |search - static| (cp), empties 25-36: depth 1 42.6, depth 2 44.7, depth 3 48.7, depth 4 49.7, 100 nodes 48.1, 1,000 52.0, 5,000 52.7; empties 13-24: 49.3, 52.9, 58.3, 58.1, 57.4, 62.7, 63.7; empties 0-12: 168, 161, 198, 197, 207, 289, 386. In the middlegame a DEPTH-1 search already makes ~80-90% of the correction that 5,000 nodes make; only the endgame keeps growing with search. No hand term explains any of these corrections (held-out R2 -0.03..+0.001 at every budget): the eval is ~45 cp inconsistent with its own 1-ply search, and that gap is not a static pattern.
NOT done: the stone-level gap fit at 1,000 nodes on undecided positions (the run's log was cut before the end; its tables printed: search values stones more than the static eval, 263 vs 215 cp; safe stones most, 444 vs 335).

## H) HCE trained like the NNUE (D977, lab hce-train)
Same data / target / loss / validation as the NNUE (see D977). Smoke on 200k rows: val loss 0.019230 (weights fitted to imitate the NNUE) -> 0.013438 (trained on the dataset), NNUE 0.008978 on the same val rows; agreement with the NNUE's sibling choice unchanged ~39%. Full 37.7M-row run pending (logs/hce/d977_hcetrain.log).

Full-data result (D986-D988), all three on the same 37.7M train / 564,834 val rows (set 4 = ours+gataxx):

| buckets | weights | init val | best val | vs init | agreement init -> trained | epoch time |
|---|---|---|---|---|---|---|
| 16 | 3,376 | 0.016864 | 0.012666 | -24.9% | 39.57% -> 40.67% | 13 s |
| **32** | **6,752** | 0.016287 | **0.012548** | -23.0% | 40.67% -> **41.67%** | 13 s |
| 45 | 9,495 | 0.016125 | 0.012496 | -22.5% | 40.12% -> 41.32% | 17 s |

Champion NNUE on the same rows: 0.008978. 32 is the knee: 45 buys -0.4% val for +41% weights and no agreement gain. Training the HCE like the NNUE is real (-23% of its own fit) but the plateau is 1.39x the NNUE's val loss at ~41% move agreement: representation-limited, not training-limited. Fresh-lr restart (D989/D990, `lab hce-train --init` from the d985 best weights, lr reset to 2.0): 45b and 32b both saturated at best epoch 0 with zero gain (0.012496 and 0.012548 unchanged, agreement unchanged) — saturation was real, not the schedule. All logs/weights: `logs/hce/d985_hcetrain_b{16,32,45}.log` + `_weights.json`, restarts `logs/hce/d989_hcetrain_b{45,32}_lr2.log`.


**D991/D992 (2026-11-16 23:3x): the plateau above was measured WITHOUT gataxx's pair-pattern residual.** gataxx = base terms + 80 pair placements of two 2x2 blocks (each exact 8-square content weighted, D4-tied, 66,420 classes) + 50x50 mobility table. Added as `lab hce-train --pairs 1` (lab/src/pairs.rs), init = trained 32b weights + zero residual (exact step-0 parity, val 0.012548). Smoke, 2M rows: 0.011285 after 4 epochs (-10.1%). Full 37.7M-row run: logs/hce/d991_hcetrain_b32_pairs.log (result below when done).

**D993/D994 results (full data, val_v9_clean 564,834 rows):** NNUE 0.008978 | ours 32b + pair tables + mobility, trained like the NNUE **0.010904** (agreement 48.72%) | gataxx 2.0.0 as shipped 0.012431 (one fitted scale k 0.407; 0.019767 raw) | ours 45b 0.012496 / 32b 0.012548 (41.3 / 41.7%) | gataxx base terms only 0.017180. Logs: logs/hce/d991_hcetrain_b32_pairs.log, logs/hce/d993_gataxx_val.txt.

**D995 Zataxx on our val:** 0.011393 (best scale k 0.573, raw 0.013560), best-move agreement 50.68%: between our graft (0.010904, 48.72%) and gataxx (0.012431). Zataxx = 36 2x2 windows -> 256 SCReLU -> 1, full forward 722 ns scalar. Log logs/hce/d995_zataxx_val.txt.

**D998 Zataxx trained like the NNUE:** 0.010283 val (quantised 0.010294), best-move agreement 49.27% (shipped 50.68%). Beats HCE+pairs on loss, not on move ranking; overfits after a few epochs (747k weights). Net: logs/hce/d996_zattrain_net.bin (Zataxx format).

**D1006 Zataxx + D4 augmentation, trained like the NNUE:** val 0.009925 (quantised 0.009928), best-move agreement 53.43%: best non-NNUE eval so far (NNUE 0.008978). Without D4: 0.010283 / 49.27% (overfit).

## 2026-11-17: can the HCE replace the policy head or the SmallNet? (D1051)
Measured on main 0120fd12 (lab node-cost / bound-cmp, logs/hce/d1051_*): the fused policy ordering costs ~0 ns per node
and saves 29% of the nodes; the SmallNet halves the node time (11.6 -> 6.0 us) by skipping big-net evals. As a lower
bound on the champion's child eval at 2% violations (940k children, held out): SmallNet 218 cp slack, fast HCE 260,
hand eval 495, linear HCE hybrid 215, SmallNet + HCE 199 (-9%). An HCE eval costs 437-711 ns vs ~88 ns for the SmallNet
child path, so the HCE replaces neither; the open lever is the big net's cost and the SmallNet-leader research (-30%).
Not checked: an incremental HCE (pair deltas per move), deeper tree nodes than val children, Elo. CORRECTION (D1052): this tested the standalone HCE at full cost; tables fit for the job (residual on the SmallNet, quantile loss, incremental deltas) are NOT tested yet; the NO is withdrawn.

## 2026-11-17: what does the champion NNUE lack? (D1053 / D1059)
Frozen champion + a linear correction from one HCE group at a time, trained on 13.75M rows of games the champion never
saw, scored on 1.53M held-out rows of other games (same label distribution), each with a calibration baseline (bias +
learned scale per bucket). Val loss: champion 0.012118, calibration 0.011658; beyond calibration: gataxx terms -0.45%,
Funes regions -0.52%, raw PST -0.53%, ours with dynamic PST and soft weaknesses -0.81%, PAIR PATTERNS -2.65%, all -3.05%.
The net misses multi-square pattern relations (mostly edges), not counts or per-stone weakness. Next: pattern inputs Split of the weakness group (D1061): dynamic PST -0.65%, soft weaknesses -0.24%, scalars -0.31%.
for the NNUE (NNUE_EFFICIENCY_PLAN step 2). Linear lower bound only; not gated.
