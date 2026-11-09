# SPSA stage 3 — what the tuner changed in the search (report 2026-11-02 20:35)

Engine: `exp/tune-policy` `1cf931f` (policy ordering + all search knobs tunable), exe `logs/tunepol_1cf931f.exe`,
net `46FCB5A4`. Run: `logs/spsa_stage3pol.log` (cmd `logs/spsa_stage3pol.cmd.txt`, resume = same + `--resume`),
knobs `tune/knobs_stage3_policy.txt` (a = 30c, x10 steps), 50 pairs, 1000-node warm-up x100 then 5000 nodes, 8 threads.
Reproduce the table: `python docs/analysis/spsa_stage3_report.py`.

## Health (k=276 of 600, 277 iterations logged)
- plus-vs-minus score of the last 50 iterations: **0.4961** (last iteration 0.4450). 100 games per iteration, so the
  per-iteration noise is ~±0.05 and the mean is at the noise floor: the tuner has no measured signal of its own.
- Theta is therefore a *direction proposal*, not a proven gain. Only the gate decides (5k + 100 ms, laws 35/35a).

## Knob deltas (default -> theta_next at k=276)

| knob | def | theta | delta | where it acts (search.rs) | what the change does |
|---|---|---|---|---|---|
| pol_w | 100 | 52 | -48% | 1352 ordering score += logit*pol_w | policy head still dominates ordering but with ~half the weight; captures/killers/history matter more again |
| pg_t | 40 | 41 | +2% | 1460-1465 policy-gap LMR threshold (logit/10) | gap must exceed ~4.1 logits before any reduction |
| pg_d | 10 | 6 | -40% | 1464 slope (logit/10 per ply) | past the threshold, reduction accumulates ~1.7x faster |
| pg_max | 2 | 0 | -100% (AT BOUND) | 1464 cap | **policy-gap reduction is OFF**: the head orders, it does not shorten search (matches D343) |
| lmr_half | 47 | 44 | -6% | 1441-1445 | base half-life slightly shorter = a bit more reduction per reduced move |
| lmr_min | 6 | 13 | +117% | 1440 | LMR starts only from move index 13 instead of 6: the first 12 moves are searched near full depth |
| lmr_dis | 0 | 36 | new | 1441 `+dis*36>>7` | at disagreement 300 the half-life grows +84 -> LMR is nearly off where the net is unsure; confident nodes keep steep reduction |
| lmr_hinge | 0 | 20 | new | 1441 `+(dis-300)+*20>>7` | same, extra gentleness beyond the knee (dis > 300) |
| fut_div | 57 | 34 | -40% | 1474, 1495, 1508 | futility removes ~1.7x more plies per cp of shortfall = much more aggressive futility reduction |
| fut_ddis | 0 | 5 | new | 1474 | futility divisor rises with disagreement (less pruning in volatile nodes) |
| fut_dhinge | 0 | 6 | new | 1474 | same beyond the knee |
| dis_fut | 24 | 23 | -4% | 1506 vol_credit | child eval credited by ~child_dis*23/128 cp before the shortfall test (uncertain children pruned less) |
| fut_hinge | 0 | -5 | new | 1506 | slightly less credit beyond the knee |
| rfp1 | 257 | 205 | -20% | rfp_margin (1246) | depth-1 RFP margin smaller: RFP fires more easily at shallow depth |
| rfp2 | 347 | 462 | +33% | rfp_margin | depth-2 RFP margin much larger: RFP at depth 2 needs a 462 cp shortfall |
| rfp3 | 478 | 507 | +6% | rfp_margin | ~unchanged |
| rfp4 | 774 | 720 | -7% | rfp_margin | ~unchanged |
| pc_t | 140 | 95 | -32% | 1284-1289 ProbCut | verify window narrower -> ProbCut attempted more often/cheaper |
| pc_k | 32 | 40 | +25% | 1287 | ...but the window still widens with disagreement (+25%), so volatile nodes keep a wide margin |
| ph_lmr0..3 | 64 | 77/72/67/67 | +20/+12/+5/+5% | 1442 | LMR phase scaler: least reduction in the opening (x1.20), tapering to x1.05 in the endgame |
| ph_fut0..3 | 64 | 72/63/40/61 | +12/-2/-38/-5% | 1475 | futility phase scaler: mild in the opening, **much more aggressive at 11-20 empties (x0.63 -> effective divisor ~21)** |

Unchanged in this run (not in the knob file, stay at default): `rfp_s1..4` (32), `rfp_hinge`, `dis_knee` (300),
`pc_d` (6), `pc_r` (3), `pc_hinge`, `nmp_*`, `iir_d`, `corr_*`, `ord_*`.

## Reading (one paragraph)
The tuner is telling a consistent story: **spend the reductions on confident nodes and the cheap moves, not on the
uncertain ones.** LMR starts much later (move 13) but is steep for the tail, and is almost cancelled wherever the
disagreement head is high (lmr_dis/lmr_hinge). Futility is far more aggressive per cp (fut_div 34 vs 57) and most
aggressive in the 11-20-empty phase, again with a disagreement brake. ProbCut is attempted more often with a smaller
base margin and a stronger disagreement widening. RFP gets easier at depth 1, much harder at depth 2. The policy head
keeps only its ordering job (pol_w halved, policy-gap reduction switched off) — exactly what D343 measured: the
gap-based reduction variants were flat or negative.

## Caveats
- Most of the drift arrived during the 1000-node warm-up (D368) and the 5k phase so far has neither confirmed nor
  undone it (mean 0.4961). `lmr_min` was pinned at the old bound 12 and was widened to 24 (D371); it now sits at 13.
- Nothing here is proven. Gate theta vs the policy main at 5k nodes and 100 ms once the run reaches k~400-600
  (or earlier if the user asks), both legs required for a merge (law 35a).
