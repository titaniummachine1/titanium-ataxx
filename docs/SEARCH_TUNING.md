# Search tuning notes (living list, newest findings first)

What we tune in Titanium's alpha-beta search, what we learned, and what to try next. Every change = own branch + gate
vs the staging tip (BEHAVIOR 24); grids run on Docker cloud sandboxes (CPU) per the cloud policy.

## Disagreement-driven horizon extension (user design, 2026-10-27; proposed, not measured)

The idea is to continue ordinary legal moves past the nominal depth horizon when a predictor expects the static score to disagree with a deeper search. This is not capture-only quiescence. Titanium currently has no qsearch: `negamax` returns `static_eval` at depth zero (`crates/titanium/src/search.rs`). Ataxx's broad branching makes an explicit extension/node budget important; a high predicted gap should earn more search budget, while a calibrated low remaining-gap estimate should stop it.

**Signal check:** a reproducible random sample of 100,000 eligible roots against the stale 5k labels had static/search Pearson r=0.862, MAE=116 cp, RMSE=175 cp; median absolute gap 80 cp and 90th percentile 249 cp. This confirms a learnable-looking target distribution, not that a predictor will improve search. Labels are from the prior search build.

**Training target:** for a position at the horizon, record the current static eval and deeper search scores at multiple fixed continuation budgets. Predict signed residuals (`search - static`) plus a calibrated magnitude/upper-quantile or per-budget residual. A single root `|static - 5k score|` predicts that a gap exists, but does not tell how many extra plies/nodes resolve it. Root labels do not substitute for horizon-node labels; collect internal-node snapshots from normal searches and split by phase, depth, mobility, and window/context. The current relabel prefix is sorted by disagreement, so it must not be used as an unbiased train/validation split.

**Model comparison and where to attach it:**
- **Futility first:** add an auxiliary disagreement head to the existing futility SmallNet trunk. In this code path the engine already makes the child accumulator and computes the small net's hidden activation before deciding whether to pay for the big NNUE child eval. Expose that activation and add the output head there: no second board transform, and only a small extra head at exactly the futility decision. This is the fastest likely route for futility, not a guarantee until measured. The small net's current output is a low-quantile lower bound on the big NNUE's *child static eval*; it must be retargeted/augmented for search residuals. Train signed child residual (`deeper-search child score - big-NNUE child static score`) and a calibrated lower-tail residual (or directly a lower quantile of deeper-search child score) at budgets/depths matching the move's remaining search. Use the lower-tail signal to back off when static may overstate the child's advantage; a mean or absolute gap alone is not a safe pruning margin. Respect child-to-move score orientation. On held-out futility calls, score saved big-net evaluations against false prunes at a fixed error rate before measuring full-engine cost/strength.
- **Horizon continuation:** an auxiliary head on the main NNUE activations is the likely cheapest generic qsearch controller, because the big NNUE has already computed its trunk for the leaf static eval. Its shared feature transformer and bucketed L1/L2 trunk can emit signed residual and per-budget/remaining-gap outputs with only a few hundred extra multiply-adds (estimate; benchmark required). Current trainer/exporter and TNN1/TNN2 formats carry only the static output, so auxiliary targets and sidecar/new-format support are needed. Freeze the trunk for the first fit, then test joint fine-tuning while protecting static-eval quality.
- **Standalone board-to-gap MLP:** clean independent baseline if either shared representation is not predictive enough. It repeats feature work; compare prediction quality and full inference cost, not parameter count alone.

The futility SmallNet accumulator is currently formed only in the futility-child path, not kept at every horizon node. This makes it attractive for the futility head but a poor first attachment for generic horizon continuation. Conversely, the main-NNUE head is available wherever the static evaluation runs, including the horizon.

**Engine targets:** prioritize the new disagreement-driven horizon continuation itself. Separately, residual confidence may inform futility/reduction margins; RFP is a smaller lever per D113. Keep the existing futility lower-bound head semantically separate. Eval choice (small vs big static network) needs a different target: small-vs-big evaluator error, not search-vs-static disagreement.

**Evaluation plan:** use random, disjoint horizon-node train/validation samples; report residual MAE, threshold calibration, and score change by extra-search budget. Then compare baseline against auxiliary-head, standalone, and smallnet-branch continuations at equal total nodes and at equal time. Require no loss of ordinary static-eval quality and measure qsearch expansion cost/stop behavior before any game gate. No model has been trained or integrated yet; active relabel remains the current long CPU job.

## Ataxx facts that shape the search (user + measurements, 2026-10-24)
- Material swings every move: a move flips 0-8 stones (typically 3-6). Losing stones is normal, not decisive as in
  chess -> the static eval jumps ply to ply; every static-eval pruning margin (futility, RFP, NMP eval term) must be
  calibrated to that swing, not copied from chess or from Moonbird's units.
- No quiet positions: almost every move can flip pieces, so a capture-only quiescence move class is not selective; current Titanium has no qsearch and returns static eval at depth zero. This does not rule out disagreement-driven horizon continuation over ordinary legal moves (above); it has different selection and training targets.
- Ordering by stones flipped is the strongest cheap signal (Moonbird: pieceDelta + 2 x history - 2.2 for jumps).
- NULL MOVE MATTERS MOST IN THE OPENING (user, from our research so far): high mobility, almost every move playable,
  so a pass is a reliable "still fine?" test; as the board fills, forced passes / zugzwang make it unsafe (our guard
  already disables NMP above 54% fill and below 11 clone targets).

## Done / measured
| lever | result | state |
|---|---|---|
| Moonbird LMR + futility reduction by child static eval (a3c9b74) | +48 @100ms, +96 @50ms | main |
| Moonbird butterfly history [stm][from][to], +d^2 / -d^2/24 (b0c4329) | +159 @100ms, +169 @50ms | main |
| Moonbird null move R = 4 + depth/2, any depth, TT miss/lower bound (nmpmb-b0) | 5k +41, 100ms +15 (3200g PASS), 50ms +2 flat | not staged |
| Root PVS | ~0 over 800g | parked |
| NMP grid (sub, div, min depth), 12 settings x 400g @100ms vs staging | all 12 within -10..+21 Elo, none conclusive -> NMP shape alone is a small lever | done (LEDGER NMPGRID1) |
| SPSA, 13 knobs at once (futility, RFP, LMR, NMP shape+guards, ordering), 6000 pairs @50 ms (training/spsa.py, exp/search-spsa) | Stockfish/fishtest method; replaces 1-D grids | running |

## Next candidates (ranked by expected Elo)
1. Futility / RFP margins: MB_FUT_DIV 57 and RFP_MARGINS [257,347,478,774] were copied or hand-set; grid them for the
   big-swing eval (fire on almost every node).
2. History: gravity update (h += bonus - h*|bonus|/max), continuation history (previous move's to-square x move),
   capture-count history (by stones flipped), history-driven LMR (reduce good-history moves less).
3. Phase-aware null move: reduction (or on/off) as a function of fill / empties and mobility, strongest in the
   opening; tune the 54% fill and 11-clone-target guards; Stockfish-style eval-margin term ((eval - beta) / k).
4. Wide-teacher search tuning (user design): teacher = no lossy pruning, same depth as the student; fitness = regret
   vs teacher + node cost (honest node counting incl. pruned-move child evals), Pareto front, then game gates.
5. Aspiration windows (Moonbird 8 units x1.9) - earlier attempt hurt at fixed nodes on an older search; retest.
