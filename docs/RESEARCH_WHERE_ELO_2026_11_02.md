# Where did the Elo come from? (user 2026-11-02, D324). Hub tool: `ti insight` now prints a per-phase table (titaniumtest games.rs `insight_phases`, built in target_phase, local commit pending)

Data: raw games in C:\gitProjects\logs\sbx_cal1\res (rwphase3_nocal_5k.games.jsonl, rwphase3_nocal_5k_b.log.games.jsonl, rwphase3_nocal_1000.log.games.jsonl,
rwphase5_nocal_5k.log.games.jsonl, rwphase6_nocal_5k.log.games.jsonl, sat16a_nocal_5k*.games.jsonl). Same exe BAF3782D (= main c86733b) both sides, baseline net F5FA8557,
std5k book, paired by seat. "ply" = half-moves from the start; buckets 0-19 / 20-39 / 40-59 / 60-79 / 80+. Value lost per move = mover's score + opponent's next score
(mates and |score| > 1500 skipped); the table shows baseline minus candidate (+ = the candidate loses less), paired z over pairs.
NOT checked: empties-based buckets, which exact moves, whether the late-game gain is stone-count or mobility, other books.

## What changed in the training (rwphase3 vs main's sat16a)
Same data (v9), same fine-tune recipe from sat16a/best.tnn, same architecture W256, same steps schedule; ONLY the target blend: result weight w in target = (1-w)*sigmoid(cp5k/400) + w*game_outcome
went from 0.25 to 0.55 for every phase, and the export no longer calibrates (calibration cost ~15 Elo, D318). Proven mates stay exact. The result is the REAL outcome of the game, which sees
past the horizon of the 5k-node label, so the net learns what a position turns into rather than only what a shallow search says.

## Result: the candidate wins LATE (rwphase3_nocal 54A6B02B, 5k nodes, 1,600 g, 800 pairs; value lost diff, cp per game this adds up to, eval level the candidate stands at)
| ply | value lost diff cp/move (z) | adds up to cp/game | candidate-view eval level cp (z) |
|---|---|---|---|
| 0-19 | -0.37 (-1.52) | -3.7 | -1.3 (-1.13) |
| 20-39 | -0.82 (-2.73) | -8.2 | -6.6 (-2.70) |
| 40-59 | +0.22 (+0.21) | +1.4 | -9.6 (-2.08) |
| 60-79 | +16.87 (+3.57) | +60.2 | +9.7 (+0.66) |
| 80+ | +26.89 (+2.80) | +84.5 | +20.7 (+0.62) |
Overall: score 0.5337 vs eval-expected 0.4960, value lost diff +1.26 cp/move z +2.01, tilt -6.6 cp z -2.04 (its evals sit LOWER than the baseline's in the opening/middle, i.e. it is more cautious),
blunders 6.2% vs 5.6% (slightly MORE blunders overall). 1000 ms leg (200 g): +47.2 +-43.8, LOS 98.4%; same shape (0-19 -9.9 cp/game, 60-79 +77.1, 80+ +85.1), value lost diff +3.71 z +2.29, speed 247 vs 273 knps.
Reading: the gain is in the second half of the game (ply 60+: about +145 cp per game saved at 5k, +160 at 1000 ms, z 2.8-3.6); the opening and early middlegame are slightly WORSE or equal. It is not
"better everywhere"; it is a better endgame/late-middlegame evaluator with a small opening price. sat16a_nocal (only calibration removed) has the same late shape (60-79 +62.7 cp/game) but weaker (+14.8 Elo).

## Result weights above 0.55 are WORSE (5k, 800 g each, vs F5FA8557)
| net | w | Elo | LOS | 60-79 diff | mid-game eval level |
|---|---|---|---|---|---|
| rwphase1 | 0.35/0.04/0.00 | -32.2 | 0.4% | - | - |
| rwphase2 | 0.4 | +3.7 (3,200 g) | 61% | - | - |
| rwphase3_nocal 54A6B02B | 0.55 | +23 pooled (1,600 g) | ~98-99% | +16.9 (z +3.6) | -6.6..-9.6 cp |
| rwphase5_nocal FB83D6ED | 0.70 | -20.0 +-23.3 | 4.6% | +8.2 (z +0.97) | -20..-55 cp |
| rwphase6_nocal CB3CED77 | 0.85 | -20.9 +-23.5 | 4.0% | -6.0 (z -0.97) | -28..-94 cp |
At 0.7 and 0.85 the net's evals drift progressively lower in the middlegame (tilt -27 and -42 cp, z -5 and -8) and it loses value from ply 0 on. The shifted eval scale probably also mis-sets the
cp-tuned search margins (RFP/futility/ProbCut margins were tuned on the 0.25 net): HYPOTHESIS, not tested. The optimum is between 0.4 and 0.7: next two runs 0.47 and 0.62 (D324 queue).
## Is the net saturated? (user question 2026-11-02, D326)
Reading of D232-D234 (old target, val MSE ~0.0176-0.0178 for W256 and W512, no gain from lr/width/inputs/probes) together with D324:
- SATURATED: fitting the 5k-label target on this data with this recipe: the val floor does not move with width (256 vs 512), lr, plateau reducers, or new simple inputs (probe R^2 ~0.5%).
- NOT SATURATED: play strength. Changing only the target (result weight 0.25 -> 0.55, no calibration) gave +23 Elo at 5k with the same W256 net and data, and the gain sits at ply 60+, which is where D232 found 42% of the loss (endgame bucket, 14% of rows). So the lever was the LABEL, not capacity or inputs.
- Caveats: val MSE of the new net is on its own target (0.052) and is not comparable; the peak weight is only bracketed (0.4 +4, 0.55 +23, 0.7 -20); capacity at the new target is UNTESTED (a W512 at 0.55 might separate from W256 now); label depth (200k-1M node labels for hard positions) is untested.
- Next levers by this reading: per-phase blend (rwphase9/10), weight peak (7/8), deeper or hard-position labels (ti mine), W512 distilled/downscaled at the new target, data scaling.
## rwphase9 (result weight 0.55 / 0.55 / 0.25 by empties <=6 / 7-12 / >=13) vs the NEW main 54A6B02B (D329, user blend idea confirmed)
5k nodes, same exe both sides, std5k book0 86000, Docker tit-cal1, 1,230 games when the gate was read: +629 -528 =73, 54.1%, Elo +28.6 +-17.9, LOS 99.9%, LLR +2.87 of 2.94 (the gate stopped at the sandbox ttl/queue; not at the SPRT bound).
ti insight (candidate = rwphase9): score 0.5411 vs eval-expected 0.5103; value lost per move baseline minus candidate +2.68 cp (z +3.26); mid-game tilt TO the candidate +17.0 cp (z +4.38; the 0.55-everywhere net was tilted to the baseline); blunders 3.2% vs 3.4%.
By phase (diff cp/move, cp per game): ply 0-19 +1.14 (+11.4 /game, z +4.3), 20-39 +0.14 (+1.4), 40-59 +0.73 (+9.9), 60-79 +10.4 (+77.9, z +2.6), 80+ +18.2 (+11.3). The opening cost of rwphase3 is gone (ply 0-19 flipped from -3.7 to +11.4 cp/game) and the late-game gain is kept.
Next in the queue: rwphase7 (0.47 uniform) and rwphase10 (0.55/0.40/0.25) 5k gates vs the same baseline (g1.sh), then a 1000 ms leg of rwphase9 (g2.sh, waits for the 5k gates), then promote if it holds (user order: merge winners).
## Would 32 buckets help? (user 2026-11-02)
Two different things share the word. (1) Finer RESULT-WEIGHT buckets: the trainer's blend is a 3-vector by empties (<=6 / 7-12 / >=13), the winning blend is a step 0.55 -> 0.25; a per-empties table (up to 32 steps, smooth ramp) is a small hub change (rw is [f32;3] in targets.rs / gpu.rs / trainer.rs / export.rs) and the natural next experiment once rwphase10 shows whether the middle bucket matters. (2) NNUE OUTPUT buckets (16 now, 8 phases x parity): more buckets = more head weights for the same per-eval cost (one bucket is evaluated); 	i train split doubles B -> 2B buckets with a val-mse parity check (no random init, LAWS C14), then fine-tune. Cost: head memory only; risk: thinner data per bucket (the endgame bucket is 14% of rows, D232). Both are queued; (1) first because it is cheaper to train and gate.
