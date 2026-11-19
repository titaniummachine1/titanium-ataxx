# LMR / search-tree statistics (exp/lmr2), 2026-11-01

Branch `exp/lmr2` (worktree `C:\gitProjects\lmr2-ataxx`) = main `c86733b` + an env-tunable extra-LMR block (`TITANIUM_TUNE`, all terms default 0,
bench nodes 68961 at depth 10 identical to the c86733b exe `logs/gate_speed4d_c86733b.exe` with net sat16a_cal) + build-time statistics (`LMR_STATS=1`,
separate target dir `target_stats`). Not committed to main, nothing gated yet.

## Measurement (stats build, selfplay 8 games, 100 ms/move, net sat16a_cal `F5FA8557`, raw output docs/analysis/lmr2_stats_selfplay_2026_11_01.txt)
Counters per (move-index class x PV x depth class); "pruned" = moves skipped because the reduction took new_depth to <= 0 (child still made and evaluated).
1. **Late moves are already almost all pruned**: move index 24+ at depth 1-2 (non-PV): 9,603,302 reached, 9,573,530 pruned (99.7%). Index 10-15: 96%. The
   tree is already very narrow; the cost of those moves is make() + the child eval inside the futility step, not subtree size.
2. **Re-search rate after a reduced, actually searched move is tiny**: about 1.4-2.3% at depth 5-8 (e.g. idx 10-15 d5-8: 888 re-searches of 140,611 reduced, of which
   101,663 pruned -> 38,948 searched -> 2.3%). Typical well-tuned LMR re-searches ~10%. Suggests reductions are conservative on the moves that ARE searched
   (candidate direction: more reduction via r_idx / r_dgrow / r_npv; the SPSA of 2026-10-24 (DKspsa1) found `lmr_half` already near its optimum on the old eval; this
   is a new view, not yet a result).
3. **User idea 2026-11-01 (captures-nothing flag) is supported by the data**: among NON-FIRST moves at >= 12 stones on the board, moves that flip no enemy stone:
   10,179,360 tried, 17,494 cutoffs (0.17%), 24,129 fail-highs (0.24%); moves that flip >= 1: 22,142,328 tried, 634,352 cutoffs (2.9%), 784,397 fail-highs (3.5%).
   A 17x lower cutoff rate. Below 12 stones (opening): no-capture 1,127,560 tried, 11,231 cutoffs (1.0%) vs capture 280,105 tried, 16,247 cutoffs (5.8%) - still 5.8x lower
   but the first move searched is a no-capture move far more often (85,357 vs 56,171), matching the user's observation that Titanium builds up in the opening.
   User refinement: apply the flag only from >= N stones (tune knob `nocap_min`, default 10). NOT yet gated.

## What was NOT analysed
8 games only (one net, 100 ms, self-play, so only the shapes above, no confidence intervals); no split by dis bucket yet; no per-phase split inside the >=12 group;
the reduced/pruned counters are per move, not per node; no time-per-move share (how much of the node cost the pruned moves' evals use is not measured).

## Tuning knobs added (TITANIUM_TUNE="name=value,...", units 1/16 ply, + = reduce more)
r_npv (null-window node), r_dgrow (x depth/8), r_idx (x min(move_idx,40)/8), r_dis (node disagreement/128), r_jump (jump move), r_hist (history score),
r_impr (improving), r_cdis (child disagreement), r_nocap (move flips nothing), r_flips (per flip, max 4), nocap_min (stones on board needed for r_nocap/r_flips).

## Idea 2026-11-01 (user, third message): tail pruning of no-capture moves
"If you are on the tail of the search, so the worst moves in the position, then you can be even more aggressive with pruning of those high-index moves if they don't capture anything.
It fights with Titanium a bit because Titanium sometimes makes very silent shifts (a silent pawn move), but that is once or twice a game."
Implemented as knob `nc_lmp` (exp/lmr2 source, not yet built or gated): no-capture moves at move index >= nc_lmp are SKIPPED before the child eval is paid (saves the make + eval cost, which
the stats show is where late moves spend their time), only at stones >= nocap_min, remaining depth <= nc_lmp_d, never when a mate score is in play. Risk named by the user: rare silent
shifts that are the best move; the gate (not the stats) decides. The reduction version is `r_nocap`; both are screened separately.

## Screen results (100 ms, 200 g each, candidate = exp/lmr2 exe 80ee39e0 with --tune, baseline = same exe on defaults; PC, 7 threads; raw logs/gates/lmr2_*_b0.log, progress.txt 'LMR2' lines)
- r_nocap=16,nocap_min=12: score 0.5000, Elo -0.0, CI [-47.7,+47.7], P(better) 50.0% (100 pairs) - flat. Knob verified active (bench depth 10: 68961 -> 65692 nodes).
- r_nocap=32,nocap_min=12: score 0.4900, Elo -6.9, CI [-54.6,+40.4], P 38.6% (100 pairs) - flat.
- r_idx=8 (extra reduction growing with move index): score 0.3118, Elo -137.5, CI [-187.7,-92.3], P(better) 0.0% (89 pairs / ~178 g, stopped at the chain end) - clearly WORSE: extra reduction by index hurts a lot. The low re-search rate (about 2%) therefore did NOT mean the reductions are too conservative.
- r_npv=8 and r_dgrow=4 were NOT run: the chain was stopped by the agent at 18:12 so the PC is free for the UBFM work the user ranked first (D270). nc_lmp not built. LMR tuning is parked (not rejected).
