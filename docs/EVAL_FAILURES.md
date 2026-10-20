# Why the eval retrains failed — measured record + hypotheses (2026-10-20)

All gates: 5k nodes, paired book, fresh lines per leg, uainewgame, seats 50/50, 0 forfeits.
Baseline = main `ea0bdf5` + champ `tupS55tree.tup` (F66A00FE).

| Arm | What it changed | Games | Elo | Verdict |
|---|---|---|---|---|
| C0 | full retrain, 5 ep, all families (heads zeroed) | 2008 | +6 | parity |
| A0 | full retrain, 6 clean ep (heads zeroed) | 1000 | +6 | parity |
| B0 | outcome-weight 0.10 | 1000 | -15 | parked |
| C0g | outcome 0.50 + 3% top-wsum golden dose | 1000 | +2 | parity |
| D0 | outcome 0.25 + 6% golden dose | 1000 | -38 | rejected (user) |
| E0 | outcome 0.25 + 10% golden dose | 1000 | -24 | rejected (user) |
| C1 | q2 head active | 104 | -30 | parked |
| C2 | q2+capconv+pa active (fixed loader) | 104 | -71 | rejected (user) |
| Wbucket1 | champ + additive piece-count correction (TUP8) | 1000 | -86 | rejected (candidate) |

## Measured explanations

1. **Teacher fit != strength (agreement lies).** Retrained nets fit the deep teacher better on
   align20k: mean|d| 110.7 (A) / 111.4 (B) / 116.2 (C ep1) vs champ 124.4, yet none beat champ by
   more than noise; correlation actually fell (0.846-0.850 vs 0.860). The 2026 self-play
   engineering paper (arXiv 2609.37447) reports the same: agreement controllers improved policy
   agreement but produced worse learning.
2. **The 5k label is a weak teacher.** Outcome audit: capconv/pa fit the 5k teacher (pa +0.2935
   R2 vs residual) but carry ~zero real-outcome signal (pa +0.0007); extra heads trained on that
   target are noise that hurts (C1 -30, C2 -71).
3. **The deep teacher is not converged.** bestmove stability: identical at fixed budget
   (150/150, deterministic) but depth changes the move: **200k vs 1M 72.7%**, **1M vs 2M 75%**,
   **2M vs 4M 77.5%**, **all three budgets agree only 62.5%** (40 positions, 1M/2M/4M).
   Child re-search favors the 1M move's child only 57.6% of the time (median +28 cp).
   A target that keeps changing with depth cannot be trusted as a policy label; the Quoridor
   doctrine demands N=3 unanimity for search labels for exactly this reason, and only ~62% of
   positions qualify even across 1M/2M/4M (deterministic repeats are identical, so unanimity
   must be across BUDGETS, not repeats).
4. **Value-only targets cannot move decisions directly.** Cloning is the only move that changes
   piece count; a per-count additive correction therefore acts as a clone/jump bias. Fitting the
   teacher's per-phase residual and adding it scored -86 Elo: the correction taught "clone more",
   not "play better". Bucket R2 for the teacher residual was 0.11 (k=4 optimal
   1-9/10-20/21-26/27-49), but only 0.0015 for the outcome residual.
5. **Outcome anchoring at high weight is noisy.** corr(deep teacher, outcome_cp) = +0.03 on the
   mined positions; the outcome target std (~967 cp) dwarfs the teacher residual std (~184 cp).
   At 0.5 weight the net spends capacity on a 90-ply credit-assignment lottery; at 0.1 it is
   too weak to matter (-15). The Apex 0.5/0.5 works because their value is win-probability and
   their rows are MCTS visits from a converged tree.
6. **Dose is not the lever.** 3% -> parity; 6% -> -38; 10% -> -24. Larger doses of a noisy
   deep-value target make it worse. `--freq-decay` at Adam scale is destructive (5%/step clamp
   wiped rare tables); disabled.

## What the winners do differently (docs/GOLDEN_DOSE.md)

- Apex: golden rows are CONVERGED deep searches (>=10k visits within 6 plies of a 1.3M-sim
  search), targets are the search's VISIT COUNTS (policy), 20x writes, 3-10% dose; value target
  0.5 game result + 0.5 search value; whole-tree data failed.
- Stockfish/Moonbird: bulk self-play labels at fixed depth/nodes (6k for Moonbird, 50M samples),
  opening diversity; best nets are first trained on self-play data, then retrained on deep
  (Lc0-like) data; "wrong binpacks" (static eval disagrees with search) are valuable.
- Quoridor doctrine (TEACHER_DERIVATION): exact verdicts override; outcomes are cheapest near
  the terminal (backward starts); own-search is dense initialization; N=3 unanimity; outcome
  k>=3 continuations; promotion only by games.

## Next experiments (ordered)

1. **Converged-teacher mining**: N=3 unanimity at 1M+ (fresh process/TT) -> keep only positions
   where the move is identical 3/3 AND stable vs 2M; then train value/decision targets on those.
   Cost measured: ~1.4 s/position/worker at 1M, so 3x1000 positions on 4 workers ~ 18 min.
2. **Policy-like signal without a policy head**: per-move sweep (each legal move gets its own
   deep search) on mined positions; keep only moves with a stable ordering; train a margin loss
   V(child_pv) > V(child_alt) + margin on the eval. This is the implementation sketch in
   docs/GOLDEN_DOSE.md 3d (needs a small trainer extension using the exact move applier).
3. **Backward-start outcome anchoring** (Quoridor): use real outcomes only close to terminal
   (short continuations), not as a global 0.5 blend on midgame rows.
4. **Balanced/opening data**: openings are 5% of cache12M; generate fresh self-play with noise
   from 2-ply starts and label them, instead of re-dosing the same midgame rows.
5. **Buckets, if revisited**: NNUE-style per-bucket OUTPUT weights (different feature weightings
   per phase), not an additive offset. Requires a TUP9-sized table multiplier; likely not worth
   it before the teacher problem is fixed.
6. **External data/engines**: autaxx ( -54 Elo at 5k) and KataAtaxx (needs GTP->UAI adapter) as
   sparring/data partners; Botzone dataset download for position diversity.
