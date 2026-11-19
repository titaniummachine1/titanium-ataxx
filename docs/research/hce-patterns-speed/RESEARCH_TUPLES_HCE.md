# Tuple / HCE study: which pattern shapes reproduce the champion NNUE, per ns (D454, user order 2026-11-06)

User: "analyse the weights, ablate the network and finish the research to try an HCE that achieves exactly the same or very close
results; then you will understand architectures better than the standard NNUE / Zataxx 2x2 tuple. What about irregular tuples:
5x1 rotated 4 times per square, diagonal tuples? Use papers."

## 1. Literature (searched 2026-11-06)
- **Jaskowski, "Systematic N-tuple Networks for Position Evaluation: Exceeding 90% in the Othello League"** (ICGA J. 37(2) 2014,
  arXiv 1406.1509): many SHORT, STRAIGHT, systematically placed tuples learn better than a few long random "snake" tuples; even
  2-tuples (288 weights) reached ~96% in the Othello League. Lesson: coverage and systematic placement beat size.
- **Buro, Logistello** (webdocs.cs.ualberta.ca/~mburo/log.html): game-STAGE-dependent tables for horizontal/vertical lines of 8,
  diagonals of 4-8, a 3x3 corner, a 2x5 corner, edge+X-square, parity; >1M parameters fitted by linear regression. Lesson: lines
  and diagonals in all directions + phase buckets + regression on many positions.
- **Rapfi (Gomoku), arXiv 2503.13178**: a compact network on LINE PATTERNS in 4 directions through every square, distilled from a CNN
  into a pattern-indexed CODEBOOK (pattern -> feature vector), summed and fed to a tiny head, with incremental updates; reaches the
  accuracy of much larger ResNets at orders of magnitude less compute. Closest published analogue of the user's "5x1 rotated 4 times
  per square" idea and of "distil the NNUE into something cheaper".
- 2048 n-tuple work (arXiv 1604.05085): evaluation cost of n-tuple nets is dominated by the NUMBER of memory accesses (tuples), not
  their size. Chinese-chess comparison training (arXiv 1801.07411): n-tuples as a linear sum of feature weights.
- Our own history (NET_ANALYSIS, tuple.rs): S57/TUP7 = 36 x 2x2 singles + 8-cell adjacent-block pairs in 4 directions + env tallies +
  mobility, fitted to SEARCH labels; the NNUE beat it (val 0.0172 vs 0.0217). Zataxx: 36 x 2x2 windows -> 256 SCReLU -> 1.
  What was never done: fit shape families to the CHAMPION NNUE's eval (clean labels) and compare them by sibling ranking and ns.

## 2. Method (tool crates/titanium-cli/src/bin/tuple_distill.rs)
Linear pattern evaluators (Logistello-style HCE), each family fitted by SGD in probability space to the champion NNUE's static eval
(distillation: labels = NNUE eval of every child of game positions, data/distil_children_400k.tdat, + game positions labelled on the
fly), 8 phase buckets by empties, position-specific tables. Cell digit = empty / own (side to move) / enemy / off-board.
Families: 2x2 windows (Zataxx), ray2 = the user's 8 overlapping 2x1 pairs per square whose union is the 3x3 (D460), ray3/ray4 = the user's 8 directional rays per square (3 or 4 cells starting at the square, rays off the board skipped, D457), line5 in 4 directions per square (Rapfi/user idea), diagonal line5 only, plus5, 3x3 (interior
centres), the NNUE's own input features used linearly (G1-G6 codes), and combinations.
Metrics on val_v9_clean parents (+ all children): corr and MAE vs the NNUE, sibling best-move agreement with the NNUE and regret by
NNUE judgment (the metric that exposed TSTG, D439), weights, lookups and measured ns per eval.

## 3. Results
- 2026-11-06 smoke test (PC, 20k child rows + 5k game rows, 1 epoch, 50 val parents; ONLY a pipeline check, numbers not meaningful):
  tool runs end to end. CAVEAT for all ns/eval numbers of this tool: they time an UNOPTIMISED reference implementation (per-cell
  digit lookup + index vector per eval; line5 = 196 tuples x 5 cells = 4,750 ns here). The fair cost proxy is the LOOKUP count:
  an optimised pattern eval costs ~1-3 ns per lookup when tables are cache-resident (incremental index updates, Rapfi-style).
- Full run: Modal CPU 16 vCPU (training/modal_tuple_distill.py), 6M NNUE-labelled children + 2M NNUE-labelled game rows,
  4 epochs, 2,000 val parents with all children; families 2x2, line5, orth5, diag5, plus5, 3x3, nnfeat, 2x2+line5, line5+3x3,
  2x2+line5+3x3, nnfeat+line5. DONE 2026-11-06 20:02 in 209 s (8,079,544 train rows = 6,182,980 children + 1,896,564 game
  rows; 1,955 val parents with all children). Raw log logs/tstg_modal/tuple_distill.log. Linear models, 8 phase buckets:

| family | weights | lookups | corr par | MAE par | corr kids | MAE kids | best-move agreement | regret cp/move |
|---|---|---|---|---|---|---|---|---|
| 2x2 (Zataxx windows) | 73,728 | 36 | 0.850 | 105.6 | 0.885 | 102.1 | 41.4% | 69.2 |
| line5 (4 dirs/square) | 1,605,632 | 196 | 0.842 | 114.6 | 0.852 | 119.0 | 34.5% | 89.9 |
| orth5 | 802,816 | 98 | 0.817 | 123.4 | 0.835 | 125.2 | 32.2% | 101.0 |
| diag5 | 802,816 | 98 | 0.778 | 139.6 | 0.802 | 138.0 | 28.6% | 110.9 |
| plus5 | 401,408 | 49 | 0.865 | 101.3 | 0.898 | 96.9 | 41.9% | 66.1 |
| 3x3 (interior centres) | 52,428,800 | 25 | 0.854 | 95.3 | 0.896 | 95.4 | 45.0% | 59.5 |
| nnfeat (champion inputs, linear) | 419,136 | 218 | 0.897 | 86.1 | 0.931 | 78.1 | 54.0% | 39.8 |
| 2x2+line5 | 1,679,360 | 232 | 0.879 | 96.4 | 0.904 | 93.9 | 43.6% | 63.6 |
| line5+3x3 | 54,034,432 | 221 | 0.896 | 85.4 | 0.918 | 84.3 | 48.8% | 55.4 |
| 2x2+line5+3x3 | 54,108,160 | 257 | 0.901 | 82.6 | 0.926 | 80.1 | 50.5% | 51.6 |
| nnfeat+line5 | 2,024,768 | 414 | 0.911 | 80.3 | 0.938 | 73.7 | 54.2% | 41.5 |
References: champion NNUE = 100% by definition (3,463 ns/eval on the same children on Modal); TSTG distilled C52C0AB4 51.6% / 41.0 cp.

Findings (D459):
1. NO LINEAR PATTERN EVALUATOR comes close to the champion: the best is 54% best-move agreement. Linear HCE cannot reproduce
   the NNUE; the champion's edge is the NONLINEAR part (hidden layers over the accumulator), as the probes said (D418/D424).
2. The champion's OWN inputs used linearly (nnfeat) = 54.0% / 39.8 cp, the same level as distilled TSTG (51.6% / 41.0).
   So TSTG's spatial mix + SCReLU products currently buy about nothing over a linear readout of the NNUE features: that is
   why TSTG loses at equal search, and it means TSTG needs a real nonlinear head, not more linear structure.
3. SHAPE: in Ataxx compact shapes beat long lines. Per lookup: 3x3 (25 lookups) 45.0% > plus5 (49) 41.9% ~ 2x2 (36) 41.4%
   >> line5 (196) 34.5% > orth5 32.2% > diag5 28.6%. Diagonal-only lines are worst. Reason: Ataxx captures and moves act on
   the 8-neighbourhood (radius 1) and jumps (radius 2); long lines mostly see far, irrelevant cells. Othello/Gomoku line
   tuples (Logistello, Rapfi) do NOT transfer directly: their rules act along lines, Ataxx rules act on neighbourhoods.
4. Adding line5 on top of compact shapes helps a little (3x3 45.0 -> line5+3x3 48.8 -> +2x2 50.5), still linear-capped.
CAVEATS: 4 SGD epochs; 3x3 has 52M weights for 8M rows (sparse, likely underfit/overfit mix, its numbers are a lower bound
of what the shape can do); one seed; 1,955 parents (+-~1.1 pp standard error on agreement); ns column = unoptimised reference.

- Run 2 (D460, launched 2026-11-06 ~20:08, Modal CPU 16 vCPU, same data/settings): ray2, ray3, ray4, ray2+2x2, ray2+3x3,
  ray3+2x2, ray3+line5, ray3+nnfeat. Log logs/tstg_modal/tuple_distill_run2.log. Question: how much of the 3x3's 45% does the
  pairwise version (ray2) keep at 16-entry tables, and do the user's rays beat line5 (they include the jump distance 2).
  DONE 20:13, 233 s, same data (logs/tstg_modal/tuple_distill_run2.log). Linear, 8 phase buckets:

| family | weights | lookups | corr par | MAE par | corr kids | MAE kids | best-move agreement | regret cp/move |
|---|---|---|---|---|---|---|---|---|
| ray2 (8 x 2x1 pairs/square) | 39,936 | 312 | 0.796 | 131.4 | 0.807 | 134.0 | 31.1% | 103.8 |
| ray3 (8 x 3x1 rays/square) | 122,880 | 240 | 0.818 | 123.6 | 0.836 | 125.1 | 34.3% | 93.8 |
| ray4 | 360,448 | 176 | 0.831 | 119.5 | 0.846 | 121.7 | 33.7% | 95.4 |
| ray2+2x2 | 113,664 | 348 | 0.842 | 109.5 | 0.877 | 105.9 | 39.9% | 72.4 |
| ray2+3x3 | 52,468,736 | 337 | 0.877 | 92.6 | 0.903 | 92.0 | 45.4% | 59.8 |
| ray3+2x2 | 196,608 | 276 | 0.860 | 102.9 | 0.892 | 98.9 | 43.5% | 63.4 |
| ray3+line5 | 1,728,512 | 436 | 0.834 | 117.2 | 0.846 | 121.7 | 33.6% | 93.7 |
| ray3+nnfeat | 542,016 | 458 | 0.897 | 86.3 | 0.930 | 78.7 | 53.7% | 41.3 |
Findings (D462): as LINEAR evaluators the rays are line-level (31-34%), below the compact windows (2x2 41.4%, 3x3 45.0%);
the pairwise factorisation of the 3x3 (ray2) keeps 31.1 of the 3x3's 45.0 -> interactions of 3+ cells carry a third of
the linear signal. Rays add almost nothing on top of 3x3 (45.4 vs 45.0) or nnfeat (53.7 vs 54.0). Caveat: lookups count
directed duplicates (ray2 312 = 156 unique pairs x 2), which is redundant for a linear model.
- Nonlinear step (D461, tool ray_nnue.rs): pattern inputs -> 256-wide accumulator (2 perspectives) -> the CHAMPION's head,
  init by regression onto the champion's pre-clip accumulator, then end-to-end fine-tune. Parity check of the head
  re-implementation: 0.42 cp mean |diff| vs the champion eval (500 val parents). PC smoke (49k rows, 2+2 epochs, 195
  parents; pipeline check only): ray2 graft 33.3% -> fine-tuned 39.0% (linear ray2 on 8M rows: 31.1%).

- ray_nnue run 2 (Modal, D464, log logs/tstg_modal/ray_nnue_run2.log; 6.18M children + 1.90M game rows, 1,955 val parents),
  interim per epoch (best-move agreement with the champion / regret / corr parents / corr kids / rows read per eval):
  - t8s (user's 8 masks, stones only): graft 50.9% / 51.6 cp / 0.925 / 0.943 | fine-tune ep0 58.2% / 36.9 cp / 0.956 / 0.965,
    375 rows per eval. Already above every linear model (best 54.0%) and distilled TSTG (51.6% / 41.0 cp).
    (preempted at 617 s and restarted; the rerun reproduced every number) ep1 59.8% / 34.7, ep2 60.8% / 34.2, ep3 61.2% / 33.9,
    FINAL ep4 61.4% / 33.8 cp, corr 0.962 / 0.969 (model /vol/runs/ray_nnue/ray_t8s.rnn, 23.0 MB).

- PC test of the user's 'only stones with legal moves scan' (D467/D471; 3.09M children + 0.95M game rows, 1+3 epochs,
  978 parents; log logs/tstg_verify/ray_nnue_d467_pc.log), final epoch:

| family | what immobile stones do | best-move agreement | regret cp | corr par / kids | rows/eval |
|---|---|---|---|---|---|
| t8s | scan their 8 masks like every stone | 59.7% | 37.0 | 0.942 / 0.966 | 379 |
| t8m | read ONE sealed-stone row (square x colour) | 61.5% | 29.3 | 0.948 / 0.968 | 335 |
| t8mx | contribute nothing | 56.2% | 37.2 | 0.893 / 0.951 | 325 |
  => mobile-only scanning is better AND 12% cheaper, as long as a sealed stone keeps one row; dropping it loses capacity.

- PC, same setup (D472/D473; log logs/tstg_verify/ray_nnue_d469_pc.log), final epoch:

| family | per stone | best-move agreement | regret cp | corr par / kids | rows/eval |
|---|---|---|---|---|---|
| n8s (user D469) | 8 single-neighbour tuples (pairwise 3x3) | 51.8% | 52.5 | 0.899 / 0.947 | 379 |
| k3s | ONE tuple = the full 3x3 (2 x 3^8 rows per square) | 64.7% | 29.0 | 0.965 / 0.972 | 58 |
  k3s graft alone (before any fine-tune) was already 63.4% / 31.6 cp. Joint patterns >> pairwise; k3s = best and cheapest.

- ENGINE + GATE (D474-D482): pattern path in nnue.rs (--nnue X.rnn; int16 = float within 0.6 cp; shift-based 3x3 index +
  blocked row sums): k3s eval 2.58 us vs champion 3.65-3.87 us (1.45x), bench 1.72x nps at the start position. Full-data k3s
  (Modal, 45.1M rows) converged at 69.0% / 24.8 cp (not downloaded: truncated over a hotspot). GATE of the PC-screen k3s
  7FA02CBA vs main at 100 ms: 100 g 28-68-4 = -147 +-72 (SPRT below bound); PLAYED nps only +3.8%; value lost/move 25.2
  vs 15.9 cp. Conclusion: a 65-69% distilled student is far too weak, and the eval is a small part of node time in games,
  so faster eval alone does not pay. k3c (shared table) = 59.6%, no faster (1.1-1.3x).

- Modal ray_nnue run 2 FINAL (D483; 6.18M children + 1.90M game rows, 2+5 epochs, 1,955 parents):

| family | best-move agreement | regret cp | rows/eval |
|---|---|---|---|
| t8 | 63.2% | 31.3 | 564 |
| t8s | 61.4% | 33.8 | 375 |
| 3x3 (interior windows, not anchored) | 64.1% | 30.1 | 50 |
| 2x2 (Zataxx windows) | 56.7% | 46.0 | 72 |
| ray2 | 53.1% | 49.4 | 312 |
| ray3 | 52.8% | 47.2 | 240 |

## 4. Network ablation (D458): which input groups the champion's decisions depend on
Champion with one input group switched off at inference (no retraining), measured against the full champion on 1,500 val parents
and all their children: sibling best-move agreement / regret by the champion's judgment.
| group removed | corr | MAE cp | best-move agreement | regret cp/move |
|---|---|---|---|---|
| G1 square state | 0.811 | 229 | 42.8% | 55.6 |
| G3 stone ring | 0.830 | 134 | 49.9% | 43.0 |
| G5 2-ply exchange | 0.954 | 74 | 70.6% | 10.7 |
| G2 anchor x stone | 0.981 | 44 | 75.1% | 10.6 |
| G6 global | 0.989 | 28 | 87.3% | 4.1 |
G1 + G3 hold the core; with retraining the loss of G2 was +5% val error (D430). Weight anatomy (D452): no dead rows, full rank,
both perspectives informative. These are the reference points the tuple families must approach.

## 5. State at 20:00 (for after compaction)
- RUNNING on Modal: tuple study run 1 (training/modal_tuple_distill.py; families 2x2, line5, orth5, diag5, plus5, 3x3, nnfeat,
  2x2+line5, line5+3x3, 2x2+line5+3x3, nnfeat+line5; local client log logs/tstg_modal/tuple_distill.log). When done: copy the
  table into sec. 3, then run 2 with TUPLE_FAMILIES env for ray3, ray4, ray3+2x2, ray3+line5, ray3+nnfeat (the launcher does
  not pass env yet: set it in the Modal function or add a CLI arg).
- RUNNING on the PC: i8 gate 1000 ms leg (logs/gates/d450_i8_1000ms.log, 120 g; marker logs/gates/d450_DONE). i8 at 100 ms pooled
  1,318 g = 0 Elo (D456).
- Next after the study: for the best family per lookup, (a) a nonlinear version (Rapfi-style codebook: pattern -> small vector,
  summed, tiny head) distilled from the champion; (b) judge by sibling agreement vs the champion (the champion itself = 100%,
  TSTG = 51.6%), then ns of an optimised implementation, then a gate.
