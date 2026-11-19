# NNUE efficiency plan: higher ceiling, lower cost (D1055, user 2026-11-17)

User question: "how do you plan to optimise our NNUE while making its ceiling of what it can learn higher for less cost?"

## Where the champion's cost and capacity sit (812C2118, nnue.rs)
- Inputs: sparse rows (crate::nnfeat), ~208 active rows per eval (D446; Zataxx ~36). FT = i16, width 256, per perspective,
  updated INCREMENTALLY per move (only changed rows). Pairwise product -> L1 -> L2 -> output over both hidden layers,
  16 output buckets = per-bucket layer stacks (L1/L2/out selected, not summed), dis head, psqt per bucket.
- Cost: full forward 5.1 us in the lab (D1051), the engine ALSO full-refreshes every eval (corrected D1072); W128 2.8 us vs W256 5.1 us (D432);
  L1 skips ~86% zero activations. The node costs 6.0 us with the SmallNet saving ~half of the big-net evals (D1051).
- What is FREE at inference: more FT rows (only active rows are touched), more buckets / layer stacks (one is selected),
  everything in TRAINING (data, loss terms, teacher, optimiser). What COSTS: width, hidden sizes, active rows per eval,
  rows changed per move.

## Plan, ranked (one change per gate, B10; judged by per-node + timed gates, val loss is a guide only, D1026)
1. Free ceiling from training (no inference cost):
   - retrain on train_v11 (53.0M rows, +40% positions, D1049);
   - D4-consistency term in the loss (the champion differs by 40 cp mean / 858 cp max across symmetries, D1035);
   - distillation from a BIGGER TEACHER trained offline (W512 or more buckets on Modal): the teacher sets the ceiling,
     the W256 student keeps today's cost (C12: distil from the single strongest teacher; C14 init from the champion);
   - optimiser test MuonAdamW vs AdamW (D1037), best-move child targets (D1038), lambda targets (D758).
2. Free capacity from richer SPARSE inputs: add the HCE groups that D1053 shows the net lacks (pair-pattern classes,
   edge strips, weakness / safe-stone codes) as extra FT rows; their cost is their active count x the incremental update.
   Drop low-value existing rows (group ablation D873) so active rows stay the same or fall.
2b. (D1056) Everything incremental: a move changes 3.18 squares and touches 31.9 of 80 pair placements; window patterns (<= 4 per square) update ~10 per move; keep HCE / child-model state on the search stack like the NNUE accumulator.
3. Same quality, cheaper: (a) fewer active rows via window patterns (2x2 / 4x2 classes carry more per row; the SmallNet
   with 2x2 windows was 25% tighter, D846); (b) a narrow net on HCE-feature inputs (~1 us, D1054 cost/quality curve);
   (c) i8 FT (lossless, +12% game nps, 0 Elo at 100 ms, D440/D465 - revisit with the new net); (d) more output buckets.
4. Fewer helper nets (D1051): one child model = SmallNet leader that already reads the policy logit (w2+pol), later
   also the ordering logit; HCE tables as residual inputs where they win (D1052 T1-T4); one hub command refits every
   dependent (policy, SmallNet, dis head) after a big-net change.

## First measurements owed
- Engine-path cost split of one big-net eval (incremental FT update per move, L1, L2, output) and big-net evals per node.
- D1053 residual groups (running) -> the input candidates for step 2.
- D1054 curve (val vs ns for HCE-input nets with H = 16/32/64).

## Not checked yet
Elo of any of the above; the Botzone CPU (one weak core) cost of each change (B13: measure single-thread on the PC).
