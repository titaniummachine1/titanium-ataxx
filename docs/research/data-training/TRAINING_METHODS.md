# Training methods: avoid local minima, spend compute efficiently

This is the training-methodology research, separate from the data recipe.
Sources: our LEDGER/HANDOFFS failure history, the Stockfish/Lc0 recipe, and
standard NNUE/RL practice.

## Failure modes already observed in Titanium

- `val loss` improves while playing strength collapses: echo chamber at 12M
  (flywheel 8-90-2), val-lies x3 on distillation retrains.
- Self-distillation fossilizes: self-teacher students reach parity, then stop.
- Capacity wall: 29k shared pairs saturate; more rows alone do not break it.
- Outcome-only training is a 90-ply credit-assignment lottery; noisy.
- Frozen base + new heads memorizes noise: `pa`/`capconv` retrains collapsed.
- Higher-dose teacher data can hurt: 3% +2, 6% -38, 10% -24.
- Agreement with a teacher is not strength: retrains fit align20k better and
  still do not win.

## Avoiding local minima

1. Curriculum, not one blend:
   - Stage 1: broad base data (own selfplay / low-depth search).
   - Stage 2: fine-tune on high-quality teacher data (KataGo/Lc0-equivalent).
   - Never train a fresh net on teacher-only data; never stop at stage 1.
2. Data diversity first:
   - Dedup exact positions, balance phases, add opening/endgame coverage.
   - Keep multiple sources; avoid a single self-teacher.
   - Prefer real outcomes + search scores over pure outcome or pure score.
3. Learning-rate exploration:
   - Cyclic LR / warm restarts (`--cyclic-T0`, `--cyclic-mult`,
     `--cyclic-lr-min` already in `train_stream.py`).
   - Multiple seeds; keep the best by **gate**, never by val loss.
4. Hard-example / regret training:
   - Mine positions where shallow search disagrees with deep search, or where
     engine eval disagrees with the game outcome.
   - Upweight regret rows; deep-relabel them instead of relabeling everything.
5. SPSA / evolution strategies on eval scalars:
   - Owner's Elo-loss idea: optimize a small number of scalar knobs with
     antithetic paired-book games, not full retrains.
   - Cheap, parallel, and directly optimizes game result.
6. Correction history / relational features:
   - Per-context eval-error signals (search vs static) were the recommended
     relational lever; evaluate under fixed nodes before believing them.

## Spending compute efficiently

- Sweep small first: fixed subset, LR / batch / optimizer / lambda sweep, then
  scale only the winners.
- Data quality > quantity: filter, dedup, balance; one clean 10M set beats
  30M noisy rows.
- Prioritized sampling: easy/duplicate positions downweighted; regret and
  phase-rare positions upweighted.
- Two-stage fine-tune is cheaper than training teacher-only from scratch.
- Gate by games, not val loss: fixed-node 5k leg for decisions, time legs for
  speed. No giant grids, no hoping.
- Checkpoint and stream: resumable training, keep every epoch candidate.
- Quantization-aware training (`q2`/W4) from the start, not after.

## Concrete Titanium plan

1. Build stage1 base export from own cache + Titanium 5k + outcome.
2. Train stage1 from the current net; gate candidates.
3. Build stage2 teacher export from KataGo Ataxx + deep labels + outcome;
   upweight regret rows.
4. Fine-tune stage1 on stage2 with cyclic LR, multiple seeds, lambda/outcome
   mix; gate every candidate.
5. SPSA on the small eval scalars after the net stops improving by gate.
