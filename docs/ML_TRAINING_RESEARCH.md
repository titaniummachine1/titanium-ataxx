# ML training research: autonomous basin search, optimizers, HPO

Purpose: stop hand-picking AdamW and hoping. Make training search the
optimizer/hyperparameter space, prefer flat/deep basins, and spend compute on
promising trials with early stopping. This is a research summary, not a claim
that any single optimizer is universally best.

## 1. Loss landscape and "deep basins"

### Sharpness-Aware Minimization (SAM)
- Paper: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
  Generalization", arXiv:2010.01412.
- Idea: minimize `max_{||eps||<=rho} L(w+eps)`, not just `L(w)`. The update
  takes a gradient ascent step to a perturbed point, then a descent step.
- Why: flatter neighborhoods generalize better and are less sensitive to
  parameter noise. SAM natively helps with label noise.
- Cost: roughly 2x forward/backward per step.

### ASAM
- Paper: Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for
  Scale-Invariant Learning", arXiv:2102.11600.
- Fixes SAM's sensitivity to parameter rescaling; better when different
  parameter groups have very different scales (our embeddings vs bias vs
  scalar tables).
- Use ASAM when applying SAM to tuple-table embeddings with heterogeneous
  scales.

### SWA / SWAD
- SWA: Izmailov et al., "Averaging Weights Leads to Wider Optima and Better
  Generalization", arXiv:1803.05407.
- Keep a running average of weights from a cyclic/high constant LR phase.
  The average lands nearer the center of a flat region; no extra inference
  cost.
- SWAD: "SWAD: Domain Generalization by Seeking Flat Minima", NeurIPS 2021.
  Dense, overfit-aware weight sampling; often flatter than plain SWA.
- Practical: average checkpoints from the last 25-50% of training, or from
  several cyclic restarts; gate the averaged net, not the raw last step.

### Lookahead
- Paper: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back",
  arXiv:1907.08610.
- Maintains fast weights updated by any inner optimizer and slow weights
  interpolated every k steps. Reduces variance, stabilizes training, and is
  less sensitive to inner-optimizer hyperparameters.
- Cost: negligible; one extra parameter copy and an interpolation.
- Useful on top of AdamW/NAdam/RAdam/Lion, and combined with SAM.

### Cyclic LR / warm restarts
- SGDR: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm
  Restarts", ICLR 2017.
- Restarts let the optimizer escape a basin and sample another; SWA then
  averages across basins. Already partly supported by `train_stream.py`
  (`--cyclic-T0`, `--cyclic-mult`, `--cyclic-lr-min`).

## 2. Optimizers worth testing

Do not assume AdamW. Benchmark on a fixed subset, then gate.

- **AdamW**: strong default; best downstream in a 2025 small-LLM comparison
  (arXiv:2507.08472), but not always lowest validation loss.
- **NAdam / RAdam**: Adam variants; cheap to test; NAdam adds Nesterov
  momentum, RAdam rectifies early adaptive variance.
- **Lion**: Chen et al., "Symbolic Discovery of Optimization Algorithms"
  (Google). Sign-momentum; ~half optimizer memory; often faster; suitable LR
  is typically 3-10x smaller than AdamW, and weight decay 3-10x larger.
- **Sophia**: Liu et al., "Sophia: A Scalable Stochastic Second-Order
  Optimizer for Language Model Pre-training", arXiv:2305.14342. Hessian-based
  preconditioner; can lower validation loss, but more complex.
- **AdEMAMix**: Pagliardini et al., "AdEMAMix: Better, Faster, Older",
  arXiv:2409.03137. Mixes a fast and a very slow EMA; can reach lower minima
  and slows forgetting. Strong candidate for long fine-tunes.
- **Schedule-free AdamW/SGD**: Defazio et al.; removes explicit LR schedule;
  empirically strong; theory for nonconvex in arXiv:2411.07061.
- **Muon**: Jordan et al.; orthogonalizes momentum with Newton-Schulz.
  PyTorch `torch.optim.Muon` exists. Muon is for **2D hidden-layer matrices**;
  embeddings, biases, and scalar parameters should stay on AdamW. ACE/Apex
  used MuonAdamW. Our tuple table is mostly embeddings, so Muon applies only to
  any dense matrix layers, not the whole table.

No optimizer dominates everywhere: Lion can be faster but weaker downstream;
Sophia can lower validation loss; AdamW is a robust baseline; AdEMAMix can help
long runs. The only honest answer is a controlled sweep + game gate.

## 3. Hyperparameter search

### Optuna / TPE
- Optuna: Akiba et al., KDD 2019; TPE: Bergstra et al., NeurIPS 2011.
- TPE models `P(x|good)` and `P(x|bad)` and proposes points maximizing the
  ratio. Optuna adds pruning (stop unpromising trials) and parallel workers.
- Use for: optimizer choice, LR, weight decay, outcome/lambda, SAM rho,
  Lookahead k/alpha, SWA start, cyclic schedule, batch size.

### ASHA / Hyperband / BOHB
- ASHA: Li et al., "A System for Massively Parallel Hyperparameter Tuning",
  arXiv:1810.05934. Asynchronous successive halving; aggressive early
  stopping; scales linearly with workers.
- Hyperband: Li et al., JMLR 2018; runs multiple successive-halving brackets.
- BOHB: Falkner et al., "BOHB: Robust and Efficient Hyperparameter
  Optimization at Scale", arXiv:1807.01774; Bayesian optimization + Hyperband.
- Use ASHA/PBT first on 4 cores; BOHB/Optuna-TPE when trials are expensive and
  sequential.

### Population Based Training (PBT)
- Jaderberg et al., "Population Based Training of Neural Networks",
  arXiv:1711.09846.
- Train a population; periodically copy weights from the best members and
  mutate hyperparameters (LR, weight decay, lambda, rho). This discovers a
  **schedule** of hyperparameters, not one fixed value.
- AlphaZero PBT: "Accelerating and Improving AlphaZero Using Population Based
  Training" (AAAI 2020): population 16, copy top 20%, perturb LR/value-loss
  ratio by 0.8/1.2.
- This is the closest practical implementation of "autonomously look for the
  deepest basin": exploit good weights, explore hyperparameters, never retrain
  from scratch.

## 4. Recommended autonomous loop for Titanium

Given 4 cores, use a small population and ASHA-style rungs.

1. Fixed subset: 2-3M rows from stage 1; fixed validation probe.
2. Population: 6-8 trials.
   - Optimizer: AdamW, NAdam, RAdam, Lion, AdEMAMix.
   - LR: logarithmic around the known-safe range.
   - Weight decay, outcome weight/lambda, SAM/ASAM rho, Lookahead k/alpha.
3. Rungs:
   - rung 0: 10% of subset, 1 epoch; prune bottom 50%.
   - rung 1: 30%, 2 epochs; prune bottom 50%.
   - rung 2: 100%, full epochs; keep top 2.
4. Inside each trial:
   - optional SAM/ASAM (rho small, e.g. 0.01-0.1 scaled to embeddings);
   - Lookahead k=5, alpha=0.5;
   - cyclic LR or warmup/flat/warmdown;
   - SWA/EMA over the last 25-50%;
   - evaluate val and a small fixed-node gate subset; final selection by
     **5k-node games**, never val alone.
5. PBT/ASHA promotion:
   - copy weights + hyperparameters from the best trial;
   - mutate LR/WD/lambda/rho by 0.8-1.2;
   - continue training; no restart from scratch.
6. After stage 1, fine-tune the winner on stage 2 teacher data (Stockfish/Lc0
   recipe); gate with 100+ paired-book games.

This is not a guarantee of the global optimum. It is a principled search over
basins and hyperparameters with early stopping, flat-minima bias, and game
truth as the final selector.

## 5. What not to do

- Do not launch one AdamW run and call it training.
- Do not select by val loss alone; our val-loss-lies history is explicit.
- Do not run full 12M-row trials for every hyperparameter.
- Do not use teacher-only data from scratch.
- Do not average checkpoints from different basins without checking the gate.
- Do not use SAM/Lookahead/SWA as a substitute for data quality and game gates.

## Sources
- SAM: https://arxiv.org/abs/2010.01412
- ASAM: https://arxiv.org/abs/2102.11600
- SWA: https://arxiv.org/abs/1803.05407
- SWAD: https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html
- Lookahead: https://arxiv.org/abs/1907.08610
- PBT: https://arxiv.org/abs/1711.09846
- AlphaZero PBT: https://ojs.aaai.org/index.php/AAAI/article/download/5454/5310
- ASHA: https://arxiv.org/abs/1810.05934
- BOHB: https://arxiv.org/abs/1807.01774
- Optuna: https://dl.acm.org/doi/pdf/10.1145/3292500.3330701
- Lion: https://github.com/google/automl/blob/master/lion/README.md
- AdEMAMix: https://arxiv.org/abs/2409.03137
- Sophia: https://arxiv.org/abs/2305.14342
- Muon: https://docs.pytorch.org/docs/2.13/generated/torch.optim.Muon.html
- Optimizer comparison: https://arxiv.org/abs/2507.08472
- ACE/Apex: `C:\Users\Terminatort8000\Downloads\ACE_APEX_ARCHITECTURE.md`

## 6. What this means for the Titanium trainer

- `train_stream.py` uses sparse `EmbeddingBag`, but adds a dense manual L2 term
  (`+ wd * sum(w^2)`), which materializes dense gradients and keeps AdamW /
  NAdam / RAdam working. This is effective but is NOT the same as the PyTorch
  sparse path. PyTorch documents SparseAdam only for sparse-layout gradients;
  the clean design is SparseAdam for embedding bags plus Adam/AdamW for dense
  parameters. Any switch must be benchmarked; it changes moment updates.
- Stockfish's loss is in WDL space:
  `wdl = lambda * sigmoid(eval_scaled) + (1-lambda) * game_result`.
  Implemented in `train_stream.py` as `--wdl-loss --wdl-scale`: the target is
  `(1-outcome_weight) * sigmoid(teacher_cp/scale) + outcome_weight * outcome`,
  and the loss is MSE in probability space. `outcome_weight` is the Stockfish
  `1-lambda`.
- Stockfish data loaders do more than dedup:
  - `random_fen_skipping` for diversity;
  - smart skipping of positions whose best move is a capture;
  - WLD-based skipping when eval and outcome do not correlate (especially for
    Lc0 data).
  Titanium currently has no equivalent filters.
- Stockfish uses fake quantization with straight-through estimation for
  activations and weights (`use_fake_act_quantization`,
  `use_fake_weight_quantization`). Titanium exports int8 tables but does not do
  QAT. Add fake quantization to the training graph before the next serious run.
- Hyperparameter search must use a representative subset. `train_stream.py`
  already random-subsamples with `np.random.choice` and splits its own 5%
  validation, which is good; the external val probe should remain a separate
  holdout.
- No method here proves the global minimum. SAM/SWA/PBT/ASHA improve the
  probability of a better basin; the final selector remains 5k/100ms games.

## 7. NNUE pipeline audit vs modern practice (2026-10-24, user: "everything algorithmically optimal")

What the NNUE pipeline (train_nnue.py, export_nnue.py, gate_nnue.py) does today, and what is still missing,
ranked by expected strength per hour of compute on this 4-core box. "Done" = in code and used by the live run.

Done
- Optimizer: Ranger = RAdam (per-parameter adaptive steps, rectified early variance) + Lookahead k8/a0.56, decoupled wd.
- Weight averaging: EMA 0.9995; live and EMA both scored each epoch, the better is kept.
- Loss: power-2.5 error in probability space (nnue-pytorch), stage target blended with outcome (lambda).
- Two-stage distillation: base on Titanium 5k labels, then KataAtaxx 200-visit values as teacher (BEHAVIOR 19).
- Base-stage hyperparameters from population based training (pbt_nnue.py); Optuna pruning hook in run().
- LR: --auto-lr (D69) = val-driven backtracking line search over epochs + SGDR warm restarts from the best state
  (search for a better basin); no hand-set LR after the start; stops only when no restart is left.
- Selection: best checkpoint over all epochs (live or EMA) by val Kata-TD MSE; promotion only by paired game gates.
- Engine parity: exported net == torch to 0 cp on 1000 positions before any gate.

Missing, in the order to add them
1. SWA across basins: average the best states of the warm-restart cycles (SWA/SWAD); keep the average only if val
   AND a 5k gate say it beats the single best. Cheap (no extra training).
2. Quantisation-aware training before the int16/int8 engine port: fake-quant weights/activations with a
   straight-through estimator in the last epochs (Stockfish use_fake_*_quantization), so the speed port keeps the Elo.
3. Teacher-stage hyperparameter search: the teacher run reuses base-stage PBT values; a short PBT/ASHA over
   lambda, wd, Lookahead and optimizer family (RAdam vs schedule-free AdamW vs AdEMAMix) on a 2-3M-row subset, final pick by 5k games.
4. Data: Stockfish-style skipping (positions whose label disagrees wildly with the game outcome, near-duplicate
   openings) and the relabel loop with the current champion (HANDOFFS 3b); data quality beats optimizer tricks.
5. Capacity: v2 (D65: 1024 + pairwise + hidden skips + S57 skip), base then teacher, once v1 is saturated and gated.
6. Val -> games check: measure whether val Kata-TD gains predict 5k-gate Elo across checkpoints (our history says val
   can lie); if not, add a search-agreement probe to the epoch metric.
Not worth it here: mixed precision (Haswell CPU, no bf16), second-order optimizers (Sophia) for a 7M-param net.
