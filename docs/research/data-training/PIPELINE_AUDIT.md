# Pipeline audit: approximations, bugs, and fixes

Date: 2026-10-21. Scope: data build, relabel, stage export, trainer, autotune.

## Fixed

1. **Double weight decay**
   - `train_stream.py`: AdamW/NAdam/RAdam applied `weight_decay=a.wd` AND the
     loss added manual `a.wd * sum(w^2)`. Every update had double decay.
   - `train_full.py`: same AdamW/RAdam bug.
   - Fix: optimizer weight decay set to `0.0`; the manual L2 remains (it also
     makes gradients dense, which the current sparse EmbeddingBag setup needs).

2. **Destructive frequency decay enabled by default**
   - `train_stream.py` and `train_full.py` defaulted `--freq-decay 2.0`.
   - BACKLOG records that this shrinks whole tables ~91%/epoch at Adam scale.
   - Fix: default is now `0.0`; it is opt-in only.

3. **Autotune invalidated by those two bugs**
   - Stopped and restarted after the fixes.
   - NAdam and Lookahead added to `train_stream.py`.

## Open approximations (not fixed yet)

1. **Centipawn MSE instead of WDL loss**
   - Stockfish: `lambda * sigmoid(eval_scaled) + (1-lambda) * outcome`.
   - Ours: direct centipawn blend `(1-ow)*teacher_cp + ow*outcome_cp`.
   - This is a real approximation; must be replaced for a true Stockfish copy.

2. **Manual L2 instead of sparse optimizer semantics**
   - PyTorch documents SparseAdam for sparse-layout gradients; our dense L2
     term materializes gradients, so AdamW/NAdam/RAdam work.
   - Correct alternative: SparseAdam for embedding bags + Adam/AdamW for dense
     parameters. Must be benchmarked; changes moment updates.

3. **No Stockfish data filters**
   - Missing `random_fen_skipping`, smart capture skipping, WLD-based skipping.
   - We have dedup and outcome validity only.

4. **No QAT/fake quantization**
   - Stockfish uses fake quantized activations/weights with STE.
   - We train float and export int8; quantization error is unmodeled.

5. **Stage-1 labels are old**
   - `stage1_base_*.npz` uses `COALESCE(engine_cp, old_teacher)`.
   - Most own-cache rows still use the old 5k labels; the main relabel was
     stopped at ~320k (mostly earlier IDs) and is not complete.

6. **Autotune is not the full researched loop**
   - Current: 8 phase-1 configs, top-3 phase-2, two PBT mutations, final.
   - Missing: true ASHA rungs, SAM/ASAM, SWA/EMA, Borda/paired game selection.
   - It prunes by validation MSE, not 5k games.

7. **Architecture capacity**
   - TUP7 is a linear sum of tuple tables; no nonlinear hidden head.
   - Prior capacity wall and zero sections (`pa`, `capconv`, etc.) remain.
   - This is an architecture limit, not a training-script bug.

8. **No final 5k gate is wired into autotune**
   - Autotune exports a candidate but does not play paired-book games against
     main. The candidate must not be merged on val loss.

## Sabotage assessment

No evidence of intentional code sabotage found. The concrete failures found
were ordinary code-level approximations:

- double weight decay;
- destructive default frequency decay;
- tool timeouts killing 50k-chunk relabel children (reported as engine exit 1);
- two relabel tasks competing for 4 cores;
- autotune launched before the optimizer fixes.

The 50k-chunk `engine exit 1` errors were not engine corruption. Standalone
50k chunks passed; scheduled 10k chunks were clean. The failures happened when
the tool killed child process trees.
