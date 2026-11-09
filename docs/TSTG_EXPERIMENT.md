# TSTG ARCHITECTURE EXPERIMENT — single entry point (read this first; D414/D416, 2026-11-06)

**Goal (user):** an evaluation that is much more capable AND much cheaper than the NNUE (Tree-Squashed Tactical Graph).
**Branch:** `exp/tstg-main` in `C:/gitProjects/titanium-ataxx` (base = origin/main 2c40ae56, verified ancestor). Main is untouched.
**Status:** NOT working as an evaluator. Verified facts: `docs/TSTG_VERIFICATION_2026_11_06.md`. Model list: `docs/TSTG_MODELS.tsv`.

Older TSTG docs (`TSTG_FOUNDATIONS.md`, `TSTG_AUDIT_REPORT_FOR_ARGON.md`, `TSTG_DISASTER_POSTMORTEM_2026_11_06.md`, DECISIONS D387-D412) were
written by a weaker agent between 11-03 and 11-06. They carry a correction banner; where they disagree with this file, **this file wins**.

## 1. Stable facts (measured, with source)
| fact | number | source |
|---|---|---|
| TSTG as **sole evaluator** vs main | **0 points / 128 games** (5k, 10k, 100 ms; SmallNet on/off) | verification sec. 1 |
| TSTG as **move-ordering head over the NNUE** vs main | -1.7 +-44.5 (200 g, 100 ms), -1.7 +-47.6 (200 g, 5k) (D411); +22.6 (D410); v11 -76.7 (198 g) | logs/gates/sprt_100ms_tstg_*; D410/D411 |
| static quality, val_v9_clean, corr vs target (stm=0) | NNUE 0.666, best TSTG v10_best 0.511, v17 0.312, v24 0.390 | logs/tstg_verify/val_full.txt |
| train_tstg_v9 colour bug | 14.5% of train_v9 (4,202,404 rows) trained colour-swapped; **fixed** in this branch (`Rec::to_board` = `Board::from_occ`) | verification sec. 3 |
| v24 / tstg_best | CORRUPT (hand overwrite by build_champion_tstg, now disabled) | TSTG_MODELS.tsv |
| **readout ceiling** of frozen v10_best Z (best linear readout, 948k rows) | corr 0.572 (pooled nonlinear 0.575) vs NNUE 0.666: the bottleneck is the 8-channel representation, not drift | verification sec. 7 |
| v10_best policy head | untrained (hand defaults); 24-float fit exists (tstg_v10_polfit / _pol_dis_fit), never gated | TSTG_MODELS.tsv |

## 2. THE RULES for this experiment (machine-enforced where possible; LAWS B14)
1. **Models are immutable.** Each model is referenced by its store path `data/nnue/tstg_store/tstg_<SHA8>.ttst`; the copies there are read-only.
   A new model = a new file from a fit/train, never an edit to an existing one. After creating one, run `python -I docs/analysis/tstg_model_registry.py`
   and give it a status line in that script's STATUS table.
2. **Every gate = a commit vs main, run through `ti gate`** (it builds the exact commit in a clean worktree):
   ```
   ti gate --kind eval --test exp/tstg-main --test-args "--tstg C:/gitProjects/titanium-ataxx/data/nnue/tstg_store/tstg_<SHA8>.ttst" --nodes 5000
   ```
   **Run it with the working directory = `C:/gitProjects/titanium-ataxx`** (the hub takes the engine repo from the cwd; from anywhere else it falls back
   to the main worktree `mainpromo-ataxx` and writes its records there).
   Enforced by the hub (`titaniumtest/src/repro.rs`):
   - `ti sprt` vs main **refuses a raw `target/release` exe** (only `--dev-raw-exe` gets past, and it stamps the log `DEV_RAW_EXE (NOT a result)`).
   - It **refuses `--tstg` without a file**.
   - It **hashes every file argument** into the log header.
   - It **appends START/DONE rows to `docs/GATE_RUNS.tsv`**.

   `ti gate` additionally writes a JSON record (test/base commit, nets, `test_args`, `test_files` sha, result). It also puts a local annotated
   tag **`gate/<run_id>`** on the tested commit: `git tag -l "gate/*" -n9` lists what was gated against what.
3. **Commit before you gate.** Local git is fine (user 2026-11-06). Uncommitted code cannot be gated: `ti gate` builds the commit, not the folder.
4. **One question per gate, written down first.** Before a gate, add a DECISIONS row: hypothesis, model sha, command, what result means what.
   After it, the result row with `ti insight` + `ti analyze` (LAWS B12).
5. **Order of evidence for an evaluator change:**
   (a) val_v9_clean correlation/MAE vs the NNUE (`verify_tstg_perspective`; minutes);
   (b) 5k-node gate vs main (`ti gate --nodes 5000`);
   (c) 100 ms and deep time legs on the PC.
   Never skip (a): a model below NNUE quality on (a) cannot win (b).
6. No hand-typed weights, no "anti-symmetry" overwrites, no copying phases into other phases.

## 3. Next steps (ranked; each is one commit + one written question)
1. ~~Linear-readout ceiling test~~ DONE (D418): ceiling 0.572 < 0.666, patching v10_best cannot reach the NNUE. Original text: On the frozen Stage 1-3 features Z of v10_best, fit the best linear readout
   (ridge, D4-orbit-shared, per phase) against the target on train_v9 rows with the correct colours. Score it on val_v9_clean.
   - If the ceiling stays well below the NNUE's 0.666 corr: the backbone cannot be rescued. Retrain it (step 2) or rethink the features.
   - If the ceiling approaches 0.666: save the fit as a new store model and go to (b).
2. **Retrain the backbone with the fixed trainer** (train_tstg_v9 now reads colours correctly), from v10_best (law C14: no random init),
   on Modal CPU or Docker. Write a ledger row first: the earlier TSTG Modal runs have no ledger row. Then (a), then (b).
3. **Keep the working path alive:** TSTG as a cheap move-ordering head over the NNUE (D410/D411 parity). Gate the fitted 24-float head
   (tstg_v10_polfit) this way, with `ti gate --test-args` (needs an engine flag for "NNUE eval + TSTG ordering": the 11-04 behaviour of `set_tstg`).
4. Only after (a) shows TSTG >= NNUE quality: speed tuning, SmallNet re-fit on TSTG, dis-scale calibration.
5. Data: run the NNUE data-scaling test (25/50/100% of v9, D415) before generating any new dataset.

## 4. Where everything is
- Code: `crates/titanium/src/tstg.rs` (model), `crates/titanium/src/search.rs` (`set_tstg`, `node_eval_with_dis`), trainers `crates/titanium-cli/src/bin/train_tstg_v9.rs` (fixed) and `train_tstg.rs`.
- Checks: `verify_tstg_perspective` (val quality), `debug_spatval`, `debug_ply3`, `debug_avx_vs_scalar`.
- Logs: `logs/gates/*.log(+.games.jsonl)`, `docs/GATE_RUNS.tsv` (new), `logs/tstg_verify/`.
- Models: `data/nnue/tstg_store/` (read-only, content-addressed), registry `docs/TSTG_MODELS.tsv`.

## 5. Status 2026-11-06 evening (D433-D447; numbers in docs/RESEARCH_ARCH_2026_11_06.md sec. 4)
- Colour-fixed retrain 28B60D6B (11 min Modal CPU): val corr 0.579 (NNUE 0.666); eval 1.34 us vs NNUE 4.96 us (3.7x cheaper).
- Eval-only A/B at equal search (same exe, --policy none --no-fut-sn, TI_TSTG_W=0, 5k, 100 g): 4-93-3 / dis-neutral 5-95-0 /
  NNUE-distilled C52C0AB4 6-92-2 (-449). Not the dis head, not the eval scale (slope 1.014).
- ROOT CAUSE: sibling ranking (best child agreement with the NNUE 49.4% -> 51.6% after distillation; regret 44.8 -> 41.0 cp/move).
- Speed in games 117k nps: the search evaluates most children for futility, SmallNet is off for TSTG -> ~4-6 evals per node.
- NEXT (ranked): (1) listwise distillation on all children (softmax of NNUE child scores / T, arXiv 1503.02531, user D445/D447);
  implementable in train_tstg_v9 by giving each child the target that reproduces the listwise gradient through the existing
  MSE step; (2) wider Z 8 -> 32 channels (Net2Net); (3) TSTG-calibrated child prefilter (~1 eval per node).
- Tools: tstg_child_check (sibling ranking), tstg_cost (ns), tstg_distil_data (children labelled by the NNUE),
  tstg_dis_neutral, verify_tstg_perspective, tstg_readout_ceiling; launchers training/modal_train_tstg_fix.py / _distil.py.
