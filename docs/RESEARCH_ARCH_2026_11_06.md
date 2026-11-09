# Architecture research 2026-11-06: what the champion knows, what costs time, what TSTG is missing (D423-D452)

Everything below is measured on the PC (i7-4900MQ, 4C/8T, L2 1 MB / L3 8 MB, AVX2, single thread unless noted) against main
2c40ae56 (net 46FCB5A4, policy head AD686DBA, SmallNet 8F479ECB). Validation = data/nnue/packed/val_v9_clean.tdat (551,301
non-proven rows; the file is ORDERED, stm=1 rows sit at the end: use interleaved splits). Raw outputs: logs/tstg_verify/.
Tools are Rust bins in crates/titanium-cli/src/bin/ (branch exp/tstg-main unless noted) and scripts in docs/analysis/.

## 1. Inside the champion NNUE (the user asked to read the neurons and the weights)
| question | answer | tool / D |
|---|---|---|
| dead neurons? | 0 of 560 (2 x 256 accumulator + 16 + 32); 2 active <1%, 3 near-constant | nnue_neuron_probe, D424 |
| does it build the user's relations internally? | yes: R^2 from neurons: reach depth 1/2/3/4 = 0.80/0.77/0.58/0.09, exposed stones 0.87, sealed 0.92, interior (anchors) 0.71, ghost holes 0.85, liability stones 0.77, territory 0.67-0.76 | D424 |
| do those concepts explain its errors? | no: each |r| <= 0.071 with (target - eval); a linear correction from all 27 concepts: test corr 0.881 -> 0.877 | D424 |
| harmful neurons (hand-edit gain)? | none: 16 looked harmful on half A (max -0.035%), replacing them HURTS held-out half B +0.53% | nnue_neuron_ablation, D431 |
| most important units | l1 neurons 2 and 6 (~11-12% MSE each when mean-ablated) | D431 |
| pruning cost (no training, held-out) | W240 +0.73%, W224 +1.52%, W192 +3.81%, W160 +6.65%, W128 +10.35% MSE | D431 |
| weight-space programming | shrink --refit (pivoted Cholesky + least-squares L1 refit, 4 s) wins back 54%; + 2-4 min Modal fine-tune: W224 val 0.028532 (+0.5%), W192 0.028638 (+0.9%) vs control 0.028381 | D431 |
| weight energy by input group | G1 square state 44.9%, G3 stone ring 34.8%, G2 anchor 13.6%, G5 2-ply exchange 5.3% (largest per-row norm), G6 1.4%; 0 dead rows | nnue_weight_anatomy.py, D452 |
| low-rank structure? | no: participation ratio 240.6 of 256 | D452 |
| is the 2nd perspective a duplicate? | no: corr(own_j, enemy_j L1 weights) mean -0.22, a per-neuron scalar x own explains only 21.8% of the enemy weights | D452 |
| remove G2 (anchor)? | +5.0% val error after a fine-tune (0.029790 vs 0.028381): keep | D429/D430 |

Conclusion: the net uses its capacity fully; the relations the user wants are already inputs (nnfeat.rs G1/G3/G5) and are
built internally where they are not. There is no free redundancy to delete.

## 2. What costs time (per eval, 20,173 val positions, best of 7)
| item | ns | tool / D |
|---|---|---|
| full champion eval | 4,962-5,154 | nnue_cost_split, D426 |
| feature extraction (all groups) | 528 (G1 119, G2 67, G3 119, G5 308, G6 32) | D426 |
| accumulator refresh + L1/L2 | ~4,500: 208 active rows x 256 i16 lanes | D426 |
| move generation (reference) | 189 | D426 |
| rows changed per move (incremental potential) | 135.8 of 208 (G1 52.9, G2 25.1, G3 35.2, G5 19.8, G6 2.8) -> incremental saves only ~35% | D426 |
| width (FIXED kernel, D446) | W256 4,987 / W192 4,092 / W128 2,829 / W64 2,300 | D446/D448 |
| i8 FT table (exp/ft-i8) | W256 4,526 (-9%), W192 3,827 (-23%) | D446/D448 |
| Zataxx (source, not timed) | 36 active rows, ONE accumulator, 1.5 MB table, no hidden layers, 1 eval per node | D451 |

CORRECTION (D446): the earlier claim 'eval is cache-bound' (D432) was wrong; ft_sum's blocked path and g6tab required
width % 128 == 0, so W224/W192/W96 ran a scalar fallback. Fixed on exp/ft-i8 (width % 64). The eval is mostly compute-bound.
Search side: the futility code evaluates most CHILDREN of every node (search.rs ~1338-1353); SmallNet pre-filters about half.

## 3. 8-bit feature table (branch exp/ft-i8, worktree C:/gitProjects/i8-ataxx, commit eef97292, from origin/main)
- Offline (nnue_i8_probe on exp/tstg-main): per-lane scale: eval error mean 1.61 cp, max 19.4 cp; same best sibling as the float
  net 98.9%, regret 0.01 cp/move (global scale: 2.98 cp, 97.5%). Lossless in practice. (D444)
- Engine: `--ft-i8` flag (default off = bit-identical, bench depth 8 = 14,932 nodes); i8 rows with per-lane integer scales
  s_j = floor(127/max|col j|), activation clamp(a_j, 0, s_j)/s_j; ft_sum/g6tab blocked for width % 64. (D445/D446)
- Gate vs main, 100 ms, PC (ti gate tag gate/2026-11-06-eef97292-aD5D7C9C0-vs-2c40ae5e-ms100): 200 g +31.4 +-48; extension
  400 g (book0 200) +6.9 +-32; pooled 600 g ~ +15 Elo, P ~87%; played-game nps +12.3%; insight: value lost/move 12.6 vs 14.2 cp,
  blunders 3.7 vs 3.9%. RUNNING: 1,000 more at 100 ms (book0 400) + 1000 ms leg 120 g (logs/gates/d450_run.sh). (D449/D450)
- RESULT (D456/D465): 100 ms pooled 1,318 g = 623-623-72 = 0 Elo; 1000 ms 120 g = -37.8 +-67.4 (LOS 13%), knps 147 vs 152
  (the speed edge vanished), value lost/move 8.0 vs 5.8 cp (z -1.11). Undecided, no Elo gain seen; not a promotion candidate now.

## 4. TSTG (the user's architecture), what was wrong and what is left
- Colour bug: train_tstg_v9 trained 14.5% of rows colour-swapped (fixed 48ce57b). Retrain from v10_best on correct colours
  (Modal CPU 16 vCPU, 20 epochs x 32 s = 11 min): val corr 0.511 -> 0.579 (NNUE 0.666), MAE 306 -> 284 (NNUE 276),
  stm=1 0.705 -> 0.875 (NNUE 0.895). Model 28B60D6B. (D433/D434)
- Speed: TSTG eval 1,335 ns vs NNUE 4,962 (3.7x cheaper). In games 117k nps (main 74k at 8 parallel games): the search evaluates
  children for futility and SmallNet is NNUE-calibrated (off for TSTG) -> ~4-6 TSTG evals per node. (D435/D442)
- Eval-only A/B (same exe, --policy none --no-fut-sn, TI_TSTG_W=0, 5k, 100 g): colour-fix 4-93-3 (-494), dis neutralised
  5-95-0 (-511), NNUE-distilled 6-92-2 (-449). Eval slope vs NNUE 1.014 (scale is fine). (D437-D441)
- ROOT CAUSE: SIBLING RANKING. TSTG picks the NNUE's best child only 49.4% (distilled 51.6%), regret 44.8 (41.0) cp/move,
  within-parent Spearman 0.84 (0.86). Cross-position correlation (0.83-0.92) hides it. (D439/D441, tstg_child_check)
- Next for TSTG: listwise distillation (Hinton et al. arXiv 1503.02531, user D445/D447: train on the teacher's softened
  distribution over ALL children, not per-child values), wider Z (8 -> 32 channels, Net2Net), TSTG-calibrated prefilter for
  ~1 eval per node. See docs/TSTG_EXPERIMENT.md.

## 5. Infrastructure added today (see docs/TSTG_EXPERIMENT.md sec. 2, LAWS B14)
- Hub (titaniumtest): repro.rs guard (raw exes refused vs main, --tstg needs a file, file hashes in headers, docs/GATE_RUNS.tsv),
  ti gate --test-args + gate/<run_id> tags, CPU trainer accepts --result-weight-phase. Commits 9bc3cb3, 39728f3.
- Modal: fine-tunes take 2-4 min (ti train modal launch with the rwphase9 recipe args from titaniumtest/docs/MODAL_RUNS.jsonl);
  TSTG trainer on Modal CPU ~32-54 s per epoch (training/modal_train_tstg_fix.py, ..._distil.py, PYTHONUTF8=1 needed on Windows;
  a local `timeout` kills the client but --detach keeps the remote run).
- Immutable TSTG model store data/nnue/tstg_store/ + docs/TSTG_MODELS.tsv (docs/analysis/tstg_model_registry.py).

## 6. Ranked next steps (architecture)
1. Finish the i8 gate (100 ms pooled to P >= 95%, 1000 ms leg); merge = user's call.
2. Fewer evals per node: measure evals/node (counter), then a cheaper child prefilter or parent-eval futility; this is where
   Zataxx-like node rates come from.
3. Fewer active rows per eval: candidate feature redesigns (Zataxx-style 2x2 windows + G3 safety; joint per-square codes), each
   trained by graft init + Modal fine-tune, judged by val + sibling ranking + eval ns before any gate.
4. W192 + i8 (-23% eval, +0.9% val error) needs a policy-head refit (ti policy) before gating: any new net disables the bundled head.
5. Champion policy head on soft targets (D447) if `ti policy` fits argmax targets.
6. Search tuning by phase: the 16-phase SearchPhaseParams (D410) were never tuned (SPSA on Docker/Modal CPU).

## 7. Not done / unknown
Exact evals per node; Zataxx node rate re-measured on this PC; AVX-512 tier for i8; per-neuron concept labels; the champion
policy head's training target; TT pollution theory; 31M-set correlation claim; endgame-only quality split.

## 8. Why Stockfish-style AB engines get a fast strong net and we struggle (user question 2026-11-06 ~20:40)
1. Incremental updates: a quiet chess move changes ~2 features per perspective (from, to) out of ~30 active, so the
   accumulator update is ~4 rows; the champion's relational features (G1 reach states, G3 rings, G5 exchange) change
   135.8 of 208 rows per move (D426), so incremental saves only ~35% and we pay ~a full refresh per node.
2. One eval per node (cached in the TT); Titanium evaluates most CHILDREN for futility (~4-6 evals per node for TSTG, D442).
3. Search tuned for years on fishtest (billions of games); our 16-phase search params were never tuned (D410), SPSA found nothing.
4. Data: billions of positions vs our ~30M.
5. Ataxx is harder per eval: a move flips up to 8 stones and branching is larger, so the static eval is more volatile.
Lesson for the architecture: inputs must be LOCAL so a move touches few rows (Zataxx 2x2: 36 rows, 1 eval/node, ~1.5M nps).
That is what the D463-D470 pattern nets test (t8s 5x5 per stone, k3s 3x3 per stone = 57 rows/eval): first match the
champion's quality, then implement the incremental update and count evals per node.
