# 📋 Titanium Ataxx Master Backlog & Roadmap Audit

Living audit tracking all completed engineering milestones, empirical gates, and remaining backlog items across Neural Network Architecture, Search Selectivity, Infrastructure, and Botzone Competition.

---

## 1. Roadmap Progress & Status Matrix

| Track | Status | Recent Milestone | Next Objective |
| :--- | :--- | :--- | :--- |
| **Neural Architecture** | 🟢 ACTIVE | +72 Elo D4 Champion (`7B25AC9D`), TNN3 Disagreement Head, Net2Net W=512 | Saturated W=512 D4 training on v7 $\to$ Net2Net downscale to W=256 |
| **Dataset & Relabeling** | 🟢 ACTIVE | 2,025,000 active-learning 5k relabels completed overnight ($0.00 cost) | Train on v7 dataset (32.62M rows + 88k Botzone anchors) |
| **Search Selectivity** | 🟡 FROZEN | Quiet LMR knob gated (neutral, D147); SmallNet modulation rejected (D147) | Policy-guided move ordering, singular extensions |
| **Botzone Ladder** | 🟢 LIVE | Version 11 crossed **1515.91 Elo** (Rank #147 in world) | Botzone AVX-512 compilation, autonomous ladder climb to top 100 |

---

## 2. Completed Backlog Milestones (Verified & Gated)

1. **Persistent Session Tree Management (D127, D138):**
   - Transposition table aging (`age: u8`), warm start, and 4.0s time management enabled.
   - Fixed startup crash (d1 invariant, Mistake #34) and universal overrun prediction (Mistake #35).
   - Powered Version 11's climb from 1477 to **1515.91 Elo** on the live Botzone arena.

2. **D4 Dihedral Symmetry Augmentation (D140, D141):**
   - Implemented online random 8-mode D4 dihedral permutation table in `train_nnue.py`.
   - Gated over 416 games: **237W – 152L – 27D (+72 Elo)**. Single largest Elo jump in project history.
   - Resolved Mistake #38 (canonical square bias and dead weights on rotated boards).

3. **Zero-Cost Auxiliary Disagreement Head (`TNN3`, D143):**
   - Closed-form Ridge Regression head over $h_1 \in \mathbb{R}^{16}$ and $h_2 \in \mathbb{R}^{32}$ (48 features) across 8 phase buckets.
   - Evaluated 0.000 cp parity vs reference; provides zero-cost tactical volatility credit in child futility and RFP dampening.

4. **SmallNet Leaf Filter & Calibrated Quantile Training (D142):**
   - Distilled $W=32$ SmallNet (`fut_small.tsn`) with exact pinball quantiles ($\tau = 0.02 \to 0.0200$).
   - Prunes 68.0% of heavy Big Net leaf evaluations at 192 ns/eval with only 1.59% false prunes.

5. **Disagreement Head Search Selectivity Tuning (D147):**
   - Proved that SmallNet volatility modulation (`dis_sn_num=48`) weakens the 68% leaf filter, dropping NPS by 20% and losing -32 Elo under time control $\to$ SmallNet filter frozen.
   - Proved softening LMR reductions on late moves (`dis_lmr_num=1`) dilutes PV search depth $\to$ LMR baseline frozen.
   - Quiet LMR boost (`dis_lmr_quiet_red=1`) tested over 416 games at 100ms: 195W–198L–23D (neutral dead heat).

6. **SmallNet D4 Augmentation Gate (D148):**
   - Saturated D4 SmallNet gated over 416 games at 100ms: 185W–208L–23D (47.24% score, Elo -19).
   - Rejected per Gate Law v5 $\to$ production builtin SmallNet stands frozen.

7. **Overnight Disagreement Relabeling (D149):**
   - Relabeled 2,025,000 top-disagreement positions ($|StaticEval - Search5k|$ descending) at 5,000 nodes/move using `nnue_main.tnn` purely on local PC ($0.00 spend).
   - Assembled into `final_train_v7.npz` (32,622,551 rows) with 88,295 Botzone $\ge 2200$ Elo anchors.

---

## 3. Open Backlog Items (Ranked by Expected Elo / Impact)

### 🔴 Phase A: Model Architecture & Training (Current Priority)

1. **Net2Net W=512 (AVX-512) Saturated Training with Online D4 Augmentation (D150, D151):**
   - **Goal:** Expand network capacity from $W=256 \to W=512$ via function-preserving Net2Net-Wider ($<0.02$ cp initial error).
   - **Data:** Trained on `final_train_v7.npz` (incorporating 2,025,000 fresh 5k search labels + 88,295 Botzone anchors).
   - **Regularization:** Online D4 dihedral augmentation across all 8 configurations per batch + uniform weighting (`diff_nudge=0.0`) to prevent over-saturation collapse (D133 vs D146).
   - **Result:** Gated over 416 games at 5,000 nodes vs reigning champion `nnue_main.tnn`: **232W – 170L – 14D (+52 Elo)**, scoreCI `[0.5270, 0.6220] > 0.5000` (**PASS**, Gate Law v5). Shard 0 determinism tripwire replayed 100% bit-identically across all 26 games.

2. **Net2Net Downscale & Noise Cleaning ($W=512 \to W=256$) (Current Priority):**
   - **Concept:** User's capacity breathing cycle: upscale gives expanded capacity to absorb new patterns and sharp disagreement targets; downscaling cleans parameter noise and compresses essential features back into a lean, fast AVX2 model for maximum NPS in search.
   - **Tooling:** Implemented in [`training/net2net_downscale.py`](file:///C:/gitProjects/titanium-ataxx/training/net2net_downscale.py).

3. **Pattern Saliency & Structural Input Features (from `WHAT_MATTERS.md` & `autaxx`):**
   - **Stone Permanence / Safe-Stone Difference:** A stone that cannot be converted next move (no enemy landing square or empty neighbour) is worth 2.5–3× a convertible stone (Kata flip test: .81 vs .22–.37).
   - **Jump-Ring Vulnerability:** Exposed singletons susceptible to 1-to-8 infection swings.
   - **Mobility & Stone Parity Delta:** Explicit clone/jump availability and endgame fill parity.

4. **Stockfish Two-Stage Training Recipe (`STOCKFISH_LC0_RECIPE.md`):**
   - Stage 1: Base Titanium selfplay data (32.5M rows).
   - Stage 2: Fine-tuning on KataGo / deep teacher positions with WDL anchor consolidation.

---

### 🟡 Phase B: Search Selectivity & Move Ordering

1. **Policy-Guided Move Ordering & Cheap Safety-After-Move (`WHAT_MATTERS.md` B):**
   - Jumps with doomed stones: 92% of jump origins in strong play are attackable stones. Jumping costs ~2.2 converted stones if the origin was safe, but only ~1 stone if it was going to be lost anyway.
   - Prioritizing safe clones over wild jumps when ahead in the endgame.
   - Ordering killers / refutations based on net conversion delta.

2. **Singular Extensions / Multi-Cut Extensions for Ataxx:**
   - Detect singular PV moves where all alternative moves fail low by $\ge 150$ cp, extending search depth by +1 ply to resolve tactical traps.

3. **Dynamic Aspiration Window Tuning:**
   - Narrow initial window ($\pm 25$ cp) to fail high/low faster on obvious tactical spikes, with exponential widening on research.

---

### 🟢 Phase C: Botzone Platform & Native SIMD

1. **Botzone Native AVX-512 Compilation Target (`tools/build_botzone.py`):**
   - Section 1.2 of `DISAGREEMENT_HEAD_SPEC.md`: Botzone servers run Xeon/EPYC CPUs with native AVX-512 support.
   - Currently, `titanium_v2.so` targets `glibc 2.14` with SSE2.
   - Compiling with dynamic runtime CPU dispatch (`is_x86_feature_detected!("avx512bw")` already implemented in `nnue.rs:194-224`) will double vector throughput on Botzone (8 loop iterations of 32-lane `i16` vs 16 iterations on AVX2 and 32 on SSE2).

2. **Autonomous Botzone Elo Climber Execution:**
   - Script [`training/botzone_elo_climber.py`](file:///C:/gitProjects/titanium-ataxx/training/botzone_elo_climber.py) ready.
   - Enforces user mandate: no rematches, challenge the highest-Elo unplayed bot above our rating (currently 1515.91), and push toward the top 100.
