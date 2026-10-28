# Titanium Technical Documentation & Research

Welcome to the engineering specifications, algorithmic research, and competitive intelligence documentation for the Titanium Ataxx Engine.

---

## Document Index

### 1. Engine Architecture & Feature Specifications
* **[Master Backlog, NNUE Saturation & Disagreement Head (TNN3 + TSN1)](./DISAGREEMENT_HEAD_SPEC.md)**
  * **Objective:** Achieve MCTS / PUCT-quality exploration at Alpha-Beta speeds while saturating the NNUE teacher on 31.57M 5k-labeled positions.
  * **Capacity vs. Information Diagnostic:** Empirical comparison of $W=256$ vs. $W=512$ validation loss plateaus before scaling runtime dimensions, plus feature saliency ablation inspired by Tuxel's `autaxx`.
  * **Dual-Target Hardware SIMD:** Zero-padding $W=256$ alignment for both **AVX-2 (`x86-64-v3`)** local play and **AVX-512 (`x86-64-v4`)** Botzone server acceleration (halving feature-transformer loop iterations from 16 to 8).
  * **Zero-Cost Disagreement Head:** Dual-headed NNUE predicting both positional valuation ($cp$) and search-tree volatility ($|\text{Search}_{5k} - \text{StaticEval}|$) packed into a single 32-bit `ecache` slot `(eval_cp: i16, dis_cp: u16)`.

* **[Persistent Search Tree vs. One-Shot Engine on Botzone](./PERSISTENT_VS_ONESHOT_BOTZONE.md)**
  * **Objective:** Overcome the per-turn time penalty of persistent mode on Botzone by eliminating TT pollution and state corruption across moves.
  * **Key Fixes (`exp/persistent-tree`):**
    1. Zero-RAM TT generation aging (`age: u8` in compiler padding) to evict unreachable earlier-turn nodes without RAM growth.
    2. Preservation of TT refutation moves on fail-low (`FLAG_UPPER`).
    3. Softened history decay (`*h * 3 / 4`) to retain learned positional heuristics.
  * **Measurement Protocol:** Asymmetric gating using `titaniumtest sprt --cand-reset-every-move / --base-reset-every-move`.

### 2. Competitive Intelligence & Community Benchmarks
* **[Community Research & Competitive Intelligence (2025–2026)](./COMMUNITY_RESEARCH_2026.md)**
  * Comprehensive survey of competitive engines, open datasets, and cutting-edge techniques:
    * **Superhuman Datasets:** KataGo Ataxx 2026 MCTS (ModelScope, 18-block/20-block ResNet/Transformer), Jamie Whiting's Monty, Tuxel's Autaxx, Mexx, Moonbird weights, and balanced 2/4-ply opening books.
    * **Hardware & Bitboards:** BMI2 `pext` parallel bit extraction for 8-neighbor infection flips and AVX-512 vectorization.
    * **Network Architectures:** $D_4$ dihedral symmetries, Karpathy-style autoregressive recipes, and SVD weight downscaling ($W=512 \to W=256$).
    * **Botzone Competition:** Environment quirks, glibc sandboxing, memory caps, and the top-ranked bot `atx`.

### 3. Historical Verification & Gates
* **[Experiment Ledger (LEDGER.md)](../LEDGER.md)**
  * Canonical, continuous log of every tested feature, bugfix, and neural network promotion from engine baseline (E0) through contemporary milestones (S57, EVORD, NNUE Stage-1, Depth 255, and Botzone Ladder).
