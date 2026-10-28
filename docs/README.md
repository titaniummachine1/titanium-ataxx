# Titanium Technical Documentation & Research

Welcome to the engineering specifications, algorithmic research, and competitive intelligence documentation for the Titanium Ataxx Engine.

---

## Document Index

### 1. Engine Architecture & Feature Specifications
* **[Zero-Cost PUCT Disagreement Head (TNN3)](./DISAGREEMENT_HEAD_SPEC.md)**
  * **Objective:** Achieve MCTS / PUCT-quality exploration at Alpha-Beta speeds.
  * **Architecture:** Dual-headed NNUE predicting both positional valuation ($cp$) and search-tree volatility ($|\text{Search}_{5k} - \text{StaticEval}|$).
  * **Zero Memory Overhead:** Packed 32-bit `ecache` slot `(eval_cp: i16, dis_cp: u16)`.
  * **Active Learning Loop:** Autonomous dataset expansion harvesting high-disagreement frontiers.

* **[Persistent Search Tree vs. One-Shot Engine on Botzone](./PERSISTENT_VS_ONESHOT_BOTZONE.md)**
  * **Objective:** Overcome the ~2-second penalty of persistent mode on Botzone by eliminating memory bloat and state corruption across moves.
  * **Key Fixes:**
    1. Zero-RAM TT generation aging (`age: u8` in compiler padding) to evict zombie nodes from turns 1–5 without RAM growth.
    2. Preservation of TT refutation moves on fail-low (`FLAG_UPPER`).
    3. Softened history decay (`*h * 3 / 4`) to retain learned positional heuristics.
  * **Measurement Protocol:** Asymmetric gating using `titaniumtest sprt --cand-reset-every-move / --base-reset-every-move`.

### 2. Competitive Intelligence & Community Benchmarks
* **[Community Research & Competitive Intelligence (2025–2026)](./COMMUNITY_RESEARCH_2026.md)**
  * Comprehensive survey of competitive engines, open datasets, and cutting-edge techniques:
    * **Superhuman Datasets:** KataGo Ataxx 2026 MCTS (ModelScope, 18-block/20-block ResNet/Transformer), Jamie Whiting's Monty, Mexx, Moonbird weights, and balanced 2/4-ply opening books.
    * **Hardware & Bitboards:** BMI2 `pext` parallel bit extraction for 8-neighbor infection flips.
    * **Network Architectures:** $D_4$ dihedral symmetries, Karpathy-style autoregressive recipes, and SVD weight downscaling ($W=512 \to W=256$).
    * **Botzone Competition:** Environment quirks, glibc sandboxing, memory caps, and the top-ranked bot `atx`.

### 3. Historical Verification & Gates
* **[Experiment Ledger (LEDGER.md)](../LEDGER.md)**
  * Canonical, continuous log of every tested feature, bugfix, and neural network promotion from engine baseline (E0) through contemporary milestones (S57, EVORD, NNUE Stage-1, Depth 255, and Botzone Ladder).
