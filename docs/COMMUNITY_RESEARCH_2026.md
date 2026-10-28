# 🌐 Ataxx Community Research & Competitive Intelligence (2025–2026)

A curated compilation of competitive benchmarks, superhuman datasets, open-source tooling, and algorithmic insights gathered from top Ataxx and chess engine developers (*jw1912*, *Ciekce*, *hzy_sigmoid*, *kz04px*, *Serdra*, *MrZéro*, *jacek*, *Twipply*).

---

## 1. Superhuman Datasets & Neural Networks

### 1.1 KataGo Ataxx 2026 MCTS Dataset (`hzy_sigmoid`)
* **Dataset:** [ModelScope: KataGo_Ataxx_202604](https://www.modelscope.cn/datasets/sigmoid/KataGo_Ataxx_202604) (older archive: [KataGo_Ataxx_202304](https://www.modelscope.cn/datasets/sigmoid/KataGo_Ataxx_202304/files))
* **Engine Releases:** [KataGomo Releases (Ataxx 20260501 & 20250131)](https://github.com/hzyhhzy/KataGomo/releases/tag/Ataxx_20260501)
* **Architecture:** 18-block / 20-block ResNet and Transformer architectures ([KataGo_Transformer](https://github.com/hzyhhzy/KataGo_Transformer)) with TensorRT acceleration and full blocked-board support (`setfen`/`getfen`).
* **Utility for Titanium:** Multi-million position superhuman MCTS evaluations, win probabilities, and policy distributions suitable for high-depth NNUE teacher distillation.

### 1.2 Jamie Whiting's Monty Ataxx (`jw1912`)
* **Repository:** [jw1912/official-monty (ataxx branch)](https://github.com/jw1912/official-monty/tree/ataxx)
* **Weights & Formats:** Pre-trained 9 MB SMP Ataxx network; uses [official-monty/montyformat](https://github.com/official-monty/montyformat) and [montytrain](https://github.com/official-monty/montytrain).
* **Utility for Titanium:** High-level sparring partner and benchmark for UAI tournament gating.

### 1.3 Knowledge Distillation: KL-Divergence
* **Reference:** [Why KL-Divergence in Knowledge Distillation](https://medium.com/@buroojghani/why-kl-divergence-in-knowledge-distillation-1375d555a728)
* **Finding:** When training an NNUE on soft MCTS teacher probabilities, KL-divergence over the policy/outcome distribution prevents overconfidence and preserves tactical edge cases far better than scalar MSE on centipawns.

---

## 2. Competitive Arenas & Platform Edge Cases

### 2.1 Botzone (`https://en.botzone.org.cn/game/Ataxx`)
* **The "3.8 Route" Exploit:**
  * Native C++ bots receive only a strict **1.0s** per-turn limit.
  * Python entries receive up to **5.0s** per turn.
  * Top entries exploit this by writing a lightweight Python wrapper that spawns or loads a statically compiled native Linux binary (`x86_64-linux-musl-g++ -static` or Zig-compiled `.so`), granting nearly **4x more search depth** (`movetime 3.8s`).
  * **Titanium Implementation:** Cross-compiled `titanium_v2.so` via Zig, loaded via Python `ctypes` (`main.py`) with `simpleio = true` and `enable_keep_running = false`, claiming the full 12s initial / 6s turn time budget cleanly.
* **Adjudication Edge Cases:** 100-turn limit, immediate win on wipeout or piece-count exhaustion, strict repetition checks (`b.bz_over()`).

### 2.2 Alphano (`https://alphano.co.kr/en`)
* South Korean competitive AI platform featuring **Ataxx** with standard I/O (C++/Python).
* Direct arena for evaluating Titanium against international RL bots and community agents.

### 2.3 CodinGame Ataxx (Gaps & Blockers)
* **Contribution:** [CodinGame Ataxx with Gaps Contribution](https://www.codingame.com/contribute/view/148977bf421781578ba30fdec16a791be51718) (by *jacek*).
* Focuses on non-empty 7x7 boards with blockers, exactly matching Titanium's core bitboard specialization.

---

## 3. Tooling Ecosystem & Distributed Testing

### 3.1 OpenBench vs Fishtest
* **OpenBench (OB):** [AndyGrant/OpenBench](https://github.com/AndyGrant/OpenBench)
  * The standard distributed testing framework for the wider computer chess/games community (*Stormphrax*, *Viridithas*, *Ethereal*, *Berserk*).
  * Unlike official Fishtest (which is hardcoded for Stockfish), OpenBench is engine-agnostic and supports community worker fleets via Django/REST.
* **TitaniumTest:** [titaniummachine1/titaniumtest](https://github.com/titaniummachine1/titaniumtest)
  * Standalone single-binary Rust framework implementing OpenBench's exact Pentanomial SPRT math (`[-2.944, +2.944]`), continuous LLR stopping, and paired book openings.
  * Tailored for native UAI Ataxx, HTTP worker coordination, and Bullet-style sparse-feature NNUE training.

### 3.2 Reference Engines & Core Implementations
* **Sanctaphraxx (Rust):** [Ciekce/sanctaphraxx](https://github.com/Ciekce/sanctaphraxx) — Ciekce's Rust Ataxx engine based on Stormphrax; reference for [NNUE inference](https://github.com/Ciekce/sanctaphraxx/blob/main/src/nnue/mod.rs#L206).
* **Moonbird (Nim):** [tsoj/Moonbird](https://github.com/tsoj/Moonbird/releases/tag/1.0.0) — One of the strongest open-source Ataxx engines (written in Nim). Notable for [2x2 tuple evaluation](https://github.com/tsoj/Moonbird/blob/5281d7d8aa17125a01af520b74a502e6a8e4262e/src/evaluation.nim#L47), [training data generation](https://github.com/tsoj/Moonbird/blob/main/src/tuning/generateTrainingData.nim), and [Lazy SMP](https://github.com/tsoj/Moonbird/releases/tag/1.1.0).
* **Mexx & ataxx-rs (Rust):** [raklaptudirm/mexx](https://github.com/raklaptudirm/mexx) — Clean pure-Rust Ataxx engine and [crates.io/crates/ataxx](https://crates.io/crates/ataxx); reference for [board representation](https://github.com/raklaptudirm/mexx/blob/f0fdfccd7a3052d3cd778a7547994ae665f2258c/ataxx/src/position.rs#L139) and MCTS tree implementation.
* **libataxx & Autaxx:** [kz04px/libataxx](https://github.com/kz04px/libataxx), [kz04px/autaxx](https://github.com/kz04px/autaxx), and [kz04px/rustaxx](https://github.com/kz04px/rustaxx) — Reference C++ and Rust movegen, [gameover logic](https://github.com/kz04px/libataxx/blob/master/src/gameover.cpp), and perft suites.
* **Anthraxx:** [Vast342/Anthraxx](https://github.com/Vast342/Anthraxx/compare/tt-cutoff) — TT cutoff and move-ordering reference.
* **ataxx-dot-cpp:** [GediminasMasaitis/ataxx-dot-cpp](https://github.com/GediminasMasaitis/ataxx-dot-cpp) — Reference C++ engine and [evaluation_hce.cpp](https://github.com/GediminasMasaitis/ataxx-dot-cpp/blob/main/src/evaluation_hce.cpp).
* **Gotaxx:** [cpirc/gotaxx](https://github.com/cpirc/gotaxx) — Go-based Ataxx engine.
* **Balanced Opening Books:** [zzzzz151/Ataxxer balanced_openings.txt](https://github.com/zzzzz151/Ataxxer/blob/main/balanced_openings.txt) — Standard community opening book for tournament gating.
* **Sotax Dataset:** [sotax.zip on Google Drive](https://drive.google.com/file/d/1Or5aB7WdD_4FgDMRNq2tfu02LWRVlWtc/view?usp=drive_link) — Community dataset by *metaroots*.
* **Botzone Top Bot `sigmoid rawnn`:** [Botzone Bot ID `6397649a1688b565e8f38fcd`](https://botzone.org.cn/game/ranklist/5809c7647f65182b044b5e3b?page=0#6397649a1688b565e8f38fcd) by *hzy_sigmoid*.

### 3.3 Tournament Runners & GUIs
* **Coliseum:** [PhelRin/Coliseum](https://github.com/PhelRin/Coliseum) — UAI-compatible tournament runner and GUI for Ataxx and chess.
* **CuteAtaxx:** [kz04px/cuteataxx](https://github.com/kz04px/cuteataxx) — Ataxx tournament CLI, [adjudication logic](https://github.com/kz04px/cuteataxx/blob/4df2abe471d8564e47beca270fa2c97a969c9d6b/src/core/ataxx/adjudicate.cpp#L11-L62), and [TUI match runner](https://github.com/kz04px/cuteataxx/blob/tui/src/core/match/worker.cpp).
* **AtaxxGUI:** [tsoj/AtaxxGUI](https://github.com/tsoj/AtaxxGUI) — UAI engine-vs-engine and human interface (with [tournament_maybe branch](https://github.com/tsoj/AtaxxGUI/tree/tournament_maybe)).

### 3.4 Training & Tuning Principles
* **Karpathy's Recipe for Training Neural Networks:** [http://karpathy.github.io/2019/04/25/recipe/](http://karpathy.github.io/2019/04/25/recipe/)
  * *Step 1: Inspect data thoroughly* — know distributions and outliers before feeding them to loss functions.
  * *Step 2: End-to-end skeleton + overfit a small batch* — verify that the model can achieve zero training error on a tiny batch before scaling.
  * *Step 3: Monitor learning rate and loss curves* — ensure gradients are neither vanishing nor blowing up.
  * *Step 4: Regularize only after capacity is proven.*
* **D4 Dihedral Symmetry Expansion:** [jw1912/monty expand.rs](https://github.com/jw1912/monty/blob/main/train/value/src/bin/expand.rs#L5) — 8-fold dihedral symmetry transforms (horizontal/vertical flip, 90-degree rotations) to expand training dataset volume by up to 8x without collecting new games.
* **Bullet:** [jw1912/bullet](https://github.com/jw1912/bullet) — Pure-Rust / CUDA high-throughput sparse NNUE trainer.
* **Texel Tuner:** [lynx-chess/texel-tuner](https://github.com/lynx-chess/texel-tuner) — SPSA and Texel parameter optimization for heuristic margins and evaluation terms.
* **Delta Universal Hashing (Nguyen & Roscoe 2011):**
  * Fast $2^{-63}$-almost delta universal hash function using 3 64-bit integer constants:
    `X = 6364136223846793005`, `Y = 1442695040888963407`, `Z = 2305843009213693951`, with bitwise complement on Black STM. High-speed alternative or complement to Zobrist keys.

---

## 4. Architectural Lessons for Titanium

1. **Win/Loss (WDL) vs Stone-Count Greed:**
   * Training or searching to maximize stone count causes engines to make greedy short-term captures that open defensive vulnerabilities.
   * Titanium optimizes strictly for **Win Probability** ($P(\text{win}) = \text{sigmoid}(\text{cp}/400)$), selecting guaranteed positional clamps over risky captures.
2. **Shortest Mate Distance:**
   * Mate distance pruning ensures Titanium closes out won positions in the minimum number of plies, avoiding repetition swindles or turn-limit draws.
3. **Graph History Interaction (GHI) in Transposition Tables:**
   * Pruning repetition branches must never store repetition-tainted values into the global Transposition Table (`rep_skipped`), preventing corruption of independent search paths.
