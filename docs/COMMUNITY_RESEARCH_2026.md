# 🌐 Ataxx Community Research & Competitive Intelligence (2025–2026)

A curated compilation of competitive benchmarks, superhuman datasets, open-source tooling, algorithmic insights, and distillation roadmaps gathered from top Ataxx and computer game engine developers (*jw1912*, *Ciekce*, *hzy_sigmoid*, *kz04px*, *Serdra*, *MrZéro*, *jacek*, *Twipply*, *tsoj*, *Gedas*).

---

## 1. Superhuman Datasets, Neural Networks & Weights

### 1.1 KataGo Ataxx 2026 MCTS Dataset (`hzy_sigmoid`)
* **Primary Dataset (ModelScope):** [KataGo_Ataxx_202604](https://www.modelscope.cn/datasets/sigmoid/KataGo_Ataxx_202604)
* **Archive Dataset:** [KataGo_Ataxx_202304](https://www.modelscope.cn/datasets/sigmoid/KataGo_Ataxx_202304/files)
* **Engine Releases:** [KataGomo Releases (Ataxx 20260501 & 20250131)](https://github.com/hzyhhzy/KataGomo/releases/tag/Ataxx_20260501)
* **Architecture:** 18-block / 20-block ResNet and Transformer architectures ([KataGo_Transformer](https://github.com/hzyhhzy/KataGo_Transformer), [arXiv:2410.05347](https://arxiv.org/abs/2410.05347)) with TensorRT int8 quantization and full blocked-board support (`setfen`/`getfen`).
* **Utility for Titanium:** Multi-million position superhuman MCTS evaluations, win probabilities, and policy distributions suitable for high-depth NNUE teacher distillation.

### 1.2 Jamie Whiting's Monty Ataxx (`jw1912`)
* **Repository:** [jw1912/official-monty (ataxx branch)](https://github.com/jw1912/official-monty/tree/ataxx)
* **Commit Reference:** [Monty ataxx commit 71f7a64](https://github.com/official-monty/Monty/commit/71f7a6430578313a2e0cd2c872d52323b0d59fbf)
* **Search Tree & Node Architecture:** [monty/src/tree/node.rs#L79](https://github.com/jw1912/monty/blob/main/src/tree/node.rs#L79)
* **Weights & Formats:** Pre-trained 9 MB SMP Ataxx network; uses [official-monty/montyformat](https://github.com/official-monty/montyformat) and [montytrain](https://github.com/official-monty/montytrain).
* **Dihedral Symmetry Data Expansion:** [jw1912/monty expand.rs](https://github.com/jw1912/monty/blob/main/train/value/src/bin/expand.rs#L5) — $D_4$ 8-fold dihedral symmetry transforms (horizontal/vertical flip, 90-degree rotations) to expand training dataset volume by up to 8x without generating new search nodes.

### 1.3 Mexx & ataxx-rs (`raklaptudirm`)
* **Repository:** [raklaptudirm/mexx](https://github.com/raklaptudirm/mexx) (branches: `mexx`, `crabgrab-monty`)
* **Crates.io Package:** [crates.io/crates/ataxx](https://crates.io/crates/ataxx) (v0.2.0)
* **Board Representation:** [mexx/ataxx/src/position.rs#L139](https://github.com/raklaptudirm/mexx/blob/f0fdfccd7a3052d3cd778a7547994ae665f2258c/ataxx/src/position.rs#L139)
* **MCTS Tree & Graph:** [mexx/src/mcts/graph.rs#L59](https://github.com/raklaptudirm/mexx/blob/mexx/src/mcts/graph.rs#L59) (recursive graph implementation) and [mexx/src/mcts/mod.rs#L134](https://github.com/raklaptudirm/mexx/blob/mexx/src/mcts/mod.rs#L134).
* **Speed Optimizations:** [Commit 144f073](https://github.com/raklaptudirm/mexx/commit/144f073d6faa52a7a7f7a0439ee7271256ba3ac5).

### 1.4 Moonbird & Community Datasets (`tsoj`, `zzzzz151`, `metaroots`)
* **Moonbird Engine:** [tsoj/Moonbird Releases](https://github.com/tsoj/Moonbird/releases/tag/1.0.0)
* **Lazy SMP Implementation:** [tsoj/Moonbird Release 1.1.0](https://github.com/tsoj/Moonbird/releases/tag/1.1.0) and [Lazy SMP commit cf570ad](https://github.com/tsoj/Moonbird/commit/cf570add56dd5ebe4b25ca53c64a830a837fea17).
* **Pre-trained Weights Archive:** [Google Drive - tsoj weights](https://drive.google.com/file/d/1X3EEvABxiu1vwt38fYIdan9phW57KQHl/view?usp=sharing).
* **Best Move Annotated Dataset:** [Google Drive - zzzzz best move dataset](https://drive.google.com/file/d/1jFsY-nGPURF5Cgqym9VP3vSfaFVmwDaF/view?usp=sharing).
* **Sotax Dataset:** [Google Drive - sotax.zip](https://drive.google.com/file/d/1Or5aB7WdD_4FgDMRNq2tfu02LWRVlWtc/view?usp=drive_link) by *metaroots*.
* **Balanced Openings Book:** [zzzzz151/Ataxxer balanced_openings.txt](https://github.com/zzzzz151/Ataxxer/blob/main/balanced_openings.txt) — standard 2-ply and 4-ply opening book for fair engine gating.

---

## 2. Competitive Arenas & Platform Edge Cases

### 2.1 Botzone (`https://en.botzone.org.cn/game/Ataxx`)
* **Top Bot:** `sigmoid rawnn` ([Botzone Bot ID `6397649a1688b565e8f38fcd`](https://botzone.org.cn/game/ranklist/5809c7647f65182b044b5e3b?page=0#6397649a1688b565e8f38fcd)) by *hzy_sigmoid*.
* **The "3.8 Route" Exploit:**
  * Native C++ bots on Botzone receive a strict **1.0s** per-turn limit.
  * Python bots receive up to **5.0s** per turn (with 12s initial startup budget).
  * Community recipe for bypassing the 1.0s limit:
    ```bash
    # Download prebuilt musl toolchain
    wget https://musl.cc/x86_64-linux-musl-native.tgz
    tar -xzf x86_64-linux-musl-native.tgz
    export PATH=$PATH:$(pwd)/x86_64-linux-musl-native/bin
    # Compile static binary
    x86_64-linux-musl-g++ -std=c++17 -O3 -static cheetax.cpp -o master
    ```
    Wrap `master` in a Python wrapper via `subprocess.Popen` or `ctypes`, setting search `movetime` to **3.8s** to cleanly consume the 5s budget and obtain **~4x more search depth**.
  * **Titanium Implementation:** Cross-compiled `titanium_v2.so` via Zig, loaded via Python `ctypes` (`main.py`) with `simpleio = true` and `enable_keep_running = false`, eliminating subprocess overhead and claiming the full time allowance.
* **Adjudication Edge Cases:** 100-turn limit, immediate win on wipeout or piece-count exhaustion, strict repetition checks (`b.bz_over()`).

### 2.2 Alphano (`https://alphano.co.kr/en`)
* South Korean competitive AI platform featuring **Ataxx** with standard stdin/stdout protocols.
* Hosts international RL bots and community agents for Elo rating evaluation.

### 2.3 CodinGame Ataxx (Gaps & Blockers)
* **Contribution:** [CodinGame Ataxx with Gaps Contribution](https://www.codingame.com/contribute/view/148977bf421781578ba30fdec16a791be51718) (by *jacek*).
* Focuses on non-empty 7x7 boards with blocker patterns, matching Titanium's bitboard gap representation.

### 2.4 Pressibus Historical Inventory
* **Reference:** [Pressibus Ataxx & Hexxagon Inventory](http://pressibus.org/ataxx/indexgb.html) — Comprehensive archive of historical Ataxx variants, board sizes, and mathematical properties.

---

## 3. Tooling Ecosystem, Protocols & Reference Engines

### 3.1 Protocols: UAI, UGI & UXI
* **UAI (Universal Ataxx Interface):** Standard UCI adaptation for Ataxx used by CuteAtaxx and AtaxxGUI. Note: clone moves are destination square (2 characters, e.g. `c5`), jumps are origin+destination (4 characters, e.g. `a1c3`).
* **UGI (Universal Game Interface):** [kz04px/cutegames ugi.md](https://github.com/kz04px/cutegames/blob/master/ugi.md) — Multi-game protocol standard designed to replace UCI for generalized board games.
* **UXI (Universal XML/X Interface):** [crates.io/crates/uxi](https://crates.io/crates/uxi) by *raklaptudirm*.

### 3.2 Tournament Runners & GUIs
* **Coliseum:** [PhelRin/Coliseum](https://github.com/PhelRin/Coliseum) — UAI-compatible tournament runner and GUI for Ataxx and chess.
* **CuteAtaxx:** [kz04px/cuteataxx](https://github.com/kz04px/cuteataxx) — Ataxx tournament CLI, [adjudication logic](https://github.com/kz04px/cuteataxx/blob/4df2abe471d8564e47beca270fa2c97a969c9d6b/src/core/ataxx/adjudicate.cpp#L11-L62), and [play.cpp](https://github.com/kz04px/cuteataxx/blob/master/src/core/play.cpp#L18).
* **AtaxxGUI:** [tsoj/AtaxxGUI](https://github.com/tsoj/AtaxxGUI) and [tournament_maybe branch](https://github.com/tsoj/AtaxxGUI/tree/tournament_maybe) — Interactive graphical board and engine tournament coordinator.

### 3.3 Engine References & Architecture
* **Sanctaphraxx (Rust):** [Ciekce/sanctaphraxx](https://github.com/Ciekce/sanctaphraxx) — Reference for [NNUE inference](https://github.com/Ciekce/sanctaphraxx/blob/main/src/nnue/mod.rs#L206) and [PR #11](https://github.com/Ciekce/sanctaphraxx/pull/11).
* **libataxx:** [kz04px/libataxx](https://github.com/kz04px/libataxx) — Standard C++ reference for [gameover.cpp](https://github.com/kz04px/libataxx/blob/master/src/gameover.cpp) and [perft.cpp](https://github.com/kz04px/libataxx/blob/master/tests/perft.cpp).
* **AI-ataxx (fffasttime):** [fffasttime/AI-ataxx](https://github.com/fffasttime/AI-ataxx) — High-speed x86-64 assembly bitboard implementation utilizing BMI2 instructions (`pext`, `popcnt`) for accelerated movegen and capture propagation.
* **Avataxx:** Classical alpha-beta searcher with bitwise ASM acceleration.
* **Cvazz (C++):** [Jigglypuffed/Cvazz](https://github.com/Jigglypuffed/Cvazz/blob/main/src/core/Move.cc) — Compact move representation and bitboard logic.
* **Anthraxx:** [Vast342/Anthraxx](https://github.com/Vast342/Anthraxx/compare/tt-cutoff) — TT cutoffs and move ordering.
* **ataxx-dot-cpp:** [GediminasMasaitis/ataxx-dot-cpp](https://github.com/GediminasMasaitis/ataxx-dot-cpp) — Hand-crafted evaluation [evaluation_hce.cpp](https://github.com/GediminasMasaitis/ataxx-dot-cpp/blob/main/src/evaluation_hce.cpp) and [changelog.md](https://github.com/GediminasMasaitis/ataxx-dot-cpp/blob/main/changelog.md).
* **Git Versioning Automation:** [tsoj/Moonbird version.nim](https://github.com/tsoj/Moonbird/blob/main/src/version.nim) — Extracts git commit hash and tags at compile time for build traceability.
* **Weather Factory Ataxx:** [tsoj/weather-factory-ataxx](https://github.com/tsoj/weather-factory-ataxx) — SPSA and Texel parameter optimization for heuristic margins.

---

## 4. What Are "N-Tuples" / "2-Tuples" in Board Game Evaluation?

In discussions of engines like *Moonbird* or *Othello* AI engines, developers frequently reference **"2-tuples"** or **"n-tuples"**:

### 4.1 Concept
An **n-tuple** is a predefined geometric sub-pattern of $n$ board squares (for example, a pair of adjacent squares = 2-tuple, a $2 \times 2$ block = 4-tuple, or a 7-square row/column = 7-tuple).

1. Each square in the tuple can be in one of $K$ states:
   * Empty ($0$)
   * Friendly Piece ($1$)
   * Opponent Piece ($2$)
   * Blocker ($3$)
2. An $n$-tuple of size $n$ has $K^n$ possible permutations. For a 2-tuple with 3 states:
   $$3^2 = 9 \text{ states per pair}$$
   For a 4-tuple with 3 states:
   $$3^4 = 81 \text{ states per quad}$$
3. **Lookup Table Evaluation:** Instead of evaluating features via matrix-vector multiplication, an n-tuple system maps the exact integer configuration index directly to a pre-tuned scalar evaluation score:
   $$\text{Score} = \sum_{\tau \in \text{Tuples}} \mathbf{W}_{\tau}[\text{index}(\tau)]$$
   This yields ultra-fast, zero-multiplication static evaluation ($>10 \text{ Mnps}$).

### 4.2 Comparison: N-Tuples vs. NNUE
| Property | N-Tuple Networks | NNUE (HalfKP / TNN) |
| :--- | :--- | :--- |
| **Feature Representation** | Exact joint configuration of $n$ specific squares | Independent `(square, piece)` features |
| **Inference Cost** | Table lookups and additions only | Incremental accumulator add/sub + dense output layer |
| **Generalization** | Memorizes specific local patterns | Generalizes across complex global board interactions |
| **Max Capacity** | Exponential in $n$ ($K^n$ RAM limit) | Linear in feature count, flexible hidden width $W$ |

---

## 5. Karpathy's Neural Net Training Recipe Applied to Game Engines

Reference: [A Recipe for Training Neural Networks (Andrej Karpathy, 2019)](http://karpathy.github.io/2019/04/25/recipe/)

### Step 1: Understand the Data
* Audit distributions: Check the balance of wins, losses, draws, and centipawn labels.
* Detect anomalies: Zero evaluations, corrupt FENs, or duplicate hash keys across different games.
* Verify input encoding: Display board features visually from the binary tensor to confirm piece bitboards align with coordinate systems.

### Step 2: Set Up the End-to-End Skeleton + Overfit a Single Batch
* Build the full training pipeline (dataloader $\to$ forward $\to$ loss $\to$ backward $\to$ step).
* Pick a tiny sample batch ($100$ positions) and train until the loss reaches $\approx 0.0$ (or accuracy reaches $100\%$).
* **Golden Rule:** If a model cannot overfit a single batch, it has a bug in its feature indexing, gradients, or loss computation. Never proceed to training on millions of positions until single-batch overfitting succeeds.

### Step 3: Regularize Only After Capacity is Proven
* Only introduce weight decay (AdamW $L_2$ decay), feature dropout, or quantization clipping after verifying that the model achieves high capacity on the validation set.
* Avoid early underfitting: A small network that trains with high regularization will underfit the subtle tactical signals of deep search labels.

---

## 6. Neural Network Distillation & Downscaling Roadmap

### 6.1 Downscaling Analogy: Image Resizing vs. Neural Network Compression
When resizing an image from $1024 \times 1024$ to $256 \times 256$, spatial pooling (bilinear/bicubic interpolation) computes a weighted average of neighboring pixels. Can we do the same with an NNUE network?

1. **Spatial Board Pooling (7x7 Grid):**
   * Unlike chess (where knights and bishops have non-local moves), Ataxx piece interactions are strictly spatial:
     * Clones are Chebyshev distance 1 ($3 \times 3$ neighborhood).
     * Jumps are Chebyshev distance 2 ($5 \times 5$ neighborhood).
   * Feature weights corresponding to neighboring squares share spatial correlation. Spatial pooling over the 49 board squares allows compressing spatial feature representations.
2. **Channel Downsampling via Truncated SVD / PCA:**
   * In NNUE, the Feature Transformer weight matrix $\mathbf{W}_{\text{FT}} \in \mathbb{R}^{F \times W_{\text{big}}}$ can be factorized:
     $$\mathbf{W}_{\text{FT}} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
   * Keeping only the top $k = W_{\text{small}}$ singular values yields the mathematically optimal lower-dimensional linear projection with minimal Frobenius norm reconstruction error:
     $$\min_{\text{rank}(\hat{\mathbf{W}}) = k} \|\mathbf{W}_{\text{FT}} - \hat{\mathbf{W}}\|_F$$
3. **Twin-Channel Pooling (Average / Max Pooling):**
   * In pairwise architectures ($W=1024 \to 512$ pairs), average adjacent twin channels to initialize a $W=256$ student, followed by 1–2 epochs of fine-tuning.

### 6.2 Knowledge Distillation (KD) with Soft Logits
Instead of training a small student network on noisy game outcomes ($z \in \{0, 1\}$), train it to match the continuous probability distribution of a large teacher model ($W=1024$ NNUE or KataGo 20-block MCTS):

$$\mathcal{L}_{\text{KD}} = (1 - \alpha) \mathcal{L}_{\text{target}} + \alpha T^2 \cdot D_{\text{KL}}\left(\sigma\left(\frac{y_{\text{teacher}}}{T}\right) \;\parallel\; \sigma\left(\frac{y_{\text{student}}}{T}\right)\right)$$

* **Temperature Scaling ($T=2.0$):** Softens the teacher's probability distribution, exposing "dark knowledge" (e.g. revealing whether a non-best move is a plausible alternative vs. an instant loss).
* **Exponential Loss:** Using $\left|\sigma(y_{\text{student}}) - \sigma(y_{\text{teacher}})\right|^{2.5}$ penalizes tactical blunders severely while smoothing noise around near-drawn states.

### 6.3 Search Depth vs. Evaluation Speed Trade-Off
In alpha-beta search with Transposition Tables, tree depth scales with node throughput (Mnps):
* **Large Model ($W=1024$):** ~0.35 Mnps, Depth ~9 at 100ms.
* **Distilled Student ($W=256$):** ~1.20 Mnps (**3.5x faster**), Depth ~12 at 100ms (**+2 to +3 plies**).
* **Micro Model ($W=128$):** ~2.40 Mnps (**7x faster**), Depth ~14 at 100ms (**+4 plies**).

In tournament play with short time controls, each additional ply of search depth provides far greater Elo gain (+40 to +60 Elo per ply) than the marginal evaluation accuracy gained from an oversized feature layer.
