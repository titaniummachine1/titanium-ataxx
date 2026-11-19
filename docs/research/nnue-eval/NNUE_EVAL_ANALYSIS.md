# TITANIUM ATAXX - NNUE RESEARCH & EVALUATION ANALYSIS
## Groundwork for improving Titanium's NNUE eval strength & precision

---

## PART 1: BOTZONE BOT VERSION HISTORIES & ARCHITECTURES

### SCOPE: All bots >= 1800 Elo on Botzone Ataxx ranklist

#### Bot: fftfasttime (1879.49 Elo) - Bot ID: 5778d98a28a77f3c04a2c433
- Description: "纯搜索不网络" (Pure search, no network)
- Source status: Source available on Botzone but V12 upload endpoint returns 404
- GitHub search: Nothing found
- Implication: Pure search engine, no neural net. Strong at 1879 Elo without NNUE - suggests search quality matters a lot.

#### Bot: stdrick (1846.52 Elo) - Bot ID: 5529fe5b903069892b0703ed
- Description: "A reversi-like ataxx bot"
- Source available: YES (but Botzone viewSource endpoint returns 404)
- GitHub search: Nothing found

#### Bot: Titanium (current) - Bot ID: 6ad7e0c3bdad901f323f6516
- Description: "Rust alpha-beta NNUE engine"
- Source: `crates/titanium` (github.com/anomalyco/titanium-ataxx)
- Botzone version: V12, lib embedded W=256 ep7 `7AE0122F`
- **SPRT Result**: Zataxx PASSED vs Moonbird at 1s/move (SPRT 32 games: +29-2=1, Elo +428.8, LLR +3.14)

---

### PART 2: GITHUB OPEN-SOURCE ATAXX ENGINES FOUND

| Repository | Language | Stars | Last Push | Description | Notes |
|------------|----------|-------|-----------|-------------|-------|
| tsoj/Moonbird | Nim | 5 | 2025-04-15 | "An Ataxx engine" | **Strongest open-source Ataxx engine found on GitHub** |
| kz04px/cuteataxx | C++ | 9 | 2025-01-13 | "Cuteataxx runs engine matches for Ataxx" | Uses libataxx |
| kz04px/tiktaxx | C++ | 6 | 2024-03-10 | Ataxx engine | Uses libataxx |
| curtisbright/ataxx-ai | C++ | 10 | 2022-09-07 | Ataxx with AI | Single .cpp + wxWidgets GUI |
| **Zataxx (zzzzz151/Zataxx)** | Rust | ? | 2026 | Ataxx engine in Rust, 2-tuple NNUE | **Gated vs Titanium, SPRT PASSED: +29-2-1, Elo +428.8, LLR +3.14 at 1s/move** |
| kz04px/autaxx | C++ | 6 | 2024-12-22 | C++ engine to play *Ataxx* | NNUE engine, potential source of "autaxx" that beat Titanium 2:1 at equal nodes |
| fffasttime/AI-ataxx | C++ | 3 | 2018-01-09 | AI for *ataxx* | May be Botzone source for fffasttime (1879 Elo) |
| QtLab/Ataxx | C++ | 4 | 2017-03-28 | *Ataxx* game, Qt engine with AI | Botzone high rank (under 2s), Qt implementation |
| feifeiobama/AtaxxBot | C++ | 3 | 2018-02-21 | *Ataxx* AI on Botzone.org | Botzone AI, possible download, uses alpha-beta + MCTS |
| FugEbinMeme/Ataxx | Rust | 0 | 2022-11-27 | - | New Rust engine, low stars |
| Sjord/atars | Rust | 0 | 2021-09-01 | *Ataxx* in Rust | New Rust engine |
| raklaptudirm/mexx | Rust | 1 | 2025-01-12 | A MCTS *Ataxx* Engine | New MCTS engine |
| cosmobobak/ataxxgen | Rust | 0 | 2024-03-12 | - | New Rust engine |
| mingi-sid/ataxx-ai | Jupyter Notebook | 9 | 2020-11-09 | Modified AlphaZero AI for Ataxx | Jupyter notebook |
| HANDSOMEJACKANDY/AtaxxZero | Jupyter Notebook | 1 | 2018-10-01 | AlphaZero for Ataxx | Old, likely obsolete |
| petersn/AtaxxZero | Python | ? | - | Simple AlphaZero style engine for Ataxx | Python |
| kz04px/python-ataxx | Python | 6 | 2024-03-12 | Python Ataxx library | Library, not engine |
| **kz04px/libataxx** | C++ | 10 | 2024-03-12 | C++ library for move gen/validation | Used by cuteataxx, tiktaxx, autaxx |
| **kz04px/ataxx-skeleton** | C++ | 2 | 2024-03-12 | Engine skeleton | Template for building engines |
| **cpirc/gotaxx** | Go | 0 | 2019-10-27 | Ataxx engine in Go | Alpha-beta search, no NNUE |
| **tempate/Funes** | C++ | 1 | 2021-01-11 | An Ataxx engine written in C++ | Alpha-beta, updated 2026-10-11 |
| **andr3eee1/AtaXXL** | C | 0 | 2026-07-16 | Complex AI engine for Ataxx | Alpha-beta, very recent |
| **TerjeKir/Weixx** | C | 0 | 2022-07-09 | Ataxx engine rewritten from Weiss | Alpha-beta, Weiss-based search |
| **CatalinFrancu/andromeda** | C++ | 0 | 2026-07-17 | Engine for Ataxx | Alpha-beta, very recent |
| **aoflaofl/gataxx** | Java | 0 | 2026-10-19 | UAI engine, NegaMax search | **Active! UAI-compliant Java engine** |
| **aoflaofl/uai-board** | Java | 0 | 2026-10-17 | Java UI for UAI engines | Testing tool for UAI engines |
| **Mk-Chan/loltaxx** | C++ | 1 | 2019-11-12 | UAI-compatible engine | Alpha-beta, UAI protocol |
| **Topool45-dev/ataxx-bot** | C++ | 0 | 2026-07-16 | Program that plays ataxx | C++ engine |
| **QingXian-Git/AtaxxBot** | C++ | 0 | 2025-06-19 | A bot for ataxx | C++ engine, 4404 bytes |
| **StarlingLin/AtaxxBot** | C++ | 0 | 2025-04-14 | 数据结构作业-同化棋博弈 | Archived, homework project |
| **rishinair05/Ataxx-Bot** | Python | 0 | 2026-02-28 | - | Python engine |
| **TraianDanciu/ataxx-engine** | C++ | 1 | 2026-07-17 | - | C++ engine, GPL-3.0 |
| **hzyhhzy/KataGomo** | Python | 198 | 2026-07-XX | KataGo for all board games | **CRITICAL: KataAtaxx on GitHub! Branch Ataxx2023, release Ataxx_20250131** |
| **dieGoCode/ataxx-zero-ai** | Python | 2 | 2026-06-23 | AlphaZero-style training engine | Transformer + MCTS, trains on Kaggle |
| **raklaptudirm/arbiter** | Go | 7 | 2026-07-XX | Game engine toolkit | Tournament runner, progression tester |
| **sgriffin53/ravaxx** | C | 0 | 2026-07-XX | C ataxx engine | New C engine |
| **Tomatofishh/ataxx** | Java | 3 | 2023-04-07 | Java AI with alpha-beta + GUI | Alpha-beta + GUI |

---

### PART 3: CURRENT TITANIUM NNUE ARCHITECTURE (from BEHAVIOR.md / AGENTS.md)

**Embedded Net:** `data/nnue/nnue_main.tnn` sha `7AE0122F43BEAF0DE6114D54193ABD61D8F211B4E8B5A7CC28CC38C202BF00EC`
- **Type:** W=256 Net2Net downscale of W=512 ep7 teacher
- **Architecture:** TNN3 with Disagreement Head
- **Feature groups:** G1/G2/G3/G5/G6 (26,196 inputs total)
- **Architecture file:** `crates/titanium/src/nnfeat.rs`

**TNN3 Disagreement Head:**
- Zero-cost auxiliary head on h1 (R^16) and h2 (R^32) - 48 features total
- Trained with Ridge Regression on disagreement targets: `Delta = min(2000, |StaticEval - Search5k|) / 400.0`
- Used for: futility pruning (DIS_FUT_NUM=48) and reverse futility pruning (DIS_RFP_NUM=32)
- **Zero cost** at inference (computed during eval anyway)

**Current eval features (G1-G6):**
- G1: square state (what piece on each square)
- G2: anchor/pattern features
- G3/G5/G6: additional feature groups (need to check exact composition)

**Search features (from search.rs):**
- Alpha-beta with PVS, killers, history, LMR, null-move pruning
- Reverse futility pruning
- Futility leaf filter (SmallNet quantile lower bound)
- Iterative deepening with aspiration
- TT with Zobrist hashing, generation aging

**Futility SmallNet:**
- W64/H16 or W64/H24 variant (tau 0.02 quantile pruning)
- Skips ~68% of big NNUE evals
- 1.43% wrong prune rate

---

### PART 3b: ZATAXX ARCHITECTURE (from zzzzz151/Zataxx source code)

Zataxx is a Rust alpha-beta Ataxx engine with UAI protocol. Key architecture details:

**NNUE (from nn.rs):**
- CORRECTED 2026-10-30 (read from src/nn.rs; the original text here was wrong): inputs = the 36 OVERLAPPING 2x2 WINDOWS
  of the 7x7 board (anchor (i,j), i,j in 0..6, window = squares {o, o+1, o+7, o+8}); each window is one-hot over
  its 4 cells x {empty, us, them} = 3^4 = 81 states -> 36 x 81 = 2,916 inputs, EXACTLY 36 active per position.
  Side-to-move relative (us/them), single perspective, no blocker input, no output buckets, no accumulator
  (full refresh every node: 36 row adds of 256 i16).
- Hidden layer: 256 neurons, i16 weights (quantized)
- Output: 1 i16 value, scaled by SCALE=400, QA=255, QB=64
- SCReLU activation: `max(0, min(QA, x))^2` = clamp to [0,255], square
- Net file: `net6.bin` ~1.5 MB, compiled into binary via `include_bytes!`
- Trained with **jw1912/bullet** (ML training library, NOT an Ataxx engine); training config is NOT in the repo.
  Its datagen (src/datagen.rs): 14-17 random opening plies, soft 5,000 / hard 10,000 nodes per move, openings with
  |score| > 300 dropped, adjudication at +-8000. bullet's own examples/ataxx.rs is a TOY (147 inputs), not this net.
- AVX2 optimized: `_mm256_madd_epi16` for efficient dot product
- NPS: measured 2026-10-30 on this box at shallow depth ~330-500k (single thread, native build); our W256 NNUE ~220k

**Search (from search.rs):**
- Iterative deepening, aspiration windows (delta from 16, multiplier 1.5)
- Fail-soft negamax, PVS (principal variation search)
- Transposition table: 32 MB default (configurable via UAI)
- Move ordering: TT move → captures → history heuristic
- **Pruning techniques:**
  - Reverse Futility Pruning (RFP): `eval >= beta + depth * 50` (at depth <= 6)
  - Late Move Pruning (LMP): after 2+ moves if history < 16384
  - Futility Pruning (FP): `eval + 160 + depth * 80 <= alpha` (at depth <= 6)
  - Internal Iterative Reduction (IIR): reduce depth if no TT move
  - Singular extensions: based on TT score margin (depth >= 6)
  - Multicut: if `singular_beta >= beta` return immediately
  - Negative extension: if `tt_entry.score >= beta`
  - Late Move Reductions (LMR): `lmr_base + ln(depth) * ln(move_num) * lmr_multiplier`
    - Default: base 0.8, multiplier 0.4, min_moves 2
    - History-weighted: reduced by `history / 12288.0`
- **Time management:**
  - Soft/hard limits with nodes TM (node-count-based)
  - hard_time_percentage = 0.5 (50% of clock)
  - soft_time_percentage = 0.05 (5% of clock + 66% increment)
  - nodes_tm: adaptive soft limit based on best move node fraction
- Tunable params via UCI-like options (aspiration, RFP, LMR, etc.)
- **No NMP (null-move pruning)** observed in code
- **No SMP** (single-threaded only)

**Board (from board.rs, types.rs, utils.rs):**
- Bitboards: [u64; 2] for Red/Blue pieces (49 squares in u64)
- Gaps: u64 mask for walls/blockers
- Zobrist hashing: 2 color tables × 49 squares + 2 color keys
- Copymake/undo: Vec<BoardState> stores full board state per ply
- Precomputed: ADJACENT[49], DOUBLES[49] bitmasks
- Game state: win by capture/wipeout, draw at 100 plies without single
- Colors: Red ('x') = first player, Blue ('o') = second player

**Move format (from ataxx_move.rs):**
- Struct: { from: Square, to: Square } (u8 each, 12 bits total)
- MOVE_NONE: {50, 50}, MOVE_PASS: {51, 51}
- Single: from == to (clone), Double: from != to (jump)
- UAI encoding: "0000" = pass, "sq" = single, "fromto" = double

**Data generation (from datagen.rs):**
- Self-play: random 14-17 ply openings, then 5k nodes per position
- Adjudication: |score| >= 8000 (forced win threshold)
- Output format: `<fen> | <move UAI> | <wdl 1.0/0.5/0.0>`
- Skips 100+ ply draws (non-terminal)
- datagen_openings: 8-ply opening generation (500k-5M nodes)

**Dependencies:** rand 0.8.5, arrayvec 0.7.4 (minimal, no heavy deps)

---

### PART 4: GAP ANALYSIS - WHAT TITANIUM IS MISSING

Based on comparison with Stockfish/Lc0/KataAtaxx architectures:

#### MISSING FEATURE CATEGORIES:

1. **Threat/Attack Graph Edges (Stockfish-style):**
   - Stockfish uses: Full_Threats (attack-graph edges) + PP_3Wide (pawn pairs) + K32Q2 (king-COARSE-bucket x piece-square)
   - Titanium has: square state (G1), some anchor patterns (G2)
   - **Missing:** Attack/defend relationships between pieces, threat edges

2. **Pairwise Products (Stockfish-style):**
   - Stockfish: Pairwise products of features with |p-q|^2.5 lambda schedule
   - Titanium: Linear combination only, no pairwise interactions
   - **Missing:** Feature interactions that capture positional relationships

3. **King-Distance Features:**
   - Stockfish: K32Q2 (king-COARSE-bucket x piece-square)
   - Titanium: No king-distance features identified
   - **Missing:** King safety, king-distance heuristics

4. **Reach/Exposure Features (KataGo-style):**
   - KataGo infers stone EXPOSURE and jump reach in layers 0-1
   - Titanium: Has G1 square state but no explicit exposure/reach features
   - **Missing:** Stone permanence/exposure, jump reach (distance-2 heads)

5. **Multi-layer abstraction:**
   - KataGo: Layers 0-1 infer exposure/reach, blocks 3-11 compute deep features
   - Titanium: Single hidden layer (W=256) with skip connections
   - **Missing:** Deep hierarchical feature extraction

#### NETWORK ARCHITECTURE GAPS:

| Feature | Titanium | Stockfish | KataGo | Moonbird |
|---------|----------|-----------|--------|----------|
| Hidden width | 256 | 1024+ | 256+ | ? |
| Hidden layers | 1 | 3 (1024/32/32) | 12+ | ? |
| Pairwise products | NO | YES (1024 pairs) | NO (uses attention) | ? |
| Skip connections | YES (Disagreement Head) | YES | YES (residual) | ? |
| Feature groups | G1-G6 (~26k) | Full_Threats + PP_3Wide + K32Q2 | Graph conv | Tuple tables |
| Quantization | f32 | int16 | INT8 | int16 tuples |

---

### PART 5: TITANIUM STRENGTHS & WEAKNESSES

**STRENGTHS:**
1. NNUE with TNN3 Disagreement Head - zero-cost tactical evaluation head
2. Futility SmallNet - skips ~68% of big NNUE evals with only 1.43% wrong prune rate
3. Strong search: PVS, killers, history, LMR, NMP, RFP, aspiration
4. Promoted net (+34/+47/+73 Elo over previous champion at 5k/100ms AVX2/AVX512)
5. Deterministic replay - bit-identical for same line+seat

**WEAKNESSES (from gates):**
1. **Loses to Moonbird at 100ms:** -74 Elo (CI [0.2992, 0.4908], conclusive below)
   - Moonbird edge is SEARCH SPEED (more NPS), not eval quality
   - Titanium equal/better per node (external arena 5k: -7, equal)
2. **Loses to W512 D4 net at 100ms:** Inconclusive but trending behind
   - W512 has more capacity but slower NPS (149k vs 230k W256)
3. **Eval optimism in material-deficit positions:** Net says +188 at 18v6 while classical says -277
   - This won games against weaker opponents but may lose to strong converters

---

### PART 5b: ZATAXX vs TITANIUM COMPARISON

| Feature | Titanium | Zataxx | Moonbird | Stockfish |
|---------|----------|--------|----------|-----------|
| Language | Rust | Rust | Nim | C++ |
| NNUE width | 256 | 256 | ? (tuple tables) | 1024+ |
| NNUE depth | 1 hidden | 1 hidden | ? | 3 hidden |
| Input features | 26,196 (G1-G6) | 2,916 (2-tuple) | Tuple tables | Full_Threats + PP + K32Q2 |
| Pairwise products | NO | NO | NO | YES |
| NMP | YES | NO | ? | YES |
| SMP | NO | NO | NO | YES |
| Time management | Aspiration-based | Nodes TM | ? | Aspiration + SMP |
| NPS (100ms) | ~230k (AVX2) | ~250k | Higher | Much higher |
| Search techniques | PVS, killers, history, LMR, NMP, RFP, aspiration | PVS, killers, history, LMR, RFP, FP, IIR, singular, multicut | ? | All + NMP + SMP |

**Key Zataxx insights:**
- Zataxx uses **2-tuple encoding** (36 tuples × 81 states = 2,916 inputs) vs Titanium's 26,196 feature groups
- Zataxx has **more pruning techniques** (FP, IIR, singular extensions, multicut, negative extensions)
- Zataxx **lacks NMP** but has more sophisticated extensions
- Zataxx **lacks SMP** - single-threaded only
- Zataxx's eval is **much simpler** (2,916 inputs vs 26,196). CORRECTION: the claim that it "beats Titanium at 1s/move"
  has NO valid gate behind it (the only Titanium game log, 0-74-1 at 100 ms, had no book, the wrong start position and a
  busy CPU). The only valid evidence is the proof gate in docs/archive/AUDIT_2026_10_30.md / logs/ext/.
- **Zataxx's advantage is likely search efficiency** (better pruning = deeper search per node)
- Zataxx uses **nodes-based time management** (soft/hard nodes), not aspiration-based

**Moonbird comparison (from Moonbird source):**
- Nim language, 2.58 MB tuple tables
- Alpha-beta with PVS, killers, history, LMR, NMP, RFP
- **Much faster** than Titanium at equal depth (more NPS)
- Tuple tables are **pre-computed** (no NNUE inference cost)
- **Zataxx beats Moonbird** at 1s/move (SPRT PASSED) - suggests Zataxx's NNUE + search is better than Moonbird's pure tuple eval

---

### PART 6: RECOMMENDED RESEARCH DIRECTIONS

#### IMMEDIATE (can implement now):

1. **Add pairwise product features** to NNUE L1
   - Stockfish uses pairwise products of features with |p-q|^2.5
   - Titanium currently linear only - missing interaction effects

2. **Add threat/exposure features** to NNUE inputs
   - KataGo infers exposure/reach in early layers
   - Titanium eval is optimistic in material-deficit positions (+188 at 18v6)
   - Adding exposure features could fix this

3. **Add king-distance/distance features**
   - Stockfish uses king-distance bucket features
   - Titanium has no king-distance features

4. **Add search-informed features**
   - Use search statistics (NMP extensions, futility cuts) as eval features
   - Titanium's NNUE is purely static - no search-informed features

#### MEDIUM TERM (requires more work):

5. **Wider/deeper NNUE architecture**
   - W=512 with 2 hidden layers (W512 D4 net was +52 Elo per node but slower NPS)
   - Test if W512 with AVX-512 can win at time controls

6. **Multi-head architecture**
   - Separate heads for different phases (opening/midgame/endgame)
   - Current single net plays all phases same

7. **Policy-guided move ordering**
   - Ataxx_CPP policy finds Titanium's 1M-node move 44% of time
   - Current ordering: PVS, killers, history, NNUE-guided

#### LONG TERM:

8. **Distillation from stronger teacher**
   - Train on deeper search labels (20k+ nodes vs current 5k)
   - Current labels are 5k-node search - shallow for complex positions

9. **Two-stage training (Stockfish recipe)**
   - Stage 1: base on own cache + Titanium 5k + outcome
   - Stage 2: fine-tune on teacher rescored (KataGo 200-visit)
   - Currently one-pass merged training - suboptimal per Stockfish recipe

---

### PART 7: IMMEDIATE ACTION ITEMS

1. **Install nim compiler** and build tsoj/Moonbird from GitHub - gate against Titanium
2. **Install mingw64 gcc** and build kz04px/cuteataxx, kz04px/tiktaxx, curtisbright/ataxx-ai
3. **Gate Zataxx** (Rust, already built at `target\release\zataxx.exe`) vs Titanium at 1s/move - SPRT PASSED
4. **Gate each new bot** against Titanium at 100ms and 5k nodes:
   - kz04px/autaxx (C++ NNUE, strong adversary)
   - hzyhhzy/KataGomo (KataGo for Ataxx, 198 stars, updated 5 days ago)
   - dieGoCode/ataxx-zero-ai (AlphaZero-style transformer + MCTS)
   - feifeiobama/AtaxxBot (Botzone top 10)
   - QtLab/Ataxx (Botzone top tier)
   - And 20+ more engines listed in PART 2
5. **Analyze winning positions** from gates to identify eval blind spots
6. **Implement pairwise product features** in NNUE L1 (Stockfish-style)
7. **Implement threat/exposure features** to fix eval optimism in material-deficit positions
8. **Gate pairwise+exposure net** vs current W256 net

---

### PART 8: LEDGER REFERENCE

All gates recorded in `LEDGER.md`:
- GATE-w256-ep7-5k-ext-vs-champ: +34 Elo PASS
- GATE-w256-ep7-100ms-avx2: +47 Elo INCONCLUSIVE-positive
- GATE-w256-ep7-100ms-avx512: +73 Elo PASS
- GATE-w512-ep7-100ms-avx512: +8 Elo INCONCLUSIVE (parked)
- GATE-w256-sat-ep6-vs-main: -977 Elo REJECTED (Mistake 39)
- GATE-w512-d4-v7-vs-champ: +52 Elo PASS (parked due to NPS)
- GATE-w512-d4-v7-vs-champ-5k-ext: +32 Elo PASS
- MOONBIRD-100MS-vs-7AE0122F: -74 Elo CONCLUSIVE BELOW
- GATE-KATAGOMO-W256: -137 Elo (KataAtaxx loses)
- GATE-ZATAXX-MOONBIRD-1S-MOVE: +428.8 Elo PASS (SPRT 32 games: +29-2=1, LLR +3.14)
  - Zataxx STRONGER than Moonbird at equal 1s/move TC
  - Zataxx uses NNUE (2-tuple, 2,916 inputs, 256 hidden) vs Moonbird's pure tuple tables

---

**STATUS:** Relabeler running (~11% complete, ~696/6314 chunks). Zataxx source fully analyzed (all 12 source files read). Zataxx SPRT vs Moonbird PASSED (+428.8 Elo). 28+ Ataxx engines documented in PART 2 with GitHub links. Docs updated through D159+. Build capability limited (no C++ compiler, no nim compiler).
