# 2-Stage Tuple Architecture for Ataxx Evaluation

## The Idea

Replace the current flat NNUE feature groups (G1-G6) with a **hierarchical 2-stage tuple** that mirrors Ataxx game mechanics:

### Stage 1: 4x4 Legal Move Region (per piece)

For each piece on the board, examine its **4x4 neighborhood**. This captures the legal moves available to that piece:
- **Clone moves** (distance 1): any empty square in the 3x3 ring
- **Jump moves** (distance 2): any empty square in the outer ring of the 4x4

Each piece's 4x4 region is processed independently. The stage-1 output is a representation of "what this piece can do."

### Stage 2: 3x3 Landing Impact (per landing square)

For each legal landing square, apply a **3x3 tuple** that captures what the landing affects:
- **Infection zone**: adjacent pieces (distance 1) flip to the landing color
- **Chain effects**: the newly flipped pieces may create new infection opportunities
- **Board state change**: what the board looks like after the move

The stage-2 output is a representation of "what this landing does to the board."

### The Connection

The two stages are connected by the **legal move relationship**: a piece at square X can move to square Y, and landing at Y flips adjacent pieces. This is the game rule that ties the stages together.

## Why This Maps to Ataxx Mechanics

| Game mechanic | Distance | Captured by | Stage |
|---|---|---|---|
| Clone move | 1 | 4x4 region around piece | Stage 1 |
| Jump move | 2 | 4x4 region around piece | Stage 1 |
| Infection (adjacent flip) | 1 | 3x3 around landing square | Stage 2 |
| Chain infection after jump | 2+ | 3x3 cascade | Stage 2 |

The current NNUE uses flat feature groups (G1-G6) computed independently per square. This proposal makes the network **implicitly learn the game dynamics** through hierarchical composition instead of hand-coding them.

## Architecture Sketch

```
Input: 7x7 board (49 squares)
  │
  ▼
Stage 1: 4x4 per piece
  - For each piece, extract 4x4 neighborhood
  - Process through small transformer (attends to all squares in 4x4)
  - Output: per-piece move representation
  │
  ▼
Stage 2: 3x3 per landing square
  - For each legal landing, extract 3x3 neighborhood
  - Process through small transformer (attends to all squares in 3x3)
  - Output: per-landing impact representation
  │
  ▼
Combine: sum over all pieces/landings → final eval
```

## Comparison to Current NNUE

| Aspect | Current NNUE (G1-G6) | Proposed 2-Stage Tuple |
|--------|---------------------|----------------------|
| Feature extraction | Independent per square | Hierarchical (piece → landing) |
| Game dynamics | Hand-coded (G5 captures 2-ply exchange) | Learned through attention |
| Receptive field | 5x5 (G1+G2+G3) | 4x4 (stage 1) + 3x3 (stage 2) |
| Move legality | Implicit (G2 anchor = landing square) | Explicit (stage 1 finds legal moves) |
| Infection modeling | Implicit (G5 reply capture) | Explicit (stage 2 3x3 infection zone) |

## What We Have

- **16-bucket phase map** in `search.rs` — already separates opening/mid/endgame
- **Training pipeline** (`train_nnue.py`) — can train any architecture
- **Gate tools** (`gate_net_vs_main.py`) — can prove strength vs main
- **Feature dumper** (`nndump`) — can extract features for any architecture

## What We'd Need

1. **New feature extractor** in `nnfeat.rs` — replace G1-G6 with 2-stage tuple
2. **Re-dump features** — regenerate `train.idx` / `val.idx` with new architecture
3. **Retrain** — train new net on new features
4. **Gate vs main** — prove it's stronger before any merge

## Assessment

**Feasibility:** High. 7x7 = 49 squares is tiny for transformer compute. The 4x4 and 3x3 receptive fields are small enough that attention can be computed efficiently.

**Risk:** The current NNUE is already strong (+28.6 Elo at 5k over previous net). A full architecture change needs a gate vs main to prove it's better.

**Key question:** Does the hierarchical composition capture game dynamics better than flat feature groups? Only a gate can answer this.

## Formalization: Tree-Squashed Tactical Graph (TSTG)

The informal 2-stage concept has been mathematically formalized in [`docs/TSTG_FOUNDATIONS.md`](file:///C:/gitProjects/titanium-ataxx/docs/TSTG_FOUNDATIONS.md) (Decision D384). It rigorously derives:
1. **Closed-form 3-ply horizon collapse** via decoupled rewrite operators (clones as idempotent local rewrites, jumps as decoupled target rewards + origin craters, 3-ply counter-strike ring bounded by $\mathcal{B}_4(t) \cup \mathcal{B}_4(s)$).
2. **Dynamic action-conditioned low-rank tensor factorization** separating morphological basis ($3 \times 3$ grid, $\phi \in [0, 6560]$), 12-bit tactical reachability vector, and spatial interaction tensor $\mathbf{W}_{\text{spatial}} \in \mathbb{R}^{49 \times 49}$.
3. **$D_4 \times \mathbb{Z}_2$ symmetry orbit compression** into 10 canonical orbits, reducing parameter size by $5\times$.
4. **"Anchor-and-Gather" Spatial-Mixer with Pairwise SCReLU:** Token channels $\mathbf{X} \in \mathbb{R}^{49 \times D}$ split into **Anchored Even Channels** $\mathbf{X}_{\text{even}} \in \mathbb{R}^{49 \times D/2}$ (un-mixed, 100% localized at destination $u$) and **Spatial Odd Channels** $\mathbf{X}_{\text{odd}} \in \mathbb{R}^{49 \times D/2}$ ($\mathbf{H}_{\text{odd}} = \mathbf{W}_{\text{spatial}} \mathbf{X}_{\text{odd}} + \mathbf{B}_{\text{odd}}$). Pairwise SCReLU $\mathbf{Z}_k(u) = \text{SCReLU}(\mathbf{X}_{\text{even}}(u, k)) \odot \text{SCReLU}(\mathbf{H}_{\text{odd}}(u, k))$ strictly couples local destination state with remote origin safety while **cutting the $49 \times 49$ MatMul FLOPs in half** ($\approx 60\text{ ns}$ in AVX2, $\ge 100\text{k}$ NPS).
5. **Turn operator as conditioning field** $\mathbf{e}_\tau$ resolving zugzwang and latent agency naturally.
6. **Second-order tactical phenomena:** Hostage evacuation duality, conversion elasticity (fortress sealing vs re-infection via 24-instruction carry-save bitboard counters), self-covering assassination jumps, and starvation horizons.
7. **Zero-cost policy readout:** Dual origin-destination decomposition for move ordering ($\mathbf{w}_{\text{clone}}^\top \mathbf{z}_t$ and $\mathbf{w}_{\text{dest}}^\top \mathbf{z}_t + \mathbf{w}_{\text{orig}}^\top \mathbf{z}_s$) from the same shared representations.
8. **Square-Based Jump Deduplication & Clone Dominance:** 
   - **Destination Invariance:** Multiple pieces jumping to the same square $t$ produce the *identical* local infection $\mathcal{N}_1(t) \cap T$. Evaluating separate $3 \times 3$ tuples for each jump to $t$ is completely redundant. The landing rewrite is evaluated **exactly once** per target square $t \in E$.
   - **Decoupled Weakness Accounting:** The unique risk of a jump is the ghost crater left behind at origin $s$. This liability is evaluated **once** per origin square $s \in U$ via the `Ghost_U` bit and the topology index of $s$.
   - **Clone Dominance vs. Hostage Evacuation:** When $t \in \text{Clone\_U} \cap \text{JumpTargets}$, a clone ($+1$ stone, $0$ hole) strictly dominates any normal jump ($+0$ stone, hole at $s$), with the sole exception of an evacuation jump rescuing a stranded hostage at $s$ ($\text{OrigScore}[s] > 0$).
9. **Zero-Cycle Stage 1 (`topo_index`) & Strict STM Normalization:**
   - Maintain `topo_index: [[u16; 49]; 2]` incrementally during `make_move` (updating only the $\le 9$ neighbors of the $1\text{–}9$ changed squares via `DELTA_LUT`), making Stage 1 cost **0 CPU cycles** inside `eval()`.
   - Always normalize perspective to Side-to-Move (`U = pos.us()`, `T = pos.them()`).

**Next step:** Prototype the incremental `topo_index`, 10-orbit canonical topology LUT, and 12-bit carry-save tactical preprocessor in `crates/titanium/src/tstg.rs` and benchmark cycle latency.
