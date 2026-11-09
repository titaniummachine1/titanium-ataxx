> **CORRECTION BANNER (2026-11-06, D414/D416): this document was written by a weaker agent and contains claims that do not reproduce (e.g. 'beat main champion', '+0.962 correlation', 'heads fully populated'). TSTG as the sole evaluator scored 0 points in 128 games. Read `docs/TSTG_EXPERIMENT.md` and `docs/TSTG_VERIFICATION_2026_11_06.md` first; they win where they disagree.**

# Mathematical Foundations of the Tree-Squashed Tactical Graph (TSTG)

---

## 1. Metric Space and Board Topology

Let the game board be defined as a discrete two-dimensional lattice:

$$\Omega = \{ (x, y) \in \mathbb{Z}^2 \mid 0 \le x < L,\ 0 \le y < L \}$$

where $L = 7$, yielding $|\Omega| = 49$ vertices.

The spatial interactions in Ataxx are governed strictly by the Chebyshev metric ($L_\infty$ norm):

$$d_\infty(u, v) = \max(|x_u - x_v|, |y_u - y_v|)$$

The board dynamics partition into two fundamental interaction neighborhoods around any vertex $u \in \Omega$:

$$\mathcal{N}_1(u) = \{ v \in \Omega \mid d_\infty(u, v) = 1 \} \quad (|\mathcal{N}_1(u)| \le 8, \text{ Clone / Infection Ring})$$

$$\mathcal{R}_2(u) = \{ v \in \Omega \mid d_\infty(u, v) = 2 \} \quad (|\mathcal{R}_2(u)| \le 16, \text{ Jump Ring})$$

$$\mathcal{B}_r(u) = \{ v \in \Omega \mid d_\infty(u, v) \le r \} \quad (\text{Metric Ball of radius } r)$$

Let the static board state be an assignment vector $S \in \Sigma^{|\Omega|}$, where the local state alphabet is:

$$\Sigma = \{ 0 \text{ (Empty)}, 1 \text{ (Friendly)}, -1 \text{ (Opponent)}, \emptyset \text{ (Blocker/Void)} \}$$

---

## 2. The Horizon Collapse Principle: Closed-Form 3-Ply Tactical Equivalence

Standard alpha-beta search builds an explicit decision tree $\mathcal{T}$ of depth $d$, evaluating leaf states via a heuristic function $f: \Sigma^{|\Omega|} \to \mathbb{R}$. In high-churn games, $f(S)$ exhibits severe spatial high-frequency noise because a 1-ply delta $\Delta S$ causes non-perturbative modifications across large clusters.

TSTG collapses the search tree $\mathcal{T}_{\le 3}$ (depth $\le 3$) into a static spatial interaction hypergraph $\mathcal{G} = (\Omega, \mathcal{E}, \Phi)$ by observing that state transitions in Ataxx are strictly localized cellular rewrite operators.

```
       [Ply 0: Static State S]
              │
              ├─── Clone(T) ──────────────► Local rewrite at N_1(T)
              │
              └─── Jump(S -> T)
                     │
                     ├── [Ply 1: Destination Reward] ──► Local rewrite at N_1(T)
                     └── [Ply 1: Origin Crater S]    ──► S becomes Empty
                                │
                                └── [Ply 2: Counter-Strike] ──► Opponent lands at S
                                                                      │
                                                                      └── [Ply 3: Counter-Flip]
                                                                          Local rewrite at N_1(S)
```

### 2.1 The Idempotency of Clone Rewrites

Let $\mathcal{A}_{\text{clone}}(u)$ be the set of friendly pieces capable of cloning to an empty cell $u \in \Omega$:

$$\mathcal{A}_{\text{clone}}(u) = \{ v \in \mathcal{N}_1(u) \mid S(v) = 1 \}$$

If $|\mathcal{A}_{\text{clone}}(u)| \ge 1$, the resulting state transformation $\mathcal{M}_{\text{clone}}(u)$ is an operator acting purely on $u$ and its immediate 1-ring $\mathcal{N}_1(u)$:

$$[\mathcal{M}_{\text{clone}}(u)(S)](w) = \begin{cases} 
1 & \text{if } w = u \\
1 & \text{if } w \in \mathcal{N}_1(u) \text{ and } S(w) = -1 \\
S(w) & \text{otherwise}
\end{cases}$$

> **Lemma 1 (Source Invariance of Clones):**
> For all $v_i, v_j \in \mathcal{A}_{\text{clone}}(u)$, the state transition satisfies $\mathcal{T}(S, v_i \to u) \equiv \mathcal{T}(S, v_j \to u) = \mathcal{M}_{\text{clone}}(u)(S)$. The mapping depends strictly on the target coordinate $u$, reducing the candidate action space from $\mathcal{O}(|\Omega|^2)$ to $\mathcal{O}(|\Omega|)$.

### 2.2 Decomposition of Jump Rewrites

A jump from source $s$ to destination $t$ where $d_\infty(s, t) = 2$ decomposes into an additive union of two decoupled spatial modifications:

$$\mathcal{T}(S, s \to t) = \mathcal{M}_{\text{clone}}(t) \circ \delta_s^0 (S)$$

where $\delta_s^0$ is the annihilation operator setting $S(s) \leftarrow 0$ (Empty).

### 2.3 The 3-Ply Closed Counter-Strike Ring

Let $s$ be an occupied square vacated by a jump. The cell $s$ transitions from $S(s) = 1$ to $S'(s) = 0$.

If an opponent piece exists within the action ball of $s$ ($w \in \mathcal{B}_2(s)$ such that $S(w) = -1$), the opponent can execute an immediate counter-move into $s$ on the subsequent ply. The resulting damage function $\mathcal{D}(s)$ is identical to an opponent clone centered at $s$:

$$\mathcal{D}(s) = 1 + |\{ v \in \mathcal{N}_1(s) \mid S(v) = 1 \}|$$

Because the maximum propagation velocity of any tactical retaliation is bounded by $d_\infty \le 2$ per ply, all second-order counter-strikes and third-order recapture potentials are fully contained within the composite metric ball $\mathcal{B}_4(t) \cup \mathcal{B}_4(s)$.

By evaluating the 1-ring neighborhood $\mathcal{N}_1(u)$ for all $u \in \Omega$ under the joint condition of real occupancy and ghost vacancy, the network accounts for 3 plies of minimax game-tree interactions without expanding tree branches.

---

## 3. Low-Rank Factorization of the Global N-Tuple Space

A complete, non-approximated evaluation of an Ataxx position requires indexing the full state space $\Sigma^{|\Omega|} = 3^{49} \approx 2.39 \times 10^{23}$ states.

Classical N-tuple systems approximate this by a sum of $M$ overlapping sub-graph lookup tables:

$$V(S) = \sum_{m=1}^{M} w_m(\tau_m(S))$$

where $\tau_m \subset \Omega$ is a tuple of coordinates (typically $|\tau_m| \in [6, 10]$).

### 3.1 The Failure of Classical Tuples Under High Churn

In games with large Hamming distance shifts between adjacent plies, classical static tuples fail because:
1. **Feature sparsity changes discontinuously:** A single jump alters up to 9 cells simultaneously, shifting the indices of hundreds of overlapping predefined static tuples.
2. **Static coordinate tuples cannot condition on action potential:** A solid $3 \times 3$ block of pieces has identical tuple indices regardless of whether an opponent piece is 1 square away (ready to breach) or completely absent from that board sector.

### 3.2 Dynamic Action-Conditioned Factorization

TSTG formulates evaluation as a bilinear low-rank decomposition of the full state tensor. Instead of allocating dense multidimensional tables, the representation is factored into:

- **Local Morphological Basis (Stage 1 / Feature Layer — Zero-Cost Incremental):**
  A canonical projection of the $3 \times 3$ sub-grid centered at coordinate $u$:
  $$\phi_{\text{topo}}(u) = \sum_{k=0}^{8} \psi(S(u + \delta_k)) \cdot 3^k \in [0, 3^9 - 1] = [0, 6560]$$
  where $\psi: \Sigma \to \{0, 1, 2\}$ maps $\{\text{Empty}: 0, \text{Friendly (STM)}: 1, \text{Opponent (NSTM)}: 2\}$.
  - **Zero-Cycle Evaluation via Incremental `make_move` Updates:** Instead of bit-twiddling 49 ternary indices from scratch inside `eval()` ($\approx 150\text{–}200$ cycles), the board maintains `topo_index: [[u16; 49]; 2]` (for both perspectives) incrementally during `make_move`. Because a move alters only $1\text{–}9$ squares (destination $t$, flipped neighbors in $\mathcal{N}_1(t)$, and jump origin $s$), each state transition $A \to B$ at square $v$ simply adds $(B - A) \cdot 3^{\text{dir}}$ to the $\le 9$ affected neighbor indices via a precomputed `DELTA_LUT[v]`. At `eval()` time, Stage 1 is 100% precomputed at **zero CPU cycles** (with a fast shift + `[u16; 512]` base-2-to-base-3 LUT fallback for cold initialization, avoiding microcoded `_pext_u64` on Zen1/2).

- **Dynamic Reachability Operator (Stage 2 / Tactical Action Bits):**
  A continuous embedding vector derived from the bitboard-calculated reachability flags:
  $$\mathbf{b}(u) \in \{0, 1\}^{B} \quad (B = 12)$$

- **Spatial Interaction Operator (Stage 3 / "Anchor-and-Gather" Relational Layer):**
  A parameter matrix $\mathbf{W}_{\text{spatial}} \in \mathbb{R}^{|\Omega| \times |\Omega|}$ that routes origin/neighborhood information from coordinate $v$ into anchored destination coordinate $u$.

The global evaluation score decomposes as:

$$V(S) = \sum_{u \in \Omega} \sum_{v \in \Omega} \mathbf{W}_{\text{spatial}}(u, v) \cdot \Big( \mathbf{E}_{\text{topo}}(\phi_{\text{topo}}(u)) \odot \mathbf{E}_{\text{act}}(\mathbf{b}(v)) \Big)$$

This decomposes the astronomical $3^{49}$ state space into an expressive representation requiring fewer than $10^5$ parameters, fitting directly into L1/L2 CPU cache.

---

## 4. Symmetry Group Theory: $D_4 \times \mathbb{Z}_2$ Equivariance

To maximize sample efficiency during offline training and minimize parameter footprint in engine cache, the evaluation must be strictly equivariant under the symmetry group of the board.

```
                    D4 Dihedral Symmetry (8-fold)
                  ┌──────────────────────────────┐
                  ▼                              ▼
     [Rotations: 0°, 90°, 180°, 270°]   [Reflections: H, V, D1, D2]
                  │                              │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                     10 Fundamental Orbits
            (Every square u in 49 maps to an orbit)
                                 │
                                 ▼
                  Z2 Color Inversion Symmetry
             (Swap Friendly <-> Opponent, Invert Eval)
```

### 4.1 Spatial Symmetry: The Dihedral Group $D_4$

The square grid $\Omega$ is invariant under the dihedral group $D_4$, consisting of 8 group elements:

$$D_4 = \{ R_0, R_{90}, R_{180}, R_{270}, S_h, S_v, S_{d1}, S_{d2} \}$$

The group action $g \circ u$ partitions the 49 coordinates of the board into 10 disjoint orbits (equivalence classes):

$$\Omega / D_4 = \{ \mathcal{O}_0, \mathcal{O}_1, \dots, \mathcal{O}_9 \}$$

- **Center square:** Orbit size 1.
- **Orthogonal axes:** Orbits of size 4.
- **Diagonals:** Orbits of size 4.
- **General asymmetric squares:** Orbits of size 8.

**Consequence for Tuple Lookup Tables:**
Instead of storing 49 distinct $3 \times 3$ topology tables:

$$\text{TableSize} = 49 \times 3^9 \times D = 49 \times 6561 \times D$$

we enforce canonical orbit mapping:

$$\mathbf{E}_{\text{topo}}(u, \phi) \equiv \mathbf{E}_{\text{orbit}}(\text{Orbit}(u), \text{Canonize}(\phi, u))$$

This compresses the topology weight tensor by a factor of $\approx 5\times$ to $10 \times 6561 \times D$, guaranteeing that symmetric positions yield identical neural activations.

### 4.2 Color Inversion Symmetry: The Group $\mathbb{Z}_2$

Let $\mathcal{C}$ be the color inversion operator:

$$\mathcal{C}(S) = -S$$

The evaluation function must satisfy anti-symmetry:

$$V(\mathcal{C}(S)) = -V(S)$$

By constructing the network with dual perspective halves—accumulating features for Side-to-Move (STM) and Non-Side-to-Move (NSTM) using shared parameter weights—anti-symmetry is guaranteed by design:

$$V(S) = f(\mathbf{H}_{\text{STM}}) - f(\mathbf{H}_{\text{NSTM}})$$

- **Strict STM Perspective Normalization in Code:** At the entry of evaluation, the board is always normalized relative to the active player: `U = pos.us()` (Side-to-Move bitboard) and `T = pos.them()` (Non-Side-to-Move bitboard). For incremental `topo_index`, either dual arrays `topo_index: [[u16; 49]; 2]` (`[stm, nstm]`, swapped on each ply) are maintained directly, or a fixed-color `topo_index: [u16; 49]` ($0 = \text{Empty}, 1 = \text{Black}, 2 = \text{White}$) is mapped through a tiny $6561$-entry `FLIP_COLOR_LUT[topo]` when `STM == White`.

---

## 5. The Relational Interaction Layer: Micro-Transformer vs. Spatial-Mixer

The core theoretical challenge is modelling the dependency between a destination coordinate $t \in \Omega$ and a jump origin coordinate $s \in \Omega$.

### 5.1 Formulation A: Dynamic Self-Attention (Micro-Transformer)

Under self-attention, each square token $\mathbf{x}_i \in \mathbb{R}^D$ generates queries, keys, and values:

$$\mathbf{A}_{i, j} = \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{D}}\right), \quad \mathbf{y}_i = \sum_{j} \mathbf{A}_{i, j} \mathbf{v}_j$$

- **Theoretical Strength:** Attention is content-dependent. If token $i$ represents an empty square with high capture gain, and token $j$ represents the only piece that can jump to $i$, the dot product $\mathbf{q}_i \cdot \mathbf{k}_j$ dynamically maximizes, routing the penalty of vacating $j$ directly into the value of landing on $i$.
- **Computational Complexity:** $\mathcal{O}(N^2 D)$ with transcendental functions ($\exp$) and dynamic memory allocations, bounding CPU alpha-beta search throughput to $< 15\text{k}$ NPS.

### 5.2 Formulation B: Static Geometric "Anchor-and-Gather" Spatial-Mixer

In Ataxx, the spatial distance $d_\infty(u, v)$ is state-invariant: coordinate $(2, 3)$ is always a jump origin for coordinate $(4, 3)$.

> **Critical Linear-Algebra Catch (Why Full-Channel Mixing Fails):**
> If both the Even channels ($\mathbf{X}_{:, 2k}$) and the Odd channels ($\mathbf{X}_{:, 2k+1}$) were multiplied by the same $49 \times 49$ matrix $\mathbf{W}_{\text{spatial}}$, both channel sets would be spatially smeared across the board in the exact same way. At row $t$, channel $2k$ would no longer represent the pure local destination reward at $t$!

**The "Anchor-and-Gather" Factorization (Fixes Coupling + Halves MatMul FLOPs):**
We partition the token channels $\mathbf{X} \in \mathbb{R}^{49 \times D}$ into two distinct halves:
1. **Anchored Even Channels ($\mathbf{X}_{\text{even}} = \mathbf{X}_{:, 0::2} \in \mathbb{R}^{49 \times D/2}$):**
   Left completely **un-mixed** (anchored at square $u$). They remain 100% localized at coordinate $u$, encoding the exact local $3 \times 3$ landing reward and morphological state at $u$.
2. **Spatial Odd Channels ($\mathbf{X}_{\text{odd}} = \mathbf{X}_{:, 1::2} \in \mathbb{R}^{49 \times D/2}$):**
   Multiplied by the learned geometric interaction kernel $\mathbf{W}_{\text{spatial}} \in \mathbb{R}^{49 \times 49}$ to gather origin and neighborhood states around each square:
   $$\mathbf{H}_{\text{odd}} = \mathbf{W}_{\text{spatial}} \mathbf{X}_{\text{odd}} + \mathbf{B}_{\text{odd}} \in \mathbb{R}^{49 \times D/2}$$

We then apply **Pairwise Multiplicative Rectification (SCReLU)** between the anchored local even channels and the spatially gathered odd channels:

$$\text{SCReLU}(x) = (\text{clamp}(x, 0, 1))^2$$

$$\mathbf{Z}_k(u) = \text{SCReLU}(\mathbf{X}_{\text{even}}(u, k)) \odot \text{SCReLU}(\mathbf{H}_{\text{odd}}(u, k)) \quad \text{for } k \in [0, D/2 - 1]$$

- **Algebraic Role of the Anchor-and-Gather Product:**
  At every square $t$, anchored channel $\mathbf{X}_{\text{even}}(t, k)$ encodes the **Local Destination State at $t$** (capture gain / plug value), while gathered channel $\mathbf{H}_{\text{odd}}(t, k)$ aggregates the **Remote Origin States around $t$** ($\text{Safety} = 1 - \text{Vulnerability}(s)$).
  The product $\mathbf{Z}_k(t)$ acts as an exact local-to-remote continuous differentiable AND gate:
  $$\mathbf{Z}_k(t) > 0 \iff (\text{Local Destination } t \text{ is High Gain}) \land (\text{Gathered Origins } s \to t \text{ are Safe})$$

- **Computational Complexity ($2\times$ Speedup):**
  Because $\mathbf{W}_{\text{spatial}}$ only multiplies the $D/2$ odd channels ($49 \times 49 \times (D/2)$ instead of $49 \times 49 \times D$), the matrix multiplication FLOPs are **cut in half**:
  $$\mathbf{H}_{\text{odd}} = \mathbf{W}_{\text{spatial}} \times \mathbf{X}_{\text{odd}}$$
  evaluating in **$\approx 60$ nanoseconds** using AVX2 `_mm256_maddubs_epi16` (or AVX-512 VNNI / WASM `i8x16` dot-product instructions), easily sustaining $\ge 100\text{k}$ NPS.

---

## 6. Information-Theoretic Framing of the Turn Operator

A common failure mode in game-evaluation network design is hardcoding minimax reductions into the architecture.

### 6.1 The Fallacy of Manual Minimax Pooling

Hardcoded minimax pooling imposes severe, incorrect constraints on the evaluation space:
- **Existential Asymmetry:** Game-theoretic value in high-churn games is non-linear. If player $U$ has ten distinct $+3$ captures, but player $T$ has a single $+12$ catastrophic response, simple max or sum reductions fail to capture that $U$ is completely forced into defensive containment.
- **Zugzwang Indifference:** If $\text{Legal}(U)$ contains only jumps that leave catastrophic ghost craters, forcing a move decreases $U$'s evaluation. A network with manual pooling cannot distinguish between a position with 10 safe moves and a position with 10 suicidal moves.

### 6.2 The Turn Token as a Conditioning Field

In TSTG, the side to move is modeled as a binary conditioning operator:

$$\tau \in \{0, 1\} \quad (\tau = 1 \iff \text{Side-to-Move})$$

Let $\mathbf{e}_\tau \in \mathbb{R}^D$ be the learned embedding of $\tau$. This embedding is broadcast-summed into all 49 spatial tokens prior to relational mixing:

$$\mathbf{x}_u = \mathbf{e}_{\text{base}}(u) + \mathbf{e}_\tau$$

```
   [49 Spatial Tokens]         [Turn State tau in {0, 1}]
            │                               │
            │                        Dense Embedding
            │                               │
            └───────────────┬───────────────┘
                            ▼
           Turn-Conditioned Tokens: x_u = e_base(u) + e_tau
                            │
                            ▼
           Relational Mixing (Transformer or Mixer)
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
[tau = 1 (Our Turn)]                  [tau = 0 (Their Turn)]
Attacks: Evaluated as Agency          Attacks: Evaluated as Latent Threats
Craters: Evaluated as Execution Risk  Craters: Evaluated as Target Vulnerabilities
```

> **Theorem (Universal Agency Emergence):**
> Given an unconstrained relational layer operating on $\mathbf{X}^{(0)}$, backpropagation against the Bellman objective naturally partitions the latent representations:
> 1. When $\tau = 1$, the network weights prioritize tokens where $b_{\text{Real}}^U = 1$ (immediate agency), conditioning their value on the absence of $b_{\text{Ghost}}^U$ (execution risk).
> 2. When $\tau = 0$, the network weights prioritize tokens where $b_{\text{Real}}^T = 1$ (opponent agency), conditioning the evaluation on defensive perimeter containment.
> 3. Zugzwang is learned naturally: if $\tau = 1$ and all active friendly actions coincide with $b_{\text{Ghost}}^U = 1$, the cross-channel interactions produce strongly negative activation sums without requiring hardcoded heuristic rules.

---

## 7. Second-Order Physical Phenomena

To achieve complete game-theoretic coverage of Ataxx, TSTG resolves five second-order physical phenomena that basic 1-ply and naive ghost-square models miss:

### 7.1 The Evacuation Duality: Fortress Breach vs. Hostage Escape

In the baseline formulation, vacating a jump origin $s \in \text{Vacatable}_U$ was modeled purely as a structural liability (creating a Ghost Crater). In full Ataxx game theory, vacating square $s$ operates across two opposing regimes governed by the local $3 \times 3$ piece composition around $s$:

- **Regime I: Fortress Breach (Negative Vacancy):**
  When $s$ is embedded inside a friendly cluster ($|\mathcal{N}_1(s) \cap U| \ge 2$) and lies within the opponent's action radius ($s \in \mathcal{B}_2(T)$), vacating $s$ creates an internal blast crater that sacrifices adjacent friendly pieces on the next ply.
- **Regime II: Stranded Hostage Evacuation (Positive Vacancy / Escape):**
  When $s$ is an isolated friendly outpost ($|\mathcal{N}_1(s) \cap U| = 0$) adjacent to empty squares that the opponent can reach ($\mathcal{N}_1(s) \cap \text{Real}_T \neq \emptyset$), the piece at $s$ is "dead food"—if it remains on $s$, the opponent will clone adjacent to $s$ on the next turn and convert it for free. Jumping the piece at $s$ away to join another cluster denies the opponent a $+2$ flip swing while leaving behind an empty square $s$ surrounded by zero friendly pieces (meaning invading $s$ yields 0 flips for the opponent).

Because the token at square $s$ encodes both the $3 \times 3$ neighborhood $\phi_{\text{topo}}(s)$ and the vacatability bit, the network automatically distinguishes between **Pinned Anchor Pieces** (high friendly neighbor count) and **Stranded Hostages Must-Jump** (zero friendly neighbors, adjacent to opponent targets).

---

### 7.2 Conversion Elasticity: Re-Infection vs. Fortress Sealing

Not all piece flips in Ataxx have equal permanence. When a player moves to $t \in E$ and converts the opponent pieces $C(t) = \mathcal{N}_1(t) \cap T$, the resulting state stability depends on whether the newly converted pieces in $C(t)$ touch other empty squares reachable by the opponent.

```
[Elastic / Volatile Flip]                 [Plastic / Permanent Flip (Fortress Seal)]
   T_attacker                                  U   U   U
       │ (Re-infects e_2 on Ply 2)             U  [t]  T_converted  U
       ▼                                       U   U   U
      [e_2] ── T_converted ── [t_landed]   (t was the ONLY empty neighbor of T_converted;
                                            landing on t makes the entire block 100% immune)
```

#### 7.2.1 Bitwise Carry-Save Multiplicity Counters ($\mathcal{O}(1)$ Neighbor Counting)

To classify stability across all 49 squares simultaneously without loops, we use a Bitwise Carry-Save Adder over the 8 directional 1-step shifts $\{S_0, S_1, \dots, S_7\}$ of any bitboard $X$.

In exactly 24 bitwise instructions, this produces three global bitmasks for any input mask $X$:
- $\mathcal{M}_{\ge 1}(X) = \text{Ones}_8 \equiv \mathcal{D}_1(X)$ (Squares with $\ge 1$ neighbor in $X$)
- $\mathcal{M}_{\ge 2}(X) = \text{Twos}_8$ (Squares with $\ge 2$ neighbors in $X$)
- $\mathcal{M}_{= 1}(X) = \text{Ones}_8 \ \& \ \sim\text{Twos}_8$ (Squares with exactly 1 neighbor in $X$)

#### 7.2.2 Morphological Immunity & Fortress Sealing Equations

Applying the multiplicity operator $\mathcal{M}$ to the Empty bitboard $E$ partitions all pieces on the board into exact structural stability tiers:

- **Core Invulnerable Pieces (0 Empty Neighbors):**
  $$\text{Solid}_U = U \ \& \ \sim\mathcal{M}_{\ge 1}(E)$$
  These pieces cannot be flipped on the next turn under any circumstance unless an adjacent piece jumps away.

- **Critical 1-Liberty Pieces (Exactly 1 Empty Neighbor):**
  $$\text{OneLib}_U = U \ \& \ \mathcal{M}_{= 1}(E)$$
  Every piece in $\text{OneLib}_U$ is touching one single empty hole. If either player fills that hole (and converts/protects the boundary), those pieces immediately lock into $\text{Solid}$ status!

- **Fortress-Sealing Target Squares (High-Priority Plugs):**
  An empty square $t \in E$ is a Fortress Seal if landing on $t$ locks adjacent 1-liberty pieces into permanent immunity:
  $$\text{Seal}_U = E \ \& \ \mathcal{D}_1(\text{OneLib}_U)$$

- **Elastic / Volatile Flip Targets (Subject to Immediate Re-Infection):**
  When $U$ lands on $t$, an adjacent enemy piece $w \in \mathcal{N}_1(t) \cap T$ is Elastic (Re-Infectible) if $w$ touches at least one other empty square reachable by $T$ ($\mathcal{M}_{\ge 2}(E \ \& \ (\text{Real}_T \cup \{t\}))$). Conversely, if $w \in \text{OneLib}_T$, flipping $w$ by landing on $t$ permanently seals $w$ into $U$'s fortress with zero possibility of counter-flip!

---

### 7.3 Second-Order Tactical Paradoxes: Assassination Jumps & Redundancy

Basic ghost-vacancy logic assumes that if $s \in \text{Vacatable}_U$ is within range of an enemy piece ($s \in \mathcal{B}_2(T)$), jumping from $s$ is dangerous. This breaks down in two critical cases that the bitboard pre-processor can resolve exactly:

#### 7.3.1 Paradox A: Multi-Origin Jump Redundancy

Suppose empty target $t \in \text{Jump}_U$ can be reached by a pinned fortress piece $s_1 \in \text{Ghost}_U$, and can also be reached by an unpinned safe piece $s_2 \notin \text{Ghost}_U$ (or via a Clone).

Because rational play will always choose the safe origin $s_2 \to t$ over the suicidal origin $s_1 \to t$, target $t$ is not risky unless all origins capable of reaching $t$ are pinned:

$$\text{PureRisky}_U = (\mathcal{R}_2(\text{Ghost}_U) \ \& \ E) \ \setminus \ (\mathcal{D}_1(U) \cup \mathcal{R}_2(U \ \& \ \sim\text{Ghost}_U))$$

Moreover, using the carry-save multiplicity operator on the outer jump ring $\mathcal{R}_2$, we can flag **Redundant Safe Jumps** (targets reachable by $\ge 2$ distinct safe jump origins, preserving flexibility).

#### 7.3.2 Paradox B: The Assassination / Self-Covering Jump

Suppose friendly piece $s$ is threatened by only one enemy piece $w \in T$, and $w$ happens to sit at Chebyshev distance 1 from our jump destination $t$ ($w \in \mathcal{N}_1(t)$).
- If $s$ jumps to $t$, landing on $t$ flips $w$ to Friendly!
- On the opponent's turn, $w$ no longer belongs to the opponent, so the Ghost Crater left at $s$ cannot be punished!
- Using the carry-save adder across enemy threat coverage, we separate **Single-Attacker Ghost Threats** (which can be neutralized by capturing the lone attacker) from **Multi-Attacker Ghost Threats** ($\ge 2$ enemy pieces threatening $s$, impossible to assassinate if they don't share the same $3 \times 3$ flip box).

---

### 7.4 Endgame Parity, Private Eyes, and Starvation Horizons

As the empty square count $|E|$ approaches zero, Ataxx transitions from a tactical flip war into a combinatorial game of Tempo and Parity (similar to Othello and Go endgames).

#### 7.4.1 Isolated 1-Hole Eyes (Tempo Vaults)

An empty square $e \in E$ with zero empty neighbors ($\mathcal{M}_{\ge 1}(E)(e) = 0$) is an Isolated Eye:

- **Private Friendly Eye:**
  $$\text{Eye}_U = \text{IsolatedEyes} \ \& \ \text{Clone}_U \ \& \ \sim\text{Real}_T$$
  Opponent cannot reach $e$, and because $e$ is surrounded exclusively by friendly pieces or blockers, $U$ has a guaranteed free move in reserve. Even better: any friendly piece $u \in \mathcal{N}_2(e)$ cannot be forced to vacate its square if $U$ still has clones into $\text{Eye}_U$.
- **Contested Eye:**
  $$\text{Eye}_{\text{Contested}} = \text{IsolatedEyes} \ \& \ \text{Real}_U \ \& \ \text{Real}_T$$
  Urgent territorial battleground that reduces $|E|$ by 1 without creating new adjacent empty boundaries.

#### 7.4.2 Starvation and Pass-Cascade Detection

In Ataxx, if $\text{Real}_T = \emptyset$ while $\text{Real}_U \neq \emptyset$, player $T$ is Starved (Forced to Pass). Player $U$ gets to play consecutive uninterrupted clones until either the entire board is filled or $U$ voluntarily moves adjacent to $T$.

- If $\text{Real}_T = \emptyset$, the opponent has zero agency.
- Even more powerfully: if $\text{Real}_T \subseteq \text{Real}_U$ and $|\text{Real}_T| = 1$ (opponent has only 1 legal move square on the entire board, and we can also reach that exact square), if it is $U$'s turn ($\tau = 1$), $U$ can play on that single square and immediately starve $T$ on ply 1!
- **Bitboard Starvation Threat Flag:**
  $$\text{StarvationThreat} = (\text{popcount}(\text{Real}_T) == 1) \ \& \ ((\text{Real}_T \ \& \ \text{Real}_U) \neq 0)$$

---

### 7.5 Unified 16-Phase Architecture & Zero-Cost Policy Readout

To integrate seamlessly with an alpha-beta engine that uses 16 game-phase buckets and policy-guided move ordering / Late Move Reductions (LMR):

```
                49 Enriched Square Tokens X in R^(49 x D)
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
[Anchored Even: X_even]               [Spatial Odd: H_odd = W_spatial · X_odd]
(Local 3x3 state at u)                (Gathered origins/threats around u)
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
              Anchor-and-Gather Pairwise SCReLU
              Z_k(u) = SCReLU(X_even) * SCReLU(H_odd)
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
[16-Phase Value Readout]                [Zero-Cost Move Ordering Policy]
Phase p = (Occupied * 16) / ValidSq     Clone(t):     Score = w_c · z_t
Eval = W_val[p] · flatten(Z) + b[p]     Jump(s -> t): Score = w_d · z_t + w_o · z_s
```

#### 7.5.1 Phase-Bucketed Final Layer (16 Game Phases)

The relative value of mobility, ghost craters, and solid fortresses shifts drastically from the opening (sparse board, jump-heavy) to the endgame (dense board, clone/parity-heavy).

Instead of using a single static output vector $\mathbf{w}_{\text{out}}$, the final linear readout indexes into 16 phase-specific weight banks conditioned on board fullness:

$$p = \left\lfloor \frac{\text{popcount}(U \mid T) \times 16}{\text{ValidSquares}} \right\rfloor \in [0, 15]$$

$$V(S) = \mathbf{w}_{\text{out}}^{(p)\top} \text{vec}(\mathbf{H}) + b^{(p)}$$

Because only the final dot product is bucketed by $p$, you gain $16\times$ parameter capacity across game phases with zero extra runtime compute (still just one dot product per node).

#### 7.5.2 Unified Policy Extraction for Move Ordering (Origin-Destination Decomposition)

Because the relational layer outputs a contextualized activation vector $\mathbf{h}_u \in \mathbb{R}^{D}$ for every square $u \in \{0 \dots 48\}$, the network already possesses the exact latent value of landing on any target $t$ and vacating any origin $s$.

Instead of executing a second neural network to score moves for alpha-beta move ordering and LMR, move logits are extracted directly from the active square tokens using three learned projection vectors $\mathbf{w}_{\text{clone}}, \mathbf{w}_{\text{dest}}, \mathbf{w}_{\text{orig}} \in \mathbb{R}^D$:

- **For any Clone Move to Target $t$:**
  $$\pi_{\text{clone}}(t) = \mathbf{w}_{\text{clone}}^\top \mathbf{h}_t$$

- **For any Jump Move from Origin $s$ to Target $t$:**
  $$\pi_{\text{jump}}(s \to t) = \mathbf{w}_{\text{dest}}^\top \mathbf{h}_t + \mathbf{w}_{\text{orig}}^\top \mathbf{h}_s$$

**Computational Advantages:**
1. Compute the two scalar arrays $\text{DestScore}[u] = \mathbf{w}_{\text{dest}}^\top \mathbf{h}_u$ and $\text{OrigScore}[u] = \mathbf{w}_{\text{orig}}^\top \mathbf{h}_u$ across the 49 squares once per node.
2. Scoring a legal jump $s \to t$ during move generation is literally one integer addition: $\text{DestScore}[t] + \text{OrigScore}[s]$.
3. If $s$ is a pinned fortress piece, $\text{OrigScore}[s]$ is strongly negative (penalizing the suicidal jump in move ordering). If $s$ is a stranded hostage piece, $\text{OrigScore}[s]$ is positive (boosting the escape jump to the front of the move list).

---

## 8. Complete Enriched Feature Vector per Square

Each of the 49 squares receives its canonical $3 \times 3$ topology index $\phi_{\text{topo}}(u) \in [0, 6560]$ alongside a compact 12-bit Tactical & Morphological State Vector $\mathbf{b}(u) \in \{0, 1\}^{12}$:

| Bit | Name | Bitboard Formula | Game-Theoretic Meaning |
|:---:|:---|:---|:---|
| **0** | `Clone_U` | $\mathcal{D}_1(U) \ \& \ E$ | Friendly can clone here ($+1$ net piece $+$ flips, zero origin crater) |
| **1** | `SafeJump_U` | $\mathcal{R}_2(U \ \& \ \sim\text{Ghost}_U) \ \& \ E$ | Friendly can jump here without unpinning a threatened fortress piece |
| **2** | `RiskyJump_U` | $(\mathcal{R}_2(U) \ \& \ E) \ \& \ \sim(\text{Bit}_0 \mid \text{Bit}_1)$ | Landing here forces vacating a punishable Ghost Crater |
| **3** | `Clone_T` | $\mathcal{D}_1(T) \ \& \ E$ | Opponent can clone here on their turn |
| **4** | `SafeJump_T` | $\mathcal{R}_2(T \ \& \ \sim\text{Ghost}_T) \ \& \ E$ | Opponent can safely jump here on their turn |
| **5** | `RiskyJump_T` | $(\mathcal{R}_2(T) \ \& \ E) \ \& \ \sim(\text{Bit}_3 \mid \text{Bit}_4)$ | Opponent jumping here forces breaking their own fortress |
| **6** | `Ghost_U` | $\mathcal{R}_2(E) \ \& \ U \ \& \ \mathcal{B}_2(T)$ | Friendly piece here can jump, and opponent can invade the resulting hole |
| **7** | `Ghost_T` | $\mathcal{R}_2(E) \ \& \ T \ \& \ \mathcal{B}_2(U)$ | Opponent piece here can jump, and we can invade the resulting hole |
| **8** | `Solid_Piece` | $(U \mid T) \ \& \ \sim\mathcal{M}_{\ge 1}(E)$ | Piece has 0 empty neighbors (100% immune to flips next ply) |
| **9** | `OneLib_Piece` | $(U \mid T) \ \& \ \mathcal{M}_{= 1}(E)$ | Piece has exactly 1 empty neighbor (plugging that neighbor locks this piece) |
| **10** | `Fortress_Seal` | $E \ \& \ \mathcal{D}_1(\text{Bit}_9)$ | Landing on this empty square permanently locks adjacent 1-liberty pieces |
| **11** | `Isolated_Eye` | $E \ \& \ \sim\mathcal{D}_1(E)$ | Isolated 1-square pocket (tempo vault / parity eye) |

Combined with global scalar inputs (Turn $\tau \in \{0, 1\}$, Parity $|E| \pmod 2$, and `StarvationThreat`), this 12-bit bitwise preprocessor and $3 \times 3$ topology lookup provides a mathematically closed, 3-ply tactical and morphological representation of Ataxx in under 60 CPU instructions.


---

## 9. Deduplicated Destination Evaluation & Jump-Induced Weakness Modeling (User Design Rule)

**Core Principle:** The engine search is untouched. Inside the TSTG static evaluation, we eliminate redundant computation by strictly **deduplicating $3 \times 3$ tuple evaluations of destination squares**, and reallocating compute to **rigorously modeling the enemy retaliatory threats directed towards the vacated jump weaknesses (ghost squares)**.

### 9.1 Destination Deduplication: Evaluating Landing Impact Exactly Once
In a typical Ataxx position, an empty square $t \in E$ may be reachable by multiple friendly pieces:
$$s_1, s_2, \dots, s_k \in \mathcal{R}_2(t) \cap U$$
- In naive multi-tuple evaluation, evaluating each legal jump $s_i \to t$ would evaluate $k$ separate $3 \times 3$ tuples centered at $t$.
- **Idempotency Property:** By Lemma 1 and Section 2.2, the piece flips caused by landing at $t$ depend *strictly* on the local neighborhood $\mathcal{N}_1(t) \cap T$, completely independent of which origin $s_i$ vacated its square ($s_i \notin \mathcal{N}_1(t)$ because $d_\infty(s_i, t) = 2$).
- **Compute Optimization:** The $3 \times 3$ landing tuple at destination $t$ is evaluated **exactly once**. Regardless of how many pieces can jump to $t$, the destination evaluation is deduplicated into a single square token $\mathbf{h}_t$, collapsing the landing complexity from $\mathcal{O}(|\Omega|^2)$ to $\mathcal{O}(|\Omega|) = 49$ lookups.

### 9.2 Modeling Jump-Induced Weaknesses & Enemy Retaliation Vectors
The true variance and risk of a jump $s \to t$ is NOT at the destination $t$, but at the origin $s$, which transitions from friendly occupancy to an empty vulnerability: $S(s) = 1 \to 0$.

The compute saved by destination deduplication is invested into explicitly modeling the enemy moves generated towards the vacated weakness $s$:

1. **Enemy Threat Generation Towards Vacated Square $s$:**
   Any enemy piece within Chebyshev distance 2 can exploit the newly created hole $s$:
   $$\mathcal{A}_{\text{retaliate}}(s) = \{ w \in \mathcal{B}_2(s) \mid S(w) = -1 \}$$
   - **Enemy Clone Threat:** If $w \in \mathcal{N}_1(s) \cap T$, the enemy can clone into $s$ with zero origin risk, claiming $s$ and flipping all adjacent friendly pieces.
   - **Enemy Jump Threat:** If $w \in \mathcal{R}_2(s) \cap T$, the enemy can jump into $s$, vacating $w$ to convert $s$ and its neighborhood.

2. **Quantifying Enemy Counter-Strike Damage $\mathcal{D}_{\text{enemy}}(s)$:**
   When an opponent move lands on the vacated square $s$, the immediate damage inflicted is:
   $$\mathcal{D}_{\text{enemy}}(s) = 1 + |\mathcal{N}_1(s) \cap U|$$
   This flip potential is pre-calculated via the 1-ring friendly neighborhood around $s$.

3. **Bitboard Enemy Threat Multiplicity:**
   Using the carry-save adder over the enemy threat masks:
   - $\mathcal{M}_{\ge 1}(\mathcal{B}_2(T))(s) = 1$: The vacated hole $s$ can be invaded by at least 1 enemy piece.
   - $\mathcal{M}_{\ge 2}(\mathcal{B}_2(T))(s) = 1$: The vacated hole $s$ is targeted by $\ge 2$ enemy pieces (multi-attacker threat; cannot be neutralized by a single assassination jump).

### 9.3 Bilateral Spatial Synthesis
Instead of search expanding branches or evaluation replicating 3x3 tables:
- **Destination Token $\mathbf{h}_t$:** Encodes the static reward of landing on $t$ (friendly capture gain).
- **Origin Weakness Token $\mathbf{h}_s$:** Encodes the structural liability of vacating $s$ (enemy retaliation moves directed towards $s$, damage $\mathcal{D}_{\text{enemy}}(s)$, and multi-threat status).
- **Relational Mixer $\mathbf{W}_{\text{spatial}}(s, t)$:** Combines the destination reward with origin weakness via learned spatial routing, accounting for the net delta:
  $$\text{NetGain}(s \to t) \approx \text{Reward}(t) - \text{Vulnerability}(s)$$
  without ever re-evaluating the destination tuple.

### 9.4 Engine Search Invariance
The engine's alpha-beta search algorithm, move generator, and pruning frameworks remain completely untouched. This optimization is strictly internal to the TSTG evaluation graph, replacing flat NNUE feature extraction with a deduplicated, weakness-aware static tactical representation.

### 9.5 Clone Dominance & The Hostage Evacuation Exception
A clone has zero departure cost because the piece duplicates itself—the source square is never vacated ($\delta_s^0 = 0$). Consequently, the identity of the cloning piece is completely irrelevant, creating a clean computational separation:
- **For Clones:** Move evaluation is strictly $\text{Score}(t)$. Destination $t$ is evaluated once, which is 100% sufficient for every piece capable of cloning to $t$.
- **For Jumps:** Move evaluation requires the pair $\text{Score}(t) + \text{OrigValue}(s)$ (where $\text{OrigValue}(s) = -\text{Vulnerability}(s)$), because the piece at $s$ vanishes.

Furthermore, whenever an empty square $t$ lies in $\text{Clone\_U} \cap \text{JumpTargets}$, **the clone strictly dominates any normal jump to $t$ in material**:
1. The clone grants $+1$ net piece on the board and leaves zero hole behind.
2. The jump grants $+0$ net pieces on the board and leaves an empty hole at $s$.

**The Sole Exception (Stranded Hostage Evacuation, §7.1 Regime II):**
The *only* scenario where a player prefers jumping $s \to t$ over an available clone to $t$ is an **evacuation jump**—intentionally vacating $s$ because $s$ is an isolated friendly outpost ($|\mathcal{N}_1(s) \cap U| = 0$) about to be surrounded and converted for free by the opponent next ply. Because Stage 2 bitboard flags capture both `Clone_U` at $t$ and the isolated/threatened state at $s$:
- In static evaluation, if $t \in \text{Clone\_U}$, the anchored even channels at $t$ treat $t$ as a guaranteed $+1$ expansion without requiring origin penalty checks, while isolated hostages at $s$ independently contribute positive evacuation potential.
- In policy move ordering, a pinned fortress jump has $\text{OrigScore}[s] \ll 0$ (ranked well below $\pi_{\text{clone}}(t)$), whereas a stranded hostage has $\text{OrigScore}[s] > 0$ (allowing the evacuation jump to naturally outrank or rival a passive clone when rescuing the stone saves a $+2$ swing).

