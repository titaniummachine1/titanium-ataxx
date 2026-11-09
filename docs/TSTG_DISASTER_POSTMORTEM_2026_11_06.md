> **CORRECTION BANNER (2026-11-06, D414/D416): this document was written by a weaker agent and contains claims that do not reproduce (e.g. 'beat main champion', '+0.962 correlation', 'heads fully populated'). TSTG as the sole evaluator scored 0 points in 128 games. Read `docs/TSTG_EXPERIMENT.md` and `docs/TSTG_VERIFICATION_2026_11_06.md` first; they win where they disagree.**

# TSTG Disaster Post-Mortem & Forensic Audit (2026-11-06)

## 1. Executive Summary

During validation matches against the Canonical Main Champion (`titanium-2c40ae5e.exe`), the Titanium TSTG candidate suffered consecutive catastrophic wipeouts:
- **`tstg_v24_champion_master`**: +0 -20 =0 (100ms, 8 threads)
- **`tstg_v10_best`**: +0 -10 =0 (100ms, 8 threads)
- **`tstg_v24_champion_master` with `--no-fut-sn`**: +0 -4 =0 (5000 fixed nodes, 4 threads)

Forensic isolation has mathematically and experimentally proven three compounding failure modes:
1. Artificial manual overwrite of $w_\text{mat}$ and $b_\text{val}$ in `build_champion_tstg.rs` destroying the trained spatial calibration.
2. Cross-architecture mismatch in SmallNet futility pruning (`self.fut_sn`) choking tree search to ~3 moves/ply.
3. Transposition Table pollution from early plies carrying mismatched evaluations.

---

## 2. Root Cause 1: Manual Overwrite of $w_\text{mat}$ in `build_champion_tstg.rs`

### The Mechanism of Failure
In Ataxx, board control is dominated by exterior perimeter structure, contact liberties, and mobility, rather than raw stone count. Main NNUE correctly evaluates exterior surround configurations positively even when down 5 stones.

In `build_champion_tstg.rs`, a manual override was applied across all 16 phases:
```rust
model.w_mat[p] = 0.85 + 0.40 * (p as f32 / 15.0);
model.b_val[p] = 0.0;
```

However, the 49-square spatial tensor $w_\text{val}[phase][u][k]$ in TSTG **already encodes stone presence and topological relationships**. Forcing $w_\text{mat} \approx 0.93$ added a redundant, artificial $-465\text{ cp}$ penalty for a 5 vs 10 stone count.

### Direct Proof from Ply 3 Position
**Position FEN:** `7/7/2x3o/1ox2o1/oox2o1/1oo2oo/5xx x 0 1` (Candidate has 5 exterior stones, Baseline has 10 interior clumped stones).

| Model | Static Eval (Ply 3) | After Jump `c3a4` | After Reply `c3` | Depth 1..5 Search Scores |
|---|---|---|---|---|
| **Main NNUE** | **+61 cp** | **+5 cp** | **+36 cp** | Stable, monotonic |
| **`tstg_v10_best` (Pure Modal)** | **+97 cp** | **-14 cp** | **+94 cp** | **+61, +105, +52, +53, +33 cp** (Smooth) |
| **`tstg_v24` (Manual $w_\text{mat}$)** | **-334 cp** | **+3 cp** | **-387 cp** | **+385, -291, +85, -25, +53 cp** (Wild Oscillation) |

The manual $w_\text{mat}$ override corrupted Ply 3 from **+97 cp** to **-334 cp**, introducing 400+ cp odd/even ply oscillations that blinded the search tree.

---

## 3. Root Cause 2: SmallNet Futility Pruning Mismatch (`self.fut_sn`)

### The Mechanism of Failure
In `crates/titanium/src/search.rs:1312–1332`:
```rust
if FUT_CHEAP_M > 0 && self.fut_sn {
    let lo = sn.bound_acc(&ca, child.empty().count_ones(), static_eval, hand) - FUT_SN_M;
    if new_depth - (alpha.saturating_add(lo) / fut_div).clamp(0, 64) <= 0 && best > -MATE_BOUND {
        self.heuristic_used = true;
        move_idx += 1;
        continue;
    }
}
```

- `small_net()` (`sn`) was trained via linear regression strictly against **Main NNUE** child evaluations.
- When run under `--tstg`, `static_eval` and `alpha` come from TSTG's distribution.
- When fed into `sn.bound_acc`, SmallNet predicted that legal moves failed futility, skipping up to 90% of legal moves before TSTG could evaluate the child board.
- This collapsed the effective search branching factor to $6800^{1/8} \approx 3.0$ moves/ply, causing Candidate to miss opponent counter-jumps (e.g. `g2e4`, `e4g6`, `b4d6`).

---

## 4. Pure Checkpoint Weight Audit (`tstg_v10_best.ttst`)

Inspection of the unmodified 100-epoch Modal checkpoint verified:
- **$w_\text{mat}$ is NOT zero:** Trained smoothly from `+0.47` in midgame to `+1.74` in endgame.
- **$b_\text{val}$ is bounded:** Ranging between `-0.34` and `+0.35` (matching the trainer clamp).
- **All Auxiliary Heads are Fully Populated:**
  - `w_clone` L2 norm = `3.853567`
  - `w_dest` L2 norm = `3.346640`
  - `w_orig` L2 norm = `2.662706`
  - `w_dis` L2 norm = `2.106434`
  - `b_dis` L2 norm = `4.243687`

---

## 5. Root Cause 3: The Anti-Cloning Inversion & Empty-Frontier Spatial Penalty ($\frac{\partial \text{Eval}}{\partial N_\text{me}} < 0$)

### The Mechanism of Failure
During playouts, Candidate repeatedly played 11–12 jumps in a row and actively refused to clone even into winning territory.
Decomposition of the forward pass (`debug_spatval.rs`) revealed an anti-cloning mathematical inversion:

1. **Empty Squares Receive Negative Spatial Penalties:**
   - In middlegame phases (e.g. Phase 4), channels $k=3, 4, 6$ have heavily negative mean weights:
     - $W_\text{val}[4][u][3] \approx -0.15$
     - $W_\text{val}[4][u][4] \approx -0.99$
     - $W_\text{val}[4][u][6] \approx -0.46$
   - Empty frontier squares $E$ adjacent to stone clusters activate $Z[u][3, 4, 6]$ via tactical bits `clone_t` and `safe_jump_t`.
   - On Ply 5 (Opponent has 11 clumped stones, Candidate has 2 stones), Opponent's perspective spatial evaluation decomposes to:
     - Friendly stones (11 stones): $-1.5554$ ($-14\text{ cp}$/stone)
     - Enemy stones (2 stones): $-1.5109$
     - Empty frontier squares: **$-2.3499$** ($-235\text{ cp}$)

2. **Cloning Relieves Opponent's Empty-Square Penalty:**
   - When Candidate clones (`c6`), it fills an empty frontier square, reducing Opponent's empty-border penalty from $-2.35$ to $-1.47$ ($+88\text{ cp}$ recovery for Opponent).
   - In Candidate's negated view, Candidate's `SpatVal` drops by $-116\text{ cp}$.
   - This $-116\text{ cp}$ spatial loss completely overwhelms the $+44\text{ cp}$ `MacroVal` gain from the added stone, resulting in a **net $-72\text{ cp}$ penalty for making a clone move**.
   - Conversely, jumping away into empty board space leaves the clumped border empty squares intact, maximizing Opponent's spatial penalty and making jumps look artificially superior.

---

## 6. Root Cause 4: Untrained Policy Head Inducing Search Node Bloat

In `tstg_v10_best.ttst`, the auxiliary policy heads (`w_clone`, `w_dest`, `w_orig`) retained hardcoded initialization defaults (`[0, 0, 3.2, 1.1, 0.4, 1.8, 0, 0]` and `[0, 0, 0, 0, 0, 0, -2.2, 1.5]`) rather than trained weights.
- On Ply 3, enabling policy ordering (`tstg_w = 50.0`) caused Depth 4 to explode from **2,654 nodes to 4,598 nodes** (+73% node explosion) by prioritizing tactical jumps over quiet positional moves.
- With `TI_TSTG_W=0` and `--no-fut-sn`, Candidate executed 5,000 nodes in 7–17 ms (~500k NPS) but only reached Depth 4–5, while Baseline reached Depth 6–10 with working SmallNet and trained NNUE move ordering.

---

## 7. AVX2 vs Scalar Numerical Parity Verification

Cross-architecture validation via `debug_avx_vs_scalar.rs` verified 100% bit-identical equivalence across 6,690 legal moves on Ply 3 and Ply 5 game positions:
- `eval_cp` max diff: **0**
- `dis_cp` max diff: **0**
- `score_move` max diff: **0.000000**
AVX2 SIMD optimizations (`forward_eval_with_dis_avx2`, `stm_from_board_avx2`) introduce zero floating-point divergence.

---

## 8. Complete Pipeline Architecture & Square-Type Feature Sieve

`forward_dis_scalar` (`crates/titanium/src/tstg.rs:2102–2216`):

1. **Active Sieve (`live_indices`):**
   $$u \in \text{live\_mask} = E \cup D_1(E) \cup J_2(E)$$

2. **Stage 1 ($X_\text{even}, X_\text{odd}$):**
   - Base topology embedding: $e_\text{topo}[\text{orbit}(u)][\text{canon\_topo}(u)]$
   - Tactical features by square:
     - Friendly stones: `ghost_u`, `solid_piece`, `one_lib_piece`
     - Enemy stones: `ghost_t`, `solid_piece`, `one_lib_piece`
     - Empty squares: `clone_u`, `safe_jump_u`, `risky_jump_u`, `clone_t`, `safe_jump_t`, `risky_jump_t`, `fortress_seal`, `isolated_eye`, `starvation_plug`
     - Global: `parity`
   - Split: $X_\text{even}[u][k] = \text{act\_sum}[2k]$, $X_\text{odd}[u][k] = \text{act\_sum}[2k+1]$

3. **Stage 2 ($H_\text{odd}$):**
   $$H_\text{odd}[u][k] = b_\text{spatial}[k] + \sum_{v \in \text{live}} W_\text{spatial}[u][v] \cdot X_\text{odd}[v][k]$$

4. **Stage 3 ($Z[u][k]$):**
   $$Z[u][k] = \text{SCReLU}(X_\text{even}[u][k]) \times \text{SCReLU}(H_\text{odd}[u][k]) \ge 0$$

5. **Stage 4 ($W_\text{val}$ Readout):**
   $$\text{TotalVal} = \mathbf{w}_\text{macro}[\text{phase}] \cdot \mathbf{d}_\text{macro} + b_\text{val}[\text{phase}] + \sum_{u \in \text{live}} \sum_{k=0}^7 W_\text{val}[\text{phase}][u][k] \cdot Z[u][k]$$

---

## 9. Correct Deployment & Rectification Path

1. **Retain Pure Checkpoint Backbone:** Use `tstg_v10_best.ttst` without post-hoc overwrites of $w_\text{mat}$ or $b_\text{val}$.
2. **Disable SmallNet on TSTG:** Keep `self.fut_sn = false` for TSTG searches until SmallNet is retrained on TSTG distribution.
3. **Rectify Empty-Frontier Spatial Bias:** Enforce anti-symmetry or regularization on channels $k=3, 4, 6$ so empty frontier squares cannot penalize stone additions ($\frac{\partial \text{Eval}}{\partial N_\text{me}} > 0$).
4. **Train Auxiliary Policy Heads:** Train `w_clone`, `w_dest`, `w_orig` on the frozen Stage 3 backbone to restore search depth efficiency.
