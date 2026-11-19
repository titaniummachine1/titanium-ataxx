# 🧠 Master Backlog, NNUE Saturation & Disagreement Head Specification (`TNN3` + `TSN1`)

This document tracks the immediate execution backlog, hardware vectorization strategy (AVX-2 & AVX-512), capacity-vs-information diagnostics, and exact code-level integration for Titanium's multi-stage roadmap.

---

## 1. Strategic Order of Operations & Branch Discipline

$$\text{Track A: Persistent Tree (`exp/persistent-tree`)} \;\longrightarrow\; \text{Phase 1: Teacher Saturation & Ablation} \;\longrightarrow\; \text{Phase 2: Distilled Big Net \& `SmallNet` (`TSN1`)} \;\longrightarrow\; \text{Phase 3: Disagreement Head (`TNN3`)} \;\longrightarrow\; \text{Phase 4: Active-Learning Loop}$$

> **Strict Branch & Gating Rule:**
> Every feature and optimization is developed on an isolated `exp/*` branch, verified with unit tests and an SPRT gate (`titaniumtest sprt`), and merged into **`staged/main`** for review. `main` remains untouched until explicit sign-off.

### 1.1 Why Big Net Saturation & Pattern Ablation Come First (`Phase 1`)
1. **No Premature Optuna Sweeps:** Running hyperparameter sweeps on an under-trained network wastes compute. We first train the Big Net on the complete **31,568,598 5k-labeled dataset (`dis_positions.bin`)** via Modal GPU (`training/modal_train.py`) until validation loss reaches its asymptotic plateau.
2. **Capacity-Bound vs. Information-Bound Diagnostic:**
   - Before permanently expanding network width from $W=256$ to $W=512$, we compare validation loss curves between $W=256$ and $W=512$ at full saturation.
   - **If validation loss plateaus at the same floor:** The model is **information-bound** (limited by input features or search target granularity, not parameter count). We keep the lean $W=256$ runtime architecture (or distill $W=512 \to W=256$) so inference stays twice as fast.
   - **If $W=512$ opens a meaningful validation gap:** We use $W=512$ as the **Master Teacher** and distill its soft predictions into the $W=256$ runtime net, `fut_small.tsn`, and the `TNN3` Disagreement Head.
3. **Pattern Saliency & Tuxel (`autaxx`) Feature Probing:**
   - Run `nnerr_analyze.py` and group-drop ablations (`--drop G*`) on the saturated checkpoint to inspect what board patterns the net seeks or misses.
   - Evaluate candidate structural inputs inspired by Tuxel's `autaxx` (`C:/gitProjects/autaxx`):
     - **Hole-risk & blocker topology:** Corner/edge stability around immutable blocker squares.
     - **Jump-ring vulnerability:** Exposed singletons susceptible to 1-to-8 infection swings.
     - **Mobility & stone parity delta:** Explicit side-to-move clone/jump availability and endgame fill parity.

### 1.2 Hardware Vectorization: AVX-2 Primary + AVX-512 Botzone Acceleration
- **Primary Local Standard (`AVX-2 / x86-64-v3`):** All local development, SPRT gates, and baseline benchmarks optimize for AVX-2 first so local hardware runs at peak efficiency.
- **Botzone Server Acceleration (`AVX-512 / x86-64-v4`):**
  - Botzone's competition cluster supports AVX-512 instructions.
  - Currently, `tools/build_botzone.py` compiles `libtitanium_ffi.so` with baseline `x86_64-unknown-linux-gnu.2.14` (SSE2 only).
  - Because our accumulator width $W=256$ is an exact multiple of 32 (`256 = 8 * 32`), **the exact same $W=256$ network aligns with zero padding on both AVX-2 and AVX-512**:
    - **SSE2 (128-bit):** 32 loop iterations of 8-lane `i16` per active feature.
    - **AVX-2 (256-bit):** 16 loop iterations of 16-lane `i16` per active feature.
    - **AVX-512 (512-bit):** **8 loop iterations** of 32-lane `i16` per active feature (**2× vector throughput over AVX-2, 4× over SSE2** on the exact same net!).
  - Branch `exp/botzone-simd` adds `--cpu <v3|v4|runtime>` support to `tools/build_botzone.py` while strictly enforcing `glibc <= 2.17`.

### 1.3 Why `SmallNet` (`TSN1`) & Distillation Come Second (`Phase 2`)
As documented in `smallnet.rs:7-8`, `fut_small.tsn` ($W=32, H=16$, `~90 ns/child`) is fitted to a specific `nnue_main.tnn` and must be re-distilled whenever `nnue_main.tnn` changes:
- **Upgraded Target:** Train `fut_small.tsn` on $\min(\text{BigNet}(\text{child}), -\text{Search}_{5k})$ so the 90 ns pre-filter never prunes sacrificial capture chains.
- **Tuxel / Fast Incremental Refinements:** Keep the incremental feature-transformer update path allocation-free and L1-resident.

### 1.4 Why Disagreement Head (`TNN3`) Comes Third (`Phase 3`)
With the new static evaluator frozen, we compute the exact 5k-vs-static disagreement $\Delta_{\text{true}}(s) = |\text{Search}_{5k}(s) - \text{StaticEval}_{\text{new}}(s)|$ across all 31.57M relabeled positions in a single pass and train the zero-cost `TNN3` auxiliary head (`dis_w[bucket][32]`, `< 1 ns` inference).

### 1.5 Why Self-Relabeled External Geometries Come Fourth (`Phase 4`)
Because Titanium outperforms external engines, **we never fine-tune directly on weaker external teacher evaluations**. External datasets (`KataGo_Ataxx_202604`, `sotax.zip`, `zzzzz`, Botzone `atx_positions.bin`) supply **raw board geometries `(occ0, occ1, blk, stm)`**, which we filter by high predicted $\hat{\Delta}$ (Active Learning) and relabel **100% with Titanium's own 5k-node search**.

---

## 2. Exact Code Integration in `crates/titanium/src/`

### 2.1 Zero-Cost `TNN3` Head in `nnue.rs` (Branch: `exp/disagreement-head`)
In [`crates/titanium/src/nnue.rs:256-324`](../crates/titanium/src/nnue.rs), `Nnue::head` already computes the shared hidden activation vector (`h1[..l1]` and `h2[..l2]`) for the active empties `bucket`.

Adding the Disagreement Head (`magic == b"TNN3"`) requires only one additional dot product per `bucket` when present (and safely returns `0` for `TNN1`/`TNN2` nets):
```rust
let dis_cp = if self.has_dis {
    let k = l1 + l2;
    let dis_row = &self.dis_w[bucket * k..(bucket + 1) * k];
    let d1: f32 = dis_row[..l1].iter().zip(&h1[..l1]).map(|(a, b)| a * b).sum();
    let d2: f32 = dis_row[l1..].iter().zip(&h2[..l2]).map(|(a, b)| a * b).sum();
    (400.0 * (d1 + d2 + self.dis_b[bucket]).max(0.0)).round() as i32
} else {
    0
};
```
* **Parameter footprint:** `8 buckets * (48 + 1) floats = 392 floats = 1,568 bytes` (for $L_1=16, L_2=32$).
* **Runtime cost:** **~0.8 ns** when `head()` is already running (`0.00 ns` on `ecache` hits).

### 2.2 Zero-Memory Packing in `ecache` (`search.rs:550`)
In [`crates/titanium/src/search.rs:550`](../crates/titanium/src/search.rs), `Searcher::ecache` is a $2^{16}$-entry table of `(u64, i32)` (`1 MiB`).
Because `eval_cp` is clamped to `[-30000, +30000]` (`i16`) and `dis_cp` is in `[0, 30000]` (`u16`), we pack both into the existing `i32` payload:
```rust
#[inline(always)]
fn pack_eval_dis(eval_cp: i32, dis_cp: i32) -> i32 {
    let e = (eval_cp.clamp(-30000, 30000) as i16 as u16) as u32;
    let d = (dis_cp.clamp(0, 30000) as u16) as u32;
    ((e << 16) | d) as i32
}

#[inline(always)]
fn unpack_eval_dis(packed: i32) -> (i32, i32) {
    let u = packed as u32;
    let eval_cp = ((u >> 16) as u16 as i16) as i32;
    let dis_cp = (u & 0xFFFF) as i32;
    (eval_cp, dis_cp)
}
```
* **Extra RAM:** `0 bytes`.
* **Extra cache lookups:** `0`.

### 2.3 PUCT-Style Selectivity in `negamax` (`search.rs:864` & `search.rs:1048`)
With `(child_eval, child_dis)` unpacked from `leaf_eval_with_dis(&child)`:
1. **Volatility-Modulated Moonbird Futility Reduction (`search.rs:1048`):**
   ```rust
   let vol_credit = (child_dis * DIS_FUT_NUM) >> 7;
   let shortfall = (alpha.saturating_add(child_eval) - vol_credit).max(0);
   new_depth -= (shortfall / MB_FUT_DIV).clamp(0, 64);
   ```
2. **Volatility-Guarded Reverse Futility Pruning (`search.rs:864`):**
   ```rust
   let rfp_margin = RFP_MARGINS[(depth.max(1) - 1) as usize] + ((node_dis * DIS_RFP_NUM) >> 7);
   if static_eval + rfp_margin < alpha {
       self.heuristic_used = true;
       return alpha;
   }
   ```
3. **Zero-Diff Fallback:** When `dis_cp == 0` (e.g., on `TNN1`/`TNN2` nets before `TNN3` weights are loaded), `vol_credit == 0` and `rfp_margin` is bit-for-bit identical to `main`.

---

## 3. Overnight Autonomous Execution Queue (Branch-Isolated)

| Step | Branch | Description | Gate / Verification | Merge Target |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `exp/persistent-tree` | Zero-RAM TT generation aging (`age: u8`), fail-low refutation preservation, `×3/4` history decay | 49 unit tests + SPRT @ 100ms & 5k nodes vs `main` | `staged/main` |
| **2** | `exp/disagreement-head` | `TNN3` parser/head in `nnue.rs`, 32-bit `(eval_cp, dis_cp)` packing in `ecache`, volatility-modulated RFP & futility in `search.rs` | Unit tests + zero-node-diff bench proof on `TNN1`/`TNN2` | `staged/main` |
| **3** | `exp/botzone-simd` | Add AVX-2 (`x86-64-v3`) and AVX-512 (`x86-64-v4`) compilation & runtime dispatch in `tools/build_botzone.py` + `nnue.rs` | `glibc <= 2.17` ELF check + Docker Linux turn test + NPS bench | `staged/main` |
| **4** | `titaniumtest` (`main`) | Add `--cand-reset-every-move` / `--base-reset-every-move` for honest one-shot vs persistent Botzone SPRT testing | `cargo test` + asymmetric match verification | `titaniumtest` |
| **5** | `exp/tonight` (`titanium-ataxx`) | Complete 31.57M 5k relabeling, auto-shutdown cloud sandboxes, assemble v5 dataset, run Capacity vs Information diagnostic (`W=256` vs `W=512`) & saturate Big Net on Modal GPU | Validation loss plateau + `nnerr_analyze.py` ablation report | `staged/main` |
| **6** | `exp/tonight` (`titanium-ataxx`) | Distill `fut_small.tsn` (`TSN1`) and train `TNN3` Disagreement Head against saturated Big Net | SPRT gate of new nets | `staged/main` |
