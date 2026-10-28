# 🧠 Master Backlog & Disagreement Head Specification (`TNN3` + `TSN1`)

This document tracks the immediate execution backlog and exact code-level integration for Titanium's 4-phase NNUE + PUCT-in-Alpha-Beta roadmap.

---

## 1. Strategic Order of Operations (Why This Exact Sequence Wins)

$$\text{Phase 1: Big Net (`TNN2`)} \;\longrightarrow\; \text{Phase 2: SmallNet (`TSN1`) \& SVD Downscale} \;\longrightarrow\; \text{Phase 3: Disagreement Head (`TNN3`)} \;\longrightarrow\; \text{Phase 4: Active-Learning Data Loop}$$

1. **Why Big Net First (`Phase 1`):**
   Both the incremental futility pre-filter (`SmallNet`, `crates/titanium/src/smallnet.rs`) and the Disagreement Head target ($\Delta_{\text{true}}(s) = |\text{Search}_{5k}(s) - \text{StaticEval}_{\text{new}}(s)|$) are defined relative to the **new Big Net's static evaluation**. Training and freezing the Big Net first guarantees zero distribution drift downstream.
2. **Why `SmallNet` & Downscaling Second (`Phase 2`):**
   As documented in `smallnet.rs:7-8`, `fut_small.tsn` ($W=32, H=16$, `~90 ns/child`) is fitted to a specific `nnue_main.tnn` and must be re-distilled whenever `nnue_main.tnn` changes. Upgrading its target to $\min(\text{BigNet}(\text{child}), -\text{Search}_{5k})$ prevents the 90 ns pre-filter from pruning sacrificial capture chains. Simultaneously, SVD + soft-logit distillation produces a $W=256$ Big Net candidate (`~2x` faster than $W=512$).
3. **Why Disagreement Head Third (`Phase 3`):**
   With the new static evaluator frozen, we compute the exact 5k-vs-static disagreement $\Delta_{\text{true}}(s)$ across all relabeled positions in a single ~45-second pass and train the zero-cost `TNN3` auxiliary head (`dis_w[bucket][32]`, `< 1 ns` inference).
4. **Why Self-Relabeled External Geometries Fourth (`Phase 4`):**
   Because Titanium outperforms external engines, **we never fine-tune directly on weaker external teacher evaluations** (no naive Lc0/external fine-tuning). Instead, external datasets (`KataGo_Ataxx_202604` on ModelScope, `sotax.zip`, `zzzzz`, Botzone `atx_positions.bin`) supply **raw board geometries `(occ0, occ1, blk, stm)`**, which we filter by high predicted $\hat{\Delta}$ (Active Learning) and relabel **100% with Titanium's own 5k-node search**.

---

## 2. Exact Code Integration in `crates/titanium/src/`

### 2.1 Zero-Cost `TNN3` Head in `nnue.rs`
In [`crates/titanium/src/nnue.rs:256-324`](../crates/titanium/src/nnue.rs), `Nnue::head` already computes the shared 32-float hidden activation vector (`h1[..16]` and `h2[..16]`) for the active empties `bucket`.

Adding the Disagreement Head requires only one additional 32-element dot product per `bucket`:
```rust
let dis_row = &self.dis_w[bucket * k..(bucket + 1) * k];
let d1: f32 = dis_row[..l1].iter().zip(&h1[..l1]).map(|(a, b)| a * b).sum();
let d2: f32 = dis_row[l1..].iter().zip(&h2[..l2]).map(|(a, b)| a * b).sum();
let dis_cp = (400.0 * (d1 + d2 + self.dis_b[bucket]).max(0.0)).round() as i32;
```
* **Parameter footprint:** `8 buckets * (32 + 1) floats = 264 floats = 1,056 bytes`.
* **Runtime cost:** 32 FMAs = **1 AVX2 instruction (~0.8 ns)** when `head()` is already running (`0.00 ns` on `ecache` hits).

### 2.2 Zero-Memory Packing in `ecache` (`search.rs:550`)
In [`crates/titanium/src/search.rs:550`](../crates/titanium/src/search.rs), `Searcher::ecache` is a $2^{16}$-entry table of `(u64, i32)` (`1 MiB`).
Because `eval_cp` is clamped to `[-30000, +30000]` (`i16`) and `dis_cp` is in `[0, 4000]` (`u16`), we pack both into the existing `i32` payload:
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
Currently, [`search.rs:1048`](../crates/titanium/src/search.rs) reduces every move after the first by how far its child static eval falls short of `alpha`, divided by a static constant `MB_FUT_DIV = 57`:
```rust
new_depth -= (alpha.saturating_add(child_eval) / MB_FUT_DIV).clamp(0, 64);
```
With `(child_eval, child_dis)` unpacked from `leaf_eval_with_dis(&child)`:
1. **Volatility-Modulated Moonbird Futility Reduction (`search.rs:1048`):**
   ```rust
   // High child_dis (tactical volatility) grants a volatility credit that prevents premature reduction/pruning;
   // low child_dis (quiet/locked board) allows tighter pruning.
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
3. **Root Disagreement Time Allocation (`search.rs`):**
   Scale the per-move time budget by $\text{clamp}(0.75 + 0.005 \cdot \hat{\Delta}_{\text{root}}, 0.70, 1.60)$ so Titanium invests extra search depth on tactically sharp turns and banks clock on routine turns.

---

## 3. Dataset Inventory & Expansion Policy

| Dataset | Rows | Current Status | Action Plan |
| :--- | :--- | :--- | :--- |
| **`dis_positions.bin`** (from `final_train_v4.npz`) | `31,568,598` | 🔄 Relabeling at 5k nodes (`3,133+` chunks dense, `92.6%` tail reach) | Finish relabel $\to$ Train Phase 1–3 nets |
| **Botzone Top-Bot Scrape (`atx_positions.bin`)** | `95,235` | ✅ On disk (`data/nnue/atx_scraped/`) | Deduplicate hashes $\to$ Relabel at 5k nodes |
| **`KataGo_Ataxx_202604` (ModelScope)** | Multi-million | 📌 Cataloged in `COMMUNITY_RESEARCH_2026.md` | Extract **raw board states only**, filter top 30% highest $\hat{\Delta}$, relabel with **Titanium 5k search** |
| **`sotax.zip` + `zzzzz` + Moonbird sets** | Multi-million | 📌 Cataloged in `COMMUNITY_RESEARCH_2026.md` | Extract unique opening/blocker geometries, relabel with **Titanium 5k search** |

---

## 4. Botzone Engine Persistence Optimization (4s Persistent vs. 6s One-Shot)
* **Detailed Strategy & Gating Plan:** [docs/PERSISTENT_VS_ONESHOT_BOTZONE.md](PERSISTENT_VS_ONESHOT_BOTZONE.md)
* **Core Concept:** Instead of restarting Python and clearing the TT every turn for 6.0s, run in persistent mode (`enable_keep_running`) for 4.0s with **5 state-preservation optimizations**:
  1. TT Subtree recycling & PV move seeding.
  2. Countermove / Continuation history (`conthist`).
  3. History table gentle aging (`*h = *h * 7 / 8`).
  4. Killer moves 2-ply shifting.
  5. Static eval cache (`ecache`) persistence across turns.
* **Gating:** Head-to-head SPRT (`3800 ms` persistent with state vs. `5700 ms` cold restart) via `titaniumtest sprt`. Gate winner deploys as default on Botzone.
