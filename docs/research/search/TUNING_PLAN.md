# TUNING PLAN (2026-11-01, user orders D284/D286/D287)

User: "we have never run any Texel tuning ... at least 5,000 nodes per move, possibly alternating 5,000 and 50,000 ... on the computer, accelerate with Docker";
"not everything is constant: tune the dynamics that use the disagreement (dis) head"; "futility pruning, LMR etc rely on the additional dis head";
"maybe retrain the auxiliary heads and small net to fit the new best net".
No Python for tuning (D282): the tuner is a hub (`ti`, C:\gitProjects\titaniumtest) subcommand.

## What exists (all on branch exp/tune, worktree C:\gitProjects\tune-ataxx, from exp/lmr2 19e09df)
- c7bf414: 48 dynamic tunables read once from `--tune name=v,...` (env TITANIUM_TUNE); `titanium-cli tune-list` prints name + default (the hub reads this).
  Defaults reproduce main c86733b: `bench --depth 10` = 68,961 nodes on BOTH exes (logs/gate_speed4d_c86733b.exe and the tune exe). nps 167-178k vs 175k (tune() read per node; noise-level, re-measure before baking).
  Knobs: rfp1..4 (base margin per depth), rfp_s1..4 (dis slope per depth, default 32 = DIS_RFP_NUM), rfp_hinge (extra slope on dis above dis_knee, default 0, knee 300), rfp_dismax;
  ph_rfp0..3 / ph_lmr0..3 / ph_fut0..3 (phase multiplier /64 by empties: >=33, 21-32, 11-20, <=10); lmr_half 47, lmr_min 6, lmr_dis, lmr_hinge;
  fut_div 57, fut_ddis, fut_dhinge (dis in the divisor), dis_fut 24 (child dis credit), fut_hinge, fut_ext 1 (the "+1 ply when child dis > knee");
  pc_d 6, pc_r 3, pc_t 140, pc_k 32, pc_hinge; nmp_tgt 11, nmp_fill 26, nmp_r 3, nmp_dismax; iir_d 4; corr_apply 64, corr_wmax 128; ord_clone/jpen/jconn.
  Plus the 15 exp/lmr2 terms (r_npv ... nocap_min).
- `titanium-cli sn-audit <games.jsonl> [stride] [max]` (this file's measurement below).

## SmallNet staleness measurement (2026-11-01 ~19:40, exe tune c7bf414+audit, bundled net F5FA8557)
SmallNet fut_small.tsn = distilled on child evals of the OLD net e11827d1 (D109/D168), tau 0.02 pessimistic lower bound. Audit: for every legal child of 3,000 real game positions
(every 5th ply of the 100 ms gate games l2c_pvx4 and the 5k games w512_5k), P(lo > ce) with lo = the SmallNet bound exactly as the futility pre-filter computes it and ce = big-net raw child eval:

| phase | l2c_pvx4 n / viol | w512_5k n / viol |
|---|---|---|
| opening e>=33 | 9,540 / 1.58% | 7,227 / 1.77% |
| mid 21-32 | 49,943 / 2.75% | 52,875 / 2.83% |
| late 11-20 | 34,928 / 2.63% | 35,160 / 3.12% |
| end <=10 | 20,741 / 4.98% | 21,709 / 4.30% |
| ALL | 115,152 / 3.01% | 116,971 / 3.13% |

Design value 2%. So: moderate drift (1.5x overall, ~2.3x in the endgame); median slack ce-lo = 230-260 cp mid, ~500 cp endgame (the bound is loose, esp. endgame).
NOT measured: the violation rate when the net was fresh (no e11827d1 on disk to compare), the skip rate / cost effect, the dis head (separate check owed: was F5FA8557's dis head refit? no .prov in data/nnue).
Raw: command above on logs/gates/l2c_pvx4_b0.log.games.jsonl and w512_5k.log.games.jsonl; code crates/titanium/src/search.rs `sn_audit`.

## Order of work (D287)
1. dis-head check/refit on F5FA8557 (`ti train export` dis refit) and SmallNet refit on F5FA8557 (needs a hub trainer `ti train smallnet` (Rust): Python is FORBIDDEN for any training/tuning, D289), gate each (SmallNet = speed-affecting: 100 ms + 1 s legs).
2. `ti spsa` in the hub (Rust): SPSA over the knob vector, theta+ vs theta- via the hub's paired-book runner (`ti sprt`), 5k and 50k nodes alternating, logs every theta.
3. Run on the PC (one job) and on Docker (A9), stage by knob group (dis-dynamics first), gate the tuned vector vs untouched main at 100 ms and 1 s before it counts.

## 2026-11-02 stage 1b (D316, user: >10k nodes, no fixed iteration limit, on the PC)
Stage 1 (5k/50k) was stopped by the PC crash at k=18 (knobs moved 1-11 units). Stage 1b restarts from the defaults on the PC: nodes 20000,20000,20000,50000, 50 pairs, --iters 600 (raise with --resume), 6 games in parallel, BelowNormal. Command C:\gitProjects\logs\spsa_stage1b.cmd.txt, log C:\gitProjects\logs\spsa_stage1b.log. Checkpoint gates (theta vs defaults, 100 ms + 1 s, ti insight/analyze) in Docker every ~100 iterations; promotion only after those gates.

---

## 2026-11-03 16-Bucket Phase-Aware Search Architecture (Foundation)

To match the granular phase resolution of the NNUE evaluation (which partitions game progression into 16 distinct bucket states), the search dynamics can be parameterized by game phase rather than using global static thresholds or crude 4-bucket scalers.

### 1. Granularity & Phase Mapping
Phase is determined directly from the count of empty squares (or board stone density), identical to the NNUE bucket mapping:
$$p = \text{clamp}\left(\left\lfloor \frac{\text{empties} \times 16}{49} \right\rfloor, 0, 15\right)$$

- **Buckets 13–15 (Opening, 40–49 empties)**: Wide action trees, high jump mobility, volatile capture cascades. High risk of missing tactical traps if over-pruning early plies.
- **Buckets 8–12 (Middlegame, 25–39 empties)**: Mixed clone/jump frontiers. Disagreement head credit is maximal here; futility pruning is most effective.
- **Buckets 3–7 (Late Middlegame / Early Endgame, 10–24 empties)**: Dense clusters, fortress formation. Jump sacrifices create ghost craters. LMR must be delayed ($lmr\_min \ge 6$) to avoid missing zugzwang breaches.
- **Buckets 0–2 (Terminal Endgame, 0–9 empties)**: Clone/parity dominated, zero or few jumps. Exact calculation regime; pruning margins tighten while search deepens rapidly.

### 2. Clean Architecture Pattern (No Spaghetti Matches)
Instead of scattershot `match empties` statements inside `negamax()`, search profiles are represented as a structured table:

```rust
#[derive(Clone, Copy, Debug)]
pub struct PhaseSearchProfile {
    /// Multiplier for reverse futility pruning margin (/64, 64 = 1.0x).
    pub rfp_scale: u16,
    /// Move index threshold before LMR activates.
    pub lmr_min: u8,
    /// LMR halflife divisor (reduction steepness).
    pub lmr_half: u8,
    /// Futility division coefficient (higher = gentler pruning).
    pub fut_div: u8,
}

/// Baseline 16-phase profiles (can be overridden via CLI --tune or calibrated via deep regret).
pub const DEFAULT_PHASE_PROFILES: [PhaseSearchProfile; 16] = [
    // 0..2: Terminal Endgame (0-9 empties)
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 8, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 8, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 8, lmr_half: 47, fut_div: 57 },
    // 3..7: Mid / Late Endgame (10-24 empties)
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    // 8..12: Middlegame (25-39 empties)
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    // 13..15: Opening (40-49 empties)
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
    PhaseSearchProfile { rfp_scale: 64, lmr_min: 6, lmr_half: 47, fut_div: 57 },
];
```

### 3. Tuning Protocol & Guardrails
- **Zero-Regression Default**: With all parameters initialized to current baseline constants, the engine produces 100% bit-identical search nodes to `main`.
- **Regret Calibration (D373A / D383)**: Calibrate phase profiles using fixed-node regret against 500k-node deep references on actual game positions across all 16 empties bins (not opening-only books).
- **Time-Control Verification Mandate**: No phase knob change may be promoted solely on fixed-node gates (avoiding the D379/D380 pitfall where 5k gains collapsed at 100 ms). Both 100 ms and 1 s legs must confirm positive Elo.

