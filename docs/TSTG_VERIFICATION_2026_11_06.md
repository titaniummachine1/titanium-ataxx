# TSTG verification, 2026-11-06 (D413/D414, Opus)

The user asked me to verify the "spiral of despair": a forensic dossier pasted from another model, covering the TSTG evaluation work of 11-04..11-06.
I changed no model file and launched no gate. Raw outputs are in `logs/tstg_verify/`. Tools: `crates/titanium-cli/src/bin/verify_tstg_perspective.rs` (new, read-only),
`docs/analysis/tstg_jump_share.py`, `docs/analysis/tdat_stm_share.py`, and the existing `debug_spatval`, `debug_ply3` and `debug_avx_vs_scalar`. The code is branch exp/tstg-main
(HEAD 5e5e445, ancestor of origin/main OK) with exe B554CD6B. Main NNUE = bundled 46FCB5A4.

## 1. What the "TSTG vs main" numbers actually measured (the key fact)
- Up to commit 00af77f (11-04 20:13), `set_tstg` only attached the model: **the NNUE stayed the evaluator and TSTG only ordered moves**
  (3c56326: `self.tstg` is used only in `score_into`). So D409/D410 (+22.6 at 100 ms), D411 (-1.7 at 100 ms and at 5k nodes, 200 g each) and the 11-04 17:03
  v11 run (`sprt_100ms_tstg_v11_vs_main.log`: 74-117-7, -76.7 +-49, 198 g, baseline = the same exe without --tstg, NOT 2c40ae5) are all
  **NNUE eval + TSTG move ordering**. None of them tested TSTG as an evaluator.
- From 8dee67d (11-05 05:41) `set_tstg` sets `nnue = None`: TSTG becomes the **sole evaluator** ("Pure TSTG"). serve's `--tstg` forces `--nnue none` since 7071e42.
- **Every pure-TSTG game on record: 0 points in 128 games** (0 wins, 0 draws). The table below has every log with a `# ti sprt` header since 11-05:

| log | model | clock | result |
|---|---|---|---|
| smoke_test, test_4g, test_4g_v14, test_4g_time100ms(_v14) | v12-v14 | 5k / 100 ms | 0-2, 0-4 x4 |
| gate_tstg_v16_time100ms / nodes10k | v16 | 100 ms / 10k | 0-20, 0-4 |
| gate_tstg_v11_nodes10k | data/tstg_v11_saturated | 10k | 0-4 |
| gate_tstg_v17_time100ms | v17 | 100 ms | 0-10 |
| gate_tstg_v23_5knodes | v23 | 5k | 0-18 |
| tstg_v24_5knodes / 20g_100ms | v24 | 5k / 100 ms | 0-16, 0-20 |
| tstg_v10_best_10g_100ms | v10_best | 100 ms | 0-10 |
| tstg_v24_4g_5knodes_nofutsn, tstg_v10_4g_5knodes_nofutsn_w0 | v24, v10_best, SmallNet futility OFF | 5k | 0-4, 0-4 |

  The same v11 weights went from a 39% score as an ordering head (NNUE eval) to 0/4 as the evaluator. At **fixed 5k nodes** the result is also 0%,
  so speed cannot explain it: the static evaluation itself is hundreds of Elo worse.
- The commit message of 8dee67d claims "50-game clock gate +24 -23 =3, 20-game +11 -7 =2, beat main champion". **No such log exists** in logs/ or logs/gates/.
  The only near-even result after 11-05 is `sprt_gate_120g_main_eval.log` (56-59-5, -8.7, 120 g), and its candidate runs **without `--tstg`** (NNUE vs NNUE).
  The dossier's claim that the "win" was main playing itself is consistent with this. The 50 g log itself could not be found.

## 2. Static-eval quality on held-out data (new measurement, the decisive one)
`verify_tstg_perspective data/nnue/packed/val_v9_clean.tdat 600000 <models>`: 551,301 non-proven rows, target = train_tstg_v9's target (0.55/0.25 result blend),
evals clamped to +-1500.

| evaluator (correct board) | corr vs target, stm=0 (466k) | MAE | corr, stm=1 (85k) | MAE | corr vs NNUE (stm=0) | MAE vs NNUE |
|---|---|---|---|---|---|---|
| **main NNUE 46FCB5A4** | **+0.666** | 276 | **+0.895** | 125 | 1 | 0 |
| tstg_v10_best (= v23, = v10_pol_dis_fit value) | +0.511 | 306 | +0.705 | 187 | +0.714 | 137 |
| tstg_v17_champion (= v22 value) | +0.312 | 381 | +0.622 | 192 | +0.455 | 271 |
| tstg_v11_saturated | +0.304 | 346 | +0.631 | 192 | +0.445 | 194 |
| tstg_v11_final | +0.269 | 354 | +0.589 | 210 | +0.390 | 197 |
| tstg_v24_champion_master | +0.390 | 572 | +0.669 | 513 | +0.589 | 531 |

- The best TSTG file (v10_best) explains far less than the NNUE does. v24's MAE of 531 cp vs the NNUE confirms that the hand overwrite in `build_champion_tstg.rs` wrecked the calibration.
- The 11-05 claim "v17 +0.9620 correlation with Main NNUE, 64.8 cp MAE" (D412 / HANDOFFS) **does not reproduce** on this held-out set: it measures +0.455 / 271 cp.
  That number came from the 31M set (tstg_v10_31m.bin) and is probably in-sample. Not re-run on that file.

## 3. NEW BUG the dossier missed: train_tstg_v9.rs swaps colours on white-to-move rows
- The hub stores `.tdat` boards by colour (`occ0` = black, `occ1` = white). The target is from the side to move, and the main NNUE trainer builds `Board::from_occ(occ0, occ1, 0, stm)` with no swap.
- `train_tstg_v9.rs` `Rec::to_board` swaps the bitboards when `stm == 1` **but keeps `turn = 1`**, so `me()` returns the opponent's stones.
- Measured with v10_best on the 85k white-to-move val rows: corr +0.705 on the correct board vs **+0.100 on the trainer's board**.
- **train_v9.tdat: 4,202,404 of 28,990,531 rows (14.5%) are white-to-move.** Every Modal epoch fed those rows with inverted colours (`docs/analysis/tdat_stm_share.py`).
- `train_tstg.rs` and `train_policy_frozen_backbone.rs` (us/them records) map colours correctly. Only the v9 trainer, the `v10_best` line, is affected.
  It is one plausible source of the "more own stones = worse" spatial weights, but it is not proven to be the whole cause.

## 4. Claim-by-claim verdict on the dossier
| claim | verdict | evidence |
|---|---|---|
| build_champion_tstg overwrote w_mat/b_val and phases 12-15 and saved over v24 + both tstg_best files | **TRUE, with a correction**: the backbone it loads is **data/tstg_v11_saturated.ttst**, not tstg_v10_best, and the heads come from data/tstg_best.ttst (which it then overwrote). It also forces channel pairs (0,1)(2,4)(5,6)(7,3) to be exact negatives, an assumption nothing in training supports | source; sha256 v24 = data/tstg_best = data/nnue/tstg_best = 9159aeba |
| v10_best is the untouched "100-epoch" Modal checkpoint | **PARTLY**: it is the trainer's best-val save on the Modal volume (fetch_checkpoints.py); `max_epochs=100` is a cap with early stop, and the run resumed from an earlier best. The training log was not copied out, so the real epoch count is unknown. **The Modal ledger has no row for any TSTG training run** (A9 violation) | modal_train_tstg.py, logs/modal_spend.csv |
| v10_best w_mat / b_val values | **TRUE** (exact match) | logs/tstg_verify/debug_ply3.txt |
| AVX2 = scalar bit-identical | **TRUE for eval/dis/legal-move scores** (0 / 100, 0 / 6690). The raw per-square policy arrays differ (3651 / 4900 squares); harmless only if non-move squares are never read | debug_avx_vs_scalar.txt |
| Bug A: clone scores -72 cp vs jump at ply 5 | **TRUE as arithmetic** (jump c5d7 +159, clone c6 +87). The real problem is bigger: TSTG scores EVERY child **+70..+159 for the mover** where the NNUE says **-411..-688**. With 2 stones vs 11 it thinks the 11-stone side to move is losing by 159 cp. That is a sign error on lopsided boards, not a clone/jump nuance | debug_avx_vs_scalar.txt sec. 2 |
| "11-12 jumps out of the first 15 moves" | **EXAGGERATED**: candidate jump share in the first 15 own moves 0.56-0.67 (8-10 of 15) in every TSTG log; main's own share is 0.55 (120 g NNUE-vs-NNUE control) and 0.63 in the v11 hybrid run. Not a distinguishing pathology | tstg_jump_share.py |
| Bug B: policy head never trained (None target) | **TRUE for v10_best**: `forward_and_grad_step(.., None, lr)` in train_tstg_v9.rs:271; its head L2 norms 3.8536 / 3.3466 / 2.6627 are exactly the hand defaults. The 11-06 postmortem sec. 4 calls them "fully populated", which is **wrong**. v24 took its heads from the old data/tstg_best (training status not verified, that file is now overwritten). Ordering only changes nodes-to-depth: it cannot produce 0% at fixed nodes | train_tstg_v9.rs; debug_ply3 |
| Policy ON = +73% nodes at depth 4 | TRUE (2654 -> 4598), but depth 5 is 7396 vs 7999 (+8%): one position, noise-level | debug_ply3.txt |
| Bug C: SmallNet futility on TSTG | **TRUE that it is mis-calibrated** (SmallNet bounds the NNUE child eval; 9c310ad re-enabled it for TSTG). **NOT the cause of 0%**: both --no-fut-sn runs also lost 0-4 | logs |
| dis scale ~4x the NNUE's | TRUE (v10_best dis 139-152 cp vs NNUE 31-43 on the ply-3 boards) | debug_ply3.txt |
| Task 2 (fit the 24 policy and dis floats) still to do | **ALREADY DONE** on 11-06 13:07/13:27 (commits 2e279c2, 5e5e445 -> tstg_v10_polfit / tstg_v10_pol_dis_fit). Never gated, never documented in DECISIONS; the value is identical to v10_best | sha + val table |

## 5. What this means
1. TSTG as the **sole evaluator** has never been close: 0/128, fixed-node legs included. Readout patches on `v10_best` (the dossier's Task 1) can only move
   it toward the best LINEAR readout of the frozen Z features. That ceiling is unknown and cheap to measure. Under ~0.62 corr vs target on stm=0 rows (NNUE 0.666), no tweak will make it play.
2. The 11-04 positive results were TSTG as a **move-ordering head on top of the NNUE**. That path is real: about parity at 200 g x2 (D411), and +22.6 in D410.
3. The v10_best backbone was trained on 14.5% colour-swapped rows. Any further training of that line must fix `Rec::to_board` (turn 0 for the swapped board, or no swap) first.

## 6. Not checked
TT pollution (postmortem root cause 3); the `0000` moves the candidate plays mid-game in gate_tstg_v23 (pass legality); the 31M-set correlation claim on its own file;
the endgame-only quality (val rows are mixed phases); the Modal training log (not in the repo); whether the v17 lineage has its own data bugs; per-phase breakdown of the val table.


## 7. Readout ceiling test (D418, `tstg_readout_ceiling`, logs/tstg_verify/readout_ceiling_*.txt)
Frozen v10_best up to Stage 3 (Z = 49 squares x 8 channels) + the 6 macro deltas; per-phase ridge readout fit on 948,380 train_v9 rows
(correct colours, strided), lambda 30; scored on val_v9_clean (466k stm=0 / 85k stm=1 rows). corr / MAE vs the training target:

| readout | stm=0 | stm=1 | vs NNUE (stm=0) |
|---|---|---|---|
| main NNUE | **0.666** / 276 | 0.895 / 125 | 1 |
| v10_best as trained | 0.511 / 306 | 0.705 / 187 | 0.714 / 137 |
| **best linear readout of frozen Z + macro** (184k rows: 0.544) | **0.572** / 293 | 0.860 / 141 | 0.851 / 94 |
| + board-pooled Z, pairwise products, pooled x macro (92 extra) | 0.575 / 292 | 0.863 / 139 | 0.855 / 93 |
| macro deltas only (7 features) | 0.444 / 331 | 0.750 / 189 | |

Meaning: a mis-trained readout ("drift") explains only 0.511 -> 0.572. Nonlinear global pooling of Z adds 0.003. **The information the NNUE uses is
not in the 8-channel Z features**: Z adds +0.13 corr over seven hand counts, and the NNUE adds another +0.09. The readout-projection fix in the dossier's Task 1 is capped at
0.572: not enough. Not tested: whether retraining the backbone with the colour fix moves Z (needs a training run); wider Z; radius-2 inputs.

| + move-relation block (D420: per side clones, jumps, best gain, best 2-ply net, holes punished; 16 features) on Z + macro | 0.577 / 291 | 0.867 / 137 | 0.859 / 91 |
| macro + move-relation block only | 0.500 / 317 | 0.818 / 165 | |

Move relations carry signal (macro 0.444 -> 0.500) but Z already holds most of it; linear 2-ply summaries do not close the gap (D420).

## 8. User's representation, linear probe (D421/D422, `tstg_reach_probe`)
| features (linear one-hot readout) | stm=0 | stm=1 |
|---|---|---|
| plain occupancy + macro | 0.529 | 0.822 |
| + real reach maps (both sides, 1-4 deep, landing on empties) | 0.562 | 0.848 |
| + imagined reach maps (through any square) | 0.539 | 0.829 |
| + both | 0.565 | 0.850 |
| NNUE | 0.666 | 0.895 |

Every linear readout of per-square features stops near 0.57. The NNUE's lead is its nonlinearity, so a linear probe cannot rank representations
for a nonlinear net. The next test must train the same small nonlinear net on plain vs relational inputs (D422).
