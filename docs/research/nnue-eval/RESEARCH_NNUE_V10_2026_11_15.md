# NNUE retrain on train_v10 (self-play data) - 2026-11-15 (D806-D819)

## Result
- Candidate `data/nnue/proxy/nn_v10sat_nocal.tnn`, sha256 **812C2118**: the champion net 46FCB5A4 fine-tuned on
  `train_v10` until saturation.
- **5,000 nodes, Botzone rules, policy head ON on both sides:** +56.8 +-26.4 Elo over 630 games (366-264-0), LLR +2.99,
  SPRT **PASS** (gate tag `gate/2026-11-15-be296ec6-a87CB1BEF-B96B31BD6-vs-be296ec6-nodes5000`).
- **Control with the same setup:** the champion trunk re-exported (7CA724B4) + policy scored -2.6 +-29.2 over 400 games
  (no tilt, value-lost diff z -0.11), so the gate setup is sound.
- **Play quality** (`ti insight`):
  - Value lost per move is 4.27 cp better (paired z +4.26). Blunders: 4.4% vs 5.6% of moves.
  - The gain sits at plies 40-59 (+4.49 cp/move, z +2.56) and plies 60-79 (+29.65 cp/move, z +4.42, +121.5 cp per game).
  - Plies 0-39 are equal and ply 80+ is undecided (z -0.49).
- **Time legs** (100 ms, then 1000 ms): running when this was written, see DECISIONS D820+.

## Pipeline (all hub / Rust, no Python)
1. **Games.** `ti gen` (binary only since D807): `batch_NNNN.tgam` holds the played move, depth and score per ply;
   `batch_NNNN.tbst` holds the engine's own best move, its score and the DAG game id. They differ in 3.6% of plies, where
   novelty played a different move; both are kept. The old JSON (1.34 GB) was converted with `ti dag jsonl-to-bin`;
   the JSON files are kept.
2. **Rows.** `ti data games-tdat --tgam a.tgam,b.tgam --best-scores`:
   - 8,631,223 rows from 83,327 games.
   - cp = the best-move score (the value of the position).
   - outcome = game result from the mover's view.
   - Duplicates kept (5.1% exact overlap with train_v9; 27.9% repeats inside the new rows); D4 used only to measure.
3. **Append.** `ti data append`: `train_v10` = train_v9 (29,092,361 rows, untouched) + new rows = 37,723,584 rows. Kata/aux
   sidecars are zero-padded for the new rows, and the chains are merged. Lambda copies `train_v10_lam9/_lam7` are made
   but not used yet.
4. **Health checks.**
   - `ti data chain-audit`: one-ply label consistency fine.
   - `check-leak` vs val_v9_clean: 0.08%.
   - New-row results contradict proven scores in 0.46% of rows (train_v9: 1.58%).
5. **Training.** `ti train modal launch`, Modal T4, about 2,645 s, about $0.45, run `nn_v10sat`:
   - Init = the champion itself (C14).
   - Champion recipe: lr 5e-5 -> 1e-6 cosine, bs 16384, ema 0.9, proof-blend 0.5, result weight 0.55/0.55/0.25.
   - **Saturation rule (user, D812):** `--plateau 0 --patience 37` = stop after 4 epochs without a val gain. Early stop
     came at checkpoint 352 (88,000 steps = 38 epochs). The last 4 epochs gained 0.007%.
   - Val mse 0.028436 -> 0.027929 (-1.8%).
6. **Export.** `ti train export --net runs/nn_v10sat/nn_v10sat/best.tnn --data data/nnue/packed/train_v10 --out
   data/nnue/proxy/nn_v10sat_nocal.tnn --result-weight-phase 0.55,0.55,0.25` (no calibration; refits the disagreement head).
7. **Gate.** `ti gate --test HEAD --base main --kind eval --rules botzone --nodes 5000 --test-args "--network NET --policy
   C:/gitProjects/mainpromo-ataxx/data/nnue/policy_main.bin --bz --tt-bits 22" --base-args "--bz --tt-bits 22"`, run
   from `C:/gitProjects/mainpromo-ataxx`.

## Pitfalls found today (each cost time; the fix or rule is next to it)
| # | What went wrong | Effect | Fix / rule |
|---|---|---|---|
| P1 | **The engine turns the bundled policy head OFF for any custom `--network`** unless `--policy FILE` is given (`crates/titanium/src/config.rs:115-118`, `policy.rs:610-614`). | Every candidate gate D814-D816 ran without policy: -83 to -150 Elo, even the champion's own trunk (-127/-134). I first blamed the data, then the dis-head refit (D816, withdrawn in D817). | `ti gate` now adds the bundled `policy_main.bin` to any side that loads a custom net without `--policy`, and says so in the header (opt-out `--no-auto-policy`). D819. |
| P2 | `ti train modal fetch` downloads **every** checkpoint of a `--keep-all` run. | About 10 GB onto the SSD for one run (stopped at 3.9 GB). | `fetch` now gets only `best.tnn`, `RECORD.txt`, `EARLY_STOP.txt`, `GUARD_TRIPPED.txt` and `gpu_state.txt` by default; `--all` gets everything. D819. |
| P3 | `ti train modal launch` called `modal run ti_modal.py` without an entrypoint; the launcher has two (`main`, `gen_main`). | Launch failed. | `src/modal.rs` names `::main`. |
| P4 | The launcher image renamed the CUDA binary to `ti-cuda` (uncommitted edit of an earlier session), but `run()` still called `ti` (= the CPU-only build). | Run exited: "built without --features cuda". | `modal/ti_modal.py` `run()` calls `ti-cuda`. |
| P5 | The old recipe stops at a fixed 30,000 steps / plateau 3. | Not saturation as the user defines it. | Saturation = `--plateau 0 --patience <4 epochs of checkpoints>`, steps only as a cap (D812). |
| P6 | The champion guard refuses `--no-fut-sn` on `main`, while GATE_RULE 8 says evaluator gates disable the SmallNet on both sides. | The rules conflict. | Gates ran with the SmallNet ON on both sides (as the engine ships). **Open for the user** (LAWS A8). |
| P7 | `titanium-ataxx` sits on `exp/tstg-main`, which does not contain `origin/main`. | Gates must not run from it (B0). | Run gates from `C:/gitProjects/mainpromo-ataxx`. |
| P8 | `ti gen` wrote 1.34 GB of JSON game files (an agent decision of 2026-11-13, never the user's). | JSON is forbidden for data (D806). | Binary `.tgam` + `.tbst` only (D807). |

## What was NOT checked
- Time legs (running), ply 80+ (undecided at 5k), and the lambda-target variant (`train_v10_lam9`).
- A low-lr anneal after the early stop: training stopped at lr about 2.9e-5 of the cosine.
- Position-level aggregated WDL.
- A SmallNet refit on the new net: the futility small net 8F479ECB is still fitted to 46FCB5A4.
- Why today's dis-head refit gives a different head than the 11-02 export: it is not shown to be harmful (P1 explained
  the losses).

## Raw data
- Gate logs: `C:/gitProjects/logs/gate_*`, `C:/gitProjects/mainpromo-ataxx/logs/gates/2026-11-15-*`, `docs/GATES.jsonl`
  (hub).
- Training log: `C:/gitProjects/logs/modal_nn_v10sat.log`. Run files: `runs/nn_v10sat/nn_v10sat/` (best.tnn BDE5BE71,
  RECORD.txt); all checkpoints stay on the Modal volume `ti-data:/runs/nn_v10sat`.
