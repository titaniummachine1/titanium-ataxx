# Futility disagreement auxiliary-head experiment

**Status:** setup started; source sample labeling and training have not run. The active PC relabel job is still the only long CPU job. Do not kill or rebuild its pinned main executable.

## Durable workspace

- Branch: `exp/futgap-disagree`
- Worktree: `C:\gitProjects\futgap-ataxx`
- Base: main `7d7ef19` (contains bundled `e11827d1` NNUE and futility SmallNet)
- Source futility data: `C:\gitProjects\titanium-ataxx\data\nnue\smallnet\sn_train.npz` (D109 unique training child positions) and `sn_eval.npz` (held-out futility calls from disjoint roots), plus `sn_w32.pt` / `sn_q32.pt` checkpoints and the raw `rec_*.bin` files.
- Current search label engine: `C:\gitProjects\main-ataxx\target\release\titanium-cli.exe`, bundled network; manifest hashes are captured by `training/futgap_label.py` at prepare time.
- `training/futgap_label.py` creates seeded, disjoint samples (32,768 train / 4,096 calibration / 8,192 final test by default) in side-to-move-relative 25-byte board format and then, only with `--label --pc-idle`, labels them at 5k nodes. The target is `child search score - D109 big-NNUE child static score`, in child-to-move cp. It writes sample row IDs, static/context inputs, labels, and executable/network hashes under `SN_DATA/futgap_probe_e3f8dc6e_20261027`. Preparation refuses to overwrite an existing run; label outputs are hash-checked and resumable by split.

## Experiment hypothesis

For the futility call, branch a disagreement output from the current SmallNet's already-computed L1 activation. The child accumulator is already built before the engine decides whether to call the big NNUE. Reusing it avoids another board feature transform. Keep the existing small-net static lower-bound head; add signed search-static residual plus a calibrated lower-tail estimate (or direct lower quantile of child search score). Child-perspective sign matters: a negative residual means search may see the child as worse than its static eval, so the parent move is more dangerous to prune.

This is a futility-specific choice. For generic horizon continuation, the main NNUE auxiliary head remains the likely cheaper attachment because its trunk has already run at the leaf. A standalone model is the clean independent baseline.

## Resume procedure

1. Confirm the active `training/relabel_overnight.py` process has finished or has been stopped at a completed chunk. The D109 source data and this experiment workspace are persistent; no source relabel progress is overwritten.
2. From this worktree, set `$env:SN_DATA='C:/gitProjects/titanium-ataxx/data/nnue/smallnet'` and run `python training/futgap_label.py --prepare`.
3. Run `python training/futgap_label.py --label --pc-idle --threads 7` only when the PC is free. The label script refuses to mix an engine or network hash changed since preparation.
4. Run `python training/futgap_train.py --threads 1`. It freezes the existing SmallNet FT/L1, selects on the calibration split, and fits a signed-mean residual head plus a 5th-percentile residual head; it reports final-test MAE/RMSE/correlation and quantile violation against a zero-gap baseline.
5. Simulate the real futility threshold using the final-test child calls; report big-NNUE evaluations saved vs wrong prunes at fixed error rates. Use the calibration split to choose any safety shift; never tune the shift on the final test.
6. Only if the offline tradeoff improves: port the small second head to the futility net format and measure per-call latency plus full-engine 5k-node and time-controlled results. Preserve the existing lower-bound path as the control.

## Stop/go criteria

- No use of the disagreement-ranked relabel prefix as a random validation split.
- No conclusion from prediction loss alone: the primary offline metric is saved expensive evals at a fixed wrong-prune rate, followed by measured full-engine cost.
- No promotion to main without the project’s normal engine gate and user decision.
