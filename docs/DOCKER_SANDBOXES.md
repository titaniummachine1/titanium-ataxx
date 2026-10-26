# Docker cloud sandboxes (`sbx --cloud`) - everything an agent needs to run, watch, STOP and pay for them

Written 2026-10-26 so ANY model can take over. If you find a sandbox running and no session is watching it, section 4
tells you how to check it, collect its results and delete it. Money rules first.

## 1. Money (read before creating anything)
- Docker credit: $250, spend cap agreed with the user: **$200** (Docker is free up to that; Modal is capped at $30/month and the
  user PREFERS Docker for CPU jobs: "docker credits are worth slightly less for us, only 40 vCPU and no GPU", and wants the PC used
  as much as possible). Announce every launch and its worst-case cost BEFORE creating it; never leave a sandbox idle.
- Price: 16 vCPU / 32 GB = $1.118/h (0.07 per vCPU-h). Worst case of a sandbox = ttl x price.
- Ledger: `logs/docker_spend.csv` (when,name,cpus,mem_gb,usd_per_h,ttl_min,charged_usd_upper,status). Add a row at creation
  (worst case = full ttl) and a row `DELETED after ~N min (actual ~$X)` when you delete. It is my estimate, the Docker billing page
  lags up to 48 h. The running total is not summed anywhere: worst-case rows overstate, DELETED rows give the real figure; add them up before a big launch.
- `--on-timeout delete --ttl Nm` at creation makes every sandbox self-destruct, so a forgotten one costs at most its ttl.
  Extend with `sbx --cloud ttl +Nm NAME` (hard ceiling 24 h from creation).

## 2. Commands (Git Bash; ALWAYS `export MSYS_NO_PATHCONV=1`, always `< /dev/null` on exec/cp/ls inside loops)
```
sbx --cloud ls                                   # list; STATUS running; name, id, age
sbx --cloud create --name tit-xyz --image-ref docker.io/library/python:3.12 --cpus 16 --memory 32g --ttl 240m \
                   --on-timeout delete --platform linux/amd64        # python:3.12 = training, rust:1.97 = Linux builds / gates
sbx --cloud exec tit-xyz bash -c "cmd"           # run a command (the exec stream can time out: retry; transient
                                                 #   "unauthenticated: resolve access token ... timed out" is NOT a failure of the job)
sbx --cloud cp LOCAL tit-xyz:/tmp/x/NAME         # upload (Windows path via cygpath -w if it fails); download: sbx --cloud cp tit-xyz:/tmp/x/NAME ./
sbx --cloud ttl +60m tit-xyz                     # extend
sbx --cloud rm --force tit-xyz                   # DELETE (stops billing). Files are gone: copy results out first
```
- Jobs must be FULLY DETACHED or they die when the exec stream closes: `(nohup setsid bash -c '...' > job.log 2>&1 < /dev/null &)`.
- Do not run `sbx exec` inside `while read` loops (it eats stdin): `mapfile` + `< /dev/null`.
- One game / one trainer per core for timing gates; pure-node 5k gates: 24-32 matches in parallel on 16 vCPU (D111).
- Uploads are slow: 1.5 GB took ~14 min. Generate big data inside the sandbox when possible.
- Never type the user's tokens/credentials; `sbx login` is the user's job.

## 3. Current state (2026-10-26 ~14:00): ONE sandbox is running
**tit-sn2** (16 vCPU, created ~12:56 local = 11:56 UTC, ttl 240m -> self-deletes ~16:56; worst case $4.48, ~$1 spent by 13:55).
Purpose: small-net v2 search (docs/DECISIONS.md D114). Directory `/tmp/sn` in the sandbox:
- `sn_train.npz` (1.5 GB), `sn_eval.npz`, `sn_*_ex.npy` extras, `smallnet_train.py`, `smallnet_eval2.py`, `smallnet_optuna.py`.
- Arms (finished): `a_ctrl.pt b_extra.pt c_extra_alpha.pt d_all_focus.pt` + `.out` logs. Winner arm D: 72.7% big-net evals skipped at 1.43% wrong prunes (control 68.6%).
- **Optuna study running** (detached): `python3 smallnet_optuna.py --jobs 4 --threads 4 --extra --alpha-in --trials 32 --patience 8`,
  log `/tmp/sn/optuna.out` (lines `trial N W .. -> skip X% cost Y us/call`, ends with `SATURATED` or `BEST`), storage
  `/tmp/sn/sn_optuna.db` (sqlite; finished trials are kept), checkpoints `/tmp/sn/opt_NNN.pt` (+ `.out` train log).
  A trial takes ~20-25 min (4 in parallel); 32 trials would take ~3 h, so the plan is to STOP at ~15:10 (about 16 trials) or at SATURATED.
- A local background watcher (Claude task) waits for `BEST|SATURATED` in optuna.out and then prints the trials. It is not needed
  for correctness: you can do section 4 by hand.

## 4. How to stop it / take over (any model)
1. Look: `sbx --cloud ls < /dev/null`, then `sbx --cloud exec tit-sn2 bash -c "grep -E '^trial|SATURATED|BEST' /tmp/sn/optuna.out; pgrep -fc smallnet_train" < /dev/null`.
2. Stop the study early (finished trials stay in the db): `sbx --cloud exec tit-sn2 bash -c "pkill -f smallnet_optuna; pkill -f smallnet_train" < /dev/null`.
3. Pick the best trial: lowest `cost` in optuna.out, or
   `python3 -c "import optuna;s=optuna.load_study(study_name='sn',storage='sqlite:////tmp/sn/sn_optuna.db');b=s.best_trial;print(b.number,b.value,b.params,b.user_attrs)"` (run inside the sandbox).
4. Copy out what you need into `data/nnue/smallnet/` (locally): `opt_<N>.pt`, `opt_<N>.pt.log`, `optuna.out`, `sn_optuna.db`,
   `d_all_focus.pt`, `a_ctrl.pt`, `b_extra.pt`, `c_extra_alpha.pt` and the `.out` logs of the arms.
5. DELETE: `sbx --cloud rm --force tit-sn2 < /dev/null`, then `sbx --cloud ls < /dev/null` must not list it.
6. Ledger row in `logs/docker_spend.csv` (`DELETED after ~N min, actual ~$X`), a line in HANDOFFS top entry, and DECISIONS D114 status.
If the sandbox is already gone (ttl): results not copied out are lost; the arms' numbers are in D114 / HANDOFFS, rerunning costs ~$3.
Never leave it running because "the study is not finished": the ttl is a hard stop and the user wants no idle instances.

## 5. After the study: next steps for the small net (the engine port is half done)
- Branch `exp/fut-sn2` (worktree `C:/gitProjects/futsn2-ataxx`, commit edf58a7): `smallnet::tactical()` (10 bit-sliced features,
  test vs brute force) is in; still to do: parse the new export format (any W,H + K=11 extra inputs: 10 tactical + alpha/400 clipped
  +-5), `exskip` per-bucket linear term, AVX2 head for the winner's W,H, `training/smallnet_export.py` for the v2 format + ref file
  with extras, `snparity` check <= 0.5 cp, then gates (100 ms + 50 ms vs main 343ab02, `training/time_gate.py`).
- Training recipes and the scoring tool (`smallnet_eval2.py`: skipped big-net evals at fixed wrong-prune rates) are in
  `docs/TRAINING_RUNBOOK.md` section "Futility small net" and D109/D110/D114.
- The small net is fitted to ONE big net (e11827d1): retrain it after every new big net.

Modal (GPU, cloud gates, the $30/month cap): docs/MODAL.md.

## 6. Other places that mention sandboxes
`docs/RELABEL_RUNBOOK.md` (relabel runs, watcher, reuse of finished sandboxes), `docs/DECISIONS.md` D111 (gate parallelism),
`logs/docker_spend.csv`, `training/par_probe.py`, `training/gate_net_vs_main.py` (5k gates, `--par`).
