# Modal (GPU training, cloud gates) - everything an agent needs to run, watch, STOP and pay for it

Written 2026-10-26 so ANY model / harness can take over. Sister document for CPU sandboxes: docs/DOCKER_SANDBOXES.md.
Training recipes themselves: docs/TRAINING_RUNBOOK.md ("GPU (Modal) - the fast path"); decisions D80-D96 in docs/DECISIONS.md.

## 1. Policy and money (read before launching anything)
- Account: Modal Starter, workspace/profile `terminatort8004`, $30/month free compute, card on file. **The user set a hard spending
  limit of $30/month in the Modal UI** (2026-10-24). Never try to raise it; do not exceed what the limit allows.
- Which cloud: **CPU work -> Docker sandboxes first** (user: "docker credits are worth slightly less for us, only 40 vCPU and no GPU";
  the PC itself first of all). **Modal is for GPU** (NNUE training) and for jobs that need more than Docker's 40 vCPU quota.
- Announce every launch (what, size, worst-case cost) before starting it; no approval needed for a cost-bounded job, but NO long
  training / relabel without the user's explicit order (CLAUDE.md "Never").
- Price list used so far (Modal pricing page 2026-10-25): L4 GPU $0.80/h + 4 cores $0.19/h + 24 GiB $0.19/h = ~$1.2/h.
  Worst case of a run = hours x $1.2 per container; every call has a hard 4 h timeout (`timeout=` in modal_train.py).
- Real spend so far (from DECISIONS): stage 1 base_v4s ~$1.6 (D86), HPO 16 trials ~$1.5 (D90), ~$8 in total on 2026-10-26.
  There is **no automatic Modal ledger**: add a row to `logs/modal_spend.csv` for every launch (header exists) and reconcile with the
  Modal billing page (user's login) when in doubt. The billing page lags.
- **Never handle tokens/secrets.** The token was created by the user via `python -m modal setup` in their terminal. If auth fails
  ("not authenticated"), ask the user to run `python -m modal setup` themselves.

## 2. Commands (Git Bash: `export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8`; CLI client 1.6.2; use `python -m modal ...`)
```
python -m modal app list                       # RUNNING / stopped apps (empty table = nothing running = nothing billing)
python -m modal app stop -y <app id>           # STOP a run (detached runs keep going after the local process exits!)
python -m modal volume list                    # volumes: ti-train (created 2026-10-25) holds uploads (/up) and checkpoints (/ckpt)
python -m modal volume put --force ti-train <local file> /up/<name>      # upload
python -m modal volume get --force ti-train /ckpt/<tag>/net.pt data/nnue/modal_ckpt/<tag>/net.pt   # download results
python -m modal run training/modal_train.py --tag smoke --args "<train_nnue args>"             # attached (dies with your shell)
python -m modal run --detach training/modal_train.py --tag <tag> --args "<train_nnue args>"    # long run, survives the shell
python -m modal run --detach training/modal_train.py --tag hpo_x --script optuna_ft.py --args "--init /vol/ckpt/<tag>/net.pt --out /vol/ckpt/hpo_x --trials 16 --epochs 2 --threads 4"
```
- `training/modal_train.py` = the same `training/train_nnue.py` on an L4 (TI_DEVICE=cuda). It uploads only small files; the 12 GB
  feature index is rebuilt in the container with the engine's own `titanium-cli nndump` from `ENGINE_SRC` (default
  C:/gitProjects/main-ataxx) and checked against the `SHA` dict in the script - **update that dict whenever the features or the
  label npz change**, else the container refuses to train (by design).
- Checkpoints, the resume file and the log are on the volume (`/ckpt/<tag>/`, log `log.txt`), committed every few minutes. A run that
  hit the 4 h timeout is relaunched with the same tag; the trainer resumes.
- `training/modal_gate.py` = paired-book gates on Modal CPUs (both engines built from source inside the image): set `CAND_SRC` /
  `OPP_SRC` to the worktrees, then `python -m modal run training/modal_gate.py --tag <t> --shards N --games G --book0 B --nodes 5000`.
  Logs come back as `logs/<tag>_pNN.txt` (pool with training/pool_gate.py / pool_glob.py). Fixed-node gates only: a time gate on Modal
  measures Modal's hardware, not ours. Linux(Modal) vs Windows(PC) replays matched only 15/20 games at 5k nodes (OS/toolchain, not CPU;
  local Docker Linux = Modal 20/20): compare Linux builds with Linux builds.
- Image builds compile the Rust engine from the worktree source: the worktree must be CLEAN and committed (law 8).

## 3. State at 2026-10-26 ~14:10
- `python -m modal app list` showed NO apps (nothing running, nothing billing). Volume `ti-train` exists (uploads + checkpoints of
  stage 1/2 and the HPO study; local copies in `data/nnue/modal_ckpt/`). Logs of the old runs: `logs/modal_*.txt`.
- Nothing Modal-related is planned right now: the current small-net work is CPU-only on Docker (D109/D114: user said NOT Modal).
- The next Modal use will be the big-net retrain after the second relabel (user plan: saturate search/speed first, then relabel,
  then train): follow the TRAINING_RUNBOOK recipe (stage 1 only; Kata fine-tune is harmful, D98/D101).

## 4. How to stop everything / take over (any model)
1. `python -m modal app list` - every row in a running state is billing.
2. For each: `python -m modal app stop -y <app id>`; run `app list` again until none is running.
3. Fetch results you still need (volume get, section 2) BEFORE deleting anything; do not delete the volume unless the user says so.
4. Add the final row to `logs/modal_spend.csv` and update the HANDOFFS top entry + DECISIONS row of the run.
5. A harness with no `modal` CLI: ask the user to run the commands in their terminal; never ask for or handle credentials.

## 5. Pitfalls
- Detached runs outlive your session: always record the app id in HANDOFFS the moment you launch (BEHAVIOR 25).
- Unicode console errors on Windows: set `PYTHONIOENCODING=utf-8`; Git Bash path mangling: `MSYS_NO_PATHCONV=1`.
- `modal_train.py` uploads nothing big; if you see GB-scale uploads something is wrong.
- Cost surprises come from forgotten `--detach` runs and from parallel runs (each container bills): count containers x hours x $1.2.
