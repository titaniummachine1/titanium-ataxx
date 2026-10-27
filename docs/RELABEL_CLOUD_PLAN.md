# Relabel overnight plan: PC + Docker (≤ $15) + Modal (≤ $8–10)

Written 2026-10-27 19:30. Goal: label as much of `dis_positions.bin` (31,568,598 rows = 6,314 chunks of 5000,
sorted from biggest static-vs-search disagreement down) as possible by about 13:00 on 2026-10-28.
Engine: `af7e376`, exe sha `e3f8dc6e…`, bundled net `e11827d1…`, 5000 nodes, reset per position.
Output dir (everything lands here): `data/nnue/relabel/out_dis_remaining_e3f8dc6e_e11827d1_d8a135ce/c_<chunk>.txt`.

## 1. What is running now (checked 19:28)

| Pool | Setup | Measured speed | Chunks it owns |
|---|---|---|---|
| PC i7-4900MQ (4 cores / 8 threads) | 8 × `labelbin --threads 1`, scheduled task `RelabelWorkers` | ~10.4 pos/s per process, **~83 pos/s total** | claims upward from 0 (`next_chunk.txt`) |
| Docker `tit-d1` (16 vCPU) | 16 × `--threads 1` via `xargs -P 16` | first 16 chunks in ≤ 4.5 min → **≥ 18.5 pos/s per vCPU** | 1200–1760 |
| Docker `tit-d2` (16 vCPU) | same | starting | 1760–2320 |
| Docker `tit-d3` (8 vCPU) | same | starting | 2320–2600 |

Correctness: the Linux build in `tit-d1` relabeled queue rows 0–99 and matched the PC's `c_0.txt` on **100/100 lines**.

## 2. Common-sense check: what is optimal and what is not

**Good**
- Every pool runs one single-thread search per CPU. No multi-threaded search anywhere.
- PC at 8 processes on 4 cores with hyperthreading is right (D111: hyperthreads add ~8–13% for pure-node work).
- Chunk files have the same names everywhere, and every worker skips a chunk that already has a valid file, so pools cannot double-label once results are collected.

**Not optimal (fix these)**
1. **Idle billing (biggest issue).** At the measured Docker speed, 40 vCPU ≈ 740 pos/s, so the 7M rows I gave them finish in about 2.6 h (~22:00). The sandboxes then sit idle and bill until their 5 h ttl (~00:20). That wastes ~\$6 of the \$15. The fix is a watcher that collects finished chunks and hands each sandbox the next slice (same cost, about twice the rows), then deletes it when the budget is used.
2. **Cloud results only reach the PC at collect time.** If a sandbox hits its ttl before collect, its labels are lost. The watcher must collect before the ttl.
3. **Each PC process loads the whole 789 MB queue** (`labelbin` does `fs::read(bin)`), so 8 processes use ~6.6 GB RAM. This doesn't slow labeling, but if RAM gets tight, give each process a 5000-row slice file instead.
4. **Dashboard overcounts.** It adds the 1.37M labels in `out_dis/` and `out_dis_remaining_13210c04…/`. Those came from the OLD exe and are not part of this queue's progress. Progress and band should count only the manifest `output_dir`. The worker count also shows 0 while 8 processes run.

## 3. Budget math

| Pool | Price | Speed | Cost per 1M rows | Budget | Rows |
|---|---|---|---|---|---|
| PC | free | ~83 pos/s | \$0 | 17.5 h to 13:00 | **~5.2M** (chunks 0–~1050) |
| Docker | \$0.07 / vCPU-h | ~18.5 pos/s / vCPU (measured) | **~\$1.05** | \$15 → 214 vCPU-h | **~14M** |
| Modal | \$0.047 / physical core-h (= 2 vCPU) + \$0.008 / GiB-h | estimated 20–25 pos/s per core, more with 2 processes per core | **~\$0.45–0.65** (unverified) | \$8 | **~12–17M** |

Remaining after PC + Docker ≈ 31.5M − 5.2M − 14M ≈ **12.3M rows**, so Modal needs roughly \$6–8 to finish the queue.
That leaves \$2–4 of the \$10 Modal band plus anything unspent for GPU training. **The user must check Modal month-to-date
spend on the billing page first** (the \$30/month hard cap is shared with training; ~\$8 was spent 2026-10-26).

If Modal turns out slower than estimated, stop it at \$8. The rows left unlabeled are then the lowest-disagreement tail, which matters least.

## 4. Implementation plan (for the implementing agent)

### Step A: one shared cloud claim counter (local)
- File `data/nnue/relabel/cloud_next_chunk.txt`, msvcrt-locked like `next_chunk.txt`. Start value **2600** (Docker already has 1200–2600).
- Both cloud drivers claim ranges from it upward. The PC keeps its own counter from 0 and will reach only ~1050 by noon.
- Upper limit 6314. A driver stops when the counter reaches it.

### Step B: Docker watcher (`training/relabel_dis_cloud.py watch`)
Add two subcommands to the existing [relabel_dis_cloud.py](../training/relabel_dis_cloud.py):
- `reuse --name N --first-chunk A --last-chunk B`: upload a new `slice.bin` + `one.sh` (new FIRST, ROWS, DEADLINE), move `out/` aside after collecting, then restart `job.sh` **skipping the build and verify** (binary already built and verified). Same as `reuse` in `relabel_run.py`.
- `watch --budget-usd 15`, a loop every 10 min:
  1. For each sandbox: if `status == DONE`, `collect --keep`, claim the next slice of `vcpu × 35` chunks (≈ 2–2.5 h of work), then `reuse`.
  2. Spend so far = Σ(vCPU × hours since creation × 0.07), including the 3 running sandboxes from 19:21–19:25. Stop handing out slices when spend + (time left on running slices × rate) ≥ budget.
  3. When a sandbox has no new slice: collect, then `sbx --cloud rm --force`, and add a `DELETED after ~N min (actual ~$X)` row to `logs/docker_spend.csv`.
  4. Extend ttl with `sbx --cloud ttl +Nm` when a new slice would outrun it (hard ceiling 24 h from creation).
- Run it as a Windows scheduled task (like `RelabelWorkers`). The agent harness restarts killed tool-launched processes twice tonight.

### Step C: Modal relabel (`training/relabel_dis_modal.py`)
- Image: debian + rustup + `git archive af7e376` of main-ataxx, `cargo build -p titanium-cli --release` (same as `modal_relabel_probe.py`).
- Function `label_batch(first_chunk, chunk_blobs)`: **`cpu=8`** (= 16 vCPU), `memory=4096`, `timeout=1800`.
  Inside: write each 5000-row chunk (125 KB) to its own file, then run 16 `labelbin --threads 1` processes at once
  (`concurrent.futures.ThreadPoolExecutor(16)` around `subprocess.run`). Return `{chunk: stdout}`.
  Input per call = 16 chunks ≈ 2 MB, output ≈ 1 MB: no volume needed.
- Local driver: claim 16 chunks at a time from `cloud_next_chunk.txt`, `label_batch.map(...)` with `max_containers=10`,
  validate each result (5000 lines, 3 integer columns) and write `c_<chunk>.txt` via `.tmp` + rename.
  Track spend locally as container-seconds × (8 × \$0.0000131 + 4 × \$0.00000222) and stop submitting at the budget.
  Failed or invalid chunks go back to a retry list. Log every launch in `logs/modal_spend.csv`.
- **Probe first (~\$0.15):** one container with `cpu=8` runs (a) the 100-row verify against `c_0.txt` (must be 100/100;
  DOCKER Linux matched, so Modal Linux should too) and (b) 16 chunks with 16 processes, then 16 chunks with 8 processes.
  Use whichever gives more pos/s per dollar. Then compute cost per 1M rows and set the chunk budget from it.
- Run detached (`modal run --detach`) or as a scheduled task, and record the app id in HANDOFFS.

### Step D: dashboard fixes
- Progress, frontier and band from the manifest `output_dir` only; show the old dirs as "previous runs" separately.
- Show which chunk ranges are done as a coverage bar (PC from 0, Docker from 1200, Modal from wherever it claims), not just the max chunk.
- Fix the worker count, which should read 8 on the PC.

### Step E: morning
- Final `collect` of every sandbox, delete all sandboxes, `python -m modal app list` must be empty.
- Count valid chunks in the output dir, list the gaps (missing chunk indices) and let the PC fill the lowest gaps first.
- Update `logs/docker_spend.csv`, `logs/modal_spend.csv`, HANDOFFS, DECISIONS.

## 5. Stop everything
```
Stop-ScheduledTask RelabelWorkers; Stop-Process -Name titanium-cli
python training/relabel_dis_cloud.py collect --name tit-d1   # then d2, d3
sbx --cloud rm --force tit-d1                                # then d2, d3
python -m modal app list ; python -m modal app stop -y <id>
```
