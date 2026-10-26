# Relabel runbook (2026-10-25) - how the full-dataset relabel works and what went wrong

Goal: label all 33,891,139 dataset positions with the strongest engine (plain big NNUE c4829350, 5000 nodes, fresh state per position - NOT the hybrid)
to retrain/gate later. Label = "score depth nodes" per position; scores include +-99998/99999 (decided positions) - clip when training.

## Files (all under data/nnue/relabel/, gitignored)
- positions.bin (25-byte records, SHUFFLED, seed 11) + rows.npy (global dataset row of each record). Shuffled => any finished set of chunks is a random sample.
- out/c_<firstrow>.txt: one chunk = 5000 rows (cloud slices start at their own lo, so chunk starts are lo+k*5000, not always multiples of 5000).
- labels.npz: `python training/relabel_run.py merge` (full) or `merge --partial` (writes a `have` mask). Overlapping chunks are harmless.
- queue.txt ("lo hi" ranges still to hand out), sandboxes.txt ("name vcpu lo hi"), next_id.txt, watch_all.log, watch.lock, pc_crawl.sh, prebuilt/titanium-cli.

## Workers
- PC: `training/relabel_run.py pc --lo A --hi B --threads 7 [--deadline unix]` (~35-41 pos/s). pc_crawl.sh = after rows 0-1.6M, crawls from row 33.9M DOWNWARD in 500k
  blocks with no deadline; stop it by hand (kill python + titanium-cli + the bash).
- Docker cloud sandboxes (`sbx --cloud`): 16 vCPU ~205-227 rows/s, 8 vCPU ~100-113 (12.5 rows/s/vCPU). ~$0.07/vCPU-h (log: logs/docker_spend.csv, my estimate, not the bill).
  QUOTA: 40 vCPU in total (2x16 + 1x8). Creating a 4th 16-vCPU sandbox fails with resource_exhausted.
  `relabel_run.py cloud --name N --lo A --hi B --vcpu V --ttl 600m --deadline unix [--prebuilt data/nnue/relabel/prebuilt/titanium-cli]`
  `relabel_run.py reuse --name N --lo A --hi B --vcpu V --deadline unix` = give a FINISHED sandbox a new slice (uploads slice.bin+job.sh, moves old out/ to out_prev_*, restarts job).
  `sbx --cloud ttl +300m NAME` extends a sandbox's lifetime (hard ceiling 24 h from creation). `relabel_run.py collect --name N --force --keep` pulls chunks and leaves it running.
- Jobs must be started FULLY DETACHED `(nohup setsid bash job.sh > job.log 2>&1 < /dev/null &)`, or they die when the exec stream closes. Use `MSYS_NO_PATHCONV=1` with sbx in Git Bash.
- The prebuilt Linux engine (zig, glibc 2.31 target, NOT 2.14: std fails to link on 2.14) of main 0f27043 gave bit-identical labels to the cargo-built sandbox on 400 rows
  (checked in a debian:bookworm-slim container). It saves the 8-10 min cargo build per new sandbox.

## Watcher (data/nnue/relabel/watch_all.sh, v4) - run with `bash data/nnue/relabel/watch_all.sh > watch_all.out 2>&1` (ONE copy; it has a lock)
Every 5 min: per sandbox DONE (or 21:10 hard stop) -> final pull -> unlabelled tail of its slice to the queue front -> REUSE the same sandbox on the next fair-share range
(share = queued rows x its vCPU / all live vCPU, capped to finish by 20:00); fallback = delete + create a new one from the prebuilt engine; no work left = delete.
Every 3rd pass pulls running sandboxes' chunks. Ends when no sandbox is left. `is_listed` retries 3x so one failed `sbx ls` cannot make a live sandbox look dead.

## Lessons / traps
- TaskStop of a background `bash watch_all.sh` did NOT kill the script: 4 watchers ran at once. Kill with `taskkill /F /T /PID` (find them with Get-CimInstance Win32_Process).
- `pgrep -f 'job.sh'` inside `bash -c "..."` matches itself -> always true. Use `'[j]ob.sh'`.
- Do not read `sbx exec` inside `while read` loops (it eats stdin): use mapfile + `< /dev/null`.
- Python on Windows cannot read Git-Bash /tmp paths: use repo-relative paths.
- Heredocs with backslash-n through the tooling got mangled once: write such code with the Write/Edit tools.
- Rates were ~25% lower than the first estimates (cloud 205 not 227/s, PC 34-41 not 48): planning numbers = 12.5 rows/s/vCPU, PC 35/s.

## Result: DONE 19:29 (all 33,891,139 rows, labels.npz written, 0 sandboxes left). The paragraph below is the 14:55 snapshot, kept for history.
## Status at the time of writing (14:55)
~25.6M of 33.9M rows labelled (75%); tit-rl4/5/6 running; rl5 already reused on rows 25,127,785-28,402,785; queue 28.4M-33.0M; PC crawl from the top. ETA ~19:15-19:30.
After "watcher finished": `python training/relabel_run.py merge` (full). Missing rows -> PC crawl or a short extra sandbox. NO training without the user's order.

## Botzone diagnosis (same day) - how to read a match log
Fetch: `curl -H "X-Requested-With: XMLHttpRequest" "https://www.botzone.org.cn/match/<id>?lite=true"` returns JSON (logs[].output has the bot's stderr/traceback + verdict).
Match 6adde289...: "titanium_ataxx_PY6" lost on its first move: `OSError ... version GLIBC_2.28 not found (required by /data/titanium_v2.so)` = the OLD titanium_v2.so
(built on a modern glibc) was still in Botzone storage while the driver was already the new one. Our builds (tools/build_botzone.py, botzone/upload/data/titanium_v2.so
438,712 B sha c4ecf46954; CI artifact 436,440 B ee3cab501b) need glibc <= 2.14. The user re-uploaded and says the bytes match; a test match link is still owed to confirm.

## User plan after the relabel (not started, needs a go)
Retrain the full-width net on the new labels, gate it, THEN ablate the net for speed (narrower first layer/head: from scratch, distillation, lottery-ticket-style neuron pruning;
unstructured pruning is useless for the dense AVX2 eval). Gate speed AND strength at 5k nodes, 100 ms, 50 ms.

## After the relabel
The labels were joined into final_*_v4.npz (training/relabel_join.py) and trained into e11827d1 - see docs/TRAINING_RUNBOOK.md.
The next relabel round should label with the CURRENT main net (e11827d1), not c4829350.
