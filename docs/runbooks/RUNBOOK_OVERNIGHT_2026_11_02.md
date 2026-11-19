# RUNBOOK + STATE for the night of 2026-11-01/02 (D330). Written before a compaction so nothing is lost. Read with PLAN_OVERNIGHT_2026_11_02.md and RESEARCH_WHERE_ELO_2026_11_02.md.

## Where main stands (local only, NOT pushed, NOT on Botzone)
- worktree C:\gitProjects\mainpromo-ataxx, branch main. Commits: c86733b (old, net F5FA8557) -> 0b2b837 (net 54A6B02B rwphase3_nocal, 2026-11-02 ~04:10) -> 32d85d6 (net 46FCB5A4 rwphase9, ~04:50).
  Only data/nnue/nnue_main.tnn changed; the net is include_bytes! in crates/titanium/src/nnue.rs. Tests 58+2+4 pass; bench depth 10 = 62420 nodes with 46FCB5A4.
- Promotion orders (user 2026-11-02): "if it wins just merge it into main"; "same net/engine, just better eval: merge if the 5k gate passed". Ledger rows PROMO-54A6B02B, PROMO-46FCB5A4.
- Owed: Botzone upload (V17 package still embeds F5FA8557; needs rebuild with tools/build_botzone.py + curl.exe upload; user order), push, PC time-leg confirmation (SPSA paused), SPSA re-gate on the new net.

## Result-weight table (all 5k, same exe BAF3782D = main c86733b; "vs old" = vs F5FA8557, "vs new" = vs 54A6B02B)
| tag | weights (<=6 / 7-12 / >=13 empties) | sha | result |
|---|---|---|---|
| rwphase1 | 0.35/0.04/0.00 | 6867B14A | -32.2 vs old (800 g) |
| rwphase2 | 0.4 uniform | 8C7A2519 | +3.7 vs old (3,200 g) |
| rwphase3_nocal | 0.55 uniform | 54A6B02B | +23 vs old (1,600 g); 1000 ms leg +47.2 +-43.8 (200 g, Docker) -> main 0b2b837 |
| rwphase5 | 0.7 | FB83D6ED | -20.0 vs old (800 g) |
| rwphase6 | 0.85 | CB3CED77 | -20.9 vs old (800 g) |
| rwphase7 | 0.47 uniform | 19111A79 | -9.1 +-13.4 vs new (2,368 g, LOS 9.1%) |
| rwphase8 | 0.62 uniform | 69244DA1 | gate queued (1,600 g vs new) |
| rwphase9 | 0.55/0.55/0.25 | 46FCB5A4 | +28.6 +-17.9 vs new (1,230 g, LOS 99.9%, LLR +2.87; the gate process ended at 1,230 g, reason not determined) -> main 32d85d6 |
| rwphase10 | 0.55/0.40/0.25 | 491D8B77 | +4.8 +-13.5 vs new (2,400 g, LOS 75.7%): middle bucket 0.40 is NOT better than 0.55 |
Reading: best so far is 0.55 for the last 12 empties and 0.25 before that. Uniform 0.47 and 0.62 and the 0.40 middle step are not better. Next blend experiments: finer per-empties ramp (hub change, rw is [f32;3] in targets.rs/gpu.rs/trainer.rs/export.rs), 0.55/0.55/0.15, 0.65/0.55/0.25, NNUE output buckets 16->32 via `ti train split` (parity, no random init).
Phase effect (ti insight phase table, docs/research/strategy/RESEARCH_WHERE_ELO_2026_11_02.md): gains sit at ply 60+; 0.25 in the opening removed the opening cost.

## The pipeline (every step used tonight; all hub/Rust, no Python for training)
1. TRAIN (Modal, ~7-10 min, ~$0.2): in C:\gitProjects\titaniumtest with the hub binary ./target_dev/release/ti.exe (do NOT rebuild target_dev while SPSA runs):
   `ti train modal launch --tag TAG --timeout-min 45 --max-usd 1.5 --args "train run --backend cuda --data /vol/data/train_v9 --val /vol/data/val_v9_clean --init /vol/runs/sat16a/best.tnn --out /vol/runs/TAG --steps 30000 --eval-every 250 --bs 16384 --lr 5e-5 --lr-min 1e-6 --warmup 500 --ram --guard 1.15 --ema 0.9 --val-rows 131072 --keep-all --proof-blend 0.5 --plateau 3 --lr-factor 0.5 --max-drops 3 --result-weight-phase A,B,C"` (A,B,C = weights for <=6, 7-12, >=13 empties). FORGETTING --out makes it exit 1 in 0.1 s. Log to C:\gitProjects\logs\modal_TAG.log. Ledger row in logs/modal_spend.csv. Env: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 MSYS_NO_PATHCONV=1.
2. FETCH: `ti train modal fetch --tag TAG --out runs/TAG` (run from C:\gitProjects\titanium-ataxx; the H2Connection AttributeError printed at the end is harmless) -> runs/TAG/TAG/best.tnn + RECORD.txt.
3. EXPORT (no --calibrate ever, D318; SAME --result-weight-phase as training): `ti train export --net runs/TAG/TAG/best.tnn --data data/nnue/packed/train_v9 --out data/nnue/proxy/TAG_nocal.tnn --result-weight-phase A,B,C` -> .tnn + .tnn.prov (provenance proven against RECORD.txt; the gate REFUSES a net without its .prov).
4. GATE in the Docker sandbox tit-cal1 (16 vCPU, \.118/h, ttl extended to 05:05 UTC 2026-11-02 = 06:05 local (local = UTC+1); ledger logs/docker_spend.csv). Files in /tmp/g: bin/base (exe BAF3782D = main c86733b), bin/ti (older hub; no end-of-gate report), book.txt, sat16a_cal.tnn (F5FA8557), rwphase3_nocal.tnn (54A6B02B), TAG_nocal.tnn(+.prov). Upload with `sbx --cloud cp FILE tit-cal1:/tmp/g/FILE` (both .tnn and .tnn.prov). Scripts (copies in C:\gitProjects\logs\sbx_cal1): g1.sh TAG BOOK0 GAMES = 5k nodes vs 54A6B02B, 30 threads; g2.sh = 1000 ms leg, 15 threads; g3.sh = 10k nodes; q.sh = sequential queue. Use a DIFFERENT book0 per gate (86000, 88000, 90000, 92000, 94000, 96000 used). NEVER use `pgrep -f`/`pkill -f` with a pattern that also appears in your own shell command line (self-match deadlock / kills your own exec shell); wait on gates/status.txt lines instead.
5. COPY games out of the sandbox (exec output drops big files): gzip|base64 in 300 KB chunks (function getf in this session; destination C:\gitProjects\logs\sbx_cal1\res), then `ti insight FILE` (use C:\gitProjects\titaniumtest\target_phase\release\ti.exe: it has the per-phase table; target_dev does not) and `ti analyze`.
6. PROMOTE (user standing order): in mainpromo-ataxx copy the .tnn over data/nnue/nnue_main.tnn, cargo test --workspace --release, build, bench, git commit with the Co-Authored-By trailer, LEDGER PROMO row. No push, no Botzone.

## Running right now (04:55 local)
- SPSA stage 1b on the PC (log C:\gitProjects\logs\spsa_stage1b.log, ~11 of 600 iterations; tuned vs F5FA8557 + exe logs/tune_6e418ce.exe: needs a re-gate on the new main net; resume = cmd in spsa_stage1b.cmd.txt + --resume).
- Sandbox tit-cal1: g2.sh rwphase9 1000 ms vs 54A6B02B (200 g, early -22.7 +-69.6 at 92 g = noise), then q.sh: rwphase8 5k (1,600 g), rwphase9 10k (800 g). Results land in /tmp/g/gates/*_vs_new_*.out; status lines in gates/status.txt.
- Worktree C:\gitProjects\cnt-ataxx (branch exp/counters from c86733b): created, NO counters written yet (plan item G0: count static evals / futility child evals pruned vs searched / ecache collisions / cutoff-index histogram via atomics in search.rs, print at the end of `bench`).
- Hub (C:\gitProjects\titaniumtest): games.rs has the new insight_phases (UNCOMMITTED, built into target_phase). Local commits 75bf96e, d373b13 (unpushed).
## Next (ranked, each with its doc): policy head per docs/research/policy/POLICY_PLAN.md (regression init, never random: LAWS C14), counters G0, finer blend ramp, 16->32 NNUE buckets, hard-position mining, W512 at the new target, SPSA re-gate.
## Clock note
Local time is UTC+1. Timestamps written as 'local' in D321-D331 entries between 03:00 and 05:10 are approximate (written from the sandbox clock plus guesses); the file mtimes and ledger CSV UTC stamps are the truth. 1000 ms leg of rwphase9 at 144 g (read 04:30 local): -26.6 +-55.4 Elo, LOS 17% = undecided noise but NOT supportive; the 5k leg is +28.6. If the 1000 ms leg ends conclusively below, tell the user (the time leg is the played clock, LAWS B9) and consider gating 46FCB5A4 vs 54A6B02B at 300 ms on the PC with SPSA paused.
