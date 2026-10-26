## 2026-10-26 ~11:40 - CURRENT STATE: futility speed-up done and gated; NOTHING RUNNING (no local job, no Docker, no Modal)

Main is still 1b286f2. Candidates (none merged, user decides): exp/fut-cheap ea3e8ec (hand-eval pre-filter, D108): vs main 100 ms +69 [+35,+103], 50 ms +59 [+26,+94], 5k -20 (inconclusive). exp/fut-sn 9888823 (on top of fut-cheap; futility small net as lower bound tau 2%, D110; 5k search x1.77 vs main): vs fut-cheap 100 ms +41 [+8,+75], 50 ms +45 [+22,+69] (832 g), vs main 5k +3 [-30,+37]. exp/fut-sn-t002 8f1d228 (tau 0.2%): vs main 5k -22 [-56,+12], no time legs. exp/nnue-int8 1e0e406: 5k -12 (inconclusive), +5% speed, parked. Not done: a DIRECT fut-sn vs main time confirmation (law 24 style) before merging. Gate parallelism (D111): time gates par 4 (physical cores), 5k gates par 10 on the PC, 24-32 on a 16-vCPU Docker sandbox. Docker spend this session ~$1.4 (logs/docker_spend.csv). The small net is fitted to e11827d1: retrain it with the big net (smallnet_* tools, D109/D110).
Ranked queue of what to try and gate next: BACKLOG.md, section CURRENT BACKLOG (2026-10-26).

## 2026-10-26 ~11:00 - RUNNING: futility small-net gates (D110); fut-cheap PASSED both time legs (awaiting user merge call)

Chain in logs/gate_futsn.out: exp/fut-sn 9888823 (tau 2%, exe A6A220D0) vs exp/fut-cheap ea3e8ec at 100 ms then 50 ms; then 5k vs main for tau 2% and tau 0.2% (exp/fut-sn-t002 8f1d228). Keep the PC idle during the time legs. Docker: nothing running (tit-sn1 deleted). Small-net tools: training/smallnet_{pack,train,eval,export}.py, data in data/nnue/smallnet/. The small net is fitted to e11827d1: retrain it (probe -> pack -> train bound mode -> export) whenever the big net changes.

## 2026-10-26 ~10:30 - RUNNING: futility small net training on Docker tit-sn1 (D109) + fut-cheap time legs

Docker sandbox tit-sn1 (16 vCPU, auto-delete at 150 min from ~10:05; DELETE it when the runs saturate) trains 3 small nets in /tmp/sn (sn_w32.pt, sn_w32nh.pt, sn_w64.pt). Gates: GATE-futcheap 5k = -20 [-54,+13] inconclusive; 100 ms then 50 ms legs running (logs/gate_futcheap.out). Data: data/nnue/smallnet/. PC stays idle during the time legs.

## 2026-10-26 ~10:30 - RUNNING: gates of the cheap futility pre-filter (exp/fut-cheap ea3e8ec, D108)

Background chain (logs/gate_futcheap.out): 5k nodes -> 100 ms -> 50 ms vs main 1b286f2, same net e11827d1. Keep the box idle during the time legs. Done: GATE-int8-vs-main 5k = -12 [-45,+22] inconclusive (LEDGER). D107 corrected: 72 ns = hand eval, S57 = 1.76 us.

## 2026-10-26 ~09:30 - network speed ablation probe (no training; nothing running)

D107: NNUE evals are ~95% of search time, 2.77 computed evals per node, 91% of futility child evals end in a prune. No free neuron pruning (all 256 used). int8 FT = exp/nnue-int8 1e0e406, +5% speed, val +0.05%, NOT gated. Next options (need the user's go): (a) time-gate int8; (b) cheaper futility child eval (search change, biggest lever); (c) retrain narrower / without G2 (Modal, ~$1-3 each), gate speed AND strength. Probe tool: training/nnue_prune_probe.py.

## 2026-10-26 ~08:30 - CURRENT STATE (read this first; everything below is history)

**Main = origin/main 1b286f2 (PUSHED)** = default NNUE **e11827d1** (+132 Elo @5k vs the old c4829350, LEDGER GATE-s1chg-vs-main)
+ critical fixes (LEDGER MAIN-critical): Botzone driver hardening, FFI depth cap 32 + release stack guard, finite-INF window /
saturating TT mate conversion (strength-neutral, -5 [-39,+28]), stale tests fixed. Workspace tests pass except the known-broken
titanium-wasm crate. Local main exe f3552dc7. NOTHING IS RUNNING (no local job, no Modal app, no Docker sandbox).

**Owed / waiting on the user**
- Botzone upload (user's login): C:/gitProjects/main-ataxx/botzone/upload = titanium_botzone.py 320c2c0f (14,288 B, new bot version,
  Python 3, simple IO, long-running OFF), data/titanium_v2.so 942faa88 (438,752 B -> storage /titanium_v2.so), data/nnue/nnue_main.tnn
  e11827d1 (27,945,148 B -> storage /nnue/nnue_main.tnn). Then a test match: read its log with
  `curl -H "X-Requested-With: XMLHttpRequest" "https://www.botzone.org.cn/match/<id>?lite=true"` (debug line = depth/nodes/time/net + "ev ..." recoveries).
- Next round choice (user): (a) self-relabel the 33.9M rows with e11827d1 + retrain stage 1, (b) deeper Titanium labels (20k+ nodes) as
  the teacher, (c) speed ablation of the net (narrower FT/head) aimed at Moonbird's speed edge. NO external teacher (see arena).
- Optional: Botzone end rules in search (either side stuck = bz_over, 401-ply judge cap) live only on exp/botzone-robust (bzrob-ataxx);
  correct per the judge, not crash fixes, slower per node, untested in Botzone Elo -> needs its own gate before main.

**What was learned (2026-10-25/26), all in docs/DECISIONS.md D73-D106, LEDGER, MISTAKES 25-30**
- Relabel: all 33,891,139 rows labelled by big NNUE c4829350 @5000 nodes (docs/RELABEL_RUNBOOK.md); joined as data/nnue/final/final_*_v4.npz
  (titanium_cp replaced, titanium_cp_old + relabel_depth kept; training/relabel_join.py, verified on all rows). train_nnue.py reads v4 by
  default (TI_DATA_TAG=v3 = old).
- Trainer fixes (training/train_nnue.py): --block 1024 mixed chunks (the npz files are in source+game order; old loader trained 245 steps at a
  time on one contiguous chunk = non-iid, D75); warm-start seed is the score to beat (D76); hypergrad fixed to Baydin-for-Adam (D78) and reset
  at every warm restart (D82); TI_DEVICE=cuda / TI_ROOT for Modal (D80).
- GPU training on Modal: training/modal_train.py (L4, volume ti-train: /up = slim npz + seeds, /ckpt/<tag> = runs; features rebuilt in the
  container by the engine's nndump and sha256-checked vs local), ~2.7 min/epoch vs 64 min on the CPU, ~$1.2/h. Spent ~$8 on Modal today.
- THE RECIPE that made e11827d1: stage 1 (base, v4 labels) seeded with c4829350, auto-lr + hypergrad, lr 4e-3, saturated 0.022533 (+50 vs main)
  -> short Optuna continuation search (training/optuna_ft.py, D83/D90: lr ~1.1-1.7e-3, Lookahead k 9-10 alpha ~0.33, bs 4096; weight decay
  irrelevant) -> continuation from trial 14 to saturation WITH hypergrad (s1c_hg): val 0.021626 = e11827d1 (+132). Without hypergrad: equal
  in Elo (+3 [-31,+36], D95), hypergrad costs +50% GPU time.
- STAGE 2 (Kata fine-tune) IS HARMFUL: -137 vs its stage-1 net (D98); scaling its output to stage 1's spread is worse (-183, D104) -> not scale;
  Kata's targets are TD values that blend each game's own result (hindsight, D101). Titanium's labels predict results better than
  Ataxx_CPP's raw net (Brier 0.162 vs 0.169; right 70% of confident disagreements).
- External arena (training/ax_vs_ti.py, Botzone end rules, 100 g/leg, D103/D105): Ataxx_CPP 5k 100-0 / 100 ms +220 / 1 s +210 (Titanium
  wins); KataAtaxx (1 thread; ignores visit limits, so no 5k leg) 100 ms -108 with ~9x Titanium's time, 1 s +191 with ~1.8x; Moonbird 1.1.0
  5k -7 (equal) / 100 ms -241 / 1 s -21 -> Moonbird's edge is SEARCH SPEED.
- Gate tools: training/gate_net_vs_main.py (any .pt/.tnn vs main or --champ net, --cand-exe, pure node limit --time 0, tripwire = shard 0
  replayed); training/run_ext_arena.sh; KataAtaxx arena config engine_cpu_arena.cfg (1 thread, no ponder). training/cpp_bridge.py is an
  UNVERIFIED UAI bridge (engine.exe lacks `set`; the arena uses engine-ctl.exe directly) - do not use without fixing.
- Harness trap (D102, MISTAKES 30): `match --nodes N` without `--time 0` caps the in-process candidate at 1 s; under CPU overload that voids
  gates. Run ONE gate at a time; never edit/build in a worktree a running job reads (MISTAKES 29).
- New law: BEHAVIOR 25 DOCS IN REAL TIME ("what you remember you shall forget, what you wrote down you will remember").

**Worktrees made today (can be removed when no longer needed):** bzrob-ataxx (exp/botzone-robust, keeps the BZ-rule commits), crit-ataxx
(exp/critical-fixes = main), netpromo-ataxx (exp/net-s1chg, merged). Nets: data/nnue/champions/MAIN_NNUE_s1chg_e11827d1.tnn,
data/nnue/modal_ckpt/{base_v4s,s1c_hg,s1c_nohg,s2_hg,hpo_ft}/.

## 2026-10-25 19:30 - FULL RELABEL DONE (all 33,891,139 rows labelled by main 0f27043 + plain big NNUE c4829350, 5000 nodes, fresh state per position)

data/nnue/relabel/labels.npz (644 MB): rows (global dataset row = train rows, then val, then test), have (all True), score, depth, nodes. 0 sandboxes left (tit-rl1..rl8 deleted).
Facts: mean depth 8.21, mean nodes 4939; 832,568 rows (2.46%) are proven wins/losses (|score| > 5000: values +-99999 minus ply, mate distance) -> clip/convert when training; 15,850 rows have depth 0 (terminal or no-move positions, score 0 or -99998/-2xxx sample values: inspect before use); non-decided scores: mean 66, p1 -1114, p99 +1190.
Cost: docker_spend.csv sums to $58.5 WORST CASE (every sandbox at full ttl); by actual runtime ~ $48 (rl1 ~9, rl2 ~10, rl3 ~5, rl4 ~6, rl5 ~9, rl6 ~4.4, rl7 ~4.3, rl8 ~0.3). Not the bill.
DONE 2026-10-26 ~08:00 (D105): external arena - Ataxx_CPP and KataAtaxx are weaker than Titanium on equal terms, Moonbird equal except at 100 ms (speed); no external teacher. Earlier: RUNNING (2026-10-26 ~04:00): (1) local gates srchfix_vs_main2 then s2scaled_vs_s1chg2 (pure node, --time 0; logs/gate_*2); (2) external arena training/run_ext_arena.sh (D103, runner log logs/arena_ext_runner.txt, results logs/arena_*.txt; ~7-8 h; timed legs wait for (1)). MAIN = 1b286f2 PUSHED (D106): default net e11827d1 + critical fixes; Botzone upload folder C:/gitProjects/crit-ataxx/botzone/upload (user uploads .py + .so + net). Main exe rebuild after gate s2scaled_vs_s1chg ends; Botzone needs the new net uploaded. The srchfix_vs_main gate is VOID (tripwire mismatch on 2 of 26 games, most likely the 82-s net swap; candidate replays deterministically) -> re-run it in full. KATA FINE-TUNE HURTS (D98): stage-2 net -137 Elo vs its stage-1 net (CI [-175,-102]), = main; best net stays the stage-1 hg net data/nnue/modal_ckpt/s1c_hg/net.pt (TNN E11827D1, +132 vs main); stage 2 parked, user to confirm. BOTZONE (D97): hardened driver on exp/botzone-robust d389bd6, upload folder C:/gitProjects/bzrob-ataxx/botzone/upload - user uploads the new titanium_botzone.py as a new bot version (storage files unchanged); merge into main needs the user's OK. GATE s1chg vs main PASSED (D94): +132 Elo at 5k, CI [+97,+170], net data/nnue/modal_ckpt/s1c_hg/net.pt (TNN E11827D1) = strongest so far, NOT staged/merged. RUNNING (D93, since ~01:50): STAGE 2 s2_hg on Modal app ap-qxEOaY2aJ5uXqKLd4WId7A from the stage-1 hg net (val 0.021626; local stream logs/modal_s2_hg.txt; stop: python -m modal app stop -y ap-qxEOaY2aJ5uXqKLd4WId7A) + local gates s1chg_vs_main then s1chg_vs_nohg (logs/gate_s1chg_*). DONE before: RUNNING (D91, since ~00:50): stage-1 continuation A/B on Modal: s1c_nohg app ap-674nks78Y9cQVVQqZ3IZgW, s1c_hg app ap-7RBJvvqQSYOaP8wweVvDoK (volume /ckpt/<tag>, local streams logs/modal_s1c_*.txt; stop: python -m modal app stop -y <app>); the better one -> gate vs main -> stage 2. D83 SEARCH DONE (D90): best t14 val 0.022166 (-1.6%), data/nnue/modal_ckpt/hpo_ft/t014.pt; next = continue stage 1 from it with its settings (auto-lr) -> gate vs main -> stage 2. STAGE 2 STOPPED (D89: the D83 search found a 0.72% better stage-1 continuation; stage 2 restarts from that net) - was app ap-GOfH0q7aRoG3PXd1hmAw0J (volume /ckpt/teacher_v4s, local stream logs/modal_teacher_v4s.txt, stop: python -m modal app stop -y ap-GOfH0q7aRoG3PXd1hmAw0J). GATE s1v4 PASSED 23:35 (D87): stage-1 net beats main at 5k, 230-170-16, +50 Elo, CI [+17,+85], both seats; NOT staged/merged (user call). STAGE 1 SATURATED 23:16 (D86): data/nnue/modal_ckpt/base_v4s/net.pt, val 0.022533 (seed 0.025629). NOW RUNNING (23:22): D84 gate s1v4 on the local CPU (logs/gate_s1v4/, logs/gate_s1v4.out) + D83 search on Modal app ap-A34IiY50Lf9bPgJbViDcVu (volume /ckpt/hpo_ft, stop: python -m modal app stop -y ap-A34IiY50Lf9bPgJbViDcVu). Plan:  D83 continue-from-saturated Optuna (optuna_ft.py, ~1 h, ~$1.5) -> D84 GATE the best stage-1 net vs the champion (training/gate_net_vs_main.py, 416 g at 5k, local CPU ~10 min) -> stage 2 (Kata fine-tune, user OK given as part of D85) -> GATE stage-2 net vs main AND vs stage-1 net = does the Kata fine-tune help? (D85: Kata is NOT assumed right; games decide). STAGE 1 NOW ON MODAL GPU (D81, since ~21:47): app https://modal.com/apps/terminatort8004/main/ap-BLbA4IHnUttQKngTjOSJGs (relaunched 21:58 from the seed with the D82 hypergrad-restart fix; stop: `python -m modal app stop -y ap-BLbA4IHnUttQKngTjOSJGs`; first try ap-qigqdHSAm6pt9VEIM3FKxf stopped after 2.4 epochs, log logs/modal_base_v4s_run1_stopped.txt), local stream logs/modal_base_v4s_launch.txt; volume ti-train /ckpt/base_v4s (log.txt, net.pt); check: `MSYS_NO_PATHCONV=1 python -m modal volume get ti-train /ckpt/base_v4s/log.txt logs/` ; relaunch = same modal run command (resumes). CPU run STOPPED 21:45. Earlier: GPU MOVE IN PREP (D80, 21:30): training/modal_train.py + slim data upload to Modal volume ti-train; switch only after a <$1 smoke shows >=5x. STAGE 1 RUNNING since 20:37 (D79) [EPOCH 1 DONE 21:41: val mse_target EMA 0.023099 (live 0.024298) vs seed 0.025629 = -9.9%, accepted; Kata-TD drifted 0.05585 -> 0.06086 (expected for the base stage, stage 2 re-fits Kata); the rising train loss did not hurt val] [21:22 WATCH: epoch-1 per-chunk TRAIN loss rose 0.0105 -> 0.0117 while hypergrad cut the LR 1.21x -> 0.80x (264 up / 4443 down) = LR too high for the warm start; epoch-1 val decides: worse than seed 0.025629 -> auto-lr rolls back + halves LR] - user 2026-10-25: "try to saturate the first stage before progressing" (stage 2 only after auto-lr SATURATED + user approval): `bash training/run_base_v4s.sh`, log logs/train_nnue_base_v4s.txt, runner log logs/run_base_v4s.txt; to resume after a reboot just run the same script (it resumes); stage 2 needs the user's approval. Earlier: STAGE 1 PREP (~20:15): trainer fixed (--block mixing D75, warm-start seed D76); hypergrad smoke showed a runaway LR (D77) -> fixing to Baydin-for-Adam and re-smoking; full stage-1 run NOT started yet (runner training/run_base_v4s.sh). JOIN DONE + VERIFIED (2026-10-25 evening, user order 'replace old labels'): `python training/relabel_join.py` wrote data/nnue/final/final_{train,val,test}_v4.npz = v3 + `titanium_cp` replaced by the relabel score, `titanium_cp_old` (previous values), `relabel_depth`; v3 files untouched. Verified on ALL rows: every other column identical, titanium_cp == mapped score, packed position == dataset position (train/val/test 32,534,256 / 677,670 / 679,213). training/train_nnue.py now loads `final_<split>_${TI_DATA_TAG:-v4}.npz` (DATA_TAG; set TI_DATA_TAG=v3 for the old labels). Old vs new on val (undecided rows): corr 0.67, new labels correlate better with the game result (0.343 vs 0.236), new sd 385 cp vs 246. Kata columns (teacher_p3, kata_td_*, teacher_kind) unchanged, so the teacher stage is unaffected. Nothing trained.
STILL TO DO (user go): join step is done; join labels into the training files by row (final_{train,val,test}_v3.npz column `titanium_cp` <- labels.score, keep Kata columns), then stage 1 base training on the new titanium_cp, stage 2 Kata fine-tune (same recipe as nnue_t1ep4: HANDOFFS lines on `train_nnue.py --stage teacher`, DECISIONS D36 warm-start question still open), export TNN2 + parity, gate vs main. Then the speed ablation (user plan below).

## 2026-10-25 afternoon - FULL RELABEL (user: relabel the whole dataset, wait for it to finish)

RUNBOOK with all recipes, lessons and the Botzone glibc diagnosis: docs/RELABEL_RUNBOOK.md (read it first for anything relabel / sbx / Botzone log related).

USER PLAN (2026-10-25 14:30, not started, needs an explicit go): after the relabel finishes and the last training run(s) are done, ABLATE the network to see how much faster it can be made (narrower first layer / head: train-from-scratch and distillation vs lottery-ticket-style neuron pruning with reset to initial weights). Gate each by speed AND strength (5k nodes, 100 ms, 50 ms). Unstructured weight pruning is useless for our dense AVX2 eval - prune whole neurons.

(13:45: watcher v4 REUSES finished sandboxes instead of delete+create: `sbx --cloud ttl +Nm` extends the ttl, `relabel_run.py reuse` uploads the next slice.bin+job.sh and restarts the job; slices are fair-shared by vCPU so all sandboxes end together; fallback = delete + new sandbox from the prebuilt engine.) At 13:20 the watcher became v3 (data/nnue/relabel/watch_all.sh): work queue queue.txt (rows 25,127,785-33,000,000 first), sandboxes.txt = 'name vcpu lo hi'.
tit-rl4/5/6 (16/16/8 vCPU) finish their slices ~14:50 (job deadline 15:00); each time one is DONE it is pulled+deleted, its unlabelled tail goes to the queue front, and a
same-size successor starts on the next range, sized to finish by ~20:00 (12.5 rows/s/vCPU), using the PREBUILT Linux engine (no cargo build): training/relabel_run.py cloud --prebuilt
data/nnue/relabel/prebuilt/titanium-cli = zig build (glibc 2.31) of main 0f27043, labels verified bit-identical to the cargo-built sandbox on 400 rows.
PC: pc_crawl.sh crawls from row 33.9M downward, no deadline (stop by hand). ETA all rows ~19:30-20:00. Extra cost ~$12-14 (total ~$57; Docker free allowance $200).
When the watcher log says 'watcher finished': check `python training/relabel_run.py merge` (full) - if rows are missing the PC crawl fills them, or relaunch a sandbox on the gap.
No training without the user's order.

## 2026-10-25 night - OVERNIGHT RELABEL RUNNING (big NNUE c4829350, 5000 nodes/position, fresh state per position)

(The entry below saying "push blocked" is outdated: main 0f27043 is pushed, Actions build artifact `botzone-upload` exists.)
Dataset packed shuffled: data/nnue/relabel/positions.bin (33,891,139 records) + rows.npy, so ANY finished set of rows is a random sample.
Workers (chunks of 5000 rows -> data/nnue/relabel/out/c_<firstrow>.txt, resumable):
- PC (terminal tab c9): rows 0-1.6M, ~41 pos/s, deadline 15:00, expected done ~12:45.
- tit-rl1 (Docker cloud, 16 vCPU): rows 1.6M-5.6M, ~227 pos/s, expected done ~07:00.
- tit-rl2 (16 vCPU): rows 5.6M-12,672,785, expected done ~10:45.
- tit-rl3 (8 vCPU): rows 12,672,785-16,672,785, ~113 pos/s, expected done ~12:05.
Docker cloud vCPU quota is 40, so the planned rl3-rl5 (16 vCPU each) could not be created (cost nothing). Rows 16.67M-33.9M (~17M, half the data) are NOT labelled
unless more is launched; ~16.7M rows (49%) will be.
Watcher data/nnue/relabel/watch_all.sh (log watch_all.log): every 30 min pulls chunks from each sandbox, restarts dead jobs, deletes a sandbox when DONE, hard stop 13:10.
Spend ~ $11.2 + $11.7 + $5.9 = ~$29 worst case (logs/docker_spend.csv; the sandboxes auto-delete after ttl 10.5 h anyway).
Merge: `python training/relabel_run.py merge --partial` -> data/nnue/relabel/labels.npz (rows, have, score, depth, nodes; scores include +-99998/99999 decided - clip when training).
UPDATE (user: let PC cook until stopped, +2 h for Docker): watcher v2 checks every 10 min; when a sandbox is DONE it is pulled+deleted and a same-size
successor (tit-rl4, 5, 6...) starts on the next unlabelled rows (from 16,672,785 upwards, cap 33,000,000), deadline 15:00, final pull+delete 15:10 (state: sandboxes.txt,
next_row.txt). After its slice (~12:45) the PC crawls the unlabelled rows downward from 33.9M (data/nnue/relabel/pc_crawl.sh) with NO deadline - stop it by hand (kill python/titanium-cli) when training starts.
Expected extra cloud spend ~ $15 (total ~ $44). Old tit-rl1 ttl ends ~11:45, rl2 ~12:30, rl3 ~12:40 (auto-delete) - all expected done before that.
No training without the user's order. After rl1 finishes (~07:00) 16 vCPU of quota is free: a new sandbox on rows 16.67M+ could add ~4M rows (~$6).

## 2026-10-25 night - Botzone crash fixed locally; main 0f27043 NOT pushed (needs GitHub workflow scope)

Botzone v5 (cf94732, deployed by another agent) crashed on the bot's first move: titanium_v2.so needed glibc 2.34
(Botzone = Ubuntu 16.04 / glibc 2.23) - built on a modern runner. It also ran S57 only (the NNUE default was CLI-only).
Fix on exp/botzone-nnue, fast-forwarded into LOCAL main 0f27043 (staging too): zig glibc 2.14 build, FFI loads
nnue_main.tnn (peak 125 MB), driver 12 s on turn 1 / 6 s later, tools/build_botzone.py (one command, Docker test),
workflow rewritten. Normal search identical to 8b51d92. Upload folder ready: C:/gitProjects/main-ataxx/botzone/upload.
PUSH BLOCKED: GitHub refuses workflow-file changes from a token without 'workflow' scope (user must run
`gh auth refresh -h github.com -s workflow` + `gh auth setup-git`, then push main). Old branch exp/botzone-ffi is superseded.

## 2026-10-25 evening - main 8b51d92 pushed; relabel ready to plan

**Main (pushed):** 923dbea (NNUE c4829350 default) -> d1e0055 (eval cache + prefetch/head speedups, bit-identical, ~2x; +135 @100ms vs 923dbea, 360 g) -> dc937b3 (search to the deadline, time check every 64 nodes; user: no gate) -> 8b51d92 (cache-only eval move ordering EVORD_W 100; 5k gate stopped at +57 in 83 g, inconclusive; user merged). Staging = main.
**Speed facts:** 5k-node search ~70-75 ms (was 131); PC relabel 48 pos/s (labelbin, 8 threads). Dropped: int8 FT, register tiling, incremental accumulator, evaluate-every-child ordering (6x slower).
**Relabel (NOT started, user said wait for my 'ready'):** training/relabel_pack.py (8M subset packed -> data/nnue/relabel/positions.bin 9.36M rows), training/relabel_run.py (plan/pc/cloud/collect/merge; EXE + cloud source now = C:/gitProjects/main-ataxx (main 8b51d92, the engine on main)). Open: user wants WHOLE dataset (33.9M) shuffled; duplicate count not checked; cloud per-vCPU speed never measured (first test failed: uncommitted code). Labeller = main 8b51d92 (record the exe sha in the run).
**Botzone:** python driver fine (6 s one-shot, margins, watchdog); engine .so on exp/botzone-ffi is old (S57, b0c4329-level) - another agent owns the update.
**Running:** nothing. Docker ledger ~$11.6 of $200.

## 2026-10-25 ~00:20 STATE (usage limit hit mid-run)
- STAGING = d61d0c4 (fast int16 NNUE) by USER ORDER; engine config --tuple none --nnue data/nnue/champions/STAGING_NNUE_t1ep4_c4829350.tnn.
  Gate vs main: 5k +207 PASS, 100 ms +29 / 50 ms +13 (ahead, not conclusive). MAIN untouched (b0c4329 + S57 53F2B29C). Nets backed up read-only in data/nnue/champions/.
- RUNNING: hybrid grids vs main (exp/nnue-hybrid caa8f0b): cloud sandbox tit-hyb (move-share HYB_PCT 10/20/35/50/100, ttl 2h auto-delete)
  + local Docker container tit-local-hyb (material switch HYB_MAT 6/8/10/12). Results: logs/sbx_hyb/res/logs, logs/local_hyb. If no session is
  watching: `sbx --cloud ls` / `sbx --cloud rm --force <id>`, `docker rm -f tit-local-hyb`, pool with training/pool_glob.py.
- Docker ledger logs/docker_spend.csv (worst case ~$10 of $200 stop). Training STOPPED (relaunch needs --force-restart --hypergrad).
- NEXT: pick best hybrid -> gate vs staging/main; then default-NNUE switch + ask user for main merge; relabel with strongest engine only after user go.

## 2026-10-24 23:25 NNUE teacher run STOPPED by user order ("just stop training, not worth it for now")
- Best net = epoch-4 EMA (0.055853 Kata-TD MSE), exported data/nnue/nnue_t1ep4.tnn (c4829350). Cycle 0 bottomed out (ep5-8 all worse).
- The resume file holds the PRE-restart state (cycle 0): if training resumes later, relaunch WITH --force-restart --hypergrad.
- The box is now free for gates (local Docker Linux workers next to the cloud sandboxes).

## 2026-10-24 ~22:50 NNUE teacher run: FORCED warm restart (user: "make it look for better basins, it hit rock bottom")
- Cycle 0 bottomed out: epochs 5-8 all within 0.1% of best 0.055853 (ep4 EMA) at lr 5e-4 .. 3.1e-5.
- train_nnue.py: --force-restart (applied ONCE at startup: basin0 saved, warm restart from the global best), first restart
  jump = lr/2 (3.75e-4), smaller after each failed cycle in a row; relaunched with --auto-restarts 3.
- If the trainer has to be relaunched again (RAM guard, reboot): use the same command WITHOUT --force-restart.
- Also: the run now uses --hypergrad (D72, per-step LR scale, persisted in the resume file) - keep the flag on relaunch.

## PAUSED 2026-10-24 ~20:50 for a PC restart (user enables Windows Hypervisor Platform for Docker Sandboxes)
- NNUE teacher run (D69 --auto-lr) paused at ep6 chunk 16 checkpoint (resume file saved). RESUME with the exact D69 command:
  python -I -u training/train_nnue.py --stage teacher --init data/nnue/nnue_pbt/best_base.pt --out data/nnue/nnue_teacher_v1 --epochs 60
  --auto-lr --cos-epochs 12 --auto-restarts 2 --min-rel 0.002 --metric mse_kata_td --lr 7.5e-4 --lr-min 2e-5 --bs 4096 --width 256
  --opt ranger --wd 3e-7 --la-k 8 --la-alpha 0.56 --ema 0.9995 --threads 4 --min-free-gb 1.5 >> logs/train_nnue_teacher_v1.txt
- Modal gates of exp/search-nmpmb-b0 1df8ddb vs staging: 5k PASS +41; 100 ms +12 inconclusive (extension book 200-399 was running);
  50 ms rerun was running. Both are killed by the restart -> rerun training/modal_gate.py (OPP_SRC staging, CAND_SRC nmpmbb0-ataxx).
- After restart: user runs `sbx login`; then test one tiny Docker Sandboxes batch job (cloud? scripted? my own spend ledger < $200).

## STAGING LIVE 2026-10-24 (BEHAVIOR 24, D70): branch staging @ b0c4329 (= main), worktree C:/gitProjects/staging-ataxx
- Candidates now gate vs the STAGING TIP; a full pass -> agent merges into staging + LEDGER STAGE-<tag>; staging -> main needs user approval
  after a staging-vs-main confirmation gate (5k + 100 ms + 50 ms). Not pushed.
- NNUE trainer: --auto-lr (D69: cycles of backtracking + SGDR warm restarts from the global best, stop after 2 failed cycles in a row, then
  SWA over basins + a 2D plane map through the best and two best basins). Smoke test then the real teacher run (resumes ep5 chunk 16).

## PROMOTED 2026-10-24 night: main = b0c4329 (Moonbird history), PUSHED. STRONGEST = main b0c4329 + S57 (tupS57v2.tup 53F2B29C)
- Gates vs a3c9b74, same S57: 5k +200, 100 ms +159, 50 ms +169 (400g each, all PASS). LEDGER Whistmb5k/100/50, PROMOTE-histmb.
- PARKED (behind, inconclusive vs a3c9b74, not rejected): exp/search-rootpvs 40dd73c (-27 @5k), exp/search-nmpmb 027e07a (-11 @5k).
  Re-test both rebased on b0c4329.
- NEXT (user design 2026-10-24): search-tuning teacher. Teacher = WIDE search (no lossy pruning: no LMR/futility/RFP/NMP/IIR,
  or ~2x the student breadth; TT/ordering/PVS kept) to the SAME depth D as the student; student = tuned pruned search.
  Fitness = regret vs teacher at depth D + node cost (honest node counting incl. pruned-move static evals); keep the
  agreement-vs-nodes Pareto front, then game gates decide. Endgame <= ~10 empties: exact solve as teacher. Measure first what
  full-width depth 6-9 costs on ~20 positions. Branch exp/teacher-search (own binary, no hot-path flag).
- NNUE teacher run training (resumed 18:47 at ep4 chunk 8). Its gate must be vs main b0c4329 (rebase exp/nnue).

## PROMOTED 2026-10-24 evening: main = a3c9b74 (Moonbird-style search), PUSHED. STRONGEST = main a3c9b74 + S57 (tupS57v2.tup 53F2B29C)
- Gates vs main 653d9a1, same S57: 5k +263 (flattered: the child static evals of pruned moves are not counted as nodes), 100 ms +48
  [0.5225,0.6138], 50 ms +96 [0.5900,0.6787], 400g each, all PASS. LEDGER WsearchmbA5k/100/50, PROMOTE-searchmbA.
- Change: LMR from move 7 depth*47/(47+n); futility REDUCTION 1 ply per 57 cp the child static eval is below alpha (from move 2);
  moves reduced to <= 0 skipped; child eval reused via pre_eval. Depth @100 ms 7.08 vs 6.50 (Moonbird 11.4).
- main exe rebuilt e6a9090b, bench d8 a1b2 13 / 3410 nodes (identical to gated candidate). titanium-wasm does not compile
  (Searcher lifetime, predates a3c9b74): fix before any web/wasm build.
- NNUE teacher run RESUMED 17:49 at epoch 4 (saturation run, patience 2). Its later NNUE gate must be vs main a3c9b74 and the
  NNUE side must be built on a3c9b74 search (rebase exp/nnue).
- Next search levers (own branch + gate each): NMP R=4+depth/2, aspiration, history malus, RFP retune; cheaper eval makes A pay more.

## PROMOTED 2026-10-24 late: main = 653d9a1 (TUP7 eval fix), PUSHED. STRONGEST = main 653d9a1 + S57 (tupS57v2.tup 53F2B29C)
- Gates vs old main 298b7ae, same S57: 5k +77 (400g), 100 ms +45 (400g), 50 ms +26 (800g), all PASS; Moonbird odds +53. LEDGER PROMOTE-tup7fix.
- main exe rebuilt (45537061), bench d8 a1a2 13 / 10844 nodes. Speed not optimised yet (~2.4x slower per node than 298b7ae).
- The NNUE teacher run (D64) is RUNNING again (resumed at chunk 16, epoch 1). Epoch 1 done: val Kata-TD MSE .0587 (EMA)
  vs S57 .0663 vs base start .0686; level with Ataxx_CPP's raw value from 25 empties up, behind at <= 24 (NET_ANALYSIS 29).
- DAG LIVE (D67): data/dag/games_dag.db, 16,940 games / 1.44M nodes from all gate logs; gates feed it automatically
  (gate_nnue.py, ext_gate.py; bat-launched gates: run `python training/dag.py ingest "logs/gate_<tag>_*.txt"` after).
- Teacher run now TRAINS TO SATURATION (training/restart_teacher_saturate.ps1 switched it at ep2 chunk 16): up to 12
  epochs, plateau stop patience 2 / 0.2% on val Kata-TD; best checkpoint over all epochs -> data/nnue/nnue_teacher_v1.pt.
- LEVER QUEUE (main = champion, branches try to beat it; side branches for exploration):
  1. finish v1 teacher (3 epochs) -> export -> gate the NNUE vs main @5k (exp/nnue has the tuple fix; NNUE side --tuple S57 not
     needed for v1, use --tuple none on the nnue side as before) -> 100/50 ms legs if ahead.
  2. v2 big (D65: W 1024 + pairwise + hidden skips + engine-exact S57 skip, data/nnue/s57/*.f32 built with the fixed engine),
     base then teacher; capacity aimed at <= 24 empties.
  3. relabel the base data with current main 653d9a1 + S57 (~20% closer to Kata, NET_ANALYSIS 28; ~14 h on 8 workers).
  3b. (user 2026-10-24) the relabel is a LOOP: relabel every row's titanium_cp with the current champion (keep Kata labels
     where they exist) -> base stage again (warm start) -> Kata teacher fine-tune -> gate -> the new champion relabels again.
  3c. (user 2026-10-24) NOT before: (1) saturate the current setup, (2) relabel with the current Titanium + saturate again.
     Moonbird must first be PROVEN a better teacher at 5k nodes per position than KataAtaxx before any Moonbird relabel.
     Then: Moonbird 1.1.0 as SECONDARY teacher = its 5k-node SEARCH values on
     all positions (not its static eval: S57 static is already >= Moonbird static vs Kata, NET_ANALYSIS 23; Moonbird's edge is
     search, it beats Titanium ~8-192 at equal 5k nodes). Existing pipeline: training/relabel_moonbird.py (stopped at 1/238
     chunks by D57, resumable); calibration rows already carry Moonbird@1k labels. Then ablate with identical budgets and
     paired games: Titanium-only / Kata-only / Moonbird-only / mixed targets; also test using Moonbird only where it disagrees
     sharply with Titanium. Keep every stage's best checkpoint; judge by paired-match Elo vs main AND vs Moonbird.
     Watch: a loss plateau can be capacity, label noise floor, optimiser or distribution mismatch, not only "saturated".
  4. speed recovery for the tuple fix (after strength levers; profile capconv + env loop).
  4b. (user 2026-10-24) NNUE PRUNING after strength is proven: skip zero activations in L1 (Stockfish find-nnz), delete
     never-active FT neurons, L1-regularised fine-tune + magnitude-prune sweep |w| < eps gated per eps (fixed-node gate =
     strength loss, time gate = speed gain; raise eps until the speed gain stops paying), structured pruning (neurons,
     input groups) before quantisation.
  5. root-move ordering by Ataxx_CPP policy (NET_ANALYSIS 22).
  6. (user idea 2026-10-24, AFTER gating what we have) OPENING/GAME DAG like Quoridor_best_AI: one node per position (Zobrist
     key) with W/D/L + visits + children; self-play walks the known DAG with temperature (explore variants) and plays the
     engine's best once off-DAG (full-budget search, every position labelled once); later MCTS-style frontier expansion
     (eval all children at a known node, pick the best unseen). Gives opening WDL statistics (book + training targets) and
     millions of diverse self-play rows. Seed from data/nnue/outcomes.db (33.9M positions with W/D/L, mostly Kata self-play)
     + gate logs. Guard against the S21-era echo chamber (self-distillation fossilised): keep temperature + outside games
     (Kata, Moonbird). Cost guide: ~1 s/game/core at 5k nodes -> 1M games ~ 35 h on 8 cores.

## ENGINE BUG FOUND 2026-10-24 late: main plays TUP7 tuple nets != how they were trained (docs/NET_ANALYSIS.md section 27)
- Engine TUP7 loader skipped capn/conv and read `pa` 1060 bytes early (S53..S57); trainer zero-constraint only block 0 while
  the engine skipped state 0 of every block; env defined per stone in the engine vs every square in the trainer.
  Engine S57 vs trained S57: 66 cp mean, corr .963.
- FIX: branch exp/tup7-fix 653d9a1 (worktree C:/gitProjects/tup7fix-ataxx, from main 298b7ae, only tuple.rs). Parity after fix
  0 cp on every family (training/tuple_parity.py). Same fix also in the exp/nnue worktree (uncommitted there).
- GATE 5k DONE: fixed vs main, same S57, 400 paired games: **239-152-9, 0.609, +77 Elo, CI [0.5625,0.6538], PASS**; won both
  seats (121-76-3 / 118-76-6). A/A control same harness 197-197-6 = 0.500 exactly. LEDGER Wtup7fix.
- MOONBIRD: equal 5k nodes both lose ~8-192; at odds Ti 5k vs Moonbird 300 nodes: fixed 63-134-3 (0.323) vs main 52-148-0
  (0.260) on the same lines = fixed +53 Elo vs the same outside engine. LEDGER WmoonTup7.
- 100 ms time leg: training/run_tup7fix_gate_100ms.bat -> logs/gate_tup7fix100ms_p*.txt (running at the time of writing).
- Fixed eval ~2.4x slower per node. User: strength first, speed later. Merge into main needs the user's explicit approval.
- Follow-up: all titanium_cp base labels in the datasets were produced by the BUGGY engine eval (main 5k search); a relabel
  with the fixed engine is a candidate data improvement (not started).
- v2 skip: engine-exact S57 via `titanium-cli s57dump` (exp/nnue) -> data/nnue/s57/{split}.f32; val done, train/test owed
  (rebuild with the FIXED engine).
- The D64 teacher run is paused at a resume checkpoint for the gate (same command resumes).

## RUNNING 2026-10-24 late: NNUE v1 TEACHER stage on Kata TD (user go: "do what's best" -> option A)
- One job: `python training/train_nnue.py --stage teacher --init data/nnue/nnue_pbt/best_base.pt --out data/nnue/nnue_teacher_v1
  --epochs 4 --lr 1e-3 --lr-min 2e-5 --bs 4096 --width 256 --opt ranger --wd 3e-7 --la-k 8 --la-alpha 0.56 --ema 0.9995 --threads 4
  --min-free-gb 1.5`, log logs/train_nnue_teacher_v1.txt, resumable (same command continues from data/nnue/nnue_teacher_v1.resume.pt).
- Pre-checks: audit_final.py --suffix _v3 ok=true (33,891,139 rows); teacher rows 20,662,234, teacher_p3 no NaN, in [0,1]; feature rows
  == npz rows; smoke 1 chunk: val Kata-TD MSE .0684 -> .0652 (S57 reference on the same val Kata rows .0663). ~37 min/epoch, ~7 GB free.
- Next after it ends: compare val mse_kata_td with S57 .0663; if better -> export + parity + 5k-node gate (gate_nnue.py, needs go).
- Not done: S57-skip variant (D63) not built; this run is the plain v1 architecture.

## RESEARCH BLOCK 2026-10-24 evening: what decides value and move choice (READ docs/WHAT_MATTERS.md FIRST)
No engine change, nothing merged; main 298b7ae + S57 untouched. All new files UNCOMMITTED on exp/tonight.
- Ranked summary: docs/WHAT_MATTERS.md. Evidence: docs/NET_ANALYSIS.md Part 3 sections 12-24. Decision proposal D63 (unsigned).
- VALUE: stone PERMANENCE (cannot be converted next move) is worth 2.5-3x to Kata's value head (colour-flip test); Moonbird env/pattern
  tables next; reach/2-ply/parity ~0. S57's static eval already holds this cheap knowledge: all cheap groups close only 13-33% of
  its gap to a deep net (Ataxx_CPP). Missing part is DEEP and endgame-heavy (Kata ~70% PASSSCORING rule mismatch NOT separated).
- POLICY: cheap features cap at ~31% top-1 / 60% top-3 on Kata's full move (2-ply material 30%/53%). Kata jumps with doomed stones (92%)
  and trades material for safety. Root ordering: Ataxx_CPP one-pass policy finds Titanium's 1M-node move 44% (Ti@1k 37%, Ti@5k 47%).
- INTERNALS: block 0 = exposure/gain units (single neurons), block 1 = reach (distance-2 heads), value built up to block 11; no score or
  ownership targets in Kata's data; its ownership head is untrained.
- New scripts (training/): probe_kata_deep.py, probe_kata_flip.py, kata_choice_model.py, kata_why_moves.py, kata_look_*.py,
  screen_features.py, s57_residual.py, root_order_probe.py. Logs in logs/ (probe_kata_*, kata_choice_model*, screen_features, s57_residual, root_order_probe).
- PROPOSAL D63 (needs user go): NNUE v2 = S57 tuple skip + hidden layers on the residual + Kata-TD teacher stage + cheap permanence inputs;
  D61 anchor/threat/jump-2ply demoted to ablation arms. Separate search lever: Ataxx_CPP root policy ordering (gate at 5k nodes).
- Not analysed: what Kata blocks 3-11 compute; rule-mismatch share of the endgame gap; any Elo for any of this.

## STATE AT END OF SESSION 2026-10-24 (READ THIS FIRST; supersedes the 2 'RUNNING' entries it replaced)
NOTHING IS RUNNING. Nothing merged. main (298b7ae, S57 net final_v2_wdl_ep8.tup sha 53F2B29C) untouched and still strongest.

### 1. Research results (docs/NET_ANALYSIS.md parts 1+2 = the ground truth for the NNUE design; read it)
- All strong engines evaluate within 2 squares (3x3 capture ring + 5x5 jump reach). KataGo's net infers stone EXPOSURE and jump reach in layers 0-1 (probe R2 .91 vs .47 raw).
- Values of Kata/Ataxx_CPP/Moonbird depend on stone safety/exposure (+.07..+.14 R2 over material); Titanium 5k search values ignore them (+.00) -> Titanium labels cannot teach safety.
- POLICIES: Kata's 200-visit search puts 60% of mass on JUMPS (top move is a jump 59%). Both Kata and Ataxx_CPP choose by the 2-PLY EXCHANGE (gain - enemy best recapture), not raw captures (Ataxx_CPP policy = best-2-ply move 52% vs random 7.5%; Kata jump top dest = best 2-ply 70%).
- ANCHOR TEST: best-2-ply square 19.8% top-move hit > max-capture square (what exp/nnue G2 uses NOW) 12.3% > block-enemy 9.1% > random 6.1%. => current anchor is the WRONG definition.
- autaxx (beat us 2:1 equal nodes): tiny NNUE, raw board 98 inputs -> 32/perspective -> 64->32->32 with skip, 38 KB. Hidden layer >> input richness.
- Stockfish 2026 (read from source): inputs Full_Threats (attack-graph edges) + PP_3Wide (pawn pairs) + K32Q2 (king-COARSE-bucket x piece-square); L1 1024, pairwise product, layer stacks 32/32, skip, |p-q|^2.5, lambda schedule.
- Moonbird we play = tag 1.1.0, ternary tables, 2.58 MB (data/engines/moonbird_1.1.0, exact Python port verified 300/300). data/engines/moonbird_src is a NEWER unreleased 25.7 MB checkout. Our `mob` feature port counts clone-only; Moonbird counts distance<=2.
- Kata data rules: 70% PASSSCORING (Botzone end rule) + 30% PASSCONTINUE, random loop rules, komi noise.

### 2. NNUE v2 design decisions (docs/NET_ANALYSIS.md section 11; each = ablation arm, games decide; NOT built yet)
1) anchor = landing square of the best 2-ply move over ALL moves (clones+jumps); arms exact vs coarse symmetry bucket, 1 vs 2 anchors. 2) Stockfish-style threat edges: (side, clone|jump, landing sq, direction of adjacent enemy stone) 1,568 inputs. 3) 2-ply exchange for jumps too (vacated origin). 4) keep G1 square state. 5) skip connections from both hidden layers; width 128-256. 6) lambda schedule start->end. 7) search: order captures by 2-ply net.

### 3. Code + data state
- Branch exp/nnue, worktree C:/gitProjects/nnue-ataxx (from main 298b7ae), COMMITTED locally as 02c3f29 (not pushed); project docs/scripts committed on exp/tonight 95c1b3b: crates/titanium/src/nnfeat.rs (v1 feature groups G1/G2/G3/G5/G6, 26,196 inputs, brute-force tests pass), nnue.rs (f32 inference, Rust==torch 0 cp), search.rs set_nnue hook (old path identical: bench d8 10915 nodes), CLI nndump/nneval/nnbench + match --nnue/--opp-nnue (use --tuple none on the nnue side).
- Speed v1 (unoptimised): features 1.5 us + f32 refresh 256w ~3.8 us vs tuple eval 0.95 us.
- Features on disk: data/nnue/nnfeat/{train,val,test}.{cnt,idx} (v1 groups; 13.2 GB train; REBUILD after v2 feature changes via `titanium-cli nndump`).
- Trainer training/train_nnue.py (Ranger/EMA/warmup+cosine, stages base|teacher, --drop groups, compact() file-read loader); optuna_nnue.py (done: study nnue_base_v1 in data/nnue/nnue_hpo/optuna.db, top.json: Ranger, bs 4096, lr 3-5e-3, la_k 8; width importance 0.0); pbt_nnue.py (stopped after round 15, state data/nnue/nnue_pbt/state.pt resumable, best_base.pt val 0.01717); export_nnue.py (+ .ref.txt parity file); gate_nnue.py (export + parity + 8x200 games @5k + tripwire; never run).
- Fitness reference (same held-out val): S57 base-target mse .0217 / Kata TD .0663 / result .2036. NNUE v1 PBT best: base .01717 (better) but Kata TD ~.0687 (worse). No NNUE has played a gate.
- Dataset v3 (data/nnue/final/final_*_v3.npz, streaming builder training/build_v3_targets.py): Kata TD h1 on Kata rows, Titanium 5k on own rows, wsum_kataonly + ow_row columns (ow_row estimator is BROKEN: do not use). Kata label sidecars kata_labels_*_nopol.npz.
- Analysis scripts: analyze_moonbird.py, probe_concepts.py, probe_kata_internals.py, probe_kata_policy.py, probe_kata_jumps.py, probe_cpp_policy.py, nnue_baseline.py.

### 4. User rules restated this session (also in BEHAVIOR/AGENTS)
- Titanium 5k labels = Stockfish BASE stage data (never call it a teacher). Kata = the teacher. Stage 1 whole dataset, stage 2 Kata rows.
- No training/relabel/long job without an explicit order; no spiralling (one agreed plan, not 3 runs in an hour); talk plainly.
- Check RAM before any big job (16 GB box crashed once); never memmap a 13 GB file in a long run; stream columns.
- Research before building: dismantle the strong engines (values AND policies) and test design choices against them.
- Invert problems with bitboards (spread targets) instead of enumerating moves.

### 5. Next (needs user go): rebuild nnfeat to v2 -> brute-force tests -> re-dump features -> short ablation trains (1 group/arm) -> Optuna seed -> PBT -> commit the v2 engine change -> gate_nnue.py vs S57.

## KATA-FIRST DATASET v3 BUILT 2026-10-24 (dataset-only; NO training, NO engine labeling; details docs/KATA_FIRST_V3.md)
- AUDIT: KataGo's own 200-visit TD value beats the v2 Ataxx raw relabel in every empties band vs the real result (MSE .008 vs .142 at <=4 empties, .143 vs .171 at 25-35). v2 had thrown TD away; own-game rows (36%) had a flat target (std .076).
- BUILT `final_{train,val,test}_v3.npz` (v2 columns untouched + teacher_p3/teacher_cp3/teacher_kind/proven/wsum3/kata_td_win/kata_td_draw/kata_visits) via `training/build_v3_targets.py`; Kata rows = TD h1 (weight share 82%), own rows = Titanium 5k sigmoid(cp/400) x0.5 weight (18%), all rows kept, row order = v2 so sidecars stay valid. Kata sidecar `kata_labels_*_nopol.npz` (restore script OOM'd on policy; --no-policy works; 1 corrupt raw file -> 1 row falls back).
- TRAINER: `--weight-col`, per-kind target summary, missing `--teacher-prob-col` is now fatal (was a silent cp fallback). Launch file `training/run_v3_wdl.bat` (same init/opt/epochs as v2, ow 0.15) NOT RUN: law 14. Forward-only mock-epoch wiring smoke only (logs/mock_v3.txt).
- DECISIONS FOR USER: (1) train v3 (explicit order), (2) lambda ow sweep after, (3) whether to Moonbird-relabel the 11.9M own rows afterwards (not run; training/teacher_bakeoff.py unrun), (4) src0 weight 0.5 / horizon h1 are my picks.
- UNCOMMITTED: all of the above is untracked/modified on exp/tonight; nothing merged.

## CORRECTION 2026-10-24: Botzone bot = 6 s per move, ONE-SHOT (no long-live); driver d514e90 on exp/botzone-ffi; true-limit test 5.912 s/move margin 87.5 ms; first move 6 s (12 s unverified). See docs/AXNET_PROBE.md section 9.

## BOTZONE LIBRARY READY 2026-10-24 (details: docs/AXNET_PROBE.md section 9)
- Branch exp/botzone-ffi (worktree C:/gitProjects/botzone-ataxx, from main 298b7ae): Botzone-rules search + stop flag + titanium-ffi C ABI + Python driver + simulator. Normal search identical to main (node counts). Package in botzone/dist, NOT uploaded; Linux .so not executed locally. Engine commits NOT merged to main (needs user approval).

## AXNET PROBE + ATAXX_CPP/KATA INVESTIGATION 2026-10-23 (full record: docs/AXNET_PROBE.md)
- STRONGEST BUILD NOW: main-ataxx 298b7ae (S57) + final_v2_wdl_ep8.tup (sha 53F2B29C; main default name tupS57v2.tup). Pushed/merged by user order.
- KEY DATASET FINDING: v2 src0 rows (11.69M, 36pct) have a calibrated near-flat teacher_p (std .076, logit slope .112 vs raw). src0 relabel with raw Ataxx W/D/L is DONE + structurally verified (data/nnue/final/ataxx_relabel_src0/, 47 chunks) but NOT MERGED. Merge = dataset v3. teacher column semantic bug (=1000*p_win) to fix. NO training without explicit order.
- Ataxx_CPP: strength is mostly its POLICY head. Policy-only (1 forward pass) == Titanium S57 at ~30k nodes. Value-only 1-ply weaker (10k nodes: Titanium 22.5pct vs 45pct vs policy-only). Titanium beats full Ataxx_CPP at 0.1 s (79-21, +230) and 1 s (29-11, +168) on this CPU (0.71 ms/eval, no VNNI). Deployed bots use 1 s (C++) / 3.94 s + AVX512 (Python #1): 4 s NOT tested.
- Grafted Ataxx value as Titanium eval (branch exp/ataxxcpp-eval, pure Rust port, parity-verified): 4-108 (-573 Elo) vs S57 @5k; 2-46 vs classical. Not a bug (parity+sign tests pass); alpha-beta cannot use a saturating win-prob value.
- Kata 2026 transformer: CPU ~2 visits/s (ignores maxTime), DirectML crashes on both GPUs (887A0006). Cannot gate at 1 s here.
- TRAPS: `go evals 1` returns first generated move (use `policy`); engine.exe lacks `set` (use engine-wdl/ctl); pool_gate.py is hardcoded to gate_v2 logs; gate_summary.py has a %-format crash; backslashes get mangled in tool heredocs.
- OWED: user decision on v3 merge; solver solve.h unvalidated; optional 4 s gate. Nothing merged or rejected this block.
## v2 NET PASSED GATE 2026-10-23 (Ataxx-relabeled teacher, +46 Elo @5k)
- Candidate `data/nnue/final_v2_wdl_ep8.tup` (sha `53F2B29C…`) vs MAIN `tupS56twostage.tup` (`36346A67`) @5k nodes, net-only change. 400 games = 4 disjoint paired-book legs × 100g (starts 0/50/100/150), 4 parallel workers, exe `D992E3CC` both sides.
- **W-L-D 225-172-3, score 0.5663, Elo +46, paired-bootstrap 95% CI [0.5212, 0.6100], eloCI [+15, +78]; seats 200/200; 56 pairs won / 29 lost / 115 split. PASS under GATE LAW v5.** Logs `logs/gate_v2_ep8_p{0..3}.txt`, pooled by `training/pool_gate.py` (gate_summary.py has no pooling).
- Infra verified: cand-vs-champ 4g replayed BIT-IDENTICAL; champ-vs-champ differs (117/92 vs 113/102 plies) → `--tuple` honored on both sides. NB match header prints `--net`, not `--tuple`, so "net: none vs none" is expected, not a bug.
- Training: 8 epochs banked (`final_v2_wdl_ep1..ep8.tup`), val_mse .039160→.036727, plateau never fired (rel_improv 3.8e-3 at ep8). Killed at ep9 chunk 40/41; resume manifest intact → `training/run_v2_wdl.bat` continues ep9-20 if wanted.
- Dataset v2: 20,662,235 outcome-only rows relabeled with Ataxx_CPP raw W/D/L; corr(teacher_p,outcome) 0.490 (was 0.157); audit_v2 ok=true; sidecars unchanged/valid.
- NOT merged: needs explicit user approval naming the net (law 10/18). D54 (8-epoch stop) + D55 (gate PASS) in docs/DECISIONS.md.

## MERGED TO MAIN 2026-10-22 + DATASET v2 READY 2026-10-23
- main-ataxx: commit `5c691cd` (pushed), `DEFAULT_TUP = data/nnue/tupS56twostage.tup` (sha `36346A67…1F29AC`). Exe `D992E3CC` (differs from gate exe `B3422646` only by the const). Gates: gate_summary 3600g 1917-1642-41, score .5382, CI [.5219,.5545]. Strongest = main+tupS56twostage.
- GATE LAW v5 now in BEHAVIOR law 8 / AGENTS law 10: PASS = 95% CI lower bound > 0.5, no magnitude.
- DATASET v2 DONE (75 min): all 20,662,235 `teacher_src==1` rows relabeled with Ataxx_CPP raw W/D/L (new `batchwdl`, engine-wdl `642DD019`, static build fixed 0xC0000135). 83 chunks train + 2 val + 2 test, 8 workers.
- `final_{train,val,test}_v2.npz` (33,891,139 rows), `final_audit_v2.json` ok=true 0 problems; p sums 1.0 (max err 1.4e-7); corr(teacher_p,outcome_p) **0.490** vs 0.157 for titanium_cp; p_draw==0 (Ataxx draw head saturated).
- teacher_src mix UNCHANGED (0:11.69M 1:20.66M 2:28.5k 3:152.5k) deliberately: `--dirs` drives sidecar feature layout, so the 52 GB SP/DE + base sidecars stay valid (row order preserved).
- `validate_pipeline.py` green on v2 → PIPELINE READY; launch config `training/run_v2_wdl.bat` (4 ep, warm start champ, sidecar-fed, resume, ckpt per ep). NOT launched: law 14 needs explicit "train".
- D51 merge SIGNED+DONE, D52 relabel SIGNED+DONE, D53 (raw eval + ply=2*pieces + src kept) NEEDS-REVERIFICATION.
- NEXT: on explicit order → run v2 training, then 5k-node gate (net-only change) + 100ms confirmation vs main `tupS56twostage`; then val/test relabel already included so no drift.

## GATE PASSED 2026-10-22 (two-stage net objectively better than champ)
- GATE LAW v5 (user order): +20 Elo / LLR +2.94 threshold REMOVED (it was agent-worded, never signed). PASS = 95% CI lower bound on score > 0.5.
- abB_twostage.tup vs champ tupS55tree.tup @5k: 3600g (8 parallel workers on 8 cores, disjoint paired book), W-L-D 1917-1642-41, score 0.5382, Elo +27, scoreCI [0.5219,0.5545], eloCI [+15,+38]. Seats 1800/1800. Tripwire bit-identical.
- Hashes: exe B3422646, champ F66A00FE, candidate 36346A67. Merge-ready; NOT merged (needs explicit user approval naming the net + promotion row).
- Harness fact: match has NO --threads; speed comes from N parallel match processes on disjoint book ranges. 8 workers = ~8x throughput.
## A/B GATED 2026-10-22 (neither arm beats champ; relabel trigger live)
- Arm A (one-pass teacher 1ep): 99-99-2 vs champ over 200g = 0.500. Arm B (base->teacher): 152-146-2 over 300g = +6 Elo. B-vs-A head-to-head: 53-44-3 (+31, inconclusive).
- Val said B ahead 0.14310 vs 0.14362; games say both ~= champ. Val lied again (law: only games crown). Determinism tripwire passed (bit-identical replay).
- D48 TRIGGERED (user condition: arms failed vs champ -> relabel KataGo rows with Ataxx evals, then retrain). NOT started; needs explicit order. Nothing merged, nothing rejected.
## FAST A/B RUNNING 2026-10-22 (prep gap-free, gate ready)
- Gaps closed with evidence: (1) sidecar + --tcol titanium_cp smoke OK (1 chunk, 59s, val .0216); (2) stage1->stage2 --init chain OK (val .1459, teacher target); (3) gate harness verified --opp self honors --opp-tuple (main.rs:1052-1055) and 2-game book smoke mirrored seats 1-1; (4) gate_summary.py written (gate_check.py absent) and parses smoke correctly.
- Debug switch added: --debug-chunks N (partial-epoch wiring test only, never for candidates).
- Running: training/run_ab_fast.bat -> armA (teacher 1ep), armB step1 (base 1ep), armB step2 (teacher 1ep). Logs abA_onepass.txt / abB_stage1.txt / abB_stage2.txt. Est ~105 min.
- Gate ready: training/run_ab_gates.bat (100g/arm vs champ tupS55tree F66A00FE at 5k nodes, book starts 0 and 50, exe B3422646 both sides). Verdict via gate_summary.py; never merge/reject on the bound.
- D44-47 unsigned (see DECISIONS). D48 = if both arms fail: relabel KataGo rows with Ataxx evals, then retrain (user-conditioned, NOT started).
## BASE-STAGE TARGET ADDED 2026-10-22 (D37-39, smoke-tested, NO training launched)
- MEASURED from final_train: teacher_p == outcome on teacher_src=1 (20.66M rows, corr 1.000). One-pass therefore trains 63.5pct of rows on pure outcome; Titanium 5k labels (titanium_cp, all rows) were NEVER a target. --tcol only offered sc/teacher.
- Added --tcol titanium_cp to train_stream.py: base-stage target = clip(titanium_cp,+/-1500), 25pct outcome mix. No new data, no new sidecars (sidecar is position-keyed).
- Smoke: 200k rows, 1 ep, val_mse 0.0224, smoke_stage1.tup written. Wiring works.
- D39: val_mse NOT comparable across target columns -> A/B must be decided by games only.
- Next (needs explicit order): A/B one-pass (teacher, 4ep) vs two-stage (base 2ep -> teacher fine-tune 2ep), same total compute, then paired-book 5k gate. Est ~4h train + ~1h gate.
## RECIPE AUDIT 2026-10-22 (found: one-pass != Stockfish; lambda IS used)
- Lambda is implemented (--outcome-weight; ow 0.25 = lambda 0.75: 0.75*teacher_p + 0.25*outcome_p). NOT avoided. Value unsigned (D11).
- Real gap: docs/STOCKFISH_LC0_RECIPE.md:48-55 says one-pass merged target_cp is NOT the Stockfish recipe; needs stage1 base (own cache) then stage2 fine-tune on teacher rows. We train final_train.npz (teacher_src 0/1/2/3 merged) in one pass. Logged D35.
- Also logged D36 (warm start from champ vs Stockfish from-scratch base). Both NEEDS-REVERIFICATION; no training launched.
- What was NOT verified before first launch: recipe-level sanity (two-stage, wsum 98x, lambda value, warm start). What WAS verified: sidecar parity, manifest counts, column presence, forward-only smoke. Lesson: verified my own new code, never audited whether the plan made sense.
## OPTUNA PORTED 2026-10-22 (built + wiring-checked, NOT launched)
- training/optuna_final.py: TPE + multistage + PBT ported to final dataset (full rows), WDL truth columns, conditional sidecar (dirs match only), fresh OUTDIR/STUDY_DB/LOG. Original optuna_stage1.py untouched.
- Wiring verified with --trials 0 (study created, clean exit, zero training executed). Single verification trial + full study both await explicit user order. D40-43 remain unsigned.
## DECISIONS FILE LIVE 2026-10-22 (governance, user-ordered)
- docs/DECISIONS.md created: every impactful decision attributed (user-order w/ quote, agent-mine, prior-session), status SIGNED vs NEEDS-REVERIFICATION. Backfill covers recipe (D10-14), pipeline (D20-25), prior sessions (D30-33), port sub-decisions (D40-43, explicitly unsigned after unclear answers).
- BEHAVIOR law 18 added (decision signing + lifecycle + no-claim-without-tool-call) + AGENTS.md pointer. User owns all verdicts; nothing of mine counts as theirs.
## TRAINING STOPPED 2026-10-22 (explicit user order, mid-ep-4)
- Killed PID 28356 clean. Progress banked: ep1 val 0.1436, ep2 val 0.1430, ep3 ckpt written (final_wdl_ep3.tup); ep4 died at chunk ~33/41. Checkpoints final_wdl.tup + ep1/ep2/ep3 + resume files all intact on disk.
- Resume point: relaunch identical command with --resume to continue ep4. Free RAM was healthy (1.8-2.1GB); stop was user-ordered, not a crash.
- Open question from user: global vs local optimization. Current run is LOCAL (Adam, 1 seed, champ init). Global options queued for after gate: (a) cyclic+SWA+multiseed, (b) Optuna TPE/ASHA with game-gate selection, (c) architecture change past the 29k-pair ceiling.
## TRAINING RESTARTED 2026-10-22 (base precomputed, PID 11208)
- Killed PID 1552 mid-startup (no artifacts, verified clean). Root cause of slow startup: base_eval_stm = ~150 full-array passes + 100x2GB popcnt churn, unmeasured in prep.
- Fix: --only-base built 41 base sidecars (~9min one-time); trainer loads precomputed base (parity-verified). Startup base phase: minutes -> 0s.
- Relaunched identical command, PID 11208, same log. Val materializing, epoch loop next. Kill-test protocol still owed after chunk 2-3.
## TRAINING STARTED 2026-10-22 (explicit user order 'launch training immediately')
- CMD: train_stream.py --cache final_train.npz --sidecar-dir sidecar --wdl-loss --wdl-scale 400 --outcome-col outcome_p --teacher-prob-col teacher_p --tcol teacher --blend 1.0 --outcome-weight 0.25 --dirs 0,1,2,3 --chunk-rows 800000 --opt adam --lr 1e-3/3e-3 --wd 0 --init tupS55tree.tup --epochs 4 --max-epochs 8 --plateau 1e-4 --seed 7 --threads 4 --resume --ckpt final_wdl_ep{ep}.tup --out final_wdl.tup. PID 1552, logs/train_final_wdl.txt.
- This is the FIRST live test of the --sidecar-dir path. Watch epoch 1 chunk 1 for crash; resume manifest bounds any kill to 1 chunk. Kill-test protocol still owed on this run.
- Expected status per chunk: rows, feat/mat/train secs, train_mse; per epoch: train_mse + val_mse. Ask for status => read log tail.
## TRAIN-OPT 2026-10-22 (config-only, no training)
- Kernel benchmark (fwd+bwd, no step, 32k rows): 4 threads optimal (307k rows/s @bs4096; 8 threads slower, contention). bs16384 only +9% over bs4096 -> kept bs=4096 (batch size changes dynamics; user's call).
- train_stream.py: l2 term skipped when --wd 0 (math-identical); launch bat set --wd 1e-6 -> 0 (negligible either way, flagged). --threads stays 4.
- Epoch estimate (from kernels, NOT a live run): compute ~2-4min + IO ~1-2min cold/seconds warm => ~3-6min/epoch. validate_pipeline.py re-run: PIPELINE READY.
## PIPELINE READY 2026-10-22 (prepared, NOT launched)
- Launch config: training/run_final_wdl_sidecar.bat (WDL, ow 0.25, champ init, sidecar-fed, resume on, ckpt per ep, logs/train_final_wdl.txt).
- training/validate_pipeline.py prints PIPELINE READY: files exist, 41 sidecar chunks cover 32,534,256 rows, WDL columns + teacher_src mix verified, forward-only smoke 2x4096 in 0.1s (no_grad, no optimizer, no writes).
- CHECKLISTS pre-training gained: validate script, sidecar match, kill-test protocol. Kill test NOT executed (executing it starts a training run). Awaiting explicit user order.
## SIDECAR DONE 2026-10-22 (dataset-only, no training)
- training/materialize_sidecar.py built all 41 chunks (32,534,256 rows, SP 332 + DE 100 int32): ~52.4GB in data/nnue/final/sidecar/ + manifest (sha per chunk). Resumed after abort at 31/41, finished clean.
- --verify-only PARITY OK on 2 random chunks (fresh recompute == sidecar bytes); gather/slice logic parity-tested standalone.
- train_stream.py gained --sidecar-dir: chunk-aligned sequential loop (shuffled chunk order + in-chunk shuffle in train_pass), resume manifest carries sidecar flag. Random fancy-gather measured ~320s/800k cold (page-fault storm) -> replaced with whole-chunk sequential read + in-RAM mask.
- Measured: per-epoch features_for+materialize32 ~16.4min over full train -> sidecar ~1-2min cold epoch 1, seconds warm. No search in this path; outputs bit-identical.
- Trust boundary (honest): sidecar bytes verified, trainer wiring syntax-checked + logic-tested standalone, but --sidecar-dir never ran inside a live loop (no training ordered). First gated run is its live test.
## DOCS CLOSED 2026-10-22 (doc-only, no training)
- Wrote the four owed files: docs/INDEX.md (map), docs/MISTAKES.md (14 consolidated mistakes), docs/CHECKLISTS.md (start/pre-train/pre-gate/end), data/nnue/INVENTORY.md (all artifacts with verified bytes/hashes/status).
- New-session coverage: laws+mistakes auto-load via opencode.json (AGENTS/BEHAVIOR/HANDOFFS/README); discoverability now via INDEX+INVENTORY. Still owed (data work): proven slice, Ataxx W/D/L rebuild, 179k gap. LEDGER dup Wenginewdl row left as-is (cosmetic).
## VERIFY 2026-10-22 (dataset-only, no training)
- Final set re-verified read-only (mmap): train 32,534,256 out .5087/op .5075/tp .5095, teacher_src 0:11691002 1:20662235 2:28501 3:152518; val 677,670 / test 679,213 same profile; audit ok=true zero NaNs/dupes/leaks.
- Stage1 (11.7M, out .497, teacher mean +7) + stage2-KataGo (20.9M, out .513, teacher mean +45, unclipped interim) exports exist with manifests; stage2 superseded by final.
- Engine audit: main-ataxx label binary B3422646 (matches record), exp build C5665FEB, engine-wdl B3C9D870, Ataxx_CPP base 4D152EEE, Kata cpu B3A67AAF, autaxx 250A710D; champ net F66A00FE re-verified.
- Prior ordered steps (libsql reload, outcomes import+merge, Ataxx rebuild, final join/clip/audit) confirmed done from artifacts; not re-run. Proven slice + Ataxx W/D/L-head rebuild + 179k gap still owed. No training run, none ordered.
## FINAL DATASET READY 2026-10-22 (training FORBIDDEN until explicit user order)
- Final set: `data/nnue/final/` train 32,534,256 / val 677,670 / test 679,213 = **33,891,139 rows**; `final_audit.json` ok=true, zero NaNs, no duplicate keys, no train/val/test leakage.
- Per row: KataGo `outcome_p` + W/L/D counts from 53.6M pre-dedup rows (22.5M seen once, 8.1M twice, 3.35M 3+, max 13,669 repeats); Titanium 5k `engine_cp` on ALL rows (32,751,139 relabeled, errors=0, main-ataxx ea0bdf5 + tupS55tree F66A00FE); deep labels kept where present (1.92M train).
- Teacher precedence locked: Ataxx-W/D/L > KataGo > deep > engine; recorded per row in `teacher_src`; KataGo rows keep the KataGo teacher (never relabeled).
- Per-source WDL calibration fitted and stored (`teacher_calibration.json`): Ataxx a=0.001,b=-0.5 (MSE 0.226 vs 0.296 at /400); deep a=0.003,b=0; engine a=0.001,b=0. `teacher_p` stamped on all splits (mean ~0.51, range [0,1]).
- `teacher` clipped to [-1500,1500] (raw Ataxx q*1000 had +/-99999 tails); `sc` recomputed; audit re-run ok=true after clip.
- Missing by design (next data work, NOT training): proven/forced-outcome column (engine-wdl `set`/`wdl`/`stat` built and verified; `training/label_proven_slice.py` ready, terminal-safe via `pass` on no-move positions, chunked with skip-if-exists); Ataxx full W/D/L-head rebuild; 179,581 own-cache rows still without Ataxx labels (fallback to deep/engine).
- Trainer hardening done and compile-checked: WDL loss (`--wdl-loss --outcome-col outcome_p --teacher-prob-col teacher_p`), double weight-decay removed, destructive freq-decay default off, per-chunk train_mse logging, atomic checkpoint writes, `--resume` manifests bounding any kill to one chunk (kill-tested protocol defined, not yet executed).
- Forgotten-but-valuable inventory (on disk, none merged, do not train from them): tonight* net zoo (A0-E0, PD0, bucket, online600 — all parked/rejected); `golden.npz` 1200 rows @500k; `lab2m_*` 2M rows @20k with matching .out scores; `stage2_katago_ep1.tup` + `stage2_katago.tup` = INTERIM KataGo-only checkpoint, NOT the final net; `cache12M_golden*` dose variants; 40+ regretloop nets (dead lines); `position_outcomes.npz` 33,957,653 unique keys (source of truth for counts).
- Box state at pause: zero python/titanium processes; TitaniumKataGate + titan-watchdog DISABLED; one-time relabel task deleted after clean completion. TitaniumRelabelFinal shows Ready/N/A (spent, will not refire).
- Rules updated this session (BEHAVIOR 14-17, user-ordered): no training without audit-ok + explicit user order; one job at a time; resume-only restarts; bytes-are-the-receipt.

## TEACHER TELEMETRY FIX 2026-10-22
- Ataxx value head verified as W/L/D (search.h backup: value[0]=win, value[1]=draw, value[2]=loss; value[3]=score logit). Old raw q=(e0-e1)/sum conflates draw with loss.
- engine-wdl.exe built (src/engine.cpp +set/wdl/stat, no change to existing commands): wdl prints q,score,v0..v3; stat prints proven,pv,visits; set loads any board.
- Plan: calibrate teacher per source (sigmoid(a*x+b) fit to outcome_p), export teacher_p column + calibration manifest; proven rows get hard forced outcome, excluded from scale fitting; gate on 5k + 100ms.
- Docs: docs/TEACHER_POLICY.md updated; script training/label_proven_slice.py added (not yet run); training/train_stream.py gained --teacher-prob-col.
## STOCKFISH LC0 RECIPE + DATASET PIPELINE 2026-10-21
- `docs/TEACHER_POLICY.md`: KataGo = positions/outcome data only; no more KataGo generation. Ataxx_CPP = strongest teacher for relabels/labels. Precedence: Ataxx > deep > main-5k > old-5k > outcome. Ataxx cache rebuilt 12,187,486 rows; corrupt old quarantined.
- `docs/ML_TRAINING_RESEARCH.md`: SAM/ASAM, SWA/SWAD, Lookahead, cyclic restarts, PBT/ASHA/Hyperband/BOHB/Optuna, AdamW/NAdam/RAdam/Lion/Sophia/AdEMAMix/schedule-free/Muon; autonomous basin-search loop + citations.
- `docs/TRAINING_METHODS.md`: local-minima avoidance (curriculum, cyclic LR/seeds, regret/hard-example mining, SPSA/ES, data diversity) + compute-efficiency rules (small sweeps first, gate by games, prioritized sampling, quant-aware).
- `docs/STOCKFISH_LC0_RECIPE.md`: exact Stockfish recipe (Stockfish gensfen depth9/5k base first, then Lc0 data fine-tune; never train teacher-only from scratch; lambda mixes score/outcome; Lc0 filter).
- Titanium mapping: stage1 base = own cache + Titanium 5k + outcome; stage2 teacher = KataGo Ataxx 202604 + deep labels (lab2m 20k, golden 500k, deep_labels 100k) + outcome. Export stage1/stage2 separately.
- Current pipeline: Turso/libSQL DB `data/nnue/titanium_train.db`; master pipeline running build/import; relabel all rows with main-ataxx 5k (8 workers); then deep import/finalize/export. Not complete.
- Critical: use main-ataxx binary (main HEAD ea0bdf5, descendant of 73a859d), net tupS55tree.tup F66A00FE. `titanium-main` binary is older 73a859d. exp/tonight slower.
## ADVERSARY TRAINING RESEARCH 2026-10-21
- New doc `docs/ADVERSARY_TRAINING.md`: architecture/training of Ataxx_CPP, KataAtaxx, Moonbird, atx, AutaxxNNUE, Fairy-Stockfish, ataxx-zero-ai; Stockfish/Lc0 data recipe; bucket analysis; Titanium plan.
- Key facts: Ataxx_CPP = 6x64x4 transformer + graph/PUCT MCTS, INT8/VNNI, 4 empty-square phase nets, trained by distilling KataAtaxx ONNX labels (top-32 policy + q) into selfplay data, q_mix 0.75, then QAT + phase specialization. Moonbird = alpha-beta + 4x2/2x4 tuple tables SGD, 25MB params. atx = closed NNUE + persistent tree.
- Recommended Titanium path: use Ataxx_CPP MCTS as the "Lc0" teacher (high-visit policy/q/score/ownership), two stages (base selfplay/5k then teacher-rescored fine-tune), then test 2 vs 4 phase buckets only after opening/endgame data exists. Do not train teacher-only from scratch; do not judge by val loss; gate by games.

## STRONGEST OPEN ADVERSARY FOUND 2026-10-21
- Ataxx_CPP (EmptyBlueBox/Ataxx, commit d412e90d549ec1fb10cd96f998dc1db2064e740c, open source + weights) ported to Windows and beat Titanium 4-0 at 1s/move startpos: 34-15, 31-18, 26-22, 33-16. Author commit 6cf2d647 claims 16/20 vs atx at 3.94s. This is the strongest local adversary found, stronger than KataAtaxx at 1s.
- Build: data/engines/ataxx_cpp/build/engine.exe (sha256 61F8C184...67B5E6), tf.h Windows aligned_alloc/signal patch, engine.cpp chrono/1LL patch; wrapper training/ataxx_cpp_match.py; log logs/ataxx_cpp_4g_1s.txt. Uses champion.bin + p0-8/p8-15/p15-30/p30-50 phase nets.
- Ataxx_CPP vs KataAtaxx b18 (50 visits; Ataxx 1s): 4-0 (30-19, 31-18, 28-21, 34-15); log logs/ataxx_cpp_vs_kata_4g.txt.
- Ataxx_CPP vs Moonbird 1.1.0 (both 1s, startpos): 3-1 (23-26 loss, 26-22, 39-10, 30-19); log logs/ataxx_cpp_vs_moon_4g.txt. Ataxx_CPP is the strongest local/open adversary found (4-game smoke, not a 100g gate).
- atx remains closed/server-only. Botzone login Terminator_PL works with lowercase password; atx source endpoint denied (owner sqz, opensource false). No public atx repo found.

## NEXT: RUST BOTZONE FIX + SF WIKI REVIEW 2026-10-21
- Rust Botzone bot id 6ad7c69fbdad901f323ed0ca latest v4 6ad7d61d...: detail shows ranked=FALSE (Python bots ranked true, +50 Elo). The 0 matches is at least partly the rank flag, not necessarily compile. First re-rank via /mybots/rankbot then verify.
- Then compile the exact v4 source (viewsrc saved temp/opencode/rust_bot_src.rs, 174244B) with Rust 1.94.1 (Botzone) vs local 1.97.1; test --edition 2021 and default; fix any compile error; upload new version if needed; re-rank.
- Python bots live: long-live 4s id 6ad7dc32bdad901f323ee01c (v ...01d); normal 6s PY6 id 6ad7e0c3bdad901f323f6516 (v ...6517). Site Python [6,4]: 6s normal/12 first/4s long-live; .so /titanium.so; +50 Elo; keep separate.
- SF nnue-pytorch wiki: data quality/learnability > raw eval; SF depth9/5k selfplay, multi-PV4 diff100, opening book; Lc0-rescored data; train SF first then Lc0 fine-tune (Lc0-only worse); quant-aware clipping; games + multiple seeds; 16GB/8GB/800GB.
- Titanium actions: (1) two-stage 5k base -> deep/teacher fine-tune (NOT 3-10pct dose); (2) data coverage 84pct mid/5pct open/11pct end -> add opening/endgame; (3) multi-PV/soft policy labels (KataAtaxx raw policy); (4) multiple seeds; (5) quant-aware q2/W4; (6) wider student/feature width; CPU/GPU gap = no HalfKA-scale net.
- KataAtaxx raw eval works (start: whiteWin .42568 whiteLoss .57081); eval-in-AB test pending; 5 NPS acceptable.

## ATAXX TWO-VARIANT NIGHT 2026-10-20

- Rust bot id 6ad7c69fbdad901f323ed0ca latest version 6ad7d61dbdad901f323edbba (970ms normal, 1600ms first); ranked; 0 matches.
- Python bot id 6ad7dc32bdad901f323ee01c version 6ad7dc32bdad901f323ee01d; py36; simpleio+keep_running; ranked; source C:/gitProjects/botzoo/titanium_botzone.py; .so at /titanium.so (365336B md5 f4adbd3bd9ac659cb757111c280323da); net /nnue/tupS55tree.tup.
- Site Python time config [6,4]: 6s normal, first doubled 12s, long-live subsequent 4s. Wrapper budgets 11.5s first / 3.96s later. Windows DLL local test: 11.1s first / 5.50s later, legal moves, marker each turn.
- .so: Rust FFI ti_new/ti_new_game/ti_apply/ti_search/ti_free; built rust 1.97 + zig 0.13 target x86_64-unknown-linux-gnu.2.23 (glibc <=2.14 symbols); storage root.
- Pondering: NONE. Platform SIGSTOPs long-live bots between turns; CPU used while waiting would count against next turn. Bundle and .so search only when the request arrives; reader/thread search joined; opponent runs separately, no shared-core contention.
- Measurement: separate bots for Elo comparison; first ranked matches still scheduled. Optional local 970ms vs 3960ms time-control match.

## BOTZONE ATAXX v3 2026-10-20 (long-live format + time fix)

- Latest version 6ad7d39abdad901f323ed9bc (v2 6ad7d1ce..., v1 ...dcb); bot id 6ad7c69fbdad901f323ed0ca; simpleio+keep_running true, ranked. 0 ranked matches so far (scheduler; first match compiles v3).
- CRITICAL long-live input format (fff source proves it): turn 1 = turn number + current request (+ empty data/globaldata); every later turn = ONLY the 4-number request, NO turn header, no history/data. v2 wrongly expected a header every turn; v3 fixed the driver.
- Budgets: 900ms normal, 1500ms first turn (platform 1s / 2s). Top bots: fff (Simple IO long-live) 910ms max; Avataxx (JSON one-shot) 900ms; RawNN is Python JSON single net eval (6x language limit). Do NOT use the full 1s: fff leaves 90ms, TLE=loss.
- Local long-live sim both colors: legal moves, marker every turn, first 1.63s, later 0.90s. Source of truth: C:/gitProjects/botzoo/titanium-ataxx-v1.rs (patched v3).

## BOTZONE ATAXX SUBMITTED 2026-10-20 (late)

- Bot: titanium_ataxx_PL, id 6ad7c69fbdad901f323ed0ca, version 6ad7c69fbdad901f323ed0cb, RANKED.
- Flags: simpleio=true, enable_keep_running=true (long-live). Net uploaded: /nnue/tupS55tree.tup (726440B, HEAD verified).
- Source: C:/gitProjects/botzoo/titanium-ataxx-v1.rs (patched): loads champ net data/nnue/tupS55tree.tup (non-native), long-live loop (TT/killers persist across turns, reset at turnID 1), 4-corner start, BZ rules + repetition guard, 23-bit TT (~197MB peak), 800ms/1500ms first move.
- KataAtaxx strength probe: task TitaniumKataGate running 2 games vs the CPU/ONNX release via training/kata_bridge.py (setfen + two-stage jumps). Measured: KataGo ignores maxTime (5.8s/move at movetime 1000) and visits barely matter (10-13s/move at 100-400 visits), ~19s model load. Result pending logs/ladder_kata_2g.txt.
- Next: verify first ranked matches/compile; if long-live desyncs, set simpleio/keep false (same binary handles EOF); add ponder thread; monitor Elo.

## FINAL 2026-10-20 ~09:55 (session close)

- exp/tonight (pushed). MAIN UNTOUCHED ea0bdf5. Strongest = main + tupS55tree (F66A00FE).
- ONLINE600 (per-step search-in-the-loop, 0.5 outcome, regret weights): leg A 1000g +14 (CI [0.489,0.550], LLR +0.63), leg B 1000g -13 (LLR -3.83) -> pooled 2000g 981-980-39 = 0.500, LLR -3.20 -> PARITY. No merge.
- All eval retrains parity/negative: C0 +6 (2008g), A0 +6, B0 -15, C0gold +2, D0 -38, E0 -24, C1 -30, C2 -71 (rejected), PD -3, bucket -86 (rejected). 5k-label/value-only line is exhausted.
- Measured brain/teacher facts: 5k-vs-200k move disagreement 47pct; 200k/1M 73pct; 1M/2M 75pct; 2M/4M 77.5pct; all-three 62.5pct. Agreement with deep teacher improves while Elo does not. pa/capconv fit teacher but have ~0 outcome signal. Piece-count bucketing: teacher R2 .11, outcome .0015; additive correction -86.
- Fixed: S50 fused-bag q2 offset bug (q2 never trained; polluted singles/pairsH).
- Ladder @5k (proper gates, 0 forfeits): autaxx +54 over us, Moonbird +485, Sanctaphraxx far below; KataAtaxx CPU/ONNX downloaded (GTP->UAI adapter needed).
- Delivered: RAM-bounded streaming trainer; full-family trainer with q2 fix; feature/outcome/bucket audits; golden-dose pipeline (top-wsum and disagreement/1M variants); TUP8 bucket loader+eval (branch-only); bestmove --nodes/--tuple; exact move applier; online_step.py (per-step search-in-loop); export_sqlite.py provenance (nodes_used/depth/share/engine+net hashes); docs/GOLDEN_DOSE.md and docs/EVAL_FAILURES.md (includes owner's train-on-Elo/ES idea).
- Data on disk (gitignored): cache12M_golden(3pct)/golden6/golden10/disagree(3pct); golden.npz (1200 @500k); disagree1m.npz (531); converged.npz (243); trees.npz (degenerate, 14 nodes - retired); nets tonightA-E, A0/B0/C0/D0/E0, PD0, bucket, online600(+_0); all gate logs.
- NEXT (do first): dataset quality - dedup cache12M by position, filter rails/inconsistent labels, reweight wsum softly (power .3, capped), rebalance phases; then a fixed-subset LR/batch/optimizer sweep with 5k gates; SPSA/ES on eval scalars (owner Elo-loss idea) with antithetic paired-book games; tune online loop (regret scale, outcome blend); KataAtaxx adapter.
- Cleanup done: scheduled tasks removed, runners in logs/, no training running.

## MORNING STATUS 2026-10-20 06:35 (all jobs finished)

- D0 (outcome 0.25 + 6pct golden dose): 5k 1000g = 438-547-15, 0.446, -38 Elo, CI [0.415,0.476], LLR -7.92 -> CONCLUSIVE BELOW (reject candidate, needs user).
- E0 (outcome 0.25 + 10pct golden dose): 5k 1000g = 455-524-21, 0.466, -24 Elo, CI [0.435,0.496], LLR -5.61 -> CONCLUSIVE BELOW (reject candidate).
- Dose response: 3pct + outcome 0.50 = +2 (parity); 6pct = -38; 10pct = -24. Value-only golden dosing does not convert; higher dose hurts.
- Nothing running; box idle. Latest commit 3168647 on exp/tonight (pushed). Main untouched (ea0bdf5).
- Awaiting user calls: C2 (-71), D0 (-38), E0 (-24) reject confirmations. C0-golden (+2) and B0 (-15) stay parked per FOMO.
- NEXT LEVER (ready): PV-margin decision-boundary training (docs/GOLDEN_DOSE.md 3d + implementation sketch): add --nodes/--tuple to bestmove, mine shallow-vs-deep disagreements, value target 0.5 outcome + margin loss forcing the deep PV move. Then KataAtaxx GTP->UAI adapter and Turso dataset promotion.

## UPDATE 2026-10-20 ~02:40

- C0-golden (outcome 0.50 + 3pct golden dose) 5k gate: 495-488-17 over 1000g fresh lines 3108-3607, +2 Elo, CI [0.473,0.534], LLR -1.23 -> PARITY. Full 5k ladder is now: C0 +6 (2008g), A0 +6, B0 -15, C0g +2, C1 -30, C2 -71 (conclusive). Every value-only retrain variant is noise-level vs champ.
- Run D (outcome 0.25 + 6pct golden) training since 02:35; auto-gate at 03:05 (waits for EXIT then exports D0 + 1000g gate). Run E (outcome 0.25 + 10pct golden) scheduled 03:15, auto-gate 03:50.
- Implication: value-only golden dosing does not move 5k strength. Next lever is decision-boundary training (PV margin loss: force shallow search to prefer the deep PV move, not just fit its value) — needs per-move deep labels via the UAI serve path; documented in docs/GOLDEN_DOSE.md 3d.

> Older entries (2026-10-20 and before) + static reference sections moved to docs/HANDOFFS_ARCHIVE.md (same content, not auto-loaded).