# AXNET probe + Ataxx_CPP/Kata investigation (2026-10-23) — read before touching teacher/eval work

Everything here is measured; each claim names its log. Written so a post-compaction session does NOT start over.

## 0. Findings that change what we do next (ranked)

1. **v2 dataset: the src0 half is nearly flat (v3 opportunity).** In `final_*_v2.npz`, `teacher_src==0` (11,691,002 rows = 36% of train) has `teacher_p` mean 0.506, **std 0.076** — a calibrated/shrunk copy of the Ataxx value:
   `logit(old_teacher_p) = 0.112 * logit(raw_p_win) + 0.01` (corr 1.0000, mean |diff| 0.193). `teacher_src==1` (KataGo rows, relabeled with raw Ataxx W/D/L) has std 0.266. So the +46 Elo v2 net trained with a near-constant target on 36% of rows.
   **The src0 relabel is DONE but NOT MERGED**: `data/nnue/final/ataxx_relabel_src0/` (47 chunks, 11,691,002 rows, unique, finite, p-sum err 1.2e-7; log `logs/relabel_src0_rerun.txt`, 2531 s). Raw p_win there has the same scale as src1. Merging it (+ re-audit) = dataset v3. Do NOT train without an explicit "train" order (law 14). Merge script to adapt: `training/merge_ataxx_relabel.py` (it currently handles src 1 only; `teacher_src` must stay unchanged so sidecars stay valid).
2. **Ataxx_CPP's strength is mostly its POLICY head; alpha-beta is a poor consumer of its value.**
   - Policy-only (one forward pass, play top-prior move) equals Titanium S57 at **~30k nodes** (table 3).
   - Value-only 1-ply is clearly weaker than policy-only once Titanium has >=10k nodes (22.5% vs 45% for Titanium; CIs do not overlap).
   - Titanium search with the Ataxx value as its eval (graft) loses **4-108 (-573 Elo)** to S57 and **2-46 to the classical hand-eval** — graft is plumbed correctly (parity + sign tests) so this is a real property, not a bug.
3. **Titanium (S57, alpha-beta) beats the full Ataxx_CPP engine at short time controls**: 0.1 s each 79-21 (+230 Elo, CI +147..+330); 1 s each 29-11 (+168, CI +70..+301). Ataxx_CPP is only ~138 evals at 0.1 s and ~1372 at 1 s on this CPU (0.71 ms/eval, AVX2, no VNNI). Its Botzone deployment is **1 s (C++ bot) / 3.94 s with AVX-512 `engine-ice` + symmetric TTA (Python bot, the actual #1)**. We have NOT tested 4 s.
4. **KataAtaxx 2026 transformer cannot run at 1 s here**: CPU engine ~2 visits/s (0.45 s/visit, ignores maxTime, 4 visits ~2 s, 16 visits ~10 s); DirectML engine crashes on BOTH GPUs (Intel HD4600 and AMD FirePro M5100): `887A0006 GPU will not respond` at session init. The earlier "Ataxx_CPP beat Kata 4-0" used the OLD 2025 b18 at 50 visits. Smoke only (4 visits): Titanium 2-0.

## 1. Setup facts (verified)

- CPU i7-4900MQ Haswell, 4C/8T, AVX2+FMA, **no AVX-512/VNNI**. "8 workers" = ~4-5x one core, not 8x.
- Ataxx_CPP = `EmptyBlueBox/Ataxx` @ `d412e90`, weights `data/engines/ataxx_cpp/weights/{champion,p0-8,p8-15,p15-30,p30-50}.bin` (6 blocks, 64 ch, 4 heads, int8). Phase nets chosen by empties [0,9) [9,16) [16,31) [31,50).
- Value head order: **logits are [win, draw, loss, score]** (search.h). `rawWDL`'s `q=(e0-e1)/sum` in engine.cpp is **win-minus-DRAW (wrong)**; correct is `(e0-e2)/sum`. The `teacher` column in v2 = 1000*(that wrong q) = 1000*p_win (0..1000, not cp). Trainer reads `teacher_p` (correct: p_win+0.5*p_draw) so training was unaffected; fix the column in v3. p_draw ~ 0 because Botzone rules make final counts sum to 49 (odd) — draws essentially impossible.
- Rules: Ataxx_CPP = Botzone end rules (game ends when EITHER side is stuck; empties go to the mobile side). Titanium search is classic-rules; arena ends games Botzone-style (small end-of-game mismatch).
- Binaries: `engine.exe` 4D152EEE (original, lacks `set`); `engine-wdl.exe` 642DD019 (+set/wdl/stat/batchwdl; `go` output byte-identical); `engine-ctl.exe` 28C9ADA9 and `engine-ctl2.exe` (+ `batchfull`, `policy`, `valuegreedy`). WinLibs g++ at `~/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_*/mingw64/bin`. Build: `g++ -O3 -std=c++17 -march=haswell -static -o build/<name>.exe src/engine.cpp`.
- **`go evals 1` is a TRAP**: with one eval no child has visits, `best()` returns edge 0 = first generated move (not the policy). Use `policy` (top-prior move, one forward pass). My first "node ladder" used `go evals 1` and was INVALID (retracted).
- Titanium reference side for arenas: main exe `C:/gitProjects/main-ataxx/target/release/titanium-cli.exe` sha 3D0E2EFD with explicit `--tuple data/nnue/final_v2_wdl_ep8.tup` (sha 53F2B29C = S57 net). `tupS57v2.tup` does not exist in titanium-ataxx/data.

## 2. Tools written this session (all in `training/` unless noted)

- `ax_vs_ti.py` — arena: Titanium vs Ataxx_CPP (`--cpp-ms` full engine, `--cpp-evals N`, `--cpp-policy`, `--cpp-valuegreedy`) or Kata (`--opp kata --kata-visits N`). Paired book (50k), both seats, Botzone end rules, paired-bootstrap CI. Needs `engine-ctl*.exe` for policy/valuegreedy (`CPP_EXE` env overrides). Runners: `run_axti.sh`, `run_axti_policy.sh`, `run_axti_policy2.sh`, `run_axti_value.sh`.
- `run_ax_gate.ps1` — 8-worker fixed-node gate for the grafted-eval exe (branch below). Args: `-Tag -OppTuple -Scale -Mat -Exe -BookBase -Workers -GamesPerWorker -Nodes`.
- `pool_gate.py` is HARDCODED to `logs/gate_v2_ep8_p*.txt` — it ignores arguments and reprints the OLD v2 gate. Use a tag-aware pooler (logic in this session's `/tmp/pool.py`; rewrite if needed).
- `gate_summary.py` crashes on a `%` format bug in the "CONCLUSIVE BELOW" branch (line ~73): escape `95%%`.
- Branch **`exp/ataxxcpp-eval`** (worktree `C:/gitProjects/ataxxeval-ataxx`, from main 298b7ae, commits c222efd, fbb1b37, 774b291c, + later): `crates/titanium/src/axnet.rs` = pure-Rust port of the int8 transformer (AVX2 `vpmaddwd` matmul, padded attention, same exp/GELU as tf.h); hook at top of `search::evaluate`; env `AXNET_DIR` (weights dir), `AXNET_SCALE` (cp = scale*atanh(q), default 800), `AXNET_MAT`; `match` strips AXNET_* for the opponent child. Parity vs C++ on 400 dataset positions: q diff median 0, p90 3e-4, max 0.025; ~2 ms/eval (vs 0.76 C++). Unit tests: `matches_cpp`, `axnet_sign`, `axnet_agreement`, `axnet_speed` (need `AXNET_DIR`, `AXNET_REF`).
- `engine.cpp` additions (Ataxx_CPP src, untracked; data/ is gitignored): `batchfull` (net outputs + terminal flag + exact endgame solver `src/solve.h`; **solver NEVER validated** — the 20M-node test was killed; use small node caps), `policy`, `valuegreedy`.

## 3. Results (all logs under `logs/`)

| What | Games | Result |
|---|---|---|
| Graft: Ataxx value as Titanium eval vs S57, 5k nodes (`gate_ax5k_p*.txt`, exe D3091D64) | 112 | **4-108-0, -573 Elo**, CI [-818,-446], seats 56/56 |
| Graft vs classical hand-eval, 5k (`gate_axhealth_classical_p*`) | 48 | 2-46 |
| Titanium S57 vs full Ataxx_CPP, 0.1 s each (`axti_100ms.txt`) | 100 | **79-21, +230** [+147,+330]; as black 43-7, as white 36-14 |
| same, 1 s each (`axti_1000ms.txt`) | 40 | **29-11, +168** [+70,+301] |
| Titanium nodes vs Ataxx **policy-only** (`axti_policy_n*.txt`) | 40 each | N=1:2.5-7.5% ... 300: 7.5%, 1000: 15%, 3000: 42.5%, 10k: 45%, **30k: 52.5%**, 100k: 62.5%, 300k: 72.5% |
| Titanium nodes vs Ataxx **value-only 1-ply** (`axti_value_n*.txt`) | 40 each | 1: 0%, 10: 7.5%, 100: 5%, 1000: 12.5%, 10k: 22.5% |
| Titanium 1 s vs Kata 2026 CPU 4 visits (`axti_kata_smoke.txt`) | 2 | 2-0 (plumbing only; one game took 599 s) |

Titanium at 1-300 nodes is flat (~5%) vs policy-only. Moves DO change with node count (8-19/30 agreement with the 100k-node move, 16-27 distinct moves), so it is not frozen; it is just weak at tiny budgets.

## 4. Mistakes made this session (do not repeat)

- Launched a 112-game gate without proving the graft plays sanely (smoke went 0-2). Always run a health check (vs classical) first.
- Built a DLL the user did not want; user rule: **UAI binary only, no DLLs, no extra steps without prompt**.
- Reported the `go evals 1` ladder (39-1, 40-0) without reading the moves. Always read a few games.
- `pool_gate.py` output was almost reported as the new result — check the file glob.
- Backslashes in heredocs/Edit strings get mangled by the tool layer: use forward slashes, `chr(92)`, or the Write/Edit tools.
- Claimed src0 was "weak label type"; truth: same Ataxx value but calibrated/shrunk (see finding 1).

## 5. Open items (next session, in order)

1. User decision: merge src0 relabel into dataset **v3** (raw W/D/L for all non-deep rows), fix `teacher` column to a real cp (`400*logit` or `800*atanh(p_win-p_loss)`), re-audit, rebuild nothing else (sidecars stay valid). Then explicit "train" order -> gate vs S57 @5k, 4 legs x100g.
2. Optional: 4 s/move gate vs Ataxx_CPP (its Python-bot setting); 16 games ~ 30 min.
3. Optional: use policy head knowledge in Titanium (move ordering / policy-guided pruning) — the finding says the policy carries most of the strength.
4. GPU: DirectML does not work on this hardware for the Kata transformer (device hung). Ataxx_CPP-on-GPU relabel speedup is modest (~30 min saved per pass); not worth it unless many relabel passes are planned.
5. Fix `gate_summary.py` % bug, make `pool_gate.py` tag-aware, validate or delete `solve.h`.

## 6. Hybrid (Ataxx policy + our NNUE as leaf value) — TRIED, ABANDONED, CODE REMOVED (2026-10-23)

Idea (user): keep Ataxx_CPP's search + policy, replace its value head with Titanium's static eval. Built as a Rust PUCT in the exp/ataxxcpp-eval worktree (policy mapping validated: top move matched the C++ `policy` command on 39/40 positions).
Measured before the user stopped it:
- 100 ms, same policy + code, only the value differs: **hybrid[our eval] vs hybrid[his value head] = 10-90 (-382 Elo, CI [-512,-288], 100 games)**. Our eval as a leaf value is clearly WORSE than his value head in a policy-guided search (low budget, ~50 expansions/move at 100 ms).
- Speed probe (machine loaded): hybrid[our NNUE] 412 exp/s, hybrid[his value] 524 exp/s, original C++ 925 evals/s (1372/s idle). **No speed gain from swapping the value**: the transformer TRUNK (needed for the policy at every expanded node) is the cost; the value head is a few percent.
- Not run: equal-expansion quality test, 0.5/1/2 s ladder (stopped by user). Possible future idea only if wanted: run the transformer only near the root and use the NNUE below (not built).
All hybrid code, scripts and exe were deleted; worktree exp/ataxxcpp-eval is back to its last commit (the eval GRAFT only). Logs kept: logs/hyb_*, logs/diag_*, logs/lad_*.

## 7. Pending small fixes
- ax_vs_ti.py `Ti` class: pass `encoding="utf-8", errors="replace"` to Popen (Moonbird prints a byte cp1250 cannot decode -> reader dies). Needed before gating original Ataxx_CPP vs Moonbird (`--a cmd:<Moonbird exe> --opp cpp`). That match (1 s) was requested as a question by the user but NOT run.
- Moonbird 1 s vs Titanium S57 DONE: 6-34-0 (-301 Elo); 100 ms: 4-91-1 (-523). autaxx(NNUE) 100 ms: 49-46-1 (+11, run overlapped benchmarks).

## 8. Later findings (2026-10-23, after the hybrid) — read this section too

**Rules from the user for interpreting results:** the old 4-game smokes in HANDOFFS (Ataxx_CPP 4-0 vs Titanium, 3-1 vs Moonbird, 4-0 vs Kata b18) are NOT evidence; ignore them. For Titanium use LOCAL measurements of the current build (S57); Botzone numbers are only used for other engines. Our Botzone bot `titanium_ataxx_PL` was submitted with an older version and sits at roughly 2000-2300 (user-reported); it joined recently.

**Local results today (this CPU, equal time, paired book, Botzone end rules, logs in logs/):**
- Moonbird 1.1.0 vs Titanium S57: 100 ms 91-4-1 (about -523 Elo for us); 1 s 34-6 (about -301).
- Moonbird vs original Ataxx_CPP, 1 s: 34-6 for Moonbird (logs/moon_vs_cpp_1s.txt; side A = Moonbird).
- Titanium vs original Ataxx_CPP: 0.1 s 79-21, 1 s 29-11 (Titanium ahead).
- Titanium vs autaxx(NNUE bridge), 100 ms: 49-46-1 (user reads this as Titanium ahead; 96 games, overlapped some benchmarks).
- Local ordering: Moonbird strongest; Titanium ahead of Ataxx_CPP; autaxx about level with or just behind Titanium. KataAtaxx cannot run at 1 s here (CPU ~2 visits/s, DirectML crashes on both GPUs) so it is unmeasured.
- Moonbird facts: alpha-beta, 25 MB eval (4x2/2x4 tuple tables) trained on 50M positions from 6k-node games (docs/ADVERSARY_TRAINING.md section 4) = similar data scale to ours but ~35x more parameters. Not on Botzone; no public rating found.
- Ataxx_CPP build flags do not matter (-march=native -flto: 0.743 vs 0.731 ms/eval); the net is hand-written AVX2; no VNNI on this CPU.

**Botzone ladder snapshot** (ratings drift: user screenshot vs live fetch differ by up to 230 points for the same bot): atx 2670, KataAtaxx_mini 2381, Kissin_Flash ~2333-2373, cheetax ~2279-2334, PyJataxx ~2260-2333, Ataxx_Python 2096-2327 (README claims #1: false now), Ataxx_CPP 2033-2088, hce 1950, RawNN (Kata net, no search) 1938, sotax ab-nnue 1938, waffle nnue 1935, AutaxxNNUE port 1870. Our bot not located on pages 1,3,4,6,16,31 (pages 2 and 5 unchecked).

**KataGo raw data finding (big):** the extraction that built v1/v2 kept ONLY the final game result (`out`==`teacher`==`sc`, corr 1.000). The raw self-play files (data/datasets/KataGo_Ataxx_202604/*/tdata/*.npz, 834 files, selfplay cfg maxVisits=200, cheapSearchProb=0) also contain per position: TD-value targets at 4 horizons (globalTargetsNC cols 4/8/12/16 win, 5/9/13/17 loss, 6/10/14/18 draw), root visits (col 60, mean 198), and the full MCTS visit distribution (policyTargetsNCMove[:,0,:], ~185 visits over ~6.2 moves).
On 575,637 matched positions, correlation with the real game result / MSE: Kata TD-h1 0.719 / 0.115; Kata TD mean of 4 horizons 0.687 / 0.125; Ataxx raw relabel 0.442 / 0.188; constant 0.5 -> 0.232. By empties 1-10: Kata 0.941 vs Ataxx 0.514. Caveat: TD targets mix later search values (TD-lambda style), so they are closer to the result by construction; still a legitimate training target.
**Plan (dataset only, no training without explicit order):** `training/restore_kata_labels.py` attaches Kata TD-values (4 horizons), draw probs, visits and policy distributions to the v2 src==1 rows by (occ0,occ1) key into sidecars `data/nnue/final/kata_labels_{train,val,test}.npz` (read-only on v2). 5-file smoke passed: on rows with all occurrences present the aggregated Kata outcome reproduces v2 outcome_p to 1e-10. FULL RUN NOT STARTED (user interrupted the launch; ~25 min, 16 GB RAM box). Recipe idea (user): target = lambda*TD win-prob + (1-lambda)*outcome (Stockfish-style), sweep lambda; do NOT use the Ataxx raw relabel for src1; own-game rows (src0) keep Titanium 5k labels + outcome instead of the squashed Ataxx scalar; decide Kata-only vs Kata+own by an A/B decided by games. No new teacher compute is needed (the Kata labels already exist).

## 9. Botzone engine library + Python driver (2026-10-23/24) — branch exp/botzone-ffi (worktree C:/gitProjects/botzone-ataxx)
- Commits on top of main 298b7ae: 4f1e998 (Botzone-rules search: const-generic BZ, search_bz, Board::bz_over/bz_counts/botzone_start, external stop flag), 2f7f35b (crates/titanium-ffi C ABI, botzone/titanium_botzone.py driver, botzone/sim_local.py simulator, BZ search uses whole time budget), lockfile commit.
- PROOF the normal search is untouched: 42 engine tests pass; bench d8 10915 / d10 78698 nodes and sperft d7x12 292988 nodes IDENTICAL to main's exe; sperft 1.12 vs 1.13 Mnps. MAIN NOT MODIFIED. Merging the engine commits into main needs the user's explicit approval (iron law 1/10).
- main had NO Botzone rules (they lived only in the ungated exp/search-buckets "tournament" line) and the old bundle used the 4-corner start; the new library ports only the rules + start + repetition guard.
- Time: driver knows 12 s first / 4 s later (site config [6,4]), searches limit - 50 ms - time already spent - 2 ms, watchdog ti_stop at budget+12 ms. Real-limit test: 11.894 s / 3.949 s (margin 50.7-51.9 ms); engine overshoot median 1.0 ms, max 2.8 ms over 162 scaled moves; 6 sim games vs greedy all legal.
- No CPU-flag variants: x86-64-v3 build measured equal (1.14 vs 1.14 Mnps).
- Linux .so (zigbuild x86_64-unknown-linux-gnu.2.14, glibc <= 2.14, 9 exports) was NOT executed: no WSL distro, Docker engine off. Same source as the tested Windows DLL.
- Upload package (ready, not uploaded): botzone-ataxx/botzone/dist/{titanium_botzone.py, data/titanium_v2.so, data/nnue/tupS57v2.tup} (net sha 53F2B29C). New file names on purpose so the ranked bot's /titanium.so is not overwritten. The user uploads; the agent never logs in to Botzone. The user pasted their Botzone password in chat once: advise rotating it.

**CORRECTION to section 9 (2026-10-24, user): the Botzone Python bot gets 6 s per move and does NOT use a long-lived session.** The "12 s first / 4 s later, long-live" figures came from an earlier session's notes and were never verified on the site. The driver (commit d514e90 on exp/botzone-ffi) now defaults to ONE-SHOT: fresh process per move, whole history sent each time, budget = 6000 ms - 50 ms - process age (read from the OS) - 2 ms, whole budget used, watchdog +12 ms. True 6 s limit test: 5.912 s per move (spawn to exit), margin 87.5 ms; engine overshoot max 1.2 ms; scaled sim: 105 moves, worst margin 39 ms, all games legal. First move also 6 s (a doubled 12 s first move is UNVERIFIED; TI_FIRST_MS=12000 enables it once the user confirms it). Long-live kept as TI_MODE=long. Package refreshed in botzone/dist (driver sha 55a0455f..., .so unchanged 7f762039, net 53F2B29C). Upload settings: Python, Simple IO, long-live OFF, 6 s.

**Update (2026-10-24, later):** Botzone wiki (Bot + Ataxx pages) VERIFIED: first turn of a match gets double time, Python multiplier 6x (6 s / 12 s first), 256 MB, one usable core, Ubuntu 16.04. The judge's REPETITION rule is NOT "revisiting a position loses": a move is illegal iff the latest placement snapshot equals an earlier one and the move yields the same successor snapshot as then. The library (exp/botzone-ffi, titanium-ffi `ti_move_banned`, exact rule) was differential-tested against the reference judge (judge_ref.h): 57,951 legal-move checks incl. 656 judge-rejected repetitions, 5,546 illegal-move checks, 0 mismatches (botzone/judge_check). First move kept at 6 s by choice (user). Package refreshed: driver unchanged in behaviour, titanium_v2.so sha 7f762039 -> rebuilt (see dist), net 53F2B29C.
