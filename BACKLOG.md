# BACKLOG - living queue (newest first)

## CURRENT ACTIVE BACKLOG (2026-10-28)

| # | Priority Task | Motivation & Evidence | Execution Path | Status |
|---|---|---|---|---|
| **1** | **Master Net Saturation ($W=512$)** | Bakeoff proved $W=512$ drops val MSE to `0.022424` (wins by $-0.000379$). 31.57M rows ready on Modal `ti-train`. | 8-10 epochs Lookahead Ranger, cosine LR ($10^{-3} \to 10^{-5}$), weak difficulty nudge (`--diff-nudge 0.15`). Checkpoints committed per-minute to `/vol/ckpt/sat_w512`. | **IN PROGRESS** (Candidate pruning `bakeoff_w512_nog5` running first) |
| **2** | **Disagreement Head Training (`TNN3`)** | Predicts $|\text{Search}_{5k} - \text{StaticEval}|$. Directly mimics PUCT epistemic exploration term in Alpha-Beta minimax. | Gather static predictions from saturated teacher, compute absolute search gap, train auxiliary quantile/residual head. | **QUEUED** (Awaiting saturated net) |
| **3** | **PUCT-Inspired AB Search Integration** | Dynamic LMR depth cancellation on high-uncertainty lines; +1 ply volatility extensions; aggressive pruning on quiescent nodes. | Wire `disagreement(s)` into `search.rs` LMR reduction formula: $\Delta d = \text{LMR}(d, m) - k \cdot \text{Disagreement}$. Gate vs `main @ 31de097`. | **DESIGNED** (Ready for implementation) |
| **4** | **Small Net Distillation (`fut_small.tsn`)** | Deployed engine requires high NPS. Distill saturated $W=512$ master net into lean $W=64 / W=32$ futility small net. | `smallnet_pack.py` $\to$ `smallnet_train.py` (bound mode) $\to$ `smallnet_export2.py`. | **QUEUED** |
| **5** | **Botzone Deployment Package** | Unified search upgrades (`31de097`) + persistent search state validated (+48.9% at 4s vs 6s). | Package hardened `.so` + new saturated NNUE, run 12-case Linux sanity test, ready for Botzone. | **PENDING MATCH COMPLETION** |

---

## CURRENT BACKLOG (2026-10-26 ~12:00; replaces everything below, which is history)

Method for every item (it is how fut-cheap/fut-sn were found and proven): PROFILE (count where the cost is) -> RECORD +
SIMULATE offline (decisions changed vs cost saved, held-out data) -> BUILD on its own exp/* branch -> GATE: 5k nodes
(per-node cost, diagnosis; 10 par on the PC or 24-32 on a 16-vCPU Docker sandbox, D111) then 100 ms + 50 ms vs main
(speed-affected: both time legs decide, law 2; one match per physical core) -> user approves the merge.
State at writing: main 1b286f2; exp/fut-sn (futility small net, D110) confirming vs main, merge approved by the user if it passes.
After that merge: profile numbers below refer to main 1b286f2 and must be re-measured on the merged build.

### A. Speed: big-net evals (D107: NNUE = ~95% of search time; fut-sn removed ~half of the futility child evals)
| # | Idea | Why / evidence | How | Gate | Cost |
|---|---|---|---|---|---|
| A1 | ~~Profile the merged build~~ DONE (D113) | big net 74% of time; 58% of ALL big-net evals = futility children pruned anyway after the big net (bound too cautious); leaves ~10% skippable; RFP <1% | exp/eval-probe 14e7fe7 + training/eval_probe_report.py | - | done |
| A1b | ~~Tighter futility bound~~ TRIED, DOES NOT PAY (D115/D116) | bound +40/+80: no Elo; better net (W64, tactical inputs): 74% vs 68.6% skipped but engine speed equal (small net costs what it saves) | exp/fut-sn2 parked | - | closed. Lesson: score small-net candidates by MEASURED ns per call |
| A2 | **Two-sided bound small net** | today only a LOWER bound (futility); leaves/RFP need "certainly above beta" AND "certainly below alpha" | add upper-quantile + median heads (pinball), train on node-entry positions + futility children (probe recorder v3) | offline: skipped evals vs changed decisions | ~1 h PC/Docker CPU |
| A3 | **Lazy leaf eval** (depth-0 nodes) | the leaf's static eval is backed up, so a guess costs strength (small-net error ~126 cp) - but a BOUND outside the window decides the cutoff | depth 0: lower bound >= beta -> return without the big net; upper bound <= alpha -> same (return the bound, fail-soft care) | 5k + 100 ms + 50 ms vs main | build ~1 h |
| A4 | RFP at node entry by bound (DEMOTED by D113: <1% of evals) | RFP asks "static eval + margin < alpha?" (and the beta side) | use the upper/lower bound; big net only when undecided | same | small |
| A5 | **Small-net move ordering** | EVORD uses cached big-net evals only (8b51d92); the median head is ~0.1 us per child | order by small-net median for uncached children | 5k first (ordering = per-node quality), then time | small |
| A6 | **Tune tau / margin of fut-sn** | tau 2% chosen from 2 points (0.2% / 2%); 1% and 5% untested in games | build tau 1%, 5% (heads already trained in sn_q32.pt) | 100 ms + 50 ms vs the merged main | 2 builds, ~1 h gates |
| A7 | **Big-net head, bit-identical speed-up** | head ~1.5 us of ~6.5 us per eval (f32 sparse L1, scalar L2/out) | AVX2/FMA like the small net, same summation order where needed for bit-identity | identity check (labels 400x5k == main), then 100 ms leg | small |
| A8 | **Feature computation speed-up** | features ~1.4 us/eval, G5 2-ply exchange the largest group | profile nnfeat per group, bit-identical rewrite | identity + 100 ms | medium |
| A9 | **int8 FT rows** (exp/nnue-int8 1e0e406) | +5% search speed, val +0.05%, 5k -12 [-45,+22] | re-test stacked on the merged build (eval share changed) | 100 ms + 50 ms | small |
| A10 | Narrower / feature-dropped big net | no free pruning (D107): width 192 +6.4% val untrained, G2 knock-out +5.7%; needs retrain | Modal/Docker retrain, distil from e11827d1 | 5k + time legs | $1-3 per arm; lower priority now |
| A11 | Incremental big-net accumulator | 64% of rows change per move; tried once, slower (SPEED-nnue) | only if A1 shows FT refresh dominating again | identity + time | low priority |

### B. Search
| # | Idea | Note |
|---|---|---|
| B1 | Policy net for move ordering / late-move pruning | ranking moves is a policy's job (Kata raw policy top-3 = search best 94%, NET_ANALYSIS); needs per-move labels; after A |
| B2 | Botzone end rules (either side stuck, 401-ply cap) | on exp/botzone-robust; correct per the judge, slower per node; own gate |
| B3 | Search-fix tripwire neutral (-5 [-39,+28]) | extend only on request |

### C. Data / training
| # | Idea | Note |
|---|---|---|
| C1 | **Self-relabel round** with e11827d1 | cheaper after fut-sn: same per-node strength (5k +3) at x1.77 speed -> ~1.7x more labels per Docker $ |
| C2 | Deeper Titanium labels (20k+ nodes) as teacher | hindsight-free search values (D101/D105: no external teacher beats Titanium) |
| C3 | **Retrain the small net with every new big net** | it is fitted to e11827d1: probe recorder -> smallnet_pack -> smallnet_train (bound mode, --init from the old small net) -> smallnet_export -> rebuild; gate 100 ms |
| C4 | Small-net capacity / Optuna | saturated at 126 cp (W32 H16); H32 only -2 cp, W64 worse; low value unless A2 changes the target |

### D. Harness / ops
| # | Idea | Note |
|---|---|---|
| D1 | Scripted Docker 5k gate | today hand-built (scratchpad job.sh); make training/docker_gate5k.py (par 24-32, tripwire, auto-delete) |
| D2 | Cross-OS determinism | Linux and Windows builds each replay bit-identically on their own machine; same-games across OS not checked |
| D3 | Botzone .so with fut-sn | after the merge: rebuild (small net is compiled in, no new storage file), tools/build_botzone.py 12-case test, user uploads |

### Done this session (2026-10-26, details in DECISIONS D107-D112, LEDGER)
- D107 speed probe (no training): eval = ~95% of search time; no free neuron pruning; int8 +5%.
- D108 exp/fut-cheap: hand-eval futility pre-filter, vs main 100 ms +69, 50 ms +59.
- D109/D110 exp/fut-sn: futility small net as a trained lower bound (tau 2%), x1.77 at 5k, vs fut-cheap 100 ms +41,
  50 ms +45 (832 g), vs main 5k +3; --no-fut-sn switch for exact full-strength search (D112).
- D111 gate parallelism sweet spots (PC 10 for 5k gates, Docker 24-32; time gates one per physical core).

---

## HISTORY (2026-10-19/20 night, superseded)

Autonomous night 2026-10-19/20. Branch `exp/tonight` (staging; user approves merges).
Iron laws unchanged: main untouchable; proven-ahead merges only; 5k = quality, time = crown;
>=100g legs; paired book by seat; uainewgame every game; probes never rejections; check FOMO.

## RUNNING

- **tonightA**: streaming full retrain, all families, 12.2M rows, 5 ep, Adam 1e-3 / conv 3e-3,
  wd 1e-6, teacher blend 0.9, init champ tupS55tree, chunk 800k, RAM guard 1200MB.
  Log `logs/tonightA_stream.log`. Task `TitaniumNightA`. Ckpt `data/nnue/tonightA_ep{ep}.tup`.
  Gate owed: 5k paired book 104g vs main ea0bdf5 + tupS55tree (candidate binary = exp/tonight, non-native tuple).

## RESULTS TONIGHT (2026-10-20 ~02:00)

- Full retrain (tonightA_fixed): val 10080.9 -> 7331.9 over 5 passes; all families trained
  (q2 21.5% nz, D1/D2 54/60%, mob 28/24%, pa 25.9% nz L1 162k), singles/pairsH/V near-stable.
- 5k ladder vs main ea0bdf5 (2008+ games, paired book, speed parity proven):
  - C0 (SP+env/mob; heads zeroed): 1009-972-27, 0.509, +6 Elo, CI [0.487,0.531], LLR -1.16
    -> PARITY-STRADDLE: park (no merge, no reject).
  - C1 (q2 active): -30 Elo straddle -> park/extend; q2 costs ~15% NPS.
  - C2 (q2+capconv+pa): 0.399, CI [0.305,0.493] -> REJECT CANDIDATE (user confirms).
- Insight: 5k legs are ~17s/104g; fixed-node gates are deterministic under CPU load -> large-N
  SPRT (2000g in ~10 min) is now the standard for any close call.
- Run B (outcome-weight 0.1, champ init): val 16824.8 -> 15690.2; B0 heads-zeroed gate 5k
  1000g fresh lines = 471-515-14, -15 Elo, CI [0.447,0.509], LLR -4.18 -> PARK (CI straddles;
  directional negative; per law never reject on a dip, extend later).
- Clean 6-epoch retrain (scheduled A re-fire, fixed q2): val 7192.9 best; A0 gate 5k 1000g fresh
  = 502-486-12, +6 Elo, CI [0.477,0.539], LLR -0.72 -> PARITY. 5k-label paradigm exhausted.
- Agreement (align20k): champ 124.4/corr .8604; A 110.7/.8463; B 111.4/.8437; C ep1 116.2/.8500
  -> every retrain fits the deep teacher better on mean error yet none wins games. Agreement is
  a diagnostic, not a promotion metric (confirmed).
- Golden labels vs real outcomes: corr(deep teacher, outcome_cp) = +0.029 on the 1200 mined
  positions (out mean 0.502). Top-wsum positions are early/midgame where the result is not
  determined -> the >=50% outcome anchor the owner ordered is intentionally noisy; Run C/D test
  whether it still helps via regularization/grounding. (Ataxx volatility: flip avalanches.)
- GOLDEN DOSE built: 1200 unique top-wsum positions deep-labelled @500k nodes (champ, native);
  cache12M_golden = 12.42M rows, dose 3% (repeat 20, wsum ~16x). Run C = outcome 0.50 +
  blend 1.0, 2 epochs from champ (user ratio: >=50% outcome anchor, rest deep/PV precision).
  Gate owed. Docs: docs/GOLDEN_DOSE.md (Apex pack1/pack2, Stockfish, teacher doctrine,
  replay-until-learn, tree tightening).

## QUEUED (in order)

1. **Gate A** (after training): rebuild candidate + main from clean refs; 4 shards x 26g,
   book-start 0/13/26/39, `--nodes 5000`, `--tuple tonightA.tup` vs `--opp-tuple tupS55tree.tup`.
   Then `count_gate.py`. If ahead -> t100 leg (speed affected: loader fix + nonzero capconv/q2).
2. **Run B — outcome grounding**: same stream recipe + `--outcome-weight 0.1` (real `out` column;
   S21g ban was on dag `sum_outcome`, not real outcomes). 3 ep. Gate identically. FOMO: parked.
3. **Run C — FAL/norm budget** (research-backed): per-row frequency-adaptive protection done
   CORRECTLY. Current `--freq-decay` is weight-shrink with hard clamp; calibrated badly for Adam
   (0.05/step wiped rare tables; see WstreamA). Options: (a) re-calibrate shrink ~1e-4..1e-3/step,
   (b) implement post-step row-update scaling (FAL: mult = log(F+1)/max log(F+1); Adam is
   scale-invariant to grad scaling, so scale the UPDATE, not the gradient), (c) norm budget on
   SP bag for epochs >1. Literature: arXiv 2505.05605 (Pinterest 2025), 2110.04844 (FA-SGD).
4. **Run D — EMA weights**: maintain shadow w <- d*w + (1-d)*w every N steps; export EMA net.
   Cheap, usually helps tables; compare vs A by gate.
5. **Candidate ladder (merge-cleanliness order)**: from the final net export three variants with
   training/zero_sections.py:
   - **C0 = zero q2+capconv+pa**: old-loader-safe AND speed-identical to champ (has_q2 all-zero
     skip path identical; env/mob compute cost content-independent). Intelligence-only change ->
     5k gate can merge alone; prove NPS parity with sperft alternating pairs.
   - **C1 = zero capconv+pa**: adds q2 knowledge, pays the quat splat per node -> needs t100 leg.
   - **C2 = full net**: needs the loader fix (exp/tonight) and t100.
   Gate order C0 -> C1 -> C2 (each vs main ea0bdf5 + tupS55tree).
6. **Deep labeling at night end** (CPU after gates): MEASURED 2026-10-19 23:47: `label --nodes
   20000` = 34.7 pos/s/worker (2000 pos in 57.6s). Full 12.4M relabel: 20k ~16.6h, 10k ~8.3h,
   5k ~4.1h (6 workers). So full-scale deep relabel does not fit a night. Better choices:
   (a) 20k relabel of a 1.5-2M hard/high-wsum subset (~2-2.7h) — S22 lesson says spend on rows;
   (b) 5k relabel with tupS55tree = self-distillation refresh (~4h, echo-chamber risk);
   (c) spend the time on Run B/extra training variants instead. Decide after gates.
   Pipeline script ready: scripts/night_label.ps1 (copies relab12m bases -> labels -> joins).

## RESEARCH NOTES (actionable)

- Pinterest 2025 (arXiv 2505.05605): embedding LR 50x dense (Sparse Optimizer); FAL row LR =
  eta*log(F[i]+1)/max log(F+1); multi-epoch overfitting hits RARE rows hardest; larger data +
  feature diversity mitigate; alternatives: re-init tables per epoch, stronger hashing.
- FA-SGD (arXiv 2110.04844): higher LR for infrequent rows speeds convergence in theory.
  Tension: FAL (ads, dense features per row) slows rare rows; FA-SGD speeds them. Our case is
  closer to ads tables (one weight per pattern, repeated rows) -> FAL direction likely right for
  multi-epoch, but only after epoch 1 (first epoch wants coverage).
- Our q2/env/mob/pa sections start at/near zero: first epoch is pure coverage; protection should
  scale in with epoch index (or frequency), never from step 0.
- SparseAdam note: torch sparse EmbeddingBag + dense L2 term keeps Adam alive (pure sparse grads
  crash Adam) — current code relies on this; keep the wd*l2 term.

## FINDINGS (fed back into queue)

- OUTCOME AUDIT (Wfeataudit2): real-result signal lives in singles/pairs/env/q2 (~+.015 R2),
  mob ~+.010, capconv ~+.005, pa +.0007. Same families vs TEACHER residual: pa .2935, pairs
  .2705, env .2233. => pa/capconv learn teacher idiosyncrasies, not wins (echo-chamber
  fingerprint, matches Wr1/Wr1dag failures). Consequence: C1 (zero capconv+pa) is now a
  first-class candidate, not just a diagnostic — it is loader-neutral and keeps the families
  with real outcome signal.
- Training speed reality: chunked featurization amortizes (12.7s/M rows vs 67s/M at 100k
  chunks); full 12.2M epoch ~5 min (feat 155s + mat 20s + train 120s).

## EXTERNAL LADDER (2026-10-20, 5k nodes, paired book, uainewgame)

- Titanium (candidate exe exp/tonight + champ F66A00FE) vs **autaxx 1.0** (built from source with
  llvm-mingw): 10-20 over 30 games (lines 4000-4014) -> autaxx is stronger at equal nodes than
  the "autaxx ~1800-1900 matches Titanium" assumption suggests.
- vs **Moonbird 1.1.0**: 1-19 over 20 games (lines 4030-4039) -> consistent with the 0-100 history.
- vs **Sanctaphraxx 0.1.11** (cargo build): 2-0 (small sample; it is an early work-in-progress).
- **KataAtaxx 20260501** CPU/ONNX release downloaded (331MB, engine_cpu/ataxx7x_cpu.exe, GTP
  engine + JRE + ONNX weights). Needs a GTP->UAI adapter to play in our harness; high value
  (Botzone #2, 2383). Autaxx/Sanctaphraxx builds work: autaxx needs the llvm-mingw bin on PATH
  for its DLLs; Sanctaphraxx runs standalone.

## DATA INFRA (started)

- `training/export_sqlite.py` writes any cache to SQLite/libSQL (Turso-compatible): positions
  with occ0/occ1/blk/stm, teacher, outcome, sc, wsum, source, **nodes_used, depth,
  best_move_share, engine_sha256, net_sha256** + indexes. Verified on golden.npz (1200 rows,
  engine 06E02C7D, net F66A00FE). JSON stays Botzone-only; Turso/libSQL is the training store.
- Golden cache builder (`make_golden_cache.py`) multiplies golden wsum instead of literal
  duplication when a weight-share is wanted; repeats are kept for Apex-style 20x writes.

## BUGS FOUND/FIXED TONIGHT

- **FUSED-BAG q2 OFFSET (Wq2fix, critical)**: q2 indices were materialized as AR2+Q (0..9215)
  with no SP_OFF['q2']=29160 addition -> q2 table NEVER trained (explains champ q2 0.0% and the
  tuple.rs "tree training never touched it" note) and q2 lookups polluted singles/pairsH.
  Inherited from the S50 fused-bag refactor (train_modern.py line 758). Fixed in both trainers;
  verifier re-proved bit-identical vs corrected reference. Epoch 1 of tonightA (buggy q2) is the
  warm start for the fixed continuation (log logs/tonightA2_stream.log).
- i8 quantization note: exported .tup rounds to integers; sub-0.5 in-memory drift disappears
  (a 24-step smoke exported byte-identical to its init). Never judge short runs by net bytes.
- Naming trap for future commands: in paths like `data\nnue\tsmoke3.tup`, a mistyped `\t`
  silently becomes a separator + 't' filename (visible only if the target name starts with t).
  Prefer forward slashes in tool commands.

- `train_stream.py` materialize32 initially used `AR` (missing *81) for singles — the historical
  S37 corruption signature. Fixed; verified bit-identical vs reference concat (5000 real rows,
  0/0 mismatches) after also fixing pair width (80 cols total, not 160).
- `train_full.py` wsum row-weights were not masked by the mate filter (12.4M vs 12.2M rows) ->
  misaligned weights on cache12M. Fixed: `sw = z['wsum'][keep]`.
- `freq-decay 2.0` with Adam: shrink = lr*fdec*rarity clamps at 5%/step for anything rare ->
  whole tables decay ~91%/epoch. Recorded; disabled for Adam runs. Needs recalibration before reuse.
- PowerShell `Start-Process -ArgumentList` arg-array does not quote elements with spaces: pass a
  single manually-quoted string, or the `--opp "path serve"` splits (AGENTS trap confirmed).
- Tool wrapper kills process trees on timeouts -> long jobs must run via SCHTASKS + .bat, never
  as tool children (training died at 22:56 this way; 2.5M-row materialization also paged the box).

## FOMO / RE-EXAMINE (never rejected on record)

- `Wtup7split` pa-only -17 Elo straddle (CI 0.380-0.572): now superseded by properly trained pa.
- `Wr1pa` -61 Elo straddle; `Wr1` -141; both frozen-base residual retrains with square-one
  labels; superseded by tonight's full retrain (same families, base unfrozen). Not verdicts on
  the FEATURES, only on that training attempt.
- `W1pext` parked (PEXT conv_key loses on this CPU); `NodeSets` neutral; W5/W6/W7 unbuilt;
  NMP-R2 0.560 straddle parked (search side; revisit after eval settles).
- Loader fix Wtup7gate -54 was weights-starved (champ pa max|w|=3). Tonight is the real test.

## BRANCH INVENTORY (local exp/*)

- exp/tonight (HEAD ff2f18d/e75d637): staging for tonight. Loader fix + training tooling.
- exp/winter-tup7fix: loader fix + night-close ledger (pushed).
- exp/winter-clean / q2skip / enrich / sets: winter speed work; winter-clean merged to main.
- exp/tuple-eval: tuple line (champ net origin; modern trainer source).
- exp/search-buckets: SMP+BZ batch + stash dirt (parked, never merged).
- Older: search-retune, corrhist, countermove, singular, sancta-incr, etc. (shelved/reverted).

## GATE PROTOCOL (verbatim for tonight)

- Candidate: `target\release\titanium-cli.exe` built at exp/tonight (clean ref, no dirty Rust).
- Baseline: `C:\gitProjects\main-ataxx\target\release\titanium-cli.exe` rebuilt at main ea0bdf5.
- Both: TUP7 tuple, non-native eval, `--nodes 5000`, opp `"<main> serve"` + `--opp-nodes 5000`.
- 4 shards: `--games 26 --start-game 1 --book data/nnue/book50k.txt --book-start {0,13,26,39}`.
- Log: `logs/gate_tonightA_5k_{a,b,c,d}.txt`; count via count_gate.py; seats must be 50/50;
  same line+seat must replay bit-identically (determinism tripwire).
- Record exe sha256 (both) + commit + net sha256 in LEDGER row. No dirty trees.
