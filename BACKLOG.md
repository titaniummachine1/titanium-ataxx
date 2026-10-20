# BACKLOG — living queue (newest first)

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
