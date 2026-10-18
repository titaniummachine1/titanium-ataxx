# MORNING 2026-10-19 — overnight shift report

## TL;DR (read this, then LEDGER tail)
- **ProbCut: artifact found + fixed, mechanism neutral.** The 163-37 was INTEGER OVERFLOW
  (`beta + margin` wraps at INF-beta PV nodes; the OFF-switch margin 2e9 was the loaded gun).
  Fix = saturating_add + skip when |beta|>MATE. Clean pooled over 4 legs: 408-392 (+7, LLR -0.4).
  UNRESOLVED, parked. Margins untuned (sloss 40it: no convergence, no fold).
- **Intent ordering (your idea): REJECTED at defaults.** 82-118 (-63, LLR -2.4, miss 0.5%).
  sloss 40it: thetas frozen (value-loss blind to ordering). Game-SPSA 56it: zero gradient
  (ordering is a step function of tier gaps; even ±1750 perturbations don't move it).
  Mechanism unresponsive to both tuners → buried, branch kept.
- **Dynamic null-R: neutral.** div=8 was a no-op (100-100); div=4 real test pooled 201-199 (+2).
  Parked.
- **LMR table/curve sloss: both no-signal** (random walk, no convergence). sloss tunes
  margins/gates (R1 worked) but not shapes. Lesson logged.
- **Box cleaned:** ~25 orphan processes killed (stale book100i wave); they don't affect
  fixed-node results, only timing.

## Branches (all pushed to origin)
- `exp/search-probcut` @3f83770: ProbCut + INF-guard + saturating. Binary-equivalent to release.
- `exp/search-intent` @1756975 (worktree HERE): + intent ordering + NullRDiv + all sloss/spsa
  PARAMS + LEDGER rows S55pcut2/3, S55intent/2, S55dynr/2. **target/release = this build**
  (sha dbdf8f69, NullRDiv live, intent defaults ON = the -63 weights; screens zero them as needed).
- `exp/search-lmrcurve` @9105a9d: mode-3 curve + night driver + HANDOFFS entry (partially stale,
  this file supersedes it). `exp/search-lmrtab` has the table code.

## Morning queue (priority order)
1. **Lambda-blend data project** (your Stockfish point): join OUTCOMES to deep positions,
   retarget loss to λ·outcome + (1−λ)·sigmoid(deep). Kills the S22 echo chamber. Then Texel
   hand-eval tune (tempo/contact/multicap) on blended labels.
2. **Move-agreement loss for ordering** (needs deep bestmoves — gen_deep doesn't save them;
   extend it, relabel 881 fresh pos @500k with moves, then sloss intent weights properly).
3. **Learned time management** (needs time-gates @100/200ms; untestable at fixed nodes).
4. **Phase-progress input (0..1) + symmetric D4 augmentation** for next training round.
5. Optional: ProbCut SPRT extension (needs user nod + 2 fresh books); SPSA retest of R1 thetas.

## Box lessons (do not relearn)
- Edit preserves mtime → `os.utime` after EVERY edit, verify `git diff` shows the change.
- Python stale `__pycache__` from mtime-preserving edits → delete `training/__pycache__`
  when imports act stale.
- place_bench orphans accumulate (engines outlive dead parents) → `taskkill /F /IM
  titanium-cli.exe /T + python.exe` when the box feels slow; they can't skew fixed-node
  results, only steal CPU. Consider adding cleanup to .cmd files.
- schtasks /tr 261-char limit; one-shot tasks + /run pattern works; STOP file =
  C:\gitProjects\logs\STOP (night driver honors it; no driver running now).
- LEDGER.md is CRLF — Edit tool fails on it; append via python script. LEDGER rows from
  tonight are on `exp/search-intent` (S55pcut2/3, S55intent/2, S55dynr/2); S55lmrtab +
  original S55pcut are on `exp/search-lmrcurve`. Collect at merge time.
- place_bench --a-setup/--b-setup apply per-engine setoptions (proven pattern for
  same-binary screens). Fresh books: af (100, seed 106) banked; ab/ac/ad/ae/aa/z/y/x used.
- uai_probe.py hangs in this env (never completed tonight) — use direct `serve` smoke
  via uai_smoke.in instead.
- Determinism proven: identical engines replay bit-identical 200g (100-100 on book z).
  Any future wild swing (like 163-37) = INFRA BUG until proven otherwise. Check beta/INF
  arithmetic first.
