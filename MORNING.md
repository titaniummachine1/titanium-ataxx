# MORNING 2026-10-19 — overnight shift report (UPDATED 08:35, shift actually worked)

## LATE ADDITIONS (after first draft — you asked, I did)
- **O1 RESTORED, parity 100-100 exact** (book ah). Line == main at defaults. CorrHist
  confirmed OFF-by-default, not ported. The "rebase" is DONE as a restore (no rebase
  needed — merge-base was already 73a859d).
- **STALE-BINARY BUG found+fixed**: sloss EXE pointed at table-era target-loss (no
  PCut/Jump/Curve) — the intent/curve/pcut tunes tuned NOTHING. Fixed EXE->release,
  ported curve mode 3 in, rebuilt, reran everything REAL.
- **Final screens (all fresh books)**: intent-tuned 91-109 (BURIED), curve-tuned 89-111
  (BURIED), pcut-tuned 104-96 (parked neutral).
- **LAMBDA PIPELINE SHIPPED**: 1117 positions played out (2234 games, stm outcomes),
  deep_fresh_out.jsonl + --lamb flag (logit-blend, MSE core untouched). Demo tune runs
  clean, no convergence in 40it (honest neutral). Texel-on-eval = your call next.

## !!! READ FIRST (original note, now RESOLVED — kept for the trail):

## !!! READ FIRST: the working line is missing O1 (must rebase before any merge)
- Commit 8cdbb6b (exp/uai-protocol, "zero behavior change") REBUILT search.rs from a
  template WITHOUT the S42 O1 hole-aware jump ordering (+110..+270). Everything tonight
  (all screens) ran O1-less. Self-play verdicts are internally valid (same binary both
  sides) but absolute magnitudes may shift on the O1 base, and ORDERING mechanisms
  (intent) interact directly with O1's ordering — intent must be RETESTED with O1 present.
- Parity clean-tree vs sterile-main (73a859d, has O1): 94-106 (af) + 97-103 (ag) =
  191-209/400, LLR -1.71. Explained (missing O1), but note the gap is only -16, not -110:
  the main-vs-intent diff is 301+/150- (more than just O1+my blocks) — REVIEW the full
  diff in the morning; something else may differ (stale sterile binary? other dropped code?).
- Morning step 0: rebase working line onto main (restore O1 + tree2 default eval if
  applicable), re-verify parity ≈100-100, THEN re-screen probcut/intent/dynr deltas.
  Do NOT merge anything until parity holds.
- Eval note: screens used --tuple titan_champ.tup explicitly (both sides, consistent).
  Main now defaults to tupS55tree.tup (73a859d). Confirm which eval the morning gates use.

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
