# Checklists — run these verbatim, no shortcuts

## Session start
- [ ] `tasklist` / process list EMPTY of python/titanium jobs (one job at a time; kill orphans by EXACT binary name before anything else).
- [ ] `git branch --show-current` (expect `exp/*`, never main) + `git status --short` (know every dirty file).
- [ ] Strongest pointer confirmed: main + `data/nnue/tupS55tree.tup` (`F66A00FE`).
- [ ] Read HANDOFFS.md top 2 entries + BEHAVIOR.md laws. State the owed work out loud.

## Pre-training (ALL must be true; user "train" is one box, not the whole list)
- [ ] Dataset audit script prints `ok=true` on the EXACT files about to be trained.
- [ ] User explicitly said "train" for THIS run.
- [ ] Spec complete: proven/forced column, W/D/L rebuild, no silent gaps (179k-style).
- [ ] Teacher precedence verified per row (`teacher_src`); outcome column never removed.
- [ ] Trainer has `--resume` manifests + atomic ckpt writes; kill-test done, not claimed.
- [ ] No other job on the box (empty process list, verified not assumed).
- [ ] `python training/validate_pipeline.py` prints PIPELINE READY on the exact files.
- [ ] Sidecar manifest dirs/rows match the launch `--dirs` and cache (when `--sidecar-dir`).
- [ ] Kill-test protocol executed on the ordered run: launch, kill after 1 banked chunk, relaunch resumes.
- [ ] Exe sha + commit + net hash recorded for the run log.

## Pre-gate (fishtest rule)
- [ ] Both sides are git refs (branch/commit), never dirty; exe sha + commit + net hash both sides.
- [ ] `uainewgame` before EVERY game, both sides.
- [ ] Paired book, 50/50 seats per side verified from the log (never assumed from game count).
- [ ] Each leg ≥ 100 games; straddle = EXTEND, never merge/reject on the bound.
- [ ] Same line + seat replays bit-identically (fixed nodes) or the run is VOID.

## Session end (`/handoff`)
- [ ] HANDOFFS.md prepended (newest on top), LEDGER.md row appended via python, BEHAVIOR.md updated if laws changed.
- [ ] `git diff` inspected: only intended files changed.
- [ ] No training/label job left running; scheduled tasks cleaned; STOP file respected.
