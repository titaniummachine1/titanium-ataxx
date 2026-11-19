# Docs index — where everything lives (read this first in a new session)

## Memory files (auto-load via `opencode.json`, newest entry on top)
- `AGENTS.md` — project briefing: strongest build, iron laws summary, commands, traps.
- `BEHAVIOR.md` — standing orders: iron laws 1–17, mistakes ledger, checkpoints.
- `HANDOFFS.md` — session state, newest entry on top. Read the top 2 entries.
- `LEDGER.md` — experiment verdict rows (`Wxxx`). CRLF — append via python, never Edit.

## Guides in `docs/`
- `MISTAKES.md` — consolidated mistake list (this file's companion). Read before acting.
- `CHECKLISTS.md` — session-start, pre-training, pre-gate, session-end checklists.
- `TEACHER_POLICY.md` — teacher precedence (Ataxx-WDL > KataGo > deep > engine), W/D/L semantics, calibration.
- `STOCKFISH_LC0_RECIPE.md` — Stockfish/Lc0 data recipe mapped onto Titanium (two stages, multi-PV, quant-aware).
- `ML_TRAINING_RESEARCH.md` — optimizers/schedulers/HORS/autotune survey + citations.
- `TRAINING_METHODS.md` — local-minima avoidance, compute-efficiency rules.
- `PIPELINE_AUDIT.md` — dataset pipeline audit record.
- `ADVERSARY_TRAINING.md` — how Ataxx_CPP/KataAtaxx/Moonbird/atx train; our path.
- `WHAT_MATTERS.md` — ranked one-page summary: what decides VALUE and MOVE CHOICE (measured on KataAtaxx), and what follows for the NNUE. READ FIRST.
- `NET_ANALYSIS.md` — 2026-10-24: what Kata/Ataxx_CPP/Moonbird/autaxx/Stockfish look at (values, internals, POLICIES, anchor test) + NNUE v2 decisions; Part 3 = Kata internals, policy, choice model, flip test, value screen.
- `KATA_FIRST_V3.md` — dataset v3: Kata TD labels on Kata rows, Titanium base labels on own rows.
- `AXNET_PROBE.md` — 2026-10-23 Ataxx_CPP/Kata investigation: graft, arenas, policy-vs-value ladders, v3 dataset finding (READ FIRST for teacher/eval work).
- `EVAL_FAILURES.md`, `GOLDEN_DOSE.md` — dead eval lines (value-only dosing does not convert).

## Data inventory
- `data/nnue/INVENTORY.md` — every dataset artifact: path, bytes, rows, hash, status.

## Training scripts (`training/`, dataset-only unless user says "train")
- Export/verify: `export_final_join.py`, `audit_final.py`, `stamp_teacher_p.py`,
  `clip_final_teacher.py`, `calibrate_teachers.py`, `export_stage1.py`,
  `export_stage2_katago.py`.
- Label: `relabel_own5k_main.py`, `relabel_katago5k_main.py`, `label_proven_slice.py`,
  `build_ataxx_db.py`, `import_ataxx_db.py`, `build_outcomes.py`, `import_outcomes_db.py`.
- Train (FORBIDDEN without explicit user order + audit-ok): `train_stream.py`
  (`--resume`, WDL loss, atomic ckpts), `train_full.py`.
- HPO (parked until dataset gates): `optuna_stage1.py`, `autotune_stage1.py`.

## Engines
- Label binary: `C:\gitProjects\main-ataxx\target\release\titanium-cli.exe`
  (`B3422646`, main `ea0bdf5` + `tupS55tree.tup` `F66A00FE`).
- W/D/L + proven tool: `data/engines/ataxx_cpp/build/engine-wdl.exe` (`B3C9D870`).
- Strongest open adversary: `data/engines/ataxx_cpp/build/engine.exe` (`4D152EEE`).
