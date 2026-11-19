# Teacher policy: KataGo for positions, Ataxx_CPP for teacher labels

Date: 2026-10-21.

## Decision

- **KataGo Ataxx 202604**: remains a source of positions and outcome data.
  Do not download or generate more KataGo data.
- **Ataxx_CPP**: the strongest available Ataxx engine. Use it as the teacher
  when new labels or relabels are needed.
- **Own Titanium search labels**: still useful for stage 1 and for coverage,
  but Ataxx_CPP is the stronger teacher.

## Precedence when merging labels

For a position, keep all labels, but prefer for the training target:

1. Ataxx_CPP teacher (`ataxx_q` / deep Ataxx label) when present.
2. Existing deep labels (20k/500k) when present.
3. Current `main-ataxx` 5k `engine_cp`.
4. Older `old_teacher`.
5. KataGo outcome only, if nothing else exists.

Never delete the original columns: `outcome`, `engine_cp`, `deep_cp`,
`ataxx_q`.

## Data flow

- Positions: KataGo + own cache + future selfplay.
- Outcome: KataGo/own game result; never overwritten by a teacher.
- Labels: Ataxx_CPP for relabeling; Titanium 5k for base/coverage.
- Final training target: WDL lambda blend of the best available teacher and
  outcome; teacher choice recorded in the manifest.

## Artifacts

- Ataxx teacher dataset: `data/nnue/cache12M_ataxx_clean.npz`
  - 12,187,486 rows, q mean +0.526, q std 0.309, outcome corr +0.186.
  - Source: Ataxx_CPP raw transformer, `champion.bin` + p0-8/p8-15/p15-30/p30-50.
- Corrupt old Ataxx file quarantined:
  `data/quarantine/cache12M_ataxx_clean_bad_20261021.npz`
- Current first run uses the already-exported KataGo/WDL stage-2 file; it does
  not generate new KataGo labels.

## What to use when more teacher labels are required

Use the Ataxx_CPP engine (`data/engines/ataxx_cpp/build/engine.exe`, extended
with `set`/`wdl`/`stat` in 2026-10-22) via the existing labeler/wrapper. Do not
restart KataGo relabeling. Prefer Ataxx labels where the engine is stronger;
KataGo positions keep their own teacher.

## Source-code semantics of the teacher (verified from source, 2026-10-22)

- The network value head is **W/L/D, not W/D/L**: `search.h` backup uses
  `value[0]=win`, `value[1]=draw`, `value[2]=loss` in `(w-l)/(w+d+l)`, and
  `value[3]` is a score logit used as `tanh`.
- The old `raw` protocol (`q`, `score`) is a lossy subset: `rawValue` computes
  `q=(e0-e1)/sum` and `score=tanh(value[3])` with **no proven flag**.
- The correct teacher protocol is:
  - `wdl` command returns `q, score, value[0..3]` for exact W/D/L use.
  - `set` loads any dataset board; `go` searches; `stat` returns
    `proven pv visits` from the search root.
  - Proven rows take hard ground truth `pv` and are excluded from eval-scale
    fitting and from the heuristic teacher column.
- KataGo stays the teacher on its own positions; Ataxx never overwrites a
  KataGo teacher except by explicit override decision.
