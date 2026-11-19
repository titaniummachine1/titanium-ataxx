# Dataset v3: Kata-first targets (2026-10-24) — audit, decisions, how to run

Nothing here was trained. Everything is read-only on v2 (`final_*_v2.npz`, sidecars untouched).

## What the audit found (measured, val split, 430k KataGo rows)

MSE of each label vs the real game result, weighted by `wsum`, by empties band:

| empties | KataGo TD h1 | KataGo TD h4 | v2 `teacher_p` (Ataxx raw) | constant 0.5 |
|---|---|---|---|---|
| 1-4   | 0.008 | 0.020 | 0.142 | 0.234 |
| 5-8   | 0.043 | 0.082 | 0.179 | 0.232 |
| 9-15  | 0.103 | 0.134 | 0.180 | 0.225 |
| 16-24 | 0.136 | 0.148 | 0.178 | 0.216 |
| 25-35 | 0.143 | 0.148 | 0.171 | 0.208 |
| 36-49 | 0.142 | 0.144 | 0.165 | 0.195 |

1. **KataGo's own 200-visit search value beats the Ataxx raw relabel in every band, by a huge margin in the endgame**
   (0.008 vs 0.142 at <=4 empties). v1/v2 threw the TD labels away (kept only the final result) and v2 then replaced them
   with the weaker 1-ply Ataxx value. TD h1 is also the best calibrated (reliability: predicted 0.45 -> actual 0.43,
   0.55 -> 0.57; h4 is biased low, mean 0.478 vs result 0.512). The four horizons correlate 0.98-0.999, so the choice matters little; h1 is the default.
2. **Own-game rows (11.9M, 36%) had a flat target in v2** (`teacher_p` std 0.076, a 0.11-slope shrink of the Ataxx value). v3 gives them the Titanium
   5k search value `sigmoid(clip(titanium_cp, +-1500)/400)` (std 0.147). It is still the weakest label (corr 0.20 with the result) -> weight x0.5.
3. **The two sources are complementary in coverage, so own rows are NOT dropped.** Kata rows: 22.6% at <=8 empties, only 1.1% at >=40 empties.
   Own rows: ~0% endgame, 34% at 30-45 empties. Dropping them would leave openings/early middlegame to ~1.4M Kata rows.
4. Trainer hole fixed: `--teacher-prob-col X` with X missing used to silently fall back to a cp path (garbage with the v2 `teacher` column = 1000*p_win). Now fatal.
5. Not changed, flagged: the trainer's validation set is a random 5% of the training file (same games), not `final_val_*`. Gates decide, so low impact.
6. `restore_kata_labels.py` OOM'd on the 2 GiB policy array; `--no-policy --out-suffix _nopol` works (280 s). One raw file is corrupt (`7F3450DB0670D210.npz`): 1 of 21.5M rows has no Kata label and falls back to the Titanium label.
7. No terminal positions exist in the dataset (0 rows full/wiped-out), so the "proven by board" rule fired on nothing. The `proven` column is a placeholder for engine-solved endgames.

## v3 file layout (`data/nnue/final/final_{train,val,test}_v3.npz`)
All v2 columns unchanged, plus `teacher_p3, teacher_cp3 (real cp, fixes the 1000*p_win column), teacher_kind (1 kata/2 own/3 terminal),
proven (-1 unknown/0 loss/1 win), wsum3, kata_td_win[N,4], kata_td_draw[N,4], kata_visits`. Build: `python training/build_v3_targets.py`
(`--horizons 1,0,0,0`, `--src0-weight 0.5`; seconds-to-minutes, idempotent). Report: `final_v3_report.json`.
Weight share train: Kata 82%, own 18%.

## Lambda (what is tunable at train time)
Target = (1 - ow) * `teacher_p3` + ow * `outcome_p`, `--outcome-weight ow`. `run_v3_wdl.bat` starts at ow=0.15 (KataGo TD is calibrated and
the single-game result is a noisy sample; v2 used 0.25 on a much weaker teacher). Sweep ow by games. Everything needed to change horizons, src0 weight or ow is stored.

## Run (needs an explicit user "train" order)
`training/run_v3_wdl.bat`: same init (S56), optimizer, 8 epochs, sidecars as v2 -> only the labels differ, so the gate vs v2/S57 attributes the result to labels.

## Teacher question (Moonbird / Ataxx_CPP for the own rows)
Not run (user: no labeling now). Own rows are the only place a relabel would matter. Ataxx_CPP is weaker than Titanium -> no. Moonbird is +485 Elo at equal nodes
and ~0.7 Mnps (probe: 20k nodes in 37 ms) -> the one credible Stockfish-style teacher for those rows; `training/teacher_bakeoff.py` (unrun) compares
Moonbird/Titanium/Kata labels against results and neutral playouts on a sample. Decide after v3 is gated.
