# Proof hunting — the one runbook (D1589–D1604, 2026-11-22)

Start here for anything about proving positions and putting proofs into the dataset. Everything you need is on this
page: the tools, the exact commands, the rules, what each output file is, how to check it, and what is not built yet.
Background on the prover: [research/proof-hunt/ZATAXX_BZ_MODIFICATION.md](../research/proof-hunt/ZATAXX_BZ_MODIFICATION.md).

## 1. The tools (all local, nothing pushed)

| What | Where | Version |
|---|---|---|
| Prover = Zataxx with Botzone rules | `C:/gitProjects/zataxx-bz`, exe `target/x86_64-pc-windows-gnu/release/zataxx.exe` | commit `5bbad56` (on `6e0a8fe` ply-401 rule, `4211e77` own movegen) |
| Hunt tool `ti proof-lines` | hub `C:/gitProjects/titaniumtest`, `src/proofline.rs`, exe `target_main/x86_64-pc-windows-gnu/release/ti.exe` | commit `b3882a2` (merge: `ti data proof-upgrade` a830e6e) |
| Rows from proof lines `ti data games-tdat` | hub `src/gamedata.rs` (`--dedup`, `--dedup-against`, `--exclude-keys`) | same hub build |
| Run script (all never-hunted games) | [`analysis/d1604_proof_hunt_v16.sh`](../analysis/d1604_proof_hunt_v16.sh) | |
| Independent check of the rows | [`analysis/d1598_check_rows.py`](../analysis/d1598_check_rows.py) (analysis only) | |
| Position key of a row (for tests) | [`analysis/d1602_key_of_row.py`](../analysis/d1602_key_of_row.py) | |

Build: Zataxx `cargo build --release --target x86_64-pc-windows-gnu` in `zataxx-bz` (PATH needs `C:/msys64/ucrt64/bin`);
hub `cargo build --release --target x86_64-pc-windows-gnu --target-dir target_main --bin ti` in `titaniumtest`.
Never build the hub while a hunt runs: `ti.exe` is in use (LAWS D13: stop the run, do not build around it).

## 2. Run it

```
ti proof-lines --tgam DIR|a.tgam[,b.tgam] --out STEM [--threads 4] [--max-ms 1000] [--hash-mb 64] [--switch 6] [--first-ms 300] [--confirm 2] [--min-stones 40] [--limit-games N]
```

- Input = game files (`.tgam` + `.tbst`, the `ti gen` format) that start at the standard start position (others are
  skipped and counted: their history is unknown).
- Whole hunt over every never-hunted game: `sh docs/analysis/d1604_proof_hunt_v16.sh` (per batch, output
  `data/proofs/hunt_v16/<run>_<batch>.*`, finished batches skipped on a rerun).
- Stop: `C:/gitProjects/STOP_ALL` (or `STEM.run/STOP`); workers stop within seconds. Resume = rerun the script.
- The tool never overwrites an existing `STEM.tgam`.

## 3. What it does, exactly

1. **Candidates**: every position of every game with >= `--min-stones` stones (first occurrence only). Prior chance of a
   proof from Titanium's own score label (D204 calibration: |cp| > 5000 -> 100%, 1500–5000 92%, 500–1500 56%, < 500 ~28%).
2. **Botzone-exact query**: each position goes to Zataxx as `position startpos moves <the whole game so far>`, so the
   judge's repetition ban and the ply-401 end are applied exactly (a bare FEN makes Zataxx print a WARNING).
3. **Proof** = |score| >= 31,000 (Zataxx mate bound) on 2 consecutive depths (`ZAX_STOP_ON_PROOF`, `ZAX_PROOF_CONFIRM`).
   Zataxx's proofs are accepted as proofs (user D1597).
4. **Dynamic queue** (D1599/D1602/D1603): after every completed depth the running search's expected proofs per ms
   (chance from its score / time of its next depth) is compared with the best waiting position; if that one is
   `--switch` (6) times better, Zataxx is stopped (UAI `stop`) and the position goes back into the queue at its new rank.
   A position being searched is locked (out of the queue); a paused position stays with the worker whose hash table
   holds its work; each position gets at most `--max-ms` in total. Engines are started once and never reset.
5. **Play-out** (no PV is ever stored, LAWS D15): the proven search's own winning line is walked move by move; every
   position on it gets the proven result for its side to move, the mate score counted down, and the line's move as its
   best move. Where the line ends before the game does, the position is proven again and walked on, to the real end.
   Checked on every move: legal (Titanium's generator), not banned (the judge's repetition rule, kept by the tool from
   the real game), and at the end the Botzone final count must give the proven result. Any failure discards the line.
6. **Output**: one game record per line (`STEM.tgam` + `.tbst`, src 202, not in the sequence DAG): a branch from the
   original game at the proven position. Lines that meet are walked through (each search picks its own winning moves).
7. **Conflicts** (D1602): a position proven win by one line and loss by another is disputed: the second line is not
   written, both proofs go to `STEM.conflicts.tsv`, the key to `STEM.conflicts.keys`; it never enters the dataset.

## 4. Outputs and how to check them

| File | Content |
|---|---|
| `STEM.tgam` / `.tbst` | the proof lines as game records (positions + best move + proven score + result) |
| `STEM.report.txt` | counts: candidates, searches, switches, proofs, lines (to the end / cut), positions, every discard reason, conflicts |
| `STEM.conflicts.tsv` / `.keys` | disputed positions with both proofs / their keys |

Turn lines into rows: `ti data games-tdat --tgam STEM.tgam --dedup --exclude-keys STEM.conflicts.keys [--dedup-against train.tdat] --out ROWS`.
Check: `python -I docs/analysis/d1598_check_rows.py ROWS.tdat ORIGINAL_GAME_ROWS.tdat` (result vs proven score,
repeats, agreement with Titanium's own proven labels and with the games' results).

Reference test (20 games of `pc_run16_15k/batch_0001`, 4 workers, D1600–D1603): ~64 s, 513 candidates, ~260 proofs,
~1,280 distinct positions (~975 not in the games), 0 discarded, 0 conflicts, Titanium's 162 shared proofs agree 162/162,
game results agree 272/272.

## 5. Into the dataset (D1611): one script, never by hand

`[BASE=train_v16] [OUT=train_v17] [EXTRA=a.tgam,..] sh docs/analysis/d1611_merge_proofs.sh` (needs `ti.exe` free: stop the
hunt first). It takes every batch with a real `.report.txt` (+ `EXTRA` = proof files of a batch stopped before its report;
the lines already written are complete) and makes NEW files only:

1. `games-tdat` (no dedup) = every proof-line position -> `ti data proof-upgrade --check-only`: positions proven both
   ways across batches are isolated (`check.contradictions.keys`); excluded keys = those + every batch's `conflicts.keys`.
2. `games-tdat --dedup --exclude-keys --dedup-against BASE.tdat` (incl. the held-out tail) = NEW positions only,
   leak + chain checks, `ti data append --before-tail 1528091` -> `OUTpre`.
3. `ti data proof-upgrade --preserve-tail 1528091` -> `OUT_upgrade.tdif`: every copy of a proven position (matched by D4 class, default; `--exact` = exact stones) in the head gets
   outcome 1/0 for its side to move and **proof_kind 6** (= this hunt); cp = the proof's mate score unless it already holds
   a same-sign proof; `cp5k` kept; an OLDER opposite proof is overwritten (latest wins, D1607) and listed in
   `upgrade.old_conflicts.tsv`; the held-out tail is never changed. `ti data patch` applies it (fingerprint-verified) -> `OUT`.
4. `split` -> `OUTh` + `val_v17tail` (must equal the old tail), then `ti data proof-check` (hub) re-checks
   everything with its own key code. Work files: `data/datasets/v17/proofs/` (the `.tdif` + `proof_new.tdat` are also
   the small diff files for Modal / ModelScope, LAWS D1).

Later batches: rerun with `BASE=train_v17 OUT=train_v18` (already merged positions dedup away and only upgrade).
`STANDARD_TRAIN.txt` is changed only by the user's order. Open user question: also replace the best moves in the
original games' `.tbst` at proven positions?

## 6. Known limits

- Proven draws are not detected (only wins and losses).
- The 534k v14 proofs (2026-11-18 hunt) were made WITHOUT the game history (bare FEN): repetition bans were not applied;
  a re-check with history is owed.
- Positions are deduplicated by board only; a position reached with different histories gets the first proof found.
