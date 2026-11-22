# Modified Zataxx (`zataxx-bz`) for proof hunting — reconstruction record (D1556)

Status 2026-11-22: RECONSTRUCTED AFTER THE FACT. The modification was made 2026-11-18
(sources touched 17:05–19:03, exe rebuilt 19:03) and used for the v13 proof hunt the same
evening, but nothing about it was written down anywhere until this file. Everything below
marked MEASURED comes from the tree diff and the hunt outputs; the "2x faster" framing is
the author's, the mechanism supports it, but no old-vs-new speed A/B exists on record.

## Identity

- Fork of upstream Zataxx (`C:/gitProjects/Zataxx`) at `C:/gitProjects/zataxx-bz` (Botzone fork).
- `main.rs` identical. Differ: `board.rs` (+265/-?), `nn.rs` (22 lines), `search.rs` (+210),
  `tests.rs` (+161). Total: 577 insertions, 81 deletions (`git diff --no-index`).
- Built 18.11 19:03 (`target/x86_64-pc-windows-gnu/release/zataxx.exe`, 2.7 MB).
- Fingerprint at hunt time: fnv `F2765B6B9F6C3595` vs `e6ca628` expected by `proof.rs`
  (printed only, not enforced). MISMATCH UNRESOLVED: proofs below come from this build.

## What was changed and why

1. **Botzone-rules port (`board.rs`).** No passes (classic history only); the game ends the
   moment EITHER side cannot move (`side_stuck`); a pass starts a fresh repetition window;
   `solve_one_empty` endgame solver; `placement_snap`, `bz_banned` repetition helpers.
   Reason: upstream Zataxx plays classic rules; proofs must be found under the exact rules
   the Botzone judge enforces, otherwise they are worthless as labels.
2. **Proof-hunting instrumentation (`search.rs`).** Forced-win subtree counters (`fw_me/fw_opp`),
   terminal-leaf counters (`tw_me/tw_opp`), a `pv_line` accessor, and a rich info line
   (pvlen, pvj, lsj/lsn, fw/tw, rm loss stats, bmjump) that feeds the hunt's deduction
   machinery (`unconfirmed_proof`, prolongation signals).
3. **Proof-throughput early-out (`ZAX_STOP_ON_PROOF`, `ZAX_PROOF_CONFIRM`, default 2).**
   A proven score seen at consecutive depths ends the search at once instead of burning the
   rest of the movetime budget re-proving it. THIS is the "faster at finding proofs"
   mechanism: dead re-proving time converts directly into more positions hunted per hour.
4. **Eval tweak (`nn.rs`, 22 lines) + tests (+161 lines).**

## What it produced (measured, `data/proofs/v13_hunt/` + `v13_apply_head/`)

- 18.11: evalcmp (5k/50k nodes, Ti-vs-Zx corr ~0.82–0.92), 200-position calibration
  (64% proven in 1s; deduction signals at 93–100% precision), queue build, full hunt.
- `results.tsv`: 445,787 entries (431,983 exact + 5,800 deduced + 8,004 WDL updates).
- Applied 20:25 into `train_v14h`: 534,068 new exact W/L proofs, 49,369 propagated to
  unlabelled duplicates, 7,633 deduced, 17,016 kept Titanium labels.
- Those proofs trained the v14muon net (current champion lineage, +44.1 @100 ms / +100 @1000 ms).

## Open items

- fnv mismatch (`F2765...` vs `e6ca628`): either rebuild Zataxx to the expected rev or bless
  the current build and record its fnv as the new expected value. Until then, hunt outputs
  are not comparable across builds.
- Hub `mod proof` is disabled (build fix, D1410 era); reactivation needs the `Searcher::pv`
  forward-port (reference: `exp/proof` commit `c1bc6a2`, search unchanged) + rebuild.
- No old-vs-new Zataxx speed comparison exists; run `proof-hunt --race` if the factor matters.
