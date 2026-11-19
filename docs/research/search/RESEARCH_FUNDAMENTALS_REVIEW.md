# Fundamentals review of main 3390ff4 (engine fe4fbabe), 2026-11-09 (D671-D673)

NOTE 2026-11-13: this file was emptied by my own mistake on 2026-11-10 (D678) and re-written from the copy read in
that session. Text of the first pass is the original; the second-pass section was re-written from the D676/D677 rows.
Later corrections by other sessions: F3 is already fixed in 8462ecf (D702); the SmallNet 8F479ECB was last fitted in
40875bb (D142) to net 7B25AC9D, NOT to 46FCB5A4 (D702 corrects D503/D677); main has moved to b834737d since.

User order D671: review engine fundamentals / architecture / docs, look for bugs in the TT and eval cache, find untried
improvements and anything that hinders Titanium, clean the BACKLOG.

## Method
- Own full read of `crates/titanium/src/search.rs` in `C:/gitProjects/mainpromo-ataxx` (origin/main 3390ff4).
- FFI entry (`crates/titanium-ffi/src/lib.rs`: `ti_new`, `search_bz_excluding`) read for sizes and the search path.
- One measurement: warm perf test, 189 positions from 24 gate games, 1000 ms, 4 processes
  (`docs/analysis/search_perf_test_bz.ps1`, raw `logs/tree/d672_bz_nmp_perf_1000ms.txt`), test build
  `C:/gitProjects/bznmp-ataxx` (branch exp/bz-nmp from 3390ff4, UNCOMMITTED env switches TI_BZ / TI_BZNMP).
- Perft cross-check against the Autaxx numbers the user pasted (D673).

## NOT analysed (say it out loud)
- First pass skipped nnue.rs, nnfeat.rs, smallnet.rs, policy.rs, ttcodec.rs, Board::make: READ in the second pass
  (D677, end of this file).
- Still NOT read: tuple.rs, sancta.rs, config.rs, cacheinfo.rs, the rest of board.rs.
- FFI time management / watchdog, the Python launcher, the hub.
- No game gate was run. Every "lever" below is a hypothesis with a guessed size until gated.

## Result on the user's question "bugs in TT or eval cache?"
No correctness bug found. All search callers use `probe_e` / `store_e` with the real empties count; the plain
`probe()` / `store()` wrappers (empties 0/255) are used only by tests. Botzone driver creates the engine with 22 bits
(full size). Inefficiencies only (F3, F9 below).

## Findings (code-verified)
- F1. Null move is OFF under Botzone rules: `if null_allowed && !BZ && depth > 2 ...`. Botzone (FFI) plays
  `search_bz_excluding` (BZ = true); every gate, SPSA run, tree study and perf test used `search()` (classic rules, null
  move ON). The played search was never the measured search. MEASURED (D672): at 1000 ms warm, vs "bz rules (Botzone today)":
  bz + null move = nodes 1.007 (CI 0.992..1.022), mean depth -0.14, same best move 90%, score +4.8 cp;
  classic rules = nodes 1.070 (CI 0.992..1.154, the excess is all at ply 66+), depth -0.01, same best move 88%.
  Reading: turning null move on gives NO depth under Botzone rules, so F1 is not a hidden big loss. It also says null
  move as coded buys no depth at all at this clock (fits D558: tries below beta+200 cut 0-3% and cost 5-18% of the tree).
  The two paths still pick a different move in 10-12% of positions, so tuning on one and playing the other is a risk.
- F2. TT / eval cache consistent (see above).
- F3. Futility pre-filter probes the eval cache twice (`ec2_get(&child).is_some()` then `leaf_eval_with_dis(&child)`),
  each hit does a move-to-front write.
- F4. Depth-0 children enter full `negamax` (stop poll, `bz_over`, mate distance, TT probe = one cache line) only to
  return the static eval the parent already holds (`pre_eval`). Leaf shortcut possible; terminal detection must stay.
- F5. Root: every root move is searched with the full window (-INF, -alpha) (no null-window PVS at the root; ProbCut is
  therefore off on each root move's leftmost path), the root position is never stored in the TT, killers are not shifted
  two plies between moves, history decays x3/4 per search.
- F6. A reduced null-window fail-high in a PV node re-searches straight at full window + full depth (no intermediate
  null-window full-depth step).
- F7. Static pruning is fail-LOW only (`static_eval + margin + dis < alpha`, depth <= 4). There is no fail-HIGH static
  pruning (`static_eval - margin >= beta`, "reverse futility"). No DECISIONS / BACKLOG row shows it was tried. Under
  Botzone rules (no null move) nothing cheap cuts nodes far above beta below ProbCut depth 6.
- F8. Null move is tried at every node with depth > 2 regardless of the static eval (D558 data).
- F9. `store_v2` always overwrites the same position whatever the depth; `probe_e` writes the age on every hit.
- F10. Stale comments: TT "16-byte, 4 per line" (now 10-byte x 6), staging described as active (STAGE_MASK = 0),
  poll "every 2048 nodes" (code: every 64), FFI doc "24 bytes per entry".
- F11 (D673). `titanium-cli perft` prints 16 / 256 / 6652 / 165656 / 5313188 / 167553060 / 6426671496: it counts one
  clone per SOURCE. Autaxx / libataxx count a clone once per destination: 16 / 256 / 6460 / 155888 / 4752668 / ...
  The move generator the search uses is the per-destination one and is pinned to the libataxx numbers by unit tests
  (`start_perft_matches_libataxx` to depth 5, `perft_suite_matches_libataxx` 11 positions). Not a bug; the CLI command
  is misleading next to other engines. Autaxx depths 1-9 are the standard published counts; depth 10
  (282,902,395,737,092, hashed perft, 4000 MB, hashfull 81%) cannot be checked here.

## Side fact
Perf test speed: ~102k nodes per 1000 ms per process with 4 processes on this PC (all three configs).

## Ranked levers -> BACKLOG.md top section (2026-11-09)

## Second pass, 2026-11-10 (D676/D677)
- F12 (D676). Move generation speed, ONE thread (branch exp/perft-dest c448878, worktree C:/gitProjects/perftdest-ataxx,
  not merged). Raw perft (bulk count at the last ply): d7 5,023,479,496 in ~2.6 s = 1.8-2.0 bnps. `perft 6 --engine`
  (the search's own path: `legal_moves_dedup_into` + `Board::make`, every position visited): 146,780,808 positions in
  2.63-2.66 s = 55 M/s = 18 ns per position. A search node costs 6-9 us (D305), so movegen + make is about 1-3% of
  node time (ESTIMATE from these two numbers, not a profile). Verdict: fast enough; a 2x faster generator would buy
  about 1% nps. A hashed perft would speed the test only, not the engine: not done. The threaded perft was removed on
  the user's order (each engine thread generates its own moves; the test must not be spread over threads).
- F13 (D677). nnue.rs, nnfeat.rs, ttcodec.rs, smallnet.rs, policy.rs, Board::make read: no correctness bug.
  - nnue.rs: full refresh per eval, int16 feature table, pre-summed G6 rows, sparse f32 first layer. `ft_sum` needs
    width % 128 == 0 (known, D446). The AVX-512 path is only a compiler-widened `ft_sum`, never measured on Botzone
    hardware (AVX-512 legs already owed).
  - nnfeat.rs: bit-sliced, pinned by brute-force reference tests. ttcodec.rs: 10-bit move code, 16-bit score code,
    round-trip tests.
  - smallnet.rs: header comment names net e11827d1 (stale); see the 2026-11-13 note at the top for the real fit.
    SmallNet stays on for any `--nnue` (known, D503).
  - policy.rs: no accumulator input on non-pairwise nets (known, D490).
  - Board: the `pst` field is live (hand eval, search.rs:491), not dead weight.
  - Incremental accumulator not re-proposed (tried, -12% nps, D188).
- No new lever beyond the BACKLOG D672 list. Nothing measured in the second pass except the perft timings. No gate.
