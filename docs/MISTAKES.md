# Mistakes — consolidated, read before acting (companions: BEHAVIOR.md ledger, LEDGER.md)

## Training discipline (user-ordered, 2026-10-22)
1. **Training without an explicit order.** Multiple runs launched (stage2-KataGo ep1,
   final V1/V2, autotune pre-WDL trials) without the user saying "train". Now law 14:
   training starts ONLY on audit-ok + explicit "train". Finishing dataset prep never
   authorizes training.
2. **Claiming READY before the spec is complete.** "FINAL DATASET READY" was written
   while proven/forced column, Ataxx W/D/L rebuild, and 179k gap were still missing;
   user corrected to not-ready. Claim = bytes + audit + spec checklist, never a vibe.
3. **No-ckpt long runs.** ~92 min wall lost (final V1/V2 died with zero checkpoints;
   restarts for code fixes). Fix shipped: per-chunk resume manifests + atomic writes
   (`train_stream.py --resume`); a kill may cost at most one chunk. Proof owed: kill test.
4. **Orphan writes (S22).** Timed-out command left the CHILD exe alive; liveness check
   used the wrong process name; 8 orphans + resume workers double-wrote outputs
   (332k vs 250k rows). Rules: exact binary name, kill orphans BEFORE resume, fresh
   outputs + count-verify; corruption signal = output rows != input rows.

## Strength measurement
5. **Val loss crowns.** S4v3/r1/r1dag: val improved every run while gates collapsed
   (down to 3–101). Val NEVER crowns — only paired-book games do.
6. **Single-number bench wins.** Bench d8 jitters 4.6–8M. Speed truth = alternating
   `sperft` pairs; fixed-node determinism is the tripwire (same line + seat must
   replay bit-identically or the run is VOID).
7. **Gates without `uainewgame`.** Opp carried killers/history across games; same seat
   diverged at move 4. Every gate game gets `uainewgame` both sides or the gate is VOID.
8. **Dirty-tree gates.** Both sides must be git refs (branch/commit), never a dirty
   worktree; record exe sha + commit + net hash both sides.
9. **Rejecting on a dip / merging early.** Straddling CI = EXTEND both ways, never merge
   or reject on the bound. Only the user calls merges and rejections.
10. **Out-of-column poison.** Training on unproven-semantics columns cost −400 in one run.
    Never train on unverified columns.

## Engine / environment
11. **Stripping net blobs to fit a loader.** Never cut/convert a net; ENGINE ADAPTS TO NET.
    Load-time allzero skips (W4 `has_q2`) are NOT damage.
12. **Aspiration without cause / stability-assuming mechanisms.** Scores swing 800cp/ply;
    countermove/correction-history/singular/aspiration all conclusively rejected.
13. **PowerShell traps:** no `head`/`sed`/`&&`; quote paths with spaces; `taskkill` orphans
    by EXACT binary name; `schtasks /tr` 261-char limit; STOP file =
    `C:\gitProjects\logs\STOP`.
14. **Edit traps:** edits preserve mtime → `os.utime` after every edit + verify `git diff`;
    stale `training/__pycache__` after mtime-preserving edits → delete it;
    `LEDGER.md` is CRLF → append via python, never Edit (dup `Wenginewdl` row is cosmetic).

## Session 2026-10-24

15. **RAM crash.** build_v3_targets held the 3.7 GB npz + temporaries in RAM -> Windows swap freeze, PC crashed. Fix: stream columns, RAM guard. Rule: check RAM before big jobs.
16. **Launching jobs without an order / spiralling.** Moonbird relabel started after 'no labeling'; Kata-only, two-stage and v3full runs started and killed within an hour; a mock-epoch run read as 'training'. Rule: one agreed plan, explicit go, then leave it.
17. **Overclaiming research.** Called the competitor analysis 'done' with no policy analysis and an untested anchor. Rule: state what was NOT analysed.
18. **Broken estimator presented as data.** Per-row lambda (ow_row) used the teacher as the truth to estimate result noise -> any teacher looks perfect. Rule: sanity-check estimators against a known case.
19. **memmap working-set creep.** PBT memmapped the 13 GB feature file; working set reached 10.7 GB (0.8 GB available). Fix: file reads per chunk.
20. **npz re-read per access.** `z['occ0'][j]` in a loop re-reads the array each time (10+ minutes). Load once.
21. **Wrong Moonbird version analysed first.** Local source = newer unreleased 25.7 MB version; the binary we play is 1.1.0 (2.58 MB). Always verify the engine binary matches the source (exact eval reproduction).
22. **No trainer-vs-engine parity test for the tuple net (S53-S57).** The engine's TUP7 loader skipped capn/conv and misread `pa`, the trainer's zero constraint covered only block 0, and env was defined differently: the engine played a function 66 cp (mean) away from the one trained. Rust tests only compared forward vs forward_fast. Rule: every new net format gets a family-by-family trainer==engine parity test (training/tuple_parity.py) before it ships.
23. **memmap in the NNUE training loop again (2026-10-24).** train_nnue.py batched through np.memmap of the 13 GB index (the PBT path used compact(), the plain run did not); available RAM fell ~0.5 GB per chunk. Fixed: each chunk is compacted with file reads (batches byte-identical).
