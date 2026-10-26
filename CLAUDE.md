@AGENTS.md

# Start of every session (Claude Code)
1. Read the TOP entry of `HANDOFFS.md` (current state, what is running, what is owed). It is the single source of truth
   for "where are we"; older entries below it are history.
2. For any eval / NNUE / teacher / dataset work read `docs/NET_ANALYSIS.md` (what the strong engines actually look at,
   values AND policies, the anchor test, NNUE v2 decisions) and `docs/KATA_FIRST_V3.md` (dataset v3).
3. What to try next: `BACKLOG.md` (CURRENT BACKLOG section, ranked). Training / relabel / Botzone recipes: `docs/TRAINING_RUNBOOK.md`, `docs/RELABEL_RUNBOOK.md`.
4. Laws: `BEHAVIOR.md` (summary in AGENTS.md section 3). Decisions: `docs/DECISIONS.md`. Mistakes: `docs/MISTAKES.md`.
5. NNUE engine code lives on branch `exp/nnue` in the worktree `C:\gitProjects\nnue-ataxx` (training scripts and data
   stay here in `training/` and `data/nnue/`).

# Never
- train, relabel or launch any long job without an explicit user order; never run several competing runs;
- load multi-GB arrays whole or memmap the 13 GB feature file in a long run (this 16 GB box froze once);
- call research "done" without saying what was not analysed.
