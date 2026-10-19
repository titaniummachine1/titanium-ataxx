---
description: Checkpoint Titanium session state to HANDOFFS.md, LEDGER.md, BEHAVIOR.md
---

Checkpoint this Titanium Ataxx session so knowledge accumulates and no fresh session starts lobotomized. Live repo state is injected below — use it verbatim, do not re-derive:

Current branch: !`git branch --show-current`
Recent commits: !`git log --oneline -5`
Dirty files: !`git status --short`
Worktrees: !`git worktree list`

Steps (do all of them):

1. Prepend a new `## STATE FOR NEW SESSION (YYYY-MM-DD <topic> — ...)` entry at the TOP of `HANDOFFS.md` (never append state to the bottom, never delete old entries). Include: branch + commit, what shipped, proof (tests/gates/logs), critical caveats (which net on BOTH sides, classical-vs-classical warning), what's NOT built, held decisions.
2. If any experiment ran games: append the LEDGER row via a python script (LEDGER.md is CRLF — the Edit tool fails on it; never edit it directly). Row needs: id, change, vs, gate, games, W-L-D, Elo, verdict.
3. If the user stated or changed a standing rule: update `BEHAVIOR.md` iron laws / mistakes ledger the moment it happens, not at the end. If a law summary changed, mirror it into `AGENTS.md` section 3 so fresh sessions get it with zero lookup.
4. If docs only (no code delta), say so explicitly in the HANDOFFS entry.
5. Verify with `git diff --stat` and report what changed. NEVER commit, merge, or push unless the user explicitly ordered it. MAIN IS UNTOUCHABLE.
