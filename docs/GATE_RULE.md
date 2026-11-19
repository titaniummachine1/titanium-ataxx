# GATE RULE (canonical, user 2026-11-01; hardened 2026-11-04 D407/D408). Read this, not the old laws.

This page replaces the gate wording in BEHAVIOR laws 2, 3, 4, 5, 10, 17c, 17d, 35, 36 and the CLAUDE.md "Gate" lines
wherever they differ. Those laws stay for history; where they disagree, THIS page wins.

## THE MAIN CHAMPION INVARIANT (LAW B0 Ă˘â‚¬â€ť ZERO AMBIGUITY, user 2026-11-04, D407/D408)
1. **`main` means `origin/main` (and local `main` fast-forwarded to it) Ă˘â‚¬â€ť the latest strongest absolute champion on GitHub.**
   - Current canonical `main`: commit **`e33966f8`** (D822) (worktree `C:/gitProjects/mainpromo-ataxx`), verified at startup by `titanium-cli info` against `data/nnue/BUNDLE.txt`:
     - `bundled_nnue_sha8 = 812C2118` (`data/nnue/nnue_main.tnn`, active by default)
     - `bundled_policy_sha8 = 380F609C` (`data/nnue/policy_main.bin`, `policy head ON`)
     - `bundled_smallnet_sha8 = 8F479ECB` (`data/nnue/fut_small.tsn`, `futility small net ON`)
2. **No stale branches Ă˘â‚¬â€ť mandatory ancestry check before coding or gating:**
   - Note that the folder `C:/gitProjects/titanium-ataxx` may be checked out on an `exp/*` branch, NOT `main`! Never assume a directory's current `HEAD` is `main`.
   - Before writing code or running a gate on any branch, verify:
     `git merge-base --is-ancestor origin/main HEAD`
   - If `origin/main` is NOT an ancestor of `HEAD`, the branch is **STALE** and **FORBIDDEN** for promotion gating until rebased or ported onto `origin/main`.
3. **Idiot-proof automated baseline resolution in `ti sprt` and `ti gate`:**
   - In `ti sprt`, `--baseline` defaults to `"main"`. Passing `--baseline main` (or omitting `--baseline`) automatically resolves `main`/`origin/main`, builds it from a clean worktree into `C:/gitProjects/_gate/exes/titanium-<main_short>.exe`, and verifies `<exe> info` against `data/nnue/BUNDLE.txt` (`812C2118`, `380F609C`, `8F479ECB`, zero warnings).
   - If a non-main `--baseline` is passed to `ti sprt` without `--non-main-ab`, **`ti sprt` REFUSES TO START**.
   - Internal A/B tests between two experimental builds must pass `--non-main-ab` (which stamps the log `NON-MAIN-AB`) and **MUST NEVER** be labeled or reported as `"vs main"`.

## How Stockfish (fishtest) does it, so we copy the mechanics and not a half-memory
- Games are played in PAIRS from a balanced opening book: each opening twice, colours swapped. Results are binned as
  pentanomial pair scores (LL, LD/DL, DD/WL, WD, WW). That is exactly what `gate_summary.py` / `ti sprt` do.
- The test is an SPRT: H0 = gain elo0, H1 = gain elo1, alpha = beta = 0.05. The LLR is accumulated over pairs;
  the bounds are  ln(beta/(1-alpha)) = -2.94  and  ln((1-beta)/alpha) = +2.94.
- LLR >= +2.94 stops the test (H1 accepted, patch passes). LLR <= -2.94 stops it (H0). In between the test is
  simply RUNNING: fishtest never calls a test on a dip, never stops it on a fixed small game count, and only a
  very large cap (hundreds of thousands of games) ends it as inconclusive.
- A passed test is merged; tests are stacked one after another on the new master (we do the same: law 17c).

## Our rule (what the agent does)
1. Every gate reports **P(candidate better than main)** = Phi((score - 0.5) / pse) from the paired pentanomial
   (gate_summary prints it). Minimum 100 paired games before a number is shown.
2. **P >= 95% -> PROMOTION CANDIDATE.** The user names the merge (main is untouchable until told).
3. **P < 95% (whatever the sign of the Elo) -> EXTEND** with more paired games from fresh book lines, pooled with the
   earlier blocks. Same pair seats, `uainewgame`, exe+net hashes logged (laws 6-8 stay).
4. The agent NEVER writes reject, rejected, "H0", "not better", "fails" about an undecided gate. A gate far below 50%
   is reported as a number ("score 41.4%, P(better) 0.0%") and the user decides what to do with it.
5. **Speed-affecting change: the time legs decide.** A 5k node gate is diagnosis for a change that alters nodes per
   second; the verdict comes from 50/100/300/1000 ms legs (every leg extended until P >= 95% or the user stops it).
   An intelligence-only change (speed provably untouched) may be decided on 5k alone.
6. Safety stop only at 3000 pooled games: report "undecided at 3000, P = x" and ask the user. Not a verdict.
7. Machine check: `training/gate_summary.py` prints P and the words PROMOTION CANDIDATE / EXTEND only
   (test: `python training/test_gate_tools.py`). The extension loop is `logs/gates/chain3.sh` (200-game blocks).
8. **Evaluator vs evaluator (user 2026-11-07, D504): disable the futility SmallNet on BOTH sides (`--no-fut-sn`)** so the gate
   measures the evaluator, not the infrastructure fitted to the champion (the SmallNet is fitted to 46FCB5A4 and stays ON for
   any --nnue; D482's -147 for the k3s pattern net had a mismatched SmallNet, D503). Search/policy changes on the SAME
   evaluator keep the SmallNet on (it is part of the engine being tested).

## MANDATORY IN EVERY GATE REPORT (user 2026-11-01, D312; LAWS B12)
The eval second opinion below is not optional and not 'when I remember': every gate result I report (final or partial) is accompanied by `ti insight <games.jsonl>` and `ti analyze <games.jsonl>` output, from the hub only (the old gate_summary.py print is the same report). The reply states the direction in play quality (value-lost z, blunder rate, tilt, sign agreement, turning points) next to the Elo. A sandbox whose hub build predates b6cc577 prints no report: copy the games.jsonl out and run the local hub.

## Eval second opinion (added 2026-11-01, user idea; INFORMATION ONLY)
`gate_summary.py` reads `<log>.games.jsonl` (per-move score, nodes, ms, depth of BOTH engines) and prints, below the pentanomial
numbers: actual score vs the eval-expected score (sigmoid(tilt/400), tilt = mean candidate-view score over plies 10..end-10),
the mid-game tilt in cp with a paired z (which engine the game leaned towards), the value lost per move by each side
(mover score + the opponent's next score), and how often the pair result and the tilt agree in sign. It is NOT part of the
verdict. First look (D247): sign agreement 69-74%; mid-game tilt is near 0 because Ataxx games are decided late, and each
net flatters itself (W128 100 ms: tilt z +2.55 while the result z is +0.7). To become a test statistic it needs a neutral
referee re-score of the positions plus calibration on old games and an A-vs-A validation (type-I error) on the sandbox.

## Early-stop advisory (user idea 2026-11-01)
`gate_summary.py` prints an ADVISORY line, never a verdict and never an automatic rejection:
- DECISIVELY BEHIND when >= 100 pairs, result z <= -1.64 AND eval-tilt z <= -1.64 (stopping early is reasonable; the user decides).
- result and eval agree on the win / result says better but eval does not agree (extend before promoting) when P(better) >= 95%.
Back-test on 32 stored gates (>= 150 pairs): result z <= -2.0 alone triggered 6 gates, 0 false stops, 43% games saved; with the eval
confirmation z <= -1.64 and -1.64: 4 gates, 0 false stops, 51% saved (tiny sample: 4 triggers).

## Engine reliability weighting (user 2026-11-01: trust the engine that wins more / the stronger side, e.g. the 200 Elo Zataxx gap)
The tilt is also reported reliability-weighted: each engine's mid-game evals (plies 20..60) get weight skill^4, skill = 0.25 minus the
Brier error of sigmoid(2*eval/400) against the pair results, fitted CROSS-FIT (odd pairs fit the even half and vice versa) so a
gate never grades itself. The two skills are printed ("the X side's evals deserve more trust"). Test on held-out halves: Moonbird
gates (large gap) sign agreement 0.762 vs 0.743 unweighted (oracle 0.785); normal gates 0.780 either way. Information only.
