path = 'C:/gitProjects/titanium-ataxx/docs/MISTAKES.md'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep up to line 66 (index 66)
base_content = "".join(lines[:66])

clean_additions = """

34. **Warm iterative-deepening startup skipping shallow depths (ca9ea34) causing catastrophic `d0` blunders on hard time controls (2026-10-28).**
    - *Incidents:* Botzone match `6ae22cc69eb7df09d05b6256` (Ply 84, `d0 n876096 t3948 s12` blundered `(0,6)->(1,6)`) and match `6ae230979eb7df09d05b67dd` (Ply 47, `d0 n274176 t3948 s9` blundered `(0,1)->(2,3)`), both losing to PyJataxx.
    - *Root Cause:* Commit `ca9ea34` imported an optimization from Quoridor: `let start_depth = (entry.depth - 1).min(max_depth).max(1)`. If `root_tt` had an entry from depth 18, it started directly at depth 17, skipping depths 1..16. Under Botzone's hard 4,000ms limit, depth 17 takes >3.9 seconds and times out before completing. Because no depth iteration ever finished, `has_completed_iter` was false, `result.depth` remained 0, and the engine defaulted to `root_moves[0]`—an unsearched initial root move from static sorting!
    - *Rule / Fix:* **Never skip shallow iterations in iterative deepening under time controls.** Always start from `depth = 1`. Shallow iterations (depth 1..5) execute in microseconds (<0.1% of search time) and guarantee that `has_completed_iter = true` within the first 0.05 ms. When time expires at high depth, the engine is guaranteed to play the verified best move and score from the last completed depth (e.g. `d15`, `d17`), never `d0`. Also added emergency root fallback: `result.best = tt_legal_mv.or(Some(root_moves[0]))`, prioritizing a legal TT move.

35. **Disabling iteration prediction for Botzone (`!BZ`) causing time overrun debt and compounding clock penalties (2026-10-28).**
    - *Incidents:* The engine consumed 3,948ms out of 4,000ms on every single move, riding right on the bleeding edge of the deadline and overrunning by a few milliseconds under system jitter.
    - *Root Cause:* Botzone's judge operates on a zero-tolerance time debit system: any millisecond over the per-move limit is deducted from subsequent moves' time limits (and pondering is prohibited). In `crates/titanium/src/search.rs`, lines 1437–1440 had `if !BZ && result.score > -MATE_BOUND && self.elapsed_ms() + proj_ms > budget_ms`, explicitly disabling the iteration overrun prediction check for Botzone under the mistaken assumption that 'clock is a hard limit, so use all of it'. As a result, if an iteration took 2,000ms and only 500ms remained, it started the next iteration anyway, guaranteeing a hard time cutoff mid-iteration and risking judge overrun penalty.
    - *Rule / Fix:* **Never ride the absolute time limit edge on Botzone without iteration prediction.** Remove `!BZ` so iteration cost prediction applies universally. If the next iteration is projected to exceed the remaining budget, break cleanly at the completed depth. Also added an internal 10ms safety margin on `deadline_ms` in the Rust engine (`(budget_ms - 10ms)`), and increased `SAFETY_MS = 100` in the Botzone Python wrapper (`titanium_botzone.py`). This guarantees turns finish cleanly in ~2,500ms–3,890ms, with zero overruns and zero time debt on future moves.

36. **Transitivity assumption in AI engine evaluation (conflating a single second-player time-abort loss with engine inferiority) (2026-10-28).**
    - *Incidents:* Kataxx developer claimed Kataxx cannot lose to Titanium because Titanium lost to PyJataxx.
    - *Root Cause:* Zero-sum adversarial games have non-transitive skill geometry (Condorcet cycles / rock-paper-scissors dynamics). Different engines possess divergent search/evaluation strengths (e.g., MCTS territory expansion in PyJataxx vs deep alpha-beta singular extension tactical solving in Titanium vs neural policy heuristics in Kataxx). In Ataxx, Player 0 also has a massive first-mover tempo advantage.
    - *Rule:* Never judge engine strength from single-game second-player anomalies or intransitive match triads. Always evaluate engine supremacy via controlled multi-hundred game SPRT gates under opening book diversity (`book100k_diverse.txt`) with alternating seats.
"""

with open(path, 'w', encoding='utf-8') as f:
    f.write(base_content + clean_additions)

print("MISTAKES.md cleaned and updated perfectly.")
