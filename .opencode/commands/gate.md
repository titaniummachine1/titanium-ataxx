---
description: Verify a Titanium gate (SPRT + seats + paired book + determinism)
---

Verify Titanium gate `$ARGUMENTS` (a tag substring, e.g. branch or net tag). The gate law, never to be forgotten:

**WHICH gate proves it (law 6):** intelligence-only change (speed provably untouched: identical sperft nodes) → the 5k-node gate CAN merge alone (fairest, most precise, zero clock noise). Speed affected (any NPS change) → BOTH time gates (100ms AND 200ms) must win; 5k is then the quality diagnosis only.

**SPRT always (law 8):** LLR H0 elo≤0 vs H1 elo≥20, PASS at ≥ +2.94. AHEAD but insufficient → EXTEND (more games, fresh book), never merge early. BEHIND → EXTEND too (FOMO: buried gems reject at our peril). Reject ONLY on conclusive CI below 0.5 AND user confirmation. User calls every merge AND every rejection. An inconclusive gate is an order to spend more gate, never a verdict.

**Paired book + seats (law 11):** every gate game runs on the prepared book (`data/nnue/book50k.txt`); each line played from BOTH seats by BOTH engines. Win-both-seats = real edge (even 1 point is signal, especially fixed-nodes). Split = parity. Every leg ≥100 games (law 10), exactly 50/50 black/white per side verified FROM THE LOG.

**Determinism tripwire (laws 11–12):** fixed nodes replay bit-identically — same line + same seats must repeat exactly. Any divergence = INFRA BUG: stop, fix, re-gate. `uainewgame` before EVERY game both sides (fresh searcher in-process, `uainewgame` to UAI children) — a run without per-game resets is VOID.

Steps, in order:

1. Binaries + nets + refs: record exe hash + COMMIT hash + `.tup` hash on BOTH sides (first question on any strength task). Both sides must be git refs (branch/commit, never dirty) — rebuild from the ref, re-hash, stop if it differs. Classical-vs-classical (≈-500 Elo vs net-main) NEVER proves strength.
2. Book + seats: confirm book lines used, both seats per line, 50/50 per side counted from the log — never assumed.
3. Determinism: spot-check repeated line+seat pairs replay identically; if not, VOID the run and fix the harness.
4. SPRT: recompute LLR (and Elo + CI); report pooled W-L-D, LLR, verdict verbatim — never upgrade. AHEAD-insufficient → extend; BEHIND → extend (FOMO); conclusive → propose merge/reject FOR THE USER TO CALL.
5. Record: LEDGER row for EVERY log (any harness), HANDOFFS state. Do NOT merge — merge needs tripwire + explicit user approval + LEDGER promotion row.
