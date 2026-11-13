# Contributing to Titanium

Welcome! Titanium is a pure-Rust Ataxx engine (`crates/titanium`, CLI in `crates/titanium-cli`). `main` is the champion that plays on Botzone, so it only moves when a change has *proven* itself in games.

## Setup
```
git clone https://github.com/titaniummachine1/titanium-ataxx.git
cd titanium-ataxx
cargo build --release -p titanium-cli
cargo test --release -p titanium
```
Speed-sensitive code assumes AVX2/FMA (`x86-64-v3`); build with the repo's default flags.

## Workflow
1. Branch off `origin/main` (`git checkout -b exp/<idea> origin/main`). One idea per branch, one change per binary.
2. Keep defaults bit-identical when adding a knob (check `go depth 9` node counts match `main`).
3. Screen cheaply first: single-thread warm nodes/s (compare *within* one interleaved A/B run, never across runs; the PC throttles) plus a fixed-position regret check.
4. Then game-test against `main` with the same settings on both sides. Botzone rules matter: use `--bz --tt-bits 22` and the Botzone ruleset, at both `100 ms` and `1000 ms` per move. A change is only merged if it clears ~95% confidence on **both**; anything weaker stays on its branch.
5. Record the result (win, loss or parked) as a row in `LEDGER.md`. Do not delete rows.

## Lessons already paid for (please don't repeat)
- A change that looks better on a position-regret screen can still lose badly in games. Example: dropping the policy head at depth 1 gave -1.2 cp regret but **-82 Elo**, because depth-1 futility relies on the policy's top move being searched first.
- Alternating two different policy tables costs ~5% nodes/s (cache thrashing). Keep one table.
- Don't hand-roll scrapers/training scripts in Python; extend the Rust tooling.

## Open ideas (good first targets)
- **Static exchange (2-ply stone swing) as a move-ordering / eval base** with a learned residual on top (policy head, SmallNet, disagreement head). Branches: `exp/tactical-mvv-lva`, `exp/policy-see`.
- **E1**: prune/reduce children by shortfall against the best sibling, not only against alpha. **E2**: reductions based on policy-logit gap instead of a flat move-index threshold. Env-knob screen branch: `exp/e1-e2` (`TI_E1_M`, `TI_E2_SAFE`, `TI_E2_EARLY`, `TI_E2_GDIV`).
- SmallNet without the unused hand-eval skip, then refit.

Please open a PR against `main` with the gate numbers in the description.
