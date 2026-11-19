# Development workflow

How a change gets into `main`. The goal is the Stockfish loop: small, testable patches; every claim backed by a number that a tool produced; nothing merged on a feeling.

## 1. One experiment, one branch, one PR
- Branch `exp/<topic>` cut from the current `main`. One hypothesis per branch. Open an *Experiment* issue first for anything that changes behaviour.
- A branch ends in exactly one of: **merged** (gate passed), **closed** (gate failed, reason recorded in the PR), or **parked** (not enough evidence, reason recorded). Stale open branches are closed on a regular sweep.
- Build and test from the branch **commit**, never from a dirty tree: `titanium-cli version` prints `commit=… dirty=0|1` and the net hashes; gate tools refuse `dirty=1`.

## 2. Every PR carries evidence (see the PR template)
1. **Tests**: `cargo test --workspace --release` (CI runs it on Linux and Windows).
2. **Behaviour signature**: `titanium-cli signature` folds fixed searches over a fixed position set into one checksum. A change that claims to be non-functional leaves it untouched; an intended behaviour change updates the committed value in the same commit and says why (`Signature: <hex>` in the commit message).
3. **Strength**, when search / eval / rules / a net changes: a paired gate against `main` on a balanced book, both seats of every line, at least 100 games, SPRT on pair scores (alpha = beta = 0.05, H0 = 0 Elo, H1 = +20 Elo by default). The result is `PASS`, `H0`, or `CONTINUE (extend)`. Nothing is judged below 100 games; a stopped-early verdict needs the SPRT bound, not a hunch.
4. **Speed**, when search or eval is touched: nodes/s on a fixed position set, and a time-limited gate (nets are judged per node *and* per second).
5. **Nets**: hash every net (`sha256`, first 8 hex digits), register it, check a parity test against the previous net; a downscaled net must keep validation-loss parity before any fine-tune.

## 3. Books
Opening books come from the standard start position (`x5o/7/7/7/7/7/o5x x`, two stones per side) and are searched near-balanced at the gate depth. Tools refuse duplicates, malformed lines and lines from non-standard starts.

## 4. Merge policy
`main` only receives a PR whose CI is green and whose evidence is complete. The maintainer approves every merge by name. Reverting a regression needs the same evidence as the change that caused it.

## 5. Results are records, not prose
Gate runs write a log with exe / net / book hashes in the header and one JSON line per game (every move with the mover's score, depth, nodes). Numbers quoted in issues, PRs or docs point at such a record. A number without a record is a claim, not a result.
