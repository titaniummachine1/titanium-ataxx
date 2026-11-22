# Ordering-oracle curve: how much is better move ordering worth? (D1585-D1587, 2026-11-22)

User question (D1585): what is the best policy head that fits alpha-beta; measure the shape of the function: how much more
accurate can ordering get, and at what accuracy do the node savings stop paying for a slower node. Ideas: two-ply view
(own move's change + opponent replies), two-stage head (cheap head on all moves, expensive head on the top 10-15).

## Method
- Code: commit a0a05d8 on exp/search-levers (C:/gitProjects/fpbreak-ataxx, not pushed): `OrdOracle` + `Searcher.ord_oracle`
  (None in play) in crates/titanium/src/search.rs; driver `titanium-cli bench 1` with env ORACLE_BOOK / ORACLE_N /
  ORACLE_DEEP / ORACLE_DEPTH / ORACLE_P (crates/titanium-cli/src/main.rs). Script docs/analysis/d1585_oracle_curve.sh,
  log C:/gitProjects/logs/d1585_oracle_curve.log.
- Pass 1: deep search (depth 11 or 12) records the best move of every non-fail-low node (full hash, deepest wins).
- Pass 2: fresh searcher (cold TT), fixed depth 9 or 10; at a deterministic share p of the nodes that have an oracle move,
  that move is searched first (replacing the TT move). p = 0 also measures today's ordering against the oracle.
- Positions: logs/tree/bz_ref_252.tsv, 126 of them (every 2nd), Botzone games, all phases. Classic rules, bundled net
  7E43F8A1 + policy + SmallNet, single thread, quiet PC.

## Results (126 positions)
| p (share of covered nodes forced) | nodes d9 (vs p0) | time d9 (vs p0) | nodes d10 (vs p0) | time d10 (vs p0) |
|---|---|---|---|---|
| 0     | 6,629,311 (1.000) | 16.11 s (1.000) | 12,862,725 (1.000) | 31.03 s (1.000) |
| 25%   | 0.942 | 0.899 | 0.965 | 0.933 |
| 50%   | 0.929 | 0.855 | 0.928 | 0.871 |
| 75%   | 0.870 | 0.754 | 0.882 | 0.801 |
| 90%   | 0.846 | 0.717 | 0.863 | 0.766 |
| 100%  | 0.823 | 0.675 | 0.840 | 0.719 |
Coverage: an oracle move exists at ~58% of probed nodes (pass 2 visits nodes pass 1 never did).
Time falls faster than nodes (nps 412k -> 502k): better ordering also makes the nodes cheaper (earlier cutoffs, fewer
evaluated children per node).

Today's ordering vs the oracle (p = 0, depth 9 / depth 10):
- TT move present: equals the oracle 80.0% / 79.6% (366k / 720k nodes).
- No TT move (policy + killers + history decide): oracle move ranked top1 69.8 / 70.6%, top2 83.9 / 84.4%, top3 90.1 / 90.5%,
  top5 95.7 / 95.9%, top10 99.28 / 99.34%, top15 99.80 / 99.82%, top20 99.94 / 99.94% (588k / 1.22M nodes).

## Reading
- The curve is close to LINEAR in the share of nodes fixed: perfect ordering at the covered 58% of nodes = -18% nodes and
  -32% time at depth 9 (-16% / -28% at depth 10). About 26 accuracy points are missing at those nodes (~74% today), so
  each +1 point of first-move accuracy is worth about -0.7% nodes / -1.1..1.2% time on the covered nodes; spread over all
  nodes it is plausibly 1.5-2x that (lower bound = the covered-node number). Upper bound of the whole lever if every node
  were perfect: very roughly -30% nodes / -45% time (extrapolation, not measured).
- BREAK-EVEN RULE: a better head pays only if its extra cost per node is below ~1.1% of node time per accuracy point it
  gains (at ~2.4 us per node in this bench, ~25 ns per node per point). A head that gains +5 points top1 can cost about
  +5% node time (~120 ns/node); one that gains +10 points about +11%.
- TWO-STAGE (user) is the right shape: the cheap head already has the oracle move in its top 10 at 99.3% and top 15 at
  99.8% of nodes. An expensive re-ranker only needs to rerank 10-15 moves instead of ~45 (mean moves per position), i.e.
  ~3-4x less cost for ~99% of the possible gain. The expensive stage can also be gated to depth >= 2 where nodes are few.
- VALIDATION of the screen: the fitted-ordering knob that lost -39.3 Elo in the D1581 gate (D1584) shows in this screen
  top1 69.80% = baseline 69.80%, top2-5 0.1-0.2 points worse, nodes +0.6%: no gain. Root top1 on 15k labels had said
  +1.2 points. So: screen ordering changes with this tool (in-search rank stats + nodes/time at fixed depth), never with
  root top1 alone.

## Not checked
- Botzone-rules search (search_bz); only classic search() was measured.
- Coverage-free upper bound (a perfect oracle at every node) and the noise between runs (one run per p; nps varies a few %).
- Elo per time: ~1% time = ~1 Elo at 100 ms is a rule of thumb, not measured here.
