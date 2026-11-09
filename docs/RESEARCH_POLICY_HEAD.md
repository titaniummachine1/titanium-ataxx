# Policy head research (2026-11-02, D334). Written before the build so compaction cannot lose it.

User orders (2026-11-02): policy head "probably just like the disagreement head"; learn from the best chess engines, online sources, KataAtaxx,
Ataxx_CPP; distill the head from Kata/Ataxx_CPP for a start better than random; NEVER start from random weights (LAWS C14); if Titanium must
diverge from pure alpha-beta to use a policy head for move ordering, do it. Only goal: Elo on one weak CPU core (LAWS B13).

## What is known (web + earlier sessions; verify before quoting)
- No mainstream alpha-beta engine ships an NNUE policy head for move ordering (Stockfish/Berserk/Viridithas order with history/killers/captures).
  A policy output only pays when it is nearly free: it must reuse the accumulator/hidden activations already computed for the position.
- Monty / bullet (chess, MCTS side): policy head = small linear layers on the same trunk, legal-move masked softmax, trained with CE against
  search visit distributions. Same recipe fits here (98 outputs, no 1858-style move table needed).
- Lc0: policy over a fixed move table, softmax restricted to legal moves, CE vs MCTS visit counts.
- KataGo: policy target = MCTS visits (surprise-weighted sampling), auxiliary heads (opp. policy, ownership, score). Our Kata rows are on disk
  (21.5M, policyTargetsNCMove, 2-stage encoding, docs/POLICY_PLAN.md data B). Weights cannot be copied ResNet -> NNUE; distillation is through targets.
- Athenan / UBFM papers (arXiv 2012.10700, 2505.04525): best-first minimax with a learned value; our lazy UBFM lost 1-99 (D276), best-first
  needs a policy to prune breadth first.
- Measured here (D332): first move causes the cutoff in only 65.5% of non-PV cutoffs (85.3% PV); EVORD informs 4.5% of children; 55 b.make per node.
  So ordering headroom is real.

## Design of the fit tool `ti policy` (hub, Rust, no Python; D289)
- Labels (free, step P0): positions + played move from existing gate games.jsonl (the move is the engine's 5k-node best move). Replay via
  games::parse_game / replay (board before each move). Split train/val BY GAME PAIR to avoid leakage.
- Logit factorisation: logit(m) = A[to] + (clone ? clone_bias : B[from] + jump_bias), 98-ish outputs, legal-masked softmax CE.
  Features: frozen net activations h1 || h2 (and optionally xin, the accumulator output) of the main net 46FCB5A4 (add `xin` to train::Fwd).
- Init (LAWS C14, no random): closed-form ridge regression of the head onto the frozen features, then CPU SGD on softmax CE.
- Metrics: top-1 / top-3 / CE vs baselines (uniform over legal, max-capture, current EVORD order). Sample cap to protect the 16 GB box.
- Next steps after P0: Ataxx_CPP labels (0.7 ms/pass) -> Kata 200-visit rows -> Titanium deep root data; engine integration replacing EVORD
  (measure nodes-to-depth with the exp/counters branch, then 5k / 100 ms / 1000 ms gates); Q head (per-move value) to replace the ~0.6 wasted
  futility child evals per node; policy LMR / singular; UBFM with policy.
- Cost rule: head must be measured in knps single thread on the PC before any gate.

## NOT checked
Exact claims about which engines ship policy heads come from memory of the literature, not a fresh survey. Re-verify before quoting externally.


## Added 2026-11-02 afternoon (user questions answered in chat; D347, D353)
- Why a policy head helps here when it is rare in other alpha-beta engines (my reasoning, partly data-backed; other games NOT checked): the old ordering (captures/killers/history) was weak: the first move caused the cutoff in only 65.5% of non-PV cutoffs (D332); killers/history are built for chess where quiet moves repeat across sibling nodes, in Ataxx nearly every move changes the board a lot. Most of the head's accuracy comes from cheap scalar features (capture flips, holes left, to-square: top-1 29.6%) and only +7.7 points from the accumulator (37.3%) (D334). It is cheap (sparse dot over ~36 inputs per move, one accumulator pass per node: 3-4% nps, D343) while a node is expensive (~1.3 big-net evals per node), so fewer nodes pays. Limits: more data did not help (p1 vs p0, D342), the linear head is capacity-limited; rank-based late-move reduction by the head LOST (-28..-80, D343).
- Caveat on the +71: it compares with the same exe without the head, which still had the EVORD child-eval ordering ON. "EVORD off, no head" was not measured separately (owed, queue it).
- The 0.6 wasted evals per node (futility child evals later pruned, D332) are NOT saved by the policy head: the head reorders, it does not decide which children get a big-net eval. What policy + EVORD-off saved is the board copy of every child for the cache lookup (55 b.make per node, 4.5% cache hits): depth 10 x 100 positions 73.2 s vs 81.2 s (EVORD on) vs 93.6 s (base). The evals/node recount on the policy engine is owed (rerun exp/counters on the policy build).
- User idea (D347): eval cache where available + policy for the rest; rank-biased / 2-stage policy; E1-E4 at the end of docs/POLICY_PLAN.md.
- Next ordering levers, ranked (BACKLOG UPDATE 12): re-measure ordering counters on the policy engine; ablate killers/history with the head on; countermove/continuation history only if headroom remains (earlier countermove and corrhist were shelved/reverted); wasted futility child evals (Q head / SmallNet filter); head refit on deeper-search labels; SPSA re-tune on the policy engine (running, D344).
