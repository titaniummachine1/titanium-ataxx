# Policy plan (user idea 2026-11-02, D319): deep-search root data + Kata policy -> a CPU-cheap policy for move ordering, then UBFM

User (verbatim, 2026-11-02): "make deep search like million nodes and get the move ordering at root and slightly above it etc and keep
dumping it until we got enough to train policy along with kataxx training dataset and its policy evals policy distilled and reshaped to
fit better to CPU based search with absolutely brutal surgeries we should manage our first prototypes with your ideas for ubfm Athenian
style ... if one policy is worth 1000 nodes of titanium then it's more efficient to have policy for that beam of search".
Search improvements continue in parallel (SPSA 1b on the PC, D316).

## What we already know (sources)
- NET_ANALYSIS sec. 22: Ataxx_CPP's raw policy (one pass) agrees with Titanium's 1M-node move .443 vs Titanium@5k .467, @50k .560,
  and in 17.3% of positions it has the deep move while Ti@5k misses it -> complementary, best where the search is shallow (root first
  iterations, every inner node of a short search).
- NET_ANALYSIS sec. 17: Kata's raw policy puts the 200-visit search's best move in its top 3 of ~22 choices 94% of the time; two thirds
  of the policy logit is NOT explained by cheap board features (so a learned head is needed, not a hand scorer).
- Kata data on disk: data/nnue/katago_2026_b12c384_*.npz with policyTargetsNCMove (200-visit search policy, 2-stage: stage 0 = clone
  square or jump origin, stage 1 = jump destination; 21.5M rows in the v3 set; KATA_FIRST_V3 item 6: the 2 GiB policy array OOMs a
  whole-file load -> stream it).
- "One Kata eval = ~1000 Titanium nodes" is the cost of Kata's ResNet on CPU. A policy HEAD on our own NNUE accumulator (already
  computed for the value) is one small extra layer: estimated a few % of a node, not 1000x. That is the "brutal surgery".

## Design
1. Data A (Titanium deep root data): new hub mode (extend `ti relabel`): per position, search to N nodes (200k-1M) and dump, for every
   root move, the subtree node count of the last completed iteration + its score/bound, the best move per ID depth; also the PV
   children (ply 1-2) with their TT best move ("slightly above root"). Soft target = node-share at the root (like visit counts).
   Cost: 1M nodes ~ 8 s single thread at ~120 knps; 100k positions = ~220 core-hours -> Docker 16 vCPU ~14 h (~$16) or the PC overnight
   at 200k nodes. Positions: the hard / turning-point positions from ti analyze first (plan sec. 2 of PLAN_SESSION).
2. Data B (Kata): stream policyTargetsNCMove from the npz shards into a hub policy dataset (Rust reader, no Python), D4-augmented.
3. Head: logits = small layer on the existing accumulator (both perspectives, W256 -> 98 outputs: 49 clone/destination + 49
   jump-origin, Kata's 2-stage factorisation) masked to legal moves; cross-entropy vs soft targets. Value trunk frozen at first.
4. Trainer: the hub needs a policy loss in `ti train` (bullet sparse inputs + softmax over masked outputs). This is the biggest build.
5. Use 1: move ordering + LMR in alpha-beta (gate 5k / 100 ms / 1000 ms). Use 2: UBFM prototype with policy-ordered expansion and
   minimax backup (docs/research/search/UBFM_PLAN.md), gated at 1 s and 4 s.

## What ordering does today and what the policy replaces (search.rs:1007-1033, main c86733b)
EVORD: for every non-TT move it calls `b.make(m)` (to get the child hash) and adds 100 x (cached child eval - static eval) to the
capture/killer/history score; with EVORD_CACHE=1 it only uses child evals already in the ecache (earlier iterations / futility
evals), otherwise adds 0. So uncached moves are ordered by captures/killers/history alone, and every move pays a Board::make
(~4% of instructions, D241). The policy scores every legal move from the parent's accumulator in one pass, before any child is
made or evaluated (user 2026-11-02: "policy preference of each move might help in move ordering before eval even runs for all of
them"). Test: policy logit as the ordering term in place of / on top of the EVORD term; measure nodes-to-depth + 5k/100 ms gates.

## Policy -> better eval? (user question 2026-11-02)
The head scores ALL legal moves of a position in ONE pass from the accumulator the value already computed (no per-child evals; that is
what search does). Two ways it can improve the eval: (a) step 2 of the trunk plan: joint fine-tune value + policy (+ the final-margin /
ownership auxiliary head of D304) on the shared feature transformer, as KataGo does: auxiliary targets often make the shared features
better for value too; gate it, it can also hurt (capacity is W256). (b) indirectly through search: better ordering = deeper effective
search = better labels in the next relabel round. Order: frozen-trunk policy first (zero risk to value strength), then the joint
fine-tune as its own gated experiment.

## Open (not decided, not built)
How Kata targets and Titanium targets are mixed (start: Titanium deep data as the main target, Kata as pre-training); whether the
policy head changes the value trunk (later fine-tune with both losses); exact Ataxx move encoding in our engine vs Kata's.

## Policy vs the extra evals per node (user 2026-11-02: "try the policy stuff to save on the 1.6 evals per node?", D322)
Measured figure is 1.40-1.48 evals/node (D241 UPDATE 3, rdtsc, bench depth 10/12), not 1.6. Before the SmallNet prefilter it was 2.77
(main 1b286f2: 4.3 futility child-eval calls per node, 91% of them ended with the move pruned).
Where the extra evals come from (search.rs:1092-1131, main c86733b): for every move after the first with new_depth > 0 the PARENT runs a
full NNUE eval of the child (leaf_eval_with_dis, line 1124) to compute the futility reduction; the SmallNet prefilter (1098-1122) skips
most of them first. If the reduced child is then pruned (new_depth <= 0, line 1138), that eval bought no node. When the child IS searched,
its own static eval is an ecache hit unless the 65,536-slot ecache (ECACHE_BITS 16) collided.
- A POLICY head alone cannot replace that eval: futility asks a VALUE question per move ("is this child below alpha by a margin?"); a
  policy is a relative preference with no cp scale. It saves evals only indirectly: earlier cutoffs (fewer moves reach line 1124) and
  policy-ranked LMR / late-move pruning.
- What CAN replace it: a per-move VALUE head ("Q head", 98 outputs = estimated child eval per move, from the parent accumulator), distilled
  from our own net. Labels are free: eval every legal child of each training position with F5FA8557 (or the deep-search score). One pass
  replaces the SmallNet prefilter AND most full child evals at line 1124. It can share the layer with the policy head (two output sets).
- Cost estimate (not measured): 512 accumulator inputs x 98 outputs int16 = ~50k MACs = ~3k cycles on AVX2, vs a full eval ~13k cycles
  (D241 split FT+psqt 7,450 + features 3,490 + head 2,220). So ~0.25 of an eval; pays only if it removes > ~0.25 evals/node of the
  ~0.4-0.5 extra. Inexact (changes the tree): gate 5k + time legs on the PC (LAWS B13).
- STEP 0 (cheap, exact, do first): counters on an exp branch: split the 1.40-1.48 into (a) node static evals, (b) futility child evals
  whose move is then pruned, (c) futility child evals whose child is searched, (d) ecache misses by collision. If (d) is large, a bigger
  ecache is a free bit-identical speedup. The counts are clock-free, so they can run on the PC beside SPSA.
## Everything that waits on the policy (user 2026-11-02: "what else is missing besides best-first minimax with policy?", D323)
Status of best-first: UBFM was tried ONLY WITHOUT a policy (D276: eager K=2 0-30, lazy K=99 1-99 at 300 ms). The policy version is untried.
Ideas blocked by the missing policy / per-move head (none tried):
1. Policy move ordering in alpha-beta (sec. "What ordering does today"): scores all moves before any child is made.
2. Policy-ranked LMR / late-move pruning: reduce by policy rank or probability instead of only move index + history (our LMR prunes
   96-99.7% of late moves already, RESEARCH_LMR2, so a better RANKING of which moves are late is where the gain would be).
3. Per-move value (Q) head for futility (D322): replaces the SmallNet prefilter + the parent-side child eval at search.rs:1124.
4. Singular extension from the policy: when the top move's probability is far above the rest, extend it (today a singular test costs a
   reduced search; the policy gives it for free).
5. UBFM with policy: expand only the top-k policy children; value unexpanded children with the Q head (D276's lazy mode used the SmallNet
   LOWER bound, a biased estimate, which is one suspected cause of the 1-99).
6. Hybrid: alpha-beta near the root, policy-guided best-first / PV tunnel in the long tactical middlegame lines (where PyJataxx sees the
   loss 15-20 plies earlier, D210).
7. Root: policy order for the first ID iterations (cold start) and multi-PV diversity for self-play / relabel position generation.
Untried UBFM fixes that do NOT need a policy (D276 list): TT/DAG merge, exploration term, final-move rule (Cohen-Solal 'safe decision'
= most-visited, not best-valued). Cheap to test on exp/ubfm, but expected small until children get better value estimates (item 5).
Other open items that are independent of the policy: hard-position mining (	i mine), final-margin / ownership auxiliary head (BACKLOG
375, D304), training-method sweep (D320), eval-source counters (D322 step 0).
Order: data A/B -> hub policy + Q loss -> head in the engine (knps measured on the PC, LAWS B13) -> items 1, 3 (cheap gates) -> 2, 4 ->
5, 6 (best-first, gated at 1 s and 4 s).
## Teacher stack for the head (user question 2026-11-02: plug in Ataxx_CPP as a quick test, or start the head from Kata/Ataxx_CPP knowledge? D327)
- Ataxx_CPP as a plug-in gate: NEVER gated in games (NET_ANALYSIS sec. 22 is an agreement probe on 300 positions only). Not worth gating: one pass costs ~0.7 ms (~100 Titanium
  nodes), so it could only order the ROOT once per move; the inner-node ordering, where most nodes are, cannot be tested with it.
- Weights cannot be transferred (ResNet -> NNUE has no weight mapping, Net2Net only works inside one architecture). What transfers is KNOWLEDGE through TARGETS (distillation).
- Stack, in training order: (C) Ataxx_CPP raw policy as a label on OUR positions: ~0.7 ms/position, 32M v9 positions = ~6 core-hours (about 1 h on the 6-core PC, or Docker): own
  distribution, cheap, 44% agreement with the 1M-node move. (B) Kata 200-visit policy rows already on disk (21.5M rows, Kata's self-play distribution, needs the move-encoding map and
  the 2-stage factorisation). (A) Titanium deep root data (200k-1M nodes, hard/turning-point positions): the MAIN and FINAL target because Titanium's own deep choice is what ordering
  must predict (Kata and Ataxx_CPP are weaker players, D105). Pre-train on C+B, fine-tune on A; value trunk frozen at first.
- Quick offline metric before any engine work (no games): agreement of the head's top-1 / top-3 with Titanium's 1M-node move on the 300 val positions of root_order_probe
  (baselines: Ataxx_CPP .443, Ti@5k .467, Ti@50k .560) and the nodes-to-depth effect once wired in.
## Warm start by regression + ablation, not a random head (user 2026-11-02: "neurons are neurons ... ablation so we align the architecture ... unless the eval already has most of it ... it needs more than a fresh random net", D328)
Why weights cannot be grafted directly: a Kata/Ataxx_CPP channel is a spatial feature computed by ~12 residual blocks from the raw board; our accumulator is a different basis (sparse linear FT over G1-G6 features, then clipped activations). The two bases differ by a map we can LEARN: regress the teacher's policy onto OUR accumulator. That is the working version of "neurons are neurons".
Step P0 (no engine change, hub Rust, ~1 day): freeze the net (54A6B02B), take N Kata positions that already carry their 200-visit policy (data/nnue/katago_*.npz policyTargetsNCMove; positions converted to Board as KATA_FIRST_V3 did), compute our accumulator (+ L1 hidden) per position, fit by ridge/least squares the teacher log-policy per move (98 outputs, legal-masked), report top-1/top-3 agreement with the teacher's best move and held-out R^2. Controls: (a) cheap board features only (NET_ANALYSIS sec. 17 says 2/3 of the Kata logit is NOT explained by them), (b) random features of the same width. Reading: R^2/top-k near the controls = the trunk does NOT hold policy knowledge, the head must learn it itself (and the trunk may need a joint fine-tune); R^2 high = the trunk already has most of it, the head is a thin readout.
Ablation set (decides the head architecture, all frozen-trunk, same metric): linear on the accumulator only; accumulator + L1 hidden (16/32); factorised from/to logits (Kata 2-stage: origin/clone-destination + jump destination); per-square outputs conditioned on move type. Pick the smallest that reaches the best metric (CPU budget: LAWS B13, ~3k cycles for 98 outputs on 512 inputs).
INIT: the closed-form least-squares solution on the teacher targets (minutes, not random), then gradient fine-tune with the stack of D327 (Ataxx_CPP labels, Kata rows, Titanium deep data last).
Open: running the Ataxx_CPP net inside the hub needs its weights/format (data/engines/ataxx_cpp) or labelling through its exe; Kata rows need no teacher run. Start with Kata.
## COUNTERS MEASURED (D332, plan item G0): where the evals go and how good the ordering is
Build: worktree C:\gitProjects\cnt-ataxx, branch exp/counters (c86733b + atomic counters in search.rs, `bench` with BENCH_BOOK/BENCH_N/BENCH_NODES). Net rwphase9_nocal 46FCB5A4 (`--network`), 300 book positions (every 166th line of data/nnue/book50k_std5k.txt) x 20,000 nodes = 6,000,000 nodes, single thread, depth/time unlimited. Raw output kept in docs/analysis/counters_2026_11_02.txt.
- NN evals per node (ecache misses) = 1.314; 12.1M ecache lookups, 65% miss. Node static eval: computed here 0.47/node, provided by the parent (pre_eval) 0.46/node.
- Futility child evals (search.rs:1124): 9.27M calls = 1.54/node; the SmallNet prefilter skipped another 13.6M candidates (59% of all candidates). Of the 9.27M calls the child is then pruned (new_depth <= 0) in 6.74M (73%) and searched in 2.53M. About 5M of the calls were real NN evals (about 0.84/node), so about 0.6 evals/node (roughly 46% of all NN evals) are spent on children that are pruned anyway. A per-move value (Q) head or a better prefilter targets exactly this.
- EVORD (search.rs:1007-1033): 131.5M children made by `b.make` (55 per node) but only 5.9M (4.5%) had a cached eval when ordering ran; 125.6M (95.5%) got no eval information at all, so they are ordered by captures/killers/history only. This is the gap a policy fills, and the make-per-child loop is the 4% instruction cost.
- Beta-cutoff index (the move that causes the cutoff): NON-PV nodes (1.34M cutoffs): first move 65.5%, second 17.3%, third 8.7%, fourth 3.4%, fifth 2.0%, sixth 1.2%, seventh 0.9%, eighth or later 1.1%. PV nodes (246k cutoffs): first 85.3%, second 6.9%, third 3.5%, rest 4.3%.
READING: strong alpha-beta engines usually cut on the first move 90%+ of the time; here 34.5% of non-PV cutoffs need a later move, so ORDERING HAS REAL ROOM (a perfect first-move rate would also shrink the subtrees searched before each cutoff). The policy head is not speculative: the headroom is measured. NOT measured: how many nodes a perfect order would save, per-depth/phase split, time-to-depth effect.

## P0 RESULT (2026-11-02, D334)
Tool: titaniumtest src/policy.rs, `ti policy --net nnue_main.tnn --games <12 *_5k.log.games.jsonl> --feat none|h|x|hx --max-samples 120000 --epochs E --lr L` (build: CARGO_TARGET_DIR=target_phase). Val top1/top3/CE: uniform 4.8/14.3/3.42, greedy capture 13.2 top1, scalars only 29.6/58.7/2.43, +h 35.7/59.9/2.37, +accumulator 37.3/62.9/2.27, +both 37.4/63.0/2.27. Next: (1) measure the engine's CURRENT ordering on the same positions (exp/counters branch) to know if 37% top-1 is above it; (2) in-engine ordering test (policy logits order moves before make/eval; nodes-to-depth at fixed positions, then 5k/100 ms/1000 ms gates); cost: per-slot dot of 98x256 only on legal moves ~ 25 x 256 MACs/node, measure knps; (3) add Ataxx_CPP / Kata labels; (4) Q head. Labels so far are only 5k best moves from old gate games.


## User ideas 2026-11-02 afternoon (D347): rank-biased and 2-stage policy, cache+policy mix
- E1 recall@K of the labelled move in head top-K; E2 logits only for the first k moves of the pre-sorted list; E3 hidden-layer reranker on the top K, rank-discounted loss, regression init (C14); E4 cache lookup via incremental hash (no b.make) + cached eval converted to logit units.
- Constraint from D343: never DROP tail moves by rank (reductions lost); ordering-only truncation is safe. Cost context: policy total 3-4% nps, so the gain must come from a better top-K head, not from saving tail logits.
- Outside opinion pasted by the user (GPT): cascade + rank-discounted loss + adaptive stage-2 + recall@K + selection-bias warning; treat as hypotheses, nothing measured yet.
