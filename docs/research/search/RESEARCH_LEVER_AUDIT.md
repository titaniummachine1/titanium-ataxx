# Lever audit: what has been pushed to its limit and what is still open (state 2026-11-09)

Built from LEDGER.md, DECISIONS.md, BACKLOG.md and KEY_FACTS.md (D-numbers in brackets). "Limit reached" means repeated gates stayed within noise;
"open" means untested or undecided. Nothing here was re-measured in this audit unless stated. Champion: origin/main 285a127.

## 1. Search levers
| lever | status | evidence |
|---|---|---|
| PVS, iterative deepening, TT, killers, history, eval cache | in main | E0-E4, S1 (LEDGER) |
| Null move, LMR, futility / reverse futility, IIR | in main; per-NNUE-bucket knob tables built (exp/tree-shape) | knob candidates all within +-10 Elo at 5k; best "B" +11.0 (D592); candidate A' -0.2 over 6,000 g (D605). Near a local optimum at knob level |
| ProbCut | merged | +19.8 +-25.7 at 100 ms, 666 g, LOS 93.5% (LEDGER MAIN-probcut); the 200 ms leg was marked owed |
| SPSA / Texel tuning | inconclusive, not exhausted | all stages found nothing usable but ran too few iterations (D364, D365) |
| Singular extensions | flat | +3 +-21.7 (D531 literature note); Moonbird has no extensions either |
| Aspiration windows | confounded | tested only inside the S7quality rewrite (LEDGER S7/S10), which regressed for other reasons; a standalone re-test is on the BACKLOG, not done |
| Correction history, countermove history | shelved | "older: shelved/reverted" (BACKLOG); no clean standalone gate found in the records |
| Razoring | not tried | no record |
| Lazy SMP | not applicable | Botzone runs one weak core (LAWS B13) |
| Time management | not studied | the engine uses about 3.3 s of the 4 s Botzone clock by design (KEY_FACTS); no adaptive allocation (instability / easy-move) recorded |
| Endgame solver for late-game labels and play | idea only | D248: exact endgame labels for >= 31-stone rows, "NOT started" |
| Tree shaping per phase | in progress | per-bucket margins, treediff tool (D547-D604) |

## 2. Transposition table and evaluation cache
| item | status | evidence |
|---|---|---|
| Old 64k-slot eval cache | thrashes at the Botzone budget (18.6% hits) | D573 |
| Ranked 8-way eval cache (2^22) | 40.9% hits, -24% network evaluations | D573-D592 |
| Ranked 16-byte TT with dead-entry first replacement | built; stone count never decreases, so entries with fewer stones than the root are dead | D570/D590 |
| Combined effect (same 96 MB) | 1.25x faster at equal nodes, +17.8% played nps at 1000 ms, +40.1 +-50.6 at fixed nodes (LOS 94%) | D594-D604 |
| Time-clock gates | undecided: pooled 1000 ms, 716 g, -9.7 +-24 (P about 21%); 3300 ms leg stopped at 60 g | D609/D610 |
| Not yet tried | root stored in the TT, probing children's TT entries before searching them (ETC), importance from policy logit instead of move index, small L2-sized front cache | BACKLOG item 4 |
Reading: the cache/TT work is not finished; its quality-per-node gain is real but the clock-level result is undecided, so it is not merged.

## 3. Network and data levers
| lever | status | evidence |
|---|---|---|
| Game-result weight in the target | pushed (0.55 best, 0.7 and 0.85 worse) | +23 at 5k (D324) |
| Per-bucket calibration | removed (hurt) | D318 |
| Narrower nets (W128, W192) | W128 wins at 50-300 ms, loses at 1000 ms; W192 pruned +0.9% error, ungated | D255/D265, D431 |
| 8-bit feature table | no Elo | D465 |
| Pattern-input nets under the champion head | rejected at 100 ms | D481/D482 |
| Policy head (ordering) | pushed through four variants, ring tables reverted | D341, D493, D508, D541, D554 |
| Kata soft policy targets | done | D541 |
| Deeper labels in the late middle game | open | opponents see outcomes 12-20 plies earlier (D210) |
| Outcome-grounded self-play loop (Jataxx-style) | open, not tested | outcome-only 3x3 screen showed nothing (D614), but no loop was run |
| Data scaling (25/50/100% of v9), external Botzone games | open | BACKLOG |
| Evaluations per node (0.6 wasted child evals), SmallNet refit | open | D332, D303 |
| Fixed-cost 25-window network (Jataxx layout) | speed measured, quality not yet | section 4 |

## 4. Jataxx-style network: speed before quality (D623)
Measured on this PC with AVX2/FMA, int16 weights, 25 windows of 3x3, real positions, children of one parent evaluated back-to-back like a search:
| hidden | int32 accumulators | int16 accumulators | (plus feature extraction) |
|---|---|---|---|
| 64 | 272 ns | 226 ns | 49 ns |
| 128 | 549 ns | 265 ns | 49 ns |
| 192 | 770 ns | 409 ns | 49 ns |
| 256 | 1,076 ns | 562 ns | 49 ns |
Cold, random positions are about 2.5x slower at hidden 128 (1,357 / 834 ns) because rows come from a 126 MB table. The champion is 5,000 ns per evaluation (D426). So a 25-window net is 8-16x cheaper per evaluation at hidden 128.
Botzone log of the match Titanium vs PyJataxx (6aef4326...): PyJataxx does about 1.14 million child evaluations per second (873 ns each including movegen, make and tree work), flat across the game, 27.7k expansions/s, 42.9 legal children per position on average. The same log shows Titanium at 226 knps (avg depth 19.1, 823k nodes per move), against 110-148 knps single thread on the PC for different positions (D305): Botzone looks about 1.2-1.8x as fast as the PC for Titanium, a rough figure.
Result of the distillation screen (D625/D626, scratch trainer, hidden 128, 12 epochs of plain Adagrad on 8 threads, about 13 minutes, random init by user exception):
target correlation on side-to-move=0 validation rows (v9_clean) 0.681 against the champion's 0.727 measured with the same metric (94%); sibling ranking against the champion's own evals (2,995 parents, 136,824 children): top-1 agreement 46.9%, top-3 77.5%, mean regret about 63 cp, correlation over children 0.883. Training plateaus (epochs 9-12 added 1.6 points of top-1). Earlier pattern nets with 61-69% agreement still lost -147 Elo at 100 ms (D481), so this net is not a drop-in replacement yet; it is globally accurate but weak at ranking close siblings (3x3 windows see radius 1, no jump-range interactions).
Open: whether wider windows (5x5) or a wider hidden layer lift sibling agreement to 65%+, and whether the net is useful as a cheap prefilter / futility bound instead of a replacement.
Original question (kept): does a distilled 25-window net reach the champion's quality? Estimate only (eval share of node time is unmeasured): if eval is about 5 of 7 us per node, replacing it gives roughly 2-3x nodes per second; that is worth about +80..+140 Elo by the node ladder only if quality matches, and the earlier k3s result (65-69% best-move agreement, -147 Elo) shows how quickly quality loss cancels it.

## 4b. Where a node's time goes, and the fast-net engine test (D627/D628, 2026-11-09)
Cycle profile of the champion (scratch copy with rdtsc counters, 8 positions to depth 9, 202,757 nodes, 7.9 us/node, single thread):
| stage | share of node time | per node |
|---|---|---|
| NNUE evaluation | 65.0% | 5,170 ns (1.28 evaluations per node, 4.05 us each) |
| small-net futility bound | 11.7% | 931 ns (2.94 per node; its hand-eval part 323 ns) |
| policy head | 7.3% | 577 ns (it cuts nodes by 21%) |
| move generation + ordering | 3.4% | 267 ns |
| pop_best | 3.1% | 245 ns |
| make child | 2.1% | 168 ns (7.3 per node) |
| TT probe + store | 0.5% | 44 ns |
| glue (recursion, repetition scan, LMR math) | 6.5% | 515 ns |
There is no large search overhead: the evaluation is two thirds of a node. Variant speeds (knps): default 130; no small-net futility 87; no policy 150 (+21% nodes); hand eval only 210.
Fast-net test (D3 25-window net quantised to int16 in the scratch engine; mode 1 = fast net for every evaluation, mode 2 = big net only at PV nodes): 1.8x / 1.7x nodes per second (217k / 203k vs 120k knps), 14-18% more nodes to reach depth 9. In-engine fast-net eval agrees with the champion at correlation 0.877 (noise about 147 cp). Equal-time screening matches at 100 ms (200 games each, scratch engine vs the unmodified champion, 4 parallel processes): mode 1 22-173-5 (about -342 Elo), mode 2 24-172-4 (about -330 Elo). Screening only, not a gate; the scratch engine carries about 2% counter overhead.
Reading: speed is not the bottleneck at this quality level; evaluation precision is. A 1.5-1.8x speed-up does not pay for ~150 cp of eval noise. Using the fast net only off the PV did not help because most search decisions (ordering, futility, null-move and LMR margins) happen at non-PV nodes.

## 5. Where the biggest remaining upside probably is (hypotheses, not results)
1. A cheaper evaluation with equal quality (section 4) multiplies everything else; test distillation fidelity first.
2. Finish the TT/cache work at the played clock (root in TT, ETC), then re-gate at 1000 ms and about 4 s.
3. Outcome-grounded self-play data (what Jataxx's strength most likely rests on) and external Botzone games.
4. Per-phase search constants and endgame handling.
