# Engine review: does every part of main do what it was trained / tuned for, and is it worth its cost? (D893)

User order 2026-11-15 23:0x (D893): "review everything, prove that it is efficient, otherwise it is AI slop".
Target: `origin/main` e33966f8 (`C:/gitProjects/mainpromo-ataxx`), net 812C2118, policy 380F609C, SmallNet 8F479ECB.
Earlier review: docs/research/search/RESEARCH_FUNDAMENTALS_REVIEW.md (D671-D677, F1-F13). That review never checked TRAIN/INFERENCE
PARITY of the learned parts; this one does that first, because that is where the SEE mistake hid.

Legend: CODE = read in the source (file:line given); MEASURED = number from a log (path given); OPEN = not checked yet.

## A. Learned components: is each one fed in play what it was trained on?

| # | Component | Trained with | Played with | Verdict |
|---|---|---|---|---|
| A1 | Policy head 380F609C (TPOL5) | `--see 2d8` (SEE table on), init EE96825B, `--max-epochs 10 --patience 1` (D744 script `train_see_2d8_full.ps1`) = capped, not saturated | SEE OFF: the SEE branch needs `depth >= pol_see_min` = 99 (search.rs:2065, 2089; `TI_POLSEE_MIN` default 99, search.rs:1361) = never | MISMATCH (CODE). Capture knowledge partly lives in the SEE table (D744), which play never reads. Fix running: no-SEE retrain from 380F609C minus SEE, to saturation (D892/D893). MEASURED (D895, same val set): as trained games top1 37.44% / Kata CE 1.8348; as played 27.27% / 1.9164 (-10.2 points top1, worse than its no-SEE parent EE96825B 28.25% / 1.8865). Elo of the gap: the retrain's gate measures it. |
| A2 | SmallNet 8F479ECB (futility lower bound) | fitted in 40875bb (D142, 2026-10-28) to net 7B25AC9D | main runs 812C2118, two nets later (46FCB5A4, then 812C2118). Inputs: parent static eval + child hand eval of the CURRENT net | MISMATCH (from D702 + BUNDLE), MEASURED: on 139,697 search children of 812C2118 it is above the big net's child eval 3.00% of the time (designed tau 2%), mean slack 266.4 cp, RMSE 375.8 (logs/snport/d889_snbench_1.log): its "lower bound" is wrong 1.5x as often as designed (each such case can prune a move that should be searched) and loose the rest of the time. Refits never resolved: D70D46DD +5.8 undecided (D834, extend owed); 45DEE55F 100 ms +20.0 LOS 95.00% but 1000 ms -8.7 +-44.5 at 200 g (D711/D713). |
| A3 | Disagreement head inside 812C2118 (dis_w/dis_b over h1||h2) | NOT trained by `ti train run` ("not trained here", titaniumtest/src/train.rs:43, :186); it is the head carried over from 46FCB5A4 while the trunk under it was fine-tuned on train_v10 | reads h1||h2 of the NEW trunk | MISMATCH (CODE). Used by RFP margin (search.rs:1873), ProbCut margin (1911), futility credit (2328), +1 depth when dis > 300 (2331). The export refit that should fix this cost about -130 Elo (D816), cause never found. The refit target is |400 y - cp5k| clipped at 2000 cp, least squares per bucket (titaniumtest/src/export.rs:51-115); whether the ORIGINAL head was fitted to the same target and scale is unknown (if not, the refit changes the scale the search knobs DIS_RFP_NUM 32 / DIS_FUT_NUM 24 / PC_K 32 / "dis > 300" were tuned on, which would explain -130). Measurement plan: dis distribution per empties bucket of 812C2118 vs 46FCB5A4 on the same positions (old probe logs/tree/d689_46FCB5A4_dis_probe.txt: mean 52-64 cp in b12..b04, 193 b01, 583 b00). Size: OPEN. |
| A4 | Big net 812C2118 eval | `ti train run`, v2 = false (no hand-eval skip input) | `eval_with_dis(b, 0)` (search.rs:1566, 2311) | OK by code (v2 false on both sides). Not re-measured here. |
| A5 | SmallNet policy input (exp/snarch-leader, not on main) | `logit_k` WITH SEE (my mistake, D891) | ordering logit WITHOUT SEE | MISMATCH, fixed in code (hub snarch.rs -> `logit_order`), retrain owed. Parity of `logit_order` with the ordering loop: CODE-checked below (A6). |
| A6 | `Policy::logit_order` vs the ordering loop `gen_and_score_into::<_, true, false>` | - | - | PARITY BY CODE: clone = bias[to] + g[9+flips] + g[27+tholes] (+0.0) + td term; jump = ((bias[to+49] + g[flips]) + g[18+tholes]) + g[36+from_holes] (+0.0) + (td+tor)/TQ; same operands, same order (policy.rs:441-497 vs 502-596, search.rs:598-719). PROVEN NUMERICALLY (2026-11-15, before 23:18): `gen_and_score_matches_reference` modes 3/4 (new, snport-ataxx search.rs tests) = the played path (`<true,true,false>` with a SEE-carrying policy) vs `logit_order`, 200 positions, every move, bundled 380F609C + synthetic TPOL5: exact score equality, PASS (log logs/snport/d893_parity_test.log). |
| A7 | The ordering unit test itself | - | - | TEST GAP (CODE): before D893 `gen_and_score_matches_reference` (search.rs:3669-3715) ran the SEE instance whenever the policy had a SEE table, i.e. it tested ONLY the path play never runs (pol_see_min 99) and never the path play runs. Fixed in snport-ataxx (modes 3/4). Main still has the gap. |

## B. Search hot path (negamax, search.rs:1740-2439): live, dead, cost

| # | Item | Where | Status | Evidence / cost | Verdict |
|---|---|---|---|---|---|
| B1 | SEE scoring in ordering | search.rs:569-579, 609-624, 646-652, 707-715 (HAS_SEE=true instance) | DEAD in play (depth >= 99) | compiled as a separate generic instance, never called; the HAS_SEE=false instance still pays `if !self.see.is_empty() { z += 0.0 }` per move (policy.rs:513, 582) because 380F609C carries a SEE table | remove with the no-SEE head (the branch disappears when `see` is empty; delete the code after the gate) |
| B2 | Hand eval of every non-first child | search.rs:2276 `evaluate(&child, None, false)` | LIVE, but only as a SmallNet input: the big net ignores it (v2 false, search.rs:2308-2312) | MEASURED today: 28.5 ns per call (logs/snport/d889_snbench_1.log, e33966f8 code + 812C2118) x 2.94 calls/node = ~84 ns/node, about 1% of a node; the older profile's 323 ns/node (DECISIONS.md line 914) is stale. Small, not the big leak | waste unless the SmallNet needs it: a no-hand SmallNet (or one fed by something already computed) is the fix |
| B3 | sancta (old S1 net) branches | 1852, 2015, 2130, 2149, 2244, 2257 | DEAD (sancta = None, search.rs:1353) | one predictable branch each per node/child | cleanup, bench-parity check |
| B4 | correction history | `corrected` 1574, `finish_node` 2484 | DEAD (corr = None, 1365) | one branch per node | cleanup |
| B5 | old 1-way eval cache path | 1551-1558, 2267-2273, 2320-2321 | DEAD (ECACHE_MODE default 1 = ranked ec2, search.rs:20) | one branch per eval | cleanup |
| B6 | tuple eval in leaf eval | 1565-1567 | DEAD for v1 nets | only on v2 nets | cleanup |
| B7 | EVORD | 2149-2176 | compiled out (EVORD_W = 0, search.rs:399) | none | delete the code |
| B8 | unused TT methods / fields | compiler warnings in the snport build: `Tt::probe`/`store` unused (search.rs:1002), field `empties` never read (794), tuple.rs `maxconv` unused | DEAD | none at run time | delete |
| B9 | history heuristic | ordering 643, 703; updates 2029, 2399, 2426 | LIVE but weak: max 2000 ordering points = 0.02 policy logit (pol_w 100000/logit) | H3 (x10) +16.7 at 100 ms LOS 97.26% (1600 g, D730) but -47.2 at 1000 ms (D731) | OPEN: "history off" never tested; update cost not measured |
| B10 | capture / clone / jump hand ordering terms | 593, 638, 698 | LIVE, 0.005-0.08 logit next to the policy | not measured | OPEN |
| B11 | non-PV re-search after a null-window fail-high | 2385-2395 | LIVE | fix D721 (`is_pv || new_depth < depth`) bench -12.2% nodes, gate +7.8 +-23.4 at 800 g LOS 74% (D723), parked, never extended (the gate rule says extend) | extend that gate |
| B12 | null move (classic rules only) | 1884-1907 | LIVE in classic search, OFF in Botzone search (BZ) | D672: no depth gain at 1000 ms; D558: costs 5-18% of the tree in buckets 7-11 | Botzone plays BZ, so the played engine does not pay it |
| B13 | clock poll every 64 nodes | check_stop 1630 | LIVE | one clock read per 64 nodes | fine |

### B status 2026-11-15 (D894/D896, LAWS A11): B1, B3, B4, B5, B7, B8 DELETED on exp/no-legacy (ee69a25, f840cca),
proven node-for-node identical to main on 58 positions (classic + Botzone) with 85/85 tests. B6 (tuple tables) and the
staging subsystem are next. Merge into main = the user's order.

## B2. Where a node's time goes (last cycle profile; OLDER engine, re-measure on e33966f8 is OPEN)
Source: DECISIONS.md line 914 (rdtsc counters in a scratch copy, 202,757 nodes, 7.9 us/node). NNUE eval 65.0%
(5,170 ns/node, 1.28 evals/node, 4,053 ns/eval); SmallNet futility bound 11.7% (931 ns/node, of which hand eval 323 ns);
policy head 7.3% (577 ns); movegen + ordering 3.4%; pop_best 3.1%; child make 2.1% (7.3 makes/node); TT probe 0.4%;
TT store 0.1%; glue 6.5%. Reading: the big-net eval is the cost; everything that removes a big-net eval (SmallNet prune,
eval cache, fewer evals per node) is worth far more than micro-cleanups of B3-B8.

## C. NOT reviewed yet (OPEN, in this order)
1. Numeric parity test of `logit_order` (A6) and of the SmallNet bound in the engine vs the trainer.
2. nnue.rs eval cost split (feature extraction vs FT sum vs head) on the CURRENT net; incremental FT was measured
   at -12% nps (D188) and "saves ~35% of rows" (D714): re-measure on 812C2118.
3. Root search (search_gen 2544-2838), time management, FFI path.
4. Ordering: pop_best selection cost per node; policy pass cost per node (measured 3.6 us in the SmallNet port, D890:
   that was the extra recompute, not main's ordering pass).
5. TT replacement and eval-cache ranking (F9 of the earlier review).
6. Hub trainers: every trainer input vs the engine call path (policy2 done for SEE; snarch done; `ti train run`
   targets and the dis-head refit (A3) still OPEN).

## Correction 2026-11-16 (D917)
A3 is WRONG as written: the dis head inside 812C2118 WAS refitted to its own trunk by `ti train export` (closed form, |400 y - cp5k| target); proof: same trunk bytes as nn_v10sat_keepdis.tnn (BDE5BE71), different last 3,136 B, and BDE5BE71's tail equals 46FCB5A4's. Open: whether this target suits the search uses; BDE5BE71 never gated with the policy ON.
