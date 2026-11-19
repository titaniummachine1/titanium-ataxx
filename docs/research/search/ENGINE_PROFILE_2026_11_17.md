# Engine profile 2026-11-17 (D1089-D1091, user: "review the engine, it is AI slop; find inefficiencies")

Build: exp/sn2-fast 2886a39 (= main 0120fd12 + per-node policy cache, node-identical) with `--features prof`
(cycle counters, rdtsc; the default build is unchanged). Bench: `titanium-cli bench --nnue nnue_main.tnn --depth 12`
(start position, 358,519 nodes, 1 thread). CAVEAT: the PC ran the policy retrain on 7 of 8 threads, so the measuring
thread shared a physical core: absolute ns are inflated; the SHARES are the usable part. Re-measure on an idle PC.

## Where node time goes (share of total bench cycles)
| Component | calls/node | cycles/call | share |
|---|---|---|---|
| child big-net eval (futility path) | 0.61 | 15,625 | **47.0%** |
| node static eval, big-net part (eval-cache miss) | 0.21 | 18,182 | **19.3%** |
| movegen + ordering (+ policy) | 0.28 | 4,719 | 6.6% |
| SmallNet child bound | 1.80 | 495 | 4.4% |
| child eval-cache lookup | 2.54 | 208 | 2.6% |
| make (loop children) | 4.55 | 105 | 2.4% |
| child hand eval (SmallNet input) | 1.80 | 156 | 1.4% |
| TT probe | 1.00 | 112 | 0.6% |
| SmallNet node acc | 0.14 | 534 | 0.4% |
| finish_node (TT store) | 0.36 | 218 | 0.4% |
=> big-net evals = 0.82 per node = **66%** of all time; the SmallNet prunes 66% of the 1.80 children/node it checks.

## Inside one big-net eval (~15.6k cycles)
| Part | cycles | share of the eval | share of total |
|---|---|---|---|
| FT row sums, 2 accumulators (~95 rows x 512 B each) | 8,952 | 57% | 36.4% |
| head (sparse L1, L2, out, dis) | 3,166 | 20% | 12.9% |
| feature extraction (both_into) | 2,404 | 15% | 9.8% |
| psqt sums | 1,083 | 7% | 4.4% |

## What the row cost is made of (rowcost probe, same lists, same code)
Real indices (13.4 MB table): 15.9-16.8 ns/row; indices folded into 512 rows (256 KB, L2): 11.6-12.0 ns/row.
=> only ~25% of the row cost is memory; ~75% is the 16 vector load + add pairs per 256-lane row. Consequences:
- partial incremental (D1088): measured NO gain (rows cut 34% but bookkeeping + per-node build eat it);
- i8 FT rows: same instruction count (load + sign-extend), saves only the memory part (D440 measured -9% eval);
- the real levers: FEWER big-net evals per node (tighter SmallNet bound: the leader design is -30..-46% slack,
  D1082), FEWER rows per eval (G1 = 49 rows per accumulator = half of all rows, one per square incl. empty ones),
  NARROWER accumulator (W128 = half the vector work: 2.8 vs 5.1 us full forward, D432) paid back by better inputs,
  and psqt folded into the FT rows (Stockfish's PSQT accumulator columns: removes the 7% separate gather).

## Not measured yet
Idle-PC absolute numbers; per-group extraction cost (G5 likely the largest); the head's L1 sparsity; Botzone CPU.
