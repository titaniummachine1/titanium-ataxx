# Code audit of main 0e9efb0 (2026-11-09, D634; read-only, nothing gated, nothing changed)

Scope: engine crate `crates/titanium` (7,173 lines: search 1,841, board 1,703, nnfeat 864, nnue 762, tuple 659, sancta 347,
policy 269, smallnet 232, config 213), CLI 2,365, FFI 404, wasm 189, plus repo layout. Read: search.rs TT / eval cache / negamax in full,
config.rs build guard, FFI/driver cache settings. NOT read in depth: board.rs move generation (perft tests exist), nnue.rs inference
internals, Botzone driver time management, the hub (titaniumtest). `cargo test` was not re-run on main in this turn (65/65 passed
on the branch that became main, D607/D632).

## Verdict
No rot in the core. The search is small and readable; shipped defaults are protected (build-knob guard `config.rs` + `titanium-cli info`
vs BUNDLE.txt). The weak foundations are the CACHES, not the tree search. Nothing found that makes the shipped engine unsafe.

## Findings, ranked by strength relevance
1. **Eval cache is a 64k-entry direct-mapped always-replace table (1 MB, `ECACHE_BITS 16`)** while Botzone searches ~536k nodes/move.
   Measured hit rate 18.6% (ranked 8-way 2^22: 40.9%, D573-D592). The fix exists on `exp/tree-shape` (ranked TT 16 B x4 per line +
   ranked eval cache, same 96 MB): +17.8..22% played knps, 1.25x faster at equal nodes, fixed-node 50k +40.1 (LOS 94%, D604). Time gates:
   1000 ms pooled 716 g -9.7 [-33.6,+14.1] P(better)=21% (D609), deep 3300 ms leg only 60 g (+5.8 +-77, D610, stopped for the user).
   UNDECIDED, not rejected. Odd point to explain: +22% knps but median depth equal (14 vs 14) at 1000 ms. Next = deep leg (Botzone clock),
   rebased on 0e9efb0.
2. **TT (`Tt`, search.rs:462-544): 24 B entries (4 B padding), 1 slot per index, no buckets.** Replacement lets a SAME-position store overwrite a
   deeper entry with a shallower one (`same_pos` is unconditional). ProbCut and null-move re-search the same hash at lower depth, so a deep
   LOWER/UPPER bound can be replaced by a shallow one. Candidate lever (cheap, gateable at fixed nodes + time legs): keep the deeper
   non-exact entry when `e.depth > depth + margin`. No static eval is stored in the TT (it relies on the eval cache). No prefetch anywhere,
   but TT probe+store is only 0.5% of node time, so prefetch is NOT a lever.
3. **Node time (cycle profile, D627/D628, 7.9 us/node)**: NNUE 65.0% (1.28 evals/node x 4.05 us), small-net futility bound 11.7% (its hand-eval
   part 323 ns = 4%), policy head 7.3% (cuts nodes 21%), movegen+ordering 3.4%, pop_best 3.1%, make 2.1%, TT 0.5%, glue 6.5%. No slug:
   to go faster the evaluation must get cheaper (W192+i8 est. -15% node time, ungated) or be called less.
4. **Compile-time / runtime env knobs**: `FUT_CHEAP_M, FUT_SN_M, EVORD_*` read `option_env!` at build time; `TI_POLW, TI_POLMIN, TI_POLICY`
   are read from the process environment at runtime. Risk is contained: `config.rs` checks `build_knobs() == DEFAULT_BUILD_KNOBS` and a test
   asserts it, so a polluted build shell is caught by `info`/tests. A stray runtime `TI_POLW` on the judge is improbable. Low priority.
5. **Legacy scaffolding still compiled into the hot function**: `sancta.rs` (S1Net, 347 l), `tuple.rs` (659 l) + 3 tracked `.tup` files (2.2 MB, not
   embedded), CorrHist (off), `EVORD_*` (compiled out), `match (self.nnue, self.sancta)` branches and per-child `update_child` in negamax,
   2 `#[allow(dead_code)]`. The classical `evaluate()` is still live (small-net bound uses it). Safe cleanup = an `exp/prune-legacy` branch that must be
   bit-identical (same nodes/bestmoves on a fixed suite); no strength effect expected, small readability + tiny branch savings.
6. **Repo layout**: `training/` holds 15 legacy Python scripts (conflicts with law A7 in spirit, unused now), `web/` + `titanium-wasm` look
   abandoned, `data/nnue/tup*.tup` unused by the shipped path. Stale comments: Tt doc "depth-0 EXACT entries double as the eval cache" and TtEntry
   `mv ... 0 = none (eval cache entries)` (the eval cache is a separate array); orphan doc comment "Move scoring is fixed" above TtEntry.
7. **Open (from earlier gates)**: old exp/* branches are behind 0e9efb0 (B0); Botzone package still from 285a127.

## Jataxx policy network: there is none
v37 source (jataxx-study/versions/v37_699cdaad.cpp): no policy / prior / softmax in the search. One value net (25 3x3 windows, 128 hidden, tanh)
scores every child (`move.heuristic`, lines ~781-823); the tree always expands by that child value (best-first), and the move played is
`sort by heuristic + log(games+3)` (lines 729-734). Softmax over `heuristic + log(games+3)` existed only in the v0/v1 self-play trainer. So Titanium's
policy head (7.3% of node time, -21% nodes) is something Jataxx lacks; nothing to borrow there. Its edge is a cheap ~1.14M child evals/s
(Botzone log, D612) plus a tree that never discards information (no TT but 12M-node pool, tree reuse).

## What I propose (not started, none changes main without a gate at P >= 95%)
A. Deep-leg gate of ranked caches on top of 0e9efb0 (Botzone clock), pooled with the 716 g already logged only if the code is the same.
B. TT same-position depth guard (item 2) at fixed nodes, then 100/1000 ms legs.
C. `exp/prune-legacy` bit-identical cleanup (item 5/6), after A and B so they do not conflict.
