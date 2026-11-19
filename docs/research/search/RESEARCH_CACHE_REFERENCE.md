# TT / eval cache: reference engines, what we measured, what we built (2026-11-09, D643-D651)

## 1. What other engines do (web search, 2026-11-09; sources at the end; not verified against current source)
- **Stockfish**: 10-byte TT entries (16-bit key check, 8-bit depth, generation + PV flag + 2-bit bound in one byte, 16-bit move,
  16-bit value, 16-bit static eval), 3 entries per 32-byte cluster (`static_assert(sizeof(Cluster) == 32)`: the cluster size
  divides the cache line so the prefetched line holds it), cluster index by `mul_hi64(key, clusterCount)` (no power-of-two size),
  table allocated with large pages, cleared in parallel threads. **It stores the static eval in the TT entry, so there is no separate
  eval cache** (forum: "if you also save the static eval into the TT you don't need a separate eval hash").
- **Rustic** (Rust): 4-entry buckets (earlier 3); **blunders-engine**: 2-entry buckets (priority slot + general slot); Reckless: README
  only (Hash default 16 MB), layout not found.
- **Ataxx engines** (Moonbird, autaxx, tiktaxx): TT + alpha-beta; no published table layout found.
- **Small vs large hash**: forum evidence is mixed. Micro-Max with a 4 MB table (L2-sized) ran about half the speed of 128 MB; a
  single-position 7-ply test showed no clear win for 128-512 KB tables. Nothing supports "small is reliably faster": it depends on
  engine, time control and what lives in the entry. Measure at the target time control (what we did, section 2).
- Corridor-style "adaptive size" (user): start inside L1/L2/L3 and grow to the full size; not guaranteed better, so gated (section 3).

## 2. What we measured on this PC (i7-4900MQ: L1d 32 KB, L2 256 KB/core, L3 8 MB shared)
Search performance test (docs/analysis/search_perf_test.ps1; positions sampled from the D640 gate games at plies 6..76, same wall time for
every config, ABBA, 4 parallel processes, cold tables; this replaces playing games for speed/cache changes, user D644):
| budget | main nodes | ranked full 96 MB | adaptive 12/13->22 | best small fixed |
|---|---|---|---|---|
| 100 ms | 9,812 | 0.972x | 1.174x (+0.64 depth) | tt18+ec16 1.220x |
| 1000 ms | 96,099 | 0.936x | 0.801x | (sweep running, D651) |
| 3300 ms | 304,791 | 1.119x (+0.14 depth) | 0.989x | (sweep running, D651) |
Reading: the best size depends on the node budget (small wins at ~10k nodes, the full 96 MB wins at ~330k = the Botzone clock). Growth by
doubling (adaptive) loses at 1000/3300 ms: every doubling strands half of the stored entries and the eval cache hit rate (eval = 65% of
node time) drops during the climb. Plan: choose the size from the node budget at the first search of a game (fixed afterwards), Botzone keeps 22 bits.
Also found (D636): the futility small-net bound runs only on an eval-cache MISS, so the cache hit rate changes the pruning and the tree
(2^16 vs bigger caches give a different result in 15 of 48 fixed-node searches).

## 3. What was built (exp/caches on origin/main 0e9efb0; commits ac2a8ce, 4be1d66, 3f71961, 3a0b443; nothing merged or pushed)
- Ranked eval cache: 8 x 8-byte entries per 64-byte line, ranked eviction (empty > dead > unimportant LRU), prefetch of the child's cache line + TT bucket.
  Spare value bits (29-30) carry 2 extra hash-check bits.
- Ranked, bit-packed TT: 64-byte bucket = 6 x 10-byte entries (check 32, score 16, depth 8, empties 6, age 4, move code 10, flag 2, importance 2)
  + the 4 spare bytes as 6 x 5 extra check bits: every bit used, 59 of 64 hash bits filter, +50% entries in the same 64 MB.
  Static 10-bit table of all 529 pseudo-legal moves (49 clone destinations + 480 jump pairs; 8 bits cannot hold 529); 16-bit score code
  (evals clamp to +-30000; mates as 30001 + distance); lookup rejects a different stone count (stones never decrease in Ataxx);
  dead entries (more empties than the search root) are evicted first.
- Runtime L1/L2/L3 detection (cacheinfo.rs, CPUID leaf 4 / 0x8000001D, fallbacks 32 KB / 256 KB / 8 MB), because the Botzone .so is built here but
  runs on the judge's CPU; start sizes come from half the L2 for each table.
- Verification so far: cargo test 69 pass; parity 0 differences vs the 16-byte ranked build (parity58, game12) at 4 and 6 ways; legacy mode identical to main.
- Not yet shown: speed/strength at the played clocks, RSS inside the 512 MB Botzone limit (D607 recipe: 155 MB peak), the game gates.

Sources: Stockfish tt.h / tt.cpp mirrors (cdn.jsdelivr.net react-native-stockfish-android 0.3.0, huggingface mirrors), TalkChess "Ratio of Transposition
Hash to Eval Hash" (p=983324), "Size of Transposition Table Entry" (p=733922), "Help with transposition table" (p=957519), Rustic progress thread (p=887429),
blunders-engine docs.rs, github.com/kz04px/autaxx, tsoj/Moonbird.
