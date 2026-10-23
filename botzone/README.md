# Titanium on Botzone (Python driver + Rust engine library)

Simulator modes: `--mode oneshot` (default, fresh process per move) and `--mode long`.

Design: **Python owns Botzone, Rust owns the game.** `titanium_botzone.py` parses the long-live Simple-IO
requests, replays the game after a restart, keeps the clock and the watchdog, and talks to the engine only
through a tiny C ABI (`crates/titanium-ffi`): `ti_new / ti_new_game / ti_apply / ti_search(budget_ms) / ti_stop / ti_stats`.
The engine knows nothing about the Botzone protocol.

## Time policy (default: ONE-SHOT, 6 s per move)
The bot is configured WITHOUT a long-lived session: Botzone starts a fresh process for every move and sends the
whole game history each time (turn count, then request/response lines). The driver replays it into the engine
(`ti_apply` validates every move) and searches for `6000 ms - 50 ms - process age - 2 ms`. The process age is read
from the OS (Windows `GetProcessTimes`, Linux `/proc/self/stat`), so interpreter start-up is charged exactly.
The first move also gets 6 s (an older note mentioned a doubled 12 s first move; that is unverified and a wrong
guess means a time-out loss; `TI_FIRST_MS=12000` enables it once confirmed). The long-live variant remains
available with `TI_MODE=long` (12 s first / 4 s later, hash table kept between moves).
Search uses the whole budget (the Botzone variant has no early exit); a watchdog calls `ti_stop` 12 ms after the budget.
Measured at the true limit (6 s): every move finished at 5.912 s from process spawn to exit, margin 87.5 ms;
engine overshoot past its own budget median 1.0 ms, max 1.2 ms. At a scaled clock: 105 moves, worst margin 39 ms.
Trade-off: no hash table carry-over between moves (fresh process), in exchange for 6 s instead of 4 s per move.

## Rules handled in the engine (Botzone judge)
No passes; the game ends the moment either side cannot move and the mobile side absorbs the empties
(`Board::bz_over`, `bz_counts`, `Searcher::search_bz`); 4-corner start (`Board::botzone_start`); a move that
would repeat an earlier position is replaced by the non-repeating move that wins the most discs.
The normal (non-Botzone) search is bit-identical to main (same node counts).

## Build
```
cargo build --release -p titanium-ffi                                       # Windows DLL (local tests)
cargo zigbuild --release -p titanium-ffi --target x86_64-unknown-linux-gnu.2.14   # Linux .so for Botzone
```
zig 0.13 is at `C:/gitProjects/toolchains/zig-0.13.0/zig-windows-x86_64-0.13.0` (put it on PATH). The `.so` needs glibc <= 2.14.
A build for AVX2/BMI2 CPUs was measured and is NOT faster (1.14 vs 1.14 Mnps), so there is a single library.

## Local test (Botzone simulator: judge rules, greedy opponent, long-live message format)
```
python botzone/sim_local.py --lib target/release/titanium_ffi.dll --net <tupS57v2.tup> --games 6 --scale 0.05
python botzone/sim_local.py --lib target/release/titanium_ffi.dll --net <tupS57v2.tup> --games 2 --scale 1 --max-driver-moves 3   # true 6 s limit
# add --mode long for the long-live variant
```

## Files to upload (new names, so the currently ranked bot keeps working)
`data/titanium_v2.so` -> storage `/titanium_v2.so`; `data/nnue/tupS57v2.tup` -> storage `/nnue/tupS57v2.tup`;
`titanium_botzone.py` as the new bot version (Python, Simple IO, **long-live OFF**, time limit 6 s). Memory: TT 22 bits is ~100 MB.

## Not verified here
The Linux `.so` was not executed (no WSL distro / Docker engine on the dev box); it is the same source as the
Windows DLL that passed every test, checked to be an x86-64 ELF exporting all functions with glibc <= 2.14.
