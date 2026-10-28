# Titanium on Botzone (Python driver + Rust engine library)

**Build the upload with one command (from main):** `python tools/build_botzone.py` -> `botzone/upload/` + `MANIFEST.txt`
(pulls main, builds the library with zig for glibc 2.14, checks it, assembles the files, test-plays them in a Linux
container). The GitHub workflow runs the same build on every push to main (artifact `botzone-upload`, no secrets).
Never upload a library built with plain `cargo build` on a modern Linux: it needs glibc 2.34 and crashes the bot on
Botzone (Ubuntu 16.04, glibc 2.23) on its first move (happened with v5, 2026-10-25).
Eval: main's strongest big NNUE is embedded in `data/titanium_v2.so`; the upload no longer needs a separate network file. Local `TI_NET` still allows an explicit replacement network.
Search: the Botzone driver passes the full u8 depth limit (255) along with its time budget; solved mates and exhausted terminal trees stop automatically.
Time: 12 s on the bot's first turn of a match, 4 s on every later turn (persistent session).

Simulator modes: `--mode long` (default, persistent session with warm ID search) and `--mode oneshot`.

Design: **Python owns Botzone, Rust owns the game.** `titanium_botzone.py` parses the long-live Simple-IO
requests, replays the game after a restart, keeps the clock and the watchdog, and talks to the engine only
through a tiny C ABI (`crates/titanium-ffi`): `ti_new / ti_new_game / ti_apply / ti_search_with_limits(time, nodes, depth) / ti_stop / ti_stats`.
The engine knows nothing about the Botzone protocol.

## Time Policy (Default: PERSISTENT SESSION / LONG-RUNNING, 4 s per move)
The bot is configured WITH a long-lived persistent session (`TI_MODE = "long"`):
- Botzone hands the bot 12,000 ms on turn 1 and 4,000 ms on all subsequent turns.
- The Python process stays alive between turns, outputting `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<`.
- The Transposition Table (TT) and engine state are preserved across turns.
- Search employs warm iterative-deepening startup from TT depth + generation aging to recover 25-30 Elo of handicap against cold-reset opponents.
- The one-shot fallback variant remains available with `TI_MODE=oneshot` (6 s per move, cold process reset every move).

## Rules handled in the engine (Botzone judge)
No passes; the game ends the moment either side cannot move and the mobile side absorbs the empties
(`Board::bz_over`, `bz_counts`, `Searcher::search_bz`); 4-corner start (`Board::botzone_start`); a move that
would be rejected by the judge as a repetition is replaced by the allowed move that wins the most discs.
The repetition rule is the judge's exact one (placement snapshots; a move is illegal iff the latest snapshot equals an earlier one and
the move produces the same successor snapshot as then), verified against the reference judge `judge_ref.h` (Ataxx_CPP repo) with
`botzone/judge_check/` : 57,951 legal-move checks (656 judge-rejected repetitions), 5,546 illegal-move checks, 0 mismatches.
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
python botzone/sim_local.py --lib target/release/titanium_ffi.dll --net <tupS57v2.tup> --games 2 --scale 1 --max-driver-moves 3   # true 4 s limit
# add --mode oneshot for the cold-reset variant
```

## Cloud Deployment (Automated via Modal)
Direct Windows socket uploads for ~28.5 MB files (`titanium_v2.so`) often timeout over international ISP routes to Beijing.
Use the cloud container uploader (`tools/upload_botzone.py` via Modal) which completes in seconds over datacenter bandwidth:

```powershell
$env:BOTZONE_EMAIL = "your_email"
$env:BOTZONE_PASSWORD = "your_password"
python -m modal run tools/upload_botzone.py
```

### Botzone API Constraints & Endpoints
1. **Description Length:** `Botzone.consts.bot.desc_length_max = 100`. Descriptions > 100 characters fail with `{"success": false, "message": "bot.desc.length\n"}`. Keep descriptions concise (e.g. `v8: persistent session, warm ID search, 4s/turn`).
2. **Bot Identifiers:**
   - Bot ID: `6ad7e0c3bdad901f323f6516` (`titanium_ataxx_PY6`)
   - Game ID: `5809c7647f65182b044b5e3b` (Ataxx)
   - Compiler: `py36` (Python 3.6+)
3. **Endpoints:**
   - **Login:** `POST https://www.botzone.org.cn/login` (`email`, `password`)
   - **Upload Binary:** `POST https://www.botzone.org.cn/userfile/uploadfile` (`file`, `path`, `filename="titanium_v2.so"`)
   - **Update Bot Mode:** `POST https://www.botzone.org.cn/mybots/modify` (`botid`, `enable_keep_running="true"`, `simpleio="true"`, `opensource="false"`)
   - **Create Bot Version:** `POST https://www.botzone.org.cn/mybots/create` (`id`, `name`, `game`, `extension="py36"`, `simpleio="true"`, `enable_keep_running="true"`, `description`, `source` file)

## Verified facts from Botzone's own wiki (Bot page / Ataxx page)
Default 1 s and 256 MB per run; the FIRST turn of a match gets double time; Python gets a 6x multiplier (so 6 s in oneshot, 4 s in long-running persistent mode; 12 s first turn);
evaluation machines are x86-64 Ubuntu 16.04 with ONE usable core (our engine is single-threaded). Ataxx rules: black moves first, 4-corner start (black (0,0),(6,6); white (6,0),(0,6)), no passes (game ends when
a side cannot move, empties go to the side that can), wipeout wins, game ends at 400 turns, repetition rule as above.
