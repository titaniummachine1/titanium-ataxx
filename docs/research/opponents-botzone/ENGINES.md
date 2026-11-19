# ENGINES: every Ataxx engine we have or met (D970/D971, 2026-11-16)

Speeds are SINGLE THREAD. "vs Titanium" = the Titanium build of that date, not today's main (0120fd12) unless said.
Clones (read-only research copies, shallow): `C:/gitProjects/engines-src/<owner>__<repo>` (51 repos: 22 on 2026-11-16 + 29 on 2026-11-17, see D1008-D1015). Older local
sources: `C:/gitProjects/{Zataxx, zataxx-bz, sanctaphraxx, moonbird-src, autaxx, tiktaxx, jataxx-study}`, binaries in
`C:/gitProjects/engines/`. Feature scan of all sources: `logs/engines_feature_scan.json`.

## 0. The five things to take away
1. **gataxx is the best-documented HCE engine there is** (Java, MIT, 95 KB tuning log with Elo for every change): hand-made features +
   a trained 2x2-pair table; beats Moonbird +151 (100 ms, 1 thread, 1200 g), Funes +393, TikTaxx +418; 4.2-4.5 M nps (13.4 M without its table).
2. **The HCE terms that win in play** (gataxx, match-tuned, 400-800 g each): edge stones **+229**, safe stones (no empty neighbour)
   **+89..+101**, NEGATIVE cohesion (penalise own-own adjacency) **+97**, "reach" (stones no enemy can hit next move = our
   0-weakness stones) **-76 when removed**; mobility, territory, corner (beyond edge), second ring, threat counts: nothing.
   These are the same things our net weights most (docs/research/hce-patterns-speed/RESEARCH_HCE_FIT.md: weaknesses, edges, 3x3 fill).
3. **Learned local tables work when blended and made cheap**: gataxx's square-class x own/enemy-neighbour table (= our "dynamic PST")
   distilled from Funes scores = +71..+81 at fixed nodes blended 25-50%, but -57 on the clock until made fast; the 2x2-pair table that
   replaced it (80 placements, 46 M self-play positions) ~+150, plus a byte-lane sum (+35 Elo, identical values). Moonbird's tuned
   eval is the same family: 2x2-block pair tables + per-square (own, enemy, blocker neighbour counts) table + mobility per phase.
4. **Ataxx-specific search tricks with measured Elo** (gataxx): quiescence on moves converting >= 3 stones; cutting dominated jumps
   (jump to a square a clone also reaches) in quiescence +124 / +88, in the main search only when the jump converts nothing +20..+35
   (cutting all = -75); graded futility reduction +125; IIR +65; lazy-eval margins on the table +274 (margin 128 vs 32); partial
   iterations +15; time model from empties +50..+73.
5. **Fitting to game outcomes over-weights tactical terms the search already sees** (gataxx `tools.Fit`): regression weights lost
   -47..-94 Elo vs match-tuned ones. Match tuning decides; a fit only filters candidates. Same caution for fits to our NNUE.

## 1. Engines with source here
| engine | where | language / size | eval | search | speed (1 thread) | strength | source |
|---|---|---|---|---|---|---|---|
| **gataxx** (aoflaofl, MIT) | engines-src/aoflaofl__gataxx | Java 21, 9,032 lines | FeatureEvaluator (material, safe 4, edge 8, reach 4, cohesion -3, tempo 2 pieces; weights in 1/16 piece) + PairEvaluator (2x2-pair residual table, phase mobility) | NegaMax PVS, TT, quiescence (>= 3 captures, 5 plies), dominated-jump cut, graded futility, IIR, LMR, Lazy SMP (DeferMinDepth) | README: 4.2-4.5 M nps with the pair table, 9.4-13.4 M without (bench) | vs Moonbird +151 (100 ms), vs Funes +393, vs TikTaxx +418 | README, docs/tuning-log.md |
| **Moonbird** (tsoj) | moonbird-src (+ engines/moonbird.exe) | Nim, 3,285 lines | SGD-tuned: 2x2-block PAIR tables (4 directions x 4^8 codes per square), environmentCounts[square][own nbrs][enemy nbrs][blocker nbrs], mobility[phase][count], turn bonus | alpha-beta, TT, PVS, null move R = 4 + depth/2, LMR, futility reductions, aspiration, IID; all search params SPSA-tunable (searchParams.nim) | 390-480 k nps (2026-11-16) | Titanium: 5k nodes -7, 100 ms -241/-74, 1 s +21 (D105, D155); gataxx +151 | README, evaluation.nim, evalParams.nim |
| **Funes** (Tempate, GPL-3) | engines-src/tempate__Funes (built: engines-src/_bin/Funes.exe) | C++, 2,704 lines | material 100 + komi for 2nd mover (+188 Elo) + 2-region PSQT (+119) + corner pockets / centre penalty (+64) + holes by own-stone count {0,0,25,50,100,150,250,350,400} (+178) + safety +-10 (+99); Elo per term in eval.cpp | TT, NMP, LMR, futility, RFP, razoring | not measured (run blocked, see 4) | gataxx +393 | src/eval.cpp |
| **autaxx** (kz04px) | autaxx (+ engines/autaxx.exe) | C++, 47,042 lines (libataxx) | several searchers: alphabeta, minimax, mcts, nnue, tryhard; tryhard eval = tuned mg/eg: piece 100/79, PST, surrounded 149/27, hole penalties by own neighbours (-17 .. -534), turn 243/394 | alpha-beta (NMP, killers, TT, LMR, RFP, aspiration), MCTS variants, Lazy SMP | not measured | autaxx 20-10 vs Titanium at 5k nodes (Oct, 30 g) | src/autaxx/search/* |
| **Zataxx** (zzzzz151) | Zataxx, zataxx-bz (Botzone rules), zataxx-hce (eval-swap copy, branch hce-eval, unchanged) | Rust, 2,462 lines | 2x2 tuple NNUE 36 x 81 -> 256 SCReLU -> 1, full forward per call (~0.3 us) | PVS, TT, LMR, aspiration, killers, history, futility, RFP, singular, IID | 1.52 M nps, depth 17 in 2 s (2026-11-16); 1.42-1.78 M (D167) | Titanium +81 at 5k; Zataxx +354 at 100 ms (D166/D167, Oct build) | D166, D167, D187 |
| **Monty** (jw1912, MIT; added D1008) | engines-src/jw1912__monty (head f8ea109, 2024-06-09) | Rust, multi-game (chess / ataxx / shatranj) | VALUE = **exactly Zataxx's net**: 36 2x2 tuples (2916 inputs) -> 256 SCReLU -> 1, QA 255 / QB 64 / SCALE 400 (src/value.rs, games/ataxx/board.rs); POLICY = 99 subnets 2916 -> 8 ReLU (49 "from" + 50 "to" incl. clone/pass), move logit = dot(from_vec, to_vec) (games/ataxx/policy.rs) | MCTS PUCT (cpuct 1.41, root_pst 1.0, variance-scaled cpuct off) | not measured (needs a net file, none in the repo) | - | value trained with bullet: wdl 0.5, lr 0.001 x0.1 every 15 superbatches, 40 superbatches x 6104 x 16384 (~4 B samples), SigmoidMSE, data from its own self-play from scratch (uniform policy + material value) (train/value/src/bin/ataxx.rs) |
| **Veritas** (cosmobobak, MIT; added D1008) | engines-src/cosmobobak__veritas (head 16c705c, 2024-04-07) | Rust, 52 files; games gomoku + ataxx (ataxxgen) | AlphaZero policy + value net run on the GPU via Kyanite (CUDA, ONNX); a `pure-mcts` build uses rollouts | MCTS PUCT with root policy softmax temperature, batching, virtual threads | GPU-only; needs ../Kyanite + a net: not runnable here | - | datagen writes policy-target.csv / value-target.csv (search visit distribution + game result); trainer is not in this repo |
| **kZero** (KarelPeeters; added D1008) | engines-src/KarelPeeters__kZero (head 5bf3343, 2026-09-27) | Rust self-play + Python (PyTorch) trainer, multi-game (chess, ataxx, go, sttt, arimaa) | AlphaZero ResNet / attention value + policy; Ataxx policy = 17 x 49 + 1 = 834 outputs (clone per target + 16 jump directions per target + pass); inference with random symmetry or averaged over all symmetries (network/symmetry.rs) | AlphaZero MCTS (MuZero not stable); GPU batching, virtual loss | GPU (README: training runs take a couple of days on ~2x RTX 3090) | Ataxx run "relatively successful" (README) | rust/kz-core/src/mapping/ataxx.rs, python/lib/train.py |
| **AtaxxZero** (petersn, public domain; added D1010) | engines-src/petersn__AtaxxZero (head 097f40d, 2019-10-27) | Python (TensorFlow 1.x) + C++ cuDNN self-play client | AlphaZero ResNet: 4 input planes, 12 residual blocks x 128 filters 3x3, policy 7 x 7 x 17 move types (same 17-type encoding as kZero), value head 1 | MCTS PUCT, Dirichlet alpha 0.15 / weight 0.25, symmetry variant (._engine_with_symmetries.py); uai_interface.py + uai_ringmaster.py | GPU only (README: a few minutes per self-improvement loop on a GTX 970) | - | starts from 2,000 uniformly random games, then looper.py self-improvement; NO trained model in the repo; movegen copied from kz04px/ataxx-engine (MIT) |
| **ataxx.cpp / ataxx-dot-cpp** (GediminasMasaitis "Gedas"; added D1015) | engines-src/GediminasMasaitis__ataxx-dot-cpp (head c1e9105, 2026-08-15) | C++, own trainer (trainer/main.cpp) | v1 HCE: material 100 + unattackable stone (no empty neighbour) +50 + tempo 400; v2-3 NNUE 98 inputs (49 own + 49 enemy) -> 768 x 2 perspectives -> 1, efficient updates + accumulator stack | AB, PVS (3-fold), TT, RFP, LMR (by moves evaluated), LMP, IIR (double), history (aging, penalty), lazy SMP, WDL / jail detection, material-only eval when 1 square off | v1.2 brute AB ~66-86 M nps at start (material eval) | **changelog.md = Elo per change** (section 1b below) | changelog.md, src/evaluation_hce.cpp, src/evaluation_nn_base.h |
| **Ataxxer** (zzzzz151 = Zataxx author; added D1015) | engines-src/zzzzz151__Ataxxer (a439941, 2024-04-14) | Python match runner + SPRT (main.py, sprt_math.py) | - | - | - | - | **balanced_openings.txt (14,308 positions) + unbalanced_openings.txt**; the 545 M-position Zataxx dataset that gave ataxx.cpp +245 Elo is NOT in any public repo |
| **Sanctaphraxx 0.1.11** (Ciekce) | sanctaphraxx | Rust, 4,376 lines | NNUE (net004.nnue 19 KB) | alpha-beta, TT, history, little pruning | **14.7 M nps**, depth 6 in 2.5 s (2026-11-16) | lost 2-0 to Titanium at 5k (Oct) | D969 |
| **tiktaxx** (kz04px) | tiktaxx (+ engines/tiktaxx.exe) | C++, 5,064 lines (libataxx) | material 100 + fixed PST (corners 30, edges 10-20) + turn bonus 300 by phase | alpha-beta, TT, NMP, LMR, aspiration, killers, futility, IID, also MCTS | not measured | gataxx +418 | src/eval.cpp |
| **AtaXXL** (andr3eee1) | engines-src/andr3eee1__AtaXXL | C, 2,153 lines | Python-tuned linear (weights.h): material 981 + "snowball" 18, mobility 17, frontier 14, 10-class PST (-25..+34) | PVS, TT, NMP, LMR, killers, history, quiescence | does not build (book.h missing from the repo) | - | logic.c, weights.h |
| **Weixx** (TerjeKir, from the chess engine Weiss) | engines-src/TerjeKir__Weixx (built: _bin/Weixx.exe) | C, 2,451 lines | material only (200 per stone) | Weiss search: TT, NMP, LMR, aspiration, history, quiescence, threads | not measured | - | src/evaluate.c |
| **ataxx-engine** (TraianDanciu) | engines-src/TraianDanciu__ataxx-engine (built: _bin/ataxx-engine.exe) | C++, 939 lines | material + incremental square-sum + clone-target mobility; final eval counts empties for the opponent | TT, NMP, LMR, aspiration, killers, history, futility | not measured | - | Eval.cpp |
| **Anthraxx** (Vast342, from the chess engine Clarity) | engines-src/Vast342__Anthraxx | C++, 1,382 lines | (in search) | TT, NMP, aspiration, history | build needs -DVersion (fixed); not built (stopped) | - | README |
| **loltaxx** (Mk-Chan) | engines-src/Mk-Chan__loltaxx | C++, 2,826 lines | material only (1000 per stone) | TT, threads | needs CLI11 header (not in repo) | - | app/eval.cpp |
| **gotaxx** (cpirc) | engines-src/cpirc__gotaxx | Go, 1,877 lines | material 1000 + fixed PSQT | TT, NMP | no Go toolchain here | gataxx ~90%+ | engine/eval/eval.go |
| **mexx** (raklaptudirm) | engines-src/raklaptudirm__mexx | Rust, 1,155 lines | MCTS (cpuct sqrt 2) | MCTS | build denied (cargo build scripts of downloaded code) | - | src/mcts |
| **andromeda** (CatalinFrancu) | engines-src/CatalinFrancu__andromeda | C++, 1,560 lines | Monte-Carlo playout score | MCTS | not built (needs Config.h) | - | src |
| small / homework engines | engines-src: curtisbright__ataxx-ai, feifeiobama__AtaxxBot (Botzone, AB + MCTS), fffasttime__AI-ataxx (Botzone 1879), QtLab__Ataxx (Qt), FugEbinMeme__Ataxx, cosmobobak__ataxxgen, Sjord__atars, Topool45-dev__ataxx-bot | 400-7,500 lines | - | - | - | - | - |
| **KataAtaxx** (KataGomo Ataxx_20260501) | engines/kataataxx | C++ / ONNX, GTP only | ResNet value + policy | MCTS (PUCT) | ~2 visits/s CPU; 1 visit ~ 1,000 Titanium nodes | Titanium +191 at 1 s; Kata wins at 100 ms only with ~9x wall time | D105, D156, D755 |
| **Jataxx v37** (jacek) | jataxx-study | C++ | 25 overlapping 3x3 windows -> 128 sq-ReLU -> 3 buckets -> tanh | NN best-first minimax, no TT | ~975 ms/move | weights private: not runnable | D611, D612 |
| **Fairy-Stockfish** | engines/fairy-stockfish.exe | C++ | NNUE or classical | alpha-beta | not measurable: no compatible Ataxx net, no info lines | - | D965 |
| **Ataxx_CPP** (EmptyBlue, Botzone #3) | (run via ax_vs_ti.py in Oct; repo EmptyBlueBox/Ataxx is gone from GitHub) | C++ | - | - | - | Titanium wins every leg (D105) | D105 |
Tools: libataxx (kz04px, C++ bitboard library used by tiktaxx / autaxx / cuteataxx), cuteataxx (kz04px, match runner, JSON settings),
arbiter (raklaptudirm, Go tournament toolkit), ataxx-skeleton (engine template), rustaxx (kz04px, Rust bitboard library: fen, movegen,
makemove, perft, gameover; engines-src/kz04px__rustaxx, added D1008). Community list (ataxx README, user paste 2026-11-17): Autaxx, kZero,
Monty, Moonbird, Sanctaphraxx, Tiktaxx, Veritas, Zataxx + cuteataxx, libataxx, rustaxx: all now cloned and in this table.
**kz04px's full account (D1009, 35 repos via the GitHub API, 2026-11-17):** Ataxx-related, all cloned to engines-src/kz04px__<name>:
faeries (Rust multi-game engine, pushed 2026-11-16: alphabeta / minimax / flat MC / random searches, Ataxx eval = material only 100 per
stone, datagen; UGI/UAI/TEI/GTP protocols), games (Rust rules library: Ataxx any size, zobrist matching libataxx, perft tests), cutergames
(Rust UGI/UAI match runner with SPRT + **ready Ataxx opening books: openings/ataxx/ply3_fair.txt = 883 ply-3 positions filtered by
Autaxx depth-12 |eval| <= 300**, ply1-ply4 full lists, tournament.txt, unusual.txt), cutegames (C++ predecessor runner), python-ataxx
(Python rules + PGN + UAI client), ataxx-tools (Python game statistics / graphs), libnnue (generic C++ NNUE helpers, no Ataxx net), ugi
(protocol). Not Ataxx: soup, dove, rawr (no Ataxx code), 4ku, Baislicka, libchess, chess-tools and older non-game repos. No new strong
engine and no Ataxx net file in the account; the useful new item is the "fair" opening set for gates.
**Discord / info-site links (D1011, user paste 2026-11-17):** kZero's PUCT (rust/kz-core/src/zero/node.rs `Uct::total`) =
q + c_explore * u + w_moves_left * clamp(sharpness * clamp(m, +-clip) * (-q), +-1), with m = this child's moves-left minus its siblings'
(a moves-left head as in Lc0: prefer shorter wins and longer losses). Folkert van Heusden's Ataxx game server (ataxx.vanheusden.com
port 28028: "user" / "pass" lines then plain UAI; stats page on the same host) = repo folkertvanheusden/Waxx (engines-src, Python
waxx.py + sqlite schema + client.py wrapper + 445 KB openings.txt); the host's TLS certificate does not match its name (2026-11-17),
so its rating page was NOT read; later servers named in talkchess threads: keetweej.vanheusden.com:28028, server.ataxx.org (login),
Litaxx (none checked). Info site komputilo.nl/games/ataxx lists TikTaxx, GoTaxx, Funes, AtaxxZero, LuaTaxx (SamuraiDangyo/luataxx:
404, gone) and the acraps runner (folkertvanheusden/acraps: 404, gone). "Crazy Zero" (connected to that server by Remi Coulom, 2023):
no public source found. D1275 (2026-11-19 re-search, user question "do we have Crazy Zero documented"): Crazy Zero = Remi Coulom's PRIVATE
AlphaZero-style zero-learning framework (author of Crazy Stone); the Polygames paper (arXiv 2001.09832) lists it as applied to Go, Chess,
Shogi, Gomoku, Renju, Othello and Ataxx but gives no Ataxx result; only public strength datum = gomoku (beat Yixin 48-34, 3 days on 1 GPU,
his 2018 post). No Ataxx rating, source, net or match record found (talkchess 'Ataxx' t=72089 page 2 has no mention; server.ataxx.org:28028
= flok's Waxx server, no rating list linked). Open-source Zero-style Ataxx engines on the EngineProgramming list: kZero (Karel Peeters),
Veritas (cosmobobak, AlphaZero Rust+Python on Kyanite), Monty (jw1912, CPU AlphaZero); Edinburgh ug4 project (DNN+MCTS Gomoku/Ataxx);
AtaxxZero (Python). Strongest-engine ranking: still unknown; our own evidence = Titanium champion > KataAtaxx, vs Zataxx equal per node at
5k / behind at 100 ms (D1237), Botzone ranks (PyJataxx 2157, Kata_mini 2224, evo 1993). Test idea: ask Coulom / join server.ataxx.org
(needs user OK, outward contact) -> BACKLOG D1275.
**D1276 (2026-11-19, user Q: 1 visit kZero vs 1 visit Kata/Crazy Zero): kZero 1 VISIT MEASURED.** Tool = scratch/kz_probe (Rust, tract-onnx CPU,
src/main.rs; pest_stack_guard stubbed to build with llvm-mingw): runs engines-src/_downloads/kzero_ataxx_net/ataxx_16x128_gaps_gen_6972.onnx
(input 4x7x7 = since-copy/100 plane, own, other, gaps; outputs scalars(5)+policy 834) = argmax of the legal-masked policy (1 visit = root
expansion only), vs Titanium main 0120fd12 `serve` UAI at fixed nodes, classic rules (passes allowed, ends when nobody moves / 100 plies
without copy, count decides), 50 random 4-ply openings x both colours = 100 g per level, seed 7. Results (kZero score, Elo): Ti 1 node 99-1
(+798); 10 nodes 80% (+241 [+165,+344]); 100 nodes 33% (-123 [-202,-55]); 300 nodes 18.5% (-258 [-364,-181]); 1000 nodes 6% (-478).
=> kZero 1 visit ~ Titanium ~45 nodes (first rough interpolation, SUPERSEDED). REFINED (200 g each, seed 11): Ti 20 nodes kZero 79%; 30 nodes 72.2%; 45 nodes 66.0% (+115); 70 nodes 46.0% (-28 [-77,+20]); 150 nodes 32.0% (-131) => EQUAL at ~65 Titanium nodes; Titanium needs ~70 nodes to match, ~150 to clearly beat, ~1000 to win 94%. KataAtaxx 1 visit ~ Titanium 850-1000 nodes
(KATA_ARENA) => Kata 1 visit is ~20x stronger than kZero 1 visit in Titanium nodes (different nets/sizes, different arenas/rules: indicative,
not paired). kZero cost 11-27 ms/move CPU (16x128 net, tract). Crazy Zero: untestable (private). N-visit kZero (PUCT in the tool, arg 3) not
run yet. Sanity: kZero plays far above random (99% vs 1 node), so the input/policy mapping is right. ZanLing-TrueZero = andreaslam/TrueZero (engines-src): an AlphaZero CHESS engine (Rust + Python); its only
Ataxx code is the mapping/games files of kZero's Python library, no Ataxx engine.
vanheusden.com/games/ataxx (D1014) = the same info site as komputilo.nl/games/ataxx; extra links there: Cuppen BSc paper on Ataxx search (project.dke.maastrichtuniversity.nl/games/files/bsc/Cuppen_BSc-paper.pdf, NOT read yet: PDF download), GGS generic game server with an Ataxx interface (skatgame.net/mburo/ggs), game-ai-forum Ataxx thread viewtopic.php?f=24&t=691, Ataxx Discord invite discord.gg/32Fjky3vS6.
**Discord dump 2 (D1015, user paste 2026-11-17):** KarelPeeters' trained Ataxx net (kZero, supports gaps, Google Drive file
1ZNwH-MOU8oRHEntxim1vZj9VALaypnzU) DOWNLOADED on the user's order (D1016): engines-src/_downloads/kzero_ataxx_net/ataxx_16x128_gaps_gen_6972.onnx (19,206,275 B, sha256 5DFF76D0..., PyTorch 1.12.0 export; 16 residual blocks x 128 channels; input 'input' = kZero AtaxxStdMapper: moves-since-last-copy scalar + own / other / gap planes on 7x7; outputs 'scalars' (value, wdl, moves-left) and 'policy' (17 x 49 + 1 = 834); not yet run or scored); hzyhhzy/KataGo branch Ataxx (engines-src, 2022: the KataAtaxx engine;
hzyhhzy = Botzone 'sigmoid', owner of RawNN) + HF dataset hzydashi/KatagoExceptGo ataxx/bruteforce/5x5.7z (2.7 GB, 5x5 Ataxx solved for all
positions, not downloaded, not 7x7); folkertvanheusden/Scarletxx (UCT engine, engines-src); crippa1337/GYARB + kurt (Rust, engines-src; same
42 files); KarelPeeters/board-game-rs (rules library, outcome edge cases); chesstastic-org/monster-chess (multi-game, barebones Ataxx);
tsoj/weather-factory-ataxx (SPSA tuner for UAI options, engines-src), tsoj/AtaxxGUI (Qt GUI, engines-src; tournament branch),
CallMeRoot-J/ataxx-ui (UAI GUI, engines-src); Moonbird: our moonbird-src (2025-04-15) is newer than its last release 1.1.0 (2024-05-26, no-gap
build + lazy SMP); **ADU hash** (raklaptudirm, after eprint.iacr.org/2011/116): hash = X*a + Y*b + hi64(Y*a) + hi64(Z*b) on the two
bitboards, measured 1 G hashes/s vs 77 M for efficient incremental Zobrist, collisions equal at 24 / 12 bits (idea for Titanium, BACKLOG);
pressibus.org/ataxx (history, rules, strategies, 2,868 symmetric gap layouts); pychess.org plays Ataxx (Fairy-Stockfish); papers: MCTS Ataxx
(leoribeiro.github.io/papers/mcts-ataxx-eniac2018.pdf) and Cuppen BSc (both NOT downloaded); ataxx crate (crates.io 'ataxx', mexx author);
bitboard-viewer.vercel.app; Veritas note: its PUCT net beat pure MCTS 7-0 at 8 s+80 ms.
**Lesson from Monty (D1008):** the Zataxx value net is Monty's Ataxx value net; Monty adds a policy net on the SAME 2x2-tuple inputs
(per-square from / to subnets of width 8, logit = dot product), a cheap factorised policy shape to compare with our policy head. Gone from GitHub: EmptyBlueBox/Ataxx, dieGoCode/ataxx-zero-ai.

**GitHub search for a 3x3 tuple net (D1287, 2026-11-19):** none public. Tuple nets found are all 2x2 (Zataxx, Monty, ataxx-dot-cpp). 3x3 = Jataxx only (jataxx-study). Newly cloned and grepped, no tuple eval: Ciekce/sanctaphraxx (Rust UAI NNUE), kz04px/autaxx (C++), kz04px/tiktaxx (C++ alpha-beta). cheetax (fab), PyJataxx, noexcept, Ataxx_CPP have no GitHub repo that search could find. Uninspected: Yangxc0414/Ataxx-AI (Botzone rank 30), Dieg0Code/ataxx-zero-ai, alexxtaurus83/AttaxTensorflow.

## 1b. ataxx.cpp tuning log (GediminasMasaitis, changelog.md; self-play SPRT with cuteataxx, books ply3_fair then openings_8ply; Elo +- 95%)
HCE era (v1): alpha-beta +362, capture-count ordering +46, **unattackable stone +50 cp = +140**, beta-bound fix +37, TT move +41, PVS + TT
cutoffs +27, **RFP +86**, LMR +46, WDL detection +62, history +18, **LMP +53**, tempo +31..+121, **min/max time management +86**, total-material-gain
ordering +37. NN era (v2, own data): side-to-move-relative inputs +418 vs white/black, int quantisation +63, NNUE updates +0, 98->32 +193,
98->64 +101, tighter RFP +53, **22 M fresh positions for 40 epochs +179**, IIR +27, history aging +27, 100 epochs +36, 64x2 perspectives +90,
accumulator stack +19, **rotation augmentation +48, + diagonal mirroring +11, 8-fold symmetry at 128 wide +27**, 128 wide +41, 256 wide +37.
v3 (Zataxx's data): **same 98->256x2 net on 545 M positions with 8-fold symmetry = +246**, double IIR +12, 512 wide +57, lazy SMP 12 threads +104,
3-fold PVS +9, improved LMR +23, LMR by moves evaluated +32, deeper RFP +9, history penalty +10, 768 wide (9 epochs) +17, **material-only eval
when one square is left +17**, more IIR +6.
**For Titanium:** (1) data volume dominated every architecture change (+246 for 25x data; matches our train_v10 gains); (2) D4 augmentation
is worth +27..+59 on a position-specific net (our D999-D1006 result); (3) exact / material eval at 1 empty square is free Elo; (4) the
gataxx + ataxx.cpp HCE lists agree: stones with no empty neighbour (safe / unattackable) are the strongest single hand term.

## 2. Botzone opponents (no binaries; read from match logs)
Ranklist 2026-11-01 (D299): atx 2591, Ataxx_Python 2554, Ataxx_CPP 2356, cheetax 2246, KataAtaxx_mini 2224, Kissin_Flash 2206,
PyJataxx 2157, Sincerely 1972, evo 1971, Daybreak 1944, sotax 1936, v1_flash 1934, hce 1916, RawNN 1915, xie 1911, waffle_ataxx 1908.
| bot | what we know | speed | vs Titanium | source |
|---|---|---|---|---|
| cheetax (fab) | alpha-beta, depth 24-26, seldepth to 99 | ~2.2 M nodes/s, 8.4 M nodes/move | 2026-11-16: won 25-24, lost 18-31 | D938-D943 |
| PyJataxx (jacek) | Jataxx lineage behind Python, ~3.4 s/move | - | 3 W / 3 L (2026-11-14) | D612, D784 |
| [noexcept]Ataxx (noexcept) | py36 driver, MCTS-family value search (PUCT-like): smooth q in [-1, 1] (max single-move jump 0.29 over 190 assessed moves), full 3.94 s/move, 146→512 MB growing tree across a game, ~11.6 s first move; ranked bot "Ataxx" is v1 (fast climber), "Ataxx1" v13 is the probable testbed | - | 0 W / 3 L (2026-11-18), both seats, 108-138 turns | D1151 |
| sotax | "ab-nnue" | 403-435 k nps (3.3 M endgame) | beat Titanium 2-0 | D278 |
| hce (dblue) | hand-crafted alpha-beta, ~1 s/move | 2.7-3.6 M nps, depth 12-17 | Titanium won 2-0 | D290 |
| evo (**cosmoz = cosmobobak**, author of Veritas + ataxxgen; user D1013) | Botzone rank 9, **1993.26** (2026-11-17, user; 1971 on 2026-11-01); user note '080805-12h' (meaning not stated) | - | not played yet in our logs | engine behind it unknown: Veritas (AlphaZero MCTS, GPU) cannot run on Botzone's CPU, so likely a CPU build or another cosmobobak engine (Viridithas is his chess engine); see engines-src/cosmobobak__veritas, cosmobobak__ataxxgen | D1013 |
| v1_flash | "alphabeta + NN" | - | Titanium won 2-0 | D296 |
| RawNN | Kata net, no search | - | Titanium won | D300 |

## 3. What this means for Titanium's HCE work
- Terms to test first (proven in gataxx play, consistent with our net-occlusion findings): edge stones, safe stones, reach /
  0-weakness stones, negative cohesion, holes graded by own-stone count (Funes, autaxx); skip mobility / territory / threat counts.
- The learned local table (our dynamic PST / weakness PST) must be BLENDED with the hand terms and guarded by a lazy-eval margin,
  and summed cheaply (byte lanes or bit-sliced counts), or it loses on the clock (gataxx -57 before the speed-ups).
- Search ideas we have not tried, with Elo measured elsewhere: quiescence on >= 3-capture moves with the dominated-jump cut; graded
  futility; jump cut only for non-converting dominated jumps; time model from empties.
- Eval-swap test bed: Zataxx (Rust, our bit layout, one eval function); copy ready at C:/gitProjects/zataxx-hce.

## 4. Not done / blocked
- Running binaries built from the downloaded sources (Funes, Weixx, ataxx-engine) and building downloaded Rust crates (mexx) were
  DENIED by the Claude Code auto-mode permission check on 2026-11-16; their speeds are therefore unmeasured. Needs the user's
  permission rule to continue.
- Not built: AtaXXL (book.h missing), loltaxx (CLI11 missing), gotaxx (no Go), gataxx (no Java / Maven), andromeda (Config.h).
- No engine here has been gated against TODAY's main.

## 6. Other engines' term sets as imitations of our net (D974, `lab hce-compare`, logs/hce/d974_hce_compare.txt, 11 s)
Each set's TERMS refitted closed-form to the champion eval (4 phases; 20k positions + children; 2k held out):
| term set | dims | R2 | sibling best-move agreement | regret cp |
|---|---|---|---|---|
| gataxx (material, safe, edge, reach, cohesion) | 24 | 0.602 | 33.95% | 106.7 |
| Funes (material, corner/edge regions, corner pockets, centre arms, graded holes, weak/safe singles) | 112 | 0.600 | 37.35% | 97.9 |
| autaxx tryhard (material, raw 49-square PST, graded holes, surrounded) | 244 | 0.629 | 37.60% | 98.7 |
| ours (hce-fast: weakness x class tables, soft weaknesses, holes, frontier, no-move flags) | 820 | 0.737 | 40.15% | 78.8 |
| ours + gataxx | 844 | 0.741 | 40.55% | 78.5 |
gataxx terms refitted to OUR net (cp per unit, phases late..open): safe 25.5 / 49.3 / 69.0 / 49.4, reach 60.0 / 16.4 / 22.2 / 49.8, edge -1.4 / 10.6 / 10.0 / -4.1, cohesion **-22.4 / -12.8 / -14.3 / -25.4** (negative in every phase): our net independently prices own-own adjacency negatively, the same as gataxx's +97 Elo finding.
Reading: our weakness-table set captures the most of what the net sees (+0.11-0.14 R2, +2.5-6 pts agreement over the other engines' sets); adding gataxx's 5 terms on top gives +0.4 pts. These are imitation scores, not play strength (gataxx warns that fitted weights lose to match-tuned ones).

## 7. Gates vs today's main (1 thread per engine)
| engine | clock | games | result for the engine | log |
|---|---|---|---|---|
| gataxx 2.0.0 | 100 ms | 104 (SPRT stop) | +24 -77 =3 = **-195 +- 80 Elo** vs Titanium main 0120fd12 | logs/ext/gataxx_100ms.log (D978) |
| gataxx 2.0.0 | 1 s | 80 (stopped, decided) | +11 -68 =1 = **-310 +- 120 Elo** vs Titanium main 0120fd12 | logs/ext/gataxx_1000ms.log (D984) |

16 buckets (NNUE bucketing, D980): gataxx 0.685, Funes 0.708, autaxx 0.722, ours 0.785, ours+gataxx 0.789 R2 (logs/hce/d980_hce_compare_16b.txt).
32 buckets (D981): gataxx 0.687, Funes 0.715, autaxx 0.727, ours 0.788, ours+gataxx 0.792 R2 (logs/hce/d981_hce_compare_32b.txt): small gain over 16.
42 buckets (D982): gataxx 0.690, Funes 0.717, autaxx 0.727, ours 0.789, ours+gataxx 0.793 R2; agreement slightly lower than 16/32: saturated on 20k positions.
45 exact buckets (one per legal stone count, D983): gataxx 0.691, Funes 0.717, autaxx 0.727, ours 0.789, ours+gataxx 0.793 R2: same as 42 on 20k positions.
