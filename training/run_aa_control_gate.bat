@echo off
REM A/A CONTROL (main in-process vs main serve, same S57; expect ~0.5). Was: TUP7-fix gate: candidate exp/tup7-fix 653d9a1 vs MAIN 298b7ae, SAME net (S57 final_v2_wdl_ep8.tup) both sides.
REM Only change: the engine now evaluates TUP7 nets exactly as trained (capn/conv loaded, pa aligned, s/q2 state 0,
REM env on every square). Speed IS affected (capconv + 49-square env) -> 5k nodes = direction (diagnosis); merge
REM needs the time-control legs too (law 2). 4 disjoint paired-book legs x 100 games, 200 per seat.
set ROOT=C:\gitProjects\titanium-ataxx
set CAND=C:\gitProjects\main-ataxx\target\release\titanium-cli.exe
set MAIN=C:\gitProjects\main-ataxx\target\release\titanium-cli.exe
set NET=data\nnue\final_v2_wdl_ep8.tup
set BOOK=data\nnue\book50k.txt
cd /d %ROOT%
echo === hashes ===
certutil -hashfile "%CAND%" SHA256 | findstr /v ":"
certutil -hashfile "%MAIN%" SHA256 | findstr /v ":"
certutil -hashfile "%NET%" SHA256 | findstr /v ":"
start "" /b "%CAND%" match --games 100 --nodes 5000 --tuple "%NET%" --opp "%MAIN% serve" --opp-tuple "%NET%" --opp-nodes 5000 --book "%BOOK%" --book-start 0 --start-game 1 --out logs\gate_aa_mainext_p0.txt
start "" /b "%CAND%" match --games 100 --nodes 5000 --tuple "%NET%" --opp "%MAIN% serve" --opp-tuple "%NET%" --opp-nodes 5000 --book "%BOOK%" --book-start 50 --start-game 1 --out logs\gate_aa_mainext_p1.txt
start "" /b "%CAND%" match --games 100 --nodes 5000 --tuple "%NET%" --opp "%MAIN% serve" --opp-tuple "%NET%" --opp-nodes 5000 --book "%BOOK%" --book-start 100 --start-game 1 --out logs\gate_aa_mainext_p2.txt
start "" /b "%CAND%" match --games 100 --nodes 5000 --tuple "%NET%" --opp "%MAIN% serve" --opp-tuple "%NET%" --opp-nodes 5000 --book "%BOOK%" --book-start 150 --start-game 1 --out logs\gate_aa_mainext_p3.txt
