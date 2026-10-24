@echo off
REM TUP7-fix gate at 50 ms per move (time leg, speed included): candidate exp/tup7-fix 653d9a1 vs MAIN 298b7ae, SAME net (S57 final_v2_wdl_ep8.tup) both sides.
REM Only change: the engine now evaluates TUP7 nets exactly as trained (capn/conv loaded, pa aligned, s/q2 state 0,
REM env on every square). Speed IS affected (capconv + 49-square env) -> 5k nodes = direction (diagnosis); merge
REM needs the time-control legs too (law 2). 4 disjoint paired-book legs x 100 games, 200 per seat.
set ROOT=C:\gitProjects\titanium-ataxx
set CAND=C:\gitProjects\tup7fix-ataxx\target\release\titanium-cli.exe
set MAIN=C:\gitProjects\main-ataxx\target\release\titanium-cli.exe
set NET=data\nnue\final_v2_wdl_ep8.tup
set BOOK=data\nnue\book50k.txt
cd /d %ROOT%
echo === hashes ===
certutil -hashfile "%CAND%" SHA256 | findstr /v ":"
certutil -hashfile "%MAIN%" SHA256 | findstr /v ":"
certutil -hashfile "%NET%" SHA256 | findstr /v ":"
start "" /b "%CAND%" match --games 100 --time 50 --tuple "%NET%" --opp "%MAIN% serve" --opp-tuple "%NET%" --opp-time 50 --book "%BOOK%" --book-start 200 --start-game 1 --out logs\gate_tup7fix50ms_p4.txt
start "" /b "%CAND%" match --games 100 --time 50 --tuple "%NET%" --opp "%MAIN% serve" --opp-tuple "%NET%" --opp-time 50 --book "%BOOK%" --book-start 250 --start-game 1 --out logs\gate_tup7fix50ms_p5.txt
start "" /b "%CAND%" match --games 100 --time 50 --tuple "%NET%" --opp "%MAIN% serve" --opp-tuple "%NET%" --opp-time 50 --book "%BOOK%" --book-start 300 --start-game 1 --out logs\gate_tup7fix50ms_p6.txt
start "" /b "%CAND%" match --games 100 --time 50 --tuple "%NET%" --opp "%MAIN% serve" --opp-tuple "%NET%" --opp-time 50 --book "%BOOK%" --book-start 350 --start-game 1 --out logs\gate_tup7fix50ms_p7.txt
