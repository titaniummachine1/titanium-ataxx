@echo off
REM v2 WDL gate: candidate final_v2_wdl_ep8.tup vs MAIN tupS56twostage.tup
REM Net-only change -> 5k fixed nodes (GATE LAW v5: PASS = 95% CI lower > 0.5).
REM 4 disjoint paired-book legs x 100 games (50 lines x 2 mirrored seats each),
REM starts 0/50/100/150 -> 400 games, 200 per seat.
REM Both sides SAME binary: main-ataxx release exe D992E3CC. Only --tuple differs.
set ROOT=C:\gitProjects\titanium-ataxx
set EXE=C:\gitProjects\main-ataxx\target\release\titanium-cli.exe
set CAND=data\nnue\final_v2_wdl_ep8.tup
set CHAMP=data\nnue\tupS56twostage.tup
set BOOK=data\nnue\book50k.txt
cd /d %ROOT%
echo === hashes ===
certutil -hashfile "%EXE%" SHA256 | findstr /v ":"
certutil -hashfile "%CAND%" SHA256 | findstr /v ":"
certutil -hashfile "%CHAMP%" SHA256 | findstr /v ":"
start "" /b "%EXE%" match --games 100 --nodes 5000 --tuple "%CAND%" --opp self --opp-tuple "%CHAMP%" --opp-nodes 5000 --book "%BOOK%" --book-start 0 --start-game 1 --out logs\gate_v2_ep8_p0.txt
start "" /b "%EXE%" match --games 100 --nodes 5000 --tuple "%CAND%" --opp self --opp-tuple "%CHAMP%" --opp-nodes 5000 --book "%BOOK%" --book-start 50 --start-game 1 --out logs\gate_v2_ep8_p1.txt
start "" /b "%EXE%" match --games 100 --nodes 5000 --tuple "%CAND%" --opp self --opp-tuple "%CHAMP%" --opp-nodes 5000 --book "%BOOK%" --book-start 100 --start-game 1 --out logs\gate_v2_ep8_p2.txt
start "" /b "%EXE%" match --games 100 --nodes 5000 --tuple "%CAND%" --opp self --opp-tuple "%CHAMP%" --opp-nodes 5000 --book "%BOOK%" --book-start 150 --start-game 1 --out logs\gate_v2_ep8_p3.txt
