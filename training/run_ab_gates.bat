@echo off
REM ============================================================================
REM Gates each A/B arm vs CHAMPION (tupS55tree.tup = F66A00FE) at 5k nodes,
REM paired book, 100 games each (50 lines x 2 seats), different book ranges.
REM Both sides run the SAME binary (main-ataxx B3422646) — only the net differs.
REM Run ONLY after abA_onepass.tup and abB_twostage.tup exist.
REM ============================================================================
set ROOT=C:\gitProjects\titanium-ataxx
set EXE=C:\gitProjects\main-ataxx\target\release\titanium-cli.exe
set CHAMP=data\nnue\tupS55tree.tup
set BOOK=data\nnue\book50k.txt
cd /d %ROOT%

echo === exe + net hashes ===
certutil -hashfile "%EXE%" SHA256 | findstr /v ":"
certutil -hashfile "%CHAMP%" SHA256 | findstr /v ":"
certutil -hashfile "data\nnue\abA_onepass.tup" SHA256 | findstr /v ":"
certutil -hashfile "data\nnue\abB_twostage.tup" SHA256 | findstr /v ":"

echo === GATE A: abA_onepass vs champion, 100g, book-start 0 ===
"%EXE%" match --games 100 --nodes 5000 --tuple data\nnue\abA_onepass.tup ^
  --opp self --opp-tuple "%CHAMP%" --opp-nodes 5000 ^
  --book "%BOOK%" --book-start 0 --start-game 1 ^
  --out logs\gate_abA_vs_champ.txt

echo === GATE B: abB_twostage vs champion, 100g, book-start 50 ===
"%EXE%" match --games 100 --nodes 5000 --tuple data\nnue\abB_twostage.tup ^
  --opp self --opp-tuple "%CHAMP%" --opp-nodes 5000 ^
  --book "%BOOK%" --book-start 50 --start-game 1 ^
  --out logs\gate_abB_vs_champ.txt

echo === SUMMARY ===
python training\gate_summary.py logs\gate_abA_vs_champ.txt logs\gate_abB_vs_champ.txt
