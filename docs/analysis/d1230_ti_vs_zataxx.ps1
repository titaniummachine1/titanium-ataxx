# D1230 (user 2026-11-18): gate Titanium main 0120fd12 vs stock Zataxx, then vs the upgraded Zataxx (zataxx-bz 02bbce8).
# Titanium = candidate (Elo > 0 = Titanium better). Each engine plays under its own rules:
#   G1 stock Zataxx (zataxx-ab/default, libataxx rules, passes) vs Titanium classic `serve`, --rules libataxx (as D166).
#   G2 zataxx-bz (Botzone judge rules) vs Titanium `serve --bz`, --rules botzone (Titanium's deployed rules).
# Legs: 100 ms both (400 g max, SPRT may stop early), then 1000 ms both (200 g max). Every gate ends with ti insight + ti analyze.
# Stop: create C:\gitProjects\titanium-ataxx\logs\ext\d1230_STOP (checked between gates) or Stop-Process the ti.exe.
$ErrorActionPreference = 'Continue'
Set-Location C:\gitProjects\titanium-ataxx
$ti  = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$tim = 'C:/gitProjects/_gate/exes/titanium-0120fd12.exe'
$zs  = 'C:/gitProjects/zataxx-ab/default/target/x86_64-pc-windows-gnu/release/zataxx.exe'
$zb  = 'C:/gitProjects/zataxx-bz/target/x86_64-pc-windows-gnu/release/zataxx.exe'
$out = 'C:\gitProjects\titanium-ataxx\logs\ext'
$rep = 'C:\gitProjects\titanium-ataxx\logs\gates\reports'
$stop = "$out\d1230_STOP"
$gates = @(
  @{ n='g1_stock_100ms'; cand="$tim serve";      base=$zs; rules='libataxx'; ms=100;  g=400 },
  @{ n='g2_bz_100ms';    cand="$tim serve --bz"; base=$zb; rules='botzone';  ms=100;  g=400 },
  @{ n='g1_stock_1000ms';cand="$tim serve";      base=$zs; rules='libataxx'; ms=1000; g=200 },
  @{ n='g2_bz_1000ms';   cand="$tim serve --bz"; base=$zb; rules='botzone';  ms=1000; g=200 }
)
foreach ($x in $gates) {
  if (Test-Path $stop) { "STOP file -> end $(Get-Date -Format HH:mm:ss)"; break }
  $log = "$out\d1230_ti_vs_zat_$($x.n).log"
  "=== $($x.n) start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  & $ti sprt --candidate $x.cand --baseline $x.base --non-main-ab --rules $x.rules --time-ms $x.ms --elo0 0 --elo1 20 `
      --max-games $x.g --threads 4 --out $log 2>&1 | Select-Object -Last 3
  $gj = "$log.games.jsonl"
  if (Test-Path $gj) {
    & $ti insight $gj --by-bucket 2>&1 | Out-File -Encoding utf8 "$rep\d1230_$($x.n)_insight.txt"
    & $ti analyze $gj 2>&1 | Out-File -Encoding utf8 "$rep\d1230_$($x.n)_analyze.txt"
  }
  "=== $($x.n) done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}
"queue end $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
