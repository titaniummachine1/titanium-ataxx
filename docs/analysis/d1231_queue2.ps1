# D1231: after the two 100 ms Titanium-vs-Zataxx legs of d1230_ti_vs_zataxx.ps1, gate the two engine candidates vs main
# (Botzone rules, 100 ms), then run the two 1000 ms Zataxx legs. One PC job at a time.
#   R2 exp/nonpv-re2 f4a3ed4 (no duplicate non-PV re-search), R8 exp/semiterm d3c3444 (exact <= 2-empty solver).
# Stop after the current step: New-Item C:\gitProjects\titanium-ataxx\logs\ext\d1231_STOP
$ErrorActionPreference = 'Continue'
Set-Location C:\gitProjects\titanium-ataxx
$ti   = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$q1   = 'C:\gitProjects\logs\d1230_ti_vs_zat.log'
$s1   = 'C:\gitProjects\titanium-ataxx\logs\ext\d1230_STOP'
$stop = 'C:\gitProjects\titanium-ataxx\logs\ext\d1231_STOP'
$rep  = 'C:\gitProjects\titanium-ataxx\logs\gates\reports'
function Reports($tag, $since) {
  $gj = Get-ChildItem -Recurse C:\gitProjects\titanium-ataxx\logs\gates, C:\gitProjects\titanium-ataxx\logs\ext -Filter *.games.jsonl -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt $since } | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($gj) {
    "reports for $tag from $($gj.FullName)"
    & $ti insight $gj.FullName --by-bucket 2>&1 | Out-File -Encoding utf8 "$rep\d1231_${tag}_insight.txt"
    & $ti analyze $gj.FullName 2>&1 | Out-File -Encoding utf8 "$rep\d1231_${tag}_analyze.txt"
  } else { "no games.jsonl found for $tag" }
}
# v2 (D1235): queue1's g2_bz_100ms crashed at start (zataxx-bz printed 'score N', the hub needs 'score cp N'; fixed in
# zataxx-bz af9eee5) and queue1 went on to g1_stock_1000ms; d1230_STOP is set, so queue1 ends after that leg.
"queue2 v2 start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): waiting for queue1 to end"
while (-not (Select-String -Path $q1 -Pattern 'STOP file|queue end' -Quiet)) { Start-Sleep 20 }
Remove-Item $s1 -Force -ErrorAction SilentlyContinue
"queue1 ended $(Get-Date -Format HH:mm:ss)"
$steps = @(
  @{ tag='zat_g2_100ms'; cmd={ & $ti sprt --candidate "C:/gitProjects/_gate/exes/titanium-0120fd12.exe serve --bz" --baseline "C:/gitProjects/zataxx-bz/target/x86_64-pc-windows-gnu/release/zataxx.exe" --non-main-ab --rules botzone --time-ms 100 --elo1 20 --max-games 400 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\d1230_ti_vs_zat_g2_bz_100ms.log 2>&1 | Select-Object -Last 3 } },
  @{ tag='r2_100ms'; cmd={ & $ti gate --test f4a3ed4 --kind search --rules botzone --base-args "--bz --tt-bits 22" --test-args "--bz --tt-bits 22" --time-ms 100 --max-games 800 --elo1 10 --threads 4 2>&1 | Select-Object -Last 4 } },
  @{ tag='r8_100ms'; cmd={ & $ti gate --test d3c3444 --kind search --rules botzone --base-args "--bz --tt-bits 22" --test-args "--bz --tt-bits 22" --time-ms 100 --max-games 800 --elo1 10 --threads 4 2>&1 | Select-Object -Last 4 } },
  @{ tag='zat_g2_1000ms'; cmd={ & $ti sprt --candidate "C:/gitProjects/_gate/exes/titanium-0120fd12.exe serve --bz" --baseline "C:/gitProjects/zataxx-bz/target/x86_64-pc-windows-gnu/release/zataxx.exe" --non-main-ab --rules botzone --time-ms 1000 --elo1 20 --max-games 200 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\d1230_ti_vs_zat_g2_bz_1000ms.log 2>&1 | Select-Object -Last 3 } }
)
foreach ($s in $steps) {
  if (Test-Path $stop) { "d1231_STOP -> end $(Get-Date -Format HH:mm:ss)"; break }
  $t0 = Get-Date
  "=== $($s.tag) start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  & $s.cmd
  Reports $s.tag $t0
  "=== $($s.tag) done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}
"queue2 end $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
