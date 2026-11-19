# D1239: gate the two engine candidates vs main, Botzone rules, 100 ms (B9 first leg), then B12 reports.
#   R2 exp/nonpv-re2 f4a3ed4 (no duplicate non-PV re-search), R8 exp/semiterm d3c3444 (exact <= 2-empty solver).
# Stop between gates: New-Item C:\gitProjects\titanium-ataxx\logs\ext\d1239_STOP
$ErrorActionPreference = 'Continue'
Set-Location C:\gitProjects\titanium-ataxx
$ti  = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$rep = 'C:\gitProjects\titanium-ataxx\logs\gates\reports'
$stop = 'C:\gitProjects\titanium-ataxx\logs\ext\d1239_STOP'
foreach ($c in @(@{tag='r2'; sha='f4a3ed4'}, @{tag='r8'; sha='d3c3444'})) {
  if (Test-Path $stop) { "STOP -> end $(Get-Date -Format HH:mm:ss)"; break }
  $t0 = Get-Date
  "=== $($c.tag) $($c.sha) start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  & $ti gate --test $c.sha --kind search --rules botzone --base-args "--bz --tt-bits 22" --test-args "--bz --tt-bits 22" --time-ms 100 --max-games 800 --elo1 10 --threads 4 2>&1 | Select-Object -Last 6
  $gj = Get-ChildItem -Recurse C:\gitProjects\titanium-ataxx\logs\gates -Filter *.games.jsonl -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt $t0 } | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($gj) {
    "games: $($gj.FullName)"
    & $ti insight $gj.FullName --by-bucket 2>&1 | Out-File -Encoding utf8 "$rep\d1239_$($c.tag)_100ms_insight.txt"
    & $ti analyze $gj.FullName 2>&1 | Out-File -Encoding utf8 "$rep\d1239_$($c.tag)_100ms_analyze.txt"
  } else { "no games.jsonl found under logs/gates" }
  "=== $($c.tag) done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}
"queue end $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
