# D1255: run the R8 gate (exp/semiterm d3c3444 vs main, Botzone 100 ms) as soon as the Modal upload (ti.exe PID 22444) ends;
# ti gate refuses a timed gate while another ti.exe / python.exe runs. Then B12 reports.
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
while (Get-Process -Id 22444 -ErrorAction SilentlyContinue) { Start-Sleep 15 }
"upload ended $(Get-Date -Format HH:mm:ss); gate start"
$t0 = Get-Date
& $ti gate --test d3c3444 --kind search --rules botzone --base-args "--bz --tt-bits 22" --test-args "--bz --tt-bits 22" --time-ms 100 --max-games 800 --elo1 10 --threads 4 2>&1 | Select-Object -Last 8
$gj = Get-ChildItem -Recurse C:\gitProjects\titanium-ataxx\logs\gates -Filter *.games.jsonl | Where-Object { $_.LastWriteTime -gt $t0 } | Sort-Object LastWriteTime | Select-Object -Last 1
if ($gj) {
  & $ti insight $gj.FullName --by-bucket 2>&1 | Out-File -Encoding utf8 C:\gitProjects\titanium-ataxx\logs\gates\reports\d1255_r8_100ms_insight.txt
  & $ti analyze $gj.FullName 2>&1 | Out-File -Encoding utf8 C:\gitProjects\titanium-ataxx\logs\gates\reports\d1255_r8_100ms_analyze.txt
  "reports written from $($gj.FullName)"
}
"done $(Get-Date -Format HH:mm:ss)"
