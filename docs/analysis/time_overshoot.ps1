# D662 TIME OVERSHOOT / UTILISATION TEST: how close to the requested movetime does an engine answer, in WALL-CLOCK time as a driver would see it
# (request written -> "bestmove" line read, includes process scheduling and pipe latency), optionally under heavy background CPU load.
# usage: powershell -File time_overshoot.ps1 -Exe <engine exe> [-EngineArgs "--tt-bits 22"] [-Ms 100,1000,4000] [-Counts 40,25,10] [-Load 0|8] -Out <report.txt>
# Reports per movetime: mean wall time and utilisation (mean/T), and the distribution of (wall - T): p50 p90 p99 max, and how many answers
# came later than T - pad for pad = 0, 5, 10, 20 ms (a late answer = a time forfeit risk on Botzone).
param(
  [Parameter(Mandatory = $true)][string]$Exe, [string]$EngineArgs = "--tt-bits 22",
  [int[]]$Ms = @(100, 1000, 4000), [int[]]$Counts = @(40, 25, 10), [int]$Load = 0, [string]$Out = "$env:TEMP\time_overshoot.txt",
  [string]$Games = "C:\gitProjects\mainpromo-ataxx\logs\gates\2026-11-09-ac2a8ce8-aB6C5A149-BB6C5A149-vs-0e9efb08-ms100-b40000\sprt.log.games.jsonl"
)
# positions: plies spread over the game from real gate games
$pos = New-Object System.Collections.Generic.List[string]; $used = 0
foreach ($l in (Get-Content $Games -TotalCount 120)) {
  if ($used -ge 40) { break }
  $g = $l | ConvertFrom-Json; $mv = @($g.moves | ForEach-Object { $_.move })
  if (@($mv | Where-Object { $_ -notmatch '^[a-g][1-7]([a-g][1-7])?$' }).Count -gt 0) { continue }
  $used++
  foreach ($p in 6, 20, 34, 48, 62, 76) { if ($p -lt $mv.Count) { $pos.Add("position fen $($g.start) moves " + ($mv[0..($p - 1)] -join ' ')) } }
}
$busy = @(); if ($Load -gt 0) { 1..$Load | ForEach-Object { $busy += Start-Process powershell -ArgumentList '-NoProfile', '-Command', 'while($true){}' -WindowStyle Hidden -PassThru } }
"# time overshoot $(Get-Date -Format 'yyyy-MM-dd HH:mm') exe $Exe args '$EngineArgs' background busy threads $Load; $($pos.Count) positions available" | Set-Content $Out
try {
  $psi = New-Object Diagnostics.ProcessStartInfo; $psi.FileName = $Exe; $psi.Arguments = "serve $EngineArgs"; $psi.RedirectStandardInput = $true; $psi.RedirectStandardOutput = $true; $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
  $p = [Diagnostics.Process]::Start($psi)   # the serve banner goes to stderr
  for ($k = 0; $k -lt $Ms.Count; $k++) {
    $T = $Ms[$k]; $n = [Math]::Min($Counts[$k], $pos.Count); $over = New-Object System.Collections.Generic.List[double]; $wall = 0.0
    for ($i = 0; $i -lt $n; $i++) {
      $p.StandardInput.WriteLine('uainewgame'); $p.StandardInput.WriteLine($pos[($i * 7) % $pos.Count]); $p.StandardInput.Flush(); Start-Sleep -Milliseconds 150   # let the engine finish the new-game table clear before the clock starts
      $sw = [Diagnostics.Stopwatch]::StartNew(); $p.StandardInput.WriteLine("go movetime $T"); $p.StandardInput.Flush()
      while (($line = $p.StandardOutput.ReadLine()) -ne $null) { if ($line -like 'bestmove*') { break } }
      $sw.Stop(); $w = $sw.Elapsed.TotalMilliseconds; $wall += $w; $over.Add($w - $T)
    }
    $s = @($over | Sort-Object); $pc = { param($q) $s[[Math]::Min($s.Count - 1, [int][Math]::Floor($q * $s.Count))] }
    $late = @(0, 5, 10, 20 | ForEach-Object { $pad = $_; "later than T-${pad}ms: {0}/{1}" -f @($s | Where-Object { $_ -gt -$pad }).Count, $s.Count }) -join ' | '
    $line = "movetime {0,5} ms n={1,3}: mean wall {2,8:N1} ms = {3:P1} of T | wall-T ms: p50 {4:N1} p90 {5:N1} p99 {6:N1} max {7:N1} | {8}" -f $T, $n, ($wall / $n), ($wall / $n / $T), (& $pc 0.5), (& $pc 0.9), (& $pc 0.99), $s[-1], $late
    $line; Add-Content $Out $line
  }
  $p.StandardInput.WriteLine('quit'); if (-not $p.HasExited) { $p.Kill() }
} finally { $busy | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } }
