# D1036 (user 2026-11-17 ~03:50, going to sleep): keep self-play generation going overnight, main engine via the hub,
# 5,000 nodes per move, 4 or 6 threads, whichever generates faster; new rows are merged into training in the morning.
# 1) waits for the D1032 queue ('done' marker in its output) so the timed gates are not disturbed (one long job, A4);
# 2) thread test: one 300-game batch at 4 threads, one at 6 threads (real standard games, kept); games per minute decides;
# 3) the normal resumable loop (docs/analysis/gen_pc_loop.ps1, 2000-game batches) with the faster thread count.
# Stop: create data\gen\pc_run11_5k\STOP (running batch finishes first) or Stop-Process -Name ti_gen.
param([string]$WaitFor = '', [string]$RunDir = 'data\gen\pc_run11_5k')
if ($WaitFor) { while (-not ((Test-Path $WaitFor) -and (Select-String -Path $WaitFor -Pattern '^done ' -Quiet))) { Start-Sleep 30 } }
Set-Location C:\gitProjects\titanium-ataxx
$loop = 'C:\gitProjects\titanium-ataxx\docs\analysis\gen_pc_loop.ps1'
$rate = @{}
foreach ($t in 4, 6) {
    if (Test-Path (Join-Path $RunDir 'STOP')) { "STOP file, leaving"; exit }
    & powershell -NoProfile -File $loop -RunDir $RunDir -Batches 1 -Games 300 -Threads $t
    # time the generator only (run.log 'start' -> 'end' of the last batch; excludes binary copy and gate import)
    $l = Get-Content (Join-Path $RunDir 'run.log')
    $s = ($l | Select-String ' start$' | Select-Object -Last 1).Line.Substring(0, 19)
    $e = ($l | Select-String ' end: ' | Select-Object -Last 1).Line
    $g = [int]([regex]::Match($e, '(\d+) games in the file').Groups[1].Value)
    $min = ([datetime]$e.Substring(0, 19) - [datetime]$s).TotalMinutes
    $rate[$t] = $g / $min
    "thread test: $t threads = {0:N1} games/min ({1} games in {2:N1} min) $(Get-Date -Format HH:mm)" -f $rate[$t], $g, $min
}
$best = if ($rate[6] -gt $rate[4]) { 6 } else { 4 }
"chosen: $best threads $(Get-Date -Format HH:mm)"
& powershell -NoProfile -File $loop -RunDir $RunDir -Batches 60 -Games 2000 -Threads $best
"done $(Get-Date -Format HH:mm)"
