$hb = Get-Content logs\heartbeat.txt -ErrorAction SilentlyContinue | Select-Object -Last 1
$stale = $true
if ($hb) {
  try {
    $age = ((Get-Date) - [datetime]$hb).TotalMinutes
    if ($age -lt 60) { $stale = $false }
  } catch { $stale = $true }
}
if (-not $stale) { exit 0 }
# No agent action for 60+ min: run one bounded deep-label batch (200 pos @1M).
$st = Get-Content logs\watchdog_queue.txt
$board = $st[0]; $off = [int]$st[1]
$lines = Get-Content $board
if ($off -ge $lines.Count) { exit 0 }
$n = [Math]::Min(200, $lines.Count - $off)
$tag = (Get-Date -Format "yyyyMMdd_HHmm")
$lines | Select-Object -Skip $off -First $n | & scripts\bench\graft_label.exe label --nodes 1000000 > ("data\nnue\watchdog_" + $tag + ".txt") 2>logs\watchdog_err.txt
Set-Content logs\watchdog_queue.txt ($board + "`n" + ($off + $n))
Get-Date -Format o | Out-File logs\heartbeat.txt
"watchdog labeled $n rows from $board offset $off" | Out-File logs\watchdog.log -Append
