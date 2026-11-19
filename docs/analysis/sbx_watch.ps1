# D911: idle-kill watcher for Docker cloud boxes (LAWS A9 d: copy results out and delete the box when done, never leave one idle).
# sbx has no idle shutdown (only --ttl), and runs without an epoch cap (C15) cannot be timed in advance, so this watcher is
# started next to EVERY box at launch. Every -EveryMin minutes it checks the box for a busy process; after 2 idle checks in a row
# it copies /tmp/sn/out home with snarch_boxcopy.ps1 (sha-checked), and only if that copy printed OK it deletes the box and
# closes its row in logs/docker_spend.csv (actual = uptime x rate). A failed copy leaves the box alive and retries next round.
# Usage: powershell -File docs/analysis/sbx_watch.ps1 -Box tit-sng -Created "2026-11-16 14:00" -Rate 1.118 -Cpus 16 -Mem 32
#        [-BusyPattern 'ti[0-9]* (train|relabel|gen|sprt|gate)'] [-EveryMin 5]
param(
    [Parameter(Mandatory)][string]$Box,
    [Parameter(Mandatory)][string]$Created,
    [Parameter(Mandatory)][double]$Rate,
    [int]$Cpus = 16, [int]$Mem = 32,
    [string]$BusyPattern = 'ti[0-9]* (train|relabel|gen|sprt|gate|policy2)',
    [int]$EveryMin = 5
)
$sbx = "$env:LOCALAPPDATA\DockerSandboxes\bin\sbx.exe"
$repo = 'C:\gitProjects\titanium-ataxx'
$log = "$repo\logs\sbx_watch_$Box.log"
$t0 = [datetime]::ParseExact($Created, 'yyyy-MM-dd HH:mm', $null)
$inv = [System.Globalization.CultureInfo]::InvariantCulture
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Tee-Object -FilePath $log -Append }
function CloseRow($status) {
    $h = ((Get-Date) - $t0).TotalHours
    $usd = [math]::Round($h * $Rate, 2).ToString($inv)
    $row = "$(Get-Date -Format 'yyyy-MM-dd HH:mm'),$Box,$Cpus,$Mem,$($Rate.ToString($inv)),0,$usd,$status (up $([math]::Round($h,2).ToString($inv))h; sbx_watch D911)"
    Add-Content -Path "$repo\logs\docker_spend.csv" -Value $row -Encoding utf8
    Say "ledger: $row"
}
Say "watching $Box (created $Created, `$$Rate/h, busy = '$BusyPattern', every $EveryMin min)"
$idle = 0
while ($true) {
    $ls = (& $sbx --cloud ls 2>&1) -join "`n"
    if ($ls -notmatch [regex]::Escape($Box)) { Say "$Box is gone (ttl?) before an idle copy"; CloseRow 'GONE before watcher copy (check logs)'; break }
    $state = ((& $sbx --cloud exec $Box bash -c "pgrep -af '$BusyPattern' | grep -v pgrep >/dev/null && echo BUSY || echo IDLE" 2>&1) -join '').Trim()
    if ($state -match 'IDLE') { $idle++ } elseif ($state -match 'BUSY') { $idle = 0 } else { Say "check failed: $state" }
    if ($idle -ge 2) {
        Say "$Box idle twice: copying"
        $out = (& powershell -NoProfile -File "$repo\docs\analysis\snarch_boxcopy.ps1" -Boxes $Box) -join "`n"
        Say $out
        if ($out -match "$Box OK") {
            & $sbx --cloud rm --force $Box 2>&1 | Out-Null
            Say "$Box deleted"
            CloseRow 'DELETED when idle after sha-verified copy'
            break
        }
        Say "copy not verified: box kept, retry next round"
    }
    Start-Sleep ($EveryMin * 60)
}
