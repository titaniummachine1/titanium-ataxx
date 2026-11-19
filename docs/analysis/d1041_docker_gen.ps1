# D1041 (user 2026-11-17 ~04:00): "use 2 x 32 vCPU Docker boxes for about 3 hours to accelerate game generation, then
# automatically pull the games and close the instances; the PC keeps generating after updating its DAG for novelty".
# User ~04:25: "auto-sync from the instances so it sees their batches" -> every SyncMin minutes the finished 500-game
# batches of each box are pulled (sha-checked) into data\gen\docker_inbox\<box>\; the local loop (gen_pc_loop.ps1
# -Inbox) imports them into the PC DAG before each of its batches (ti dag seq-add, one DAG writer at a time).
# Inputs in $Stage: ti (Linux hub with --main-fingerprint / --shard), fp.txt (ti gen-fingerprint on the PC),
# seq_dag.tgz (DAG snapshot taken while no generator ran), d1041_box_gen.sh (LF). Shards: PC 0/3, box 1 = 1/3, box 2 = 2/3.
# Per box: create (ttl 200 min, self-deletes) -> ledger row -> upload -> start detached -> identity check of batch 1 ->
# sbx_watch.ps1 (deletes an idle box after a sha-checked copy) -> sync every SyncMin -> STOP at RunMin (the running
# batch finishes) -> final sync -> delete -> ledger row. Then, once the inbox is imported, the local loop restarts
# WITHOUT shard (full novelty over the merged DAG).
param(
    [string]$Stage = 'C:\gitProjects\titanium-ataxx\data\gen\docker_stage',
    [string[]]$Boxes = @('tit-gen1', 'tit-gen2'),
    [int]$Cpus = 32, [int]$MemGb = 64, [double]$Rate = 2.236, [int]$TtlMin = 200, [int]$RunMin = 170, [int]$SyncMin = 30,
    [string]$LocalRunDir = 'data\gen\pc_run12_5k',    # the sharded local loop (0/3)
    [string]$NextRunDir = 'data\gen\pc_run13_5k',     # the unsharded local loop after the boxes
    [int]$LocalThreads = 6
)
$sbx = "$env:LOCALAPPDATA\DockerSandboxes\bin\sbx.exe"
$repo = 'C:\gitProjects\titanium-ataxx'
$inbox = "$repo\data\gen\docker_inbox"
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$log = "$repo\logs\d1041_docker_gen.log"
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Tee-Object -FilePath $log -Append }
function X($box, $cmd) { ((& $sbx --cloud exec $box bash -c $cmd 2>&1) -join "`n").Trim() }
function Sync($b) {
    # finished batches not yet synced -> one tgz -> sha check -> extract into the inbox -> mark synced on the box
    # (no double quotes inside sbx exec arguments: PowerShell 5.1 does not escape them for native programs)
    $list = X $b 'cd /tmp/sn/out 2>/dev/null && touch synced.txt && for x in $(grep -oE ''batch_[0-9]+ end rc 0'' run.log | cut -d'' '' -f1); do grep -qx $x synced.txt || echo $x; done | tr ''\n'' '' '''
    $names = @($list -split '\s+' | Where-Object { $_ -match '^batch_\d+$' })
    if ($names.Count -eq 0) { Say "$b sync: nothing new"; return }
    $files = ($names | ForEach-Object { "$_.tgam $_.tbst $_.log $_.engine.json" }) -join ' '
    $r = X $b "cd /tmp/sn/out && tar czf /tmp/sync.tgz $files && sha256sum /tmp/sync.tgz | cut -c1-64"
    $d = "$inbox\$b"; New-Item -ItemType Directory -Force $d | Out-Null
    $f = "$repo\data\gen\docker_stage\sync_$b.tgz"; Remove-Item $f -ErrorAction SilentlyContinue
    & $sbx --cloud cp "${b}:/tmp/sync.tgz" $f 2>&1 | Out-Null
    if ((Test-Path $f) -and ((Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -eq $r)) {
        tar -xzf $f -C $d
        X $b ("cd /tmp/sn/out && printf '%s\n' " + ($names -join ' ') + " >> synced.txt") | Out-Null
        Say "$b sync OK: $($names -join ', ')"
    } else { Say "$b sync FAILED (remote sha $r); retry next round" }
}
Set-Location $repo
New-Item -ItemType Directory -Force $inbox | Out-Null

$started = @{}
for ($i = 0; $i -lt $Boxes.Count; $i++) {
    $b = $Boxes[$i]; $k = $i + 1
    $created = Get-Date -Format 'yyyy-MM-dd HH:mm'
    Say "create $b ($Cpus vCPU, $MemGb GB, `$$Rate/h, ttl $TtlMin min, shard $k/3)"
    & $sbx --cloud create --name $b --image-ref docker.io/library/python:3.12 --cpus $Cpus --memory "${MemGb}g" --ttl "${TtlMin}m" --on-timeout delete --platform linux/amd64 2>&1 | ForEach-Object { Say "  $_" }
    if (-not ((& $sbx --cloud ls 2>&1) -join "`n" -match [regex]::Escape($b))) { Say "$b NOT created, skipping"; continue }
    $worst = [math]::Round($Rate * $TtlMin / 60, 2).ToString($inv)
    Add-Content "$repo\logs\docker_spend.csv" "$created,$b,$Cpus,$MemGb,$($Rate.ToString($inv)),$TtlMin,$worst,CREATED (D1041 self-play generation shard $k/3; sbx_watch + d1041_docker_gen.ps1)" -Encoding utf8
    Say (X $b 'mkdir -p /tmp/sn && echo ok')
    Push-Location $Stage
    foreach ($f in 'ti', 'fp.txt', 'd1041_box_gen.sh', 'seq_dag.tgz') {
        for ($t = 1; $t -le 3; $t++) {
            & $sbx --cloud cp $f "${b}:/tmp/sn/$f" 2>&1 | Out-Null
            $rs = X $b "sha256sum /tmp/sn/$f | cut -c1-64"
            if ($rs -eq (Get-FileHash $f -Algorithm SHA256).Hash.ToLower()) { Say "$b upload $f OK"; break }
            Say "$b upload $f try $t failed ($rs)"
        }
    }
    Pop-Location
    Say (X $b "cd /tmp/sn && (nohup setsid bash d1041_box_gen.sh $k $Cpus $b > job.log 2>&1 < /dev/null &) ; sleep 2; echo started")
    $started[$b] = @{ t = Get-Date; created = $created }
    # D1044: Start-Process joins ArgumentList with spaces WITHOUT quoting, so "2026-11-17 04:42" split into two args and
    # '04:42' became sbx_watch's BusyPattern (boxes judged idle and deleted after 13 min); quote values with spaces.
    Start-Process powershell -ArgumentList @('-NoProfile', '-File', "`"$repo\docs\analysis\sbx_watch.ps1`"", '-Box', $b, '-Created', "`"$created`"", '-Rate', $Rate.ToString($inv), '-Cpus', $Cpus, '-Mem', $MemGb, '-BusyPattern', '"ti gen"') -WindowStyle Hidden
    Say "$b job started, sbx_watch started"
}
if ($started.Count -eq 0) { Say 'no box started; done'; exit }
Start-Sleep 240
$chk = 'grep -h -E ''engine = current main|REFUSED|FAILED|games [(]'' /tmp/sn/out/batch_0001.log | tail -2'
foreach ($b in $started.Keys) { $o = X $b $chk; Say "$b batch 1: $o" }
$t0 = ($started.Values | ForEach-Object { $_.t } | Sort-Object | Select-Object -First 1)
$next = (Get-Date).AddMinutes($SyncMin)
while ((Get-Date) -lt $t0.AddMinutes($RunMin)) {
    if ((Get-Date) -ge $next) { foreach ($b in $started.Keys) { Sync $b; $o = X $b 'tail -1 /tmp/sn/out/run.log'; Say "$b progress: $o" }; $next = (Get-Date).AddMinutes($SyncMin) }
    Start-Sleep 60
}
foreach ($b in $started.Keys) { $o = X $b 'touch /tmp/sn/STOP && echo ok'; Say "$b STOP: $o" }
# wait for the running batch to finish (no ti process), final sync, delete
foreach ($b in $started.Keys) {
    for ($w = 0; $w -lt 20; $w++) { if ((X $b 'pgrep -f ''ti gen'' >/dev/null && echo BUSY || echo IDLE') -match 'IDLE') { break }; Start-Sleep 60 }
    Sync $b
    if ((& $sbx --cloud ls 2>&1) -join "`n" -match [regex]::Escape($b)) {
        & $sbx --cloud rm --force $b 2>&1 | Out-Null
        $h = ((Get-Date) - [datetime]::ParseExact($started[$b].created, 'yyyy-MM-dd HH:mm', $null)).TotalHours
        Add-Content "$repo\logs\docker_spend.csv" "$(Get-Date -Format 'yyyy-MM-dd HH:mm'),$b,$Cpus,$MemGb,$($Rate.ToString($inv)),0,$([math]::Round($h * $Rate, 2).ToString($inv)),DELETED after final sync (D1041; up $([math]::Round($h,2).ToString($inv))h)" -Encoding utf8
        Say "$b deleted (up $([math]::Round($h,2)) h)"
    } else { Say "$b already gone (sbx_watch); its copy is in logs\snarch\sbx_$($b -replace '^tit-','')\out" }
}
# anything sbx_watch copied that the syncs missed goes to the inbox too
foreach ($b in $started.Keys) {
    $w = "$repo\logs\snarch\sbx_$($b -replace '^tit-','')\out"
    if (Test-Path $w) { foreach ($g in Get-ChildItem $w -Filter 'batch_*.tgam') { $dst = "$inbox\$b\$($g.Name)"; if (-not (Test-Path $dst)) { Copy-Item "$w\$($g.BaseName).*" "$inbox\$b\" ; Say "$b from sbx_watch copy: $($g.Name)" } } }
}
# wait until the local loop has imported the whole inbox, then restart it without shard
while (@(Get-ChildItem $inbox -Recurse -Filter 'batch_*.tgam' | Where-Object { -not (Test-Path "$($_.FullName).imported") }).Count -gt 0) { Start-Sleep 60 }
Say 'inbox imported; restarting the local loop without shard'
New-Item -ItemType File -Force "$LocalRunDir\STOP" | Out-Null
while (Get-Process ti_gen -ErrorAction SilentlyContinue) { Start-Sleep 20 }
Start-Process powershell -ArgumentList @('-NoProfile', '-File', "$repo\docs\analysis\gen_pc_loop.ps1", '-RunDir', $NextRunDir, '-Batches', '60', '-Games', '2000', '-Threads', $LocalThreads) -WindowStyle Hidden
Say 'done'
