# D1225 (user order 2026-11-18 18:18): Wait for all 47-stone positions in v13_hunt
# (qidx 0..453220 = 24 at 48st + 453,197 at 47st) to complete, stop proof-hunt cleanly via STOP file,
# log 47-stone totals, then launch self-play game generation (gen_pc_loop.ps1 -> data\gen\pc_run16_5k, 7 threads, 5000 nodes).
Set-Location C:\gitProjects\titanium-ataxx
$log = 'C:\gitProjects\logs\d1225_47_then_gen.log'
New-Item -ItemType Directory -Force 'C:\gitProjects\logs' | Out-Null
function Say($m) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m"
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Output $line
}

$results = 'C:\gitProjects\titanium-ataxx\data\proofs\v13_hunt\results.tsv'
$huntStop = 'C:\gitProjects\titanium-ataxx\data\proofs\v13_hunt\STOP'
$genDir = 'data\gen\pc_run16_5k'
$genStop = Join-Path $genDir 'STOP'
$huntPid = 2896

Say "watcher start: waiting for 47-stone tier to complete (target qidx >= 453221 across all 7 workers)"
while ($true) {
    $tiAlive = [bool](Get-Process -Id $huntPid -ErrorAction SilentlyContinue)
    if (-not $tiAlive) {
        Say "ti.exe PID $huntPid already exited"
        break
    }
    if (Test-Path $results) {
        $tail = Get-Content $results -Tail 60 -ErrorAction SilentlyContinue
        if ($tail) {
            $n46 = 0
            $maxQ = -1
            foreach ($l in $tail) {
                $f = $l.Split("`t")
                if ($f.Count -ge 12 -and $f[3] -match '^\d+$' -and $f[-1] -match '^\d+$') {
                    if ([int]$f[3] -le 46) { $n46++ }
                    $q = [int]$f[-1]
                    if ($q -gt $maxQ) { $maxQ = $q }
                }
            }
            if ($n46 -ge 15 -and $maxQ -ge 453235) {
                Say "47-stone tier complete (tail n46=$n46, maxQ=$maxQ): creating $huntStop"
                New-Item -ItemType File -Path $huntStop -Force | Out-Null
                break
            }
        }
    }
    Start-Sleep -Seconds 15
}

for ($i = 0; $i -lt 120; $i++) {
    $p = Get-Process -Id $huntPid -ErrorAction SilentlyContinue
    if (-not $p) { break }
    Start-Sleep -Milliseconds 500
}
Start-Sleep -Seconds 1
Remove-Item $huntStop -ErrorAction SilentlyContinue
Remove-Item $genStop -ErrorAction SilentlyContinue

if (Test-Path $results) {
    $exact47 = 0; $ded47 = 0; $kept47 = 0; $n46done = 0
    $sr = [System.IO.StreamReader]::new($results)
    $null = $sr.ReadLine()
    while (-not $sr.EndOfStream) {
        $f = $sr.ReadLine().Split("`t")
        if ($f.Count -lt 11) { continue }
        $st = [int]$f[3]
        $pr = if ($f[5] -in @('zataxx','terminal','solver48','titanium')) { $f[7] } else { $f[10] }
        if ($st -ge 47) {
            if ($pr -in @('1','2','5')) { $exact47++ }
            elseif ($pr -eq '4') { $ded47++ }
            else { $kept47++ }
        } else {
            $n46done++
        }
    }
    $sr.Close()
    $tot47 = $exact47 + $ded47 + $kept47
    Say ("47+48 stone summary: total=$tot47 (exact=$exact47, deduced=$ded47, kept=$kept47); overflow 46st=$n46done")
}

Say "launching gen_pc_loop.ps1 -> $genDir (7 threads, 5000 nodes, 60 batches)"
& powershell -NoProfile -ExecutionPolicy Bypass -File docs\analysis\gen_pc_loop.ps1 -RunDir $genDir -Threads 7 -Batches 60
Say "gen_pc_loop finished"
