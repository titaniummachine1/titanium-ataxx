# D1156 (user order 2026-11-18): when the 128-hidden 3x3 net (lab jat3-train d1152_jat3_128) saturates, start
# overnight self-play generation so the dataset grows. One long job per machine: this waits for the training to end.
#   1. wait until logs/hce/d1152_jat3_128.log says SATURATED, or the lab process is gone (crash / stop);
#   2. lab jat3-compare on the 128 net's best weights (1-2 min): loss by empties band vs the champion, blend, move agreement;
#   3. self-play generation = docs/analysis/gen_pc_loop.ps1 (current main 0120fd12 via the canonical hub, 5,000 nodes per
#      move, sequence DAG data/dag/seq_dag.db, binary .tgam/.tbst), run dir data/gen/pc_run15_5k, 7 threads, until
#      60 batches or a STOP file.
# Stop: create data\gen\pc_run15_5k\STOP (the running batch finishes first) or Stop-Process -Name ti_gen; this script:
# Stop-Process -Id <PID in KEY_FACTS RUNNING NOW>.
Set-Location C:\gitProjects\titanium-ataxx
$log = 'C:\gitProjects\logs\d1156_overnight_queue.log'
function Say($s) { Add-Content -Path $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $s" -Encoding utf8 }
$tl = 'logs\hce\d1152_jat3_128.log'
Say 'queue start: waiting for d1152_jat3_128 to saturate'
while ($true) {
  $sat = (Test-Path $tl) -and (Select-String -Path $tl -Pattern 'SATURATED' -Quiet)
  $alive = [bool](Get-Process lab -ErrorAction SilentlyContinue)
  if ($sat -or -not $alive) { break }
  Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 20   # let the trainer write its json / agreement line
foreach ($l in (Select-String -Path $tl -Pattern 'SATURATED|best weights|^epoch' | Select-Object -Last 4)) { Say ("training: " + $l.Line) }
if (-not (Select-String -Path $tl -Pattern 'SATURATED' -Quiet)) { Say 'WARNING: the training ended WITHOUT a SATURATED line (crash or stop); generation starts anyway' }
while (Get-Process lab -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 10 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_std\x86_64-pc-windows-gnu\release\lab.exe'
Say 'jat3-compare on the 128 net'
& $lab jat3-compare --model logs\hce\d1152_jat3_128_jat3.bin 2>&1 | Out-File -FilePath logs\arch\d1156_jat3_128_compare.txt -Encoding utf8
foreach ($l in (Get-Content logs\arch\d1156_jat3_128_compare.txt | Select-Object -Last 8)) { Say ("compare: " + $l) }
# D1157: the one quiet window of the night: per-group timing of the champion's feature extraction (lab feat-bench)
Say 'feat-bench on the quiet PC (200,000 positions x 5 rounds)'
& 'C:\gitProjects\titaniumtest\lab\target_fb\x86_64-pc-windows-gnu\release\lab.exe' feat-bench --n 200000 --rounds 5 2>&1 | Out-File -FilePath logs\arch\d1157_feat_bench.txt -Encoding utf8
foreach ($l in (Get-Content logs\arch\d1157_feat_bench.txt)) { Say ("feat-bench: " + $l) }
Say 'starting self-play generation (gen_pc_loop.ps1, data\gen\pc_run15_5k, 7 threads, 5000 nodes)'
& powershell -NoProfile -ExecutionPolicy Bypass -File docs\analysis\gen_pc_loop.ps1 -RunDir data\gen\pc_run15_5k -Threads 7 -Batches 60
Say 'generation loop ended'
