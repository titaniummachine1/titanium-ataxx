# D1204 (user: attention first): replaces d1203_queue (PID 1760). When the running lever run l1 (lab PID 2364) ends:
# stop the lever-screen queue 16120 and any lab job it just started (a fresh run killed within seconds is renamed
# *.killed, never deleted), pair l1, run LEVER ATTENTION v1 (attention arm, control arm), then the remaining lever runs,
# then remove data\gen\pc_run16_5k\STOP and start self-play generation.
Set-Location C:\gitProjects\titanium-ataxx
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
while (Get-Process -Id 2364 -ErrorAction SilentlyContinue) { Start-Sleep 10 }
Stop-Process -Id 16120 -Force -ErrorAction SilentlyContinue
Start-Sleep 2
Get-CimInstance Win32_Process -Filter "Name='lab.exe'" | Where-Object { $_.CommandLine -match 'hce-train|residual-pair' } | ForEach-Object { Say "stopping lab PID $($_.ProcessId): $($_.CommandLine.Substring([Math]::Max(0,$_.CommandLine.Length-80)))"; Stop-Process -Id $_.ProcessId -Force }
$lab = 'C:\gitProjects\titaniumtest\lab\target_std\x86_64-pc-windows-gnu\release\lab.exe'
$pair = 'C:\gitProjects\titaniumtest\lab\target_pair\x86_64-pc-windows-gnu\release\lab.exe'
$lev = 'C:\gitProjects\titaniumtest\lab\target_lev\x86_64-pc-windows-gnu\release\lab.exe'
$h = 'logs\hce'
$gb = "$h\d1176_res_gB_calibration_weights.json"
function Pair($stem) {
  if (Test-Path "${stem}_weights.json") {
    & $pair residual-pair --a $gb --b "${stem}_weights.json" 2>&1 | Select-String 'rows' | ForEach-Object { Say ("PAIR " + (Split-Path $stem -Leaf) + ": " + $_.Line) }
  } else { Say "MISSING ${stem}_weights.json" }
}
$runs = @(
  @('l1_funes_regions', @('--set','1')),
  @('l2_raw_pst',       @('--set','2')),
  @('l3_ours_all',      @('--set','3')),
  @('l6_ours_scalars',  @('--set','6')),
  @('l7_dynamic_pst',   @('--set','7')),
  @('l8_soft_weak',     @('--set','8')),
  @('l_win2',           @('--set','5','--win','2')),
  @('l_gataxx_pairs',   @('--set','5','--pairs','1','--no-mob','1')),
  @('l_spairs_d1',      @('--set','5','--spairs','1')),
  @('l_spairs_d2',      @('--set','5','--spairs','2')),
  @('l_spairs_d3',      @('--set','5','--spairs','3'))
)
# a run that was started and killed above has a log but no SATURATED line: keep it as *.killed and redo the run
foreach ($r in $runs) {
  $stem = "$h\d1193_res_$($r[0])"
  if ((Test-Path "$stem.log") -and -not (Select-String -Path "$stem.log" -Pattern 'SATURATED' -Quiet)) { Move-Item "$stem.log" "$stem.log.killed" -Force; Say "partial $($r[0]) kept as .killed" }
}
Pair "$h\d1193_res_l1_funes_regions"
foreach ($arm in @(@('d1203_levatt_attn', @()), @('d1203_levatt_ctrl', @('--no-attn')))) {
  Say "levatt-train $($arm[0]) start"
  & $lev levatt-train @($arm[1]) --threads 7 --out "$h\$($arm[0])" 2>&1 | Select-String 'step 0|epoch [0-9]+:|SATURATED|RESTART' | ForEach-Object { Say ("$($arm[0]): " + $_.Line) }
}
foreach ($r in $runs) {
  $stem = "$h\d1193_res_$($r[0])"
  if (Test-Path "$stem.log") { continue }
  Say "$($r[0]) start"
  & $lab hce-train @($r[1]) --buckets 16 --on-nnue 1 --nnue-scale 1 --lr 0.1 --threads 7 --out $stem 2>&1 | Select-String 'SATURATED' | ForEach-Object { Say ("$($r[0]): " + $_.Line) }
  Pair $stem
}
Remove-Item data\gen\pc_run16_5k\STOP -ErrorAction SilentlyContinue
Say 'starting self-play generation (gen_pc_loop.ps1, data\gen\pc_run16_5k, 7 threads, 5000 nodes, clean STOP)'
& powershell -NoProfile -ExecutionPolicy Bypass -File docs\analysis\gen_pc_loop.ps1 -RunDir data\gen\pc_run16_5k -Threads 7 -Batches 60
Say 'queue end'
