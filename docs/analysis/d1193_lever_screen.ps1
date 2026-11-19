# D1193 (user: 'research the missing levers and the architecture reading them'): every candidate input group on the
# frozen champion (lab standard train_v13, held-out 1,528,091 rows), each to saturation, then PAIRED vs calibration gB
# (lab residual-pair, block SE, D1188). Then self-play generation (D1191). Runs after the D1185 cluster residual (PID 21256).
Set-Location C:\gitProjects\titanium-ataxx
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
while (Get-Process -Id 21256 -ErrorAction SilentlyContinue) { Start-Sleep 30 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_std\x86_64-pc-windows-gnu\release\lab.exe'
$pair = 'C:\gitProjects\titaniumtest\lab\target_pair\x86_64-pc-windows-gnu\release\lab.exe'
$h = 'logs\hce'
$gb = "$h\d1176_res_gB_calibration_weights.json"
function Pair($stem) {
  if (Test-Path "${stem}_weights.json") {
    & $pair residual-pair --a $gb --b "${stem}_weights.json" 2>&1 | Select-String 'rows' | ForEach-Object { Say ("PAIR " + (Split-Path $stem -Leaf) + ": " + $_.Line) }
  } else { Say "MISSING ${stem}_weights.json" }
}
Pair "$h\d1185_res_s10_erosion"
Pair "$h\d1185_res_w3_windows"
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
foreach ($r in $runs) {
  $stem = "$h\d1193_res_$($r[0])"
  if (Test-Path "$stem.log") { Say "skip $($r[0]) (exists)"; Pair $stem; continue }
  Say "$($r[0]) start"
  & $lab hce-train @($r[1]) --buckets 16 --on-nnue 1 --nnue-scale 1 --lr 0.1 --threads 7 --out $stem 2>&1 | Select-String 'SATURATED' | ForEach-Object { Say ("$($r[0]): " + $_.Line) }
  Pair $stem
}
Say 'starting self-play generation (gen_pc_loop.ps1, data\gen\pc_run16_5k, 7 threads, 5000 nodes, clean STOP)'
& powershell -NoProfile -ExecutionPolicy Bypass -File docs\analysis\gen_pc_loop.ps1 -RunDir data\gen\pc_run16_5k -Threads 7 -Batches 60
Say 'queue end'
