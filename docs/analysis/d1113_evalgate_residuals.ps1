# D1113 (user: "use the basic alpha-beta, without tuning bias, to see directly how the architecture changes strength"):
# frozen champion + learned correction (hce-train --on-nnue residual models) vs the CALIBRATION model gB (champion +
# per-bucket bias/scale only), under lab eval-gate's plain alpha-beta (no TT / pruning / policy / SmallNet), fixed depth,
# 400 book pairs (seats swapped). Residual evaluator parity: lab residual-check reproduces the logged val losses exactly.
# Waits until the D1111 stone-pair screens have finished (one long job per machine).
Set-Location C:\gitProjects\titanium-ataxx
$q = 'C:\gitProjects\logs\d1111_spairs.log'
while (-not ((Test-Path $q) -and (Select-String -Path $q -Pattern '^done' -Quiet))) { Start-Sleep -Seconds 60 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_ext\x86_64-pc-windows-gnu\release\lab.exe'
$gB = 'logs\hce\d1053_res_gB_calibration_weights.json'
$runs = @(
  @('g4_pairs', 'logs\hce\d1053_res_g4_pairs_weights.json', 3),
  @('g4_pairs', 'logs\hce\d1053_res_g4_pairs_weights.json', 4),
  @('g5_all', 'logs\hce\d1053_res_g5_all_weights.json', 4),
  @('g3_ours', 'logs\hce\d1053_res_g3_ours_dynpst_weights.json', 4),
  @('sp1234', 'logs\hce\d1111_res_sp1234_weights.json', 4),
  @('sp1', 'logs\hce\d1111_res_sp1_weights.json', 4)
)
foreach ($r in $runs) {
  if (-not (Test-Path $r[1])) { "skip $($r[0]) (no model)"; continue }
  "$($r[0]) depth $($r[2]) start $(Get-Date -Format HH:mm)"
  & $lab eval-gate --a $r[1] --b $gB --depth $r[2] --pairs 400 --threads 7 2>&1 | Select-String 'eval-gate:|FINAL' | ForEach-Object { $_.Line }
}
"done $(Get-Date -Format HH:mm)"
