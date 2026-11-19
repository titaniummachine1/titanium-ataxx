# D1176 (D1169/D1170 follow-up): does a GLOBAL safe-stone count x phase explain the champion's error?
# Frozen champion 812C2118 + one linear group (lab hce-train --on-nnue 1 --nnue-scale 1, init 0 = exact champion),
# LAB STANDARD (train_v13 whole, last 1,528,091 rows held out, champion target). gB = bias + scale only (calibration
# baseline); s9 = gB + one-hot safe counts own/enemy per bucket; g0 = gataxx terms (material, LINEAR safe diff, edge, reach, cohesion).
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_std\x86_64-pc-windows-gnu\release\lab.exe'
foreach ($r in @(@('gB_calibration','5'), @('s9_safe_counts','9'), @('g0_gataxx_terms','0'))) {
  "$($r[0]) start $(Get-Date -Format HH:mm)"
  & $lab hce-train --set $r[1] --buckets 16 --on-nnue 1 --nnue-scale 1 --lr 0.1 --threads 7 --out "logs\hce\d1176_res_$($r[0])" 2>&1 | Select-String 'STANDARD|reference|SATURATED|agreement' | ForEach-Object { $_.Line }
}
"done $(Get-Date -Format HH:mm)"
