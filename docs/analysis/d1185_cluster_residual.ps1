# D1185 (user): 3x3 clusters + inside-pieces (erosion) masks as features. Frozen champion 812C2118 + one linear group,
# lab standard (train_v13 whole, held-out 1,528,091 rows), compare with the calibration gB 0.011786 (D1183).
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_std\x86_64-pc-windows-gnu\release\lab.exe'
foreach ($r in @(@('s10_erosion', @('--set','10')), @('w3_windows', @('--set','5','--win','3')))) {
  "$($r[0]) start $(Get-Date -Format HH:mm)"
  & $lab hce-train @($r[1]) --buckets 16 --on-nnue 1 --nnue-scale 1 --lr 0.1 --threads 7 --out "logs\hce\d1185_res_$($r[0])" 2>&1 | Select-String 'STANDARD|reference|SATURATED|agreement|weights' | ForEach-Object { $_.Line }
}
"done $(Get-Date -Format HH:mm)"
