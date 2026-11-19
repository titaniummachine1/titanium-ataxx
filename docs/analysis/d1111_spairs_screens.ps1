# D1111 / ARCH HUNT queue step 1a: stone pairs by Chebyshev distance band as a residual on the FROZEN champion
# (D1053 protocol, same rows / holdout / calibration as docs/analysis/d1053_residual_groups.ps1; baseline gB 0.011658).
# Waits until the D1108 bundle gate has finished (one long job per machine), then runs bands 1, 2, 3, 4 (=4..6) and 1-4.
Set-Location C:\gitProjects\titanium-ataxx
$drv = 'C:\gitProjects\logs\gate_d1108_driver.log'
while (-not (Select-String -Path $drv -Pattern '^done' -Quiet)) { Start-Sleep -Seconds 60 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_ext\x86_64-pc-windows-gnu\release\lab.exe'
$tr = 'data\datasets\v11\selfplay\sp11_rows.tdat'
$va = 'data\nnue\packed\val_v9_clean.tdat'
foreach ($b in @('1', '2', '3', '4', '1,2,3,4')) {
  $n = 'sp' + ($b -replace ',', '')
  "$n start $(Get-Date -Format HH:mm)"
  & $lab hce-train --data $tr --val $va --holdout 0.1 --buckets 16 --on-nnue 1 --nnue-scale 1 --val-target 1 --lr 0.1 --threads 7 --set 5 --spairs $b --out "logs\hce\d1111_res_$n" 2>&1 | Select-String 'reference|SATURATED|agreement' | ForEach-Object { $_.Line }
}
"done $(Get-Date -Format HH:mm)"
