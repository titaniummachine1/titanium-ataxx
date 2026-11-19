# D1057 (user: "so weaknesses and soft weaknesses are missing from Titanium?"): split the D1053 g3 group ("ours") into
# its parts, same protocol as D1053 (frozen champion + bias + learned scale + ONE part, sp11_rows, --holdout 0.1 whole
# games, --val-target 1, saturation). Compare each with the calibration baseline gB (0.011658).
# Waits for the D1053 queue ('done' in its output) so only one job runs.
param([string]$WaitFor = '')
if ($WaitFor) { while (-not ((Test-Path $WaitFor) -and (Select-String -Path $WaitFor -Pattern '^done ' -Quiet))) { Start-Sleep 30 } }
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_hyb\x86_64-pc-windows-gnu\release\lab.exe'
$tr = 'data\datasets\v11\selfplay\sp11_rows.tdat'
$va = 'data\nnue\packed\val_v9_clean.tdat'
$runs = @(
  @{ n = 'g3a_scalars';      a = @('--set', '6') },   # material, frontier, private / shared holes, no-move flags
  @{ n = 'g3b_dynpst';       a = @('--set', '7') },   # square class x weakness count (per stone), both sides
  @{ n = 'g3c_softweak';     a = @('--set', '8') }    # soft weaknesses (per stone: enemy stones near its empty squares)
)
foreach ($r in $runs) {
  "$($r.n) start $(Get-Date -Format HH:mm)"
  & $lab hce-train --data $tr --val $va --holdout 0.1 --buckets 16 --on-nnue 1 --nnue-scale 1 --val-target 1 --lr 0.1 --threads 7 --out "logs\hce\d1057_res_$($r.n)" @($r.a) 2>&1 | Select-String 'reference|SATURATED' | ForEach-Object { $_.Line }
}
"done $(Get-Date -Format HH:mm)"
