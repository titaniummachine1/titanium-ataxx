# D1032 (user: 'ablate what the pair tables approximate to cut the speed cost'): pruned HCE retrains, grafted from D991 (C14),
# then speed + per-node and timed gates vs D991 inside Zataxx's search (C:/gitProjects/zataxx-ab/hcew, --hce FILE).
# Waits for the D1030 gate queue to finish (marker 'done' in its output) so only one job runs at a time.
param([string]$WaitFor = '')
if ($WaitFor) { while (-not ((Test-Path $WaitFor) -and (Select-String -Path $WaitFor -Pattern '^done ' -Quiet))) { Start-Sleep 30 } }
$lab = 'C:\gitProjects\titaniumtest\lab\target_ext\x86_64-pc-windows-gnu\release\lab.exe'
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$h = 'C:\gitProjects\titanium-ataxx\logs\hce'
$d = 'C:\gitProjects\titanium-ataxx\data\nnue\packed'
$base = "$h\d991_hcetrain_b32_pairs_weights.json"
$z = 'C:\gitProjects\zataxx-ab\hcew\target\x86_64-pc-windows-gnu\release\zataxx.exe'
$keep44 = '0,1,2,3,4,5,6,7,8,11,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,34,35,36,37,40,41,42,43,44,45,46,47,48,63,67,76'
$keep24 = '0,1,2,3,4,7,16,19,20,21,22,23,24,25,28,29,30,35,36,41,42,43,46,47'
$runs = @(
  @{ n = 'd1032_p80_nomob'; a = @('--no-mob', '1') },
  @{ n = 'd1032_p44_nomob'; a = @('--no-mob', '1', '--pair-keep', $keep44) },
  @{ n = 'd1032_p24_nomob'; a = @('--no-mob', '1', '--pair-keep', $keep24) }
)
Set-Location C:\gitProjects\titanium-ataxx
foreach ($r in $runs) {
  "train $($r.n) start $(Get-Date -Format HH:mm)"
  & $lab hce-train --data "$d\train_v10.tdat" --val "$d\val_v9_clean.tdat" --set 4 --buckets 32 --pairs 1 --init $base --threads 7 --out "$h\$($r.n)" @($r.a) 2>&1 | Select-String 'SATURATED|agreement' | ForEach-Object { $_.Line }
  & $lab hce-eval --weights "$h\$($r.n)_weights.json" --val "$d\val_v9_clean.tdat" --n 200000 --bench 1 | Select-String 'bench' | ForEach-Object { $_.Line }
}
"d991 bench: $(& $lab hce-eval --weights $base --val "$d\val_v9_clean.tdat" --n 200000 --bench 1 | Select-String 'bench')"
foreach ($r in $runs) {
  foreach ($lim in @(@('--nodes', '5000'), @('--time-ms', '100'))) {
    $tag = if ($lim[0] -eq '--nodes') { '5k' } else { '100ms' }
    "gate $($r.n) $tag start $(Get-Date -Format HH:mm)"
    & $ti sprt --candidate "$z --hce $h\$($r.n)_weights.json" --baseline "$z --hce $base" --non-main-ab --rules libataxx @lim --elo1 20 --max-games 400 --threads 4 --out "C:\gitProjects\titanium-ataxx\logs\ext\zat_$($r.n)_vs_d991_$tag.log" 2>&1 | Select-String 'Games :|Elo   :|knps' | ForEach-Object { $_.Line }
  }
}
"done $(Get-Date -Format HH:mm)"
