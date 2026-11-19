# D1116 (user: "maybe use nodes, not time: 1k nodes etc. to compare efficiently"): plain alpha-beta (lab eval-gate, no
# margins / TT / pruning / policy / SmallNet) at EQUAL NODE budgets, 400 book pairs each, after the D1113 queue.
# (1) new net BB9CECE8 vs champion 812C2118: if the new net wins here but lost at 100 ms in the engine (D1114), the
#     engine's fixed cp margins are the cause (D1115), not eval quality. (2) residual corrections at 1k / 5k nodes.
Set-Location C:\gitProjects\titanium-ataxx
$q = 'C:\gitProjects\logs\d1113_evalgate.log'
while (-not ((Test-Path $q) -and (Select-String -Path $q -Pattern '^done' -Quiet))) { Start-Sleep -Seconds 60 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_ext\x86_64-pc-windows-gnu\release\lab.exe'
$new = 'data\nnue\proxy\nn_v11banneal_nocal.tnn'
$champ = 'C:\gitProjects\mainpromo-ataxx\data\nnue\nnue_main.tnn'
$gB = 'logs\hce\d1053_res_gB_calibration_weights.json'
$runs = @(
  @('net BB9CECE8 vs champion', $new, $champ, '--nodes', 1000),
  @('net BB9CECE8 vs champion', $new, $champ, '--nodes', 5000),
  @('net BB9CECE8 vs champion', $new, $champ, '--depth', 4),
  @('g4 pairs vs gB', 'logs\hce\d1053_res_g4_pairs_weights.json', $gB, '--nodes', 1000),
  @('g4 pairs vs gB', 'logs\hce\d1053_res_g4_pairs_weights.json', $gB, '--nodes', 5000)
)
foreach ($r in $runs) {
  if (-not (Test-Path $r[1])) { "skip $($r[0]) (no model)"; continue }
  "$($r[0]) $($r[3]) $($r[4]) start $(Get-Date -Format HH:mm)"
  & $lab eval-gate --a $r[1] --b $r[2] $r[3] $r[4] --pairs 400 --threads 7 2>&1 | Select-String 'eval-gate:|FINAL' | ForEach-Object { $_.Line }
}
"done $(Get-Date -Format HH:mm)"
