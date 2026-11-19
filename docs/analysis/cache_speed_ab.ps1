# D581: speed A/B of TT / eval-cache sizes on a replayed game at the Botzone budget (47 own searches x 500k nodes,
# TT kept all game). Run ONLY on an idle machine (no gates). ABBA order, each config twice; prints wall time per run.
# usage: powershell -File docs\analysis\cache_speed_ab.ps1
$exe = "C:\gitProjects\tree-ataxx\target_default\release\titanium-cli.exe"
$uai = "C:\gitProjects\titanium-ataxx\logs\tree\ttcensus_game.uai"
$out = "C:\gitProjects\titanium-ataxx\logs\tree\cache_speed_ab.txt"
$cfgs = @("--tt-bits 22 --ecache-bits 16", "--tt-bits 22 --ecache-mode 1 --ecache-bits 22", "--tt-mode 1 --tt-bits 22 --ecache-mode 1 --ecache-bits 22", "--tt-mode 1 --tt-bits 22 --ecache-mode 1 --ecache-bits 22", "--tt-bits 22 --ecache-mode 1 --ecache-bits 22", "--tt-bits 22 --ecache-bits 16")
"# cache speed A/B $(Get-Date -Format 'yyyy-MM-dd HH:mm') exe $exe" | Set-Content $out
foreach ($c in $cfgs) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $a = @("serve") + ($c -split " "); $lines = Get-Content $uai | & $exe @a 2>$null
    $sw.Stop()
    $nodes = ($lines | Select-String "nodes (\d+)" | ForEach-Object { [int64]$_.Matches[0].Groups[1].Value } | Measure-Object -Sum).Sum
    $bm = @($lines | Select-String "^bestmove").Count; $line = "[{0}] {1:N1} s for {2} searches of 500k nodes" -f $c, $sw.Elapsed.TotalSeconds, $bm
    $line
    Add-Content $out $line
}
