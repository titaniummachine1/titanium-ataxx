# D894 proof that a no-legacy deletion changes nothing: same best move + same node count on 58 real game positions at
# depth 9, classic rules and Botzone rules, vs main's cached gate exe. Usage: nolegacy_parity.ps1 -Exe <candidate exe> -Tag <name>
param([Parameter(Mandatory)][string]$Exe, [Parameter(Mandatory)][string]$Tag)
$f = 'C:\gitProjects\titanium-ataxx\logs\tree\parity58.uai'
$out = 'C:\gitProjects\titanium-ataxx\logs\nolegacy'
Set-Location C:\gitProjects\mainpromo-ataxx
$norm = { param($lines) $lines | ForEach-Object { $_ -replace ' time \d+ nps \d+', '' } }
foreach ($mode in @(@('classic', @()), @('bz', @('--bz')))) {
    $name = $mode[0]; $extra = $mode[1]
    (Get-Content $f) + 'quit' | & $Exe serve @extra 2>$null | Where-Object { $_ -match '^bestmove|nodes' } | Out-File -Encoding ascii "$out\$Tag`_$name.txt"
    $a = & $norm (Get-Content "$out\base_$name.txt")
    $b = & $norm (Get-Content "$out\$Tag`_$name.txt")
    $d = Compare-Object $a $b -SyncWindow 0
    $nodes = ($b | Select-String 'nodes (\d+)' | ForEach-Object { [int64]$_.Matches[0].Groups[1].Value } | Measure-Object -Sum).Sum
    "{0} {1}: lines {2} vs {3}, differences {4}, total nodes {5}" -f $Tag, $name, $a.Count, $b.Count, @($d).Count, $nodes
}
