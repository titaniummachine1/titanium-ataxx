# D690 Bit-Identity & Speed Verification for Botzone Hot-Path Refactors
param(
  [string]$BaseExe = 'C:\gitProjects\_gate\exes\titanium-bfec1623.exe',
  [string]$TestExe = 'C:\gitProjects\exp-max-nps\target\x86_64-pc-windows-gnu\release\titanium-cli.exe',
  [string]$Games = 'C:\gitProjects\exp-max-nps\logs\gates\2026-11-11-bfec1623-a52849400-vs-dbda6e11-ms100\sprt.log.games.jsonl',
  [int]$NGames = 15,
  [string]$Plies = '6,20,40,60,76'
)

$plyList = @($Plies -split ',' | ForEach-Object { [int]$_ })
$lines = Get-Content $Games -TotalCount ($NGames * 2)
$cmds = New-Object System.Collections.Generic.List[string]
$cmds.Add('uai')
$cmds.Add('isready')
$cmds.Add('uainewgame')
$cmds.Add('position startpos')
$cmds.Add('go depth 8')
$cmds.Add('position startpos moves a7b6 g7f7 a1b2 g1f2')
$cmds.Add('go nodes 10000')

$used = 0
foreach ($l in $lines) {
  if ($used -ge $NGames) { break }
  $g = $l | ConvertFrom-Json
  $mv = @($g.moves | ForEach-Object { $_.move })
  if (@($mv | Where-Object { $_ -notmatch '^[a-g][1-7]([a-g][1-7])?$' }).Count -gt 0) { continue }
  $used++
  $cmds.Add('uainewgame')
  foreach ($p in $plyList) {
    if ($p -ge $mv.Count) { continue }
    $pos = "position fen $($g.start)"; if ($p -gt 0) { $pos += " moves " + ($mv[0..($p - 1)] -join ' ') }
    $cmds.Add($pos)
    $cmds.Add('go depth 6')
    $cmds.Add($pos)
    $cmds.Add('go nodes 4000')
  }
}
$cmds.Add('quit')

$uaiFile = "$env:TEMP\d690_parity.uai"
$cmds | Set-Content $uaiFile -Encoding ascii

function RunUai($exe) {
  $out = New-Object System.Collections.Generic.List[string]
  $lastInfo = ''
  foreach ($line in (Get-Content $uaiFile | & $exe serve --bz --tt-bits 22 2>$null)) {
    if ($line -match '^info depth \d+ score ') {
      $lastInfo = ($line -replace ' time \d+ nps \d+', '')
    } elseif ($line -like 'bestmove*') {
      $out.Add("$lastInfo | $line")
      $lastInfo = ''
    }
  }
  return ,$out
}

$a = RunUai $BaseExe
$b = RunUai $TestExe
if ($a.Count -ne $b.Count) {
  Write-Error "Count mismatch: base=$($a.Count) test=$($b.Count)"
  exit 1
}
$diffs = 0
for ($i = 0; $i -lt $a.Count; $i++) {
  if ($a[$i] -ne $b[$i]) {
    if ($diffs -lt 10) { Write-Host "DIFF #$i`n  BASE: $($a[$i])`n  TEST: $($b[$i])" }
    $diffs++
  }
}
Write-Host "Bit-identity check (--bz --tt-bits 22): checked $($a.Count) searches, diffs=$diffs"
if ($diffs -gt 0) { exit 2 }
