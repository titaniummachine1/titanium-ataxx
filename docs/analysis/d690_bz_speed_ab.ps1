# D690 Warm Search Performance A/B: arbitrary binaries/flags under InvariantCulture ASCII output
param(
  [string]$Games = "C:\gitProjects\exp-max-nps\logs\gates\2026-11-11-bfec1623-a52849400-vs-dbda6e11-ms100\sprt.log.games.jsonl",
  [string]$Out = "C:\gitProjects\titanium-ataxx\logs\tree\d690_bz_speed1_warm.txt",
  [string]$BaseExe = "C:\gitProjects\_gate\exes\titanium-bfec1623.exe",
  [string]$TestExe = "C:\gitProjects\exp-max-nps\target\x86_64-pc-windows-gnu\release\titanium-cli.exe",
  [int]$Ms = 100, [int]$NGames = 25, [int]$Reps = 2, [int]$Procs = 4, [switch]$Warm, [string]$Plies = "6,16,26,36,46,56,66,76,84"
)
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

$cfg = [ordered]@{
  'bfec classic'   = @($BaseExe, '--tt-bits 22', '')
  'bfec --bz'      = @($BaseExe, '--tt-bits 22 --bz', '')
  'cand --bz'      = @($TestExe, '--tt-bits 22 --bz', '')
}
$plyList = @(($Plies -split '[,\s]+') | Where-Object { $_ } | ForEach-Object { [int]$_ })

$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

# 1. Sample positions
$lines = Get-Content $Games -TotalCount ($NGames * 3)
$pos = New-Object System.Collections.Generic.List[object]; $used = 0
foreach ($l in $lines) {
  if ($used -ge $NGames) { break }
  $g = $l | ConvertFrom-Json
  $mv = @($g.moves | ForEach-Object { $_.move })
  if (@($mv | Where-Object { $_ -notmatch '^[a-g][1-7]([a-g][1-7])?$' }).Count -gt 0) { continue }
  $used++
  foreach ($p in $plyList) {
    if ($p -ge $mv.Count) { continue }
    $line = "position fen $($g.start)"; if ($p -gt 0) { $line += " moves " + ($mv[0..($p - 1)] -join ' ') }
    $pos.Add([pscustomobject]@{ id = $pos.Count; g = $used; ply = $p; line = $line })
  }
}
$N = $pos.Count

# 2. Chunks
$chunkFiles = @(); $chunkIds = @()
for ($c = 0; $c -lt $Procs; $c++) {
  $ids = New-Object System.Collections.Generic.List[int]; $txt = New-Object System.Collections.Generic.List[string]
  $lastG = -1
  foreach ($p in $pos) {
    $mine = if ($Warm) { (($p.g - 1) % $Procs) -eq $c } else { ($p.id % $Procs) -eq $c }
    if (-not $mine) { continue }
    if (-not $Warm -or $p.g -ne $lastG) { $txt.Add('uainewgame'); $lastG = $p.g }
    $txt.Add($p.line); $txt.Add("go movetime $Ms"); $ids.Add($p.id)
  }
  $pf = "$env:TEMP\spt_d690_chunk_$c.uai"; $txt | Set-Content $pf -Encoding ascii; $chunkFiles += $pf; $chunkIds += ,@($ids)
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$hdr = "# D690 search performance test $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss')): $N positions from $used games ($Games), $Ms ms each, mode $(if($Warm){'warm'}else{'cold'}), reps $Reps, $Procs parallel processes`n"
$hdr += "# BaseExe=$BaseExe | TestExe=$TestExe`n"
[System.IO.File]::WriteAllText($Out, $hdr, $utf8NoBom)

# 3. Worker
$worker = {
  param($exe, $argstr, $file)
  $a = @('serve') + ($argstr -split '\s+' | Where-Object { $_ })
  $res = New-Object System.Collections.Generic.List[object]; $last = $null
  foreach ($line in (Get-Content $file | & $exe @a 2>$null)) {
    if ($line -match '^info depth (\d+) score cp (-?\d+) nodes (\d+) time (\d+)') { $last = @{ d = [int]$matches[1]; s = [int]$matches[2]; n = [int64]$matches[3]; t = [int64]$matches[4]; mate = $false } }
    elseif ($line -match '^info depth (\d+) score mate (-?\d+) nodes (\d+) time (\d+)') { $last = @{ d = [int]$matches[1]; s = 99999; n = [int64]$matches[3]; t = [int64]$matches[4]; mate = $true } }
    elseif ($line -like 'bestmove*') { if (-not $last) { $last = @{ d = 0; s = 0; n = 0; t = 0; mate = $false } }; $last.bm = $line.Split(' ')[1]; $res.Add([pscustomobject]$last); $last = $null }
  }
  return ,$res
}
function RunCfg($exe, $argstr, $envs) {
  $set = @()
  foreach ($kv in ($envs -split ';')) { if ($kv) { $p = $kv -split '='; Set-Item "env:$($p[0])" $p[1]; $set += $p[0] } }
  $jobs = for ($c = 0; $c -lt $Procs; $c++) { Start-Job -ScriptBlock $worker -ArgumentList $exe, $argstr, $chunkFiles[$c] }
  $null = $jobs | Wait-Job
  foreach ($e in $set) { Remove-Item "env:$e" }
  $out = New-Object object[] $N
  for ($c = 0; $c -lt $Procs; $c++) {
    $r = @($jobs[$c] | Receive-Job)
    if ($r.Count -eq 1 -and $r[0] -is [System.Collections.IEnumerable] -and $r[0] -isnot [string]) { $r = @($r[0]) }
    if ($r.Count -ne $chunkIds[$c].Count) { Write-Warning "chunk $c : $($r.Count) results for $($chunkIds[$c].Count) positions" }
    for ($i = 0; $i -lt [Math]::Min($r.Count, $chunkIds[$c].Count); $i++) { $out[$chunkIds[$c][$i]] = $r[$i] }
  }
  $jobs | Remove-Job -Force
  return ,$out
}
$keys = @($cfg.Keys); $order = @()
foreach ($r in 1..$Reps) { $order += $keys; $rev = $keys.Clone(); [array]::Reverse($rev); $order += $rev }
$runs = @{}; foreach ($k in $keys) { $runs[$k] = New-Object System.Collections.Generic.List[object] }
foreach ($k in $order) { $r = RunCfg $cfg[$k][0] $cfg[$k][1] $cfg[$k][2]; $runs[$k].Add($r); "ran $k : $(@($r | Where-Object { $_ }).Count)/$N searches" }

# 4. Aggregate
function Med($v) { $s = @($v | Sort-Object); return $s[[int][Math]::Floor($s.Count / 2)] }
$agg = @{}
foreach ($k in $keys) {
  $agg[$k] = for ($i = 0; $i -lt $N; $i++) {
    $rs = @($runs[$k] | ForEach-Object { $_[$i] } | Where-Object { $_ })
    if ($rs.Count -eq 0) { [pscustomobject]@{ n = 0; d = 0; s = 0; bm = ''; t = 0; mate = $false } }
    else { [pscustomobject]@{ n = (Med ($rs | ForEach-Object { $_.n })); d = (Med ($rs | ForEach-Object { $_.d })); s = (Med ($rs | ForEach-Object { $_.s })); bm = $rs[0].bm; t = (Med ($rs | ForEach-Object { $_.t })); mate = $rs[0].mate } }
  }
}

$inv = [System.Globalization.CultureInfo]::InvariantCulture
function FmtLines($baseKey) {
  $sb = New-Object System.Text.StringBuilder
  $b = $agg[$baseKey]
  [void]$sb.AppendLine("--- vs $baseKey ---")
  foreach ($k in $keys) {
    $a = $agg[$k]
    $lr = 0.0; $dd = 0.0; $agree = 0; $ds = 0.0; $nn = 0; $lrs = New-Object System.Collections.Generic.List[double]
    $sumN = 0L; $sumT = 0L; $mates = 0
    $nmSumN = 0L; $nmSumT = 0L
    for ($i = 0; $i -lt $N; $i++) {
      $sumN += [int64]$a[$i].n; $sumT += [int64]$a[$i].t
      if ($a[$i].mate) { $mates++ }
      if ($b[$i].n -gt 0 -and $a[$i].n -gt 0) { $x = [Math]::Log($a[$i].n / $b[$i].n); $lr += $x; $lrs.Add($x); $nn++ }
      if (-not $a[$i].mate) { $nmSumN += [int64]$a[$i].n; $nmSumT += [int64]$a[$i].t }
      $dd += ($a[$i].d - $b[$i].d); if ($a[$i].bm -eq $b[$i].bm) { $agree++ }
      if ([Math]::Abs($a[$i].s) -lt 5000 -and [Math]::Abs($b[$i].s) -lt 5000) { $ds += ($a[$i].s - $b[$i].s) }
    }
    $mu = $lr / [Math]::Max(1, $nn); $var = 0.0; foreach ($x in $lrs) { $var += ($x - $mu) * ($x - $mu) }; $se = if ($nn -gt 1) { [Math]::Sqrt($var / ($nn - 1) / $nn) } else { 0 }
    $knps = $sumN / [Math]::Max(1.0, [double]$sumT)
    $nmKnps = $nmSumN / [Math]::Max(1.0, [double]$nmSumT)
    $avgN = ($a | Measure-Object n -Average).Average
    $line = [string]::Format($inv, "{0,-14} ratio={1:N3} (+-{2:P1} SE, 95% CI {3:N3}..{4:N3}) | agg {5:N1} knps (non-mate {6:N1}, mates {7}) | ddiff {8:+0.00;-0.00} | agree {9:P0} | ds {10:+0.0;-0.0} cp | meanN {11:N0} (n={12})", $k, [Math]::Exp($mu), $se, [Math]::Exp($mu - 1.96 * $se), [Math]::Exp($mu + 1.96 * $se), $knps, $nmKnps, $mates, ($dd / $N), ($agree / $N), ($ds / $N), $avgN, $N)
    [void]$sb.AppendLine($line)
  }
  return $sb.ToString()
}

$textOut = (FmtLines 'bfec classic') + (FmtLines 'bfec --bz')
$sbPly = New-Object System.Text.StringBuilder
[void]$sbPly.AppendLine("--- by ply (ratio vs bfec --bz): geomean nodes, ddiff, agg knps ---")
$b = $agg['bfec --bz']
foreach ($p in $plyList) {
  $idx = @($pos | Where-Object { $_.ply -eq $p } | ForEach-Object { $_.id }); if ($idx.Count -eq 0) { continue }
  $row = [string]::Format($inv, "ply {0,2} (n={1,2}): ", $p, $idx.Count)
  foreach ($k in $keys) {
    $s = 0.0; $c = 0; $ddp = 0.0; $sN = 0L; $sT = 0L; $mc = 0
    foreach ($i in $idx) {
      $sN += [int64]$agg[$k][$i].n; $sT += [int64]$agg[$k][$i].t
      if ($agg[$k][$i].mate) { $mc++ }
      if ($b[$i].n -gt 0 -and $agg[$k][$i].n -gt 0) { $s += [Math]::Log($agg[$k][$i].n / $b[$i].n); $ddp += ($agg[$k][$i].d - $b[$i].d); $c++ }
    }
    $row += [string]::Format($inv, "{0}: ratio={1:N3} ddiff={2:+0.00;-0.00} knps={3:N1} (m={4})  ", $k, [Math]::Exp($s / [Math]::Max(1, $c)), ($ddp / [Math]::Max(1, $c)), ($sN / [Math]::Max(1.0, [double]$sT)), $mc)
  }
  [void]$sbPly.AppendLine($row)
}
$fullOut = $textOut + $sbPly.ToString()
[System.IO.File]::AppendAllText($Out, $fullOut, $utf8NoBom)
$fullOut
