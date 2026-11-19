# D643/D644 SEARCH PERFORMANCE TEST (the "unit test before the gate", user D644): sample positions across the course of real games,
# search each with the baseline and the candidates for the same wall time, compare nodes, depth and best move. No games are played.
# usage: powershell -File search_perf_test.ps1 [-Games <gate sprt.log.games.jsonl>] [-Out <report.txt>] [-Ms 100] [-NGames 40] [-Reps 1] [-Procs 4] [-Warm] [-Only "name|name|..."]
# Positions run on -Procs parallel worker processes (user: use 4 of the 8 threads); every config runs under the same load.
# Cold mode (default): `uainewgame` before every position (fresh tables, pure unit test). -Warm: tables persist along each game
# (like real play). Order is ABBA over configs, -Reps full passes; the per-position median over passes is used.
# Edit the $cfg table for the configs; the first entry is the baseline.
param(
  [string]$Games = "C:\gitProjects\mainpromo-ataxx\logs\gates\2026-11-09-ac2a8ce8-aB6C5A149-BB6C5A149-vs-0e9efb08-ms100-b40000\sprt.log.games.jsonl",
  [string]$Out = "C:\gitProjects\titanium-ataxx\logs\tree\d644_search_perf.txt",
  [int]$Ms = 100, [int]$NGames = 40, [int]$Reps = 1, [int]$Procs = 4, [switch]$Warm, [string]$Only, [string]$Plies = ""
)
# D672 variant: third cfg element = environment switches ("A=1;B=1") set for that config only (exp/bz-nmp build: TI_BZ, TI_BZNMP)
$new  = 'C:\gitProjects\bznmp-ataxx\target_bz\x86_64-pc-windows-gnu\release\titanium-cli.exe'
$all = [ordered]@{
  'bz rules (Botzone today)' = @($new, '--tt-bits 22', 'TI_BZ=1')
  'bz rules + null move'     = @($new, '--tt-bits 22', 'TI_BZ=1;TI_BZNMP=1')
  'classic rules (gates)'    = @($new, '--tt-bits 22', '')
}
$cfg = [ordered]@{}; foreach ($k in $all.Keys) { if (-not $Only -or ($Only -split '\|') -contains $k) { $cfg[$k] = $all[$k] } }
$plyList = if ($Plies) { @(($Plies -split '[,\s]+') | Where-Object { $_ } | ForEach-Object { [int]$_ }) } else { 6,16,26,36,46,56,66,76 }
# 1. positions (id = index)
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
# 2. chunks: cold = positions round-robin; warm = whole games round-robin (a game stays in order in one chunk)
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
  $pf = "$env:TEMP\spt_chunk_$c.uai"; $txt | Set-Content $pf -Encoding ascii; $chunkFiles += $pf; $chunkIds += ,@($ids)
}
"# search performance test $(Get-Date -Format 'yyyy-MM-dd HH:mm'): $N positions from $used games ($Games), $Ms ms each, mode $(if($Warm){'warm'}else{'cold'}), reps $Reps, $Procs parallel processes" | Set-Content $Out
# 3. run one config over all chunks in parallel; returns an array indexed by position id
$worker = {
  param($exe, $argstr, $file)
  $a = @('serve') + ($argstr -split ' ')
  $res = New-Object System.Collections.Generic.List[object]; $last = $null
  foreach ($line in (Get-Content $file | & $exe @a 2>$null)) {
    if ($line -match '^info depth (\d+) score cp (-?\d+) nodes (\d+) time (\d+)') { $last = @{ d = [int]$matches[1]; s = [int]$matches[2]; n = [int64]$matches[3]; t = [int]$matches[4] } }
    elseif ($line -match '^info depth (\d+) score mate (-?\d+) nodes (\d+) time (\d+)') { $last = @{ d = [int]$matches[1]; s = 99999; n = [int64]$matches[3]; t = [int]$matches[4] } }
    elseif ($line -like 'bestmove*') { if (-not $last) { $last = @{ d = 0; s = 0; n = 0; t = 0 } }; $last.bm = $line.Split(' ')[1]; $res.Add([pscustomobject]$last); $last = $null }
  }
  return ,$res
}
function RunCfg($exe, $argstr, $envs) { $set = @(); foreach ($kv in ($envs -split ';')) { if ($kv) { $p = $kv -split '='; Set-Item "env:$($p[0])" $p[1]; $set += $p[0] } }
  $jobs = for ($c = 0; $c -lt $Procs; $c++) { Start-Job -ScriptBlock $worker -ArgumentList $exe, $argstr, $chunkFiles[$c] }
  $null = $jobs | Wait-Job; foreach ($e in $set) { Remove-Item "env:$e" }
  $out = New-Object object[] $N
  for ($c = 0; $c -lt $Procs; $c++) { $r = @($jobs[$c] | Receive-Job); if ($r.Count -eq 1 -and $r[0] -is [System.Collections.IEnumerable] -and $r[0] -isnot [string]) { $r = @($r[0]) }; if ($r.Count -ne $chunkIds[$c].Count) { Write-Warning "chunk $c : $($r.Count) results for $($chunkIds[$c].Count) positions" }; for ($i = 0; $i -lt [Math]::Min($r.Count, $chunkIds[$c].Count); $i++) { $out[$chunkIds[$c][$i]] = $r[$i] } }
  $jobs | Remove-Job -Force
  return ,$out
}
$keys = @($cfg.Keys); $order = @()
foreach ($r in 1..$Reps) { $order += $keys; $rev = $keys.Clone(); [array]::Reverse($rev); $order += $rev }
$runs = @{}; foreach ($k in $keys) { $runs[$k] = New-Object System.Collections.Generic.List[object] }
foreach ($k in $order) { $r = RunCfg $cfg[$k][0] $cfg[$k][1] $cfg[$k][2]; $runs[$k].Add($r); "ran $k : $(@($r | Where-Object { $_ }).Count)/$N searches" }
# 4. aggregate: per position median across runs
function Med($v) { $s = @($v | Sort-Object); return $s[[int][Math]::Floor($s.Count / 2)] }
$agg = @{}
foreach ($k in $keys) { $agg[$k] = for ($i = 0; $i -lt $N; $i++) { $rs = @($runs[$k] | ForEach-Object { $_[$i] } | Where-Object { $_ }); if ($rs.Count -eq 0) { [pscustomobject]@{ n = 0; d = 0; s = 0; bm = ''; t = 0 } } else { [pscustomobject]@{ n = (Med ($rs | ForEach-Object { $_.n })); d = (Med ($rs | ForEach-Object { $_.d })); s = (Med ($rs | ForEach-Object { $_.s })); bm = $rs[0].bm; t = (Med ($rs | ForEach-Object { $_.t })) } } } }
$base = $keys[0]
foreach ($k in $keys) {
  $a = $agg[$k]; $b = $agg[$base]
  $lr = 0.0; $dd = 0.0; $agree = 0; $ds = 0.0; $nn = 0; $lrs = New-Object System.Collections.Generic.List[double]
  for ($i = 0; $i -lt $N; $i++) {
    if ($b[$i].n -gt 0 -and $a[$i].n -gt 0) { $x = [Math]::Log($a[$i].n / $b[$i].n); $lr += $x; $lrs.Add($x); $nn++ }
    $dd += ($a[$i].d - $b[$i].d); if ($a[$i].bm -eq $b[$i].bm) { $agree++ }
    if ([Math]::Abs($a[$i].s) -lt 5000 -and [Math]::Abs($b[$i].s) -lt 5000) { $ds += ($a[$i].s - $b[$i].s) }
  }
  $mu = $lr / [Math]::Max(1, $nn); $var = 0.0; foreach ($x in $lrs) { $var += ($x - $mu) * ($x - $mu) }; $se = if ($nn -gt 1) { [Math]::Sqrt($var / ($nn - 1) / $nn) } else { 0 }
  $line = "{0,-19} nodes in {1} ms: ratio vs {2} = {3:N3} (+-{9:P1} SE, 95% CI {10:N3}..{11:N3}) | mean depth diff {4:+0.00;-0.00} | same bestmove {5:P0} | mean score diff {6:+0.0;-0.0} cp | mean nodes {7:N0} (n={8})" -f $k, $Ms, $base, [Math]::Exp($mu), ($dd / $N), ($agree / $N), ($ds / $N), (($a | Measure-Object n -Average).Average), $N, $se, [Math]::Exp($mu - 1.96 * $se), [Math]::Exp($mu + 1.96 * $se)
  $line; Add-Content $Out $line
}
Add-Content $Out "--- nodes ratio vs $base by ply (candidate nodes / baseline nodes in the same time)"
foreach ($p in $plyList) {
  $idx = @($pos | Where-Object { $_.ply -eq $p } | ForEach-Object { $_.id }); if ($idx.Count -eq 0) { continue }
  $row = "ply {0,2}: " -f $p
  foreach ($k in $keys) { $s = 0.0; $c = 0; foreach ($i in $idx) { if ($agg[$base][$i].n -gt 0 -and $agg[$k][$i].n -gt 0) { $s += [Math]::Log($agg[$k][$i].n / $agg[$base][$i].n); $c++ } }; $row += "{0}={1:N3}  " -f $k, [Math]::Exp($s / [Math]::Max(1, $c)) }
  $row; Add-Content $Out $row
}



