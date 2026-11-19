param(
    [string]$GamesPath = "C:\gitProjects\exp-max-nps\logs\gates\2026-11-11-bfec1623-a52849400-vs-dbda6e11-ms100\sprt.log.games.jsonl",
    [string]$AnalyzePath = "C:\gitProjects\exp-max-nps\logs\gates\reports\d687_bfec1623_bz_100ms_analyze.txt"
)

$ErrorActionPreference = "Stop"

# 1. Analyze-file summary (blind walks vs seen mates)
$analyzeRows = Get-Content $AnalyzePath | Select-Object -Skip 1
$wTotal = 0; $lTotal = 0; $wNoMate = 0; $lNoMate = 0
$wNoMateCandSawMate = 0; $wNoMateBasePos = 0; $lNoMateBaseSawMate = 0
$wSeats = @{ x = 0; o = 0 }; $lSeats = @{ x = 0; o = 0 }
$wDisExamples = New-Object System.Collections.Generic.List[string]
$lNoMateExamples = New-Object System.Collections.Generic.List[string]

foreach ($line in $analyzeRows) {
    if (-not $line.Trim()) { continue }
    $parts = $line -split '\|'
    if ($parts.Count -lt 5) { continue }
    $head = ($parts[0].Trim() -split '\s+')
    $seat = $head[2]
    $res = $head[3]
    $loserSaw = $parts[2].Trim()
    $loserMate = $parts[3].Trim()
    $big = $parts[4].Trim()

    if ($res -eq 'W') {
        $wTotal++
        $wSeats[$seat]++
        if ($loserMate -eq '-') {
            $wNoMate++
            if ($big -match '(-?\d+)\s+vs\s+(-?\d+)') {
                $s1 = [int]$matches[1]; $s2 = [int]$matches[2]
                if ($s1 -gt 19000 -or $s2 -lt -19000) { $wNoMateCandSawMate++ }
                if (($s2 -lt -19000 -and $s1 -ge -50) -or ($s1 -gt 19000 -and $s2 -ge -50)) { $wNoMateBasePos++ }
            }
            if ($wDisExamples.Count -lt 10) { $wDisExamples.Add($line.Trim()) }
        }
    } elseif ($res -eq 'L') {
        $lTotal++
        $lSeats[$seat]++
        if ($loserMate -eq '-') {
            $lNoMate++
            if ($big -match '(-?\d+)\s+vs\s+(-?\d+)') {
                $s1 = [int]$matches[1]; $s2 = [int]$matches[2]
                if ($s1 -gt 19000 -or $s2 -lt -19000) { $lNoMateBaseSawMate++ }
            }
            if ($lNoMateExamples.Count -lt 10) { $lNoMateExamples.Add($line.Trim()) }
        }
    }
}

Write-Output "=== ANALYZE BREAKDOWN ($($wTotal + $lTotal) games) ==="
Write-Output ("Candidate W: {0} (as x: {1}, as o: {2}) | Loser (baseline) NEVER proved mate ('-'): {3}/{0} ({4:P1})" -f $wTotal, $wSeats['x'], $wSeats['o'], $wNoMate, ($wNoMate / [Math]::Max(1,$wTotal)))
Write-Output ("  of W with '-' baseline mate: cand proved +-1999x in biggest-disagreement = {0}; baseline still >= -50 cp when cand saw mate = {1}" -f $wNoMateCandSawMate, $wNoMateBasePos)
Write-Output ("Candidate L: {0} (as x: {1}, as o: {2}) | Loser (candidate --bz) NEVER proved mate ('-'): {3}/{0} ({4:P1})" -f $lTotal, $lSeats['x'], $lSeats['o'], $lNoMate, ($lNoMate / [Math]::Max(1,$lTotal)))
Write-Output ("  of L with '-' cand mate: base proved +-1999x in biggest-disagreement = {0}" -f $lNoMateBaseSawMate)
Write-Output "Examples of L where cand had '-':"
$lNoMateExamples | ForEach-Object { Write-Output "  $_" }

# 2. Raw games JSONL breakdown
$games = Get-Content $GamesPath | ForEach-Object { $_ | ConvertFrom-Json }
$pairs = @{}
$pliesW = New-Object System.Collections.Generic.List[int]
$pliesL = New-Object System.Collections.Generic.List[int]
$marginW = New-Object System.Collections.Generic.List[int]
$marginL = New-Object System.Collections.Generic.List[int]
$wipeoutW = 0; $wipeoutL = 0
$sumLt49W = 0; $sumLt49L = 0
$maxPlies = 0; $ge400 = 0
$candEndMateW = 0; $candEndMateL = 0
$baseEndMateW = 0; $baseEndMateL = 0
$candBlindTerminalL = 0; $baseBlindTerminalW = 0

# Phase stats
$phases = @(
    @{ Name='ply 0-19'; Lo=0; Hi=19 },
    @{ Name='ply 20-39'; Lo=20; Hi=39 },
    @{ Name='ply 40-59'; Lo=40; Hi=59 },
    @{ Name='ply 60-79'; Lo=60; Hi=79 },
    @{ Name='ply 80+'; Lo=80; Hi=999 }
)
$pStats = @{}
foreach ($ph in $phases) {
    $pStats[$ph.Name] = @{
        candNodes = 0L; candMs = 0L; candDepth = 0L; candN = 0L
        baseNodes = 0L; baseMs = 0L; baseDepth = 0L; baseN = 0L
    }
}

foreach ($g in $games) {
    $pairId = [string]$g.pair
    if (-not $pairs.ContainsKey($pairId)) { $pairs[$pairId] = 0.0 }
    $pairs[$pairId] += [double]$g.candidate_points

    $nMoves = $g.moves.Count
    if ($nMoves -gt $maxPlies) { $maxPlies = $nMoves }
    if ($nMoves -ge 400) { $ge400++ }

    $fx = [int]$g.final_x; $fo = [int]$g.final_o
    $cx = if ($g.candidate_seat -eq 'x') { $fx } else { $fo }
    $bx = if ($g.candidate_seat -eq 'x') { $fo } else { $fx }

    $candLast = $null; $baseLast = $null
    foreach ($mv in $g.moves) {
        if ($mv.mover -eq 'candidate') { $candLast = $mv } else { $baseLast = $mv }
        foreach ($ph in $phases) {
            if ($mv.ply -ge $ph.Lo -and $mv.ply -le $ph.Hi) {
                $st = $pStats[$ph.Name]
                if ($mv.mover -eq 'candidate') {
                    $st.candNodes += [int64]$mv.nodes; $st.candMs += [int64]$mv.ms; $st.candDepth += [int64]$mv.depth; $st.candN++
                } else {
                    $st.baseNodes += [int64]$mv.nodes; $st.baseMs += [int64]$mv.ms; $st.baseDepth += [int64]$mv.depth; $st.baseN++
                }
                break
            }
        }
    }

    if ($g.candidate_points -eq 1) {
        $pliesW.Add($nMoves)
        $marginW.Add($cx - $bx)
        if ($bx -eq 0) { $wipeoutW++ }
        if (($fx + $fo) -lt 49) { $sumLt49W++ }
        if ($candLast -and $null -ne $candLast.mate -and $candLast.mate -gt 0) { $candEndMateW++ }
        if ($baseLast -and $null -ne $baseLast.mate -and $baseLast.mate -lt 0) { $baseEndMateW++ }
        if ($baseLast -and $null -eq $baseLast.mate) { $baseBlindTerminalW++ }
    } elseif ($g.candidate_points -eq 0) {
        $pliesL.Add($nMoves)
        $marginL.Add($cx - $bx)
        if ($cx -eq 0) { $wipeoutL++ }
        if (($fx + $fo) -lt 49) { $sumLt49L++ }
        if ($candLast -and $null -ne $candLast.mate -and $candLast.mate -lt 0) { $candEndMateL++ }
        if ($baseLast -and $null -ne $baseLast.mate -and $baseLast.mate -gt 0) { $baseEndMateL++ }
        if ($candLast -and $null -eq $candLast.mate) { $candBlindTerminalL++ }
    }
}

$ww = @($pairs.Values | Where-Object { $_ -eq 2.0 }).Count
$wd = @($pairs.Values | Where-Object { $_ -eq 1.5 }).Count
$dl = @($pairs.Values | Where-Object { $_ -eq 1.0 }).Count
$ld = @($pairs.Values | Where-Object { $_ -eq 0.5 }).Count
$ll = @($pairs.Values | Where-Object { $_ -eq 0.0 }).Count

Write-Output ""
Write-Output "=== GAME & TERMINAL STRUCTURE ($($games.Count) games, $($pairs.Count) pairs) ==="
Write-Output ("Pentanomial [LL, LD, WL/DD, WD, WW] = [{0}, {1}, {2}, {3}, {4}]" -f $ll, $ld, $dl, $wd, $ww)
Write-Output ("Plies: max={0}, >=400={1} | W mean={2:N1} | L mean={3:N1}" -f $maxPlies, $ge400, ($pliesW | Measure-Object -Average).Average, ($pliesL | Measure-Object -Average).Average)
Write-Output ("Margins: W mean=+{0:N1}, wipeout(bx=0)={1}, sum<49={2} | L mean={3:N1}, wipeout(cx=0)={4}, sum<49={5}" -f ($marginW | Measure-Object -Average).Average, $wipeoutW, $sumLt49W, ($marginL | Measure-Object -Average).Average, $wipeoutL, $sumLt49L)
Write-Output ("On last move of winner/loser:")
Write-Output ("  In W (213): cand last move saw +mate={0}; base last move saw -mate={1}, base last move had NO mate (blind to terminal)={2}" -f $candEndMateW, $baseEndMateW, $baseBlindTerminalW)
Write-Output ("  In L (187): base last move saw +mate={0}; cand last move saw -mate={1}, cand last move had NO mate (blind to terminal)={2}" -f $baseEndMateL, $candEndMateL, $candBlindTerminalL)

Write-Output ""
Write-Output "=== PHASE SPEED & DEPTH (candidate --bz vs baseline dbda6e11) ==="
foreach ($ph in $phases) {
    $st = $pStats[$ph.Name]
    if ($st.candN -eq 0 -or $st.baseN -eq 0) { continue }
    $cknps = ($st.candNodes / [Math]::Max(1.0, [double]$st.candMs))
    $bknps = ($st.baseNodes / [Math]::Max(1.0, [double]$st.baseMs))
    $cd = $st.candDepth / [double]$st.candN
    $bd = $st.baseDepth / [double]$st.baseN
    $cnm = $st.candNodes / [double]$st.candN
    $bnm = $st.baseNodes / [double]$st.baseN
    Write-Output ("{0,-10}: cand {1,6:N1} knps ({2:P1}), d={3:N2}, nodes/mv={4,7:N0} | base {5,6:N1} knps, d={6:N2}, nodes/mv={7,7:N0} | d_diff={8:+0.00;-0.00}" -f $ph.Name, $cknps, ($cknps/$bknps - 1.0), $cd, $cnm, $bknps, $bd, $bnm, ($cd - $bd))
}
