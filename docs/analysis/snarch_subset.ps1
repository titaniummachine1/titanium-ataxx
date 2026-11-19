# D840: write the EXACT rows `ti train snarch` reads (trainer.rs load_val: blocks of 1024 rows at (i*nblk/blocks)*1024) into a
# small new .tdat, so a cloud sandbox builds the SAME samples as the PC without uploading the whole dataset. Source untouched.
#   powershell -NoProfile -File docs\analysis\snarch_subset.ps1 -Src data\nnue\packed\train_v10.tdat -Rows 1200000 -Out OUT.tdat
param([string]$Src, [int]$Rows, [string]$Out)
$REC = 48; $BLOCK = 1024
$fin = [System.IO.File]::OpenRead((Resolve-Path $Src))
$total = [int64]($fin.Length / $REC)
$blocks = [Math]::Max(1, [Math]::Floor($Rows / $BLOCK))
$nblk = [Math]::Floor($total / $BLOCK)
$n = [Math]::Min($blocks, [Math]::Max(1, $nblk))
$fout = [System.IO.File]::Create($Out)
$buf = New-Object byte[] ($BLOCK * $REC)
for ($i = 0; $i -lt $n; $i++) {
    $start = [int64]([Math]::Floor($i * $nblk / $blocks)) * $BLOCK
    $cnt = [Math]::Min($BLOCK, $total - $start)
    [void]$fin.Seek($start * $REC, 'Begin')
    $got = $fin.Read($buf, 0, $cnt * $REC)
    $fout.Write($buf, 0, $got)
}
$fout.Close(); $fin.Close()
"$Out : $n blocks, $((Get-Item $Out).Length / $REC) rows from $total"

