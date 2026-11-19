# D1022: Zataxx i16 overflow audit of a net6.bin-format file (2916x256 fw, 256 fb, 256 ow, 1 ob; i16 LE).
#  (a) output product: AVX2 keeps 16 bits of clamp(h,0,255)*ow -> needs |ow| <= 128
#  (b) accumulator: h = fb + sum over 36 tuples of fw[tuple*81 + state] in i16 -> worst case per neuron
#      = |fb| + sum over tuples of max over the 81 states of |fw| (a bound; real positions may stay below it)
# -Fix OUT writes a NEW file with ow clamped to +-127 (the input file is never changed).
param([Parameter(Mandatory)][string]$Net, [string]$Fix = '')
$b = [IO.File]::ReadAllBytes($Net)
$H = 256; $NF = 2916 * $H
$ow = 0..($H - 1) | ForEach-Object { [BitConverter]::ToInt16($b, 2 * ($NF + $H + $_)) }
$bad = @($ow | Where-Object { [Math]::Abs($_) -gt 128 })
$w = New-Object 'int[]' ($NF)
for ($i = 0; $i -lt $NF; $i++) { $w[$i] = [Math]::Abs([int][BitConverter]::ToInt16($b, 2 * $i)) }
$worst = 0; $over = 0
for ($k = 0; $k -lt $H; $k++) {
  $s = [Math]::Abs([int][BitConverter]::ToInt16($b, 2 * ($NF + $k)))
  for ($t = 0; $t -lt 36; $t++) { $m = 0; $base = $t * 81 * $H + $k; for ($st = 0; $st -lt 81; $st++) { $v = $w[$base + $st * $H]; if ($v -gt $m) { $m = $v } }; $s += $m }
  if ($s -gt $worst) { $worst = $s }; if ($s -gt 32767) { $over++ }
}
"{0}: output weights |ow|>128: {1} (max {2}) | accumulator worst-case bound {3} (i16 max 32767), neurons that could wrap: {4}" -f (Split-Path $Net -Leaf), $bad.Count, (($ow | ForEach-Object { [Math]::Abs($_) }) | Measure-Object -Maximum).Maximum, $worst, $over
if ($Fix) {
  $o = [byte[]]$b.Clone(); $n = 0
  for ($k = 0; $k -lt $H; $k++) { $p = 2 * ($NF + $H + $k); $v = [BitConverter]::ToInt16($b, $p); $c = [Math]::Max(-127, [Math]::Min(127, [int]$v)); if ($c -ne $v) { $n++; $x = [BitConverter]::GetBytes([int16]$c); $o[$p] = $x[0]; $o[$p + 1] = $x[1] } }
  [IO.File]::WriteAllBytes($Fix, $o)
  "wrote $Fix ($n output weights clamped to +-127), sha $((Get-FileHash $Fix).Hash.Substring(0,8))"
}
