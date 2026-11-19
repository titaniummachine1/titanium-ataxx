# D886: upload a SmallNet payload to a Docker cloud box and start its jobs. One tar.gz (sbx cp of a directory fails), sha256
# checked on the box against the local hash before anything runs. Source files are only read.
# Usage: powershell -NoProfile -File docs\analysis\snarch_boxsetup.ps1 -Box tit-sne -Set e -Files ti4,nnue_main.tnn,val_v9_clean.tdat,train_v10_snarch1m2.tdat,run_p5.sh
param([string]$Box, [string]$Set, [string]$Files, [string]$Dir = 'C:\gitProjects\titanium-ataxx\scratch\sbx_snarch\p5')
$sbx = "$env:LOCALAPPDATA\DockerSandboxes\bin\sbx.exe"
$list = $Files -split ','
$tgz = Join-Path $Dir "payload_$Set.tgz"
Push-Location $Dir
tar -czf $tgz @list
Pop-Location
$h = (Get-FileHash $tgz -Algorithm SHA256).Hash.ToLower()
"$(Get-Date -Format HH:mm:ss) $Box payload $((Get-Item $tgz).Length) B sha $($h.Substring(0,16))"
for ($try = 1; $try -le 3; $try++) {
    & $sbx --cloud cp $tgz "${Box}:/tmp/payload.tgz" | Out-Null
    $remote = ((& $sbx --cloud exec $Box bash -c "sha256sum /tmp/payload.tgz | cut -c1-64") -join '').Trim()
    if ($remote -eq $h) { break }
    "$(Get-Date -Format HH:mm:ss) $Box try ${try}: sha mismatch / missing (remote '$remote')"
}
if ($remote -ne $h) { "$(Get-Date -Format HH:mm:ss) $Box UPLOAD FAILED"; exit 1 }
& $sbx --cloud exec $Box bash -c "mkdir -p /tmp/sn && tar -xzf /tmp/payload.tgz -C /tmp/sn && cd /tmp/sn && sed -i 's/\r$//' run_p5.sh && sha256sum ti4 nnue_main.tnn val_v9_clean.tdat train_v10_*.tdat | cut -c1-16 && bash run_p5.sh $Set && sleep 20 && pgrep -c -x ti4 && tail -n 2 out/*.log | cut -c1-200"
"$(Get-Date -Format HH:mm:ss) $Box started set $Set"
