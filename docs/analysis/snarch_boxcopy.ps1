# D859: copy /tmp/sn/out of each SmallNet research box home. Method: tar on the box, then `sbx --cloud cp` of that ONE file
# (sbx cp of a directory fails with "no response received"; base64 over exec fails above ~1 MB). Verifies sha256 on both ends;
# writes logs/snarch/sbx_<box>/out_<HHmm>.tgz and extracts it into logs/snarch/sbx_<box>/out.
# Usage: powershell -File docs/analysis/snarch_boxcopy.ps1 [-At HH:mm] [-Boxes tit-sna,...]   (-At: wait until that local time)
param([string]$At = '', [string[]]$Boxes = @('tit-sna', 'tit-snb', 'tit-snc'))
$sbx = "$env:LOCALAPPDATA\DockerSandboxes\bin\sbx.exe"
if ($At) { $t = [datetime]::ParseExact($At, 'HH:mm', $null); while ((Get-Date) -lt $t) { Start-Sleep 30 } }
foreach ($n in $Boxes) {
    $short = $n -replace '^tit-', ''
    $d = "C:\gitProjects\titanium-ataxx\logs\snarch\sbx_$short"
    New-Item -ItemType Directory -Force $d | Out-Null
    $f = "$d\out_$(Get-Date -Format 'HHmm').tgz"
    for ($try = 1; $try -le 3; $try++) {
        $remote = ((& $sbx --cloud exec $n bash -c "cd /tmp/sn && tar czf /tmp/sn_out.tgz out && sha256sum /tmp/sn_out.tgz | cut -c1-64") -join '').Trim()
        & $sbx --cloud cp "${n}:/tmp/sn_out.tgz" $f | Out-Null
        if (-not (Test-Path $f)) { "$(Get-Date -Format HH:mm:ss) $n try ${try}: no file"; continue }
        $local = (Get-FileHash $f -Algorithm SHA256).Hash.ToLower()
        if ($local -eq $remote) {
            tar -xzf $f -C $d
            "$(Get-Date -Format HH:mm:ss) $n OK $((Get-Item $f).Length) B sha $($local.Substring(0,16))"
            break
        }
        "$(Get-Date -Format HH:mm:ss) $n try ${try}: sha mismatch local $local remote $remote"
    }
}
