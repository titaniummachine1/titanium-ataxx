$jobs = @()
for ($i = 0; $i -lt 4; $i++) {
    $skip = $i * 4500
    $jobs += Start-Job -ScriptBlock {
        param($s, $i)
        $lines = Get-Content C:\gitProjects\titanium-ataxx\data\nnue\align18k_clean.txt | Select-Object -Skip $s -First 4500
        $out = 'C:\gitProjects\titanium-ataxx\data\nnue\align18k_s' + $i + '.txt'
        $err = 'C:\gitProjects\titanium-ataxx\logs\align18k_s' + $i + '.err'
        $lines | & C:\gitProjects\mainwt\target\release\titanium-cli.exe label --nodes 1000000 --tuple C:\gitProjects\mainwt\data\nnue\tupS53r0.tup > $out 2>$err
        Write-Output ("shard " + $i + " done")
    } -ArgumentList $skip, $i
}
Wait-Job $jobs | Out-Null
Receive-Job $jobs
foreach ($j in $jobs) { Remove-Job $j }
Get-Date -Format o | Out-File C:\gitProjects\titanium-ataxx\logs\align18k_done.txt
