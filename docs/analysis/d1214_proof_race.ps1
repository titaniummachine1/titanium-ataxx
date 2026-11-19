# D1210-D1214: build the proof queue (unproven train_v13 positions, >= 40 stones, stones DESC then |cp| DESC) and race
# Titanium (exp/proof c1bc6a2, search = main 0120fd12) vs Zataxx (zataxx-bz e6ca628, Botzone rules aligned) on proofs.
# The levatt training (lab PID given) is SUSPENDED for the race so the timing is clean, and ALWAYS resumed (finally).
param([int]$LabPid = 14964, [int]$PerTier = 20, [int]$RaceMs = 10000, [int]$Threads = 4)
Set-Location C:\gitProjects\titanium-ataxx
Add-Type -Namespace W -Name N -MemberDefinition '[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h); [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
$ti = 'C:\gitProjects\titaniumtest\target_proof\x86_64-pc-windows-gnu\release\ti.exe'
$out = 'data\proofs\v13_hunt'
New-Item -ItemType Directory -Force $out | Out-Null
$p = Get-Process -Id $LabPid -ErrorAction SilentlyContinue
try {
  if ($p) { [W.N]::NtSuspendProcess($p.Handle) | Out-Null; "suspended $LabPid $(Get-Date -Format HH:mm:ss)" }
  & $ti proof-hunt --tdat data\nnue\packed\train_v13 --out $out --race $PerTier --race-ms $RaceMs --threads $Threads 2>&1 | Tee-Object -FilePath "$out\race_stdout.log"
} finally {
  if ($p) { [W.N]::NtResumeProcess($p.Handle) | Out-Null; "resumed $LabPid $(Get-Date -Format HH:mm:ss)" }
}
