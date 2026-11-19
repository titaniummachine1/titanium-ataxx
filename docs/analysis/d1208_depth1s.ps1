# D1208: suspend the levatt training (lab PID given) so the depth test runs on a quiet PC, run it, ALWAYS resume.
param([int]$LabPid = 14964)
Add-Type -Namespace W -Name N -MemberDefinition '[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h); [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
$p = Get-Process -Id $LabPid
try { [W.N]::NtSuspendProcess($p.Handle) | Out-Null; "suspended $LabPid $(Get-Date -Format HH:mm:ss)"
  & 'C:\Program Files\Git\bin\bash.exe' docs/analysis/d1208_depth1s.sh }
finally { [W.N]::NtResumeProcess($p.Handle) | Out-Null; "resumed $LabPid $(Get-Date -Format HH:mm:ss)" }
