param(
  [switch]$Verbose
)

# PowerShell wrapper to run the cmd batch under cmd.exe to avoid PowerShell parsing issues
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Join-Path $scriptDir ".."
Set-Location $repoRoot

$batch = Join-Path $scriptDir "build_windows.bat"
if (-not (Test-Path $batch)) {
  Write-Error "Batch file not found: $batch"
  exit 1
}

$arg = ""
if ($Verbose) { $arg = "--verbose" }

Write-Host "Running build via cmd.exe: $batch $arg"
$cmd = "cmd.exe /c `"$batch $arg`""
Write-Host "Command: $cmd"

# Start the process and stream output
$procInfo = New-Object System.Diagnostics.ProcessStartInfo
$procInfo.FileName = 'cmd.exe'
$procInfo.Arguments = "/c `"$batch $arg`""
$procInfo.RedirectStandardOutput = $true
$procInfo.RedirectStandardError = $true
$procInfo.UseShellExecute = $false
$procInfo.CreateNoWindow = $false

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $procInfo
$proc.Start() | Out-Null

# Stream stdout and stderr
while (-not $proc.HasExited) {
  $outLine = $proc.StandardOutput.ReadLine()
  if ($outLine -ne $null) { Write-Host $outLine }
  Start-Sleep -Milliseconds 10
}

# Drain remaining output
while (($outLine = $proc.StandardOutput.ReadLine()) -ne $null) { Write-Host $outLine }
while (($errLine = $proc.StandardError.ReadLine()) -ne $null) { Write-Host $errLine }

$exit = $proc.ExitCode
if ($exit -ne 0) {
  Write-Host "Build failed with exit code $exit"
  Write-Host "If verbose mode was used, the build log is in %TEMP%\ai_crawler_build.log"
}
else {
  Write-Host "Build finished successfully. Check dist\\AI_Crawler_Assistant_Server.exe"
}

exit $exit
