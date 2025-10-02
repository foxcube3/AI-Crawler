param(
  [switch]$BuildVerbose,
  [switch]$Native, # Run natively in PowerShell without batch/cmd
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

# Map built-in PowerShell -Verbose to our -BuildVerbose flag if provided
if ($PSBoundParameters.ContainsKey('Verbose') -and $PSBoundParameters['Verbose']) {
  $BuildVerbose = $true
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Join-Path $scriptDir ".."
Set-Location $repoRoot

# Logging
$buildLog = Join-Path $env:TEMP "ai_crawler_build.log"
if ($BuildVerbose) { Write-Host "[VERBOSE] Build log: $buildLog" }

function Invoke-OrFail {
  param(
    [string]$ExePath,
    [string[]]$Args
  )
  if ($BuildVerbose) { Write-Host "[VERBOSE] Command: $ExePath $($Args -join ' ')" }
  $p = Start-Process -FilePath $ExePath -ArgumentList $Args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $buildLog -RedirectStandardError $buildLog
  if ($p.ExitCode -ne 0) {
    Write-Host "Command failed: $ExePath $($Args -join ' ')"
    Get-Content $buildLog | Select-Object -Last 200 | ForEach-Object { Write-Host $_ }
    exit $p.ExitCode
  }
}

if ($Native) {
  # Determine python
  $pyCmd = "python"
  try {
    & $pyCmd -c "import sys" | Out-Null
  } catch {
    Write-Error "Python interpreter not found or not usable on PATH."
    exit 1
  }

  # Create venv
  if ($BuildVerbose) { Write-Host "[VERBOSE] Creating virtual env at .venv" }
  if (Test-Path ".venv") { if ($BuildVerbose) { Write-Host "[VERBOSE] Removing existing .venv" }; Remove-Item -Recurse -Force ".venv" }
  & $pyCmd -m venv ".venv"
  if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create virtual environment."; exit 1 }

  $venvPy = Join-Path ".venv\Scripts" "python.exe"
  if (-not (Test-Path $venvPy)) { Write-Error "Virtual environment python not found at $venvPy"; exit 1 }
  if ($BuildVerbose) { Write-Host "[VERBOSE] Using venv python: $venvPy" }

  # Upgrade pip
  Invoke-OrFail -ExePath $venvPy -Args @("-m","pip","install","--upgrade","pip")

  # Install requirements
  if (-not (Test-Path "requirements.txt")) { Write-Error "requirements.txt not found in project root ($PWD)"; exit 1 }
  Invoke-OrFail -ExePath $venvPy -Args @("-m","pip","install","-r","requirements.txt")

  # Install pyinstaller
  Invoke-OrFail -ExePath $venvPy -Args @("-m","pip","install","pyinstaller")

  # Ensure templates/data present
  if (-not (Test-Path "templates")) { Write-Error "templates directory not found. Ensure templates/ exists."; exit 1 }
  if (-not (Test-Path "data")) { if ($BuildVerbose) { Write-Host "[VERBOSE] Creating data directory" }; New-Item -ItemType Directory -Path "data" | Out-Null }

  # PyInstaller build
  $pyiArgs = @("-m","pyinstaller","--noconfirm","--clean","--onefile","--name","AI_Crawler_Assistant_Server","--add-data","templates;templates","--add-data","data;data","server.py")
  if (Test-Path "installers") { if ($BuildVerbose) { Write-Host "[VERBOSE] Including installers in build" }; $pyiArgs += @("--add-data","installers;installers") }
  Invoke-OrFail -ExePath $venvPy -Args $pyiArgs

  Write-Host "Build complete. Executable located at dist\AI_Crawler_Assistant_Server.exe"
  exit 0
}

# Fallback to existing batch wrapper flow
$batch = Join-Path $scriptDir "build_windows.bat"
if (-not (Test-Path $batch)) {
  Write-Error "Batch file not found: $batch"
  exit 1
}

if ($BuildVerbose) { $env:BUILD_VERBOSE = "1" }

$forwardArgs = @()
if ($BuildVerbose) { $forwardArgs += "--verbose" }
if ($ExtraArgs -and $ExtraArgs.Length -gt 0) { $forwardArgs += $ExtraArgs }
$argLine = ($forwardArgs -join ' ')

Write-Host "Running build via cmd.exe: $batch $argLine"
$cmd = "cmd.exe /c `"$batch $argLine`""
Write-Host "Command: $cmd"

$procInfo = New-Object System.Diagnostics.ProcessStartInfo
$procInfo.FileName = 'cmd.exe'
$procInfo.Arguments = "/c `"$batch $argLine`""
$procInfo.RedirectStandardOutput = $true
$procInfo.RedirectStandardError = $true
$procInfo.UseShellExecute = $false
$procInfo.CreateNoWindow = $false

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $procInfo
$proc.Start() | Out-Null

while (-not $proc.HasExited) {
  $outLine = $proc.StandardOutput.ReadLine()
  if ($outLine -ne $null) { Write-Host $outLine }
  Start-Sleep -Milliseconds 10
}

while (($outLine = $proc.StandardOutput.ReadLine()) -ne $null) { Write-Host $outLine }
while (($errLine = $proc.StandardError.ReadLine()) -ne $null) { Write-Host $errLine }

$exit = $proc.ExitCode
if ($exit -ne 0) {
  Write-Host "Build failed with exit code $exit"
  Write-Host "If verbose mode was used, the build log is in `%TEMP%\\ai_crawler_build.log"
} else {
  Write-Host "Build finished successfully. Check dist\\AI_Crawler_Assistant_Server.exe"
}

exit $exit
