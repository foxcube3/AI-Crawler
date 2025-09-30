# Quickstart PowerShell script for Windows users
# - Creates venv, installs requirements, runs the FastAPI server
# - Shows example API calls via Invoke-RestMethod
# Usage:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#   .\scripts\quickstart.ps1

param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "Checking Python..." -ForegroundColor Cyan
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    Write-Error "Python not found on PATH. Install Python 3.10+ and ensure 'python' or 'py' is available."
}

$venvDir = ".venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
& $pythonCmd.Path -m venv $venvDir
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
$env:VIRTUAL_ENV = (Resolve-Path $venvDir).Path
$env:Path = "$($env:VIRTUAL_ENV)\Scripts;$env:Path"

Write-Host "Upgrading pip and installing requirements..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Starting server on port $Port..." -ForegroundColor Cyan
$env:PORT = "$Port"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python server.py" -WindowStyle Normal
Start-Sleep -Seconds 3

Write-Host "Opening Web UI..." -ForegroundColor Cyan
Start-Process "http://localhost:$Port/ui"

# Example API calls
Write-Host "`nExample: Crawl by query" -ForegroundColor Yellow
$crawlBody = @{
    query = "large language models retrieval"
    output_dir = "data"
    max_results = 5
    crawl_depth = 1
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/crawl" -ContentType "application/json" -Body $crawlBody | Format-List

Write-Host "`nExample: Build index" -ForegroundColor Yellow
$indexBody = @{ output_dir = "data"; model_name = "sentence-transformers/all-MiniLM-L6-v2" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/build-index" -ContentType "application/json" -Body $indexBody | Format-List

Write-Host "`nExample: Ask question" -ForegroundColor Yellow
$askBody = @{ output_dir = "data"; question = "What are best practices for web crawling?"; top_k = 5; synthesize = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/ask" -ContentType "application/json" -Body $askBody | Format-List

Write-Host "`nExample: Generate report" -ForegroundColor Yellow
$reportBody = @{ output_dir = "data"; question = "Summarize main findings"; top_k = 5; synthesize = $true; page_size = 50; theme = "light" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/report" -ContentType "application/json" -Body $reportBody | Format-List

Write-Host "`nQuickstart complete." -ForegroundColor Green