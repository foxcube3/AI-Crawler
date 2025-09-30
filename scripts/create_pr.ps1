param(
    [string]$Branch = "fix/pyinstaller-build",
    [string]$Base = "main"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating pull request against origin/$Base..." -ForegroundColor Cyan

# Check for git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git not found on PATH. Install Git: https://git-scm.com/download/win"
}

# Check for gh
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "GitHub CLI (gh) not found on PATH. Install: https://github.com/cli/cli/releases"
    Write-Host "After install, run: gh auth login" -ForegroundColor Yellow
    exit 1
}

# Ensure we are in a Git repository
git rev-parse --is-inside-work-tree | Out-Null

# Show remote
Write-Host "Remote 'origin' URL:" -ForegroundColor Cyan
git remote get-url origin

Write-Host "Fetching latest changes..." -ForegroundColor Cyan
git fetch origin

Write-Host "Creating branch $Branch from origin/$Base..." -ForegroundColor Cyan
git checkout -b $Branch "origin/$Base"

Write-Host "Staging changes..." -ForegroundColor Cyan
git add requirements.txt README.md scripts/build_windows.bat .github/pull_request_template.md

Write-Host "Committing..." -ForegroundColor Cyan
git commit -m "Fix PyInstaller build: add dependency, correct module invocation, update README troubleshooting"

Write-Host "Pushing branch to origin..." -ForegroundColor Cyan
git push -u origin $Branch

Write-Host "Creating PR via GitHub CLI..." -ForegroundColor Cyan
gh pr create --fill --base $Base --head $Branch

Write-Host "Pull request created successfully." -ForegroundColor Green