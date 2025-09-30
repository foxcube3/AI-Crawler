@echo off
setlocal

REM Create a pull request against origin/main with the prepared changes.

REM Check for git
where git >nul 2>&1
if errorlevel 1 (
  echo git not found on PATH. Install Git and retry: https://git-scm.com/download/win
  exit /b 1
)

REM Check for GitHub CLI (gh)
where gh >nul 2>&1
if errorlevel 1 (
  echo GitHub CLI (gh) not found on PATH.
  echo - Install from https://github.com/cli/cli/releases and then run:
  echo   gh auth login
  exit /b 1
)

REM Ensure we are in a Git repository
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo Not inside a Git repository. Please run this script from your repository root.
  exit /b 1
)

REM Show current remote
echo Checking remote 'origin'...
git remote get-url origin
if errorlevel 1 (
  echo Remote 'origin' not configured. Please set it with:
  echo   git remote add origin https://github.com/<owner>/<repo>.git
  exit /b 1
)

REM Ensure we have latest main
echo Fetching latest changes...
git fetch origin
if errorlevel 1 (
  echo Failed to fetch from origin.
  exit /b 1
)

REM Create branch
set "BRANCH=fix/pyinstaller-build"
echo Creating branch %BRANCH%...
git checkout -b %BRANCH% origin/main
if errorlevel 1 (
  echo Failed to create branch based on origin/main.
  exit /b 1
)

REM Stage files
echo Staging changes...
git add requirements.txt README.md scripts\build_windows.bat .github\pull_request_template.md
if errorlevel 1 (
  echo Failed to stage files.
  exit /b 1
)

REM Commit
echo Committing changes...
git commit -m "Fix PyInstaller build: add dependency, correct module invocation, update README troubleshooting"
if errorlevel 1 (
  echo Commit failed. If there were no changes, ensure files are updated and try again.
  exit /b 1
)

REM Push
echo Pushing branch to origin...
git push -u origin %BRANCH%
if errorlevel 1 (
  echo Push failed. Check your permissions to the remote.
  exit /b 1
)

REM Create PR via GitHub CLI
echo Creating pull request to base 'main'...
gh pr create --fill --base main --head %BRANCH%
if errorlevel 1 (
  echo PR creation via gh failed. You can open the PR in the browser:
  echo   https://github.com/(owner)/(repo)/compare/main...%BRANCH%
  exit /b 1
)

echo Pull request created successfully.
endlocal