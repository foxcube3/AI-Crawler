@echo off
setlocal

REM Build Windows executable for AI Crawler Assistant (FastAPI server)
REM Requirements: Python 3.10+, pip, and internet access to install dependencies

REM Prefer the Python launcher (py -3) to avoid Microsoft Store alias issues
set "PY_CMD="
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 -c "import sys" >nul 2>&1
  if %ERRORLEVEL%==0 set "PY_CMD=py -3"
)

REM Fallback to python if usable
if not defined PY_CMD (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    python -c "import sys" >nul 2>&1
    if %ERRORLEVEL%==0 set "PY_CMD=python"
  )
)

if not defined PY_CMD (
  echo Python interpreter not found or not usable on PATH.
  echo - Install Python 3.10+ from https://www.python.org/downloads/windows/
  echo - Or disable the "App execution aliases" for python/python3 if they point to Microsoft Store:
  echo   Settings ^> Apps ^> Advanced app settings ^> App execution aliases.
  exit /b 1
)

REM Create venv
set "VENV_DIR=.venv"
%PY_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 (
  echo Failed to create virtual environment.
  exit /b 1
)

REM Use the venv's python explicitly for all subsequent steps
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo Virtual environment python not found at "%VENV_PY%".
  exit /b 1
)

REM Activate venv (optional, but keeps PATH consistent for subprocesses)
call "%VENV_DIR%\Scripts\activate"

REM Upgrade pip and install requirements
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo Failed to upgrade pip.
  exit /b 1
)

"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  exit /b 1
)

REM Install PyInstaller
"%VENV_PY%" -m pip install pyinstaller
if errorlevel 1 (
  echo Failed to install PyInstaller.
  exit /b 1
)

REM Ensure templates are bundled with the executable
set "TEMPLATES=templates"
if not exist "%TEMPLATES%" (
  echo templates directory not found. Ensure templates/ exists.
  exit /b 1
)

REM Ensure data directory exists to include in bundle
set "DATA_DIR=data"
if not exist "%DATA_DIR%" (
  echo data directory not found. Creating empty data directory.
  mkdir "%DATA_DIR%"
)

REM Include installers directory if present
set "INSTALLERS_DIR=installers"
if exist "%INSTALLERS_DIR%" (
  set PYI_INSTALLERS=--add-data "installers;installers"
) else (
  echo installers directory not found. Skipping installers inclusion.
  set PYI_INSTALLERS=
)

REM Clean previous build artifacts
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist AI_Crawler_Assistant_Server.spec del /q AI_Crawler_Assistant_Server.spec

REM Build onefile executable for server.py using venv python
"%VENV_PY%" -m pyinstaller --noconfirm --clean --onefile --name "AI_Crawler_Assistant_Server" ^
  --add-data "templates;templates" ^
  --add-data "data;data" ^
  %PYI_INSTALLERS% ^
  server.py

if errorlevel 1 (
  echo PyInstaller build failed.
  exit /b 1
)

echo Build complete. Executable located at dist\AI_Crawler_Assistant_Server.exe
endlocal