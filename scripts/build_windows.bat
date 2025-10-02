@echo off
setlocal

REM Build Windows executable for AI Crawler Assistant (FastAPI server)
REM Requirements: Python 3.10+, pip, and internet access to install dependencies

REM Verbose logging: set BUILD_VERBOSE=1 in environment or pass --verbose / -v as first arg
set "VERBOSE=0"
if /I "%~1"=="--verbose" set "VERBOSE=1"
if /I "%~1"=="-v" set "VERBOSE=1"
if defined BUILD_VERBOSE set "VERBOSE=1"
if "%VERBOSE%"=="1" (
  echo [VERBOSE] Verbose mode enabled
  echo [VERBOSE] Args: %*
)

REM Build log (captures pip/pyinstaller output to avoid confusing the batch parser)
set "BUILD_LOG=%TEMP%\ai_crawler_build.log"
if "%VERBOSE%"=="1" echo [VERBOSE] Build log: %BUILD_LOG%

REM python
if not defined PY_CMD (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    call python -c "import sys" >nul 2>&1
    if %ERRORLEVEL%==0 set "PY_CMD=python"
  )
)

REM Check that Python can import a standard library module (os)
if defined PY_CMD (
  call %PY_CMD% -c "import os" >nul 2>&1
  if errorlevel 1 (
    echo Python installation is broken or missing standard libraries.
    echo Please reinstall Python from https://www.python.org/downloads/windows/
    exit /b 1
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
REM Change working directory to repository root (script is in scripts/)
if "%VERBOSE%"=="1" echo [VERBOSE] Changing directory to repository root: "%~dp0\.."
pushd "%~dp0\.." >nul 2>&1
if errorlevel 1 (
  echo Failed to change directory to repository root.
  exit /b 1
)

set "VENV_DIR=.venv"
REM Remove existing virtual environment to avoid venvlauncher.exe copy issues
if exist "%VENV_DIR%\Scripts\python.exe" (
  if "%VERBOSE%"=="1" echo [VERBOSE] Removing existing virtual env at %VENV_DIR%
  rmdir /s /q "%VENV_DIR%"
)
if "%VERBOSE%"=="1" echo [VERBOSE] Creating virtual env at %VENV_DIR%
call %PY_CMD% -m venv "%VENV_DIR%"
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
if "%VERBOSE%"=="1" echo [VERBOSE] Using venv python: %VENV_PY%

REM Do not activate venv; use explicit venv python for all commands
REM Upgrade pip and install requirements
if "%VERBOSE%"=="1" echo [VERBOSE] Upgrading pip in virtualenv
if "%VERBOSE%"=="1" echo [VERBOSE] Command: "%VENV_PY%" -m pip install --upgrade pip
cmd.exe /c ""%VENV_PY%" -m pip install --upgrade pip" > "%BUILD_LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo Failed to upgrade pip.& echo --- pip upgrade log ---& type "%BUILD_LOG%"& exit /b 1

REM Ensure requirements.txt exists in repository root and install
if not exist "requirements.txt" (
  echo requirements.txt not found in project root (%CD%).
  popd >nul 2>&1
  exit /b 1
)
if "%VERBOSE%"=="1" echo [VERBOSE] Installing requirements from %CD%\requirements.txt
if "%VERBOSE%"=="1" echo [VERBOSE] Command: "%VENV_PY%" -m pip install -r "requirements.txt"
cmd.exe /c ""%VENV_PY%" -m pip install -r "requirements.txt"" >> "%BUILD_LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo Failed to install requirements from project root.& echo --- pip install log ---& type "%BUILD_LOG%"& popd >nul 2>&1& exit /b 1

REM Install PyInstaller
if "%VERBOSE%"=="1" echo [VERBOSE] Installing PyInstaller into virtualenv
if "%VERBOSE%"=="1" echo [VERBOSE] Command: "%VENV_PY%" -m pip install pyinstaller
cmd.exe /c ""%VENV_PY%" -m pip install pyinstaller" >> "%BUILD_LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo Failed to install PyInstaller.& echo --- pyinstaller pip install log ---& type "%BUILD_LOG%"& exit /b 1

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
  if "%VERBOSE%"=="1" echo [VERBOSE] Including installers in build: %INSTALLERS_DIR%
) else (
  echo installers directory not found. Skipping installers inclusion.
  set PYI_INSTALLERS=
)

REM Clean previous build artifacts
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist AI_Crawler_Assistant_Server.spec del /q AI_Crawler_Assistant_Server.spec

REM Build onefile executable for server.py using venv python
if "%VERBOSE%"=="1" echo [VERBOSE] Running PyInstaller (this may take a while)
if "%VERBOSE%"=="1" echo [VERBOSE] Command: "%VENV_PY%" -m PyInstaller --noconfirm --clean --onefile --name "AI_Crawler_Assistant_Server" --add-data "templates;templates" --add-data "data;data" %PYI_INSTALLERS% server.py
cmd.exe /c ""%VENV_PY%" -m PyInstaller --noconfirm --clean --onefile --name "AI_Crawler_Assistant_Server" --add-data "templates;templates" --add-data "data;data" %PYI_INSTALLERS% server.py" > "%BUILD_LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 echo PyInstaller build failed.& echo --- pyinstaller log ---& type "%BUILD_LOG%"& exit /b 1

echo Build complete. Executable located at dist\AI_Crawler_Assistant_Server.exe
if "%VERBOSE%"=="1" echo [VERBOSE] Build finished successfully
popd >nul 2>&1
endlocal