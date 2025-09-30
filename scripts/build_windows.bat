@echo off
setlocal

REM Build Windows executable for AI Crawler Assistant (FastAPI server)
REM Requirements: Python 3.10+, pip, and internet access to install dependencies

REM Detect Python or py launcher
set PY_CMD=python
where %PY_CMD% >nul 2>&1
IF ERRORLEVEL 1 (
  where py >nul 2>&1
  IF ERRORLEVEL 1 (
    echo Python not found on PATH. Please install Python and ensure 'python' or 'py' is available.
    exit /b 1
  ) ELSE (
    set PY_CMD=py
  )
)

REM Create venv
set VENV_DIR=.venv
%PY_CMD% -m venv %VENV_DIR%
IF ERRORLEVEL 1 (
  echo Failed to create virtual environment.
  exit /b 1
)

REM Activate venv
call %VENV_DIR%\Scripts\activate

REM Upgrade pip and install requirements
python -m pip install --upgrade pip
IF ERRORLEVEL 1 (
  echo Failed to upgrade pip.
  exit /b 1
)

python -m pip install -r requirements.txt
IF ERRORLEVEL 1 (
  echo Failed to install requirements.
  exit /b 1
)

REM Install PyInstaller
python -m pip install pyinstaller
IF ERRORLEVEL 1 (
  echo Failed to install PyInstaller.
  exit /b 1
)

REM Ensure templates are bundled with the executable
set TEMPLATES=templates
IF NOT EXIST %TEMPLATES% (
  echo templates directory not found. Ensure templates/ exists.
  exit /b 1
)

REM Ensure data directory exists to include in bundle
set DATA_DIR=data
IF NOT EXIST %DATA_DIR% (
  echo data directory not found. Creating empty data directory.
  mkdir %DATA_DIR%
)

REM Clean previous build artifacts
IF EXIST build rmdir /s /q build
IF EXIST dist rmdir /s /q dist
IF EXIST AI_Crawler_Assistant_Server.spec del /q AI_Crawler_Assistant_Server.spec

REM Build onefile executable for server.py using venv python
python -m pyinstaller --noconfirm --clean --onefile --name "AI_Crawler_Assistant_Server" ^
  --add-data "templates;templates" ^
  --add-data "data;data" ^
  server.py

IF ERRORLEVEL 1 (
  echo PyInstaller build failed.
  exit /b 1
)

echo Build complete. Executable located at dist\AI_Crawler_Assistant_Server.exe
endlocal