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

if not exist "requirements.txt" (
  echo requirements.txt not found. Creating a default requirements.txt file.
  echo requests>=2.31.0 >> requirements.txt
  echo beautifulsoup4>=4.12.3 >> requirements.txt
  echo lxml>=4.9.3 >> requirements.txt  
  echo trafilatura>=1.6.3 >> requirements.txt
  echo langchain>=0.0.208 >> requirements.txt
  echo duckduckgo-search>=6.3.5 >> requirements.txt
  echo PyMuPDF>=1.22.5 >> requirements.txt
  echo pdfminer.six>=20221105 >> requirements.txt
  echo python-pptx>=0.6.21 >> requirements.txt
  echo python-docx>=0.8.11 >> requirements.txt
  echo openpyxl>=3.1.2 >> requirements.txt
  echo ebooklib>=0.18 >> requirements.txt
  echo pypdf>=3.14.0 >> requirements.txt
  echo numpy>=1.25.0 >> requirements.txt
  echo pandas>=2.1.0 >> requirements.txt
  echo faiss-cpu>=1.7.4 >> requirements.txt
  echo scikit-learn>=1.3.2 >> requirements.txt
  echo sentence-transformers>=2.2.2 >> requirements.txt
  echo python-multipart>=0.0.6 >> requirements.txt
  echo fastapi>=0.103.0 >> requirements.txt
  echo uvicorn>=0.23.0 >> requirements.txt
  echo aiofiles>=23.1.0 >> requirements.txt
  echo jinja2>=3.1.2 >> requirements.txt
  echo itsdangerous>=2.1.2 >> requirements.txt
  echo sse-starlette>=1.6.5 >> requirements.txt
  echo transformers>=4.41.0 >> requirements.txt
  echo torch>=2.2.0 >> requirements.txt
  echo croniter>=1.4.1 >> requirements.txt
  echo apscheduler>=3.10.4 >> requirements.txt
  echo pyinstaller>=6.16.0 >> requirements.txt
)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  exit /b 1
)
REM Install requirements from parent directory
if not exist "..\requirements.txt" (
  echo requirements.txt not found in parent directory.
  exit /b 1
)
"%VENV_PY%" -m pip install -r "..\requirements.txt"
if errorlevel 1 (
  echo Failed to install requirements from parent directory.
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