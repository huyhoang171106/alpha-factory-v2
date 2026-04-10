@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

echo [setup] Alpha Factory bootstrap at "%ROOT%"

set "PY_CMD="
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 set "PY_CMD=python"
)

if not defined PY_CMD (
  echo [error] Python 3 not found. Install Python 3.10+ and retry.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment...
  %PY_CMD% -m venv .venv
  if %ERRORLEVEL% neq 0 (
    echo [error] Failed to create venv.
    exit /b 1
  )
)

echo [setup] Upgrading pip/setuptools/wheel...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% neq 0 (
  echo [error] Failed to upgrade pip tools.
  exit /b 1
)

if exist "requirements.txt" (
  echo [setup] Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if %ERRORLEVEL% neq 0 (
    echo [error] Dependency installation failed.
    exit /b 1
  )
) else (
  echo [warn] requirements.txt not found. Skip dependency installation.
)

if not exist ".env" (
  if exist ".env.example" (
    copy /Y ".env.example" ".env" >nul
    echo [setup] Created .env from .env.example
    echo [next] Fill WQ_EMAIL and WQ_PASSWORD in .env then run again.
  ) else (
    echo [warn] Missing .env and .env.example
  )
)

echo [ok] Setup complete.
exit /b 0
