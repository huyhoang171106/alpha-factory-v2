@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "LOG_DIR=%ROOT%\results"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%i"
set "LOG_FILE=%LOG_DIR%\overnight-%TS%.log"

set "PYTHONUNBUFFERED=1"
set "LLM_MODEL=openrouter/auto"
set "WQ_MAX_CONCURRENT=2"
set "WQ_POLL_INTERVAL=20"
if not defined WQ_INTERACTIVE_AUTH set "WQ_INTERACTIVE_AUTH=0"

call "%ROOT%\setup.bat"
if %ERRORLEVEL% neq 0 (
  echo [%date% %time%] SETUP failed code=%ERRORLEVEL%>>"%LOG_FILE%"
  exit /b %ERRORLEVEL%
)

:loop
echo [%date% %time%] START run_daily --continuous>>"%LOG_FILE%"
".venv\Scripts\python.exe" "%ROOT%\run_daily.py" --continuous --submit --level 4 --candidates 10 --cooldown 60 --max-submit 15 --pre-rank-score 50 >>"%LOG_FILE%" 2>&1
set "CODE=%ERRORLEVEL%"
echo [%date% %time%] EXIT code=%CODE%, restart in 20s>>"%LOG_FILE%"
timeout /t 20 /nobreak >nul
goto loop
