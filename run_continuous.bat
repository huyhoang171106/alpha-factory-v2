@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

call "%ROOT%\setup.bat"
if %ERRORLEVEL% neq 0 (
  echo [error] setup failed.
  exit /b %ERRORLEVEL%
)

set "PYTHONUNBUFFERED=1"
if not defined LLM_MODEL set "LLM_MODEL=openrouter/auto"
if not defined WQ_MAX_CONCURRENT set "WQ_MAX_CONCURRENT=4"
if not defined WQ_POLL_INTERVAL set "WQ_POLL_INTERVAL=10"
if not defined WQ_INTERACTIVE_AUTH set "WQ_INTERACTIVE_AUTH=0"

echo [run] starting continuous pipeline...
".venv\Scripts\python.exe" "%ROOT%\run_daily.py" --continuous --submit --level 5 --candidates 60 --cooldown 60 --max-submit 4 --pre-rank-score 50
set "CODE=%ERRORLEVEL%"
echo [exit] code=%CODE%
exit /b %CODE%
