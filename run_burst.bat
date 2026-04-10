@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

call "%ROOT%\setup.bat"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

set "PYTHONUNBUFFERED=1"
if not defined WQ_INTERACTIVE_AUTH set "WQ_INTERACTIVE_AUTH=0"

echo [run] starting one burst cycle...
".venv\Scripts\python.exe" "%ROOT%\run_daily.py" --submit --target 1 --candidates 40 --max-submit 4 --pre-rank-score 50
exit /b %ERRORLEVEL%
