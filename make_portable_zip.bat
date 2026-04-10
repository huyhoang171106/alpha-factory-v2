@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PARENT=%ROOT%\.."

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%i"
set "ZIP_NAME=alpha-factory-portable-%TS%.zip"
set "ZIP_PATH=%PARENT%\%ZIP_NAME%"
set "STAGE=%TEMP%\alpha-factory-stage-%TS%"

echo [zip] creating %ZIP_PATH%
powershell -NoProfile -Command ^
  "$root = Resolve-Path '%ROOT%';" ^
  "$stage = '%STAGE%';" ^
  "if (Test-Path $stage) { Remove-Item -Recurse -Force $stage };" ^
  "New-Item -ItemType Directory -Path $stage | Out-Null;" ^
  "Copy-Item -Path ($root.Path + '\*') -Destination $stage -Recurse -Force;" ^
  "$excludeDirs = @('.venv','results','__pycache__');" ^
  "foreach ($d in $excludeDirs) { Get-ChildItem -Path $stage -Recurse -Directory -Filter $d -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue };" ^
  "$excludeFiles = @('.env','alpha_results.db');" ^
  "foreach ($f in $excludeFiles) { Get-ChildItem -Path $stage -Recurse -File -Filter $f -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue };" ^
  "Get-ChildItem -Path $stage -Recurse -File -Include *.log,*.pyc -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue;" ^
  "Compress-Archive -Path ($stage + '\*') -DestinationPath '%ZIP_PATH%' -CompressionLevel Optimal -Force;" ^
  "Remove-Item -Recurse -Force $stage"

if %ERRORLEVEL% neq 0 (
  echo [error] failed to create zip.
  exit /b %ERRORLEVEL%
)

echo [ok] %ZIP_PATH%
exit /b 0
