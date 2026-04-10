Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$cli = Join-Path $root "alpha_factory_cli.py"
$resultsDir = Join-Path $root "results"
$logFile = Join-Path $resultsDir "windows_auto_runner.log"
$errFile = Join-Path $resultsDir "windows_auto_runner.err.log"

if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

if (-not (Test-Path $python)) {
    throw "Missing venv python at: $python. Run setup first."
}

if (-not (Test-Path $cli)) {
    throw "Missing CLI file at: $cli"
}

function Test-AlphaFactoryRunning {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like "python*.exe" -and (
            ($_.CommandLine -like "*alpha_factory_cli.py*" -and $_.CommandLine -like "*auto*" -and $_.CommandLine -like "*--profile local*") -or
            ($_.CommandLine -like "*run_async_pipeline.py*" -and $_.CommandLine -like "*--limit 0*" -and $_.CommandLine -like "*--score 50*")
        )
    }
    return ($procs | Measure-Object).Count -gt 0
}

if (Test-AlphaFactoryRunning) {
    Add-Content -Path $logFile -Value "$(Get-Date -Format s) [runner] existing local auto process detected; skip."
    exit 0
}

Add-Content -Path $logFile -Value "$(Get-Date -Format s) [runner] starting supervisor loop"

while ($true) {
    if (Test-AlphaFactoryRunning) {
        Add-Content -Path $logFile -Value "$(Get-Date -Format s) [runner] detected active process; sleep 20s"
        Start-Sleep -Seconds 20
        continue
    }
    Add-Content -Path $logFile -Value "$(Get-Date -Format s) [runner] launch: auto --profile local --skip-install"
    & $python $cli auto --profile local --skip-install 1>> $logFile 2>> $errFile
    $code = $LASTEXITCODE
    Add-Content -Path $logFile -Value "$(Get-Date -Format s) [runner] exit code=$code, restart in 15s"
    Start-Sleep -Seconds 15
}
