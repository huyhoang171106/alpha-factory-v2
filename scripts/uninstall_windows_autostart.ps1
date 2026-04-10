Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$taskName = "AlphaFactoryAutoLocal"

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "[ok] scheduled task removed: $taskName"
} else {
    Write-Host "[ok] scheduled task not found: $taskName"
}
