<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-10T20:45:00 | Updated: 2026-04-10T20:45:00 -->

# scripts

## Purpose
Operational automation scripts — deployment runners, Windows autostart management, and public metrics export.

## Key Files

| File | Description |
|------|-------------|
| `run_vps.sh` | VPS startup runner — launch async pipeline on a remote Linux server |
| `windows_auto_runner.ps1` | Windows scheduled-task runner for local autostart |
| `install_windows_autostart.ps1` | Register Windows autostart task via Task Scheduler |
| `uninstall_windows_autostart.ps1` | Remove Windows autostart task |
| `export_public_metrics.py` | Script to export and optionally sync sanitized KPI JSON to a public profile repo |

## For AI Agents

### Windows Autostart
Run as Administrator to install the scheduled task. The runner script executes `python alpha_factory_cli.py auto --profile local --no-hybrid`.

### Public Metrics Mirror
Uses `PUBLIC_STATUS_REPO` and `PUBLIC_STATUS_TOKEN` env vars. Only safe (sanitized) metrics are exported.

<!-- MANUAL: -->
