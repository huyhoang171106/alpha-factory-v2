<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-10T20:45:00 | Updated: 2026-04-10T20:45:00 -->

# deploy

## Purpose
Production deployment files — currently a single systemd service unit for running Alpha Factory on a VPS.

## Key Files

| File | Description |
|------|-------------|
| `alpha-factory.service` | systemd unit for VPS autostart and supervision |

## For AI Agents

### Usage
Install on a Linux VPS with systemd:
```bash
sudo cp alpha-factory.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable alpha-factory
sudo systemctl start alpha-factory
```

See also `scripts/run_vps.sh` for the startup command.

<!-- MANUAL: -->
