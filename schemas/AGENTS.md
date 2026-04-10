<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-10T20:45:00 | Updated: 2026-04-10T20:45:00 -->

# schemas

## Purpose
JSON schemas for structured output and external contract definitions.

## Key Files

| File | Description |
|------|-------------|
| `public_report_schema.json` | JSON schema for the sanitized public KPI report exported via `python alpha_factory_cli.py public-report` |

## For AI Agents

### Public Report
The CLI exports `results/public_report.json` — this schema defines its structure. Safe fields include: throughput, acceptance rate, DLQ rate, QD archive stats. Never include alpha expressions, credentials, or raw API responses.

<!-- MANUAL: -->
