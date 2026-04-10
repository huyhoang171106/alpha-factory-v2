# Alpha Factory Portable Setup (Windows)

This folder is prepared for zip/unzip portability.
For full runtime architecture and operational semantics, read `README.md`.
For deploy handoff contract, read `CODEX_HANDOFF_SCOPE.md`.
For execution scenarios, read `USE_CASES.md`.

## 1) What to zip

Zip the whole `alpha-factory` directory, but exclude:

- `.venv`
- `.env`
- `results`
- `alpha_results.db`

The target machine will recreate what it needs.
Or run `make_portable_zip.bat` to generate a clean zip automatically.

## 2) First run on a new machine

1. Unzip folder.
2. Open folder.
3. Run one CLI file:
   - `python alpha_factory_cli.py` (default = bootstrap + continuous start), or
   - `python alpha_factory_cli.py start`
4. Edit `.env` and fill (if first run created it):
   - `WQ_EMAIL`
   - `WQ_PASSWORD`
   - optional `OPENROUTER_API_KEY`
5. Useful commands:
   - `python alpha_factory_cli.py start` for 24/7 mode
   - `python alpha_factory_cli.py start --burst` for one short run
   - `python alpha_factory_cli.py async --limit 500 --score 50` for async streaming mode
   - `python alpha_factory_cli.py test` to run unit tests
   - `python alpha_factory_cli.py zip` to build portable zip

## 3) Minimum machine requirements

- Windows 10/11
- Python 3.10+ installed and available as `py -3` or `python`
- Stable internet

## 4) Recommended defaults

In `.env`:

- `WQ_INTERACTIVE_AUTH=0` for unattended mode
- `WQ_MAX_CONCURRENT=4`
- `WQ_POLL_INTERVAL=10`
- `WQ_MAX_WAIT_TIME=600`

If WQ API is unstable, reduce load:

- set `WQ_MAX_CONCURRENT=2`
- run with smaller `--candidates`

## 5) Troubleshooting

- If `setup.bat` fails: install Python 3.10+ and retry.
- If login requires biometric: set `WQ_INTERACTIVE_AUTH=1` and run manually once.
- If OpenRouter credits are low: keep `OPENROUTER_API_KEY` empty; deterministic generation still works.

## 6) GitHub Actions note

GitHub schedule minimum is every 5 minutes, not every minute.
Use `run_burst.bat` equivalent logic in CI and keep `--continuous` on self-hosted/VPS only.
