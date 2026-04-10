"""
Export public-safe runtime metrics without leaking strategy expressions or credentials.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tracker import AlphaTracker


def build_public_report(lookback_minutes: int) -> dict:
    tracker = AlphaTracker()
    try:
        return {
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "lookback_minutes": max(1, int(lookback_minutes)),
            "pipeline": tracker.minute_kpis(lookback_minutes=lookback_minutes),
            "qd_archive": tracker.qd_archive_stats(),
        }
    finally:
        tracker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export public-safe Alpha Factory metrics.")
    parser.add_argument("--minutes", type=int, default=60, help="KPI lookback window in minutes")
    parser.add_argument("--out", default="results/public_report.json", help="Output JSON path")
    args = parser.parse_args()

    payload = build_public_report(args.minutes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
