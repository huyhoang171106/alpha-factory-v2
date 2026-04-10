"""
alpha_factory_cli.py

Single-entry Windows-friendly launcher:
- One command to bootstrap and start.
- No .bat required for daily operation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
import zipfile
from datetime import datetime
from pathlib import Path

from tracker import AlphaTracker
from submit_governor import SubmitGovernor
from wq_client import WQClient

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE_FILE = ROOT / ".env.example"
RUN_DAILY = ROOT / "run_daily.py"
RUN_ASYNC = ROOT / "run_async_pipeline.py"
PROFILE_DEFAULTS = {
    "local": {
        "ASYNC_RANKER_WORKERS": "2",
        "ASYNC_SIMULATOR_WORKERS": "1",
        "ASYNC_USE_RAG": "1",
    },
    "vps": {
        "ASYNC_RANKER_WORKERS": "3",
        "ASYNC_SIMULATOR_WORKERS": "1",
        "ASYNC_USE_RAG": "0",
    },
    "gha": {
        "ASYNC_RANKER_WORKERS": "1",
        "ASYNC_SIMULATOR_WORKERS": "1",
        "ASYNC_USE_RAG": "0",
    },
}


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT))
    return proc.returncode


def _profile_env(profile: str) -> dict[str, str]:
    profile = (profile or "local").strip().lower()
    defaults = PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["local"])
    merged = {}
    for key, value in defaults.items():
        merged[key] = os.getenv(key, value)
    return merged


def ensure_venv() -> None:
    py = _venv_python()
    if py.exists():
        return
    print("[setup] creating virtual environment...")
    venv.create(str(VENV_DIR), with_pip=True)


def ensure_requirements() -> None:
    py = _venv_python()
    if not REQ_FILE.exists():
        print("[warn] requirements.txt not found, skip installation.")
        return
    print("[setup] installing requirements...")
    code = _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    if code != 0:
        raise RuntimeError("Failed to upgrade pip tooling")
    code = _run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)])
    if code != 0:
        raise RuntimeError("Failed to install requirements")


def ensure_env() -> None:
    if ENV_FILE.exists():
        return
    if ENV_EXAMPLE_FILE.exists():
        shutil.copyfile(ENV_EXAMPLE_FILE, ENV_FILE)
        print("[setup] created .env from .env.example")
        print("[next] fill WQ_EMAIL and WQ_PASSWORD in .env")
        return
    print("[warn] missing .env and .env.example")


def bootstrap(skip_install: bool = False) -> None:
    ensure_venv()
    if not skip_install:
        ensure_requirements()
    ensure_env()


def run_pipeline(continuous: bool, submit: bool, candidates: int, cooldown: int, max_submit: int, pre_rank: float) -> int:
    py = _venv_python()
    cmd = [
        str(py),
        str(RUN_DAILY),
        "--level",
        "5",
        "--candidates",
        str(candidates),
        "--max-submit",
        str(max_submit),
        "--pre-rank-score",
        str(pre_rank),
    ]
    if continuous:
        cmd.extend(["--continuous", "--cooldown", str(cooldown)])
    if submit:
        cmd.append("--submit")
    print("[run]", " ".join(cmd))
    return _run(cmd)


def run_async_pipeline(limit: int, score: float, use_rag: bool, profile: str = "local") -> int:
    py = _venv_python()
    cmd = [
        str(py),
        str(RUN_ASYNC),
        "--limit",
        str(limit),
        "--score",
        str(score),
    ]
    env = os.environ.copy()
    env.update(_profile_env(profile))
    env["ASYNC_USE_RAG"] = "1" if use_rag else env.get("ASYNC_USE_RAG", "0")
    print(
        "[run]",
        " ".join(cmd),
        f"(profile={profile} ASYNC_USE_RAG={env['ASYNC_USE_RAG']} "
        f"rankers={env.get('ASYNC_RANKER_WORKERS')} simulators={env.get('ASYNC_SIMULATOR_WORKERS')})",
    )
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return proc.returncode


def run_sync_submit(limit: int) -> int:
    tracker = AlphaTracker()
    client = WQClient()
    governor = SubmitGovernor(tracker, client)
    out = governor.reconcile_submitted(limit=limit)
    tracker.close()
    print({"submit_review_sync": out})
    return 0


def run_tests() -> int:
    py = _venv_python()
    if (ROOT / "tests").exists():
        return _run([str(py), "-m", "unittest", "discover", "-s", "tests", "-v"])
    return _run([str(py), "-m", "unittest", "discover", "-p", "test_*.py", "-v"])


def make_zip() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path = ROOT.parent / f"alpha-factory-portable-{ts}.zip"
    excluded_dirs = {".venv", "results", "__pycache__"}
    excluded_files = {".env", "alpha_results.db"}
    excluded_suffixes = {".log", ".pyc"}

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT.rglob("*"):
            rel = path.relative_to(ROOT)
            parts = set(rel.parts)
            if any(p in excluded_dirs for p in parts):
                continue
            if path.is_dir():
                continue
            if path.name in excluded_files:
                continue
            if path.suffix.lower() in excluded_suffixes:
                continue
            zf.write(path, rel.as_posix())
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha Factory one-click CLI")
    sub = parser.add_subparsers(dest="command")

    p_setup = sub.add_parser("setup", help="bootstrap venv, deps and .env")
    p_setup.add_argument("--skip-install", action="store_true", help="skip pip install")

    p_start = sub.add_parser("start", help="bootstrap then run pipeline")
    p_start.add_argument("--burst", action="store_true", help="run one burst instead of continuous")
    p_start.add_argument("--no-submit", action="store_true", help="run without submit")
    p_start.add_argument("--candidates", type=int, default=60)
    p_start.add_argument("--cooldown", type=int, default=60)
    p_start.add_argument("--max-submit", type=int, default=4)
    p_start.add_argument("--pre-rank-score", type=float, default=50.0)
    p_start.add_argument("--skip-install", action="store_true")

    p_async = sub.add_parser("async", help="run async streaming pipeline")
    p_async.add_argument("--limit", type=int, default=0, help="stop after N simulations (0 = run forever)")
    p_async.add_argument("--score", type=float, default=50.0, help="min pre-rank score")
    p_async.add_argument("--use-rag", action="store_true", help="enable RAG in async producer")
    p_async.add_argument("--profile", choices=["local", "vps", "gha"], default="local", help="runtime profile")
    p_async.add_argument("--skip-install", action="store_true")

    p_auto = sub.add_parser("auto", help="one-click auto mode for local/vps/gha")
    p_auto.add_argument("--profile", choices=["local", "vps", "gha"], default="local")
    p_auto.add_argument("--skip-install", action="store_true")

    p_replay = sub.add_parser("replay-dlq", help="requeue dead-lettered submit jobs")
    p_replay.add_argument("--limit", type=int, default=50)

    p_kpi = sub.add_parser("kpi", help="print minute-level KPIs")
    p_kpi.add_argument("--minutes", type=int, default=60)

    p_sync_submit = sub.add_parser("sync-submit", help="poll WQ review status for submitted alphas")
    p_sync_submit.add_argument("--limit", type=int, default=30)

    sub.add_parser("test", help="run unit tests")
    sub.add_parser("zip", help="create clean portable zip")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command or "start"  # one-click default

    try:
        if command == "setup":
            bootstrap(skip_install=args.skip_install)
            print("[ok] setup complete")
            return 0

        if command == "start":
            bootstrap(skip_install=args.skip_install)
            code = run_pipeline(
                continuous=not args.burst,
                submit=not args.no_submit,
                candidates=args.candidates,
                cooldown=args.cooldown,
                max_submit=args.max_submit,
                pre_rank=args.pre_rank_score,
            )
            return code

        if command == "test":
            bootstrap(skip_install=False)
            return run_tests()

        if command == "async":
            bootstrap(skip_install=args.skip_install)
            return run_async_pipeline(limit=args.limit, score=args.score, use_rag=args.use_rag, profile=args.profile)

        if command == "auto":
            bootstrap(skip_install=args.skip_install)
            if args.profile == "gha":
                limit = int(os.getenv("GHA_BURST_LIMIT", "80"))
                score = float(os.getenv("GHA_PRE_RANK_SCORE", "50"))
                sync_limit = int(os.getenv("GHA_SYNC_LIMIT", "40"))
                code = run_async_pipeline(limit=limit, score=score, use_rag=False, profile="gha")
                if code != 0:
                    return code
                return run_sync_submit(limit=sync_limit)
            # local/vps defaults run continuous engine mode
            return run_async_pipeline(limit=0, score=50.0, use_rag=False, profile=args.profile)

        if command == "replay-dlq":
            tracker = AlphaTracker()
            moved = tracker.replay_dlq(limit=args.limit)
            tracker.close()
            print(f"[ok] replayed={moved}")
            return 0

        if command == "kpi":
            tracker = AlphaTracker()
            kpi = tracker.minute_kpis(lookback_minutes=args.minutes)
            qd = tracker.qd_archive_stats()
            tracker.close()
            print({"pipeline": kpi, "qd_archive": qd})
            return 0

        if command == "sync-submit":
            return run_sync_submit(limit=args.limit)

        if command == "zip":
            out = make_zip()
            print(f"[ok] {out}")
            return 0

        print("[error] unknown command")
        return 2
    except Exception as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
