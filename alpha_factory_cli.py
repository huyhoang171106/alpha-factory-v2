"""
alpha_factory_cli.py

Single-entry Windows-friendly launcher:
- One command to bootstrap and start.
- No .bat required for daily operation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import venv
import zipfile
from datetime import datetime
from pathlib import Path

from tracker import AlphaTracker
from submit_governor import SubmitGovernor

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
        "ASYNC_USE_RAG": "0",
        "ASYNC_BATCH_SIZE": "1",
        "ASYNC_GEN_BATCH_SIZE": "12",
        "ASYNC_GEN_QUEUE_SIZE": "240",
        "ASYNC_SIM_QUEUE_SIZE": "120",
        "ASYNC_MIN_CRITIC_SCORE": "0.30",
        "ASYNC_COMPLEXITY_MIN": "0.40",
        "ASYNC_ROBUST_SCORE_MIN": "1.00",
        "ASYNC_MIN_SHARPE": "1.25",
        "ASYNC_MIN_FITNESS": "0.80",
        "ASYNC_TURNOVER_MIN": "1.0",
        "ASYNC_TURNOVER_MAX": "75.0",
        "ASYNC_REQUIRE_ALL_CHECKS": "0",
        "ASYNC_MIN_CHECKS_RATIO": "0.50",
        "ASYNC_ENABLE_D0": "1",
        "ASYNC_D1_SHARE": "0.90",
        "ASYNC_HYPO_TEMPLATE_RATIO": "0.35",
        "ASYNC_HYPO_ADV_WRAP_PROB": "0.00",
        "GEN_REQUIRE_LOCAL_BT_SUPPORT": "1",
        "ASYNC_SIM_BATCH_TIMEOUT": "600",
        "WQ_MAX_CONCURRENT": "1",
        "WQ_MAX_WAIT_TIME": "600",
        "WQ_SIM_SUBMIT_RETRIES": "3",
        "WQ_SIM_SUBMIT_BACKOFF": "8",
        "GENERATOR_MODE": "hybrid_hypothesis",
    },
    "vps": {
        "ASYNC_RANKER_WORKERS": "3",
        "ASYNC_SIMULATOR_WORKERS": "1",
        "ASYNC_USE_RAG": "0",
        "ASYNC_MIN_CRITIC_SCORE": "0.30",
        "ASYNC_COMPLEXITY_MIN": "0.40",
        "ASYNC_HYPO_ADV_WRAP_PROB": "0.00",
        "ASYNC_HYPO_TEMPLATE_RATIO": "0.00",
        "GEN_REQUIRE_LOCAL_BT_SUPPORT": "1",
        "GENERATOR_MODE": "hypothesis_driven",
    },
    "gha": {
        "ASYNC_RANKER_WORKERS": "1",
        "ASYNC_SIMULATOR_WORKERS": "1",
        "ASYNC_USE_RAG": "0",
    },
}


def _is_windows() -> bool:
    return os.name == "nt"


def _default_global_bin_dir() -> Path:
    if _is_windows():
        local_appdata = os.getenv("LOCALAPPDATA", "")
        candidate = Path(local_appdata) / "Microsoft" / "WindowsApps"
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate
    return Path.home() / ".alpha-bin"


def _path_contains(target: Path) -> bool:
    entries = [
        p.strip().lower() for p in os.getenv("PATH", "").split(os.pathsep) if p.strip()
    ]
    return str(target.resolve()).lower() in entries


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT))
    return proc.returncode


def install_global_command(bin_dir: Path | None = None, name: str = "alpha") -> int:
    target_dir = (bin_dir or _default_global_bin_dir()).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    cli_path = (ROOT / "alpha_factory_cli.py").resolve()
    python_path = Path(sys.executable).resolve()

    if _is_windows():
        cmd_path = target_dir / f"{name}.cmd"
        ps1_path = target_dir / f"{name}.ps1"
        cmd_path.write_text(
            f'@echo off\r\n"{python_path}" "{cli_path}" %*\r\n',
            encoding="utf-8",
        )
        ps1_path.write_text(
            f'& "{python_path}" "{cli_path}" @args\r\n',
            encoding="utf-8",
        )
        print(f"[ok] installed command wrappers: {cmd_path}")
    else:
        sh_path = target_dir / name
        sh_path.write_text(
            f'#!/usr/bin/env bash\n"{python_path}" "{cli_path}" "$@"\n',
            encoding="utf-8",
        )
        sh_path.chmod(0o755)
        print(f"[ok] installed command wrapper: {sh_path}")

    if not _path_contains(target_dir):
        print(
            "[warn] global bin dir is not on PATH.\n"
            f"Add this directory to PATH, then open a new terminal:\n{target_dir}"
        )
    else:
        print(f"[ok] '{name}' command is now available in this terminal PATH.")
    print(f"[hint] try: {name} --help")
    return 0


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    return True


def _acquire_singleton_lock(lock_path: Path, stale_after_seconds: int = 86400):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        # Keep the lock if we cannot prove it is stale/dead.
        can_reclaim = False
        try:
            content = lock_path.read_text(encoding="utf-8").strip().split(",")
            pid = int(content[0]) if content and content[0] else 0
            ts = int(content[1]) if len(content) > 1 and content[1] else 0
            stale = ts <= 0 or (int(time.time()) - ts) > stale_after_seconds
            if (not _process_exists(pid)) or stale:
                can_reclaim = True
        except Exception:
            can_reclaim = False

        if can_reclaim:
            try:
                lock_path.unlink(missing_ok=True)
            except PermissionError:
                return None
        else:
            return None

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError:
        return None
    os.write(fd, f"{os.getpid()},{int(time.time())}".encode("utf-8"))
    return fd


def _release_singleton_lock(fd: int | None, lock_path: Path):
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def _profile_env(profile: str) -> dict[str, str]:
    profile = (profile or "local").strip().lower()
    defaults = PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["local"])
    merged = {}
    for key, value in defaults.items():
        merged[key] = os.getenv(key, value)

    # Safety Hardening for Local Profile
    if profile == "local" and os.getenv("ALPHA_ALLOW_UNSAFE_PACING") != "1":
        sim_workers = int(merged.get("ASYNC_SIMULATOR_WORKERS", "1"))
        if sim_workers > 1:
            print(
                f"[safety] local profile: capping ASYNC_SIMULATOR_WORKERS=1 (current={sim_workers}). Set ALPHA_ALLOW_UNSAFE_PACING=1 to override."
            )
            merged["ASYNC_SIMULATOR_WORKERS"] = "1"

        max_concurrent = int(merged.get("WQ_MAX_CONCURRENT", "1"))
        if max_concurrent > 2:
            print(
                f"[safety] local profile: capping WQ_MAX_CONCURRENT=2 (current={max_concurrent}) for stability."
            )
            merged["WQ_MAX_CONCURRENT"] = "2"

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
    code = _run(
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    )
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


def run_pipeline(
    continuous: bool,
    submit: bool,
    candidates: int,
    cooldown: int,
    max_submit: int,
    pre_rank: float,
) -> int:
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


def run_async_pipeline(
    limit: int, score: float, use_rag: bool, profile: str = "local"
) -> int:
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


def run_local_hybrid(profile: str = "local") -> int:
    env = os.environ.copy()
    env.update(_profile_env(profile))
    env["ASYNC_USE_RAG"] = env.get("ASYNC_USE_RAG", "0")

    sync_interval = max(30, int(env.get("LOCAL_HYBRID_SYNC_INTERVAL", "300")))
    kpi_interval = max(30, int(env.get("LOCAL_HYBRID_KPI_INTERVAL", "300")))
    sync_limit = max(1, int(env.get("LOCAL_HYBRID_SYNC_LIMIT", "40")))
    restart_backoff = max(5, int(env.get("LOCAL_HYBRID_RESTART_BACKOFF", "20")))
    score = float(env.get("LOCAL_HYBRID_SCORE", "50"))
    lock_path = Path(
        env.get("LOCAL_SINGLETON_LOCKFILE", str(ROOT / "results" / "local_auto.lock"))
    )

    lock_fd = _acquire_singleton_lock(lock_path)
    if lock_fd is None:
        print(f"[skip] another local hybrid instance is already running ({lock_path})")
        return 0

    py = _venv_python()
    cmd = [
        str(py),
        str(RUN_ASYNC),
        "--limit",
        "0",
        "--score",
        str(score),
    ]
    print(
        f"[hybrid] starting profile={profile} "
        f"sync_every={sync_interval}s kpi_every={kpi_interval}s"
    )

    proc = None
    try:
        while True:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
            # Avoid immediate sidecar calls right after spawn; let the async engine warm up.
            last_sync = time.time()
            last_kpi = time.time()
            while proc.poll() is None:
                now = time.time()
                if now - last_sync >= sync_interval:
                    try:
                        run_sync_submit(limit=sync_limit)
                    except Exception as exc:
                        print(f"[warn] hybrid sync-submit failed: {exc}")
                    last_sync = now
                if now - last_kpi >= kpi_interval:
                    try:
                        tracker = AlphaTracker()
                        kpi = tracker.minute_kpis(lookback_minutes=60)
                        tracker.close()
                        print(
                            f"[hybrid] kpi gate={kpi.get('gate_pass_rate', 0.0):.1%} submit={kpi.get('submit_success_rate', 0.0):.1%}"
                        )
                    except Exception as exc:
                        print(f"[warn] hybrid kpi check failed: {exc}")
                    last_kpi = now
                time.sleep(5)

            code = int(proc.returncode or 0)
            print(
                f"[hybrid] engine exited with code={code}; restarting in {restart_backoff}s"
            )
            time.sleep(restart_backoff)
    except KeyboardInterrupt:
        print("[hybrid] stop requested")
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return 130
    finally:
        _release_singleton_lock(lock_fd, lock_path)


def run_sync_submit(limit: int) -> int:
    from wq_client import WQClient

    tracker = AlphaTracker()
    client = WQClient()
    governor = SubmitGovernor(tracker, client)
    out = governor.reconcile_submitted(limit=limit)
    tracker.close()
    print({"submit_review_sync": out})
    return 0


def run_public_report(minutes: int, out_path: str) -> int:
    tracker = AlphaTracker()
    try:
        payload = {
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "lookback_minutes": max(1, int(minutes)),
            "pipeline": tracker.minute_kpis(lookback_minutes=minutes),
            "qd_archive": tracker.qd_archive_stats(),
        }
    finally:
        tracker.close()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(f"[ok] public report exported to {output}")
    return 0


def append_thinking_log(title: str, note: str, tags: str = "") -> int:
    """
    Append a structured engineering thinking entry for long-term system learning.
    """
    log_path = ROOT / "DEVELOPER_THINKING_LOG.md"
    ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    tag_line = f"\nTags: {tags}" if tags else ""
    entry = (
        f"\n## {ts} - {title.strip() or 'Session Note'}\n{note.strip()}\n{tag_line}\n"
    )
    if not log_path.exists():
        header = (
            "# Developer Thinking Log\n\n"
            "Append-only log of architecture reasoning, failures, fixes, and upgrade directions.\n"
            "Use this as institutional memory for future quant projects.\n"
        )
        log_path.write_text(header, encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[ok] thinking log updated: {log_path}")
    return 0


def _run_timed_async_leg(
    profile: str,
    score: float,
    minutes: int,
    env_overrides: dict[str, str],
) -> dict:
    """
    Run one timed async leg for safe A/B cadence comparison.
    """
    py = _venv_python()
    cmd = [str(py), str(RUN_ASYNC), "--limit", "0", "--score", str(score)]
    env = os.environ.copy()
    env.update(_profile_env(profile))
    env.update(env_overrides)
    started_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    deadline = t0 + max(1, int(minutes)) * 60
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(3)
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    elapsed = int(time.time() - t0)
    tracker = AlphaTracker()
    try:
        kpi = tracker.minute_kpis(lookback_minutes=max(5, int(minutes)))
        qd = tracker.qd_archive_stats()
    finally:
        tracker.close()
    return {
        "profile": profile,
        "overrides": env_overrides,
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed,
        "exit_code": int(proc.returncode or 0),
        "kpi": kpi,
        "qd_archive": qd,
    }


def run_ab_safe(
    profile: str,
    score: float,
    minutes_per_leg: int,
    cycles: int,
    d1_share_a: float,
    d1_share_b: float,
    out_path: str,
) -> int:
    """
    Safe A/B comparison by cadence (A then B), avoiding concurrent submit collisions.
    """
    report = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "mode": "ab_safe_cadence",
        "profile": profile,
        "minutes_per_leg": minutes_per_leg,
        "cycles": cycles,
        "legs": [],
    }
    for i in range(max(1, int(cycles))):
        print(f"[ab-safe] cycle {i + 1}/{cycles} leg A (D1 share={d1_share_a:.2f})")
        leg_a = _run_timed_async_leg(
            profile=profile,
            score=score,
            minutes=minutes_per_leg,
            env_overrides={"ASYNC_D1_SHARE": f"{max(0.0, min(1.0, d1_share_a)):.2f}"},
        )
        leg_a["label"] = "A"
        leg_a["cycle"] = i + 1
        report["legs"].append(leg_a)

        print(f"[ab-safe] cycle {i + 1}/{cycles} leg B (D1 share={d1_share_b:.2f})")
        leg_b = _run_timed_async_leg(
            profile=profile,
            score=score,
            minutes=minutes_per_leg,
            env_overrides={"ASYNC_D1_SHARE": f"{max(0.0, min(1.0, d1_share_b)):.2f}"},
        )
        leg_b["label"] = "B"
        leg_b["cycle"] = i + 1
        report["legs"].append(leg_b)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[ok] ab-safe report: {out}")
    return 0


def run_ab_rag(
    profile: str,
    score: float,
    minutes_per_leg: int,
    out_path: str,
) -> int:
    """
    Automated A/B test: Normal (RAG=0) vs RAG (RAG=1).
    """
    report = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "mode": "ab_rag_comparison",
        "profile": profile,
        "minutes_per_leg": minutes_per_leg,
        "legs": [],
    }

    # Leg A: Normal (RAG=0)
    print(f"\n[ab-rag] starting Leg A: Normal (RAG=0) - {minutes_per_leg} mins")
    run_id_a = f"ab_normal_{int(time.time())}"
    leg_a = _run_timed_async_leg(
        profile=profile,
        score=score,
        minutes=minutes_per_leg,
        env_overrides={"ASYNC_USE_RAG": "0", "RUN_ID": run_id_a},
    )
    leg_a["label"] = "Normal"
    leg_a["run_id"] = run_id_a
    report["legs"].append(leg_a)

    # Leg B: RAG (RAG=1)
    print(f"\n[ab-rag] starting Leg B: RAG (RAG=1) - {minutes_per_leg} mins")
    run_id_b = f"ab_rag_{int(time.time())}"
    leg_b = _run_timed_async_leg(
        profile=profile,
        score=score,
        minutes=minutes_per_leg,
        env_overrides={"ASYNC_USE_RAG": "1", "RUN_ID": run_id_b},
    )
    leg_b["label"] = "RAG"
    leg_b["run_id"] = run_id_b
    report["legs"].append(leg_b)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"\n[ok] ab-rag report generated: {out}")

    # Preliminary command-line analysis
    print("\n" + "=" * 40)
    print("      A/B COMPARISON PREVIEW")
    print("=" * 40)
    for leg in report["legs"]:
        kpi = leg.get("kpi", {})
        print(f"Leg {leg['label']} ({leg['run_id']}):")
        print(f"  Simulated: {kpi.get('simulated', 0)}")
        print(
            f"  Gated:     {kpi.get('gated', 0)} ({kpi.get('gate_pass_rate', 0.0):.1%})"
        )
        print(f"  Accepted:  {kpi.get('accepted', 0)}")
    print("=" * 40)

    # Suggest running the full comparison script
    print(
        f"\n[hint] Run 'python compare_reports.py --a {run_id_a} --b {run_id_b}' for deep analysis."
    )
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
    p_start.add_argument(
        "--burst", action="store_true", help="run one burst instead of continuous"
    )
    p_start.add_argument("--no-submit", action="store_true", help="run without submit")
    p_start.add_argument("--candidates", type=int, default=60)
    p_start.add_argument("--cooldown", type=int, default=60)
    p_start.add_argument("--max-submit", type=int, default=4)
    p_start.add_argument("--pre-rank-score", type=float, default=50.0)
    p_start.add_argument("--skip-install", action="store_true")

    p_async = sub.add_parser("async", help="run async streaming pipeline")
    p_async.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after N simulations (0 = run forever)",
    )
    p_async.add_argument("--score", type=float, default=50.0, help="min pre-rank score")
    p_async.add_argument(
        "--use-rag", action="store_true", help="enable RAG in async producer"
    )
    p_async.add_argument(
        "--profile",
        choices=["local", "vps", "gha"],
        default="local",
        help="runtime profile",
    )
    p_async.add_argument("--skip-install", action="store_true")

    p_auto = sub.add_parser("auto", help="one-click auto mode for local/vps/gha")
    p_auto.add_argument("--profile", choices=["local", "vps", "gha"], default="local")
    p_auto.add_argument(
        "--no-hybrid", action="store_true", help="disable local hybrid supervisor mode"
    )
    p_auto.add_argument("--skip-install", action="store_true")

    p_replay = sub.add_parser("replay-dlq", help="requeue dead-lettered submit jobs")
    p_replay.add_argument("--limit", type=int, default=50)

    p_kpi = sub.add_parser("kpi", help="print minute-level KPIs")
    p_kpi.add_argument("--minutes", type=int, default=60)

    p_sync_submit = sub.add_parser(
        "sync-submit", help="poll WQ review status for submitted alphas"
    )
    p_sync_submit.add_argument("--limit", type=int, default=30)

    p_public = sub.add_parser(
        "public-report", help="export sanitized public-safe KPI report"
    )
    p_public.add_argument("--minutes", type=int, default=60)
    p_public.add_argument("--out", default="results/public_report.json")

    p_ab = sub.add_parser(
        "ab-safe", help="safe cadence A/B compare (no parallel submit collision)"
    )
    p_ab.add_argument("--profile", choices=["local", "vps", "gha"], default="local")
    p_ab.add_argument("--score", type=float, default=50.0)
    p_ab.add_argument("--minutes-per-leg", type=int, default=10)
    p_ab.add_argument("--cycles", type=int, default=1)
    p_ab.add_argument("--d1-share-a", type=float, default=0.90)
    p_ab.add_argument("--d1-share-b", type=float, default=0.75)
    p_ab.add_argument("--out", default="results/ab_safe_report.json")
    p_ab.add_argument("--skip-install", action="store_true")

    p_rag = sub.add_parser("compare-rag", help="A/B test: Normal vs RAG generation")
    p_rag.add_argument("--profile", choices=["local", "vps", "gha"], default="local")
    p_rag.add_argument("--score", type=float, default=50.0)
    p_rag.add_argument(
        "--minutes-per-leg", type=int, default=180, help="minutes per leg (default 3h)"
    )
    p_rag.add_argument("--out", default="results/ab_rag_report.json")
    p_rag.add_argument("--skip-install", action="store_true")

    p_tlog = sub.add_parser("thinking-log", help="append developer thinking log entry")
    p_tlog.add_argument("--title", default="Session Note")
    p_tlog.add_argument("--note", required=True)
    p_tlog.add_argument("--tags", default="")

    p_install = sub.add_parser(
        "install-global", help="install global command wrapper (alpha)"
    )
    p_install.add_argument("--name", default="alpha")
    p_install.add_argument(
        "--bin-dir", default="", help="optional custom directory for command wrapper"
    )

    sub.add_parser("test", help="run unit tests")
    sub.add_parser("zip", help="create clean portable zip")

    return parser.parse_args()


def main() -> int:
    raw_argv = sys.argv[1:]
    if raw_argv and raw_argv[0] == "--yolo":
        bootstrap(skip_install=False)
        return run_local_hybrid(profile="local")

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
            return run_async_pipeline(
                limit=args.limit,
                score=args.score,
                use_rag=args.use_rag,
                profile=args.profile,
            )

        if command == "auto":
            bootstrap(skip_install=args.skip_install)
            if args.profile == "gha":
                limit = int(os.getenv("GHA_BURST_LIMIT", "80"))
                score = float(os.getenv("GHA_PRE_RANK_SCORE", "50"))
                sync_limit = int(os.getenv("GHA_SYNC_LIMIT", "40"))
                code = run_async_pipeline(
                    limit=limit, score=score, use_rag=False, profile="gha"
                )
                if code != 0:
                    return code
                return run_sync_submit(limit=sync_limit)
            if args.profile == "local" and not args.no_hybrid:
                return run_local_hybrid(profile="local")
            # local/vps defaults run continuous engine mode
            return run_async_pipeline(
                limit=0, score=50.0, use_rag=False, profile=args.profile
            )

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

        if command == "public-report":
            return run_public_report(minutes=args.minutes, out_path=args.out)

        if command == "ab-safe":
            bootstrap(skip_install=args.skip_install)
            return run_ab_safe(
                profile=args.profile,
                score=args.score,
                minutes_per_leg=args.minutes_per_leg,
                cycles=args.cycles,
                d1_share_a=args.d1_share_a,
                d1_share_b=args.d1_share_b,
                out_path=args.out,
            )

        if command == "compare-rag":
            bootstrap(skip_install=args.skip_install)
            return run_ab_rag(
                profile=args.profile,
                score=args.score,
                minutes_per_leg=args.minutes_per_leg,
                out_path=args.out,
            )

        if command == "thinking-log":
            return append_thinking_log(title=args.title, note=args.note, tags=args.tags)

        if command == "install-global":
            custom = Path(args.bin_dir) if args.bin_dir else None
            return install_global_command(bin_dir=custom, name=args.name)

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
