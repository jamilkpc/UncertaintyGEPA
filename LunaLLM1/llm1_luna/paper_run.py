"""Supervise the five full LLM1 benchmarks in an isolated paper output directory."""
from __future__ import annotations

import argparse
import faulthandler
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

from .benchmarks import BENCHMARKS, R
from .llm1_client import CallFailure, load_environment
from .run_llm1 import run_benchmark
from .utils import (DEFAULT_WORKDIR, PREFIX, PROJECT_ROOT, atomic_write,
                    json_dumps, load_benchmark, utc_now)
from .verify_outputs import verify_output

PAPER_DIR = DEFAULT_WORKDIR / f"{PREFIX}_paper"


def read_progress(directory):
    """Read atomic export metadata; never compete with the SQLite writer."""
    progress = {}
    for name in BENCHMARKS:
        source = directory / f"{PREFIX}_inputs_{name}.json"
        expected = len(json.loads(source.read_text())["records"]) if source.exists() else None
        entry = dict(expected_items=expected, completed_items=0, recorded_requests=0,
                     successful_responses=0, item_errors=0)
        path = directory / f"{PREFIX}_{name}.metadata.json"
        if path.exists():
            metadata = json.loads(path.read_text())
            entry.update(completed_items=metadata.get("items_complete", 0),
                         item_errors=metadata.get("item_errors", 0),
                         recorded_requests=metadata.get("usage", {}).get("total_requests", 0),
                         successful_responses=metadata.get("usage", {}).get("successful_responses", 0),
                         as_of=metadata.get("updated_at"))
        progress[name] = entry
    return progress


def run_paper(directory=PAPER_DIR, workers=4):
    # Native faults bypass Python exceptions; retain thread stacks in the run log.
    # faulthandler records stack locations, not local variables or credentials.
    faulthandler.enable(all_threads=True)
    load_environment()
    directory = Path(directory).resolve()
    if directory == DEFAULT_WORKDIR.resolve():
        raise ValueError("Paper runs must use a separate directory from smoke-test outputs.")
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "paper_run.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("The paper run is already running.") from None
    state_lock = threading.Lock()
    stopped = threading.Event()
    state = dict(status="preparing", pid=os.getpid(), started_at=utc_now(),
                 workers=workers, smoke_annotations_reused=False, active_benchmark=None,
                 output_dir=str(directory), benchmark_reports={})
    status_path = directory / "paper_run_status.json"
    manifest_path = directory / "paper_run_manifest.json"
    heartbeat = None
    caffeinate = None

    def save_status():
        progress = read_progress(directory)
        with state_lock:
            state["updated_at"] = utc_now()
            state["progress"] = progress
            atomic_write(status_path, json_dumps(state))

    def monitor():
        while not stopped.wait(30):
            try:
                save_status()
            except (sqlite3.Error, OSError):
                # The runner's ledger remains authoritative if a reader meets a lock.
                pass

    def terminate(signum, frame):
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, terminate)
    exit_code = 0
    try:
        # Copy only immutable source data, never any smoke annotation or call ledger.
        for name in BENCHMARKS:
            source = DEFAULT_WORKDIR / f"{PREFIX}_inputs_{name}.json"
            target = directory / source.name
            if not target.exists():
                if source.exists():
                    atomic_write(target, source.read_text())
                else:
                    load_benchmark(name, directory)
        if not manifest_path.exists():
            expected = {name: len(load_benchmark(name, directory)) for name in BENCHMARKS}
            manifest = dict(created_at=utc_now(), scope="LLM1 only; MHS excluded",
                            fresh_annotations=True, expected_items=expected,
                            expected_successful_responses=sum(expected.values()) * (R + 1),
                            source_files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob("*.py") if not p.name.startswith("._")},
                            input_snapshots_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in directory.glob(f"{PREFIX}_inputs_*.json")})
            atomic_write(manifest_path, json_dumps(manifest))
        # Preserve each launch's source fingerprint, including storage/parser fixes.
        history_path = directory / "paper_run_launch_history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(dict(started_at=state["started_at"], pid=os.getpid(),
                            workers=workers,
                            source_files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob("*.py") if not p.name.startswith("._")}))
        atomic_write(history_path, json_dumps(history))
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
        with state_lock:
            state["status"] = "running"
        save_status()
        heartbeat = threading.Thread(target=monitor, daemon=True)
        heartbeat.start()
        for name in BENCHMARKS:
            with state_lock:
                state["active_benchmark"] = name
            save_status()
            # A bounded extra pass can recover transient failures without replacing
            # any received stochastic samples or completed items.
            previous_complete = -1
            for pass_number in range(1, 4):
                print(f"PAPER RUN {name}, pass {pass_number}", flush=True)
                result = run_benchmark(name, workers=workers, output_dir=directory,
                                       checkpoint_every=25)
                complete = int((result.status == "complete").sum())
                if result.attrs["run_report"]["incomplete_selected"] == 0:
                    break
                if complete <= previous_complete:
                    break
                previous_complete = complete
                if pass_number < 3:
                    time.sleep(15)
            audits = verify_output(name, directory)
            expected = len(load_benchmark(name, directory))
            report = dict(**result.attrs["run_report"], expected_items=expected,
                          completed_items=complete, audited_items=len(audits),
                          logprob_missing=int(result.logprob_var.isna().sum()),
                          logprob_status_counts=result.logprob_status.value_counts().to_dict(),
                          csv=str(directory / f"{PREFIX}_{name}.csv"))
            assert len(audits) == complete
            with state_lock:
                state["benchmark_reports"][name] = report
            save_status()
        with state_lock:
            reports = state["benchmark_reports"]
            all_complete = all(r["completed_items"] == r["expected_items"] for r in reports.values())
            state["status"] = "complete" if all_complete else "needs_review"
            state["active_benchmark"] = None
            state["finished_at"] = utc_now()
            state["total_usage"] = {key: sum(r["cumulative"][key] for r in reports.values())
                                    for key in ("input_tokens", "output_tokens", "cached_input_tokens",
                                                "total_requests", "successful_responses")}
            state["approximate_usd"] = sum(r["approximate_usd"] for r in reports.values())
        exit_code = 0 if all_complete else 2
    except KeyboardInterrupt:
        with state_lock:
            state.update(status="interrupted", finished_at=utc_now())
        exit_code = 130
    except Exception as exc:
        # Persist only sanitized errors; never arbitrary SDK bodies or environment.
        with state_lock:
            state.update(status="failed", finished_at=utc_now(), error_type=type(exc).__name__)
            if isinstance(exc, CallFailure):
                state["error"] = str(exc)
            elif isinstance(exc, sqlite3.Error):
                state["sqlite_error_name"] = getattr(exc, "sqlite_errorname", "unknown")
                state["error"] = str(exc)
        print(f"Paper run stopped: {type(exc).__name__}; received responses are checkpointed.", flush=True)
        exit_code = 2
    finally:
        stopped.set()
        if heartbeat is not None:
            heartbeat.join(timeout=15)
        try:
            save_status()
        finally:
            if caffeinate is not None:
                caffeinate.terminate()
            signal.signal(signal.SIGTERM, old_handler)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--background", action="store_true", help="Detach and write progress to paper_run.log/status.json.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        if args.background:
            load_environment()
            directory = args.output_dir.resolve()
            if directory == DEFAULT_WORKDIR.resolve():
                raise ValueError("Paper results require a separate directory.")
            directory.mkdir(parents=True, exist_ok=True)
            # Refuse a second launch when the supervisor already holds its lock.
            with (directory / "paper_run.lock").open("a+") as guard:
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
            command = [sys.executable, "-u", "-m", "llm1_luna.paper_run",
                       "--workers", str(args.workers), "--output-dir", str(directory)]
            with (directory / "paper_run.log").open("a", buffering=1) as log:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            launch = dict(pid=process.pid, launched_at=utc_now(), output_dir=str(directory),
                          workers=args.workers, log=str(directory / "paper_run.log"))
            atomic_write(directory / "paper_run_launch.json", json_dumps(launch))
            print(json_dumps(launch), flush=True)
            return 0
        return run_paper(args.output_dir, args.workers)
    except Exception as exc:
        print(f"Cannot start paper run: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
