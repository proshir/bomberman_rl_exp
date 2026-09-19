#!/usr/bin/env python3
"""Small, durable process controller for Bomberman experiments.

The controller deliberately manages process *groups* rather than matching
process names.  Every started job gets a stable id, a JSON record, and one
combined stdout/stderr log.  Jobs are detached from the terminal, and a job
may depend on one or more earlier jobs.

Examples are documented in ``tools/README.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = Path(os.environ.get("BOMBERMAN_JOBCTL_DIR", ROOT / ".jobctl"))
JOBS_ROOT = STATE_ROOT / "jobs"
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
TERMINAL_STATES = {"succeeded", "failed", "stopped", "blocked"}
ACTIVE_STATES = {"starting", "queued", "running", "stopping"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def die(message: str, code: int = 2) -> None:
    print(f"jobctl: {message}", file=sys.stderr)
    raise SystemExit(code)


def validate_id(job_id: str) -> str:
    if not JOB_ID_RE.fullmatch(job_id):
        die("job id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
    return job_id


def job_dir(job_id: str) -> Path:
    return JOBS_ROOT / validate_id(job_id)


def job_file(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def lock_file(job_id: str) -> Path:
    return job_dir(job_id) / ".lock"


@contextlib.contextmanager
def job_lock(job_id: str):
    path = lock_file(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_job(job_id: str) -> dict[str, Any]:
    path = job_file(job_id)
    if not path.is_file():
        die(f"unknown job id: {job_id}", 1)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        die(f"cannot read {path}: {error}", 1)
    return {}  # unreachable, keeps type checkers happy


def update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    with job_lock(job_id):
        job = read_job(job_id)
        job.update(changes)
        atomic_json_write(job_file(job_id), job)
        return job


def parse_env(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value or not value.split("=", 1)[0]:
            die(f"--env expects KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        result[key] = item
    return result


def pid_starttime(pid: int) -> str | None:
    """Return Linux's process start tick, used to avoid PID-reuse kills."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[21]
    except (FileNotFoundError, IndexError, PermissionError, ValueError):
        return None


def process_alive(pid: int | None, expected_start: str | None = None) -> bool:
    if not pid or not Path(f"/proc/{pid}").exists():
        return False
    return expected_start is None or pid_starttime(pid) == expected_start


def command_for_display(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(
        subprocess.list2cmdline([item]) for item in command
    )


def launch(args: argparse.Namespace) -> None:
    job_id = validate_id(args.id)
    directory = job_dir(job_id)
    if directory.exists():
        die(f"job id already exists: {job_id}")
    if not args.command:
        die("start needs a command after --")
    command = list(args.command)
    if command[0] == "--":
        command = command[1:]
    if not command:
        die("start needs a command after --")

    dependencies = args.after or []
    for dependency in dependencies:
        validate_id(dependency)
        if not job_file(dependency).is_file():
            die(f"dependency does not exist: {dependency}")

    directory.mkdir(parents=True)
    log_path = directory / "stdout.log"
    working_directory = Path(args.cwd).expanduser().resolve()
    progress_files = []
    for raw_path in args.progress_file:
        path = Path(raw_path).expanduser()
        progress_files.append(str(path if path.is_absolute() else working_directory / path))
    metadata: dict[str, Any] = {
        "format": 1,
        "id": job_id,
        "state": "starting",
        "created_at": now(),
        "cwd": str(working_directory),
        "command": command,
        "command_display": command_for_display(command),
        "dependencies": dependencies,
        "environment": parse_env(args.env),
        "log": str(log_path),
        "worker_pid": None,
        "command_pid": None,
        "progress_files": progress_files,
    }
    atomic_json_write(directory / "job.json", metadata)

    worker = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_worker", job_id],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    update_job(
        job_id,
        worker_pid=worker.pid,
        worker_starttime=pid_starttime(worker.pid),
        state="queued" if dependencies else "starting",
    )
    print(job_id)
    print(f"  log: {log_path}")
    if dependencies:
        print(f"  waiting for: {', '.join(dependencies)}")


def dependency_state(job_id: str) -> str:
    try:
        return str(read_job(job_id).get("state", "missing"))
    except SystemExit:
        return "missing"


def worker(job_id: str) -> None:
    job = read_job(job_id)
    dependencies = job.get("dependencies", [])
    while dependencies:
        current = read_job(job_id)
        if current.get("stop_requested"):
            update_job(job_id, state="stopped", finished_at=now(), reason="cancelled while queued")
            return
        states = {dependency: dependency_state(dependency) for dependency in dependencies}
        if any(state in {"failed", "stopped", "blocked"} for state in states.values()):
            update_job(
                job_id,
                state="blocked",
                finished_at=now(),
                reason=f"dependency did not succeed: {states}",
            )
            return
        if all(state == "succeeded" for state in states.values()):
            break
        time.sleep(3)

    current = read_job(job_id)
    if current.get("stop_requested"):
        update_job(job_id, state="stopped", finished_at=now(), reason="cancelled before start")
        return

    update_job(job_id, state="running", started_at=now())
    environment = os.environ.copy()
    environment.update(job.get("environment", {}))
    try:
        cwd = Path(job["cwd"])
        if not cwd.is_dir():
            raise FileNotFoundError(f"working directory does not exist: {cwd}")
        with Path(job["log"]).open("ab", buffering=0) as log:
            log.write(f"\n[jobctl] started {now()}\n".encode())
            process = subprocess.Popen(
                job["command"],
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # The worker already owns a detached session/process group.
                # Keeping the command in that group makes stop reliable even
                # during the tiny window before command_pid is recorded.
                start_new_session=False,
                close_fds=True,
            )
            update_job(
                job_id,
                command_pid=process.pid,
                command_starttime=pid_starttime(process.pid),
                command_pgid=os.getpgid(process.pid),
            )
            return_code = process.wait()
            log.write(f"[jobctl] exited {return_code} at {now()}\n".encode())
    except Exception as error:  # the error is part of the durable log/status
        with Path(job["log"]).open("ab") as log:
            log.write(f"[jobctl] could not start command: {error}\n".encode())
        update_job(job_id, state="failed", return_code=127, finished_at=now(), error=str(error))
        return

    current = read_job(job_id)
    stopped = bool(current.get("stop_requested")) or current.get("state") == "stopping"
    update_job(
        job_id,
        state="stopped" if stopped else ("succeeded" if return_code == 0 else "failed"),
        return_code=return_code,
        finished_at=now(),
    )


def all_processes() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    command = ["ps", "-eo", "pid=,ppid=,pgid=,stat=,etimes=,%cpu=,%mem=,args="]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return result
    for line in output.splitlines():
        fields = line.strip().split(None, 7)
        if len(fields) < 8:
            continue
        try:
            pid = int(fields[0])
            result[pid] = {
                "pid": pid,
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "stat": fields[3],
                "elapsed_seconds": int(fields[4]),
                "cpu": float(fields[5]),
                "memory": float(fields[6]),
                "command": fields[7],
            }
        except ValueError:
            continue
    return result


def process_tree(job: dict[str, Any]) -> list[dict[str, Any]]:
    processes = all_processes()
    roots = {
        int(pid) for pid in (job.get("worker_pid"), job.get("command_pid"))
        if pid
    }
    selected: set[int] = set()
    while roots:
        pid = roots.pop()
        if pid in selected or pid not in processes:
            continue
        selected.add(pid)
        roots.update(item["pid"] for item in processes.values() if item["ppid"] == pid)
    return [processes[pid] for pid in sorted(selected)]


def progress(job: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for raw_path in job.get("progress_files", []):
        path = Path(raw_path).expanduser()
        if not path.is_file():
            values.append(f"{path}: missing")
            continue
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list) and data:
                last = data[-1]
                episode = last.get("episode", last.get("round", "?")) if isinstance(last, dict) else "?"
                values.append(f"{path}: {len(data)} records, latest episode/round {episode}")
            elif isinstance(data, dict):
                values.append(f"{path}: JSON present ({len(data)} keys)")
            else:
                values.append(f"{path}: present")
        except (OSError, json.JSONDecodeError):
            values.append(f"{path}: present but not valid JSON yet")
    return values


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def job_age(job: dict[str, Any]) -> float | None:
    stamp = job.get("started_at") or job.get("created_at")
    if not stamp:
        return None
    try:
        return (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(stamp)).total_seconds()
    except ValueError:
        return None


def tail_lines(path: Path, count: int) -> list[str]:
    if not path.is_file():
        return [f"(log not created yet: {path})"]
    try:
        return path.read_text(errors="replace").splitlines()[-count:]
    except OSError as error:
        return [f"(cannot read log: {error})"]


def print_status(job: dict[str, Any], tail: int = 0) -> None:
    tree = process_tree(job)
    live = [item for item in tree if item["stat"] and "Z" not in item["stat"]]
    cpu = sum(item["cpu"] for item in live)
    memory = sum(item["memory"] for item in live)
    state = job.get("state", "?")
    if state in ACTIVE_STATES and not live:
        state += "/worker-missing"
    print(f"{job['id']}: {state}")
    print(f"  command: {job.get('command_display', '')}")
    print(f"  age: {format_duration(job_age(job))} | processes: {len(live)} | CPU: {cpu:.1f}% | MEM: {memory:.1f}%")
    print(f"  worker pid: {job.get('worker_pid') or '-'} | command pid: {job.get('command_pid') or '-'}")
    if job.get("dependencies"):
        states = ", ".join(f"{item}={dependency_state(item)}" for item in job["dependencies"])
        print(f"  dependencies: {states}")
    if job.get("return_code") is not None:
        print(f"  return code: {job['return_code']}")
    if job.get("reason"):
        print(f"  reason: {job['reason']}")
    print(f"  log: {job.get('log')}")
    for item in progress(job):
        print(f"  progress: {item}")
    if tail:
        print(f"  --- last {tail} log lines ---")
        print("\n".join(tail_lines(Path(job["log"]), tail)))


def list_jobs(args: argparse.Namespace) -> None:
    if not JOBS_ROOT.is_dir():
        print("No jobs.")
        return
    jobs = []
    for path in sorted(JOBS_ROOT.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.is_dir() and (path / "job.json").is_file():
            jobs.append(json.loads((path / "job.json").read_text()))
    if not args.all:
        jobs = [job for job in jobs if job.get("state") in ACTIVE_STATES]
    if not jobs:
        print("No matching jobs.")
        return
    print(f"{'ID':<28} {'STATE':<18} {'PID':>8} {'AGE':>12} COMMAND")
    for job in jobs:
        command = job.get("command_display", "")
        print(f"{job['id']:<28} {job.get('state', '?'):<18} {str(job.get('command_pid') or '-'):>8} {format_duration(job_age(job)):>12} {command[:100]}")


def stop_one(job_id: str, force: bool, timeout: float) -> None:
    job = read_job(job_id)
    if job.get("state") in TERMINAL_STATES:
        print(f"{job_id}: already {job['state']}")
        return
    update_job(job_id, state="stopping", stop_requested=True, stop_requested_at=now())
    # The command and worker each have their own session/process group.  The
    # start-time check prevents a stale job record from killing a reused PID.
    targets = [
        (job.get("command_pid"), job.get("command_starttime")),
        (job.get("worker_pid"), job.get("worker_starttime")),
    ]
    for pid, expected in targets:
        if not process_alive(pid, expected):
            continue
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGKILL if force else signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(process_alive(pid, expected) for pid, expected in targets):
            break
        time.sleep(0.1)
    if not force:
        for pid, expected in targets:
            if process_alive(pid, expected):
                try:
                    os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    update_job(job_id, state="stopped", finished_at=now(), reason="stopped by user")
    print(f"{job_id}: stopped")


def tail_job(args: argparse.Namespace) -> None:
    job = read_job(args.id)
    path = Path(job["log"])
    if not args.follow:
        print("\n".join(tail_lines(path, args.lines)))
        return

    # Print the current window once, then follow by byte offset. Tracking the
    # number of displayed lines would lose output as soon as the window rolls.
    initial = tail_lines(path, args.lines)
    print("\n".join(initial), flush=True)
    try:
        offset = path.stat().st_size
    except FileNotFoundError:
        offset = 0
    while True:
        try:
            with path.open("r", errors="replace") as handle:
                handle.seek(offset)
                new_text = handle.read()
                offset = handle.tell()
            if new_text:
                print(new_text, end="", flush=True)
        except FileNotFoundError:
            pass
        if read_job(args.id).get("state") in TERMINAL_STATES:
            return
        time.sleep(args.interval)


def show_processes(job_id: str) -> None:
    tree = process_tree(read_job(job_id))
    if not tree:
        print("No live processes found.")
        return
    print(f"{'PID':>8} {'PPID':>8} {'PGID':>8} {'CPU':>6} {'MEM':>6} COMMAND")
    for item in tree:
        print(f"{item['pid']:>8} {item['ppid']:>8} {item['pgid']:>8} {item['cpu']:>5.1f}% {item['memory']:>5.1f}% {item['command']}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="action", required=True)

    start = sub.add_parser("start", help="start a detached command")
    start.add_argument("--id", required=True, help="stable id used by status/tail/stop")
    start.add_argument("--after", action="append", default=[], help="wait for this job to succeed; repeatable")
    start.add_argument("--cwd", default=str(ROOT), help="working directory (default: experiment root)")
    start.add_argument("--env", action="append", default=[], help="environment override KEY=VALUE; repeatable")
    start.add_argument("--progress-file", action="append", default=[], help="JSON file to summarize in status; repeatable")
    start.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    start.set_defaults(function=launch)

    listing = sub.add_parser("list", help="list active jobs")
    listing.add_argument("--all", action="store_true", help="include finished jobs")
    listing.set_defaults(function=list_jobs)

    status = sub.add_parser("status", help="show job state, resources, and optional log tail")
    status.add_argument("ids", nargs="*", help="job ids; omit with --all")
    status.add_argument("--all", action="store_true", help="show all jobs")
    status.add_argument("--tail", type=int, default=0, help="include this many log lines")
    status.set_defaults(function=status_command)

    tail = sub.add_parser("tail", help="show or follow a job log")
    tail.add_argument("id")
    tail.add_argument("-n", "--lines", type=int, default=40)
    tail.add_argument("-f", "--follow", action="store_true")
    tail.add_argument("--interval", type=float, default=2.0)
    tail.set_defaults(function=tail_job)

    processes = sub.add_parser("ps", help="show processes belonging to a job")
    processes.add_argument("id")
    processes.set_defaults(function=lambda args: show_processes(args.id))

    stop = sub.add_parser("stop", aliases=["cancel"], help="stop a job and its process group")
    stop.add_argument("ids", nargs="+")
    stop.add_argument("--force", action="store_true", help="send SIGKILL immediately")
    stop.add_argument("--timeout", type=float, default=10.0)
    stop.set_defaults(function=lambda args: [stop_one(item, args.force, args.timeout) for item in args.ids])

    hidden = sub.add_parser("_worker")
    hidden.add_argument("id")
    hidden.set_defaults(function=lambda args: worker(args.id))
    return root


def status_command(args: argparse.Namespace) -> None:
    if args.all:
        if not JOBS_ROOT.is_dir():
            print("No jobs.")
            return
        ids = [path.name for path in JOBS_ROOT.iterdir() if (path / "job.json").is_file()]
    else:
        ids = args.ids
    if not ids:
        die("status needs one or more ids, or --all")
    for index, job_id in enumerate(ids):
        if index:
            print()
        print_status(read_job(job_id), args.tail)


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
