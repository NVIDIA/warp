# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postmortem attribution and reporting for Warp's test runner."""

import dataclasses
import os
import pathlib
import sys
from typing import Any

from warp._src.test_runner.common import (
    EVENT_HISTORY_LIMIT,
    ProcessExitProvenance,
    SuiteClassification,
    atomic_write_json,
    atomic_write_text,
    suite_name_for,
)


def classify_suites(test_suites, results_by_index, future_states, snapshots, unit_type: str = "class"):
    """Classify every discovered suite using lifecycle evidence in priority order."""
    classifications = []
    for suite_index, suite in enumerate(test_suites):
        state = future_states.get(suite_index)
        worker_index = None
        pid = None
        if suite_index in results_by_index and state != "skipped_by_failfast":
            status = "confirmed"
        else:
            snapshot = next((item for item in snapshots if item.suite_index == suite_index), None)
            if snapshot is None:
                snapshot = next(
                    (item for item in snapshots if suite_index in item.started_suite_indexes),
                    None,
                )
            if snapshot is not None:
                status = "started"
                worker_index = snapshot.worker_index
                pid = snapshot.pid
            elif state in {"cancelled", "skipped_by_failfast"}:
                status = "skipped"
            else:
                status = "never_started"
        classifications.append(
            SuiteClassification(
                suite_index=suite_index,
                suite_name=suite_name_for(suite, unit_type),
                test_count=suite.countTestCases(),
                status=status,
                worker_index=worker_index,
                pid=pid,
            )
        )
    return tuple(classifications)


def _worker_candidate(worker: dict[str, Any]) -> dict[str, str | None]:
    phase = worker["phase"]
    if phase == "initializing":
        return {"kind": "initializer", "name": "worker initialization"}
    if phase in {"test_running", "test_outcome", "test_cleanup"} and worker.get("current_test_id"):
        return {"kind": "test", "name": worker["current_test_id"]}
    if phase in {"test_stopped", "suite_finalizing"} and worker.get("suite_name"):
        return {"kind": "suite_finalization", "name": worker["suite_name"]}
    if worker.get("suite_name"):
        return {"kind": "suite", "name": worker["suite_name"]}
    return {"kind": "none", "name": None}


def _artifact_evidence(run_dir: pathlib.Path | None, worker_index: int, pid: int) -> dict[str, dict[str, Any]]:
    names = {
        "journal": f"worker-{worker_index}-{pid}.events.jsonl",
        "output": f"worker-{worker_index}-pid-{pid}.output.log",
        "fault": f"worker-{worker_index}-pid-{pid}.fault.log",
    }
    artifacts = {}
    for name, file_name in names.items():
        evidence = {"path": None, "state": "missing", "size_bytes": None}
        if run_dir is not None:
            path = run_dir / file_name
            evidence["path"] = str(path)
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                pass
            except OSError:
                evidence["state"] = "unreadable"
            else:
                evidence["size_bytes"] = size
                evidence["state"] = "empty" if size == 0 else "non_empty"
        if name == "fault":
            state = evidence["state"]
            evidence["fatal_traceback_evidence"] = {"non_empty": True, "empty": False}.get(state)
        artifacts[name] = evidence
    return artifacts


def build_crash_snapshot(failure, classifications, snapshots, process_exits, run_dir):
    """Build a JSON-serializable postmortem without inferring crash causality."""
    run_path = pathlib.Path(run_dir) if run_dir is not None else None
    exits = sorted((dataclasses.asdict(item) for item in process_exits), key=lambda item: item["pid"])
    exits_by_pid = {item["pid"]: item for item in exits}
    default_exit = {
        "exit_code": None,
        "signal_name": None,
        "provenance": ProcessExitProvenance.UNRESOLVED.value,
    }

    workers = []
    tracked_pids = set()
    for snapshot in snapshots:
        tracked_pids.add(snapshot.pid)
        worker = dataclasses.asdict(snapshot)
        worker["recent_tests"] = [dataclasses.asdict(test) for test in snapshot.recent_tests]
        worker["started_suite_indexes"] = list(snapshot.started_suite_indexes)
        process_exit = exits_by_pid.get(snapshot.pid, default_exit)
        worker.update({name: value for name, value in process_exit.items() if name != "pid"})
        worker["artifacts"] = _artifact_evidence(run_path, snapshot.worker_index, snapshot.pid)
        workers.append(worker)

    for process_exit in exits:
        if process_exit["pid"] in tracked_pids:
            continue
        workers.append(
            {
                "worker_index": None,
                "pid": process_exit["pid"],
                "phase": "unregistered",
                "suite_index": None,
                "suite_name": None,
                "current_test_id": None,
                "current_outcome": None,
                "current_elapsed_seconds": None,
                "last_transition_ns": None,
                "age_seconds": None,
                "transition_age_seconds": None,
                "recent_tests": [],
                "started_suite_indexes": [],
                "initial_gil_enabled": None,
                "gil_enabled": None,
                "first_gil_state_change": None,
                "last_non_ok_test": None,
                **{name: value for name, value in process_exit.items() if name != "pid"},
                "artifacts": _artifact_evidence(None, 0, process_exit["pid"]),
            }
        )

    workers.sort(key=lambda worker: (worker["worker_index"] is None, worker["worker_index"] or 0, worker["pid"]))
    exception_type = type(failure).__name__
    reason = str(failure) or exception_type
    confirmed_indexes = [item.suite_index for item in classifications if item.status == "confirmed"]
    return {
        "failure": {"type": exception_type, "reason": reason},
        "suite_counts": {
            "discovered": len(classifications),
            "confirmed": len(confirmed_indexes),
        },
        "workers": workers,
        "process_exits": exits,
        "suite_classifications": [dataclasses.asdict(item) for item in classifications],
        "diagnostics": {
            "run_dir": str(run_path) if run_path is not None else None,
            "crash_snapshot": str(run_path / "crash-snapshot.json") if run_path is not None else None,
            "pool_failure": str(run_path / "pool-failure.txt") if run_path is not None else None,
        },
    }


_REPORT_PRIORITY = {
    ProcessExitProvenance.INDEPENDENTLY_ABNORMAL.value: 0,
    ProcessExitProvenance.UNRESOLVED.value: 1,
    ProcessExitProvenance.PARENT_TERMINATED.value: 2,
}

# A cleanup signal lands on "unresolved" for a different reason per signal, so
# each note points at the evidence that actually separates the candidates.
_UNRESOLVED_SIGNAL_NOTES = {
    "SIGKILL": (
        "SIGKILL exits are unresolved because the pool's own forced cleanup and an "
        "external killer such as the OOM killer are indistinguishable here; compare the "
        "fault and output logs against the job's memory headroom."
    ),
    "SIGTERM": (
        "SIGTERM exits are unresolved only when the worker's journal did not show work "
        "in progress; check its journal position to see whether it had already finished."
    ),
}


def _workers_by_report_priority(snapshot) -> list[dict[str, Any]]:
    """Return workers in causal-evidence-first presentation order."""
    return sorted(
        snapshot["workers"],
        key=lambda worker: (
            _REPORT_PRIORITY.get(worker["provenance"], len(_REPORT_PRIORITY)),
            worker["worker_index"] is None,
            worker["worker_index"] if worker["worker_index"] is not None else 0,
            worker["pid"],
        ),
    )


def _gil_state_text(gil_enabled: bool | None) -> str:
    if gil_enabled is None:
        return "unknown"
    return "enabled" if gil_enabled else "disabled"


def _format_detailed_worker(worker: dict[str, Any]) -> list[str]:
    identity = f"worker {worker['worker_index']}" if worker["worker_index"] is not None else "unregistered worker"
    candidate = _worker_candidate(worker)
    signal_suffix = f" ({worker['signal_name']})" if worker["signal_name"] is not None else ""
    current_elapsed = worker["current_elapsed_seconds"]
    transition_age = worker["transition_age_seconds"]
    lines = [
        f"  {identity} (PID {worker['pid']}): exit={worker['exit_code']}{signal_suffix}; "
        f"exit provenance={worker['provenance']}",
        f"    last known state={worker['phase']}; suite={worker['suite_name'] or 'none'}; "
        f"candidate={candidate['kind']}:{candidate['name'] or 'none'}; "
        f"current/partial elapsed={f'{current_elapsed:.3f}s' if current_elapsed is not None else 'none'}; "
        f"current outcome={worker['current_outcome'] or 'none'}; "
        f"transition age={f'{transition_age:.3f}s' if transition_age is not None else 'none'}; "
        f"final GIL state={_gil_state_text(worker.get('gil_enabled'))}",
    ]

    gil_change = worker.get("first_gil_state_change")
    if gil_change is not None:
        lines.append(
            "    first observed GIL change="
            f"{_gil_state_text(gil_change['previous_gil_enabled'])}->"
            f"{_gil_state_text(gil_change['gil_enabled'])} at {gil_change['observed_at']}; "
            f"suite={gil_change['suite_name'] or 'none'}; "
            f"last completed test={gil_change['test_id'] or 'none'}"
        )

    last_non_ok = worker.get("last_non_ok_test")
    if last_non_ok is not None:
        duration = last_non_ok["elapsed_seconds"]
        lines.append(
            f"    last non-OK test={last_non_ok['test_id']} outcome={last_non_ok['outcome'] or 'none'} "
            f"duration={f'{duration:.3f}s' if duration is not None else 'none'}"
        )

    recent = worker["recent_tests"][-EVENT_HISTORY_LIMIT:]
    if recent:
        recent_items = []
        for test in recent:
            duration = test["elapsed_seconds"]
            duration_text = f"{duration:.3f}s" if duration is not None else "none"
            recent_items.append(f"{test['test_id']} outcome={test['outcome'] or 'none'} duration={duration_text}")
        formatted_recent = "; ".join(recent_items)
    else:
        formatted_recent = "none"
    lines.append(f"    recent completed tests (up to 3): {formatted_recent}")

    formatted_artifacts = []
    for source in ("journal", "output", "fault"):
        artifact = worker["artifacts"][source]
        artifact_summary = (
            f"{source}={artifact['path'] or 'none'} state={artifact['state']} size={artifact['size_bytes']}"
        )
        if source == "fault":
            fatal = artifact["fatal_traceback_evidence"]
            fatal_text = "unknown" if fatal is None else ("yes" if fatal else "no")
            artifact_summary += f" fatal traceback evidence={fatal_text}"
        formatted_artifacts.append(artifact_summary)
    lines.append("    artifacts: " + "; ".join(formatted_artifacts))
    return lines


def format_pool_failure(snapshot) -> str:
    failure = snapshot["failure"]
    lines = [f"Parallel worker pool failed: {failure['type']}: {failure['reason']}"]
    classifications = snapshot["suite_classifications"]
    confirmed = [item["suite_index"] for item in classifications if item["status"] == "confirmed"]
    lines.append(
        f"  confirmed {len(confirmed)}/{len(classifications)} discovered suites; suite indexes: "
        + (", ".join(str(index) for index in confirmed) or "none")
    )
    workers = _workers_by_report_priority(snapshot)
    parent_terminated = []
    for worker in workers:
        if worker["provenance"] == ProcessExitProvenance.PARENT_TERMINATED.value:
            parent_terminated.append(worker)
            continue
        lines.extend(_format_detailed_worker(worker))

    unresolved_signals = {
        worker["signal_name"]
        for worker in workers
        if worker["provenance"] == ProcessExitProvenance.UNRESOLVED.value
        and worker["signal_name"] in _UNRESOLVED_SIGNAL_NOTES
    }
    for signal_name in sorted(unresolved_signals):
        lines.append(f"  note: {_UNRESOLVED_SIGNAL_NOTES[signal_name]}")

    if parent_terminated:
        identities = []
        for worker in parent_terminated:
            identity = (
                f"worker {worker['worker_index']}" if worker["worker_index"] is not None else "unregistered worker"
            )
            signal_suffix = f", {worker['signal_name']}" if worker["signal_name"] is not None else ""
            gil_suffix = f", final GIL state={_gil_state_text(worker.get('gil_enabled'))}"
            identities.append(f"{identity} (PID {worker['pid']}{signal_suffix}{gil_suffix})")
        lines.append(f"  parent-terminated workers ({len(parent_terminated)}): " + "; ".join(identities))
        snapshot_path = snapshot["diagnostics"]["crash_snapshot"]
        lines.append(f"    complete per-worker state and artifacts: {snapshot_path or 'unavailable'}")
    run_dir = snapshot["diagnostics"]["run_dir"]
    if run_dir is None:
        lines.append("Durable diagnostics disabled; pass --diagnostics-dir to retain worker evidence.")
    else:
        lines.append(f"  durable diagnostics: {run_dir}")
    return "\n".join(lines)


def write_pool_failure_reports(run_dir, snapshot) -> None:
    run_path = pathlib.Path(run_dir)
    atomic_write_json(run_path / "crash-snapshot.json", snapshot)
    atomic_write_text(run_path / "pool-failure.txt", format_pool_failure(snapshot))


def _read_artifact_tail(path: pathlib.Path, limit: int = 32 * 1024) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        length = stream.tell()
        stream.seek(max(0, length - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def print_pool_failure_evidence(snapshot) -> None:
    print(format_pool_failure(snapshot), file=sys.stderr)
    for worker in _workers_by_report_priority(snapshot):
        if worker["provenance"] == ProcessExitProvenance.PARENT_TERMINATED.value:
            continue
        for source in ("fault", "output"):
            artifact = worker["artifacts"][source]
            if artifact["state"] != "non_empty":
                continue
            path = pathlib.Path(artifact["path"])
            try:
                tail = _read_artifact_tail(path)
            except OSError as error:
                print(f"Unable to read worker artifact {path}: {error}", file=sys.stderr)
                continue
            print(
                f"--- worker {worker['worker_index']} PID {worker['pid']} {source} tail "
                f"(last 32768 bytes; complete artifact: {path}) ---",
                file=sys.stderr,
            )
            print(tail, file=sys.stderr, end="" if tail.endswith("\n") else "\n")


def make_pool_failure_test_record(snapshot, diagnostics_path):
    formatted_summary = format_pool_failure(snapshot)
    if diagnostics_path is not None:
        formatted_summary += f"\nComplete crash snapshot: {diagnostics_path}"
    return (
        "warp.tests.parallel",
        "WorkerPoolCrash",
        0.0,
        "ERROR",
        "Parallel worker pool failed",
        formatted_summary,
    )
