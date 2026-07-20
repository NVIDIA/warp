# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic artifacts produced by Warp's internal test runner."""

import dataclasses
import os
import pathlib
import platform
import re
import time
from typing import Any

from warp._src.test_runner.common import atomic_write_json, warn

DIAGNOSTICS_DIR_ENV = "WARP_TEST_DIAGNOSTICS_DIR"


def resolve_diagnostics_root(cli_value: str | None) -> pathlib.Path | None:
    value = cli_value if cli_value is not None else os.environ.get(DIAGNOSTICS_DIR_ENV)
    if value is None or not value.strip():
        return None
    return pathlib.Path(value.strip())


def create_diagnostics_run_dir(root: pathlib.Path, wall_time_ns: int, parent_pid: int) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(wall_time_ns / 1_000_000_000))
    base_name = f"run-{timestamp}-{parent_pid}"
    suffix = 0
    while True:
        name = base_name if suffix == 0 else f"{base_name}-{suffix}"
        run_dir = root / name
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return run_dir


def build_run_metadata(
    args,
    process_count: int,
    run_start_monotonic_ns: int,
    run_start_wall_time_ns: int,
    parent_gil_enabled_initial: bool | None,
    finished: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build safe run metadata from an explicit system and environment allowlist."""
    ci_providers = []
    if os.environ.get("GITHUB_ACTIONS") == "true":
        ci_providers.append("GitHub Actions")
    if os.environ.get("GITLAB_CI") == "true":
        ci_providers.append("GitLab CI")
    if os.environ.get("BUILDKITE") == "true":
        ci_providers.append("Buildkite")
    if os.environ.get("TF_BUILD") == "True":
        ci_providers.append("Azure Pipelines")
    if os.environ.get("JENKINS_URL") is not None:
        ci_providers.append("Jenkins")

    metadata = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "start_method": "spawn",
        "worker_count": process_count,
        "level": args.level,
        "selected_suite": args.suite,
        "selected_pattern": args.pattern,
        "test_name_patterns": list(args.testNamePatterns or ()),
        "coverage": bool(args.coverage),
        "coverage_branch": bool(args.coverage_branch),
        "debug": bool(args.warp_debug),
        "ci_providers": ci_providers,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "warp_cache_path": os.environ.get("WARP_CACHE_PATH"),
        "parent_pid": os.getpid(),
        "run_start_monotonic_ns": run_start_monotonic_ns,
        "run_start_wall_time_ns": run_start_wall_time_ns,
    }
    metadata["parent_gil_enabled_initial"] = parent_gil_enabled_initial
    if finished is not None:
        metadata.update(finished)
    return metadata


def write_suite_timings(run_dir, metadata, records) -> None:
    run_path = pathlib.Path(run_dir)
    atomic_write_json(
        run_path / "suite-timings.json",
        {
            "metadata": metadata,
            "suites": [dataclasses.asdict(record) for record in records],
        },
    )


def format_slowest_suites(records, limit: int = 20) -> str:
    completed = sorted(
        (record for record in records if record.status == "complete" and record.elapsed_seconds is not None),
        key=lambda record: (-record.elapsed_seconds, record.suite_index),
    )[:limit]
    incomplete = sorted(
        (record for record in records if record.status != "complete" or record.elapsed_seconds is None),
        key=lambda record: record.suite_index,
    )
    lines = [f"Slowest {limit} completed suites:"]
    if completed:
        for rank, record in enumerate(completed, start=1):
            worker = f"worker {record.worker_index:02d}" if record.worker_index is not None else "no worker"
            lines.append(
                f"  {rank}. {record.suite_name} [{record.unit_type}] {record.test_count} tests "
                f"on {worker} in {record.elapsed_seconds:.3f}s"
            )
    else:
        lines.append("  none")

    lines.extend(("", "Incomplete suites:"))
    if incomplete:
        for record in incomplete:
            worker = f"worker {record.worker_index:02d}" if record.worker_index is not None else "no worker"
            lines.append(f"  {record.suite_name} [{record.unit_type}] status={record.status}; {worker}")
    else:
        lines.append("  none")
    return "\n".join(lines)


_WORKER_EVIDENCE_PATTERN = re.compile(r"worker-\d+-(?:\d+\.events\.jsonl|pid-\d+\.(?:output|fault)\.log)")


def finalize_diagnostics(run_dir, retain_worker_evidence: bool) -> None:
    """Remove generated detailed evidence after a clean run only."""
    if run_dir is None or retain_worker_evidence:
        return

    run_path = pathlib.Path(run_dir)
    try:
        candidates = tuple(run_path.iterdir())
    except OSError as error:
        warn(f"Failed to inspect diagnostics run directory {run_path}: {error}")
        return

    for path in candidates:
        if not (
            _WORKER_EVIDENCE_PATTERN.fullmatch(path.name) or path.name in {"crash-snapshot.json", "pool-failure.txt"}
        ):
            continue
        try:
            path.unlink()
        except OSError as error:
            warn(f"Failed to remove clean-run diagnostic artifact {path}: {error}")
