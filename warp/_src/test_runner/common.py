# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data models and helpers for Warp's internal test runner."""

import contextlib
import dataclasses
import enum
import json
import os
import pathlib
import sys
import tempfile
import unittest
from typing import Any

try:
    import coverage

    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

EVENT_HISTORY_LIMIT = 3
PARALLEL_RUN_TIMEOUT = 3600
_WARP_CACHE_PATH_ENV = "WARP_CACHE_PATH"


def warn(message: str) -> None:
    """Print a best-effort diagnostic warning without masking the test run."""
    try:
        print(f"Warp warning: {message}", file=sys.stderr)
    except Exception:
        pass


def get_gil_enabled() -> bool | None:
    """Return the current process GIL state when CPython exposes it."""
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is None:
        return None
    return bool(is_gil_enabled())


def get_warp_cache_base_path() -> str | None:
    cache_path = os.environ.get(_WARP_CACHE_PATH_ENV)
    if cache_path is None:
        return None
    cache_path = cache_path.strip()
    return cache_path or None


def iter_test_cases(test_suite):
    """Iterate the test cases (methods) contained in a test suite."""
    if isinstance(test_suite, unittest.TestCase):
        yield test_suite
    else:
        for suite in test_suite:
            yield from iter_test_cases(suite)


def suite_name_for(test_suite, unit_type: str = "class") -> str:
    """Return a human-readable name for the submitted parallelization unit."""
    first_test = next(iter_test_cases(test_suite), None)
    if first_test is None:
        return "unknown"
    if unit_type == "module":
        return type(first_test).__module__
    return type(first_test).__name__


@contextlib.contextmanager
def coverage_context(args, temp_dir):
    """Measure coverage around the wrapped block when ``--coverage`` is active."""
    if args.coverage:
        # The file is deleted along with the containing temporary directory.
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as coverage_file:
            pass
        cov = coverage.Coverage(
            branch=args.coverage_branch,
            data_file=coverage_file.name,
            config_file=True,  # Configuration lives in pyproject.toml (requires coverage[toml]).
        )
        try:
            cov.start()
            yield cov
        finally:
            cov.stop()
            cov.save()
    else:
        yield None


def _atomic_write(path: pathlib.Path, write_payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            write_payload(stream)
            stream.flush()
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path, payload: Any) -> None:
    def write_payload(stream):
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

    _atomic_write(pathlib.Path(path), write_payload)


def atomic_write_text(path, contents: str) -> None:
    def write_payload(stream):
        stream.write(contents)
        if not contents.endswith("\n"):
            stream.write("\n")

    _atomic_write(pathlib.Path(path), write_payload)


class EventKind(str, enum.Enum):
    WORKER_STARTED = "worker_started"
    WORKER_INITIALIZED = "worker_initialized"
    GIL_STATE_CHANGED = "gil_state_changed"
    SUITE_STARTED = "suite_started"
    TEST_STARTED = "test_started"
    TEST_OUTCOME = "test_outcome"
    TEST_CLEANUP_STARTED = "test_cleanup_started"
    TEST_STOPPED = "test_stopped"
    SUITE_FINALIZING = "suite_finalizing"
    SUITE_FINISHED = "suite_finished"
    WORKER_SHUTDOWN = "worker_shutdown"


class ProcessExitProvenance(str, enum.Enum):
    PARENT_TERMINATED = "parent_terminated"
    INDEPENDENTLY_ABNORMAL = "independently_abnormal"
    UNRESOLVED = "unresolved"


@dataclasses.dataclass(frozen=True)
class WorkerEvent:
    sequence: int
    event: EventKind
    worker_index: int
    pid: int
    monotonic_ns: int
    wall_time_ns: int
    suite_index: int | None = None
    suite_name: str | None = None
    test_id: str | None = None
    outcome: str | None = None
    elapsed_seconds: float | None = None
    test_count: int | None = None
    gil_enabled: bool | None = None
    previous_gil_enabled: bool | None = None
    observed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["event"] = self.event.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkerEvent":
        field_names = {field.name for field in dataclasses.fields(cls)}
        values = {name: value for name, value in payload.items() if name in field_names}
        values["event"] = EventKind(values["event"])
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class CompletedTest:
    test_id: str
    outcome: str | None
    elapsed_seconds: float | None


@dataclasses.dataclass(frozen=True)
class GilStateChange:
    previous_gil_enabled: bool
    gil_enabled: bool
    observed_at: str
    suite_index: int | None
    suite_name: str | None
    test_id: str | None


@dataclasses.dataclass(frozen=True)
class WorkerSnapshot:
    worker_index: int
    pid: int
    phase: str
    suite_index: int | None
    suite_name: str | None
    current_test_id: str | None
    current_outcome: str | None
    current_elapsed_seconds: float | None
    last_transition_ns: int
    age_seconds: float
    transition_age_seconds: float
    recent_tests: tuple[CompletedTest, ...]
    started_suite_indexes: tuple[int, ...]
    initial_gil_enabled: bool | None
    gil_enabled: bool | None
    first_gil_state_change: GilStateChange | None
    last_non_ok_test: CompletedTest | None


@dataclasses.dataclass(frozen=True)
class ProcessExit:
    pid: int
    exit_code: int | None
    signal_name: str | None
    provenance: str


@dataclasses.dataclass(frozen=True)
class SuiteClassification:
    suite_index: int
    suite_name: str
    test_count: int
    status: str
    worker_index: int | None = None
    pid: int | None = None


@dataclasses.dataclass(frozen=True)
class SuiteTiming:
    suite_index: int
    suite_name: str
    unit_type: str
    test_count: int
    worker_index: int | None
    pid: int | None
    started_offset_seconds: float | None
    finished_offset_seconds: float | None
    elapsed_seconds: float | None
    completion_order: int | None
    status: str
    outcomes: dict[str, int]


@dataclasses.dataclass(frozen=True)
class PoolFailure:
    exception_type: str
    reason: str
    snapshot: dict[str, Any]
    formatted_summary: str


@dataclasses.dataclass(frozen=True)
class ParallelRunResult:
    results_by_index: dict[int, tuple]
    pool_failure: PoolFailure | None
    suite_classifications: tuple[SuiteClassification, ...]
    diagnostics_degraded: bool = False
