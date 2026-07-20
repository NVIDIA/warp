# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics and worker events for Warp's internal test runner."""

import atexit
import collections
import dataclasses
import faulthandler
import json
import os
import pathlib
import sys
import threading
import time
from typing import Any

from warp._src.test_runner.common import (
    EVENT_HISTORY_LIMIT,
    CompletedTest,
    EventKind,
    GilStateChange,
    SuiteClassification,
    SuiteTiming,
    WorkerEvent,
    WorkerSnapshot,
    get_gil_enabled,
    warn,
)

_GIL_UNOBSERVED = object()


class WorkerEventReporter:
    def __init__(
        self,
        event_queue: Any,
        worker_index: int,
        run_dir: pathlib.Path | None,
        run_start_monotonic_ns: int,
    ):
        self._event_queue = event_queue
        self._worker_index = worker_index
        self._run_start_monotonic_ns = run_start_monotonic_ns
        self._sequence = 0
        self._journal_fd: int | None = None
        self._suite_index: int | None = None
        self._suite_name: str | None = None
        self._suite_test_count: int | None = None
        self._suite_started_monotonic_ns: int | None = None
        self._last_gil_enabled: bool | object | None = _GIL_UNOBSERVED
        self._last_completed_test_id: str | None = None

        if run_dir is not None:
            journal_path = run_dir / f"worker-{worker_index}-{os.getpid()}.events.jsonl"
            try:
                self._journal_fd = os.open(
                    journal_path,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o644,
                )
            except Exception as error:
                warn(f"Failed to open worker event journal {journal_path}: {error}")

    def emit(self, kind: EventKind, **fields: Any) -> WorkerEvent:
        kind = EventKind(kind)
        monotonic_ns = time.monotonic_ns() - self._run_start_monotonic_ns
        if kind is EventKind.TEST_STOPPED and fields.get("test_id") is not None:
            self._last_completed_test_id = fields["test_id"]
        if kind is EventKind.SUITE_STARTED:
            self._suite_index = fields.get("suite_index")
            self._suite_name = fields.get("suite_name")
            self._suite_test_count = fields.get("test_count")
            self._suite_started_monotonic_ns = monotonic_ns

        if self._suite_index is not None:
            fields.setdefault("suite_index", self._suite_index)
        if self._suite_name is not None:
            fields.setdefault("suite_name", self._suite_name)
        if kind is EventKind.SUITE_FINISHED:
            fields.setdefault("test_count", self._suite_test_count)
            if self._suite_started_monotonic_ns is not None:
                fields.setdefault(
                    "elapsed_seconds",
                    (monotonic_ns - self._suite_started_monotonic_ns) / 1_000_000_000,
                )

        self._sequence += 1
        event = WorkerEvent(
            sequence=self._sequence,
            event=kind,
            worker_index=self._worker_index,
            pid=os.getpid(),
            monotonic_ns=monotonic_ns,
            wall_time_ns=time.time_ns(),
            **fields,
        )

        if kind is EventKind.SUITE_FINISHED:
            self._suite_index = None
            self._suite_name = None
            self._suite_test_count = None
            self._suite_started_monotonic_ns = None

        if self._event_queue is not None:
            try:
                self._event_queue.put(event)
            except Exception as error:
                self._event_queue = None
                warn(f"Failed to send worker event: {error}")

        if self._journal_fd is not None:
            payload = (json.dumps(event.to_dict(), sort_keys=True) + "\n").encode()
            try:
                written = os.write(self._journal_fd, payload)
                if written != len(payload):
                    raise OSError(f"wrote {written} of {len(payload)} bytes")
            except Exception as error:
                self._disable_journal()
                warn(f"Failed to append worker event journal: {error}")

        return event

    def emit_gil_observed(self, kind: EventKind, **fields: Any) -> WorkerEvent:
        """Emit a worker boundary annotated with the process-local GIL state."""
        kind = EventKind(kind)
        gil_enabled = get_gil_enabled()
        previous_gil_enabled = self._last_gil_enabled
        if (
            previous_gil_enabled is not _GIL_UNOBSERVED
            and previous_gil_enabled is not None
            and gil_enabled is not None
            and gil_enabled != previous_gil_enabled
        ):
            change_fields = dict(fields)
            if self._last_completed_test_id is not None:
                change_fields.setdefault("test_id", self._last_completed_test_id)
            self.emit(
                EventKind.GIL_STATE_CHANGED,
                previous_gil_enabled=previous_gil_enabled,
                gil_enabled=gil_enabled,
                observed_at=kind.value,
                **change_fields,
            )
        self._last_gil_enabled = gil_enabled
        return self.emit(kind, gil_enabled=gil_enabled, **fields)

    def _disable_journal(self) -> None:
        journal_fd = self._journal_fd
        self._journal_fd = None
        if journal_fd is not None:
            try:
                os.close(journal_fd)
            except OSError:
                pass

    def close(self) -> None:
        self._disable_journal()


_worker_reporter: WorkerEventReporter | None = None
_worker_output_file = None
_worker_fault_file = None
_worker_faulthandler_enabled = False


def install_worker_reporter(reporter: WorkerEventReporter | None) -> None:
    global _worker_reporter
    _worker_reporter = reporter


def emit_worker_event(kind: EventKind, **fields: Any) -> WorkerEvent | None:
    if _worker_reporter is None:
        return None
    return _worker_reporter.emit(kind, **fields)


def emit_worker_event_with_gil(kind: EventKind, **fields: Any) -> WorkerEvent | None:
    """Emit a worker boundary after observing the process-local GIL state."""
    if _worker_reporter is None:
        return None
    return _worker_reporter.emit_gil_observed(kind, **fields)


def _test_id(test) -> str:
    identifier = getattr(test, "id", None)
    return identifier() if identifier is not None else str(test)


def emit_test_started(test) -> int:
    started_ns = time.monotonic_ns()
    emit_worker_event(
        EventKind.TEST_STARTED,
        test_id=_test_id(test),
    )
    return started_ns


def emit_test_outcome(test, outcome: str, started_ns: int) -> None:
    emit_worker_event(
        EventKind.TEST_OUTCOME,
        test_id=_test_id(test),
        outcome=outcome,
        elapsed_seconds=(time.monotonic_ns() - started_ns) * 1.0e-9,
    )


def emit_test_cleanup_started(test) -> None:
    emit_worker_event(EventKind.TEST_CLEANUP_STARTED, test_id=_test_id(test))


def emit_test_stopped(test) -> None:
    emit_worker_event(EventKind.TEST_STOPPED, test_id=_test_id(test))


def configure_worker_diagnostics(
    event_queue,
    worker_index: int,
    run_dir: pathlib.Path | None,
    run_start_monotonic_ns: int,
) -> WorkerEventReporter:
    global _worker_faulthandler_enabled, _worker_fault_file, _worker_output_file

    reporter = WorkerEventReporter(
        event_queue=event_queue,
        worker_index=worker_index,
        run_dir=run_dir,
        run_start_monotonic_ns=run_start_monotonic_ns,
    )

    install_worker_reporter(reporter)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as error:
        warn(f"Failed to flush worker output before redirection: {error}")

    if run_dir is not None:
        pid = os.getpid()
        output_path = run_dir / f"worker-{worker_index}-pid-{pid}.output.log"
        output_file = None
        try:
            output_file = open(output_path, "ab", buffering=0)
            os.dup2(output_file.fileno(), 1)
            os.dup2(output_file.fileno(), 2)
        except OSError as error:
            warn(f"Failed to configure worker output sink {output_path}: {error}")
        finally:
            _worker_output_file = output_file

        fault_path = run_dir / f"worker-{worker_index}-pid-{pid}.fault.log"
        fault_file = None
        try:
            fault_file = open(fault_path, "ab", buffering=0)
            faulthandler.disable()
            faulthandler.enable(file=fault_file, all_threads=True)
        except (OSError, RuntimeError, ValueError) as error:
            warn(f"Failed to configure faulthandler sink {fault_path}: {error}")
        else:
            _worker_faulthandler_enabled = True
        finally:
            _worker_fault_file = fault_file

    emit_worker_event(EventKind.WORKER_STARTED)
    atexit.register(close_worker_diagnostics)
    return reporter


def close_worker_diagnostics() -> None:
    global _worker_faulthandler_enabled, _worker_fault_file, _worker_output_file

    reporter = _worker_reporter
    if reporter is None:
        return

    try:
        emit_worker_event(EventKind.WORKER_SHUTDOWN)
    except BaseException:
        pass

    if _worker_faulthandler_enabled:
        try:
            faulthandler.disable()
        except BaseException:
            pass
        _worker_faulthandler_enabled = False

    reporter.close()
    install_worker_reporter(None)

    fault_file = _worker_fault_file
    _worker_fault_file = None
    output_file = _worker_output_file
    _worker_output_file = None
    for handle in (fault_file, output_file):
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass


PHASE_BY_EVENT = {
    EventKind.WORKER_STARTED: "initializing",
    EventKind.WORKER_INITIALIZED: "idle",
    EventKind.SUITE_STARTED: "suite_running",
    EventKind.TEST_STARTED: "test_running",
    EventKind.TEST_OUTCOME: "test_outcome",
    EventKind.TEST_CLEANUP_STARTED: "test_cleanup",
    EventKind.TEST_STOPPED: "test_stopped",
    EventKind.SUITE_FINALIZING: "suite_finalizing",
    EventKind.SUITE_FINISHED: "idle",
    EventKind.WORKER_SHUTDOWN: "shutdown",
}


@dataclasses.dataclass
class _WorkerState:
    worker_index: int
    pid: int
    phase: str
    suite_index: int | None
    suite_name: str | None
    current_test_id: str | None
    current_outcome: str | None
    current_test_started_ns: int | None
    current_outcome_elapsed_seconds: float | None
    last_transition_ns: int
    recent_tests: collections.deque[CompletedTest]
    started_suite_indexes: set[int]
    initial_gil_enabled: bool | None
    gil_enabled: bool | None
    first_gil_state_change: GilStateChange | None
    last_non_ok_test: CompletedTest | None
    last_sequence: int


@dataclasses.dataclass
class _SuiteTimingState:
    suite_index: int
    suite_name: str
    test_count: int
    worker_index: int
    pid: int
    started_monotonic_ns: int
    finished_monotonic_ns: int | None
    completion_order: int | None
    outcomes: collections.Counter[str]


class WorkerStateTracker:
    def __init__(self, history_limit: int = EVENT_HISTORY_LIMIT):
        self._history_limit = history_limit
        self._states: dict[int, _WorkerState] = {}
        self._suite_states: dict[int, _SuiteTimingState] = {}
        self._completion_count = 0
        self._lock = threading.Lock()
        self._monitor_error: BaseException | None = None

    def handle_event(self, event: WorkerEvent) -> None:
        with self._lock:
            state = self._states.get(event.worker_index)
            if state is None:
                state = _WorkerState(
                    worker_index=event.worker_index,
                    pid=event.pid,
                    phase=PHASE_BY_EVENT.get(event.event, "unknown"),
                    suite_index=event.suite_index,
                    suite_name=event.suite_name,
                    current_test_id=None,
                    current_outcome=None,
                    current_test_started_ns=None,
                    current_outcome_elapsed_seconds=None,
                    last_transition_ns=event.monotonic_ns,
                    recent_tests=collections.deque(maxlen=self._history_limit),
                    started_suite_indexes=set(),
                    initial_gil_enabled=None,
                    gil_enabled=None,
                    first_gil_state_change=None,
                    last_non_ok_test=None,
                    last_sequence=event.sequence,
                )
                self._states[event.worker_index] = state
            elif event.sequence <= state.last_sequence:
                return
            else:
                state.last_sequence = event.sequence

            state.pid = event.pid
            if event.event is EventKind.GIL_STATE_CHANGED:
                state.gil_enabled = event.gil_enabled
                if (
                    state.first_gil_state_change is None
                    and event.previous_gil_enabled is not None
                    and event.gil_enabled is not None
                    and event.observed_at is not None
                ):
                    recent_test_id = state.recent_tests[-1].test_id if state.recent_tests else None
                    state.first_gil_state_change = GilStateChange(
                        previous_gil_enabled=event.previous_gil_enabled,
                        gil_enabled=event.gil_enabled,
                        observed_at=event.observed_at,
                        suite_index=event.suite_index,
                        suite_name=event.suite_name,
                        test_id=event.test_id or recent_test_id,
                    )
                return

            state.phase = PHASE_BY_EVENT[event.event]
            state.last_transition_ns = event.monotonic_ns
            if event.gil_enabled is not None:
                state.gil_enabled = event.gil_enabled
                if event.event is EventKind.WORKER_INITIALIZED:
                    state.initial_gil_enabled = event.gil_enabled

            if event.event is EventKind.WORKER_STARTED:
                state.suite_index = None
                state.suite_name = None
                state.current_test_id = None
                state.current_outcome = None
                state.current_test_started_ns = None
                state.current_outcome_elapsed_seconds = None
            elif event.event is EventKind.WORKER_INITIALIZED:
                state.current_test_id = None
                state.current_outcome = None
                state.current_test_started_ns = None
                state.current_outcome_elapsed_seconds = None
            elif event.event is EventKind.SUITE_STARTED:
                state.suite_index = event.suite_index
                state.suite_name = event.suite_name
                state.current_test_id = None
                state.current_outcome = None
                state.current_test_started_ns = None
                state.current_outcome_elapsed_seconds = None
                if event.suite_index is not None:
                    state.started_suite_indexes.add(event.suite_index)
                    self._suite_states[event.suite_index] = _SuiteTimingState(
                        suite_index=event.suite_index,
                        suite_name=event.suite_name or "unknown",
                        test_count=event.test_count or 0,
                        worker_index=event.worker_index,
                        pid=event.pid,
                        started_monotonic_ns=event.monotonic_ns,
                        finished_monotonic_ns=None,
                        completion_order=None,
                        outcomes=collections.Counter(),
                    )
            elif event.event is EventKind.TEST_STARTED:
                state.current_test_id = event.test_id
                state.current_outcome = None
                state.current_test_started_ns = event.monotonic_ns
                state.current_outcome_elapsed_seconds = None
            elif event.event is EventKind.TEST_OUTCOME:
                state.current_test_id = event.test_id
                state.current_outcome = event.outcome
                state.current_outcome_elapsed_seconds = event.elapsed_seconds
                if event.test_id is not None and event.outcome in {"ERROR", "FAIL"}:
                    state.last_non_ok_test = CompletedTest(
                        test_id=event.test_id,
                        outcome=event.outcome,
                        elapsed_seconds=event.elapsed_seconds,
                    )
                if state.suite_index is not None and event.outcome is not None:
                    timing_state = self._suite_states.get(state.suite_index)
                    if timing_state is not None:
                        timing_state.outcomes[event.outcome] += 1
            elif event.event is EventKind.TEST_CLEANUP_STARTED:
                state.current_test_id = event.test_id
            elif event.event is EventKind.TEST_STOPPED:
                test_id = event.test_id or state.current_test_id
                if test_id is not None:
                    state.recent_tests.append(
                        CompletedTest(
                            test_id=test_id,
                            outcome=event.outcome or state.current_outcome,
                            elapsed_seconds=(
                                event.elapsed_seconds
                                if event.elapsed_seconds is not None
                                else state.current_outcome_elapsed_seconds
                            ),
                        )
                    )
                state.current_test_id = None
                state.current_outcome = None
                state.current_test_started_ns = None
                state.current_outcome_elapsed_seconds = None
            elif event.event is EventKind.SUITE_FINALIZING:
                state.current_test_id = None
                state.current_outcome = None
                state.current_test_started_ns = None
                state.current_outcome_elapsed_seconds = None
            elif event.event is EventKind.SUITE_FINISHED:
                if state.suite_index is not None:
                    timing_state = self._suite_states.get(state.suite_index)
                    if timing_state is not None:
                        self._completion_count += 1
                        timing_state.finished_monotonic_ns = event.monotonic_ns
                        timing_state.completion_order = self._completion_count
                state.suite_index = None
                state.suite_name = None
                state.current_test_id = None
                state.current_outcome = None
                state.current_test_started_ns = None
                state.current_outcome_elapsed_seconds = None

    def suite_timings(
        self,
        classifications: tuple[SuiteClassification, ...] = (),
        unit_type: str = "class",
    ) -> tuple[SuiteTiming, ...]:
        """Return timing records merged with weaker suite classifications."""
        with self._lock:
            classifications_by_index = {item.suite_index: item for item in classifications}
            records = []
            for suite_index, timing_state in self._suite_states.items():
                classification = classifications_by_index.get(suite_index)
                finished_ns = timing_state.finished_monotonic_ns
                if finished_ns is not None:
                    status = "complete"
                elif classification is not None and classification.status == "started":
                    status = "started"
                else:
                    status = "incomplete"
                records.append(
                    SuiteTiming(
                        suite_index=suite_index,
                        suite_name=timing_state.suite_name,
                        unit_type=unit_type,
                        test_count=timing_state.test_count,
                        worker_index=timing_state.worker_index,
                        pid=timing_state.pid,
                        started_offset_seconds=timing_state.started_monotonic_ns / 1_000_000_000,
                        finished_offset_seconds=finished_ns / 1_000_000_000 if finished_ns is not None else None,
                        elapsed_seconds=(
                            (finished_ns - timing_state.started_monotonic_ns) / 1_000_000_000
                            if finished_ns is not None
                            else None
                        ),
                        completion_order=timing_state.completion_order,
                        status=status,
                        outcomes=dict(sorted(timing_state.outcomes.items())),
                    )
                )

            observed_indexes = self._suite_states.keys()
            for classification in classifications:
                if classification.suite_index in observed_indexes:
                    continue
                records.append(
                    SuiteTiming(
                        suite_index=classification.suite_index,
                        suite_name=classification.suite_name,
                        unit_type=unit_type,
                        test_count=classification.test_count,
                        worker_index=classification.worker_index,
                        pid=classification.pid,
                        started_offset_seconds=None,
                        finished_offset_seconds=None,
                        elapsed_seconds=None,
                        completion_order=None,
                        status=classification.status,
                        outcomes={},
                    )
                )

            return tuple(sorted(records, key=lambda record: record.suite_index))

    def snapshots(self, now_ns: int) -> tuple[WorkerSnapshot, ...]:
        with self._lock:
            return tuple(
                WorkerSnapshot(
                    worker_index=state.worker_index,
                    pid=state.pid,
                    phase=state.phase,
                    suite_index=state.suite_index,
                    suite_name=state.suite_name,
                    current_test_id=state.current_test_id,
                    current_outcome=state.current_outcome,
                    current_elapsed_seconds=(
                        max(0, now_ns - state.current_test_started_ns) / 1_000_000_000
                        if state.current_test_id is not None and state.current_test_started_ns is not None
                        else None
                    ),
                    last_transition_ns=state.last_transition_ns,
                    age_seconds=(now_ns - state.last_transition_ns) / 1_000_000_000,
                    transition_age_seconds=(now_ns - state.last_transition_ns) / 1_000_000_000,
                    recent_tests=tuple(state.recent_tests),
                    started_suite_indexes=tuple(sorted(state.started_suite_indexes)),
                    initial_gil_enabled=state.initial_gil_enabled,
                    gil_enabled=state.gil_enabled,
                    first_gil_state_change=state.first_gil_state_change,
                    last_non_ok_test=state.last_non_ok_test,
                )
                for state in sorted(self._states.values(), key=lambda state: state.worker_index)
            )

    def handle_monitor_error(self, error: BaseException) -> BaseException:
        with self._lock:
            if self._monitor_error is None:
                self._monitor_error = error
            return self._monitor_error

    def raise_monitor_error(self) -> None:
        with self._lock:
            if self._monitor_error is not None:
                raise self._monitor_error


class WorkerEventMonitor:
    def __init__(self, event_queue: Any, tracker: WorkerStateTracker, on_event: Any = None):
        self._event_queue = event_queue
        self._tracker = tracker
        self._on_event = on_event
        self._lifecycle_condition = threading.Condition()
        self._sealed = False
        self._handler_active = False
        self._thread = threading.Thread(target=self._run, name="worker-event-monitor", daemon=True)
        self.error: BaseException | None = None
        self.stopped = False

    def start(self) -> None:
        self._thread.start()

    def seal(self) -> None:
        """Prevent new event handling and wait for the active handler to finish."""
        with self._lifecycle_condition:
            self._sealed = True
            while self._handler_active:
                self._lifecycle_condition.wait()

    def _begin_handler(self) -> bool:
        with self._lifecycle_condition:
            if self._sealed or self.error is not None:
                return False
            self._handler_active = True
            return True

    def _finish_handler(self) -> None:
        with self._lifecycle_condition:
            self._handler_active = False
            self._lifecycle_condition.notify_all()

    def _handle_queue_error(self, error: BaseException) -> None:
        with self._lifecycle_condition:
            if self._sealed:
                return
            self.error = self._tracker.handle_monitor_error(error)

    def _run(self) -> None:
        while True:
            try:
                event = self._event_queue.get()
            except BaseException as error:
                self._handle_queue_error(error)
                return

            if event is None:
                return
            if not self._begin_handler():
                continue

            try:
                self._tracker.handle_event(event)
                if self._on_event is not None:
                    self._on_event(event)
            except BaseException as error:
                self.error = self._tracker.handle_monitor_error(error)
            finally:
                self._finish_handler()

    def stop_and_drain(self) -> None:
        self._tracker.raise_monitor_error()
        try:
            self._event_queue.put(None)
        except BaseException as write_error:
            try:
                self._tracker.raise_monitor_error()
            except BaseException as monitor_error:
                if isinstance(monitor_error, (KeyboardInterrupt, SystemExit)) or not isinstance(
                    write_error, (KeyboardInterrupt, SystemExit)
                ):
                    raise monitor_error from None
            raise
        self._thread.join()
        self.stopped = True
        self.seal()
        self._tracker.raise_monitor_error()


def replay_worker_event_journals(run_dir: str | pathlib.Path, tracker: WorkerStateTracker) -> tuple[str, ...]:
    """Replay durable worker events that the live monitor may not have handled."""
    run_path = pathlib.Path(run_dir)
    errors = []
    try:
        journal_paths = sorted(run_path.glob("worker-*-*.events.jsonl"))
    except OSError as error:
        return (f"{run_path}: {error}",)

    for path in journal_paths:
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        event = WorkerEvent.from_dict(json.loads(line))
                        tracker.handle_event(event)
                    except Exception as error:
                        errors.append(f"{path}:{line_number}: {error}")
        except OSError as error:
            errors.append(f"{path}: {error}")
    return tuple(errors)
