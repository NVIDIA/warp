# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each test imports the runner internals it exercises inside the test body, so
# the module header stays free of the code under test and every test states its
# own dependencies.
# ruff: noqa: PLC0415

import concurrent.futures
import contextlib
import importlib
import io
import json
import multiprocessing
import os
import pathlib
import queue as queue_module
import signal
import sys
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ElementTree
from types import SimpleNamespace
from unittest import mock

FIXTURE_CASES_SOURCE = """import gc
import os
import pathlib
import time
import unittest

MARKER_DIR = pathlib.Path(os.environ["WARP_RUNNER_FIXTURE_DIR"])


class BlockingCase(unittest.TestCase):
    def test_wait_for_pool_shutdown(self):
        MARKER_DIR.joinpath("blocking-started").write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 30.0
        while not MARKER_DIR.joinpath("release-blocker").exists():
            if time.monotonic() >= deadline:
                self.fail("Parent did not terminate the blocked worker")
            time.sleep(0.01)


class PassingCase(unittest.TestCase):
    def test_pass(self):
        MARKER_DIR.joinpath("passing-complete").write_text("ready", encoding="utf-8")


class HardExitCase(unittest.TestCase):
    def test_exit(self):
        deadline = time.monotonic() + 30.0
        marker = MARKER_DIR.joinpath("passing-complete")
        while not marker.exists():
            if time.monotonic() >= deadline:
                self.fail("Passing suite did not complete")
            time.sleep(0.01)
        os.write(2, b"hard-exit-marker\\n")
        os._exit(86)


class AbortCase(unittest.TestCase):
    def test_abort(self):
        attempts_path = MARKER_DIR / "abort-attempts"
        attempts = int(attempts_path.read_text(encoding="utf-8")) if attempts_path.exists() else 0
        attempts_path.write_text(str(attempts + 1), encoding="utf-8")
        os.write(2, b"abort-marker\\n")
        os.abort()


class CleanupAbortCase(unittest.TestCase):
    def test_abort_during_gc(self):
        gc.collect = os.abort


class FinalizationAbortCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        os.abort()

    def test_completes_before_class_finalization(self):
        pass


class OrdinaryFailureCase(unittest.TestCase):
    def test_failure(self):
        self.fail("ordinary assertion marker")


class ShouldNotStartCase(unittest.TestCase):
    def test_not_started_after_failfast(self):
        MARKER_DIR.joinpath("unexpected-start").write_text("started", encoding="utf-8")


class OutputPressureCase(unittest.TestCase):
    def test_output_pressure(self):
        chunk = b"output-pressure-marker " + b"x" * 4060 + b"\\n"
        for _ in range(256):
            os.write(1, chunk)
            os.write(2, chunk)


class FirstPidCase(unittest.TestCase):
    def test_record_pid(self):
        MARKER_DIR.joinpath("first-pid").write_text(str(os.getpid()), encoding="utf-8")


class SecondPidCase(unittest.TestCase):
    def test_record_pid(self):
        MARKER_DIR.joinpath("second-pid").write_text(str(os.getpid()), encoding="utf-8")
"""


JUNIT_FIXTURE_SOURCES = {
    "setUpClass": """import unittest


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise RuntimeError("class setup failed")

    def test_runs(self):
        pass
""",
    "tearDownClass": """import time
import unittest


class FixtureCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        raise RuntimeError("class teardown failed")

    def test_runs(self):
        time.sleep(0.02)
""",
    "setUpModule": """import unittest


def setUpModule():
    raise RuntimeError("module setup failed")


class FixtureCase(unittest.TestCase):
    def test_runs(self):
        pass
""",
    "tearDownModule": """import time
import unittest


def tearDownModule():
    raise RuntimeError("module teardown failed")


class FixtureCase(unittest.TestCase):
    def test_runs(self):
        time.sleep(0.02)
""",
    "fixtureSkip": """import unittest


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise unittest.SkipTest("class fixture skipped")

    def test_runs(self):
        pass
""",
}


JUNIT_ORDINARY_SOURCE = """import time
import unittest


class OrdinaryCase(unittest.TestCase):
    def test_passes(self):
        time.sleep(0.02)


class SubtestCase(unittest.TestCase):
    def test_subtest_failure(self):
        time.sleep(0.02)
        with self.subTest(value=1):
            self.fail("subtest failure")
"""


@contextlib.contextmanager
def temporary_fixture_cases():
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        import_directory = root / "importable"
        marker_directory = root / "markers"
        import_directory.mkdir()
        marker_directory.mkdir()
        import_directory.joinpath("fixture_cases.py").write_text(FIXTURE_CASES_SOURCE, encoding="utf-8")

        import_path = str(import_directory)
        with mock.patch.dict(os.environ, {"WARP_RUNNER_FIXTURE_DIR": str(marker_directory)}, clear=False):
            sys.path.insert(0, import_path)
            importlib.invalidate_caches()
            module = importlib.import_module("fixture_cases")
            try:
                yield module, marker_directory, root
            finally:
                sys.modules.pop("fixture_cases", None)
                try:
                    sys.path.remove(import_path)
                except ValueError:
                    pass
                importlib.invalidate_caches()


@contextlib.contextmanager
def temporary_junit_module(scenario, source):
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        module_name = f"junit_fixture_{scenario.lower()}"
        root.joinpath(f"{module_name}.py").write_text(source, encoding="utf-8")

        import_path = str(root)
        sys.path.insert(0, import_path)
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
        try:
            yield module, root
        finally:
            sys.modules.pop(module_name, None)
            try:
                sys.path.remove(import_path)
            except ValueError:
                pass
            importlib.invalidate_caches()


class RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class RaisingQueue:
    def __init__(self, exception):
        self.exception = exception

    def put(self, item):
        raise self.exception


def assert_thread_stops(test_case, thread, timeout=5.0):
    thread.join(timeout)
    test_case.assertFalse(thread.is_alive())


def publish_large_events(queue, count, payload_size):
    payload = "x" * payload_size
    for index in range(count):
        queue.put({"index": index, "payload": payload})


class LocalManager:
    @staticmethod
    def Event():
        return threading.Event()

    @staticmethod
    def Lock():
        return threading.Lock()

    @staticmethod
    def Value(value_type, initial_value):
        return SimpleNamespace(value=initial_value)


class StubExecutor:
    def __init__(self, submit_results, manager_thread=None):
        self._submit_results = iter(submit_results)
        self._executor_manager_thread = manager_thread
        self._processes = {}

    def submit(self, function, index, suite):
        result = next(self._submit_results)
        if isinstance(result, BaseException):
            raise result
        return result

    def shutdown(self, wait=True, cancel_futures=False):
        self._executor_manager_thread = None


class NonCancellingFuture(concurrent.futures.Future):
    def cancel(self):
        return False


class PublishingManagerThread:
    def __init__(self, future, result):
        self.future = future
        self.result = result
        self.joined = False

    def join(self):
        self.joined = True
        self.future.set_result(self.result)


class StubSinkFile:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.closed = False

    def fileno(self):
        return self.descriptor

    def close(self):
        self.closed = True


class TestDiagnosticEvents(unittest.TestCase):
    def test_reporter_observes_gil_at_suite_boundaries(self):
        from warp._src.test_runner import common as gil
        from warp._src.test_runner import events as diagnostics
        from warp._src.test_runner.common import EventKind

        queue = RecordingQueue()
        reporter = diagnostics.WorkerEventReporter(
            event_queue=queue,
            worker_index=3,
            run_dir=None,
            run_start_monotonic_ns=0,
        )
        self.assertTrue(hasattr(diagnostics, "get_gil_enabled"))
        self.assertTrue(hasattr(reporter, "emit_gil_observed"))
        with mock.patch.object(
            gil.sys,
            "_is_gil_enabled",
            side_effect=(False, False, True),
            create=True,
        ):
            reporter.emit_gil_observed(EventKind.WORKER_INITIALIZED)
            reporter.emit_gil_observed(
                EventKind.SUITE_STARTED,
                suite_index=27,
                suite_name="fixture.TestGilTransition",
                test_count=1,
            )
            reporter.emit(
                EventKind.TEST_STOPPED,
                test_id="fixture.TestGilTransition.test_import",
            )
            reporter.emit_gil_observed(EventKind.SUITE_FINISHED)

        self.assertEqual(
            [event.event for event in queue.items],
            [
                EventKind.WORKER_INITIALIZED,
                EventKind.SUITE_STARTED,
                EventKind.TEST_STOPPED,
                EventKind.GIL_STATE_CHANGED,
                EventKind.SUITE_FINISHED,
            ],
        )
        initialized, started, _, changed, finished = queue.items
        self.assertFalse(initialized.gil_enabled)
        self.assertFalse(started.gil_enabled)
        self.assertFalse(changed.previous_gil_enabled)
        self.assertTrue(changed.gil_enabled)
        self.assertEqual(changed.observed_at, EventKind.SUITE_FINISHED.value)
        self.assertEqual(changed.suite_index, 27)
        self.assertEqual(changed.suite_name, "fixture.TestGilTransition")
        self.assertEqual(changed.test_id, "fixture.TestGilTransition.test_import")
        self.assertTrue(finished.gil_enabled)
        self.assertEqual(finished.suite_index, 27)

    def test_reporter_sends_event_and_journal_line(self):
        from warp._src.test_runner.common import EventKind
        from warp._src.test_runner.events import WorkerEventReporter

        queue = RecordingQueue()
        with tempfile.TemporaryDirectory() as directory:
            reporter = WorkerEventReporter(
                event_queue=queue,
                worker_index=3,
                run_dir=pathlib.Path(directory),
                run_start_monotonic_ns=100,
            )
            event = reporter.emit(
                EventKind.TEST_STARTED,
                suite_index=27,
                suite_name="fixture.TestCrash",
                test_id="fixture.TestCrash.test_abort",
            )
            reporter.close()

            self.assertEqual(event.worker_index, 3)
            self.assertEqual(event.sequence, 1)
            self.assertEqual(queue.items, [event])

            journal = next(pathlib.Path(directory).glob("*.events.jsonl"))
            payload = json.loads(journal.read_text(encoding="utf-8"))
            self.assertEqual(payload, event.to_dict())

    def test_tracker_distinguishes_outcome_cleanup_and_completion(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker(history_limit=3)

        def event(kind, sequence, test_id=None, outcome=None, elapsed_seconds=None):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=2,
                pid=8123,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=4,
                suite_name="fixture.TestCleanup",
                test_id=test_id,
                outcome=outcome,
                elapsed_seconds=elapsed_seconds,
            )

        tracker.handle_event(event(EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(EventKind.WORKER_INITIALIZED, 2))
        tracker.handle_event(event(EventKind.SUITE_STARTED, 3))
        tracker.handle_event(event(EventKind.TEST_STARTED, 4, "fixture.TestCleanup.test_ok"))
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                5,
                "fixture.TestCleanup.test_ok",
                outcome="OK",
                elapsed_seconds=0.75,
            )
        )
        tracker.handle_event(event(EventKind.TEST_CLEANUP_STARTED, 6, "fixture.TestCleanup.test_ok"))

        snapshot = tracker.snapshots(now_ns=6_500_000_000)[0]
        self.assertEqual(snapshot.phase, "test_cleanup")
        self.assertEqual(snapshot.current_outcome, "OK")
        self.assertEqual(snapshot.current_test_id, "fixture.TestCleanup.test_ok")
        self.assertEqual(snapshot.current_elapsed_seconds, 2.5)
        self.assertEqual(snapshot.transition_age_seconds, 0.5)

        tracker.handle_event(event(EventKind.TEST_STOPPED, 7, "fixture.TestCleanup.test_ok"))
        snapshot = tracker.snapshots(now_ns=7_500_000_000)[0]
        self.assertEqual(snapshot.phase, "test_stopped")
        self.assertEqual(snapshot.recent_tests[-1].test_id, "fixture.TestCleanup.test_ok")
        self.assertEqual(snapshot.recent_tests[-1].outcome, "OK")
        self.assertEqual(snapshot.recent_tests[-1].elapsed_seconds, 0.75)
        self.assertIsNone(snapshot.current_test_id)
        self.assertIsNone(snapshot.current_outcome)
        self.assertIsNone(snapshot.current_elapsed_seconds)
        self.assertEqual(snapshot.transition_age_seconds, 0.5)

        tracker.handle_event(event(EventKind.SUITE_FINALIZING, 8))
        snapshot = tracker.snapshots(now_ns=8_500_000_000)[0]
        self.assertEqual(snapshot.phase, "suite_finalizing")
        self.assertIsNone(snapshot.current_test_id)

    def test_tracker_records_gil_change_without_changing_phase(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker(history_limit=3)

        def event(kind, sequence, **fields):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=2,
                pid=8123,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=4,
                suite_name="fixture.TestGilTransition",
                **fields,
            )

        tracker.handle_event(event(EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(EventKind.WORKER_INITIALIZED, 2, gil_enabled=False))
        tracker.handle_event(event(EventKind.SUITE_STARTED, 3, gil_enabled=False))
        tracker.handle_event(
            event(
                EventKind.TEST_STARTED,
                4,
                test_id="fixture.TestGilTransition.test_import",
            )
        )
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                5,
                test_id="fixture.TestGilTransition.test_import",
                outcome="OK",
                elapsed_seconds=0.5,
            )
        )
        tracker.handle_event(
            event(
                EventKind.TEST_STOPPED,
                6,
                test_id="fixture.TestGilTransition.test_import",
            )
        )

        before_change = tracker.snapshots(now_ns=6_500_000_000)[0]
        self.assertTrue(hasattr(before_change, "initial_gil_enabled"))
        tracker.handle_event(
            event(
                EventKind.GIL_STATE_CHANGED,
                7,
                previous_gil_enabled=False,
                gil_enabled=True,
                observed_at=EventKind.SUITE_FINISHED.value,
            )
        )
        after_change = tracker.snapshots(now_ns=7_500_000_000)[0]

        self.assertEqual(after_change.phase, "test_stopped")
        self.assertEqual(after_change.last_transition_ns, 6_000_000_000)
        self.assertFalse(after_change.initial_gil_enabled)
        self.assertTrue(after_change.gil_enabled)
        self.assertEqual(after_change.first_gil_state_change.previous_gil_enabled, False)
        self.assertEqual(after_change.first_gil_state_change.gil_enabled, True)
        self.assertEqual(after_change.first_gil_state_change.observed_at, "suite_finished")
        self.assertEqual(after_change.first_gil_state_change.suite_index, 4)
        self.assertEqual(
            after_change.first_gil_state_change.test_id,
            "fixture.TestGilTransition.test_import",
        )

    def test_tracker_preserves_last_non_ok_outcome_beyond_recent_history(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker(history_limit=3)

        def event(kind, sequence, test_id=None, outcome=None):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=2,
                pid=8123,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=4,
                suite_name="fixture.TestOutcomes",
                test_id=test_id,
                outcome=outcome,
                elapsed_seconds=0.25 if outcome is not None else None,
            )

        tracker.handle_event(event(EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(EventKind.SUITE_STARTED, 2))
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                3,
                test_id="fixture.TestOutcomes.test_failure",
                outcome="FAIL",
            )
        )
        tracker.handle_event(event(EventKind.TEST_STOPPED, 4, test_id="fixture.TestOutcomes.test_failure"))
        sequence = 5
        for index in range(4):
            test_id = f"fixture.TestOutcomes.test_ok_{index}"
            tracker.handle_event(event(EventKind.TEST_OUTCOME, sequence, test_id=test_id, outcome="OK"))
            tracker.handle_event(event(EventKind.TEST_STOPPED, sequence + 1, test_id=test_id))
            sequence += 2

        snapshot = tracker.snapshots(now_ns=14_000_000_000)[0]
        self.assertTrue(hasattr(snapshot, "last_non_ok_test"))
        self.assertEqual(
            [test.test_id for test in snapshot.recent_tests],
            [
                "fixture.TestOutcomes.test_ok_1",
                "fixture.TestOutcomes.test_ok_2",
                "fixture.TestOutcomes.test_ok_3",
            ],
        )
        self.assertEqual(snapshot.last_non_ok_test.test_id, "fixture.TestOutcomes.test_failure")
        self.assertEqual(snapshot.last_non_ok_test.outcome, "FAIL")
        self.assertEqual(snapshot.last_non_ok_test.elapsed_seconds, 0.25)

    def test_journal_replay_recovers_lagging_tracker_without_regression(self):
        from warp._src.test_runner import events as diagnostics
        from warp._src.test_runner.common import EventKind, WorkerEvent

        self.assertTrue(hasattr(diagnostics, "replay_worker_event_journals"))
        tracker = diagnostics.WorkerStateTracker(history_limit=3)

        def event(kind, sequence, **fields):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=2,
                pid=8123,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=4,
                suite_name="fixture.TestJournalRecovery",
                **fields,
            )

        journal_events = (
            event(EventKind.WORKER_STARTED, 1),
            event(EventKind.WORKER_INITIALIZED, 2, gil_enabled=False),
            event(EventKind.SUITE_STARTED, 3, gil_enabled=False),
            event(
                EventKind.TEST_OUTCOME,
                4,
                test_id="fixture.TestJournalRecovery.test_failure",
                outcome="ERROR",
                elapsed_seconds=0.5,
            ),
            event(
                EventKind.TEST_STOPPED,
                5,
                test_id="fixture.TestJournalRecovery.test_failure",
            ),
            event(
                EventKind.GIL_STATE_CHANGED,
                6,
                previous_gil_enabled=False,
                gil_enabled=True,
                observed_at=EventKind.SUITE_FINISHED.value,
                test_id="fixture.TestJournalRecovery.test_failure",
            ),
            event(EventKind.SUITE_FINISHED, 7, gil_enabled=True),
        )
        tracker.handle_event(journal_events[0])
        tracker.handle_event(journal_events[1])

        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory, "worker-2-8123.events.jsonl")
            journal.write_text(
                "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in journal_events),
                encoding="utf-8",
            )
            errors = diagnostics.replay_worker_event_journals(directory, tracker)

        self.assertEqual(errors, ())
        recovered = tracker.snapshots(now_ns=8_000_000_000)[0]
        self.assertEqual(recovered.phase, "idle")
        self.assertEqual(
            recovered.last_non_ok_test.test_id,
            "fixture.TestJournalRecovery.test_failure",
        )
        self.assertTrue(recovered.gil_enabled)

        tracker.handle_event(journal_events[2])
        after_late_duplicate = tracker.snapshots(now_ns=9_000_000_000)[0]
        self.assertEqual(after_late_duplicate.phase, "idle")
        self.assertTrue(after_late_duplicate.gil_enabled)

    def test_tracker_accepts_gil_annotation_as_first_recovered_event(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker()
        event = WorkerEvent(
            sequence=6,
            event=EventKind.GIL_STATE_CHANGED,
            worker_index=2,
            pid=8123,
            monotonic_ns=6_000_000_000,
            wall_time_ns=6000,
            suite_index=4,
            suite_name="fixture.TestJournalRecovery",
            previous_gil_enabled=False,
            gil_enabled=True,
            observed_at=EventKind.SUITE_FINISHED.value,
        )

        try:
            tracker.handle_event(event)
        except KeyError as error:
            self.fail(f"GIL recovery annotation raised {error!r}")
        snapshot = tracker.snapshots(now_ns=7_000_000_000)[0]
        self.assertEqual(snapshot.phase, "unknown")
        self.assertTrue(snapshot.gil_enabled)

    def test_initializing_snapshot_names_worker_initializer(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker
        from warp._src.test_runner.postmortem import build_crash_snapshot, format_pool_failure

        tracker = WorkerStateTracker()
        tracker.handle_event(
            WorkerEvent(
                sequence=1,
                event=EventKind.WORKER_STARTED,
                worker_index=2,
                pid=8123,
                monotonic_ns=1,
                wall_time_ns=1,
            )
        )

        crash_snapshot = build_crash_snapshot(
            RuntimeError("initialization failed"),
            (),
            tracker.snapshots(now_ns=2),
            (),
            None,
        )

        self.assertEqual(crash_snapshot["workers"][0]["phase"], "initializing")
        self.assertIn("candidate=initializer:worker initialization", format_pool_failure(crash_snapshot))

    def test_nonempty_fault_log_is_fatal_evidence(self):
        from warp._src.test_runner import postmortem as diagnostics

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            fault_path = run_dir / "worker-2-pid-8123.fault.log"
            cases = (
                b"Fatal Python error: Segmentation fault\n" + b"x" * (33 * 1024),
                b"Windows fatal exception: access violation\r\n",
            )
            for contents in cases:
                with self.subTest(header=contents[:32]):
                    fault_path.write_bytes(contents)
                    with mock.patch.object(
                        diagnostics,
                        "_read_artifact_tail",
                        side_effect=AssertionError("fault evidence must not scan the console tail"),
                    ):
                        evidence = diagnostics._artifact_evidence(run_dir, 2, 8123)

                    self.assertEqual(evidence["fault"]["state"], "non_empty")
                    self.assertIs(evidence["fault"]["fatal_traceback_evidence"], True)

            fault_path.write_bytes(cases[0])
            bounded_tail = diagnostics._read_artifact_tail(fault_path)
            self.assertEqual(len(bounded_tail.encode("utf-8")), 32 * 1024)
            self.assertNotIn("Fatal Python error", bounded_tail)

            fault_path.write_bytes(b"")
            evidence = diagnostics._artifact_evidence(run_dir, 2, 8123)
            self.assertEqual(evidence["fault"]["state"], "empty")
            self.assertIs(evidence["fault"]["fatal_traceback_evidence"], False)

            fault_path.unlink()
            evidence = diagnostics._artifact_evidence(run_dir, 2, 8123)
            self.assertEqual(evidence["fault"]["state"], "missing")
            self.assertIsNotNone(evidence["fault"]["path"])
            self.assertIsNone(evidence["fault"]["fatal_traceback_evidence"])

            evidence = diagnostics._artifact_evidence(None, 2, 8123)
            self.assertIsNone(evidence["fault"]["path"])
            self.assertIsNone(evidence["fault"]["fatal_traceback_evidence"])

    def test_pool_report_prioritizes_abnormal_workers_and_compacts_parent_exits(self):
        from warp._src.test_runner.common import EventKind, ProcessExit, ProcessExitProvenance, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker
        from warp._src.test_runner.postmortem import (
            build_crash_snapshot,
            format_pool_failure,
            print_pool_failure_evidence,
        )

        tracker = WorkerStateTracker(history_limit=3)

        def event(worker_index, pid, kind, sequence, **fields):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=worker_index,
                pid=pid,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=worker_index,
                suite_name=f"fixture.Worker{worker_index}",
                **fields,
            )

        tracker.handle_event(event(0, 7001, EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(0, 7001, EventKind.SUITE_STARTED, 2, gil_enabled=False))
        tracker.handle_event(event(1, 7002, EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(1, 7002, EventKind.WORKER_INITIALIZED, 2, gil_enabled=False))
        tracker.handle_event(event(1, 7002, EventKind.SUITE_STARTED, 3, gil_enabled=False))
        tracker.handle_event(
            event(
                1,
                7002,
                EventKind.TEST_OUTCOME,
                4,
                test_id="fixture.Worker1.test_failure",
                outcome="ERROR",
                elapsed_seconds=0.75,
            )
        )
        tracker.handle_event(
            event(
                1,
                7002,
                EventKind.TEST_STOPPED,
                5,
                test_id="fixture.Worker1.test_failure",
            )
        )
        tracker.handle_event(
            event(
                1,
                7002,
                EventKind.GIL_STATE_CHANGED,
                6,
                previous_gil_enabled=False,
                gil_enabled=True,
                observed_at=EventKind.SUITE_FINISHED.value,
            )
        )
        tracker.handle_event(event(1, 7002, EventKind.SUITE_FINISHED, 7, gil_enabled=True))
        tracker.handle_event(event(2, 7003, EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(2, 7003, EventKind.SUITE_STARTED, 2, gil_enabled=False))
        tracker.handle_event(
            event(
                2,
                7003,
                EventKind.TEST_STARTED,
                3,
                test_id="fixture.Worker2.test_candidate",
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            for worker_index, pid, marker in (
                (0, 7001, "parent-tail-marker"),
                (1, 7002, "abnormal-tail-marker"),
                (2, 7003, "unresolved-tail-marker"),
            ):
                run_dir.joinpath(f"worker-{worker_index}-pid-{pid}.output.log").write_text(
                    marker,
                    encoding="utf-8",
                )

            snapshot = build_crash_snapshot(
                RuntimeError("pool failed"),
                (),
                tracker.snapshots(now_ns=8_000_000_000),
                (
                    ProcessExit(7001, -signal.SIGTERM, "SIGTERM", ProcessExitProvenance.PARENT_TERMINATED.value),
                    ProcessExit(7002, 86, None, ProcessExitProvenance.INDEPENDENTLY_ABNORMAL.value),
                    ProcessExit(7003, -signal.SIGTERM, "SIGTERM", ProcessExitProvenance.UNRESOLVED.value),
                ),
                run_dir,
            )

            summary = format_pool_failure(snapshot)
            self.assertIn("parent-terminated workers (1)", summary)
            self.assertIn("worker 0 (PID 7001, SIGTERM, final GIL state=disabled)", summary)
            self.assertLess(
                summary.index("exit provenance=independently_abnormal"),
                summary.index("parent-terminated workers (1)"),
            )
            self.assertIn(
                "last non-OK test=fixture.Worker1.test_failure outcome=ERROR duration=0.750s",
                summary,
            )
            self.assertIn("final GIL state=enabled", summary)
            self.assertIn(
                "first observed GIL change=disabled->enabled at suite_finished",
                summary,
            )
            self.assertIn(
                "note: SIGTERM exits are unresolved only when the worker's journal did not show "
                "work in progress; check its journal position to see whether it had already finished.",
                summary,
            )
            self.assertNotIn("SIGKILL exits are unresolved", summary)

            console = io.StringIO()
            with contextlib.redirect_stderr(console):
                print_pool_failure_evidence(snapshot)
            output = console.getvalue()
            self.assertIn("abnormal-tail-marker", output)
            self.assertIn("unresolved-tail-marker", output)
            self.assertNotIn("parent-tail-marker", output)
            self.assertLess(output.index("abnormal-tail-marker"), output.index("unresolved-tail-marker"))

    def test_atomic_json_replaces_complete_document(self):
        from warp._src.test_runner.artifacts import atomic_write_json

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory, "run.json")
            path.write_text('{"old":true}', encoding="utf-8")
            atomic_write_json(path, {"version": 1, "status": "complete"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "complete")
            self.assertEqual(list(path.parent.glob(".run.json.*.tmp")), [])

    def test_reporter_emit_propagates_process_controls(self):
        from warp._src.test_runner.common import EventKind
        from warp._src.test_runner.events import WorkerEventReporter

        for exception in (KeyboardInterrupt("queue interrupted"), SystemExit(74)):
            with self.subTest(exception=type(exception).__name__):
                reporter = WorkerEventReporter(
                    event_queue=RaisingQueue(exception),
                    worker_index=0,
                    run_dir=None,
                    run_start_monotonic_ns=0,
                )
                with self.assertRaises(type(exception)) as raised:
                    reporter.emit(EventKind.WORKER_STARTED)

                self.assertIs(raised.exception, exception)
                reporter.close()

    def test_environment_configuration_prefers_cli_and_ignores_blank_values(self):
        from warp._src.test_runner.artifacts import resolve_diagnostics_root

        with mock.patch.dict(
            os.environ,
            {"WARP_TEST_DIAGNOSTICS_DIR": "from-environment"},
            clear=False,
        ):
            self.assertEqual(resolve_diagnostics_root("from-cli"), pathlib.Path("from-cli"))
            self.assertEqual(
                resolve_diagnostics_root(None),
                pathlib.Path("from-environment"),
            )
            self.assertIsNone(resolve_diagnostics_root("   "))

    def test_shutdown_seals_handlers_before_closing_the_queue(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerEventMonitor, WorkerStateTracker
        from warp._src.test_runner.runner import _RunDiagnostics

        class ClosableQueue:
            def __init__(self):
                self.queue = queue_module.Queue()
                self.published = []
                self.closed = threading.Event()

            def get(self):
                return self.queue.get()

            def put(self, item):
                self.published.append(item)
                self.queue.put(item)

            def close(self):
                self.closed.set()
                self.queue.put(None)

        def event(sequence):
            return WorkerEvent(
                sequence=sequence,
                event=EventKind.WORKER_STARTED,
                worker_index=sequence,
                pid=7000 + sequence,
                monotonic_ns=sequence,
                wall_time_ns=sequence,
            )

        event_queue = ClosableQueue()
        tracker = WorkerStateTracker()
        callback_started = threading.Event()
        release_callback = threading.Event()
        handled_sequences = []
        callback_error = KeyboardInterrupt("callback interrupted")

        def blocking_callback(item):
            handled_sequences.append(item.sequence)
            callback_started.set()
            if not release_callback.wait(5.0):
                raise TimeoutError("callback release timed out")
            raise callback_error

        monitor = WorkerEventMonitor(event_queue, tracker, on_event=blocking_callback)
        seal_started = threading.Event()
        original_seal = monitor.seal

        def observed_seal():
            seal_started.set()
            original_seal()

        monitor.seal = observed_seal
        diagnostics = _RunDiagnostics()
        diagnostics.attach_monitor(event_queue, tracker, monitor)
        diagnostics.degrade_event_monitor()
        monitor.start()
        event_queue.put(event(1))
        event_queue.put(event(2))
        try:
            self.assertTrue(callback_started.wait(5.0))

            stop_errors = []

            def stop_diagnostics():
                try:
                    diagnostics.stop_monitor()
                except BaseException as error:
                    stop_errors.append(error)

            stop_thread = threading.Thread(target=stop_diagnostics, daemon=True)
            stop_thread.start()
            self.assertTrue(seal_started.wait(5.0))
            self.assertTrue(stop_thread.is_alive())
            release_callback.set()
            assert_thread_stops(self, stop_thread)
            assert_thread_stops(self, monitor._thread)

            self.assertEqual(stop_errors, [callback_error])
            self.assertEqual(handled_sequences, [1])
            self.assertTrue(event_queue.closed.is_set())
            self.assertNotIn(None, event_queue.published)
        finally:
            release_callback.set()
            if not event_queue.closed.is_set():
                event_queue.close()
            monitor._thread.join(5.0)

    def test_spawn_simple_queue_pressure_preserves_every_event(self):
        from warp._src.test_runner.events import WorkerEventMonitor

        class PressureTracker:
            def __init__(self):
                self.events = []
                self.error = None

            def handle_event(self, event):
                self.events.append(event)

            def handle_monitor_error(self, error):
                if self.error is None:
                    self.error = error
                return self.error

            def raise_monitor_error(self):
                if self.error is not None:
                    raise self.error

        spawn_context = multiprocessing.get_context("spawn")
        event_queue = spawn_context.SimpleQueue()
        tracker = PressureTracker()
        monitor = WorkerEventMonitor(event_queue, tracker)
        producer = spawn_context.Process(
            target=publish_large_events,
            args=(event_queue, 256, 262_144),
        )
        monitor.start()
        producer.start()
        try:
            assert_thread_stops(self, producer)
            self.assertEqual(producer.exitcode, 0)

            stop_errors = []

            def stop_monitor():
                try:
                    monitor.stop_and_drain()
                except BaseException as error:
                    stop_errors.append(error)

            stop_thread = threading.Thread(target=stop_monitor, daemon=True)
            stop_thread.start()
            assert_thread_stops(self, stop_thread)

            self.assertEqual(stop_errors, [])
            self.assertEqual([event["index"] for event in tracker.events], list(range(256)))
            self.assertTrue(all(len(event["payload"]) == 262_144 for event in tracker.events))
        finally:
            if producer.is_alive():
                producer.terminate()
                producer.join(5.0)
            event_queue.close()

    def test_run_directory_creation_never_reuses_an_existing_run(self):
        from warp._src.test_runner.artifacts import create_diagnostics_run_dir

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            first = create_diagnostics_run_dir(root, wall_time_ns=0, parent_pid=1234)
            second = create_diagnostics_run_dir(root, wall_time_ns=0, parent_pid=1234)

            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


class TestWorkerSinkIsolation(unittest.TestCase):
    def _configure(self, *, journal_error=None, open_side_effect=None, dup2_side_effect=None, enable_error=None):
        from warp._src.test_runner import events as diagnostics

        queue = RecordingQueue()
        output_file = StubSinkFile(80)
        fault_file = StubSinkFile(81)
        warnings = []

        if open_side_effect is None:

            def open_side_effect(path, *args, **kwargs):
                return output_file if str(path).endswith(".output.log") else fault_file

        if dup2_side_effect is None:

            def dup2_side_effect(source, target):
                return None

        open_journal = mock.Mock(return_value=82)
        if journal_error is not None:
            open_journal.side_effect = journal_error
        enable = mock.Mock()
        if enable_error is not None:
            enable.side_effect = enable_error

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(diagnostics.os, "open", open_journal),
                mock.patch("builtins.open", side_effect=open_side_effect),
                mock.patch.object(diagnostics.os, "dup", side_effect=(90, 91)),
                mock.patch.object(diagnostics.os, "dup2", side_effect=dup2_side_effect) as dup2,
                mock.patch.object(diagnostics.os, "close"),
                mock.patch.object(diagnostics.os, "write", side_effect=lambda descriptor, payload: len(payload)),
                mock.patch.object(diagnostics.faulthandler, "disable"),
                mock.patch.object(diagnostics.faulthandler, "enable", enable),
                mock.patch.object(diagnostics.atexit, "register"),
                mock.patch.object(diagnostics, "warn", side_effect=warnings.append),
            ):
                reporter = diagnostics.configure_worker_diagnostics(
                    event_queue=queue,
                    worker_index=4,
                    run_dir=pathlib.Path(directory),
                    run_start_monotonic_ns=0,
                )
                reporter.emit(diagnostics.EventKind.TEST_STARTED, test_id="fixture.Case.test_lives")
                diagnostics.close_worker_diagnostics()

        return SimpleNamespace(
            queue=queue,
            warnings=warnings,
            output_file=output_file,
            fault_file=fault_file,
            dup2=dup2,
            enable=enable,
        )

    def test_journal_open_failure_keeps_queue_and_other_sinks(self):
        result = self._configure(journal_error=OSError("journal unavailable"))

        self.assertEqual(
            [event.event.value for event in result.queue.items],
            ["worker_started", "test_started", "worker_shutdown"],
        )
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("event journal", result.warnings[0])
        self.assertTrue(result.output_file.closed)
        self.assertTrue(result.fault_file.closed)
        result.enable.assert_called_once()

    def test_output_open_failure_keeps_queue_and_fault_sink(self):
        fault_file = StubSinkFile(81)

        def open_sink(path, *args, **kwargs):
            if str(path).endswith(".output.log"):
                raise OSError("output unavailable")
            return fault_file

        result = self._configure(open_side_effect=open_sink)

        self.assertEqual(len(result.queue.items), 3)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("worker output", result.warnings[0])
        self.assertTrue(fault_file.closed)
        result.enable.assert_called_once()

    def test_fault_open_failure_keeps_queue_and_output_sink(self):
        output_file = StubSinkFile(80)

        def open_sink(path, *args, **kwargs):
            if str(path).endswith(".fault.log"):
                raise OSError("fault unavailable")
            return output_file

        result = self._configure(open_side_effect=open_sink)

        self.assertEqual(len(result.queue.items), 3)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("faulthandler sink", result.warnings[0])
        self.assertTrue(output_file.closed)
        result.enable.assert_not_called()

    def test_faulthandler_enable_failure_disables_only_fault_sink(self):
        result = self._configure(enable_error=RuntimeError("faulthandler unavailable"))

        self.assertEqual(len(result.queue.items), 3)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("faulthandler", result.warnings[0])
        self.assertTrue(result.fault_file.closed)
        self.assertTrue(result.output_file.closed)


class TestResultLifecycle(unittest.TestCase):
    class PassingTest(unittest.TestCase):
        def test_passes(self):
            pass

    class TwoPassingTests(unittest.TestCase):
        def test_first_passes(self):
            pass

        def test_second_passes(self):
            pass

    class FailingSubTest(unittest.TestCase):
        def test_subtest_failure(self):
            with self.subTest(value=1):
                self.fail("subtest failure")

    def setUp(self):
        from warp._src.test_runner.events import (
            WorkerEventReporter,
            install_worker_reporter,
        )

        self.queue = RecordingQueue()
        self.reporter = WorkerEventReporter(
            event_queue=self.queue,
            worker_index=0,
            run_dir=None,
            run_start_monotonic_ns=0,
        )
        install_worker_reporter(self.reporter)
        self.addCleanup(install_worker_reporter, None)
        self.addCleanup(self.reporter.close)

    def _assert_result_lifecycle(self, result_class):
        cleanup_calls = []
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.PassingTest)
        with mock.patch("gc.collect", side_effect=lambda: cleanup_calls.append("gc")):
            runner = unittest.TextTestRunner(
                resultclass=result_class,
                stream=io.StringIO(),
                verbosity=0,
            )
            result = runner.run(suite)

        self.assertTrue(result.wasSuccessful())
        expected = [
            "test_started",
            "test_outcome",
            "test_cleanup_started",
            "test_stopped",
        ]
        self.assertEqual([event.event.value for event in self.queue.items], expected)
        self.assertEqual(cleanup_calls, ["gc"])
        self.assertEqual(self.queue.items[1].outcome, "OK")
        self.assertGreaterEqual(self.queue.items[1].elapsed_seconds, 0.0)

    def test_parallel_junit_result_emits_test_lifecycle(self):
        from warp.tests.unittest_utils import ParallelJunitTestResult

        self._assert_result_lifecycle(ParallelJunitTestResult)

    def test_parallel_junit_result_preserves_subtest_failures_without_worker_rendering(self):
        from warp.tests.unittest_utils import ParallelJunitTestResult

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.FailingSubTest)
        runner = unittest.TextTestRunner(
            resultclass=ParallelJunitTestResult,
            stream=io.StringIO(),
            verbosity=0,
        )
        result = runner.run(suite)

        self.assertEqual(len(result.failures), 1)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.test_record[0][3], "FAIL")

    def test_manager_emits_suite_and_test_lifecycles(self):
        from warp._src.test_runner.worker import ParallelTestManager

        manager = mock.Mock()
        manager.Event.return_value = threading.Event()
        args = SimpleNamespace(
            buffer=False,
            coverage=False,
            coverage_branch=False,
            failfast=False,
            junit_report_xml=None,
            verbose=0,
        )
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.TwoPassingTests)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ParallelTestManager(manager, args, temp_dir).run_tests(4, suite)

        self.assertEqual(result[0], 2)
        self.assertEqual(
            [event.event.value for event in self.queue.items],
            [
                "suite_started",
                "test_started",
                "test_outcome",
                "test_cleanup_started",
                "test_stopped",
                "test_started",
                "test_outcome",
                "test_cleanup_started",
                "test_stopped",
                "suite_finalizing",
                "suite_finished",
            ],
        )

    def test_reporter_emit_failure_does_not_deadlock_worker_completion(self):
        from warp._src.test_runner import events
        from warp._src.test_runner.events import WorkerEventReporter, install_worker_reporter
        from warp._src.test_runner.worker import ParallelTestManager

        reporter = WorkerEventReporter(
            event_queue=RaisingQueue(OSError("event queue unavailable")),
            worker_index=0,
            run_dir=None,
            run_start_monotonic_ns=0,
        )
        install_worker_reporter(reporter)

        def assert_reporter_uninstalled():
            self.assertIsNone(events.emit_worker_event(events.EventKind.WORKER_STARTED))

        self.addCleanup(assert_reporter_uninstalled)
        self.addCleanup(install_worker_reporter, None)
        self.addCleanup(reporter.close)
        args = SimpleNamespace(
            buffer=False,
            coverage=False,
            coverage_branch=False,
            failfast=False,
            junit_report_xml="enabled",
            level="class",
            verbose=0,
        )
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.PassingTest)
        results = []
        errors = []
        warnings = []

        def run_worker():
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    results.append(ParallelTestManager(LocalManager(), args, temp_dir).run_tests(0, suite))
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(events, "warn", side_effect=warnings.append):
            worker_thread = threading.Thread(target=run_worker, daemon=True)
            worker_thread.start()
            assert_thread_stops(self, worker_thread)

        self.assertEqual(errors, [])
        self.assertEqual(results[0][:6], (1, [], [], 0, 0, 0))
        self.assertEqual(len(results[0][6]), 1)
        self.assertEqual(results[0][6][0][3], "OK")
        self.assertEqual(warnings, ["Failed to send worker event: event queue unavailable"])


class TestParallelJunitFixtures(unittest.TestCase):
    def setUp(self):
        from warp._src.test_runner.events import WorkerEventReporter, install_worker_reporter

        self.queue = RecordingQueue()
        self.reporter = WorkerEventReporter(
            event_queue=self.queue,
            worker_index=0,
            run_dir=None,
            run_start_monotonic_ns=0,
        )
        install_worker_reporter(self.reporter)
        self.addCleanup(install_worker_reporter, None)
        self.addCleanup(self.reporter.close)

    @staticmethod
    def _args():
        return SimpleNamespace(
            buffer=False,
            coverage=False,
            coverage_branch=False,
            failfast=False,
            junit_report_xml="enabled",
            level="module",
            verbose=0,
        )

    def _run_module(self, scenario, source):
        from warp._src.test_runner.worker import ParallelTestManager
        from warp.tests.unittest_utils import write_junit_results

        with temporary_junit_module(scenario, source) as (module, root):
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            first_event = len(self.queue.items)
            result = ParallelTestManager(LocalManager(), self._args(), root).run_tests(
                0,
                suite,
                suite_name=module.__name__,
            )
            events = self.queue.items[first_event:]

            junit_path = root / "results.xml"
            write_junit_results(
                str(junit_path),
                result[6],
                sum(record[2] for record in result[6]),
            )
            xml_root = ElementTree.parse(junit_path).getroot()

        return module.__name__, result, events, xml_root

    def _assert_fixture_result(self, scenario, hook, target_kind, outcome, *, ordinary_test=False):
        module_name, result, events, xml_root = self._run_module(scenario, JUNIT_FIXTURE_SOURCES[scenario])
        target = module_name if target_kind == "module" else f"{module_name}.FixtureCase"
        fixture_id = f"{hook} ({target})"

        self.assertEqual(xml_root.tag, "testsuite")
        fixture_records = [record for record in result[6] if record[1] == hook]
        self.assertEqual(len(fixture_records), 1)
        self.assertEqual(fixture_records[0][:4], (target, hook, 0.0, outcome))

        fixture_cases = [case for case in xml_root.findall("testcase") if case.get("name") == hook]
        self.assertEqual(len(fixture_cases), 1)
        fixture_case = fixture_cases[0]
        self.assertEqual(fixture_case.get("classname"), target)
        self.assertEqual(float(fixture_case.get("time")), 0.0)
        if outcome == "ERROR":
            self.assertIsNotNone(fixture_case.find("error"))
            self.assertEqual(len(result[1]), 1)
            self.assertEqual(result[3], 0)
        else:
            self.assertIsNotNone(fixture_case.find("skipped"))
            self.assertEqual(len(result[1]), 0)
            self.assertEqual(result[3], 1)
        self.assertEqual(len(result[2]), 0)

        fixture_events = [event for event in events if event.test_id == fixture_id]
        self.assertEqual(
            [event.event.value for event in fixture_events],
            ["test_started", "test_outcome", "test_cleanup_started", "test_stopped"],
        )
        self.assertEqual(fixture_events[1].outcome, outcome)

        if ordinary_test:
            self.assertEqual(result[0], 1)
            ordinary_records = [record for record in result[6] if record[1] == "test_runs"]
            self.assertEqual(len(ordinary_records), 1)
            ordinary_record = ordinary_records[0]
            self.assertEqual(ordinary_record[0], "FixtureCase")
            self.assertEqual(ordinary_record[3], "OK")
            self.assertGreater(ordinary_record[2], 0.0)

            ordinary_cases = [case for case in xml_root.findall("testcase") if case.get("name") == "test_runs"]
            self.assertEqual(len(ordinary_cases), 1)
            self.assertEqual(ordinary_cases[0].get("classname"), "FixtureCase")
            self.assertEqual(float(ordinary_cases[0].get("time")), ordinary_record[2])

            ordinary_id = f"{module_name}.FixtureCase.test_runs"
            lifecycle_events = [event.event.value for event in events if event.test_id in (ordinary_id, fixture_id)]
            self.assertEqual(
                lifecycle_events,
                [
                    "test_started",
                    "test_outcome",
                    "test_cleanup_started",
                    "test_stopped",
                    "test_started",
                    "test_outcome",
                    "test_cleanup_started",
                    "test_stopped",
                ],
            )
        else:
            self.assertEqual(result[0], 0)
            self.assertEqual(len(result[6]), 1)

    def test_records_set_up_class_error(self):
        self._assert_fixture_result("setUpClass", "setUpClass", "class", "ERROR")

    def test_records_tear_down_class_error_after_normal_test(self):
        self._assert_fixture_result(
            "tearDownClass",
            "tearDownClass",
            "class",
            "ERROR",
            ordinary_test=True,
        )

    def test_records_set_up_module_error(self):
        self._assert_fixture_result("setUpModule", "setUpModule", "module", "ERROR")

    def test_records_tear_down_module_error_after_normal_test(self):
        self._assert_fixture_result(
            "tearDownModule",
            "tearDownModule",
            "module",
            "ERROR",
            ordinary_test=True,
        )

    def test_records_fixture_skip(self):
        self._assert_fixture_result("fixtureSkip", "setUpClass", "class", "SKIP")

    def test_preserves_ordinary_and_subtest_records(self):
        module_name, result, events, xml_root = self._run_module("ordinary", JUNIT_ORDINARY_SOURCE)

        self.assertEqual(result[0], 2)
        self.assertEqual(len(result[1]), 0)
        self.assertEqual(len(result[2]), 1)
        self.assertEqual(result[3], 0)
        self.assertEqual(
            [(record[0], record[1], record[3]) for record in result[6]],
            [
                ("OrdinaryCase", "test_passes", "OK"),
                ("SubtestCase", "test_subtest_failure", "FAIL"),
            ],
        )
        self.assertTrue(all(record[2] > 0.0 for record in result[6]))

        xml_cases = xml_root.findall("testcase")
        self.assertEqual(
            [(case.get("classname"), case.get("name")) for case in xml_cases],
            [
                ("OrdinaryCase", "test_passes"),
                ("SubtestCase", "test_subtest_failure"),
            ],
        )
        self.assertEqual(
            [float(case.get("time")) for case in xml_cases],
            [record[2] for record in result[6]],
        )
        self.assertIsNone(xml_cases[0].find("failure"))
        self.assertIsNotNone(xml_cases[1].find("failure"))

        expected_lifecycles = []
        for class_name, test_name, _ in (
            ("OrdinaryCase", "test_passes", "OK"),
            ("SubtestCase", "test_subtest_failure", "FAIL"),
        ):
            test_id = f"{module_name}.{class_name}.{test_name}"
            test_events = [event for event in events if event.test_id == test_id]
            expected_lifecycles.append(
                (
                    [event.event.value for event in test_events],
                    test_events[1].outcome,
                )
            )
        self.assertEqual(
            expected_lifecycles,
            [
                (["test_started", "test_outcome", "test_cleanup_started", "test_stopped"], "OK"),
                (["test_started", "test_outcome", "test_cleanup_started", "test_stopped"], "FAIL"),
            ],
        )

    def test_records_unknown_test_like_objects_without_changing_test_count(self):
        from warp.tests.unittest_utils import ParallelJunitTestResult, write_junit_results

        class IdentifiedTestLike:
            def id(self):
                return "unknown.fixture.hook"

        class StringOnlyTestLike:
            def __str__(self):
                return "undotted-holder"

        runner = unittest.TextTestRunner(
            resultclass=ParallelJunitTestResult,
            stream=io.StringIO(),
            verbosity=0,
        )
        result = runner._makeResult()
        result.addSkip(IdentifiedTestLike(), "identified skip")
        result.addSkip(StringOnlyTestLike(), "string skip")

        self.assertEqual(result.testsRun, 0)
        self.assertEqual(
            [record[:4] for record in result.test_record],
            [
                ("unknown.fixture", "hook", 0.0, "SKIP"),
                ("StringOnlyTestLike", "undotted-holder", 0.0, "SKIP"),
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            junit_path = pathlib.Path(directory, "unknown.xml")
            write_junit_results(
                str(junit_path),
                result.test_record,
                0.0,
            )
            xml_cases = ElementTree.parse(junit_path).getroot().findall("testcase")

        self.assertEqual(
            [(case.get("classname"), case.get("name"), float(case.get("time"))) for case in xml_cases],
            [
                ("unknown.fixture", "hook", 0.0),
                ("StringOnlyTestLike", "undotted-holder", 0.0),
            ],
        )
        for test_id in ("unknown.fixture.hook", "undotted-holder"):
            test_events = [event for event in self.queue.items if event.test_id == test_id]
            self.assertEqual(
                [event.event.value for event in test_events],
                ["test_started", "test_outcome", "test_cleanup_started", "test_stopped"],
            )
            self.assertEqual(test_events[1].outcome, "SKIP")

    def test_unknown_test_like_object_falls_back_when_id_raises(self):
        from warp.tests.unittest_utils import ParallelJunitTestResult, write_junit_results

        class RaisingIdTestLike:
            def __init__(self):
                self.id_calls = 0

            def id(self):
                self.id_calls += 1
                raise RuntimeError("identifier unavailable")

            def __str__(self):
                return "fallback.fixture.skip"

        runner = unittest.TextTestRunner(
            resultclass=ParallelJunitTestResult,
            stream=io.StringIO(),
            verbosity=0,
        )
        result = runner._makeResult()
        test = RaisingIdTestLike()
        result.addSkip(test, "raising ID skip")

        self.assertEqual(result.testsRun, 0)
        self.assertEqual(result.skipped, [(test, "raising ID skip")])
        self.assertEqual(test.id_calls, 1)
        self.assertEqual(
            [record[:4] for record in result.test_record],
            [("fallback.fixture", "skip", 0.0, "SKIP")],
        )

        with tempfile.TemporaryDirectory() as directory:
            junit_path = pathlib.Path(directory, "raising-id.xml")
            write_junit_results(
                str(junit_path),
                result.test_record,
                0.0,
            )
            xml_cases = ElementTree.parse(junit_path).getroot().findall("testcase")

        self.assertEqual(len(xml_cases), 1)
        self.assertEqual(xml_cases[0].get("classname"), "fallback.fixture")
        self.assertEqual(xml_cases[0].get("name"), "skip")
        self.assertEqual(float(xml_cases[0].get("time")), 0.0)
        self.assertEqual(xml_cases[0].find("skipped").get("message"), "raising ID skip")

        test_events = [event for event in self.queue.items if event.test_id == "fallback.fixture.skip"]
        self.assertEqual(
            [event.event.value for event in test_events],
            ["test_started", "test_outcome", "test_cleanup_started", "test_stopped"],
        )
        self.assertEqual(test_events[1].outcome, "SKIP")


class TestParallelCrashIntegration(unittest.TestCase):
    def test_runner_main_uses_warp_owned_module(self):
        from warp._src.test_runner.runner import main

        self.assertEqual(main.__module__, "warp._src.test_runner.runner")

    @staticmethod
    def _args(**overrides):
        values = {
            "buffer": False,
            "coverage": False,
            "coverage_branch": False,
            "failfast": False,
            "junit_report_xml": "enabled",
            "level": "class",
            "no_shared_cache": False,
            "verbose": 0,
            "warp_debug": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _suites(module, *case_names):
        loader = unittest.TestLoader()
        return [loader.loadTestsFromTestCase(getattr(module, case_name)) for case_name in case_names]

    @staticmethod
    def _confirmed_result(class_name):
        return (
            1,
            [],
            [],
            0,
            0,
            0,
            [(class_name, "test_confirmed", 0.0, "OK", None, None)],
        )

    def _run_stub_executor(self, executor, suites):
        from warp._src.test_runner.events import WorkerEventMonitor, WorkerStateTracker
        from warp._src.test_runner.pool import run_parallel_suites

        event_queue = queue_module.Queue()
        tracker = WorkerStateTracker()
        monitor = WorkerEventMonitor(event_queue, tracker)
        monitor.start()
        try:
            with mock.patch(
                "warp._src.test_runner.pool.concurrent.futures.ProcessPoolExecutor",
                return_value=executor,
            ):
                return run_parallel_suites(
                    suites,
                    1,
                    LocalManager(),
                    self._args(),
                    tempfile.gettempdir(),
                    event_queue,
                    tracker,
                    monitor,
                    None,
                    time.monotonic_ns(),
                )
        finally:
            monitor.stop_and_drain()

    def test_monitor_drains_events_before_stopping(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerEventMonitor, WorkerStateTracker
        from warp._src.test_runner.pool import run_parallel_suites

        event_queue = queue_module.Queue()
        tracker = WorkerStateTracker(history_limit=3)
        handled_sequences = []
        monitor = WorkerEventMonitor(
            event_queue,
            tracker,
            on_event=lambda event: handled_sequences.append(event.sequence),
        )
        monitor.start()
        for index in range(64):
            event_queue.put(
                WorkerEvent(
                    sequence=index,
                    event=EventKind.WORKER_STARTED,
                    worker_index=index,
                    pid=1234 + index,
                    monotonic_ns=100 + index,
                    wall_time_ns=1000 + index,
                )
            )

        completed = concurrent.futures.Future()
        completed.set_result(self._confirmed_result("PassingCase"))
        executor = StubExecutor([completed])

        with temporary_fixture_cases() as (module, _, _):
            with mock.patch(
                "warp._src.test_runner.pool.concurrent.futures.ProcessPoolExecutor",
                return_value=executor,
            ):
                result = run_parallel_suites(
                    self._suites(module, "PassingCase"),
                    1,
                    LocalManager(),
                    self._args(),
                    tempfile.gettempdir(),
                    event_queue,
                    tracker,
                    monitor,
                    None,
                    time.monotonic_ns(),
                )

        # `run_parallel_suites` drains the monitor itself on the success path (no
        # separate barrier round-trip needed), so every event queued before the
        # call is reflected in tracker state by the time it returns.
        self.assertIsNone(result.pool_failure)
        self.assertFalse(result.diagnostics_degraded)
        self.assertEqual(handled_sequences, list(range(64)))
        self.assertEqual(
            [snapshot.pid for snapshot in tracker.snapshots(now_ns=1000)],
            list(range(1234, 1298)),
        )
        self.assertIsNone(monitor.error)
        self.assertTrue(monitor.stopped)

    def test_salvages_accepted_future_when_later_submit_fails(self):
        from concurrent.futures.process import BrokenProcessPool

        with temporary_fixture_cases() as (module, _, _):
            completed = concurrent.futures.Future()
            completed.set_result(self._confirmed_result("PassingCase"))
            executor = StubExecutor(
                [
                    completed,
                    BrokenProcessPool("later submission failed"),
                ]
            )

            result = self._run_stub_executor(
                executor,
                self._suites(module, "PassingCase", "FirstPidCase"),
            )

            self.assertIsNotNone(result.pool_failure)
            self.assertEqual(set(result.results_by_index), {0})
            self.assertEqual(result.results_by_index[0][6][0][0], "PassingCase")
            self.assertEqual(result.pool_failure.exception_type, "BrokenProcessPool")
            self.assertEqual(result.pool_failure.reason, "later submission failed")
            self.assertEqual(result.pool_failure.snapshot["failure"]["type"], "BrokenProcessPool")
            self.assertEqual(result.pool_failure.snapshot["failure"]["reason"], "later submission failed")

    def test_salvages_result_published_while_failed_executor_quiesces(self):
        from concurrent.futures.process import BrokenProcessPool

        with temporary_fixture_cases() as (module, _, _):
            broken = concurrent.futures.Future()
            broken.set_exception(BrokenProcessPool("worker exited"))
            published_during_shutdown = NonCancellingFuture()
            published_result = self._confirmed_result("FirstPidCase")
            manager_thread = PublishingManagerThread(published_during_shutdown, published_result)
            executor = StubExecutor([broken, published_during_shutdown], manager_thread=manager_thread)

            result = self._run_stub_executor(
                executor,
                self._suites(module, "PassingCase", "FirstPidCase"),
            )

            self.assertTrue(manager_thread.joined)
            self.assertIsNotNone(result.pool_failure)
            self.assertEqual(set(result.results_by_index), {1})
            self.assertEqual(result.results_by_index[1][6][0][0], "FirstPidCase")

    def test_classification_reports_started_without_result(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker
        from warp._src.test_runner.postmortem import classify_suites

        with temporary_fixture_cases() as (module, _, _):
            suites = self._suites(
                module,
                "BlockingCase",
                "HardExitCase",
                "PassingCase",
                "OrdinaryFailureCase",
                "ShouldNotStartCase",
            )
            tracker = WorkerStateTracker()
            for worker_index, pid, suite_index, suite_name in (
                (0, 7001, 0, "BlockingCase"),
                (1, 7002, 1, "HardExitCase"),
                (2, 7003, 2, "PassingCase"),
                (3, 7004, 3, "OrdinaryFailureCase"),
                (4, 7005, 4, "ShouldNotStartCase"),
            ):
                tracker.handle_event(
                    WorkerEvent(
                        sequence=1,
                        event=EventKind.SUITE_STARTED,
                        worker_index=worker_index,
                        pid=pid,
                        monotonic_ns=1,
                        wall_time_ns=1,
                        suite_index=suite_index,
                        suite_name=suite_name,
                        test_count=1,
                    )
                )
            snapshots = tracker.snapshots(now_ns=2)

            classifications = classify_suites(
                suites,
                {},
                dict.fromkeys(range(len(suites)), "unresolved"),
                snapshots,
            )

        snapshots_by_suite_index = {snapshot.suite_index: snapshot for snapshot in snapshots}
        for classification in classifications:
            self.assertEqual(classification.status, "started")
            snapshot = snapshots_by_suite_index[classification.suite_index]
            self.assertEqual(classification.worker_index, snapshot.worker_index)
            self.assertEqual(classification.pid, snapshot.pid)

    def _run_parallel(self, module, root, case_names, process_count, **arg_overrides):
        from warp._src.test_runner.common import EVENT_HISTORY_LIMIT
        from warp._src.test_runner.events import WorkerEventMonitor, WorkerStateTracker
        from warp._src.test_runner.pool import run_parallel_suites

        spawn_context = multiprocessing.get_context("spawn")
        event_queue = spawn_context.SimpleQueue()
        tracker = WorkerStateTracker(history_limit=EVENT_HISTORY_LIMIT)
        monitor = WorkerEventMonitor(event_queue, tracker)
        run_dir = root / "diagnostics" / "run"
        temp_dir = root / "coverage"
        run_dir.mkdir(parents=True)
        temp_dir.mkdir()
        suites = self._suites(module, *case_names)
        args = self._args(**arg_overrides)
        monitor.start()
        try:
            with spawn_context.Manager() as manager:
                result = run_parallel_suites(
                    suites,
                    process_count,
                    manager,
                    args,
                    str(temp_dir),
                    event_queue,
                    tracker,
                    monitor,
                    run_dir,
                    time.monotonic_ns(),
                )
        finally:
            monitor.stop_and_drain()
            event_queue.close()
        return result, run_dir

    @staticmethod
    def _classification(result, suite_index):
        return next(item for item in result.suite_classifications if item.suite_index == suite_index)

    @staticmethod
    def _worker_for_test(snapshot, test_name):
        for worker in snapshot["workers"]:
            if test_name in json.dumps(worker, sort_keys=True):
                return worker
        raise AssertionError(f"No worker retained lifecycle state for {test_name}")

    @staticmethod
    def _worker_artifact(run_dir, worker, source):
        pattern = f"worker-{worker['worker_index']}-pid-{worker['pid']}.{source}.log"
        matches = list(run_dir.glob(pattern))
        if len(matches) != 1:
            raise AssertionError(f"Expected one artifact matching {pattern}, found {matches}")
        return matches[0]

    @staticmethod
    def _autodiscovered_suite(module, *case_names):
        loader = unittest.TestLoader()
        return unittest.TestSuite(loader.loadTestsFromTestCase(getattr(module, case_name)) for case_name in case_names)

    def _run_main(self, module, root, case_names, *extra_args):
        from warp._src.test_runner.runner import main

        diagnostics_root = root / "main-diagnostics"
        suite = self._autodiscovered_suite(module, *case_names)
        argv = [
            "-s",
            "autodetect",
            "-j",
            "1",
            "--maxjobs",
            "1",
            "-q",
            "--diagnostics-dir",
            str(diagnostics_root),
            *extra_args,
        ]
        with mock.patch("warp.tests.unittest_suites.auto_discover_suite", return_value=suite):
            result = main(argv)
        return result, diagnostics_root

    def test_preserves_completed_out_of_order_result(self):
        with temporary_fixture_cases() as (module, marker_dir, root):
            result, _ = self._run_parallel(
                module,
                root,
                ("BlockingCase", "PassingCase", "HardExitCase"),
                process_count=2,
            )

            self.assertIsNotNone(result.pool_failure)
            self.assertEqual(set(result.results_by_index), {1})
            self.assertTrue(marker_dir.joinpath("blocking-started").exists())
            self._worker_for_test(result.pool_failure.snapshot, "BlockingCase.test_wait_for_pool_shutdown")
            self._worker_for_test(result.pool_failure.snapshot, "HardExitCase.test_exit")
            records = [record for value in result.results_by_index.values() for record in value[6]]
            self.assertFalse(any(record[3] == "ERROR" for record in records))

    def test_hard_exit_lists_all_worker_states(self):
        from warp._src.test_runner.common import ProcessExitProvenance

        with temporary_fixture_cases() as (module, _, root):
            result, _ = self._run_parallel(
                module,
                root,
                ("BlockingCase", "PassingCase", "HardExitCase"),
                process_count=2,
            )

            snapshot = result.pool_failure.snapshot
            self.assertEqual(len(snapshot["workers"]), 2)
            self.assertEqual({worker["worker_index"] for worker in snapshot["workers"]}, {0, 1})
            self.assertIn("last known state", result.pool_failure.formatted_summary)
            self.assertIn("candidate", result.pool_failure.formatted_summary)
            self.assertNotIn("caused by", result.pool_failure.formatted_summary)

            blocking = self._worker_for_test(snapshot, "BlockingCase.test_wait_for_pool_shutdown")
            crashed = self._worker_for_test(snapshot, "HardExitCase.test_exit")
            self.assertEqual(blocking["provenance"], ProcessExitProvenance.PARENT_TERMINATED.value)
            self.assertEqual(self._classification(result, 0).status, "started")
            self.assertEqual(crashed["provenance"], ProcessExitProvenance.INDEPENDENTLY_ABNORMAL.value)
            self.assertEqual(crashed["exit_code"], 86)
            self.assertEqual(self._classification(result, 2).status, "started")

    def test_cleanup_abort_reports_test_cleanup(self):
        with temporary_fixture_cases() as (module, _, root):
            result, _ = self._run_parallel(module, root, ("CleanupAbortCase",), process_count=1)

            worker = self._worker_for_test(result.pool_failure.snapshot, "CleanupAbortCase.test_abort_during_gc")
            self.assertEqual(worker["phase"], "test_cleanup")

    def test_class_finalization_abort_names_suite_not_completed_test(self):
        from warp._src.test_runner.postmortem import make_pool_failure_test_record

        with temporary_fixture_cases() as (module, _, root):
            console = io.StringIO()
            with contextlib.redirect_stderr(console):
                result, run_dir = self._run_parallel(
                    module,
                    root,
                    ("FinalizationAbortCase",),
                    process_count=1,
                    verbose=0,
                )

            worker = result.pool_failure.snapshot["workers"][0]
            completed_test = "fixture_cases.FinalizationAbortCase.test_completes_before_class_finalization"
            self.assertEqual(worker["phase"], "test_stopped")
            self.assertIsNone(worker["current_test_id"])
            self.assertIsNone(worker["current_outcome"])
            self.assertIsNone(worker["current_elapsed_seconds"])
            self.assertEqual(worker["recent_tests"][-1]["test_id"], completed_test)
            self.assertEqual(worker["recent_tests"][-1]["outcome"], "OK")
            self.assertIsNotNone(worker["recent_tests"][-1]["elapsed_seconds"])
            self.assertTrue(worker["artifacts"]["fault"]["fatal_traceback_evidence"])
            on_disk = json.loads(run_dir.joinpath("crash-snapshot.json").read_text(encoding="utf-8"))
            self.assertTrue(on_disk["workers"][0]["artifacts"]["fault"]["fatal_traceback_evidence"])
            report = run_dir.joinpath("pool-failure.txt").read_text(encoding="utf-8")
            self.assertIn("candidate=suite_finalization:FinalizationAbortCase", report)
            self.assertIn("fatal traceback evidence=yes", report)
            junit_record = make_pool_failure_test_record(on_disk, run_dir / "crash-snapshot.json")
            self.assertIn("fatal traceback evidence=yes", junit_record[5])
            self.assertNotIn("Current thread", junit_record[5])
            self.assertIn("Parallel worker pool failed", console.getvalue())

    def test_abort_retains_fault_and_output_logs(self):
        with temporary_fixture_cases() as (module, _, root):
            result, run_dir = self._run_parallel(module, root, ("AbortCase",), process_count=1)

            worker = self._worker_for_test(result.pool_failure.snapshot, "AbortCase.test_abort")
            output_path = self._worker_artifact(run_dir, worker, "output")
            fault_path = self._worker_artifact(run_dir, worker, "fault")
            self.assertIn(b"abort-marker", output_path.read_bytes())
            self.assertIn(b"Fatal Python error", fault_path.read_bytes())

    def test_partial_junit_has_one_pool_failure_error(self):
        from warp._src.test_runner.postmortem import make_pool_failure_test_record
        from warp._src.test_runner.runner import main
        from warp.tests.unittest_utils import write_junit_results

        with temporary_fixture_cases() as (module, _, root):
            result, run_dir = self._run_parallel(
                module,
                root,
                ("BlockingCase", "PassingCase", "HardExitCase"),
                process_count=2,
            )
            snapshot = result.pool_failure.snapshot
            self.assertEqual(snapshot["suite_counts"], {"discovered": 3, "confirmed": 1})
            self.assertNotIn("confirmed_indexes", snapshot)
            self.assertNotIn("recent_tests", snapshot)
            for worker in snapshot["workers"]:
                self.assertIn("worker_index", worker)
                self.assertIn("pid", worker)
                self.assertIn("exit_code", worker)
                self.assertIn("signal_name", worker)
                self.assertIn("provenance", worker)
                self.assertIn("phase", worker)
                self.assertNotIn("candidate", worker)
                self.assertNotIn("diagnostics", worker)
                self.assertIn("current_elapsed_seconds", worker)
                self.assertIn("current_outcome", worker)
                self.assertIn("transition_age_seconds", worker)
                self.assertLessEqual(len(worker["recent_tests"]), 3)
                self.assertEqual(set(worker["artifacts"]), {"journal", "output", "fault"})
                for artifact in worker["artifacts"].values():
                    self.assertIn(artifact["state"], {"missing", "empty", "non_empty", "unreadable"})
                    self.assertIn("path", artifact)
                    self.assertIn("size_bytes", artifact)
            summary = result.pool_failure.formatted_summary
            self.assertIn("confirmed 1/3 discovered suites", summary)
            self.assertIn("transition age=", summary)
            self.assertIn("current/partial elapsed=", summary)
            self.assertIn("recent completed tests", summary)
            self.assertIn("artifacts:", summary)
            on_disk = json.loads(run_dir.joinpath("crash-snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["suite_counts"], {"discovered": 3, "confirmed": 1})
            report = run_dir.joinpath("pool-failure.txt").read_text(encoding="utf-8")
            self.assertIn("confirmed 1/3 discovered suites", report)
            self.assertIn("recent completed tests", report)
            test_records = [
                record for index in sorted(result.results_by_index) for record in result.results_by_index[index][6]
            ]
            pool_record = make_pool_failure_test_record(
                result.pool_failure.snapshot,
                run_dir / "crash-snapshot.json",
            )
            junit_path = root / "rspec.xml"
            write_junit_results(
                str(junit_path),
                test_records,
                0.0,
                extra_records=(pool_record,),
            )

            xml_root = ElementTree.parse(junit_path).getroot()
            pool_errors = xml_root.findall("./testcase[@classname='warp.tests.parallel']")
            self.assertEqual(len(pool_errors), 1)
            self.assertEqual(pool_errors[0].attrib["name"], "WorkerPoolCrash")
            self.assertIsNotNone(pool_errors[0].find("error"))
            junit_error = pool_errors[0].find("error").text
            self.assertIn("confirmed 1/3 discovered suites", junit_error)
            self.assertIn("current/partial elapsed=", junit_error)
            self.assertIn("recent completed tests", junit_error)
            self.assertNotIn("hard-exit-marker", junit_error)
            self.assertEqual(int(xml_root.attrib["errors"]), 1)
            self.assertEqual(int(xml_root.attrib["tests"]), len(xml_root.findall("testcase")))
            self.assertEqual(list(junit_path.parent.glob(".rspec.xml.*.tmp")), [])

            main_junit_path = root / "main-rspec.xml"
            main_diagnostics = root / "main-junit-diagnostics"
            suite = self._autodiscovered_suite(module, "BlockingCase", "PassingCase", "HardExitCase")
            argv = [
                "-s",
                "autodetect",
                "-j",
                "2",
                "--maxjobs",
                "2",
                "-q",
                "--diagnostics-dir",
                str(main_diagnostics),
                "--junit-report-xml",
                str(main_junit_path),
            ]
            with mock.patch("warp.tests.unittest_suites.auto_discover_suite", return_value=suite):
                with self.assertRaises(SystemExit) as raised:
                    main(argv)
            self.assertEqual(raised.exception.code, 1)

            main_root = ElementTree.parse(main_junit_path).getroot()
            main_pool_errors = main_root.findall("./testcase[@classname='warp.tests.parallel']")
            self.assertEqual(len(main_pool_errors), 1)
            self.assertEqual(main_pool_errors[0].attrib["name"], "WorkerPoolCrash")
            self.assertIsNotNone(main_pool_errors[0].find("error"))
            self.assertEqual(int(main_root.attrib["errors"]), 1)
            self.assertEqual(int(main_root.attrib["tests"]), len(main_root.findall("testcase")))
            self.assertEqual(int(main_root.attrib["tests"]), 2)

    def test_clean_results_remain_in_discovery_order(self):
        with temporary_fixture_cases() as (module, _, root):
            result, _ = self._run_parallel(
                module,
                root,
                ("FirstPidCase", "PassingCase", "SecondPidCase"),
                process_count=2,
            )

            self.assertIsNone(result.pool_failure)
            self.assertEqual(list(result.results_by_index), [0, 1, 2])
            self.assertEqual(
                [result.results_by_index[index][6][0][0] for index in result.results_by_index],
                ["FirstPidCase", "PassingCase", "SecondPidCase"],
            )

            main_result, diagnostics_root = self._run_main(
                module,
                root,
                ("FirstPidCase", "PassingCase", "SecondPidCase"),
            )
            self.assertIsNone(main_result)
            run_dir = next(diagnostics_root.glob("run-*"))
            timing_payload = json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [record["suite_name"] for record in timing_payload["suites"]],
                ["FirstPidCase", "PassingCase", "SecondPidCase"],
            )
            self.assertTrue(all(record["status"] == "complete" for record in timing_payload["suites"]))
            self.assertEqual(list(run_dir.glob("worker-*")), [])

    def test_ordinary_failure_does_not_become_pool_failure(self):
        with temporary_fixture_cases() as (module, _, root):
            result, _ = self._run_parallel(module, root, ("OrdinaryFailureCase",), process_count=1)

            self.assertIsNone(result.pool_failure)
            self.assertEqual(len(result.results_by_index[0][2]), 1)
            self.assertEqual(self._classification(result, 0).status, "confirmed")

            with self.assertRaises(SystemExit) as raised:
                self._run_main(module, root, ("OrdinaryFailureCase",))
            self.assertEqual(raised.exception.code, 1)
            run_dir = next((root / "main-diagnostics").glob("run-*"))
            timing_payload = json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertEqual(len(timing_payload["suites"]), 1)
            self.assertEqual(timing_payload["suites"][0]["status"], "complete")
            self.assertEqual(timing_payload["suites"][0]["outcomes"], {"FAIL": 1})
            self.assertTrue(list(run_dir.glob("worker-*.events.jsonl")))
            self.assertTrue(list(run_dir.glob("worker-*.output.log")))
            self.assertTrue(list(run_dir.glob("worker-*.fault.log")))

    def test_diagnostics_setup_failure_preserves_primary_test_failure(self):
        with temporary_fixture_cases() as (module, _, root):
            console = io.StringIO()
            with (
                mock.patch(
                    "warp._src.test_runner.runner.create_diagnostics_run_dir",
                    side_effect=OSError("diagnostics setup unavailable"),
                ),
                contextlib.redirect_stderr(console),
                self.assertRaises(SystemExit) as raised,
            ):
                self._run_main(
                    module,
                    root,
                    ("OrdinaryFailureCase",),
                )

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("ordinary assertion marker", console.getvalue())
            self.assertIn(
                "Failed to create diagnostics run directory: diagnostics setup unavailable",
                console.getvalue(),
            )

    def test_diagnostics_finalization_failure_does_not_replace_test_failure(self):
        with temporary_fixture_cases() as (module, _, root):
            console = io.StringIO()
            with (
                mock.patch(
                    "warp._src.test_runner.runner.finalize_diagnostics",
                    side_effect=OSError("diagnostics finalization unavailable"),
                ),
                contextlib.redirect_stderr(console),
                self.assertRaises(SystemExit) as raised,
            ):
                self._run_main(
                    module,
                    root,
                    ("OrdinaryFailureCase",),
                )

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("ordinary assertion marker", console.getvalue())
            self.assertIn(
                "Failed to clean up durable diagnostics: diagnostics finalization unavailable",
                console.getvalue(),
            )

    def test_process_controls_escape_after_success_and_failure(self):
        primary_cases = (("PassingCase", "OK"), ("OrdinaryFailureCase", "FAILED"))
        for control in (KeyboardInterrupt("stop now"), SystemExit(73)):
            for case_name, expected_status in primary_cases:
                with self.subTest(control=type(control).__name__, primary=case_name):
                    with temporary_fixture_cases() as (module, _, root):
                        console = io.StringIO()
                        with (
                            mock.patch(
                                "warp._src.test_runner.runner.finalize_diagnostics",
                                side_effect=control,
                            ),
                            contextlib.redirect_stderr(console),
                            self.assertRaises(type(control)) as raised,
                        ):
                            self._run_main(
                                module,
                                root,
                                (case_name,),
                            )

                        self.assertIs(raised.exception, control)
                        self.assertIn(expected_status, console.getvalue())
                        run_dir = next((root / "main-diagnostics").glob("run-*"))
                        self.assertTrue(run_dir.joinpath("run.json").is_file())
                        self.assertTrue(list(run_dir.glob("worker-*.events.jsonl")))
                        self.assertFalse(any(thread.name == "worker-event-monitor" for thread in threading.enumerate()))

    def test_failfast_prevents_unstarted_suite_execution(self):
        with temporary_fixture_cases() as (module, marker_dir, root):
            result, _ = self._run_parallel(
                module,
                root,
                ("OrdinaryFailureCase", "ShouldNotStartCase"),
                process_count=1,
                failfast=True,
            )

            self.assertIsNone(result.pool_failure)
            self.assertEqual(len(result.results_by_index[0][2]), 1)
            self.assertFalse(marker_dir.joinpath("unexpected-start").exists())
            self.assertEqual(self._classification(result, 1).status, "skipped")

    def test_output_pressure_keeps_worker_attribution(self):
        with temporary_fixture_cases() as (module, _, root):
            result, run_dir = self._run_parallel(
                module,
                root,
                ("OutputPressureCase", "AbortCase"),
                process_count=2,
            )

            pressure_worker = self._worker_for_test(
                result.pool_failure.snapshot, "OutputPressureCase.test_output_pressure"
            )
            abort_worker = self._worker_for_test(result.pool_failure.snapshot, "AbortCase.test_abort")
            pressure_output_path = self._worker_artifact(run_dir, pressure_worker, "output")
            abort_output_path = self._worker_artifact(run_dir, abort_worker, "output")
            pressure_output = pressure_output_path.read_bytes()
            abort_output = abort_output_path.read_bytes()
            self.assertIn(b"output-pressure-marker", pressure_output)
            self.assertIn(b"abort-marker", abort_output)
            for output_path in run_dir.glob("worker-*.output.log"):
                output = output_path.read_bytes()
                if output_path != pressure_output_path:
                    self.assertNotIn(b"output-pressure-marker", output)
                if output_path != abort_output_path:
                    self.assertNotIn(b"abort-marker", output)
            if pressure_worker["pid"] == abort_worker["pid"]:
                self.assertEqual(pressure_worker["worker_index"], abort_worker["worker_index"])
                lifecycle_state = json.dumps(pressure_worker, sort_keys=True)
                self.assertIn("OutputPressureCase.test_output_pressure", lifecycle_state)
                self.assertIn("AbortCase.test_abort", lifecycle_state)

    def test_crash_remains_failure(self):
        with temporary_fixture_cases() as (module, marker_dir, root):
            with self.assertRaises(SystemExit) as raised:
                self._run_main(module, root, ("AbortCase",))

            self.assertEqual(raised.exception.code, 1)
            snapshot_path = next((root / "main-diagnostics").glob("run-*/crash-snapshot.json"))
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["failure"]["type"], "BrokenProcessPool")
            attempts_path = marker_dir / "abort-attempts"
            self.assertTrue(attempts_path.is_file())
            self.assertEqual(attempts_path.read_text(encoding="utf-8"), "1")

    @unittest.skipIf(sys.version_info < (3, 11), "Process isolation requires Python 3.11 or newer")
    def test_isolation_uses_fresh_worker_pids(self):
        with temporary_fixture_cases() as (module, marker_dir, root):
            result, _ = self._run_main(
                module,
                root,
                ("FirstPidCase", "SecondPidCase"),
                "--isolate-test-processes",
            )

            self.assertIsNone(result)
            first_pid = marker_dir.joinpath("first-pid").read_text(encoding="utf-8")
            second_pid = marker_dir.joinpath("second-pid").read_text(encoding="utf-8")
            self.assertNotEqual(first_pid, second_pid)


class TestModuleLoadCollection(unittest.TestCase):
    def test_collection_aggregates_churn_repeats_and_slowest(self):
        from warp._src.test_runner.module_loads import collect_module_load_summary

        with tempfile.TemporaryDirectory() as temp_root:
            run_dir = pathlib.Path(temp_root)
            log = run_dir / "worker-0-pid-100.output.log"
            log.write_text(
                "Module wp.sim abc1234 load on device 'cuda:0' took 100.0 ms (compiled)\n"
                "Module wp.sim abc1234 load on device 'cuda:0' took 50.0 ms (compiled)\n"
                "Module wp.sim def5678 load on device 'cuda:0' took 25.0 ms (error)\n"
                "Module wp.fem 1111111 load on device 'cpu' took 10.0 ms (cached)\n"
                "Module wp.top load on device 'cpu' took 500.0 ms (compiled)\n",
                encoding="utf-8",
            )
            summary = collect_module_load_summary(run_dir, no_shared_cache=False)

        self.assertEqual(summary.status_counts, {"compiled": 3, "cached": 1, "error": 1})
        self.assertEqual(len(summary.hash_churn), 1)
        churn = summary.hash_churn[0]
        self.assertEqual((churn.module, churn.device), ("wp.sim", "cuda:0"))
        self.assertEqual(churn.hashes, ("abc1234", "def5678"))
        self.assertEqual((churn.attempts, churn.compiled_count, churn.error_count), (3, 2, 1))
        self.assertEqual(churn.aggregate_ms, 175.0)
        self.assertEqual(len(summary.repeated_compilations), 1)
        repeat = summary.repeated_compilations[0]
        self.assertEqual((repeat.module, repeat.module_hash, repeat.compilations), ("wp.sim", "abc1234", 2))
        self.assertEqual(repeat.aggregate_ms, 150.0)
        self.assertEqual([record.module for record in summary.slowest_compiled], ["wp.top", "wp.sim", "wp.sim"])


class TestExitProvenance(unittest.TestCase):
    """Attribution of worker exits when the pool tears itself down.

    CPython's ``_ExecutorManagerThread._terminate_broken()`` publishes
    ``BrokenProcessPool`` to the pending futures *before* it SIGTERMs the
    surviving workers, so by the time the runner samples liveness the blocked
    worker may already be reaped. Liveness alone therefore reports whichever
    thread won that race; the worker's last journaled phase does not.
    """

    @staticmethod
    def _provenance(exit_code, alive_at_failure=False, terminated_while_working=False):
        from warp._src.test_runner.pool import _derive_provenance

        return _derive_provenance(exit_code, alive_at_failure, terminated_while_working).value

    def test_sigterm_while_working_is_parent_terminated_without_liveness(self):
        from warp._src.test_runner.common import ProcessExitProvenance

        self.assertEqual(
            self._provenance(-signal.SIGTERM, alive_at_failure=False, terminated_while_working=True),
            ProcessExitProvenance.PARENT_TERMINATED.value,
        )

    def test_liveness_and_journal_agree_on_parent_termination(self):
        losing_the_race = self._provenance(-signal.SIGTERM, alive_at_failure=False, terminated_while_working=True)
        winning_the_race = self._provenance(-signal.SIGTERM, alive_at_failure=True, terminated_while_working=True)
        self.assertEqual(losing_the_race, winning_the_race)

    def test_sigterm_after_shutdown_stays_unresolved(self):
        from warp._src.test_runner.common import ProcessExitProvenance

        self.assertEqual(
            self._provenance(-signal.SIGTERM, terminated_while_working=False),
            ProcessExitProvenance.UNRESOLVED.value,
        )

    def test_sigkill_while_working_stays_unresolved(self):
        from warp._src.test_runner.common import ProcessExitProvenance

        self.assertEqual(
            self._provenance(-signal.SIGKILL, terminated_while_working=True),
            ProcessExitProvenance.UNRESOLVED.value,
        )

    def test_hard_exit_code_remains_independently_abnormal(self):
        from warp._src.test_runner.common import ProcessExitProvenance

        self.assertEqual(
            self._provenance(86, terminated_while_working=True),
            ProcessExitProvenance.INDEPENDENTLY_ABNORMAL.value,
        )

    def test_report_explains_unresolved_sigkill_as_cleanup_or_external_killer(self):
        from warp._src.test_runner.common import ProcessExit, ProcessExitProvenance
        from warp._src.test_runner.postmortem import build_crash_snapshot, format_pool_failure

        exits = (ProcessExit(9001, -signal.SIGKILL, "SIGKILL", ProcessExitProvenance.UNRESOLVED.value),)
        summary = format_pool_failure(build_crash_snapshot(RuntimeError("pool failed"), (), (), exits, None))

        self.assertIn("SIGKILL exits are unresolved", summary)
        self.assertIn("OOM killer", summary)
        self.assertNotIn("SIGTERM exits are unresolved", summary)


class TestSuiteTimings(unittest.TestCase):
    @staticmethod
    def _timing(**overrides):
        from warp._src.test_runner.common import SuiteTiming

        values = {
            "suite_index": 0,
            "suite_name": "fixture.TestTiming",
            "unit_type": "class",
            "test_count": 1,
            "worker_index": 0,
            "pid": 4818,
            "started_offset_seconds": 1.0,
            "finished_offset_seconds": 2.0,
            "elapsed_seconds": 1.0,
            "completion_order": 1,
            "status": "complete",
            "outcomes": {"OK": 1},
        }
        values.update(overrides)
        return SuiteTiming(**values)

    def test_tracker_measures_worker_occupancy_from_suite_events(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker()

        def event(kind, sequence, monotonic_ns, **fields):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=4,
                pid=4818,
                monotonic_ns=monotonic_ns,
                wall_time_ns=1_000_000_000 + monotonic_ns,
                suite_index=9,
                suite_name="fixture.TestSlow",
                test_count=184,
                **fields,
            )

        tracker.handle_event(event(EventKind.SUITE_STARTED, 1, 2_000_000_000))
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                2,
                3_000_000_000,
                test_id="fixture.TestSlow.test_ok",
                outcome="OK",
            )
        )
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                3,
                4_000_000_000,
                test_id="fixture.TestSlow.test_skip",
                outcome="SKIP",
            )
        )
        tracker.handle_event(event(EventKind.SUITE_FINISHED, 4, 7_500_000_000))

        timing = tracker.suite_timings()[0]
        self.assertEqual(timing.suite_index, 9)
        self.assertEqual(timing.suite_name, "fixture.TestSlow")
        self.assertEqual(timing.unit_type, "class")
        self.assertEqual(timing.test_count, 184)
        self.assertEqual(timing.worker_index, 4)
        self.assertEqual(timing.pid, 4818)
        self.assertEqual(timing.started_offset_seconds, 2.0)
        self.assertEqual(timing.finished_offset_seconds, 7.5)
        self.assertEqual(timing.elapsed_seconds, 5.5)
        self.assertEqual(timing.completion_order, 1)
        self.assertEqual(timing.status, "complete")
        self.assertEqual(timing.outcomes, {"OK": 1, "SKIP": 1})

    def test_tracker_keeps_active_and_classification_only_records_separate(self):
        from warp._src.test_runner.common import EventKind, SuiteClassification, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker()
        tracker.handle_event(
            WorkerEvent(
                sequence=1,
                event=EventKind.SUITE_STARTED,
                worker_index=2,
                pid=8123,
                monotonic_ns=4_000_000_000,
                wall_time_ns=5_000_000_000,
                suite_index=1,
                suite_name="fixture.TestActive",
                test_count=3,
            )
        )
        classifications = (
            SuiteClassification(1, "fixture.TestActive", 3, "started", 2, 8123),
            SuiteClassification(2, "fixture.TestCancelled", 4, "skipped"),
            SuiteClassification(3, "fixture.TestPending", 5, "never_started"),
        )

        records = tracker.suite_timings(classifications, unit_type="module")

        self.assertEqual([record.suite_index for record in records], [1, 2, 3])
        self.assertEqual(records[0].status, "started")
        self.assertEqual(records[0].started_offset_seconds, 4.0)
        self.assertIsNone(records[0].finished_offset_seconds)
        self.assertIsNone(records[0].elapsed_seconds)
        self.assertEqual(records[0].worker_index, 2)
        self.assertEqual(records[0].unit_type, "module")
        self.assertEqual(records[1].status, "skipped")
        self.assertIsNone(records[1].worker_index)
        self.assertEqual(records[2].status, "never_started")

    def test_slowest_summary_ranks_only_twenty_completed_records(self):
        from warp._src.test_runner.artifacts import format_slowest_suites

        records = [
            self._timing(
                suite_index=index,
                suite_name=f"fixture.TestTiming{index:02d}",
                elapsed_seconds=float(index + 1),
                finished_offset_seconds=float(index + 2),
                completion_order=index + 1,
            )
            for index in range(25)
        ]
        records.append(
            self._timing(
                suite_index=25,
                suite_name="fixture.TestIncomplete",
                finished_offset_seconds=None,
                elapsed_seconds=None,
                completion_order=None,
                status="started",
                outcomes={},
            )
        )

        summary = format_slowest_suites(records, limit=20)
        completed_block, incomplete_block = summary.split("Incomplete suites:")
        completed_rows = [line for line in completed_block.splitlines() if line.startswith("  ")]

        self.assertEqual(len(completed_rows), 20)
        self.assertEqual(
            [line.split()[1] for line in completed_rows],
            [f"fixture.TestTiming{index:02d}" for index in range(24, 4, -1)],
        )
        self.assertNotIn("fixture.TestIncomplete", completed_block)
        self.assertIn("fixture.TestIncomplete", incomplete_block)

    def test_writes_compact_timing_json_atomically(self):
        from warp._src.test_runner.artifacts import write_suite_timings

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            write_suite_timings(run_dir, {"worker_count": 2}, [self._timing()])

            payload = json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"], {"worker_count": 2})
            self.assertEqual(payload["suites"][0]["suite_name"], "fixture.TestTiming")
            self.assertEqual(list(run_dir.glob(".suite-timings.json.*.tmp")), [])

    def test_finalization_removes_only_current_run_worker_evidence_on_success(self):
        from warp._src.test_runner.artifacts import finalize_diagnostics

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            retained = (run_dir / "run.json", run_dir / "suite-timings.json", run_dir / "notes.txt")
            generated = (
                run_dir / "worker-0-4818.events.jsonl",
                run_dir / "worker-0-pid-4818.output.log",
                run_dir / "worker-0-pid-4818.fault.log",
            )
            for path in (*retained, *generated):
                path.write_text(path.name, encoding="utf-8")

            finalize_diagnostics(run_dir, retain_worker_evidence=False)

            self.assertTrue(all(path.exists() for path in retained))
            self.assertTrue(all(not path.exists() for path in generated))

    def test_finalization_keeps_all_evidence_when_requested(self):
        from warp._src.test_runner.artifacts import finalize_diagnostics

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            paths = (
                run_dir / "run.json",
                run_dir / "suite-timings.json",
                run_dir / "worker-0-4818.events.jsonl",
                run_dir / "worker-0-pid-4818.output.log",
                run_dir / "worker-0-pid-4818.fault.log",
                run_dir / "crash-snapshot.json",
                run_dir / "pool-failure.txt",
            )
            for path in paths:
                path.write_text(path.name, encoding="utf-8")

            finalize_diagnostics(run_dir, retain_worker_evidence=True)

            self.assertTrue(all(path.exists() for path in paths))

    def test_metadata_allowlists_environment_values(self):
        from warp._src.test_runner.artifacts import build_run_metadata

        args = SimpleNamespace(
            coverage=True,
            coverage_branch=True,
            level="class",
            pattern="test_timing*.py",
            suite="autodetect",
            testNamePatterns=["*timing*"],
            warp_debug=True,
        )
        environment = {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "WARP_CACHE_PATH": "/opt/warp-cache",
            "CI_JOB_ID": "123456",
            "GITHUB_ACTIONS": "true",
            "SECRET_TOKEN": "do-not-serialize",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            metadata = build_run_metadata(
                args,
                process_count=3,
                run_start_monotonic_ns=10,
                run_start_wall_time_ns=20,
                parent_gil_enabled_initial=None,
            )

        encoded = json.dumps(metadata, sort_keys=True)
        self.assertEqual(metadata["worker_count"], 3)
        self.assertEqual(metadata["level"], "class")
        self.assertEqual(metadata["ci_providers"], ["GitHub Actions"])
        self.assertEqual(metadata["cuda_visible_devices"], "0,1")
        self.assertEqual(metadata["warp_cache_path"], "/opt/warp-cache")
        self.assertNotIn("CI_JOB_ID", encoded)
        self.assertNotIn("123456", encoded)
        self.assertNotIn("SECRET_TOKEN", encoded)
        self.assertNotIn("do-not-serialize", encoded)
        self.assertNotIn("environment", metadata)
        self.assertNotIn("serial_fallback", metadata)

    def test_run_metadata_distinguishes_parent_gil_endpoints(self):
        import inspect

        from warp._src.test_runner.artifacts import build_run_metadata

        self.assertIn("parent_gil_enabled_initial", inspect.signature(build_run_metadata).parameters)
        args = SimpleNamespace(
            coverage=False,
            coverage_branch=False,
            level="class",
            pattern="test*.py",
            suite="default",
            testNamePatterns=[],
            warp_debug=False,
        )
        metadata = build_run_metadata(
            args,
            process_count=2,
            run_start_monotonic_ns=10,
            run_start_wall_time_ns=20,
            parent_gil_enabled_initial=False,
            finished={
                "run_finished_monotonic_ns": 30,
                "run_finished_wall_time_ns": 40,
                "parent_gil_enabled_final": True,
            },
        )

        self.assertFalse(metadata["parent_gil_enabled_initial"])
        self.assertTrue(metadata["parent_gil_enabled_final"])
        self.assertNotIn("gil_enabled", metadata)

    def test_quiet_run_suppresses_ordinary_worker_lifecycle_output(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.runner import _make_worker_event_callback

        events = (
            WorkerEvent(
                sequence=1,
                event=EventKind.TEST_STARTED,
                worker_index=3,
                pid=4818,
                monotonic_ns=93_000_000_000,
                wall_time_ns=99_000_000_000,
                suite_index=9,
                suite_name="fixture.TestSlow",
                test_id="fixture.TestSlow.test_waits",
            ),
            WorkerEvent(
                sequence=2,
                event=EventKind.SUITE_FINISHED,
                worker_index=3,
                pid=4818,
                monotonic_ns=93_427_000_000,
                wall_time_ns=100_000_000_000,
                suite_index=9,
                suite_name="fixture.TestSlow",
                elapsed_seconds=93.427,
                test_count=184,
            ),
        )
        output = io.StringIO()

        with mock.patch("sys.stderr", output):
            callback = _make_worker_event_callback(SimpleNamespace(verbose=0))
            for event in events:
                callback(event)

        self.assertEqual(output.getvalue(), "")

    def test_quiet_run_suppresses_final_timing_summary(self):
        from warp._src.test_runner.runner import _RunDiagnostics

        args = SimpleNamespace(
            coverage=False,
            coverage_branch=False,
            level="class",
            pattern=None,
            suite="autodetect",
            testNamePatterns=None,
            verbose=0,
            warp_debug=False,
        )
        tracker = mock.Mock()
        tracker.suite_timings.return_value = (self._timing(),)
        diagnostics = _RunDiagnostics()
        diagnostics.configure(args, 1, 1, 1, None)
        diagnostics.tracker = tracker
        output = io.StringIO()

        with mock.patch("sys.stderr", output):
            diagnostics.finalize()

        self.assertEqual(output.getvalue(), "")
        tracker.suite_timings.assert_called_once_with((), unit_type="class")

    def test_verbose_test_lifecycle_prefix_includes_worker_and_pid(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.runner import _make_worker_event_callback

        event = WorkerEvent(
            sequence=2,
            event=EventKind.TEST_STARTED,
            worker_index=3,
            pid=4818,
            monotonic_ns=10,
            wall_time_ns=20,
            suite_index=9,
            suite_name="fixture.TestSlow",
            test_id="fixture.TestSlow.test_waits",
        )
        output = io.StringIO()

        with mock.patch("sys.stderr", output):
            _make_worker_event_callback(SimpleNamespace(verbose=2))(event)

        self.assertEqual(output.getvalue(), "[worker 3 pid=4818] fixture.TestSlow.test_waits ...\n")

    def test_cleanup_exception_after_clean_body_retains_evidence(self):
        from warp._src.test_runner.runner import main

        class CleanupFailure(RuntimeError):
            pass

        with temporary_fixture_cases() as (module, _, root):
            diagnostics_root = root / "diagnostics"
            coverage_root = root / "coverage"
            coverage_root.mkdir()
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.PassingCase)

            class RaisingTemporaryDirectory:
                def __enter__(self):
                    return str(coverage_root)

                def __exit__(self, exception_type, exception, traceback):
                    raise CleanupFailure("temporary directory cleanup failed")

            argv = [
                "-s",
                "autodetect",
                "-j",
                "1",
                "--maxjobs",
                "1",
                "-q",
                "--diagnostics-dir",
                str(diagnostics_root),
            ]
            with mock.patch("warp.tests.unittest_suites.auto_discover_suite", return_value=suite):
                with mock.patch(
                    "warp._src.test_runner.runner.tempfile.TemporaryDirectory",
                    RaisingTemporaryDirectory,
                ):
                    with self.assertRaisesRegex(CleanupFailure, "temporary directory cleanup failed"):
                        main(argv)

            run_dir = next(diagnostics_root.glob("run-*"))
            json.loads(run_dir.joinpath("run.json").read_text(encoding="utf-8"))
            json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertTrue(list(run_dir.glob("worker-*.events.jsonl")))
            self.assertFalse(any(thread.name == "worker-event-monitor" for thread in threading.enumerate()))

    def test_missing_coverage_preflight_writes_compact_diagnostics(self):
        from warp._src.test_runner.runner import main

        with tempfile.TemporaryDirectory() as directory:
            diagnostics_root = pathlib.Path(directory, "diagnostics")
            stderr = io.StringIO()
            argv = ["--coverage", "--diagnostics-dir", str(diagnostics_root)]

            with mock.patch("warp._src.test_runner.runner.COVERAGE_AVAILABLE", False):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(argv)

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("coverage was not found", stderr.getvalue())
            run_dir = next(diagnostics_root.glob("run-*"))
            run_metadata = json.loads(run_dir.joinpath("run.json").read_text(encoding="utf-8"))
            timings = json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertTrue(run_metadata["coverage"])
            self.assertEqual(timings["suites"], [])
            self.assertFalse(any(thread.name == "worker-event-monitor" for thread in threading.enumerate()))


HARNESS_TEST_CASES = [
    TestDiagnosticEvents,
    TestWorkerSinkIsolation,
    TestResultLifecycle,
    TestParallelJunitFixtures,
    TestParallelCrashIntegration,
    TestModuleLoadCollection,
    TestExitProvenance,
    TestSuiteTimings,
]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_case in HARNESS_TEST_CASES:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    return suite


if __name__ == "__main__":
    unittest.main()
