# Licensed under the MIT License
# https://github.com/craigahobbs/unittest-parallel/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp's internal parallel test-runner command-line implementation."""

import argparse
import dataclasses
import multiprocessing
import os
import sys
import tempfile
import time
import unittest

import warp.tests.unittest_suites  # NVIDIA Modification
import warp.tests.unittest_utils
from warp._src.test_runner.artifacts import (
    atomic_write_json,
    build_run_metadata,
    create_diagnostics_run_dir,
    finalize_diagnostics,
    format_slowest_suites,
    resolve_diagnostics_root,
    write_suite_timings,
)
from warp._src.test_runner.common import (
    COVERAGE_AVAILABLE,
    EVENT_HISTORY_LIMIT,
    EventKind,
    coverage_context,
    get_gil_enabled,
    get_warp_cache_base_path,
    warn,
)
from warp._src.test_runner.events import (
    WorkerEventMonitor,
    WorkerStateTracker,
)
from warp._src.test_runner.module_loads import (
    collect_module_load_summary,
    format_module_load_summary,
)
from warp._src.test_runner.pool import run_parallel_suites
from warp._src.test_runner.postmortem import make_pool_failure_test_record
from warp.tests.unittest_utils import write_junit_results  # NVIDIA modification

# The following variables are NVIDIA Modifications
START_DIRECTORY = os.path.join(os.path.dirname(__file__), "..")  # The directory to start test discovery


def _convert_select_pattern(pattern):
    if "*" not in pattern:
        return f"*{pattern}*"
    return pattern


class _RunDiagnostics:
    """Best-effort diagnostics around the primary test run.

    Diagnostic failures must never mask test results: ordinary errors are
    reported and skipped, while KeyboardInterrupt/SystemExit re-raise after
    the remaining cleanup steps have run.
    """

    def __init__(self):
        self.args = None
        self.process_count = 0
        self.run_start_monotonic_ns = 0
        self.run_start_wall_time_ns = 0
        self.parent_gil_enabled_initial = None
        self.run_dir = None
        self.event_queue = None
        self.tracker = None
        self.monitor = None
        self.suite_classifications = ()
        self.unit_type = "class"
        self.retain_worker_evidence = True
        self._evidence_required = False
        self._monitor_degraded = False
        self._monitor_stopped = False
        self._finalized = False

    def configure(self, args, process_count, run_start_monotonic_ns, run_start_wall_time_ns, run_dir):
        self.args = args
        self.process_count = process_count
        self.run_start_monotonic_ns = run_start_monotonic_ns
        self.run_start_wall_time_ns = run_start_wall_time_ns
        self.parent_gil_enabled_initial = get_gil_enabled()
        self.run_dir = run_dir
        self.unit_type = args.level
        self._write_run_metadata_or_disable("initialize durable diagnostics")

    def attach_monitor(self, event_queue, tracker, monitor):
        self.event_queue = event_queue
        self.tracker = tracker
        self.monitor = monitor

    def update_process_count(self, process_count):
        self.process_count = process_count
        self._write_run_metadata_or_disable("update durable diagnostics")

    def require_evidence(self):
        self._evidence_required = True
        self.retain_worker_evidence = True

    def degrade_event_monitor(self):
        self.require_evidence()
        self._monitor_degraded = True

    def mark_clean(self):
        if not self._evidence_required:
            self.retain_worker_evidence = False

    def best_effort(self, label, operation):
        """Run one diagnostic step; report ordinary failures instead of masking results."""
        try:
            operation()
        except (KeyboardInterrupt, SystemExit):
            self.require_evidence()
            raise
        except Exception as error:
            self.require_evidence()
            warn(f"Failed to {label}: {error}")

    def _write_run_metadata_or_disable(self, label):
        try:
            self._write_run_metadata()
        except (KeyboardInterrupt, SystemExit):
            self.require_evidence()
            raise
        except Exception as error:
            self.require_evidence()
            warn(f"Failed to {label}: {error}")
            self.run_dir = None

    def _metadata(self, finished=False):
        if self.args is None:
            return None
        options = None
        if finished:
            options = {
                "run_finished_monotonic_ns": time.monotonic_ns(),
                "run_finished_wall_time_ns": time.time_ns(),
                "parent_gil_enabled_final": get_gil_enabled(),
            }
        return build_run_metadata(
            self.args,
            self.process_count,
            self.run_start_monotonic_ns,
            self.run_start_wall_time_ns,
            parent_gil_enabled_initial=self.parent_gil_enabled_initial,
            finished=options,
        )

    def _write_run_metadata(self, finished=False):
        if self.run_dir is None:
            return
        metadata = self._metadata(finished=finished)
        if metadata is not None:
            atomic_write_json(self.run_dir / "run.json", metadata)

    def stop_monitor(self):
        """Stop live event handling; safe to call more than once."""
        if self._monitor_stopped:
            return
        self._monitor_stopped = True
        try:
            if self.monitor is not None:
                if self._monitor_degraded:
                    self.monitor.seal()
                    if self.tracker is not None:
                        self.tracker.raise_monitor_error()
                elif not self.monitor.stopped:
                    try:
                        self.monitor.stop_and_drain()
                    except BaseException:
                        self.require_evidence()
                        self.monitor.seal()
                        raise
        finally:
            if self.event_queue is not None:
                close = getattr(self.event_queue, "close", None)
                if close is not None:
                    close()

    def _finalize_module_loads(self):
        if self.run_dir is None or self.args is None:
            return
        summary = collect_module_load_summary(
            self.run_dir,
            no_shared_cache=bool(getattr(self.args, "no_shared_cache", False)),
        )
        if summary is None:
            return
        if summary.read_errors:
            self.require_evidence()
            for error in summary.read_errors:
                warn(f"Failed to read worker output for module-load diagnostics: {error}")
        self.best_effort(
            "write module-load diagnostics",
            lambda: atomic_write_json(self.run_dir / "module-load-summary.json", summary.to_dict()),
        )
        if self.args.verbose > 0:
            report = format_module_load_summary(summary)
            print(file=sys.stderr)
            print(report, file=sys.stderr)

    def _finalize_summaries(self):
        records = ()
        if self.tracker is not None:
            records = self.tracker.suite_timings(self.suite_classifications, unit_type=self.unit_type)
            if self.args is not None and self.args.verbose > 0:
                print(file=sys.stderr)
                print(format_slowest_suites(records), file=sys.stderr)
        metadata = self._metadata(finished=True)
        if self.run_dir is not None and metadata is not None:
            atomic_write_json(self.run_dir / "run.json", metadata)
            write_suite_timings(self.run_dir, metadata, records)

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True
        steps = (
            ("stop diagnostics monitor", self.stop_monitor),
            ("finalize module-load diagnostics", self._finalize_module_loads),
            ("finalize compact diagnostics", self._finalize_summaries),
            ("clean up durable diagnostics", lambda: finalize_diagnostics(self.run_dir, self.retain_worker_evidence)),
        )
        control_error = None
        for label, operation in steps:
            try:
                self.best_effort(label, operation)
            except (KeyboardInterrupt, SystemExit) as error:
                if control_error is None:
                    control_error = error
        if control_error is not None:
            raise control_error


def _make_worker_event_callback(args):
    def on_event(event):
        if args.verbose <= 0:
            return
        if event.event is EventKind.SUITE_FINISHED:
            elapsed = event.elapsed_seconds if event.elapsed_seconds is not None else 0.0
            test_count = event.test_count if event.test_count is not None else 0
            print(
                f"[worker {event.worker_index:02d} pid={event.pid}] SUITE {event.suite_name} "
                f"completed {test_count} tests in {elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
            )
            return
        prefix = f"[worker {event.worker_index} pid={event.pid}]"
        if event.event is EventKind.TEST_STARTED:
            if args.verbose > 1:
                print(f"{prefix} {event.test_id} ...", file=sys.stderr, flush=True)
        elif event.event is EventKind.TEST_OUTCOME:
            if args.verbose > 1:
                elapsed = f" ({event.elapsed_seconds:.3f}s)" if event.elapsed_seconds is not None else ""
                print(f"{prefix} {event.test_id} ... {event.outcome}{elapsed}", file=sys.stderr, flush=True)

    return on_event


def main(argv=None):
    diagnostics = _RunDiagnostics()
    try:
        return _main(argv, diagnostics)
    finally:
        diagnostics.finalize()


def _create_argument_parser():
    parser = argparse.ArgumentParser(
        prog="warp.tests",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Example usage:
        python -m warp.tests -s autodetect -p 'test_a*.py'
        python -m warp.tests -s debug
        python -m warp.tests -k 'mgpu' -k 'cuda'
        """,
    )
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_const", const=0, default=2, help="Quiet output")
    parser.add_argument("-f", "--failfast", action="store_true", default=False, help="Stop on first fail or error")
    parser.add_argument(
        "-b", "--buffer", action="store_true", default=False, help="Buffer stdout and stderr during tests"
    )
    parser.add_argument(
        "-k",
        dest="testNamePatterns",
        action="append",
        type=_convert_select_pattern,
        help="Only run tests which match the given substring",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        metavar="PATTERN",
        default="test*.py",
        help="'autodetect' suite only: Pattern to match tests ('test*.py' default)",
    )
    parser.add_argument(
        "-t",
        "--top-level-directory",
        metavar="TOP",
        help="Top level directory of project (defaults to start directory)",
    )
    parser.add_argument("--junit-report-xml", metavar="FILE", help="Generate JUnit report format XML file")
    parser.add_argument(
        "--diagnostics-dir",
        help="Directory for durable per-worker crash evidence. Defaults to WARP_TEST_DIAGNOSTICS_DIR when set.",
    )
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        default="default",
        choices=["autodetect", *warp.tests.unittest_suites.SUITE_FACTORIES],
        help="Name of the test suite to run (default is 'default').",
    )
    group_parallel = parser.add_argument_group("parallelization options")
    group_parallel.add_argument(
        "-j",
        "--jobs",
        metavar="COUNT",
        type=int,
        default=0,
        help="The number of test processes (default is 0, all cores)",
    )
    group_parallel.add_argument(
        "-m",
        "--maxjobs",
        metavar="MAXCOUNT",
        type=int,
        default=8,
        help="The maximum number of test processes (default is 8)",
    )
    group_parallel.add_argument(
        "--level",
        choices=["module", "class"],
        default="class",
        help="Set the test parallelism level (default is 'class')",
    )
    group_parallel.add_argument(
        "--isolate-test-processes",
        action="store_true",
        help="Run each test suite in a fresh process (requires Python 3.11 or newer).",
    )
    group_coverage = parser.add_argument_group("coverage options")
    group_coverage.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    group_coverage.add_argument("--coverage-branch", action="store_true", help="Run tests with branch coverage")
    group_coverage.add_argument(
        "--coverage-html",
        metavar="DIR",
        help="Generate coverage HTML report",
        default=os.path.join(START_DIRECTORY, "..", "..", "htmlcov"),
    )
    group_coverage.add_argument("--coverage-xml", metavar="FILE", help="Generate coverage XML report")
    group_coverage.add_argument(
        "--coverage-fail-under", metavar="MIN", type=float, help="Fail if coverage percentage under min"
    )
    group_warp = parser.add_argument_group("NVIDIA Warp options")
    group_warp.add_argument(
        "--no-shared-cache", action="store_true", help="Use a separate kernel cache per test process."
    )
    group_warp.add_argument("--warp-debug", action="store_true", help="Set warp.config.mode to 'debug'")
    return parser


@dataclasses.dataclass
class _AggregatedResults:
    tests_run: int = 0
    errors: list = dataclasses.field(default_factory=list)
    failures: list = dataclasses.field(default_factory=list)
    skipped: int = 0
    expected_failures: int = 0
    unexpected_successes: int = 0
    test_records: list = dataclasses.field(default_factory=list)
    extra_records: list = dataclasses.field(default_factory=list)

    @property
    def pool_error_count(self):
        return len(self.extra_records)

    @property
    def is_success(self):
        return not (self.errors or self.failures or self.unexpected_successes or self.pool_error_count)


def _aggregate_results(results, pool_failure):
    summary = _AggregatedResults()
    for result in results:
        summary.tests_run += result[0]
        summary.errors.extend(result[1])
        summary.failures.extend(result[2])
        summary.skipped += result[3]
        summary.expected_failures += result[4]
        summary.unexpected_successes += result[5]
        summary.test_records.extend(result[6])

    if pool_failure is not None:
        diagnostics_path = pool_failure.snapshot["diagnostics"]["crash_snapshot"]
        summary.extra_records.append(make_pool_failure_test_record(pool_failure.snapshot, diagnostics_path))
        summary.tests_run += 1
    return summary


def _report_results(parser, args, summary, test_duration):
    infos = []
    if summary.failures:
        infos.append(f"failures={len(summary.failures)}")
    if summary.errors or summary.pool_error_count:
        infos.append(f"errors={len(summary.errors) + summary.pool_error_count}")
    if summary.skipped:
        infos.append(f"skipped={summary.skipped}")
    if summary.expected_failures:
        infos.append(f"expected failures={summary.expected_failures}")
    if summary.unexpected_successes:
        infos.append(f"unexpected successes={summary.unexpected_successes}")

    if summary.errors or summary.failures:
        print(file=sys.stderr)
        for error in summary.errors:
            print(error, file=sys.stderr)
        for failure in summary.failures:
            print(failure, file=sys.stderr)
    elif args.verbose > 0:
        print(file=sys.stderr)

    print(unittest.TextTestResult.separator2, file=sys.stderr)
    test_label = "tests" if summary.tests_run > 1 else "test"
    print(f"Ran {summary.tests_run} {test_label} in {test_duration:.3f}s", file=sys.stderr)
    print(file=sys.stderr)
    status = "OK" if summary.is_success else "FAILED"
    print(f"{status}{' (' + ', '.join(infos) + ')' if infos else ''}", file=sys.stderr)

    if (summary.test_records or summary.extra_records) and args.junit_report_xml:
        write_junit_results(
            args.junit_report_xml,
            summary.test_records,
            test_duration,
            extra_records=summary.extra_records,
        )

    if not summary.is_success:
        error_count = (
            len(summary.errors) + len(summary.failures) + summary.unexpected_successes + summary.pool_error_count
        )
        parser.exit(status=min(255, error_count))


def _write_coverage_reports(parser, args, temp_dir):
    if not args.coverage:
        return

    import coverage  # noqa: PLC0415

    cov = coverage.Coverage(config_file=True)
    cov.combine(data_paths=[os.path.join(temp_dir, name) for name in os.listdir(temp_dir)])

    print(file=sys.stderr)
    percent_covered = cov.report(ignore_errors=True, file=sys.stderr)
    print(f"Total coverage is {percent_covered:.2f}%", file=sys.stderr)
    if args.coverage_html:
        cov.html_report(directory=args.coverage_html, ignore_errors=True)
    if args.coverage_xml:
        cov.xml_report(outfile=args.coverage_xml, ignore_errors=True)
    if args.coverage_fail_under and percent_covered < args.coverage_fail_under:
        parser.exit(status=2)


def _main(argv, diagnostics):
    """
    unittest-parallel command-line script main entry point
    """

    parser = _create_argument_parser()
    args = parser.parse_args(args=argv)

    if args.coverage_branch:
        args.coverage = args.coverage_branch

    process_count = max(0, args.jobs)
    if process_count == 0:
        process_count = multiprocessing.cpu_count()
    process_count = min(process_count, args.maxjobs)  # NVIDIA Modification

    run_start_monotonic_ns = time.monotonic_ns()
    run_start_wall_time_ns = time.time_ns()
    diagnostics_root = resolve_diagnostics_root(args.diagnostics_dir)
    diagnostics_run_dir = None
    if diagnostics_root is not None:
        try:
            diagnostics_run_dir = create_diagnostics_run_dir(
                diagnostics_root,
                wall_time_ns=run_start_wall_time_ns,
                parent_pid=os.getpid(),
            )
        except Exception as error:
            warn(f"Failed to create diagnostics run directory: {error}")
    diagnostics.configure(
        args,
        process_count,
        run_start_monotonic_ns,
        run_start_wall_time_ns,
        diagnostics_run_dir,
    )
    diagnostics_run_dir = diagnostics.run_dir

    if args.isolate_test_processes and sys.version_info < (3, 11):
        parser.error("--isolate-test-processes requires Python 3.11 or newer")

    if args.coverage and not COVERAGE_AVAILABLE:
        parser.exit(
            status=2, message="--coverage was used, but coverage was not found. Is it installed?\n"
        )  # NVIDIA Modification

    spawn_context = multiprocessing.get_context(method="spawn")
    event_queue = spawn_context.SimpleQueue()
    tracker = WorkerStateTracker(history_limit=EVENT_HISTORY_LIMIT)
    monitor = WorkerEventMonitor(event_queue, tracker, on_event=_make_worker_event_callback(args))
    diagnostics.attach_monitor(event_queue, tracker, monitor)
    monitor.start()

    import warp as wp  # noqa: PLC0415 NVIDIA Modification

    # Clear the Warp cache (NVIDIA Modification).  Honor WARP_CACHE_PATH
    # before the clear so concurrent worktrees pinned to the same Warp
    # version do not wipe each other's default cache.  Workers key on the
    # same env var.
    warp_cache_base_path = get_warp_cache_base_path()
    if warp_cache_base_path is not None:
        wp.config.kernel_cache_dir = warp_cache_base_path

    wp.clear_lto_cache()
    wp.clear_kernel_cache()
    print(f"Main process cleared Warp kernel cache: {wp.config.kernel_cache_dir}")

    # Create the temporary directory (for coverage files)
    with tempfile.TemporaryDirectory() as temp_dir:
        discover_suite, test_suites = _discover_suites(args, temp_dir)
        # Don't use more processes than test suites
        process_count = max(1, min(len(test_suites), process_count))
        diagnostics.update_process_count(process_count)
        diagnostics_run_dir = diagnostics.run_dir

        # Report test suites and processes
        print(
            f"Running {len(test_suites)} test suites ({discover_suite.countTestCases()} total tests) across {process_count} processes",
            file=sys.stderr,
        )
        if args.verbose > 1:
            print(file=sys.stderr)

        # Run the tests in parallel
        start_time = time.perf_counter()
        report_pool_failure = None
        manager = spawn_context.Manager()
        try:
            parallel_run = run_parallel_suites(
                test_suites,
                process_count,
                manager,
                args,
                temp_dir,
                event_queue,
                tracker,
                monitor,
                diagnostics_run_dir,
                run_start_monotonic_ns,
            )
            diagnostics.suite_classifications = parallel_run.suite_classifications
            if parallel_run.diagnostics_degraded:
                diagnostics.degrade_event_monitor()
            results = [parallel_run.results_by_index[index] for index in sorted(parallel_run.results_by_index)]

            if parallel_run.pool_failure is not None:
                diagnostics.require_evidence()
                report_pool_failure = parallel_run.pool_failure
                unresolved = len(test_suites) - len(parallel_run.results_by_index)
                print(
                    f"Warning: Parallel execution failed after confirming "
                    f"{len(parallel_run.results_by_index)}/{len(test_suites)} suites; "
                    f"{unresolved} remain unresolved.",
                    file=sys.stderr,
                )
        finally:
            manager.shutdown()

        diagnostics.best_effort("stop diagnostics monitor", diagnostics.stop_monitor)

        stop_time = time.perf_counter()
        test_duration = stop_time - start_time

        summary = _aggregate_results(results, report_pool_failure)
        _report_results(parser, args, summary, test_duration)
        _write_coverage_reports(parser, args, temp_dir)

    diagnostics.mark_clean()


def _discover_suites(args, temp_dir):
    with coverage_context(args, temp_dir):
        test_loader = unittest.TestLoader()
        if args.testNamePatterns:
            test_loader.testNamePatterns = args.testNamePatterns

        auto_discover_suite = warp.tests.unittest_suites.auto_discover_suite(test_loader, args.pattern)
        if args.suite == "autodetect":
            discover_suite = auto_discover_suite
        else:
            discover_suite = warp.tests.unittest_suites.compare_unittest_suites(
                test_loader,
                args.suite,
                auto_discover_suite,
            )

    if args.level == "class":
        test_suites = list(_iter_class_suites(discover_suite))
    else:
        test_suites = list(_iter_module_suites(discover_suite))
    return discover_suite, test_suites


def _iter_module_suites(test_suite):
    for module_suite in test_suite:
        if module_suite.countTestCases():
            yield module_suite


# Iterate class-level test suites - test suites that contains test cases
def _iter_class_suites(test_suite):
    has_cases = any(isinstance(suite, unittest.TestCase) for suite in test_suite)
    if has_cases:
        yield test_suite
    else:
        for suite in test_suite:
            yield from _iter_class_suites(suite)


if __name__ == "__main__":  # pragma: no cover
    main()
