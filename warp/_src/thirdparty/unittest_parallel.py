# Licensed under the MIT License
# https://github.com/craigahobbs/unittest-parallel/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
unittest-parallel command-line script main module
"""

import argparse
import concurrent.futures  # NVIDIA Modification
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from io import StringIO

import warp.tests.unittest_suites  # NVIDIA Modification
import warp.tests.unittest_utils
from warp._src.thirdparty import appdirs
from warp.tests.unittest_utils import (  # NVIDIA modification
    ParallelJunitTestResult,
    write_junit_results,
)

try:
    import coverage

    COVERAGE_AVAILABLE = True  # NVIDIA Modification
except ImportError:
    COVERAGE_AVAILABLE = False  # NVIDIA Modification


# The following variables are NVIDIA Modifications
START_DIRECTORY = os.path.join(os.path.dirname(__file__), "..")  # The directory to start test discovery
_SUITE_TIMEOUT = (
    3600  # Timeout in seconds: total wall-clock limit for parallel execution, per-suite limit during isolated fallback
)
_WARP_CACHE_PATH_ENV = "WARP_CACHE_PATH"


def _get_warp_cache_base_path():
    cache_path = os.environ.get(_WARP_CACHE_PATH_ENV)
    if cache_path is None:
        return None
    cache_path = cache_path.strip()
    return cache_path or None


def _kill_process_pool(executor):
    """Kill all executor workers without waiting for running futures.

    Python 3.14 added :meth:`~concurrent.futures.ProcessPoolExecutor.kill_workers`.
    Mirror its implementation for older supported Python versions.
    """
    kill_workers = getattr(executor, "kill_workers", None)
    if kill_workers is not None:
        kill_workers()
        return

    processes = executor._processes.copy() if executor._processes else {}
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes.values():
        try:
            if not process.is_alive():
                continue
        except ValueError:
            # The process has already exited and closed.
            continue

        try:
            process.kill()
        except ProcessLookupError:
            # The process exited between the liveness check and kill.
            continue


def main(argv=None):
    """
    unittest-parallel command-line script main entry point
    """

    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="unittest-parallel",
        # NVIDIA Modifications follow:
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Example usage:
        python -m warp.tests -s autodetect -p 'test_a*.py'
        python -m warp.tests -s debug
        python -m warp.tests -k 'mgpu' -k 'cuda'
        """,
    )
    # parser.add_argument("-v", "--verbose", action="store_const", const=2, default=1, help="Verbose output")
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
        help="'autodetect' suite only: Pattern to match tests ('test*.py' default)",  # NVIDIA Modification
    )
    parser.add_argument(
        "-t",
        "--top-level-directory",
        metavar="TOP",
        help="Top level directory of project (defaults to start directory)",
    )
    parser.add_argument(
        "--junit-report-xml", metavar="FILE", help="Generate JUnit report format XML file"
    )  # NVIDIA Modification
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        default="default",
        # "autodetect" is a special mode; the named suites are derived from the
        # registry so a choice cannot exist without a factory behind it.
        choices=["autodetect", *warp.tests.unittest_suites.SUITE_FACTORIES],
        help="Name of the test suite to run (default is 'default').",
    )  # NVIDIA Modification
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
    )  # NVIDIA Modification
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
    group_parallel.add_argument(
        "--serial-fallback",
        action="store_true",
        default=False,
        help="Run in a single-process (no spawning) mode without multiprocessing or concurrent.futures.",
    )  # NVIDIA Modification
    group_parallel.add_argument(
        "--fallback-on-crash",
        action="store_true",
        default=False,
        help="If parallel execution fails, re-run each test suite one-at-a-time in "
        "isolated single-process pools. Slow; mainly to salvage partial results. "
        "Off by default: a crash fails loudly instead.",
    )  # NVIDIA Modification
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
    group_warp = parser.add_argument_group("NVIDIA Warp options")  # NVIDIA Modification
    group_warp.add_argument(
        "--no-shared-cache", action="store_true", help="Use a separate kernel cache per test process."
    )
    group_warp.add_argument("--warp-debug", action="store_true", help="Set warp.config.mode to 'debug'")
    args = parser.parse_args(args=argv)

    if args.isolate_test_processes and sys.version_info < (3, 11):
        parser.error("--isolate-test-processes requires Python 3.11 or newer")
    if args.isolate_test_processes and args.serial_fallback:
        parser.error("--isolate-test-processes cannot be used with --serial-fallback")

    if args.coverage_branch:
        args.coverage = args.coverage_branch

    if args.coverage and not COVERAGE_AVAILABLE:
        parser.exit(
            status=2, message="--coverage was used, but coverage was not found. Is it installed?\n"
        )  # NVIDIA Modification

    process_count = max(0, args.jobs)
    if process_count == 0:
        process_count = multiprocessing.cpu_count()
    process_count = min(process_count, args.maxjobs)  # NVIDIA Modification

    import warp as wp  # noqa: PLC0415 NVIDIA Modification

    # Clear the Warp cache (NVIDIA Modification).  Honor WARP_CACHE_PATH
    # before the clear so concurrent worktrees pinned to the same Warp
    # version do not wipe each other's default cache.  Workers key on the
    # same env var.
    warp_cache_base_path = _get_warp_cache_base_path()
    if warp_cache_base_path is not None:
        wp.config.kernel_cache_dir = warp_cache_base_path

    wp.clear_lto_cache()
    wp.clear_kernel_cache()
    print(f"Main process cleared Warp kernel cache: {wp.config.kernel_cache_dir}")

    # Create the temporary directory (for coverage files)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Discover tests
        with _coverage(args, temp_dir):
            test_loader = unittest.TestLoader()
            if args.testNamePatterns:
                test_loader.testNamePatterns = args.testNamePatterns

            auto_discover_suite = warp.tests.unittest_suites.auto_discover_suite(
                test_loader, args.pattern
            )  # NVIDIA Modification

            # NVIDIA Modification
            if args.suite != "autodetect":
                # Print notices for test classes missing from the suite when compared to auto-discovered tests
                discover_suite = warp.tests.unittest_suites.compare_unittest_suites(
                    test_loader, args.suite, auto_discover_suite
                )
            else:
                discover_suite = auto_discover_suite

        # Get the parallelizable test suites
        if args.level == "class":
            test_suites = list(_iter_class_suites(discover_suite))
        else:  # args.level == "module"
            test_suites = list(_iter_module_suites(discover_suite))

        # Don't use more processes than test suites
        process_count = max(1, min(len(test_suites), process_count))

        if not args.serial_fallback:
            # Report test suites and processes
            print(
                f"Running {len(test_suites)} test suites ({discover_suite.countTestCases()} total tests) across {process_count} processes",
                file=sys.stderr,
            )
            if args.verbose > 1:
                print(file=sys.stderr)

            # Create the shared index object used in Warp caches (NVIDIA Modification)
            manager = multiprocessing.Manager()
            shared_index = manager.Value("i", -1)

            # Run the tests in parallel
            start_time = time.perf_counter()

            # NVIDIA Modification: added concurrent.futures with crash handling and per-suite isolated fallback
            results = []
            parallel_failed = False
            parallel_fail_reason = "unknown"

            try:
                executor_options = {}
                if args.isolate_test_processes:
                    executor_options["max_tasks_per_child"] = 1

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=process_count,
                    mp_context=multiprocessing.get_context(method="spawn"),
                    initializer=initialize_test_process,
                    initargs=(manager.Lock(), shared_index, args, temp_dir),
                    **executor_options,
                ) as executor:
                    test_manager = ParallelTestManager(manager, args, temp_dir)
                    # Iterate results explicitly so we can report which suite timed out
                    try:
                        for result in executor.map(test_manager.run_tests, test_suites, timeout=_SUITE_TIMEOUT):
                            results.append(result)
                    except concurrent.futures.TimeoutError:
                        _kill_process_pool(executor)
                        raise

            except concurrent.futures.TimeoutError:
                pending_index = len(results)
                total = len(test_suites)
                suite_name = _get_suite_name(test_suites[pending_index]) if pending_index < total else "unknown"
                print(
                    f"Warning: Parallel execution timed out (total timeout={_SUITE_TIMEOUT}s). "
                    f"Next pending result was suite "
                    f"{pending_index + 1}/{total} ({suite_name}), "
                    f"but a different suite may be the actual blocker.",
                    file=sys.stderr,
                )
                parallel_failed = True
                parallel_fail_reason = "timed out"
            except BrokenProcessPool:
                print("Warning: Process pool broken during parallel execution.", file=sys.stderr)
                parallel_failed = True
                parallel_fail_reason = "process pool broken"
            except Exception as e:
                # Handle other pool-level exceptions
                print(f"Warning: Process pool error: {e}.", file=sys.stderr)
                parallel_failed = True
                parallel_fail_reason = str(e)

            # When parallel execution fails, either fail loudly with a partial
            # report (default) or re-run each suite in an isolated single-process
            # pool to salvage results (opt-in via --fallback-on-crash).
            if parallel_failed and not args.fallback_on_crash:
                # Keep the results collected before the failure, mark the remaining
                # suites as crashed without re-running them, and fall through to the
                # normal report (which exits non-zero because of the crash errors).
                completed = len(results)
                pending = test_suites[completed:]
                print(
                    f"Warning: Parallel execution failed ({parallel_fail_reason}). "
                    f"{completed}/{len(test_suites)} suites completed before the failure; "
                    f"marking the remaining {len(pending)} as crashed (some may have passed "
                    f"but cannot be confirmed after the failure). Pass --fallback-on-crash "
                    f"to re-run all suites in isolated single-process mode.",
                    file=sys.stderr,
                )
                for suite in pending:
                    results.append(
                        create_crash_result(suite, reason=f"Parallel execution failed: {parallel_fail_reason}")
                    )
            elif parallel_failed:
                print("Running all tests in isolated single-process mode...", file=sys.stderr)
                # Run all test suites in isolated single-process pools
                results = []
                fallback_manager = ParallelTestManager(manager, args, temp_dir)
                for i, suite in enumerate(test_suites):
                    try:
                        # Create a new single-process pool for each test suite
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=1,
                            mp_context=multiprocessing.get_context(method="spawn"),
                            initializer=initialize_test_process,
                            initargs=(manager.Lock(), shared_index, args, temp_dir),
                        ) as executor:
                            future = executor.submit(fallback_manager.run_tests, suite)
                            try:
                                result = future.result(timeout=_SUITE_TIMEOUT)
                                results.append(result)
                            except concurrent.futures.TimeoutError:
                                _kill_process_pool(executor)
                                suite_name = _get_suite_name(suite)
                                print(
                                    f"Warning: Isolated test suite {i + 1}/{len(test_suites)} ({suite_name}) timed out (timeout={_SUITE_TIMEOUT}s). Marking tests as crashed.",
                                    file=sys.stderr,
                                )
                                crash_result = create_crash_result(
                                    suite, reason=f"Process timed out (timeout={_SUITE_TIMEOUT}s)"
                                )
                                results.append(crash_result)
                            except BrokenProcessPool:
                                print(
                                    f"Warning: Process crashed or was terminated unexpectedly in isolated execution for test suite {i + 1}/{len(test_suites)}. Marking tests as crashed.",
                                    file=sys.stderr,
                                )
                                crash_result = create_crash_result(suite)
                                results.append(crash_result)
                            except Exception as e:
                                print(
                                    f"Warning: Error in isolated test suite {i + 1}/{len(test_suites)}: {e}. Marking tests as crashed.",
                                    file=sys.stderr,
                                )
                                error_result = create_crash_result(suite)
                                results.append(error_result)
                    except Exception as e:
                        print(
                            f"Warning: Failed to create isolated process for test suite {i + 1}/{len(test_suites)}: {e}. Marking tests as crashed.",
                            file=sys.stderr,
                        )
                        error_result = create_crash_result(suite)
                        results.append(error_result)
        else:
            # This entire path is an NVIDIA Modification

            # Report test suites and processes
            print(f"Running {discover_suite.countTestCases()} total tests (serial fallback)", file=sys.stderr)
            if args.verbose > 1:
                print(file=sys.stderr)

            if args.warp_debug:
                wp.config.mode = "debug"

            # Run the tests in serial
            start_time = time.perf_counter()

            with multiprocessing.Manager() as manager:
                test_manager = ParallelTestManager(manager, args, temp_dir)
                results = [test_manager.run_tests(discover_suite)]

        stop_time = time.perf_counter()
        test_duration = stop_time - start_time

        # Aggregate parallel test run results
        tests_run = 0
        errors = []
        failures = []
        skipped = 0
        expected_failures = 0
        unexpected_successes = 0
        test_records = []  # NVIDIA Modification
        for result in results:
            tests_run += result[0]
            errors.extend(result[1])
            failures.extend(result[2])
            skipped += result[3]
            expected_failures += result[4]
            unexpected_successes += result[5]
            test_records += result[6]  # NVIDIA Modification
        is_success = not (errors or failures or unexpected_successes)

        # Compute test info
        infos = []
        if failures:
            infos.append(f"failures={len(failures)}")
        if errors:
            infos.append(f"errors={len(errors)}")
        if skipped:
            infos.append(f"skipped={skipped}")
        if expected_failures:
            infos.append(f"expected failures={expected_failures}")
        if unexpected_successes:
            infos.append(f"unexpected successes={unexpected_successes}")

        # Report test errors
        if errors or failures:
            print(file=sys.stderr)
            for error in errors:
                print(error, file=sys.stderr)
            for failure in failures:
                print(failure, file=sys.stderr)
        elif args.verbose > 0:
            print(file=sys.stderr)

        # Test report
        print(unittest.TextTestResult.separator2, file=sys.stderr)
        print(f"Ran {tests_run} {'tests' if tests_run > 1 else 'test'} in {test_duration:.3f}s", file=sys.stderr)
        print(file=sys.stderr)
        print(f"{'OK' if is_success else 'FAILED'}{' (' + ', '.join(infos) + ')' if infos else ''}", file=sys.stderr)

        if test_records and args.junit_report_xml:
            # NVIDIA modification to report results in Junit XML format
            write_junit_results(
                args.junit_report_xml,
                test_records,
                tests_run,
                len(failures) + unexpected_successes,
                len(errors),
                skipped,
                test_duration,
            )

        # Return an error status on failure. Clamp to 255 because process exit
        # codes wrap modulo 256; a large crash error count that is a multiple of
        # 256 would otherwise exit 0 and report a failed run as success. The
        # `not is_success` guard guarantees the count is at least 1.
        if not is_success:
            parser.exit(status=min(255, len(errors) + len(failures) + unexpected_successes))

        # Coverage?
        if args.coverage:
            # Combine the coverage files
            cov_options = {}
            cov_options["config_file"] = True  # Grab configuration from pyproject.toml (must install coverage[toml])
            cov = coverage.Coverage(**cov_options)
            cov.combine(data_paths=[os.path.join(temp_dir, x) for x in os.listdir(temp_dir)])

            # Coverage report
            print(file=sys.stderr)
            percent_covered = cov.report(ignore_errors=True, file=sys.stderr)
            print(f"Total coverage is {percent_covered:.2f}%", file=sys.stderr)

            # HTML coverage report
            if args.coverage_html:
                cov.html_report(directory=args.coverage_html, ignore_errors=True)

            # XML coverage report
            if args.coverage_xml:
                cov.xml_report(outfile=args.coverage_xml, ignore_errors=True)

            # Fail under
            if args.coverage_fail_under and percent_covered < args.coverage_fail_under:
                parser.exit(status=2)


def _convert_select_pattern(pattern):
    if "*" not in pattern:
        return f"*{pattern}*"
    return pattern


@contextmanager
def _coverage(args, temp_dir):
    # Running tests with coverage?
    if args.coverage:
        # Generate a random coverage data file name - file is deleted along with containing directory
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as coverage_file:
            pass

        # Create the coverage object
        cov_options = {
            "branch": args.coverage_branch,
            "data_file": coverage_file.name,
            # NVIDIA Modification removed unneeded options
        }
        cov_options["config_file"] = True  # Grab configuration from pyproject.toml (must install coverage[toml])
        cov = coverage.Coverage(**cov_options)
        try:
            # Start measuring code coverage
            cov.start()

            # Yield for unit test running
            yield cov
        finally:
            # Stop measuring code coverage
            cov.stop()

            # Save the collected coverage data to the data file
            cov.save()
    else:
        # Not running tests with coverage - yield for unit test running
        yield None


# Iterate module-level test suites - all top-level test suites returned from TestLoader.discover
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


def _get_suite_name(test_suite):
    """Return a human-readable name for a test suite (e.g. 'TestTileMatmul')."""
    first_test = next(_iter_test_cases(test_suite), None)
    return type(first_test).__name__ if first_test is not None else "unknown"


# Iterate test cases (methods)
def _iter_test_cases(test_suite):
    if isinstance(test_suite, unittest.TestCase):
        yield test_suite
    else:
        for suite in test_suite:
            yield from _iter_test_cases(suite)


def create_crash_result(test_suite, reason="Process crashed or was terminated unexpectedly"):
    """Create a result indicating the process failed while running this test suite.

    This entire function is an NVIDIA modification.
    """
    test_count = test_suite.countTestCases()
    crash_errors = []
    crash_test_records = []

    # Create error entries for each test in the suite
    # Note: We don't know which specific test caused the failure, just that the process failed
    for test in _iter_test_cases(test_suite):
        error_msg = f"{reason} while running this test suite (unknown which test caused the failure): {test}"
        crash_errors.append(
            "\n".join(
                [
                    unittest.TextTestResult.separator1,
                    str(test),
                    unittest.TextTestResult.separator2,
                    error_msg,
                ]
            )
        )
        crash_test_records.append((test.__class__.__name__, test._testMethodName, 0.0, "ERROR", error_msg, error_msg))

    # Return the same format as run_tests: (test_count, errors, failures, skipped, expected_failures, unexpected_successes, test_records)
    return (test_count, crash_errors, [], 0, 0, 0, crash_test_records)


class ParallelTestManager:
    def __init__(self, manager, args, temp_dir):
        self.args = args
        self.temp_dir = temp_dir
        self.failfast = manager.Event()

    def run_tests(self, test_suite):
        # Fail fast?
        if self.failfast.is_set():
            return [0, [], [], 0, 0, 0, []]  # NVIDIA Modification

        # NVIDIA Modification for GitLab
        warp.tests.unittest_utils.coverage_enabled = self.args.coverage
        warp.tests.unittest_utils.coverage_temp_dir = self.temp_dir
        warp.tests.unittest_utils.coverage_branch = self.args.coverage_branch

        if self.args.junit_report_xml:
            resultclass = ParallelJunitTestResult
        else:
            resultclass = ParallelTextTestResult

        # Run unit tests
        with _coverage(self.args, self.temp_dir):
            runner = unittest.TextTestRunner(
                stream=StringIO(),
                resultclass=resultclass,  # NVIDIA Modification
                verbosity=self.args.verbose,
                failfast=self.args.failfast,
                buffer=self.args.buffer,
            )
            result = runner.run(test_suite)

            # Set failfast, if necessary
            if result.shouldStop:
                self.failfast.set()

            # Return (test_count, errors, failures, skipped_count, expected_failure_count, unexpected_success_count)
            return (
                result.testsRun,
                [self._format_error(result, error) for error in result.errors],
                [self._format_error(result, failure) for failure in result.failures],
                len(result.skipped),
                len(result.expectedFailures),
                len(result.unexpectedSuccesses),
                result.test_record,  # NVIDIA modification
            )

    @staticmethod
    def _format_error(result, error):
        return "\n".join(
            [
                unittest.TextTestResult.separator1,
                result.getDescription(error[0]),
                unittest.TextTestResult.separator2,
                error[1],
            ]
        )


class ParallelTextTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        stream = type(stream)(sys.stderr)
        super().__init__(stream, descriptions, verbosity)
        self.test_record = []  # NVIDIA modification

    def startTest(self, test):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ...")
            self.stream.flush()
        super(unittest.TextTestResult, self).startTest(test)

    def stopTest(self, test):
        super().stopTest(test)
        # Force garbage collection of CPU-side allocations to reduce peak
        # host RSS in parallel test runs.
        import gc  # noqa: PLC0415

        gc.collect()

    def _add_helper(self, test, dots_message, show_all_message):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ... {show_all_message}")
        elif self.dots:
            self.stream.write(dots_message)
        self.stream.flush()

    def addSuccess(self, test):
        super(unittest.TextTestResult, self).addSuccess(test)
        self._add_helper(test, ".", "ok")

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self._add_helper(test, "E", "ERROR")

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self._add_helper(test, "F", "FAIL")

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        self._add_helper(test, "s", f"skipped {reason!r}")

    def addExpectedFailure(self, test, err):
        super(unittest.TextTestResult, self).addExpectedFailure(test, err)
        self._add_helper(test, "x", "expected failure")

    def addUnexpectedSuccess(self, test):
        super(unittest.TextTestResult, self).addUnexpectedSuccess(test)
        self._add_helper(test, "u", "unexpected success")

    def printErrors(self):
        pass


def initialize_test_process(lock, shared_index, args, temp_dir):
    """Necessary operations to be executed at the start of every test process.

    Currently this function can be used to set a separate Warp cache. (NVIDIA modification)
    If the environment variable `WARP_CACHE_PATH` is detected, the cache will be placed in the provided path.

    It also ensures that Warp is initialized prior to running any tests.
    """

    with lock:
        shared_index.value += 1
        worker_index = shared_index.value

    with _coverage(args, temp_dir):
        import warp as wp  # noqa: PLC0415

        if args.warp_debug:
            wp.config.mode = "debug"

        warp_cache_base_path = _get_warp_cache_base_path()

        # init_kernel_cache() appends warp.config.version, so we set
        # kernel_cache_dir to a base path and let Warp add the version segment.
        if args.no_shared_cache:
            if warp_cache_base_path is not None:
                cache_root_dir = os.path.join(warp_cache_base_path, f"worker-{worker_index:03d}")
            else:
                cache_root_dir = appdirs.user_cache_dir(
                    appname="warp", appauthor="NVIDIA", version=f"worker-{worker_index:03d}"
                )

            wp.config.kernel_cache_dir = cache_root_dir

            wp.clear_lto_cache()
            wp.clear_kernel_cache()
        elif warp_cache_base_path is not None:
            # Using a shared cache for all test processes
            wp.config.kernel_cache_dir = warp_cache_base_path


if __name__ == "__main__":  # pragma: no cover
    main()
