# Licensed under the MIT License
# https://github.com/craigahobbs/unittest-parallel/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from contextlib import contextmanager
from io import StringIO

import warp.tests.unittest_suites  # NVIDIA Modification
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
START_DIRECTORY = os.path.dirname(__file__)  # The directory to start test discovery


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
        python -m warp.tests -s kit
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
        choices=["autodetect", "default", "kit"],
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
        choices=["module", "class", "test"],
        default="class",
        help="Set the test parallelism level (default is 'class')",
    )
    group_parallel.add_argument(
        "--disable-process-pooling",
        action="store_true",
        default=False,
        help="Do not reuse processes used to run test suites",
    )
    group_parallel.add_argument(
        "--disable-concurrent-futures",
        action="store_true",
        default=False,
        help="Use multiprocessing instead of concurrent.futures.",
    )  # NVIDIA Modification
    group_parallel.add_argument(
        "--serial-fallback",
        action="store_true",
        default=False,
        help="Run in a single-process (no spawning) mode without multiprocessing or concurrent.futures.",
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
    args = parser.parse_args(args=argv)

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

    import warp as wp  # NVIDIA Modification

    # Clear the Warp cache (NVIDIA Modification)
    wp.clear_lto_cache()
    wp.clear_kernel_cache()
    print("Cleared Warp kernel cache")

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
        if args.level == "test":
            test_suites = list(_iter_test_cases(discover_suite))
        elif args.level == "class":
            test_suites = list(_iter_class_suites(discover_suite))
        else:  # args.level == 'module'
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

            if args.disable_concurrent_futures:
                multiprocessing_context = multiprocessing.get_context(method="spawn")
                maxtasksperchild = 1 if args.disable_process_pooling else None
                with multiprocessing_context.Pool(
                    process_count,
                    maxtasksperchild=maxtasksperchild,
                    initializer=initialize_test_process,
                    initargs=(manager.Lock(), shared_index, args, temp_dir),
                ) as pool:
                    test_manager = ParallelTestManager(manager, args, temp_dir)
                    results = pool.map(test_manager.run_tests, test_suites)
            else:
                # NVIDIA Modification added concurrent.futures
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=process_count,
                    mp_context=multiprocessing.get_context(method="spawn"),
                    initializer=initialize_test_process,
                    initargs=(manager.Lock(), shared_index, args, temp_dir),
                ) as executor:
                    test_manager = ParallelTestManager(manager, args, temp_dir)
                    results = list(executor.map(test_manager.run_tests, test_suites, timeout=2400))
        else:
            # This entire path is an NVIDIA Modification

            # Report test suites and processes
            print(f"Running {discover_suite.countTestCases()} total tests (serial fallback)", file=sys.stderr)
            if args.verbose > 1:
                print(file=sys.stderr)

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

        # Return an error status on failure
        if not is_success:
            parser.exit(status=len(errors) + len(failures) + unexpected_successes)

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


# Iterate test cases (methods)
def _iter_test_cases(test_suite):
    if isinstance(test_suite, unittest.TestCase):
        yield test_suite
    else:
        for suite in test_suite:
            yield from _iter_test_cases(suite)


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
        import warp.tests.unittest_utils

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
    If the environment variable `WARP_CACHE_ROOT` is detected, the cache will be placed in the provided path.

    It also ensures that Warp is initialized prior to running any tests.
    """

    with lock:
        shared_index.value += 1
        worker_index = shared_index.value

    with _coverage(args, temp_dir):
        import warp as wp

        if args.no_shared_cache:
            from warp.thirdparty import appdirs

            if "WARP_CACHE_ROOT" in os.environ:
                cache_root_dir = os.path.join(os.getenv("WARP_CACHE_ROOT"), f"{wp.config.version}-{worker_index:03d}")
            else:
                cache_root_dir = appdirs.user_cache_dir(
                    appname="warp", appauthor="NVIDIA", version=f"{wp.config.version}-{worker_index:03d}"
                )

            wp.config.kernel_cache_dir = cache_root_dir

            wp.clear_lto_cache()
            wp.clear_kernel_cache()
        elif "WARP_CACHE_ROOT" in os.environ:
            # Using a shared cache for all test processes
            wp.config.kernel_cache_dir = os.path.join(os.getenv("WARP_CACHE_ROOT"), wp.config.version)


if __name__ == "__main__":  # pragma: no cover
    main()
