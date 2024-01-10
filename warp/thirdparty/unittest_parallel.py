# Licensed under the MIT License
# https://github.com/craigahobbs/unittest-parallel/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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

try:
    import coverage

    COVERAGE_AVAILABLE = True  # NVIDIA Modification
except ImportError:
    COVERAGE_AVAILABLE = False  # NVIDIA Modification


# The following variables are NVIDIA Modifications
RUNNING_IN_TEAMCITY = os.environ.get("TEAMCITY_VERSION") is not None
TEST_SUITE_NAME = "WarpTests"
START_DIRECTORY = os.path.dirname(__file__)  # The directory to start test discovery


def main(argv=None):
    """
    unittest-parallel command-line script main entry point
    """

    # Command line arguments
    parser = argparse.ArgumentParser(prog="unittest-parallel")
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
        "-p", "--pattern", metavar="PATTERN", default="test*.py", help="Pattern to match tests ('test*.py' default)"
    )
    parser.add_argument(
        "-t",
        "--top-level-directory",
        metavar="TOP",
        help="Top level directory of project (defaults to start directory)",
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

    # Create the temporary directory (for coverage files)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Discover tests
        with _coverage(args, temp_dir):
            test_loader = unittest.TestLoader()
            if args.testNamePatterns:
                test_loader.testNamePatterns = args.testNamePatterns
            discover_suite = warp.tests.unittest_suites.auto_discover_suite(
                test_loader, args.pattern
            )  # NVIDIA Modification
            # discover_suite = warp.tests.unittest_suites.explicit_suite()

        # Get the parallelizable test suites
        if args.level == "test":
            test_suites = list(_iter_test_cases(discover_suite))
        elif args.level == "class":
            test_suites = list(_iter_class_suites(discover_suite))
        else:  # args.level == 'module'
            test_suites = list(_iter_module_suites(discover_suite))

        # Don't use more processes than test suites
        process_count = max(1, min(len(test_suites), process_count))

        if RUNNING_IN_TEAMCITY:
            print(f"##teamcity[testSuiteStarted name='{TEST_SUITE_NAME}']")  # NVIDIA Modification for TC

        if not args.serial_fallback:
            # Report test suites and processes
            print(
                f"Running {len(test_suites)} test suites ({discover_suite.countTestCases()} total tests) across {process_count} processes",
                file=sys.stderr,
            )
            if args.verbose > 1:
                print(file=sys.stderr)

            # Run the tests in parallel
            start_time = time.perf_counter()

            if args.disable_concurrent_futures:
                multiprocessing_context = multiprocessing.get_context(method="spawn")
                maxtasksperchild = 1 if args.disable_process_pooling else None
                with multiprocessing_context.Pool(
                    process_count,
                    maxtasksperchild=maxtasksperchild,
                    initializer=set_worker_cache,
                    initargs=(args, temp_dir),
                ) as pool, multiprocessing.Manager() as manager:
                    test_manager = ParallelTestManager(manager, args, temp_dir)
                    results = pool.map(test_manager.run_tests, test_suites)
            else:
                # NVIDIA Modification added concurrent.futures
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=process_count,
                    mp_context=multiprocessing.get_context(method="spawn"),
                    initializer=set_worker_cache,
                    initargs=(args, temp_dir),
                ) as executor, multiprocessing.Manager() as manager:
                    test_manager = ParallelTestManager(manager, args, temp_dir)
                    results = list(executor.map(test_manager.run_tests, test_suites, timeout=3600))
        else:
            # This entire path is an NVIDIA Modification

            # Report test suites and processes
            print(f"Running {discover_suite.countTestCases()} total tests (serial fallback)", file=sys.stderr)
            if args.verbose > 1:
                print(file=sys.stderr)

            import warp as wp

            # force rebuild of all kernels
            wp.build.clear_kernel_cache()
            print("Cleared Warp kernel cache")

            # Run the tests in serial
            start_time = time.perf_counter()

            with multiprocessing.Manager() as manager:
                test_manager = ParallelTestManager(manager, args, temp_dir)
                results = [test_manager.run_tests(discover_suite)]

        stop_time = time.perf_counter()
        test_duration = stop_time - start_time

        if RUNNING_IN_TEAMCITY:
            print(f"##teamcity[testSuiteFinished name='{TEST_SUITE_NAME}']")  # NVIDIA Modification for TC

        # Aggregate parallel test run results
        tests_run = 0
        errors = []
        failures = []
        skipped = 0
        expected_failures = 0
        unexpected_successes = 0
        for result in results:
            tests_run += result[0]
            errors.extend(result[1])
            failures.extend(result[2])
            skipped += result[3]
            expected_failures += result[4]
            unexpected_successes += result[5]
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
        print(f'Ran {tests_run} {"tests" if tests_run > 1 else "test"} in {test_duration:.3f}s', file=sys.stderr)
        print(file=sys.stderr)
        print(f'{"OK" if is_success else "FAILED"}{" (" + ", ".join(infos) + ")" if infos else ""}', file=sys.stderr)

        # Return an error status on failure
        if not is_success:
            if RUNNING_IN_TEAMCITY:
                print("##teamcity[buildStatus status='FAILURE']")  # NVIDIA Modification for TC
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
            return [0, [], [], 0, 0, 0]

        # Run unit tests
        with _coverage(self.args, self.temp_dir):
            runner = unittest.TextTestRunner(
                stream=StringIO(),
                resultclass=ParallelTeamCityTestResult
                if RUNNING_IN_TEAMCITY
                else ParallelTextTestResult,  # NVIDIA Modification for TC
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


# NVIDIA Modifications from here until end of file


def set_worker_cache(args, temp_dir):
    """This function is run at the start of ever new process. It changes the Warp cache to avoid conflicts."""

    with _coverage(args, temp_dir):
        import warp as wp
        from warp.thirdparty import appdirs

        pid = os.getpid()
        cache_root_dir = appdirs.user_cache_dir(
            appname="warp", appauthor="NVIDIA Corporation", version=f"{wp.config.version}-{pid}"
        )

        wp.config.kernel_cache_dir = cache_root_dir

        wp.build.clear_kernel_cache()


def _tc_escape(s):
    s = s.replace("|", "||")
    s = s.replace("\n", "|n")
    s = s.replace("\r", "|r")
    s = s.replace("'", "|'")
    s = s.replace("[", "|[")
    s = s.replace("]", "|]")
    return s


class ParallelTeamCityTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        stream = type(stream)(sys.stderr)
        super().__init__(stream, descriptions, verbosity)

    def startTest(self, test):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ...")
            self.stream.flush()
        self.start_time = time.perf_counter_ns()
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
        self.reportSuccess(test)

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self._add_helper(test, "E", "ERROR")
        self.reportFailure(test, err)

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self._add_helper(test, "F", "FAIL")
        self.reportFailure(test, err)

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        self._add_helper(test, "s", f"skipped {reason!r}")
        self.reportIgnored(test, reason)

    def addExpectedFailure(self, test, err):
        super(unittest.TextTestResult, self).addExpectedFailure(test, err)
        self._add_helper(test, "x", "expected failure")
        self.reportSuccess(test)

    def addUnexpectedSuccess(self, test):
        super(unittest.TextTestResult, self).addUnexpectedSuccess(test)
        self._add_helper(test, "u", "unexpected success")
        self.reportFailure(test, "unexpected success")

    def addSubTest(self, test, subtest, err):
        super(unittest.TextTestResult, self).addSubTest(test, subtest, err)
        if err is not None:
            self._add_helper(test, "E", "ERROR")
            self.reportSubTestFailure(test, err)

    def printErrors(self):
        pass

    def reportIgnored(self, test, reason):
        test_id = test.id()
        reason_str = str(reason)
        print(reason_str)
        self.stream.writeln(f"##teamcity[testIgnored name='{test_id}' message='{_tc_escape(str(reason))}']")
        self.stream.flush()

    def reportSuccess(self, test):
        duration = round((time.perf_counter_ns() - self.start_time) / 1e6)  # [ms]
        test_id = test.id()
        self.stream.writeln(f"##teamcity[testStarted name='{test_id}']")
        self.stream.writeln(f"##teamcity[testFinished name='{test_id}' duration='{duration}']")
        self.stream.flush()

    def reportFailure(self, test, err):
        test_id = test.id()
        self.stream.writeln(f"##teamcity[testStarted name='{test_id}']")
        self.stream.writeln(
            f"##teamcity[testFailed name='{test_id}' message='{_tc_escape(str(err[1]))}' details='{_tc_escape(str(err[2]))}']"
        )
        self.stream.writeln(f"##teamcity[testFinished name='{test_id}']")
        self.stream.flush()

    def reportSubTestFailure(self, test, err):
        test_id = test.id()
        self.stream.writeln(f"##teamcity[testStarted name='{test_id}']")
        self.stream.writeln(
            f"##teamcity[testFailed name='{test_id}' message='{_tc_escape(str(err[1]))}' details='{_tc_escape(self._exc_info_to_string(err, test))}']"
        )
        self.stream.writeln(f"##teamcity[testFinished name='{test_id}']")
        self.stream.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
