# Licensed under the MIT License
# https://github.com/craigahobbs/unittest-parallel/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side unittest execution for Warp's internal test runner."""

import os
import unittest
from io import StringIO

import warp.tests.unittest_utils
from warp._src.test_runner.common import coverage_context, get_warp_cache_base_path, suite_name_for
from warp._src.test_runner.events import (
    EventKind,
    configure_worker_diagnostics,
    emit_worker_event,
    emit_worker_event_with_gil,
)
from warp._src.thirdparty import appdirs
from warp.tests.unittest_utils import ParallelJunitTestResult


class ParallelTestManager:
    def __init__(self, manager, args, temp_dir):
        self.args = args
        self.temp_dir = temp_dir
        self.failfast = manager.Event()

    def run_tests(self, suite_index, test_suite, suite_name=None):
        # Fail fast?
        if self.failfast.is_set():
            return [0, [], [], 0, 0, 0, []]  # NVIDIA Modification

        suite_name = suite_name or suite_name_for(test_suite, getattr(self.args, "level", "class"))
        emit_worker_event_with_gil(
            EventKind.SUITE_STARTED,
            suite_index=suite_index,
            suite_name=suite_name,
            test_count=test_suite.countTestCases(),
        )

        result_tuple = None
        with coverage_context(self.args, self.temp_dir):
            runner = self._create_runner()
            result = runner.run(test_suite)

            # Set failfast, if necessary
            if result.shouldStop:
                self.failfast.set()

            emit_worker_event(EventKind.SUITE_FINALIZING)
            result_tuple = self._result_tuple(result)

        emit_worker_event_with_gil(EventKind.SUITE_FINISHED)
        return result_tuple

    def _create_runner(self):
        # NVIDIA Modification for GitLab
        warp.tests.unittest_utils.coverage_enabled = self.args.coverage
        warp.tests.unittest_utils.coverage_temp_dir = self.temp_dir
        warp.tests.unittest_utils.coverage_branch = self.args.coverage_branch

        resultclass = ParallelJunitTestResult

        return unittest.TextTestRunner(
            stream=StringIO(),
            # Descriptions append each test's docstring summary, which is noise
            # across Warp's full suite. Test identifiers alone are enough.
            descriptions=False,  # NVIDIA Modification
            resultclass=resultclass,  # NVIDIA Modification
            verbosity=self.args.verbose,
            failfast=self.args.failfast,
            buffer=self.args.buffer,
        )

    def _result_tuple(self, result):
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


def initialize_test_process(
    lock,
    shared_index,
    args,
    temp_dir,
    event_queue,
    diagnostics_run_dir,
    run_start_monotonic_ns,
):
    """Necessary operations to be executed at the start of every test process.

    Currently this function can be used to set a separate Warp cache. (NVIDIA modification)
    If the environment variable `WARP_CACHE_PATH` is detected, the cache will be placed in the provided path.

    It also ensures that Warp is initialized prior to running any tests.
    """

    with lock:
        shared_index.value += 1
        worker_index = shared_index.value

    configure_worker_diagnostics(
        event_queue=event_queue,
        worker_index=worker_index,
        run_dir=diagnostics_run_dir,
        run_start_monotonic_ns=run_start_monotonic_ns,
    )

    with coverage_context(args, temp_dir):
        import warp as wp  # noqa: PLC0415

        if args.warp_debug:
            wp.config.mode = "debug"

        warp_cache_base_path = get_warp_cache_base_path()

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

        wp.init()

    emit_worker_event_with_gil(EventKind.WORKER_INITIALIZED)
