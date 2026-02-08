# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import signal
import subprocess
import unittest

import warp as wp
from warp.tests.unittest_utils import *


def _run_in_subprocess(func_name: str, timeout: int = 60):
    """Run a module-level function in a subprocess, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-c", f"import warp.tests.test_assert; warp.tests.test_assert.{func_name}()"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _assert_aborted(test_case, returncode):
    """Assert process was killed by signal."""
    if sys.platform == "win32":
        test_case.assertNotEqual(returncode, 0)
    else:
        test_case.assertIn(returncode, [-signal.SIGABRT, -signal.SIGILL, -signal.SIGTRAP])


@wp.kernel
def expect_ones(a: wp.array(dtype=int)):
    i = wp.tid()

    assert a[i] == 1


@wp.kernel
def expect_ones_with_msg(a: wp.array(dtype=int)):
    i = wp.tid()

    assert a[i] == 1, "Array element must be 1"


@wp.kernel
def expect_ones_compound(a: wp.array(dtype=int)):
    i = wp.tid()

    assert a[i] > 0 and a[i] < 2


@wp.func
def expect_ones_function(value: int):
    assert value == 1, "Array element must be 1"


@wp.kernel
def expect_ones_call_function(a: wp.array(dtype=int)):
    i = wp.tid()
    expect_ones_function(a[i])


# Functions that trigger assertions (called from subprocess)
def _trigger_basic_assert():
    wp.config.cache_kernels = False
    wp.config.mode = "debug"
    with wp.ScopedDevice("cpu"):
        wp.launch(expect_ones, (1,), inputs=[wp.zeros(1, dtype=int)])


def _trigger_assert_with_msg():
    wp.config.cache_kernels = False
    wp.config.mode = "debug"
    with wp.ScopedDevice("cpu"):
        wp.launch(expect_ones_with_msg, (1,), inputs=[wp.zeros(1, dtype=int)])


def _trigger_compound_assert():
    wp.config.cache_kernels = False
    wp.config.mode = "debug"
    with wp.ScopedDevice("cpu"):
        wp.launch(expect_ones_compound, (1,), inputs=[wp.full(1, value=3, dtype=int)])


def _trigger_function_assert():
    wp.config.cache_kernels = False
    wp.config.mode = "debug"
    with wp.ScopedDevice("cpu"):
        wp.launch(expect_ones_call_function, (1,), inputs=[wp.full(1, value=3, dtype=int)])


def _trigger_basic_assert_true():
    wp.config.cache_kernels = False
    wp.config.mode = "debug"
    with wp.ScopedDevice("cpu"):
        wp.launch(expect_ones, (1,), inputs=[wp.ones(1, dtype=int)])


def _trigger_compound_assert_true():
    wp.config.cache_kernels = False
    wp.config.mode = "debug"
    with wp.ScopedDevice("cpu"):
        wp.launch(expect_ones_compound, (1,), inputs=[wp.ones(1, dtype=int)])


def _trigger_mode_switch():
    wp.config.cache_kernels = False
    # Release mode first - should not abort
    wp.config.mode = "release"
    with wp.ScopedDevice("cpu"):
        wp.launch(expect_ones, (1,), inputs=[wp.zeros(1, dtype=int)])
    wp.synchronize()
    print("RELEASE_OK", flush=True)

    # Switch to debug mode and use a NEW kernel to ensure it's compiled with debug mode
    # (We use a new kernel to avoid relying on hash invalidation logic for mode changes)
    wp.config.mode = "debug"

    @wp.kernel
    def expect_ones_debug(a: wp.array(dtype=int)):
        i = wp.tid()
        assert a[i] == 1

    with wp.ScopedDevice("cpu"):
        wp.launch(expect_ones_debug, (1,), inputs=[wp.zeros(1, dtype=int)])


class TestAssertRelease(unittest.TestCase):
    """Assert test cases that are to be run with Warp in release mode."""

    @classmethod
    def setUpClass(cls):
        cls._saved_mode = wp.get_module_options()["mode"]
        cls._saved_cache_kernels = wp.config.cache_kernels

        wp.config.mode = "release"
        wp.config.cache_kernels = False

    @classmethod
    def tearDownClass(cls):
        wp.set_module_options({"mode": cls._saved_mode})
        wp.config.cache_kernels = cls._saved_cache_kernels

    def test_basic_assert_false_condition(self):
        with wp.ScopedDevice("cpu"):
            wp.load_module(device=wp.get_device())

            input_array = wp.zeros(1, dtype=int)

            capture = StdErrCapture()
            capture.begin()

            wp.launch(expect_ones, input_array.shape, inputs=[input_array])

            output = capture.end()

            self.assertEqual(output, "", f"Kernel should not print anything to stderr, got {output}")

    def test_basic_assert_with_msg(self):
        with wp.ScopedDevice("cpu"):
            wp.load_module(device=wp.get_device())

            input_array = wp.zeros(1, dtype=int)

            capture = StdErrCapture()
            capture.begin()

            wp.launch(expect_ones_with_msg, input_array.shape, inputs=[input_array])

            output = capture.end()

            self.assertEqual(output, "", f"Kernel should not print anything to stderr, got {output}")

    def test_compound_assert_false_condition(self):
        with wp.ScopedDevice("cpu"):
            wp.load_module(device=wp.get_device())

            input_array = wp.full(1, value=3, dtype=int)

            capture = StdErrCapture()
            capture.begin()

            wp.launch(expect_ones_compound, input_array.shape, inputs=[input_array])

            output = capture.end()

            self.assertEqual(output, "", f"Kernel should not print anything to stderr, got {output}")

    def test_basic_assert_false_condition_function(self):
        with wp.ScopedDevice("cpu"):
            wp.load_module(device=wp.get_device())

            input_array = wp.full(1, value=3, dtype=int)

            capture = StdErrCapture()
            capture.begin()

            wp.launch(expect_ones_call_function, input_array.shape, inputs=[input_array])

            output = capture.end()

            self.assertEqual(output, "", f"Kernel should not print anything to stderr, got {output}")


# NOTE: Failed assertions abort the process on CPU and leave CUDA context unrecoverable,
# so all debug assertion tests run in subprocesses.
class TestAssertDebug(unittest.TestCase):
    """Assert test cases that are to be run with Warp in debug mode."""

    def test_basic_assert_false_condition(self):
        returncode, _, stderr = _run_in_subprocess("_trigger_basic_assert")
        _assert_aborted(self, returncode)
        self.assertRegex(stderr, r"Assertion failed: .*assert a\[i\] == 1")

    def test_basic_assert_true_condition(self):
        returncode, _, stderr = _run_in_subprocess("_trigger_basic_assert_true")
        self.assertEqual(returncode, 0)
        self.assertNotIn("Assertion failed", stderr)

    def test_basic_assert_with_msg(self):
        returncode, _, stderr = _run_in_subprocess("_trigger_assert_with_msg")
        _assert_aborted(self, returncode)
        self.assertRegex(stderr, r"Assertion failed: .*assert a\[i\] == 1.*Array element must be 1")

    def test_compound_assert_true_condition(self):
        returncode, _, stderr = _run_in_subprocess("_trigger_compound_assert_true")
        self.assertEqual(returncode, 0)
        self.assertNotIn("Assertion failed", stderr)

    def test_compound_assert_false_condition(self):
        returncode, _, stderr = _run_in_subprocess("_trigger_compound_assert")
        _assert_aborted(self, returncode)
        self.assertRegex(stderr, r"Assertion failed: .*assert a\[i\] > 0 and a\[i\] < 2")

    def test_basic_assert_false_condition_function(self):
        returncode, _, stderr = _run_in_subprocess("_trigger_function_assert")
        _assert_aborted(self, returncode)
        self.assertRegex(stderr, r"Assertion failed: .*assert value == 1.*Array element must be 1")


class TestAssertModeSwitch(unittest.TestCase):
    """Test that a new kernel defined in debug mode has assertions enabled, even after running in release mode."""

    def test_switch_to_debug_mode(self):
        """Test that a new kernel defined after switching to debug mode has assertions enabled."""
        returncode, stdout, stderr = _run_in_subprocess("_trigger_mode_switch")
        # Verify release mode completed without aborting
        self.assertIn("RELEASE_OK", stdout, "Release mode should complete without aborting")
        # Verify debug mode aborted
        _assert_aborted(self, returncode)
        self.assertRegex(stderr, r"Assertion failed: .*assert a\[i\] == 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
