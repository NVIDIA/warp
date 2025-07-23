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

import unittest

import warp as wp
from warp.tests.unittest_utils import *


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


# NOTE: Failed assertions on CUDA devices leaves the CUDA context in an unrecoverable state,
# so we currently do not test them.
class TestAssertDebug(unittest.TestCase):
    """Assert test cases that are to be run with Warp in debug mode."""

    @classmethod
    def setUpClass(cls):
        cls._saved_mode = wp.get_module_options()["mode"]
        cls._saved_cache_kernels = wp.config.cache_kernels

        wp.set_module_options({"mode": "debug"})
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

            # Older Windows C runtimes have a bug where stdout sometimes does not get properly flushed.
            if output != "" or sys.platform != "win32":
                self.assertRegex(output, r"Assertion failed: .*assert a\[i\] == 1")

    def test_basic_assert_true_condition(self):
        with wp.ScopedDevice("cpu"):
            wp.load_module(device=wp.get_device())

            input_array = wp.ones(1, dtype=int)

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

            # Older Windows C runtimes have a bug where stdout sometimes does not get properly flushed.
            if output != "" or sys.platform != "win32":
                self.assertRegex(output, r"Assertion failed: .*assert a\[i\] == 1.*Array element must be 1")

    def test_compound_assert_true_condition(self):
        with wp.ScopedDevice("cpu"):
            wp.load_module(device=wp.get_device())

            input_array = wp.ones(1, dtype=int)

            capture = StdErrCapture()
            capture.begin()

            wp.launch(expect_ones_compound, input_array.shape, inputs=[input_array])

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

            # Older Windows C runtimes have a bug where stdout sometimes does not get properly flushed.
            if output != "" or sys.platform != "win32":
                self.assertRegex(output, r"Assertion failed: .*assert a\[i\] > 0 and a\[i\] < 2")

    def test_basic_assert_false_condition_function(self):
        with wp.ScopedDevice("cpu"):
            wp.load_module(device=wp.get_device())

            input_array = wp.full(1, value=3, dtype=int)

            capture = StdErrCapture()
            capture.begin()

            wp.launch(expect_ones_call_function, input_array.shape, inputs=[input_array])

            output = capture.end()

            # Older Windows C runtimes have a bug where stdout sometimes does not get properly flushed.
            if output != "" or sys.platform != "win32":
                self.assertRegex(output, r"Assertion failed: .*assert value == 1.*Array element must be 1")


class TestAssertModeSwitch(unittest.TestCase):
    """Test that switching from release mode to debug mode rebuilds the module with assertions enabled."""

    @classmethod
    def setUpClass(cls):
        cls._saved_mode = wp.config.mode
        cls._saved_mode_module = wp.get_module_options()["mode"]
        cls._saved_cache_kernels = wp.config.cache_kernels

        # Don't set any mode initially - use whatever the default is
        wp.config.cache_kernels = False

    @classmethod
    def tearDownClass(cls):
        wp.config.mode = cls._saved_mode
        wp.set_module_options({"mode": cls._saved_mode_module})
        wp.config.cache_kernels = cls._saved_cache_kernels

    def test_switch_to_debug_mode(self):
        """Test that switching from release mode to debug mode rebuilds the module with assertions enabled."""
        with wp.ScopedDevice("cpu"):
            # Create an array that will trigger an assertion
            input_array = wp.zeros(1, dtype=int)

            # In default mode, this should not assert
            capture = StdErrCapture()
            capture.begin()
            wp.launch(expect_ones, input_array.shape, inputs=[input_array])
            output = capture.end()

            # Should not have any assertion output in release mode
            self.assertEqual(output, "", f"Kernel should not print anything to stderr in release mode, got {output}")

            # Now switch to debug mode and have it compile a new kernel
            wp.config.mode = "debug"

            @wp.kernel
            def expect_ones_debug(a: wp.array(dtype=int)):
                i = wp.tid()
                assert a[i] == 1

            # In debug mode, this should assert
            capture = StdErrCapture()
            capture.begin()
            wp.launch(expect_ones_debug, input_array.shape, inputs=[input_array])
            output = capture.end()

            # Should have assertion output in debug mode
            # Older Windows C runtimes have a bug where stdout sometimes does not get properly flushed.
            if output != "" or sys.platform != "win32":
                self.assertRegex(output, r"Assertion failed: .*assert a\[i\] == 1")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
