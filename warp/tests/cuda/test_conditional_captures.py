# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import warp as wp
from warp.context import assert_conditional_graph_support
from warp.tests.unittest_utils import *


def check_conditional_graph_support():
    try:
        assert_conditional_graph_support()
    except Exception:
        return False
    return True


@wp.kernel
def multiply_by_one_kernel(array: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    array[tid] = array[tid] * 1.0


def launch_multiply_by_one(array: wp.array(dtype=wp.float32)):
    wp.launch(multiply_by_one_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_two_kernel(array: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    array[tid] = array[tid] * 2.0


def launch_multiply_by_two(array: wp.array(dtype=wp.float32)):
    wp.launch(multiply_by_two_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_two_kernel_limited(
    array: wp.array(dtype=wp.float32), condition: wp.array(dtype=wp.int32), limit: float
):
    tid = wp.tid()
    array[tid] = array[tid] * 2.0

    # set termination condition if limit exceeded
    if array[tid] > limit:
        condition[0] = 0


def launch_multiply_by_two_until_limit(array: wp.array(dtype=wp.float32), cond: wp.array(dtype=wp.int32), limit: float):
    wp.launch(multiply_by_two_kernel_limited, dim=array.size, inputs=[array, cond, limit])


@wp.kernel
def multiply_by_three_kernel(array: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    array[tid] = array[tid] * 3.0


def launch_multiply_by_three(array: wp.array(dtype=wp.float32)):
    wp.launch(multiply_by_three_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_five_kernel(array: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    array[tid] = array[tid] * 5.0


def launch_multiply_by_five(array: wp.array(dtype=wp.float32)):
    wp.launch(multiply_by_five_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_seven_kernel(array: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    array[tid] = array[tid] * 7.0


def launch_multiply_by_seven(array: wp.array(dtype=wp.float32)):
    wp.launch(multiply_by_seven_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_eleven_kernel(array: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    array[tid] = array[tid] * 11.0


def launch_multiply_by_eleven(array: wp.array(dtype=wp.float32)):
    wp.launch(multiply_by_eleven_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_thirteen_kernel(array: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    array[tid] = array[tid] * 13.0


def launch_multiply_by_thirteen(array: wp.array(dtype=wp.float32)):
    wp.launch(multiply_by_thirteen_kernel, dim=array.size, inputs=[array])


def launch_multiply_by_two_or_thirteen(array: wp.array(dtype=wp.float32), cond: wp.array(dtype=wp.int32)):
    wp.capture_if(
        cond,
        lambda: launch_multiply_by_two(array),
        lambda: launch_multiply_by_thirteen(array),
    )


def launch_multiply_by_three_or_eleven(array: wp.array(dtype=wp.float32), cond: wp.array(dtype=wp.int32)):
    wp.capture_if(
        cond,
        lambda: launch_multiply_by_three(array),
        lambda: launch_multiply_by_eleven(array),
    )


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_if_capture(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        array = wp.zeros(4, dtype=wp.float32)
        condition = wp.zeros(1, dtype=wp.int32)

        # preload module before graph capture
        wp.load_module(device=device)

        # capture graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(
                condition,
                launch_multiply_by_two,
                array=array,
            )

        # test different conditions
        for cond in [0, 1]:
            array.assign([1.0, 2.0, 3.0, 4.0])
            condition.assign([cond])

            wp.capture_launch(capture.graph)

            if cond == 0:
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            else:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_if_capture_with_subgraph(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        array = wp.zeros(4, dtype=wp.float32)
        condition = wp.zeros(1, dtype=wp.int32)

        # preload module before graph capture
        wp.load_module(device=device)

        # capture if branch graph
        with wp.ScopedCapture(force_module_load=False) as if_capture:
            launch_multiply_by_two(array)

        # capture main graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(
                condition,
                if_capture.graph,
                array=array,
            )

        # test different conditions
        for cond in [0, 1]:
            array.assign([1.0, 2.0, 3.0, 4.0])
            condition.assign([cond])

            wp.capture_launch(capture.graph)

            if cond == 0:
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            else:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


def test_if_nocapture(test, device):
    with wp.ScopedDevice(device):
        # test different conditions
        for cond in [0, 1]:
            array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
            condition = wp.array([cond], dtype=wp.int32)

            wp.capture_if(
                condition,
                launch_multiply_by_two,
                array=array,
            )

            if cond == 0:
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            else:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


def test_if_with_subgraph(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        # test different conditions
        for cond in [0, 1]:
            array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
            condition = wp.array([cond], dtype=wp.int32)

            # capture if branch graph
            with wp.ScopedCapture(force_module_load=False) as if_capture:
                launch_multiply_by_two(array)

            wp.capture_if(
                condition,
                if_capture.graph,
            )

            if cond == 0:
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            else:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_if_else_capture(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        array = wp.zeros(4, dtype=wp.float32)
        condition = wp.zeros(1, dtype=wp.int32)

        # preload module before graph capture
        wp.load_module(device=device)

        # capture graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(
                condition,
                launch_multiply_by_two,
                launch_multiply_by_three,
                array=array,
            )

        # test different conditions
        for cond in [0, 1]:
            array.assign([1.0, 2.0, 3.0, 4.0])
            condition.assign([cond])

            wp.capture_launch(capture.graph)

            if cond == 0:
                expected = np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32)
            else:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_if_else_capture_with_subgraph(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        array = wp.zeros(4, dtype=wp.float32)
        condition = wp.zeros(1, dtype=wp.int32)

        # preload module before graph capture
        wp.load_module(device=device)

        with wp.ScopedCapture(force_module_load=False) as capture_true:
            launch_multiply_by_two(array)

        with wp.ScopedCapture(force_module_load=False) as capture_false:
            launch_multiply_by_three(array)

        # capture graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(
                condition,
                capture_true.graph,
                capture_false.graph,
                array=array,
            )

            launch_multiply_by_one(array)

        # test different conditions
        for cond in [0, 1]:
            array.assign([1.0, 2.0, 3.0, 4.0])
            condition.assign([cond])

            wp.capture_launch(capture.graph)

            if cond == 0:
                expected = np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32)
            else:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


def test_if_else_nocapture(test, device):
    with wp.ScopedDevice(device):
        # test different conditions
        for cond in [0, 1]:
            array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
            condition = wp.array([cond], dtype=wp.int32)

            wp.capture_if(
                condition,
                launch_multiply_by_two,
                launch_multiply_by_three,
                array=array,
            )

            if cond == 0:
                expected = np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32)
            else:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


def test_if_else_with_subgraph(test, device):
    with wp.ScopedDevice(device):
        # test different conditions
        for cond in [0, 1]:
            array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
            condition = wp.array([cond], dtype=wp.int32)

            # capture if-true branch graph
            with wp.ScopedCapture(force_module_load=False) as if_true_capture:
                launch_multiply_by_two(array)
            if_true_graph = if_true_capture.graph

            # capture if-false branch graph
            with wp.ScopedCapture(force_module_load=False) as if_false_capture:
                launch_multiply_by_three(array)
            if_false_graph = if_false_capture.graph

            wp.capture_if(
                condition,
                if_true_graph,
                if_false_graph,
            )

            if cond == 0:
                expected = np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32)
            else:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_else_capture(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        array = wp.zeros(4, dtype=wp.float32)
        condition = wp.zeros(1, dtype=wp.int32)

        # preload module before graph capture
        wp.load_module(device=device)

        # capture graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(
                condition,
                on_false=launch_multiply_by_two,
                array=array,
            )

        # test different conditions
        for cond in [0, 1]:
            array.assign([1.0, 2.0, 3.0, 4.0])
            condition.assign([cond])

            wp.capture_launch(capture.graph)

            if cond == 0:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
            else:
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_else_capture_with_subgraph(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        array = wp.zeros(4, dtype=wp.float32)
        condition = wp.zeros(1, dtype=wp.int32)

        # preload module before graph capture
        wp.load_module(device=device)

        # capture subgraph for multiply by two
        with wp.ScopedCapture(force_module_load=False) as multiply_capture:
            launch_multiply_by_two(array=array)
        multiply_graph = multiply_capture.graph

        # capture main graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_if(
                condition,
                on_false=multiply_graph,
                array=array,
            )

        # test different conditions
        for cond in [0, 1]:
            array.assign([1.0, 2.0, 3.0, 4.0])
            condition.assign([cond])

            wp.capture_launch(capture.graph)

            if cond == 0:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
            else:
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


def test_else_nocapture(test, device):
    with wp.ScopedDevice(device):
        # test different conditions
        for cond in [0, 1]:
            array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
            condition = wp.array([cond], dtype=wp.int32)

            wp.capture_if(
                condition,
                on_false=launch_multiply_by_two,
                array=array,
            )

            if cond == 0:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
            else:
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


def test_else_with_subgraph(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        # test different conditions
        for cond in [0, 1]:
            array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
            condition = wp.array([cond], dtype=wp.int32)

            # capture else branch graph
            with wp.ScopedCapture(force_module_load=False) as else_capture:
                launch_multiply_by_two(array)
            else_graph = else_capture.graph

            wp.capture_if(
                condition,
                on_false=else_graph,
            )

            if cond == 0:
                expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
            else:
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_while_capture(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        array = wp.zeros(4, dtype=wp.float32)
        condition = wp.zeros(1, dtype=wp.int32)

        # preload module before graph capture
        wp.load_module(device=device)

        # capture graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(
                condition,
                launch_multiply_by_two_until_limit,
                array=array,
                cond=condition,
                limit=1000,
            )

        # test different conditions
        for cond in [0, 1]:
            array.assign([1.0, 2.0, 3.0, 4.0])
            condition.assign([cond])

            wp.capture_launch(capture.graph)

            # Check the output matches expected values
            if cond == 0:
                # No iterations executed since condition was false
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            else:
                # Multiple iterations until limit reached
                expected = np.array([256.0, 512.0, 768.0, 1024.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_while_capture_with_subgraph(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        array = wp.zeros(4, dtype=wp.float32)
        condition = wp.zeros(1, dtype=wp.int32)

        # preload module before graph capture
        wp.load_module(device=device)

        # capture subgraph for body of while loop
        with wp.ScopedCapture(force_module_load=False) as body_capture:
            launch_multiply_by_two_until_limit(array=array, cond=condition, limit=1000)

        # capture main graph with while node
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(
                condition,
                body_capture.graph,
                array=array,
                cond=condition,
                limit=1000,
            )

        # test different conditions
        for cond in [0, 1]:
            array.assign([1.0, 2.0, 3.0, 4.0])
            condition.assign([cond])

            wp.capture_launch(capture.graph)

            # Check the output matches expected values
            if cond == 0:
                # No iterations executed since condition was false
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            else:
                # Multiple iterations until limit reached
                expected = np.array([256.0, 512.0, 768.0, 1024.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


def test_while_nocapture(test, device):
    with wp.ScopedDevice(device):
        # test different conditions
        for cond in [0, 1]:
            array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
            condition = wp.array([cond], dtype=wp.int32)

            wp.capture_while(
                condition,
                launch_multiply_by_two_until_limit,
                array=array,
                cond=condition,
                limit=1000,
            )

            # Check the output matches expected values
            if cond == 0:
                # No iterations executed since condition was false
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            else:
                # Multiple iterations until limit reached
                expected = np.array([256.0, 512.0, 768.0, 1024.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


def test_while_with_subgraph(test, device):
    with wp.ScopedDevice(device):
        # test different conditions
        for cond in [0, 1]:
            array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
            condition = wp.array([cond], dtype=wp.int32)

            # capture body graph
            with wp.ScopedCapture(force_module_load=False) as body_capture:
                launch_multiply_by_two_until_limit(array=array, cond=condition, limit=1000)
            body_graph = body_capture.graph

            wp.capture_while(
                condition,
                body_graph,
            )

            # Check the output matches expected values
            if cond == 0:
                # No iterations executed since condition was false
                expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            else:
                # Multiple iterations until limit reached
                expected = np.array([256.0, 512.0, 768.0, 1024.0], dtype=np.float32)

            np.testing.assert_array_equal(array.numpy(), expected)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_complex_capture(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        # data array
        array = wp.zeros(4, dtype=wp.float32)

        # condition arrays
        condition1 = wp.zeros(1, dtype=wp.int32)
        condition2 = wp.zeros(1, dtype=wp.int32)
        while_condition = wp.zeros(1, dtype=wp.int32)

        limit = 1000

        # preload module before graph capture
        wp.load_module(device=device)

        # capture graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(
                while_condition,
                launch_multiply_by_two_until_limit,
                array=array,
                cond=while_condition,
                limit=limit,
            )

            launch_multiply_by_seven(array)

            wp.capture_if(
                condition1,
                launch_multiply_by_two_or_thirteen,  # nested if-else
                launch_multiply_by_three_or_eleven,  # nested if-else
                array=array,
                cond=condition2,
            )

            launch_multiply_by_five(array)

        # test different conditions
        for cond1 in [0, 1]:
            for cond2 in [0, 1]:
                for while_cond in [0, 1]:
                    # reset data
                    array.assign([1.0, 2.0, 3.0, 4.0])

                    # set conditions
                    condition1.assign([cond1])
                    condition2.assign([cond2])
                    while_condition.assign([while_cond])

                    # launch the graph
                    wp.capture_launch(capture.graph)

                    # calculate expected values based on conditions
                    base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
                    cond = while_cond
                    while cond != 0:
                        base = 2 * base
                        # set cond to zero if any value exceeds limit
                        if np.any(base > limit):
                            cond = 0

                    # multiply by 7
                    base *= 7.0

                    # apply nested conditions
                    if cond1:
                        if cond2:
                            base *= 2.0  # multiply by 2
                        else:
                            base *= 13.0  # multiply by 13
                    else:
                        if cond2:
                            base *= 3.0  # multiply by 3
                        else:
                            base *= 11.0  # multiply by 11

                    # multiply by 5
                    base *= 5.0

                    if not np.array_equal(array.numpy(), base):
                        # print(f"Conditions: while_cond={while_cond}, cond1={cond1}, cond2={cond2}, limit={limit}")
                        np.testing.assert_array_equal(array.numpy(), base)


@unittest.skipUnless(check_conditional_graph_support(), "Conditional graph nodes not supported")
def test_complex_capture_with_subgraphs(test, device):
    assert device.is_cuda

    with wp.ScopedDevice(device):
        # data array
        array = wp.zeros(4, dtype=wp.float32)

        # condition arrays
        condition1 = wp.zeros(1, dtype=wp.int32)
        while_condition = wp.zeros(1, dtype=wp.int32)

        limit = 1000

        # preload module before graph capture
        wp.load_module(device=device)

        # capture subgraphs
        with wp.ScopedCapture(force_module_load=False) as while_capture:
            launch_multiply_by_two_until_limit(array, while_condition, limit)
        while_graph = while_capture.graph

        with wp.ScopedCapture(force_module_load=False) as if_true_capture:
            launch_multiply_by_two(array)
            launch_multiply_by_thirteen(array)
        if_true_graph = if_true_capture.graph

        with wp.ScopedCapture(force_module_load=False) as if_false_capture:
            launch_multiply_by_three(array)
            launch_multiply_by_eleven(array)
        if_false_graph = if_false_capture.graph

        # capture main graph
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(while_condition, while_graph)

            launch_multiply_by_seven(array)

            wp.capture_if(condition1, if_true_graph, if_false_graph)

            launch_multiply_by_five(array)

        # test different conditions
        for cond1 in [0, 1]:
            for while_cond in [0, 1]:
                # reset data
                array.assign([1.0, 2.0, 3.0, 4.0])

                # set conditions
                condition1.assign([cond1])
                while_condition.assign([while_cond])

                # launch the graph
                wp.capture_launch(capture.graph)

                # calculate expected values based on conditions
                base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
                cond = while_cond
                while cond != 0:
                    base = 2 * base
                    # set cond to zero if any value exceeds limit
                    if np.any(base > limit):
                        cond = 0

                # multiply by 7
                base *= 7.0

                # apply nested conditions
                if cond1:
                    base *= 2.0  # multiply by 2
                    base *= 13.0  # multiply by 13
                else:
                    base *= 3.0  # multiply by 3
                    base *= 11.0  # multiply by 11

                # multiply by 5
                base *= 5.0

                if not np.array_equal(array.numpy(), base):
                    # print(f"Conditions: while_cond={while_cond}, cond1={cond1}, cond2={cond2}, limit={limit}")
                    np.testing.assert_array_equal(array.numpy(), base)


def test_complex_nocapture(test, device):
    with wp.ScopedDevice(device):
        limit = 1000

        # test different conditions
        for cond1 in [0, 1]:
            for cond2 in [0, 1]:
                for while_cond in [0, 1]:
                    # set data
                    array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)

                    # set conditions
                    condition1 = wp.array([cond1], dtype=wp.int32)
                    condition2 = wp.array([cond2], dtype=wp.int32)
                    while_condition = wp.array([while_cond], dtype=wp.int32)

                    wp.capture_while(
                        while_condition,
                        launch_multiply_by_two_until_limit,
                        array=array,
                        cond=while_condition,
                        limit=limit,
                    )

                    launch_multiply_by_seven(array)

                    wp.capture_if(
                        condition1,
                        launch_multiply_by_two_or_thirteen,  # nested if-else
                        launch_multiply_by_three_or_eleven,  # nested if-else
                        array=array,
                        cond=condition2,
                    )

                    launch_multiply_by_five(array)

                    # calculate expected values based on conditions
                    base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
                    cond = while_cond
                    while cond != 0:
                        base = 2 * base
                        # set cond to zero if any value exceeds limit
                        if np.any(base > limit):
                            cond = 0

                    # multiply by 7
                    base *= 7.0

                    # apply nested conditions
                    if cond1:
                        if cond2:
                            base *= 2.0  # multiply by 2
                        else:
                            base *= 13.0  # multiply by 13
                    else:
                        if cond2:
                            base *= 3.0  # multiply by 3
                        else:
                            base *= 11.0  # multiply by 11

                    # multiply by 5
                    base *= 5.0

                    if not np.array_equal(array.numpy(), base):
                        # print(f"Conditions: while_cond={while_cond}, cond1={cond1}, cond2={cond2}, limit={limit}")
                        np.testing.assert_array_equal(array.numpy(), base)


def test_complex_with_subgraphs(test, device):
    with wp.ScopedDevice(device):
        limit = 1000

        # test different conditions
        for cond1 in [0, 1]:
            for while_cond in [0, 1]:
                # set data
                array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)

                # set conditions
                condition1 = wp.array([cond1], dtype=wp.int32)
                while_condition = wp.array([while_cond], dtype=wp.int32)

                # capture while loop body graph
                with wp.ScopedCapture(force_module_load=False) as while_body_capture:
                    launch_multiply_by_two_until_limit(array=array, cond=while_condition, limit=limit)
                while_body_graph = while_body_capture.graph

                # capture nested if-else true branch
                with wp.ScopedCapture(force_module_load=False) as if_true_capture:
                    launch_multiply_by_two(array=array)
                    launch_multiply_by_thirteen(array=array)
                if_true_graph = if_true_capture.graph

                # capture nested if-else false branch
                with wp.ScopedCapture(force_module_load=False) as if_false_capture:
                    launch_multiply_by_three(array=array)
                    launch_multiply_by_eleven(array=array)
                if_false_graph = if_false_capture.graph

                wp.capture_while(
                    while_condition,
                    while_body_graph,
                )

                launch_multiply_by_seven(array)

                wp.capture_if(
                    condition1,
                    if_true_graph,
                    if_false_graph,
                )

                launch_multiply_by_five(array)

                # calculate expected values based on conditions
                base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
                cond = while_cond
                while cond != 0:
                    base = 2 * base
                    # set cond to zero if any value exceeds limit
                    if np.any(base > limit):
                        cond = 0

                # multiply by 7
                base *= 7.0

                # apply nested conditions
                if cond1:
                    base *= 2.0  # multiply by 2
                    base *= 13.0  # multiply by 13
                else:
                    base *= 3.0  # multiply by 3
                    base *= 11.0  # multiply by 11

                # multiply by 5
                base *= 5.0

                if not np.array_equal(array.numpy(), base):
                    # print(f"Conditions: while_cond={while_cond}, cond1={cond1}, cond2={cond2}, limit={limit}")
                    np.testing.assert_array_equal(array.numpy(), base)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestConditionalCaptures(unittest.TestCase):
    pass


# tests with graph capture
add_function_test(TestConditionalCaptures, "test_if_capture", test_if_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures, "test_if_capture_with_subgraph", test_if_capture_with_subgraph, devices=cuda_devices
)
add_function_test(TestConditionalCaptures, "test_if_else_capture", test_if_else_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures,
    "test_if_else_capture_with_subgraph",
    test_if_else_capture_with_subgraph,
    devices=cuda_devices,
)
add_function_test(TestConditionalCaptures, "test_else_capture", test_else_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures, "test_else_capture_with_subgraph", test_else_capture_with_subgraph, devices=cuda_devices
)
add_function_test(TestConditionalCaptures, "test_while_capture", test_while_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures, "test_while_capture_with_subgraph", test_while_capture_with_subgraph, devices=cuda_devices
)
add_function_test(TestConditionalCaptures, "test_complex_capture", test_complex_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures,
    "test_complex_capture_with_subgraphs",
    test_complex_capture_with_subgraphs,
    devices=cuda_devices,
)


# tests without graph capture
add_function_test(TestConditionalCaptures, "test_if_nocapture", test_if_nocapture, devices=devices)
add_function_test(TestConditionalCaptures, "test_if_with_subgraph", test_if_with_subgraph, devices=cuda_devices)
add_function_test(TestConditionalCaptures, "test_if_else_nocapture", test_if_else_nocapture, devices=devices)
add_function_test(
    TestConditionalCaptures, "test_if_else_with_subgraph", test_if_else_with_subgraph, devices=cuda_devices
)
add_function_test(TestConditionalCaptures, "test_else_nocapture", test_else_nocapture, devices=devices)
add_function_test(TestConditionalCaptures, "test_else_with_subgraph", test_else_with_subgraph, devices=cuda_devices)
add_function_test(TestConditionalCaptures, "test_while_nocapture", test_while_nocapture, devices=devices)
add_function_test(TestConditionalCaptures, "test_while_with_subgraph", test_while_with_subgraph, devices=cuda_devices)
add_function_test(TestConditionalCaptures, "test_complex_nocapture", test_complex_nocapture, devices=devices)
add_function_test(
    TestConditionalCaptures,
    "test_complex_with_subgraphs",
    test_complex_with_subgraphs,
    devices=cuda_devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
