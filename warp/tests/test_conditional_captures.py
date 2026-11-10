# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def multiply_by_one_kernel(array: wp.array[wp.float32]):
    tid = wp.tid()
    array[tid] = array[tid] * 1.0


def launch_multiply_by_one(array: wp.array[wp.float32]):
    wp.launch(multiply_by_one_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_two_kernel(array: wp.array[wp.float32]):
    tid = wp.tid()
    array[tid] = array[tid] * 2.0


def launch_multiply_by_two(array: wp.array[wp.float32]):
    wp.launch(multiply_by_two_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_two_kernel_limited(array: wp.array[wp.float32], condition: wp.array[wp.int32], limit: float):
    tid = wp.tid()
    array[tid] = array[tid] * 2.0

    # set termination condition if limit exceeded
    if array[tid] > limit:
        condition[0] = 0


def launch_multiply_by_two_until_limit(array: wp.array[wp.float32], cond: wp.array[wp.int32], limit: float):
    wp.launch(multiply_by_two_kernel_limited, dim=array.size, inputs=[array, cond, limit])


@wp.kernel
def multiply_by_three_kernel(array: wp.array[wp.float32]):
    tid = wp.tid()
    array[tid] = array[tid] * 3.0


def launch_multiply_by_three(array: wp.array[wp.float32]):
    wp.launch(multiply_by_three_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_five_kernel(array: wp.array[wp.float32]):
    tid = wp.tid()
    array[tid] = array[tid] * 5.0


def launch_multiply_by_five(array: wp.array[wp.float32]):
    wp.launch(multiply_by_five_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_seven_kernel(array: wp.array[wp.float32]):
    tid = wp.tid()
    array[tid] = array[tid] * 7.0


def launch_multiply_by_seven(array: wp.array[wp.float32]):
    wp.launch(multiply_by_seven_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_eleven_kernel(array: wp.array[wp.float32]):
    tid = wp.tid()
    array[tid] = array[tid] * 11.0


def launch_multiply_by_eleven(array: wp.array[wp.float32]):
    wp.launch(multiply_by_eleven_kernel, dim=array.size, inputs=[array])


@wp.kernel
def multiply_by_thirteen_kernel(array: wp.array[wp.float32]):
    tid = wp.tid()
    array[tid] = array[tid] * 13.0


def launch_multiply_by_thirteen(array: wp.array[wp.float32]):
    wp.launch(multiply_by_thirteen_kernel, dim=array.size, inputs=[array])


def launch_multiply_by_two_or_thirteen(array: wp.array[wp.float32], cond: wp.array[wp.int32]):
    wp.capture_if(
        cond,
        lambda: launch_multiply_by_two(array),
        lambda: launch_multiply_by_thirteen(array),
    )


def launch_multiply_by_three_or_eleven(array: wp.array[wp.float32], cond: wp.array[wp.int32]):
    wp.capture_if(
        cond,
        lambda: launch_multiply_by_three(array),
        lambda: launch_multiply_by_eleven(array),
    )


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
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


def test_if_capture_with_subgraph(test, device):
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graph nodes not supported")

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


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
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


def test_if_else_capture_with_subgraph(test, device):
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graph nodes not supported")

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


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
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


def test_else_capture_with_subgraph(test, device):
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graph nodes not supported")

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


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
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


def test_while_capture_with_subgraph(test, device):
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graph nodes not supported")

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


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
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


def test_complex_capture_with_subgraphs(test, device):
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graph nodes not supported")

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


def test_graph_debug_dot_print(test, device):
    """Test debug graph output in DOT format."""
    array = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.launch(multiply_by_two_kernel, dim=array.size, inputs=[array], device=device)
        wp.launch(multiply_by_three_kernel, dim=array.size, inputs=[array], device=device)
        wp.launch(multiply_by_five_kernel, dim=array.size, inputs=[array], device=device)
        wp.launch(multiply_by_seven_kernel, dim=array.size, inputs=[array], device=device)

    with tempfile.TemporaryDirectory() as temp_dir:
        dot_path = os.path.join(temp_dir, "test_graph.dot")
        wp.capture_debug_dot_print(capture.graph, dot_path, verbose=True)
        with open(dot_path) as dot_file:
            dot = dot_file.read()

    test.assertTrue(dot.startswith("digraph"))
    test.assertRegex(dot, r"multiply_by_two_kernel_[0-9a-f]+_cuda_kernel_forward")
    test.assertRegex(dot, r"multiply_by_three_kernel_[0-9a-f]+_cuda_kernel_forward")
    test.assertRegex(dot, r"multiply_by_five_kernel_[0-9a-f]+_cuda_kernel_forward")
    test.assertRegex(dot, r"multiply_by_seven_kernel_[0-9a-f]+_cuda_kernel_forward")


# ================================================================================================================
# freeing parent-graph allocations during body capture
# ================================================================================================================


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
def test_free_during_body_capture(test, device):
    """Test freeing a parent-graph allocation while a conditional body graph is being captured.

    Dropping the last reference to an array allocated in the parent graph while a
    conditional body graph is being captured must not add a free node to the body
    graph or latch a CUDA error. The allocation is retained for the lifetime of the
    graph instead.
    """
    with wp.ScopedDevice(device):
        # preload module before graph capture
        wp.load_module(device=device)

        for branch in ("on_true", "on_false", "while"):
            array = wp.zeros(4, dtype=wp.float32)
            condition = wp.zeros(1, dtype=wp.int32)
            holder = {}

            def if_body(holder=holder, array=array):
                launch_multiply_by_two(array)
                holder.pop("temp")  # drop the last reference during body capture

            def while_body(holder=holder, array=array, condition=condition):
                launch_multiply_by_two_until_limit(array, condition, 16.0)
                holder.pop("temp")  # drop the last reference during body capture

            stderr_capture = StdErrCapture()
            stderr_capture.begin()
            try:
                with wp.ScopedCapture(force_module_load=False) as capture:
                    # allocated in the parent graph, freed during body capture
                    holder["temp"] = wp.zeros(1024, dtype=wp.float32)
                    launch_multiply_by_one(holder["temp"])
                    if branch == "on_true":
                        wp.capture_if(condition, on_true=if_body)
                    elif branch == "on_false":
                        wp.capture_if(condition, on_false=if_body)
                    else:
                        wp.capture_while(condition, while_body=while_body)
            finally:
                output = stderr_capture.end()

            test.assertEqual(output, "", f"unexpected stderr output during capture ({branch}): {output}")

            # test different conditions
            for cond in [0, 1]:
                array.assign([1.0, 2.0, 3.0, 4.0])
                condition.assign([cond])

                wp.capture_launch(capture.graph)

                if branch == "on_true":
                    factor = 2.0 if cond else 1.0
                    expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * factor
                elif branch == "on_false":
                    factor = 1.0 if cond else 2.0
                    expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * factor
                else:
                    if cond:
                        # multiplied by two until an element exceeds the limit of 16
                        expected = np.array([8.0, 16.0, 24.0, 32.0], dtype=np.float32)
                    else:
                        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

                np.testing.assert_array_equal(array.numpy(), expected)

            # a latched CUDA error would poison this non-contiguous clone
            src = wp.zeros((8, 8), dtype=wp.float32)
            clone = wp.clone(src[::2])
            wp.synchronize_device()
            test.assertEqual(clone.shape, (4, 8))


# ================================================================================================================
# test exceptions
# ================================================================================================================


# Separately captured child graphs still cannot contain allocation nodes.
def body_with_alloc():
    wp.zeros(10)


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
def test_body_allocation_lifetime(test, device):
    """Callable bodies own persistent backing across replay and either destruction order."""
    with wp.ScopedDevice(device):
        wp.load_module(device=device)
        for branch in ("on_true", "on_false", "while"):
            for release_array_first in (False, True):
                with test.subTest(branch=branch, release_array_first=release_array_first):
                    condition = wp.zeros(1, dtype=wp.int32)
                    result = wp.zeros(4, dtype=wp.float32)
                    holder = {}

                    def body(result=result, holder=holder, condition=condition):
                        temporary = wp.empty_like(result)
                        temporary.fill_(2.0)
                        launch_multiply_by_two(temporary)
                        wp.copy(result, temporary)
                        holder["temporary"] = temporary
                        condition.zero_()

                    with wp.ScopedCapture(force_module_load=False) as capture:
                        if branch == "on_true":
                            wp.capture_if(condition, on_true=body)
                        elif branch == "on_false":
                            wp.capture_if(condition, on_false=body)
                        else:
                            wp.capture_while(condition, while_body=body)
                        if release_array_first:
                            # Drop the array after resuming the parent capture: no free
                            # node may be recorded for the body's persistent backing.
                            holder.clear()
                    for cond in (0, 1, 0, 1):
                        result.fill_(1.0)
                        condition.assign([cond])
                        wp.capture_launch(capture.graph)
                        executed = not cond if branch == "on_false" else bool(cond)
                        np.testing.assert_array_equal(result.numpy(), np.full(4, 4.0 if executed else 1.0))

                    # If the array outlives its graph, its backing must remain usable.
                    del capture
                    wp.synchronize_device()
                    if not release_array_first:
                        holder["temporary"].fill_(3.0)
                        np.testing.assert_array_equal(holder["temporary"].numpy(), np.full(4, 3.0))
                        holder.clear()


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
def test_failed_body_allocation_lifetime(test, device):
    """Callback aborts and failed capture ends preserve live scratch without leaking dropped scratch."""
    with wp.ScopedDevice(device):
        wp.load_module(device=device)
        condition = wp.ones(1, dtype=wp.int32)
        result = wp.zeros(4, dtype=wp.float32)
        for failure in ("callback", "capture_end"):
            for retain_array in (False, True):
                with test.subTest(failure=failure, retain_array=retain_array):
                    wp.synchronize_device()
                    baseline = wp.get_mempool_used_mem_current(device)
                    holder = {}

                    def body(retain_array=retain_array, holder=holder, failure=failure):
                        temporary = wp.empty_like(result)
                        temporary.fill_(2.0)
                        if retain_array:
                            holder["temporary"] = temporary
                        if failure == "callback":
                            raise ValueError("rejected callback")

                    error = ValueError if failure == "callback" else RuntimeError
                    message = "rejected callback" if failure == "callback" else "CUDA graph capture failed"
                    stderr = StdErrCapture()
                    stderr.begin()
                    try:
                        with test.assertRaisesRegex(error, message):
                            with wp.ScopedCapture(
                                force_module_load=False, capture_mode=wp.CaptureMode.THREAD_LOCAL
                            ) as capture:
                                wp.capture_if(condition, on_true=body)
                                # Invalidate only the parent: its conditional body must remain valid during teardown.
                                test.assertFalse(
                                    wp._src.context.runtime.core.wp_cuda_context_synchronize(device.context)
                                )
                    finally:
                        diagnostics = stderr.end()  # Expected capture errors, not invalid graph releases.
                    test.assertNotIn("invalid argument", diagnostics)
                    test.assertFalse(device.is_capturing)
                    # A callback exception may finalize a partial graph. Never replay an aborted capture.
                    del capture
                    if retain_array:
                        holder["temporary"].fill_(3.0)
                        np.testing.assert_array_equal(holder["temporary"].numpy(), np.full(4, 3.0))
                        holder.clear()
                    wp.synchronize_device()
                    test.assertLessEqual(wp.get_mempool_used_mem_current(device), baseline)
                    with wp.ScopedCapture(force_module_load=False) as recovery:
                        result.fill_(5.0)
                    wp.capture_launch(recovery.graph)
                    np.testing.assert_array_equal(result.numpy(), np.full(4, 5.0))


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
def test_error_alloc_if_subgraph(test, device):
    with wp.ScopedDevice(device):
        # capture body subgraph
        with wp.ScopedCapture() as body_capture:
            body_with_alloc()

        cond = wp.ones(1, dtype=wp.int32)
        with test.assertRaisesRegex(
            RuntimeError, r"Child graph contains an unsupported operation \(memory allocation\)"
        ):
            with wp.ScopedCapture():
                wp.capture_if(condition=cond, on_true=body_capture.graph)


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
def test_error_alloc_else_subgraph(test, device):
    with wp.ScopedDevice(device):
        # capture body subgraph
        with wp.ScopedCapture() as body_capture:
            body_with_alloc()

        cond = wp.ones(1, dtype=wp.int32)
        with test.assertRaisesRegex(
            RuntimeError, r"Child graph contains an unsupported operation \(memory allocation\)"
        ):
            with wp.ScopedCapture():
                wp.capture_if(condition=cond, on_false=body_capture.graph)


@unittest.skipUnless(wp.is_conditional_graph_supported(), "Conditional graph nodes not supported")
def test_error_alloc_while_subgraph(test, device):
    with wp.ScopedDevice(device):
        # capture body subgraph
        with wp.ScopedCapture() as body_capture:
            body_with_alloc()

        cond = wp.ones(1, dtype=wp.int32)
        with test.assertRaisesRegex(
            RuntimeError, r"Child graph contains an unsupported operation \(memory allocation\)"
        ):
            with wp.ScopedCapture():
                wp.capture_while(condition=cond, while_body=body_capture.graph)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()
cuda_devices_with_mempool = get_cuda_test_devices_with_mempool()


class TestConditionalCaptures(unittest.TestCase):
    pass


# tests with graph capture
add_function_test(TestConditionalCaptures, "test_if_capture", test_if_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures, "test_if_capture_with_subgraph", test_if_capture_with_subgraph, devices=devices
)
add_function_test(TestConditionalCaptures, "test_if_else_capture", test_if_else_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures,
    "test_if_else_capture_with_subgraph",
    test_if_else_capture_with_subgraph,
    devices=devices,
)
add_function_test(TestConditionalCaptures, "test_else_capture", test_else_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures, "test_else_capture_with_subgraph", test_else_capture_with_subgraph, devices=devices
)
add_function_test(TestConditionalCaptures, "test_while_capture", test_while_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures, "test_while_capture_with_subgraph", test_while_capture_with_subgraph, devices=devices
)
add_function_test(TestConditionalCaptures, "test_complex_capture", test_complex_capture, devices=cuda_devices)
add_function_test(
    TestConditionalCaptures,
    "test_complex_capture_with_subgraphs",
    test_complex_capture_with_subgraphs,
    devices=devices,
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


add_function_test(
    TestConditionalCaptures, "test_graph_debug_dot_print", test_graph_debug_dot_print, devices=cuda_devices
)

add_function_test(
    TestConditionalCaptures,
    "test_free_during_body_capture",
    test_free_during_body_capture,
    devices=cuda_devices_with_mempool,
)

add_function_test(
    TestConditionalCaptures,
    "test_body_allocation_lifetime",
    test_body_allocation_lifetime,
    devices=cuda_devices_with_mempool,
)
add_function_test(
    TestConditionalCaptures,
    "test_failed_body_allocation_lifetime",
    test_failed_body_allocation_lifetime,
    devices=cuda_devices_with_mempool,
)
add_function_test(
    TestConditionalCaptures,
    "test_error_alloc_if_subgraph",
    test_error_alloc_if_subgraph,
    devices=cuda_devices_with_mempool,
)
add_function_test(
    TestConditionalCaptures,
    "test_error_alloc_else_subgraph",
    test_error_alloc_else_subgraph,
    devices=cuda_devices_with_mempool,
)
add_function_test(
    TestConditionalCaptures,
    "test_error_alloc_while_subgraph",
    test_error_alloc_while_subgraph,
    devices=cuda_devices_with_mempool,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
