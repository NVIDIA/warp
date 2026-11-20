# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import sys
import unittest
import weakref
from typing import ClassVar

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices


class RecordingArray(wp.array):
    calls: ClassVar[list] = []

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        result = func(*args, **({} if kwargs is None else kwargs))
        cls.calls.append(func)
        return result


@wp.kernel
def square(values: wp.array[float], result: wp.array[float]):
    i = wp.tid()
    result[i] = values[i] * values[i]


@wp.kernel
def increment(values: wp.array[float]):
    i = wp.tid()
    values[i] += 1.0


def test_tape_delegation(test, device):
    values = RecordingArray([2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    result = wp.zeros(2, dtype=wp.float32, device=device, requires_grad=True)
    RecordingArray.calls = []
    tape = wp.Tape()
    with tape:
        call_line = sys._getframe().f_lineno + 1
        wp.launch(square, 2, inputs=[values], outputs=[result], device=device)

    test.assertEqual(RecordingArray.calls, [wp.launch])
    test.assertEqual(len(tape.launches), 1)
    test.assertIs(tape.launches[0][3][0], values)
    test.assertEqual(
        tape.launches[0][7]["caller"],
        {"file": __file__, "lineno": call_line, "func": "test_tape_delegation"},
    )
    tape.backward(grads={result: wp.ones_like(result)})
    np.testing.assert_allclose(values.grad.numpy(), [4.0, 6.0])
    np.testing.assert_allclose(result.numpy(), [4.0, 9.0])


def test_plain_tape_callsite(test, device):
    values = wp.ones(2, dtype=wp.float32, device=device, requires_grad=True)
    result = wp.zeros_like(values)
    tape = wp.Tape()
    with tape:
        call_line = sys._getframe().f_lineno + 1
        wp.launch(square, 2, inputs=[values], outputs=[result], device=device)
    test.assertEqual(
        tape.launches[0][7]["caller"],
        {"file": __file__, "lineno": call_line, "func": "test_plain_tape_callsite"},
    )
    np.testing.assert_allclose(result.numpy(), [1.0, 1.0])


def test_capture_delegation(test, device):
    values = RecordingArray([0.0, 0.0], dtype=wp.float32, device=device)
    alias = values.view(wp.float32)
    wp.load_module(module=increment.module, device=device)
    RecordingArray.calls = []
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(increment, 2, inputs=[values], device=device)
    test.assertEqual(RecordingArray.calls, [wp.launch])

    values.zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(alias.numpy(), [1.0, 1.0])
    test.assertEqual(RecordingArray.calls, [wp.launch])


def test_utils_capture_delegation(test, device):
    scan_values = RecordingArray([1, 2, 3], dtype=wp.int32, device=device)
    scan_out = wp.zeros(3, dtype=wp.int32, device=device)
    sum_values = RecordingArray([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    sum_out = wp.zeros(1, dtype=wp.float32, device=device)
    inner_b = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=device)
    inner_out = wp.zeros(1, dtype=wp.float32, device=device)
    sort_keys = RecordingArray([3, 1, 2, 0, 0, 0], dtype=wp.int32, device=device)
    sort_values = wp.array([30, 10, 20, 0, 0, 0], dtype=wp.int32, device=device)
    segment_keys = RecordingArray([3, 1, 4, 2, 0, 0, 0, 0], dtype=wp.int32, device=device)
    segment_values = wp.array([30, 10, 40, 20, 0, 0, 0, 0], dtype=wp.int32, device=device)
    segment_offsets = wp.array([0, 2, 4], dtype=wp.int32, device=device)
    encode_values = RecordingArray([1, 1, 2, 2, 2, 3], dtype=wp.int32, device=device)
    run_values = wp.zeros(6, dtype=wp.int32, device=device)
    run_lengths = wp.zeros(6, dtype=wp.int32, device=device)
    run_count = wp.zeros(1, dtype=wp.int32, device=device)
    expected_calls = [
        wp.utils.array_scan,
        wp.utils.array_sum,
        wp.utils.array_inner,
        wp.utils.radix_sort_pairs,
        wp.utils.segmented_sort_pairs,
        wp.utils.runlength_encode,
    ]

    RecordingArray.calls = []
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        test.assertIsNone(wp.utils.array_scan(scan_values, scan_out))
        test.assertIs(wp.utils.array_sum(sum_values, out=sum_out), sum_out)
        test.assertIs(wp.utils.array_inner(sum_values, inner_b, out=inner_out), inner_out)
        test.assertIsNone(wp.utils.radix_sort_pairs(sort_keys, sort_values, 3))
        test.assertIsNone(wp.utils.segmented_sort_pairs(segment_keys, segment_values, 4, segment_offsets))
        test.assertIs(wp.utils.runlength_encode(encode_values, run_values, run_lengths, run_count=run_count), run_count)
    test.assertEqual(RecordingArray.calls, expected_calls)

    # CUDA capture executes these operations while recording. Reset their
    # outputs and sort inputs so the checks require replay to execute them.
    for output in (scan_out, sum_out, inner_out, run_values, run_lengths, run_count):
        output.zero_()
    sort_keys.view(sort_keys.dtype).assign(np.array([3, 1, 2, 0, 0, 0], dtype=np.int32))
    sort_values.assign(np.array([30, 10, 20, 0, 0, 0], dtype=np.int32))
    segment_keys.view(segment_keys.dtype).assign(np.array([3, 1, 4, 2, 0, 0, 0, 0], dtype=np.int32))
    segment_values.assign(np.array([30, 10, 40, 20, 0, 0, 0, 0], dtype=np.int32))

    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(scan_out.numpy(), [1, 3, 6])
    np.testing.assert_allclose(sum_out.numpy(), [6.0])
    np.testing.assert_allclose(inner_out.numpy(), [32.0])
    np.testing.assert_array_equal(sort_keys.view(sort_keys.dtype).numpy()[:3], [1, 2, 3])
    np.testing.assert_array_equal(sort_values.numpy()[:3], [10, 20, 30])
    np.testing.assert_array_equal(segment_keys.view(segment_keys.dtype).numpy()[:4], [1, 3, 2, 4])
    np.testing.assert_array_equal(segment_values.numpy()[:4], [10, 30, 20, 40])
    np.testing.assert_array_equal(run_values.numpy()[:3], [1, 2, 3])
    np.testing.assert_array_equal(run_lengths.numpy()[:3], [2, 3, 1])
    np.testing.assert_array_equal(run_count.numpy(), [3])
    test.assertEqual(RecordingArray.calls, expected_calls)


def test_config_disabled_tape_callsite(test, device):
    wp.config.enable_array_function_overrides = False
    values = wp.array([2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    result = wp.zeros_like(values)
    tape = wp.Tape()
    with tape:
        call_line = sys._getframe().f_lineno + 1
        wp.launch(square, 2, inputs=[values], outputs=[result], device=device)

    test.assertEqual(
        tape.launches[0][7]["caller"],
        {"file": __file__, "lineno": call_line, "func": "test_config_disabled_tape_callsite"},
    )
    tape.backward(grads={result: wp.ones_like(result)})
    np.testing.assert_allclose(values.grad.numpy(), [4.0, 6.0])
    np.testing.assert_allclose(result.numpy(), [4.0, 9.0])


def test_config_enabled_after_command_recording(test, device):
    wp.config.enable_array_function_overrides = False
    RecordingArray.calls = []
    values = RecordingArray([0.0, 0.0], dtype=wp.float32, device=device)
    command = wp.launch(increment, 2, inputs=[values], device=device, record_cmd=True)
    test.assertEqual(RecordingArray.calls, [])

    wp.config.enable_array_function_overrides = True
    command.launch()
    command.launch()
    np.testing.assert_allclose(values.view(values.dtype).numpy(), [2.0, 2.0])
    test.assertEqual(RecordingArray.calls, [])


def test_config_enabled_after_graph_capture(test, device):
    wp.config.enable_array_function_overrides = False
    RecordingArray.calls = []
    values = RecordingArray([0.0, 0.0], dtype=wp.float32, device=device)
    wp.load_module(module=increment.module, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(increment, 2, inputs=[values], device=device)
    values.zero_()
    test.assertEqual(RecordingArray.calls, [])

    wp.config.enable_array_function_overrides = True
    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(values.view(values.dtype).numpy(), [2.0, 2.0])
    test.assertEqual(RecordingArray.calls, [])


def test_recorded_command_delegation(test, device):
    values = RecordingArray([0.0, 0.0], dtype=wp.float32, device=device)
    owner = weakref.ref(values)
    RecordingArray.calls = []
    command = wp.launch(increment, 2, inputs=[values], device=device, record_cmd=True)
    test.assertEqual(RecordingArray.calls, [wp.launch])
    np.testing.assert_allclose(values.view(values.dtype).numpy(), [0.0, 0.0])
    del values
    gc.collect()
    test.assertIsNotNone(owner())
    command.launch()
    command.launch()
    np.testing.assert_allclose(owner().view(wp.float32).numpy(), [2.0, 2.0])
    test.assertEqual(RecordingArray.calls, [wp.launch])


class TestFunctionDispatchRecording(unittest.TestCase):
    def setUp(self):
        self.addCleanup(
            setattr, wp.config, "enable_array_function_overrides", wp.config.enable_array_function_overrides
        )
        wp.config.enable_array_function_overrides = True


devices = get_test_devices()
add_function_test(TestFunctionDispatchRecording, "test_tape_delegation", test_tape_delegation, devices=devices)
add_function_test(TestFunctionDispatchRecording, "test_plain_tape_callsite", test_plain_tape_callsite, devices=devices)
add_function_test(
    TestFunctionDispatchRecording,
    "test_config_disabled_tape_callsite",
    test_config_disabled_tape_callsite,
    devices=devices,
)
add_function_test(
    TestFunctionDispatchRecording,
    "test_config_enabled_after_command_recording",
    test_config_enabled_after_command_recording,
    devices=devices,
)
add_function_test(
    TestFunctionDispatchRecording,
    "test_config_enabled_after_graph_capture",
    test_config_enabled_after_graph_capture,
    devices=devices,
)
add_function_test(TestFunctionDispatchRecording, "test_capture_delegation", test_capture_delegation, devices=devices)
add_function_test(
    TestFunctionDispatchRecording, "test_utils_capture_delegation", test_utils_capture_delegation, devices=devices
)
add_function_test(
    TestFunctionDispatchRecording, "test_recorded_command_delegation", test_recorded_command_delegation, devices=devices
)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
