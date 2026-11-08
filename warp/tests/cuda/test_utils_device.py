# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that ``wp.utils`` helpers run on their arrays' CUDA device, not the current one."""

import gc
import itertools
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

_secondary_ids = itertools.count()


class _SecondaryContext:
    """A second CUDA context on the same GPU, used as a distinct Warp device."""

    def __init__(self, device):
        core = wp._src.context.runtime.core
        self.context = core.wp_cuda_context_create(device.ordinal)
        if not self.context:
            raise RuntimeError(f"Failed to create a CUDA context on {device}")
        # creating a context pushes it; setting the previous one instead would leave an extra stack entry
        core.wp_cuda_context_pop_current()
        # a context left alive by a failed test keeps its alias mapped until it is unmapped here
        self.alias = f"{device.alias}_utils_device_test_{next(_secondary_ids)}"
        self.device = wp.map_cuda_device(self.alias, self.context)

    def close(self):
        wp.synchronize()
        wp.unmap_cuda_device(self.alias)
        # release the device's default stream while its context is still alive
        self.device = None
        gc.collect()
        wp._src.context.runtime.core.wp_cuda_context_destroy(self.context)

    def abandon(self):
        """Unmap the device but keep its context alive, for resources a failed test left behind."""
        wp.unmap_cuda_device(self.alias)
        self.device = None


def _run_beside_and_inside_captures(device, other, make):
    """Call a helper on ``other``'s arrays inside a capture on ``other``, then beside a capture on ``device``.

    ``make(other)`` allocates fresh arrays on ``other`` and returns ``(run, read)``: ``run()`` calls
    the helper, and ``read()`` returns its result as a NumPy array.

    Returns:
        The result before the call, after the capture on ``other`` but before its launch, after
        that launch, and right after the call beside a capture on ``device``. Every array is
        released on return, so the caller can destroy ``other``.
    """
    # Warp sets up a context on its first use, which a capture on another context
    # can't contain, so use ``device`` before capturing
    own = wp.zeros(1, dtype=float, device=device)

    run, read = make(other)
    initial = read()
    # ``device`` stays current, so only the arrays tell the helper to use ``other``
    with wp.ScopedCapture(other, force_module_load=False) as capture:
        with wp.ScopedDevice(device):
            run()
    inside_before_launch = read()
    wp.capture_launch(capture.graph)
    inside_after_launch = read()

    run, read = make(other)
    # in the default thread-local mode, CUDA refuses allocations on other devices during a capture
    with wp.ScopedCapture(device, force_module_load=False, capture_mode=wp.CaptureMode.RELAXED):
        own.zero_()  # an op of the capture's own, so the capture is not empty
        run()
    wp.synchronize_device(other)
    beside = read()

    return initial, inside_before_launch, inside_after_launch, beside


def _check_runs_on_array_device(test, device, make, expected):
    """Check that a helper's call on another device's arrays runs on that device, not the current one.

    A second CUDA context on the same GPU stands in for the other device.
    """
    secondary = _SecondaryContext(device)
    try:
        initial, inside_before_launch, inside_after_launch, beside = _run_beside_and_inside_captures(
            device, secondary.device, make
        )
    except BaseException:
        # a failed capture can leave arrays and graph resources that are freed later, so
        # destroying the context under them would corrupt the process
        secondary.abandon()
        raise
    secondary.close()

    test.assertFalse(np.array_equal(initial, expected))
    np.testing.assert_array_equal(inside_before_launch, initial)
    np.testing.assert_array_equal(inside_after_launch, expected)
    np.testing.assert_array_equal(beside, expected)


def test_array_scan_runs_on_array_device(test, device):
    """Run ``array_scan()`` on its arrays' device."""

    def make(other):
        values = wp.array([1, 2, 3, 4], dtype=int, device=other)
        result = wp.zeros(4, dtype=int, device=other)
        return (lambda: wp.utils.array_scan(values, result)), result.numpy

    _check_runs_on_array_device(test, device, make, np.array([1, 3, 6, 10], dtype=np.int32))


def test_array_sum_runs_on_array_device(test, device):
    """Run ``array_sum()`` on its arrays' device."""

    def make(other):
        values = wp.array([1.0, 2.0, 3.0, 4.0], dtype=float, device=other)
        result = wp.zeros(1, dtype=float, device=other)
        return (lambda: wp.utils.array_sum(values, out=result)), result.numpy

    _check_runs_on_array_device(test, device, make, np.array([10.0], dtype=np.float32))


def test_array_sum_axis_runs_on_array_device(test, device):
    """Run ``array_sum()`` along an axis on its arrays' device."""

    def make(other):
        values = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float, device=other)
        result = wp.zeros((1, 3), dtype=float, device=other)
        return (lambda: wp.utils.array_sum(values, axis=0, out=result)), result.numpy

    _check_runs_on_array_device(test, device, make, np.array([[5.0, 7.0, 9.0]], dtype=np.float32))


def test_array_inner_runs_on_array_device(test, device):
    """Run ``array_inner()`` on its arrays' device."""

    def make(other):
        a = wp.array([1.0, 2.0, 3.0, 4.0], dtype=float, device=other)
        b = wp.array([2.0, 2.0, 2.0, 2.0], dtype=float, device=other)
        result = wp.zeros(1, dtype=float, device=other)
        return (lambda: wp.utils.array_inner(a, b, out=result)), result.numpy

    _check_runs_on_array_device(test, device, make, np.array([20.0], dtype=np.float32))


def test_array_inner_axis_runs_on_array_device(test, device):
    """Run ``array_inner()`` along an axis on its arrays' device."""

    def make(other):
        a = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float, device=other)
        b = wp.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]], dtype=float, device=other)
        result = wp.zeros((1, 3), dtype=float, device=other)
        return (lambda: wp.utils.array_inner(a, b, axis=0, out=result)), result.numpy

    _check_runs_on_array_device(test, device, make, np.array([[6.0, 9.0, 12.0]], dtype=np.float32))


def test_radix_sort_pairs_runs_on_array_device(test, device):
    """Run ``radix_sort_pairs()`` on its arrays' device."""

    def make(other):
        # the sort uses the second half of each array as scratch space
        keys = wp.array([3, 1, 2, 0, 0, 0, 0, 0], dtype=int, device=other)
        values = wp.array([0, 1, 2, 3, 0, 0, 0, 0], dtype=int, device=other)

        def read():
            return np.stack([keys.numpy()[:4], values.numpy()[:4]])

        return (lambda: wp.utils.radix_sort_pairs(keys, values, 4)), read

    _check_runs_on_array_device(test, device, make, np.array([[0, 1, 2, 3], [3, 1, 2, 0]], dtype=np.int32))


def test_segmented_sort_pairs_runs_on_array_device(test, device):
    """Run ``segmented_sort_pairs()`` on its arrays' device."""

    def make(other):
        # the sort uses the second half of each array as scratch space
        keys = wp.array([3, 1, 2, 0, 0, 0, 0, 0], dtype=int, device=other)
        values = wp.array([0, 1, 2, 3, 0, 0, 0, 0], dtype=int, device=other)
        segment_starts = wp.array([0], dtype=int, device=other)
        segment_ends = wp.array([4], dtype=int, device=other)

        def read():
            return np.stack([keys.numpy()[:4], values.numpy()[:4]])

        return (lambda: wp.utils.segmented_sort_pairs(keys, values, 4, segment_starts, segment_ends)), read

    _check_runs_on_array_device(test, device, make, np.array([[0, 1, 2, 3], [3, 1, 2, 0]], dtype=np.int32))


def test_runlength_encode_runs_on_array_device(test, device):
    """Run ``runlength_encode()`` on its arrays' device."""

    def make(other):
        values = wp.array([1, 1, 2, 2, 2, 3], dtype=int, device=other)
        run_values = wp.zeros(6, dtype=int, device=other)
        run_lengths = wp.zeros(6, dtype=int, device=other)
        run_count = wp.zeros(1, dtype=int, device=other)

        def read():
            return np.concatenate([run_count.numpy(), run_values.numpy()[:3], run_lengths.numpy()[:3]])

        return (lambda: wp.utils.runlength_encode(values, run_values, run_lengths, run_count=run_count)), read

    _check_runs_on_array_device(test, device, make, np.array([3, 1, 2, 3, 2, 3, 1], dtype=np.int32))


class TestUtilsDevice(unittest.TestCase):
    pass


# in-capture calls allocate temporary storage from the memory pool
devices = get_selected_cuda_test_devices_with_mempool()

add_function_test(
    TestUtilsDevice, "test_array_scan_runs_on_array_device", test_array_scan_runs_on_array_device, devices=devices
)
add_function_test(
    TestUtilsDevice, "test_array_sum_runs_on_array_device", test_array_sum_runs_on_array_device, devices=devices
)
add_function_test(
    TestUtilsDevice,
    "test_array_sum_axis_runs_on_array_device",
    test_array_sum_axis_runs_on_array_device,
    devices=devices,
)
add_function_test(
    TestUtilsDevice, "test_array_inner_runs_on_array_device", test_array_inner_runs_on_array_device, devices=devices
)
add_function_test(
    TestUtilsDevice,
    "test_array_inner_axis_runs_on_array_device",
    test_array_inner_axis_runs_on_array_device,
    devices=devices,
)
add_function_test(
    TestUtilsDevice,
    "test_radix_sort_pairs_runs_on_array_device",
    test_radix_sort_pairs_runs_on_array_device,
    devices=devices,
)
add_function_test(
    TestUtilsDevice,
    "test_segmented_sort_pairs_runs_on_array_device",
    test_segmented_sort_pairs_runs_on_array_device,
    devices=devices,
)
add_function_test(
    TestUtilsDevice,
    "test_runlength_encode_runs_on_array_device",
    test_runlength_encode_runs_on_array_device,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
