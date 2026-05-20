# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``wp.array.fill_`` under multi-stream graph capture.

Targets a composability bug. Before the fix, ``wp_array_fill_device``
staged the fill scalar through ``capturable_tmp_alloc``, which itself relied on
``cudaStreamEndCapture`` followed by ``cudaStreamBeginCaptureToGraph`` to pause and
resume the capture so the host-to-device staging memcpy was not recorded into the
graph. CUDA only allows ``cudaStreamEndCapture`` on the *begin* stream of a
capture, so as soon as the fill ran on a forked stream the pause call returned
``cudaErrorStreamCaptureWrongThread (903)``, the capture was invalidated, and
``capture_end`` raised.

The fix passes the fill bytes inline through CUDA kernel arguments via bucketed
``FillValue<N>`` PODs; no host->device staging, no pause/resume. Fill values that
exceed the largest inline buffer fall back to the legacy
``capturable_tmp_alloc`` + ``_devptr`` kernel path.
"""

import ctypes
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# Larger than the largest inline ``FillValue<N>`` bucket (3968 bytes) so
# ``wp.array.fill_`` is forced through the legacy ``capturable_tmp_alloc`` +
# ``_devptr`` kernel fallback path.
@wp.struct
class BigFillStruct:
    pad: wp.types.vector(1000, wp.float32)  # 4000 bytes


class TestArrayFillCapture(unittest.TestCase):
    pass


def test_array_fill_forked_stream_capture(test, device):
    """``arr.fill_`` on a forked stream must not invalidate a captured graph."""
    with wp.ScopedDevice(device):
        wp.synchronize_device()

        outer = wp.Stream(device)
        inner = wp.Stream(device)

        # A non-contiguous slice forces ``arr.fill_`` to route through
        # ``wp_array_fill_device`` on the device side. The contiguous fast path
        # uses ``device.memtile`` (out of scope for this test).
        arr = wp.zeros((4, 8), dtype=wp.float32, device=device)
        view = arr[:, ::2]
        test.assertFalse(view.is_contiguous)

        with wp.ScopedCapture(device=device, stream=outer, force_module_load=False) as capture:
            # Hand-fork ``inner`` into ``outer``'s capture: record a captured
            # event on ``outer``, then have ``inner`` wait on it. Under default
            # ThreadLocal this is the standard CUDA fork-by-event-wait pattern
            # and does not require Relaxed.
            fork_event = wp.Event(device)
            outer.record_event(fork_event)
            inner.wait_event(fork_event)

            # ``sync_enter=False`` keeps ``wp_cuda_context_set_stream`` out of its
            # cached-event ``cuStreamWaitEvent`` branch, which would otherwise
            # raise ``cudaErrorStreamCaptureIsolation (905)`` here.
            with wp.ScopedStream(inner, sync_enter=False):
                view.fill_(7.0)

            # Join ``inner`` back into ``outer`` before the capture closes.
            join_event = wp.Event(device)
            inner.record_event(join_event)
            outer.wait_event(join_event)

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        np.testing.assert_array_equal(view.numpy(), np.full(view.shape, 7.0, dtype=np.float32))


def test_contiguous_vec3_fill_forked_stream_capture(test, device):
    """Contiguous fills through ``device.memtile`` must also avoid pause/resume capture."""
    with wp.ScopedDevice(device):
        wp.synchronize_device()

        outer = wp.Stream(device)
        inner = wp.Stream(device)
        arr = wp.zeros(8, dtype=wp.vec3, device=device)

        with wp.ScopedCapture(device=device, stream=outer, force_module_load=False) as capture:
            fork_event = wp.Event(device)
            outer.record_event(fork_event)
            inner.wait_event(fork_event)

            with wp.ScopedStream(inner, sync_enter=False):
                arr.fill_(wp.vec3(1.0, 2.0, 3.0))

            join_event = wp.Event(device)
            inner.record_event(join_event)
            outer.wait_event(join_event)

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        expected = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (arr.shape[0], 1))
        np.testing.assert_array_equal(arr.numpy(), expected)


def test_array_fill_oversized_value_fallback(test, device):
    """Fill values larger than the largest inline bucket use the ``_devptr`` fallback path correctly.

    User struct dtypes containing very large vector/matrix members (or several
    ``wp.array`` fields) can exceed the inline kernel-arg buckets. For those,
    ``wp_array_fill_device`` falls back to ``capturable_tmp_alloc`` plus the
    legacy ``_devptr`` kernels. This test exercises the fallback by filling a
    non-contiguous slice of a ``BigFillStruct`` array and checks the fill
    propagated to every element.
    """
    s = BigFillStruct()
    for i in range(1000):
        s.pad[i] = float(i + 1)

    cvalue = s.__ctype__()
    test.assertGreater(
        ctypes.sizeof(cvalue), 3968, "test must exercise the >3968B fallback path; struct layout changed"
    )

    with wp.ScopedDevice(device):
        wp.synchronize_device()

        arr = wp.zeros((4, 8), dtype=BigFillStruct, device=device)
        view = arr[:, ::2]  # non-contiguous -> wp_array_fill_device
        test.assertFalse(view.is_contiguous)

        view.fill_(s)
        wp.synchronize_device()

        nptype = BigFillStruct.numpy_dtype()
        ns = s.numpy_value()
        expected = np.empty(view.shape, dtype=nptype)
        expected.fill(ns)
        np.testing.assert_array_equal(view.numpy(), expected)


devices = get_selected_cuda_test_devices()

# ``check_output=False`` because the failing path on origin/main emits CUDA
# error 903 + cascading 901s on stderr. Once the fix is in place those should
# no longer appear.
add_function_test(
    TestArrayFillCapture,
    "test_array_fill_forked_stream_capture",
    test_array_fill_forked_stream_capture,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestArrayFillCapture,
    "test_contiguous_vec3_fill_forked_stream_capture",
    test_contiguous_vec3_fill_forked_stream_capture,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestArrayFillCapture,
    "test_array_fill_oversized_value_fallback",
    test_array_fill_oversized_value_fallback,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
