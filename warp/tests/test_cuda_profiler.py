# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CUDA profiler control API (:func:`warp.cuda_profiler_start`,
:func:`warp.cuda_profiler_stop`, and :class:`warp.ScopedCudaProfiler`)."""

import unittest
from unittest import mock

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Initialize the runtime so that ``runtime.core`` (patched by the mock tests below)
# is available even on CPU-only builds.
devices = get_test_devices()


@wp.kernel
def inc_one_kernel(a: wp.array[float]):
    """Increment each element of ``a`` by one (used to give the smoke test work to launch)."""
    i = wp.tid()
    a[i] = a[i] + 1.0


def _core():
    """Return the native core library handle holding the profiler entry points."""
    return wp._src.context.runtime.core


class TestCudaProfiler(unittest.TestCase):
    """Tests for the CUDA profiler control wrappers.

    The wrappers are thin passthroughs to ``cuProfilerStart``/``cuProfilerStop`` with
    no observable in-process state, so these tests verify the Python-level contract (that
    the native entry points are invoked correctly) rather than that profiler collection
    actually toggled, which can only be observed by an external profiler such as Nsight
    Systems.
    """

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_start_stop_invoke_core(self):
        """``cuda_profiler_start``/``cuda_profiler_stop`` each call their native entry point once,
        forwarding the selected device's CUDA context."""
        core = _core()
        device = wp.get_device()
        with (
            mock.patch.object(core, "wp_cuda_profiler_start") as start,
            mock.patch.object(core, "wp_cuda_profiler_stop") as stop,
        ):
            wp.cuda_profiler_start(device=device)
            wp.cuda_profiler_stop(device=device)
        start.assert_called_once_with(device.context)
        stop.assert_called_once_with(device.context)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_scoped_capture_invokes_start_then_stop(self):
        """``ScopedCudaProfiler`` starts on entry and stops on exit, forwarding the resolved
        device's CUDA context to both calls."""
        core = _core()
        device = wp.get_device()
        with (
            mock.patch.object(core, "wp_cuda_profiler_start") as start,
            mock.patch.object(core, "wp_cuda_profiler_stop") as stop,
        ):
            with wp.ScopedCudaProfiler(device=device):
                # start runs on entry, stop is deferred until exit
                start.assert_called_once_with(device.context)
                stop.assert_not_called()
            stop.assert_called_once_with(device.context)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_scoped_capture_stops_on_exception(self):
        """``ScopedCudaProfiler`` still stops profiling when the body raises."""
        core = _core()
        device = wp.get_device()
        with (
            mock.patch.object(core, "wp_cuda_profiler_start"),
            mock.patch.object(core, "wp_cuda_profiler_stop") as stop,
        ):
            with self.assertRaises(ValueError):
                with wp.ScopedCudaProfiler(device=device):
                    raise ValueError("boom")
            # the finally block must still stop profiling when the body raises
            stop.assert_called_once_with(device.context)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_scoped_capture_preserves_body_exception(self):
        """A stop failure does not mask an exception raised inside the captured region."""
        device = wp.get_device()
        with (
            mock.patch.object(wp, "cuda_profiler_start"),
            mock.patch.object(wp, "cuda_profiler_stop", side_effect=RuntimeError("stop failed")),
        ):
            with self.assertRaisesRegex(ValueError, "body failed"):
                with wp.ScopedCudaProfiler(device=device):
                    raise ValueError("body failed")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_scoped_capture_raises_stop_exception(self):
        """A stop failure is raised when the captured region completes successfully."""
        device = wp.get_device()
        with (
            mock.patch.object(wp, "cuda_profiler_start"),
            mock.patch.object(wp, "cuda_profiler_stop", side_effect=RuntimeError("stop failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                with wp.ScopedCudaProfiler(device=device):
                    pass

    def test_cpu_device_is_noop(self):
        """Passing a non-CUDA device is a no-op: the native entry points are never invoked
        and no error is raised (a CPU device has no CUDA context to profile)."""
        core = _core()
        with (
            mock.patch.object(core, "wp_cuda_profiler_start") as start,
            mock.patch.object(core, "wp_cuda_profiler_stop") as stop,
        ):
            wp.cuda_profiler_start(device="cpu")
            wp.cuda_profiler_stop(device="cpu")
            with wp.ScopedCudaProfiler(device="cpu"):
                pass
        start.assert_not_called()
        stop.assert_not_called()

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_smoke_on_cuda(self):
        """Exercise the real native binding end-to-end on a CUDA device.

        Confirms the ctypes signature is valid and the calls do not raise. Whether
        profiler collection actually toggled is not asserted (see class docstring).
        """
        device = wp.get_cuda_device()
        a = wp.zeros(16, dtype=float, device=device)

        wp.cuda_profiler_start(device=device)
        wp.launch(inc_one_kernel, dim=16, inputs=[a], device=device)
        wp.cuda_profiler_stop(device=device)

        with wp.ScopedCudaProfiler(device=device):
            wp.launch(inc_one_kernel, dim=16, inputs=[a], device=device)

        assert_np_equal(a.numpy(), np.full(16, 2.0, dtype=np.float32))

    @unittest.skipUnless(not wp._src.context.runtime.is_cuda_enabled, "Requires a CPU-only build")
    def test_noop_on_cpu_build(self):
        """Exercise the real bindings on a CPU-only build to confirm they are safe no-ops.

        On a non-CUDA build the native entry points are empty stubs. Calling them without
        mocks catches a regression in the CPU stub or ctypes binding without needing a GPU.
        """
        wp.cuda_profiler_start()
        wp.cuda_profiler_stop()
        with wp.ScopedCudaProfiler():
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
