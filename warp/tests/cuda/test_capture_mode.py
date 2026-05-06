# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``capture_mode`` parameter of :class:`wp.ScopedCapture`.

The CUDA driver rejects most runtime APIs while a stream is being captured
under ``cudaStreamCaptureModeGlobal`` / ``cudaStreamCaptureModeThreadLocal``
and invalidates the capture on the first such call.
``cudaStreamCaptureModeRelaxed`` tolerates those calls, which lets Warp
compose with libraries that still perform lazy / capture-unsafe CUDA
runtime calls (e.g. ``cudaFree(0)`` during lazy context / allocator init)
inside a Warp graph capture.

The tests below exercise both directions using ``cudaFree(0)`` as a minimal
"capture-unsafe" runtime call:

* ``test_relaxed_allows_capture_unsafe_runtime_call``:
  ``cudaFree(0)`` succeeds under ``CaptureMode.RELAXED`` and the capture
  ends cleanly with a non-null graph.
* ``test_thread_local_rejects_capture_unsafe_runtime_call``:
  ``cudaFree(0)`` is rejected under ``CaptureMode.THREAD_LOCAL`` and
  ``wp.capture_end`` raises because the capture was invalidated.
"""

import ctypes
import ctypes.util
import unittest

import warp as wp
from warp.tests.unittest_utils import *


def _load_cudart():
    """Load the CUDA runtime library and bind ``cudaFree``.

    Returns ``None`` if the library cannot be located, in which case the
    tests below will skip (see :func:`unittest.TestCase.skipTest`).
    """
    name = ctypes.util.find_library("cudart")
    if name is None:
        return None
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        return None
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    return lib


CUDART = _load_cudart()


class TestCaptureMode(unittest.TestCase):
    pass


def test_relaxed_allows_capture_unsafe_runtime_call(test, device):
    """``CaptureMode.RELAXED`` tolerates ``cudaFree(0)`` during capture."""
    if CUDART is None:
        test.skipTest("libcudart not available")

    with wp.ScopedDevice(device):
        # Make sure the CUDA context is fully initialized before the capture
        # opens, so cudaFree(0) exercises the runtime path under capture
        # rather than lazy context init.
        wp.synchronize_device()

        with wp.ScopedCapture(capture_mode=wp.CaptureMode.RELAXED, force_module_load=False) as capture:
            err = CUDART.cudaFree(0)
            test.assertEqual(err, 0, f"cudaFree(0) failed with CUDA error {err} under RELAXED capture")

        test.assertIsNotNone(capture.graph)


def test_thread_local_rejects_capture_unsafe_runtime_call(test, device):
    """``CaptureMode.THREAD_LOCAL`` rejects ``cudaFree(0)`` during capture.

    The rejection invalidates the capture, so ``wp.capture_end`` (called
    from ``ScopedCapture.__exit__``) raises a ``RuntimeError``. We catch it
    to confirm that the stricter mode is indeed doing its job.
    """
    if CUDART is None:
        test.skipTest("libcudart not available")

    with wp.ScopedDevice(device):
        wp.synchronize_device()

        with test.assertRaisesRegex(RuntimeError, "CUDA graph capture failed"):
            with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL, force_module_load=False):
                err = CUDART.cudaFree(0)
                # cudaFree during a strict capture returns a non-zero CUDA
                # error (typically cudaErrorStreamCaptureUnsupported = 900).
                test.assertNotEqual(err, 0, "cudaFree(0) unexpectedly succeeded under THREAD_LOCAL capture")


devices = get_selected_cuda_test_devices()

# check_output is disabled on the ThreadLocal test because the CUDA runtime
# may emit a diagnostic line when it rejects cudaFree(0) during capture.
add_function_test(
    TestCaptureMode,
    "test_relaxed_allows_capture_unsafe_runtime_call",
    test_relaxed_allows_capture_unsafe_runtime_call,
    devices=devices,
)
add_function_test(
    TestCaptureMode,
    "test_thread_local_rejects_capture_unsafe_runtime_call",
    test_thread_local_rejects_capture_unsafe_runtime_call,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
