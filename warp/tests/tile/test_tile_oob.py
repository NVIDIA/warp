# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import signal
import subprocess
import sys
import unittest

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices

# The OOB bounds checks only fire in debug mode, so compile this whole module
# in debug. Subprocesses then amortize compilation through the normal kernel cache.
wp.set_module_options({"mode": "debug"})


def _run_in_subprocess(func_name: str, device, timeout: int = 60):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import warp.tests.tile.test_tile_oob as mod; getattr(mod, sys.argv[1])(sys.argv[2])"),
            func_name,
            str(device),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _assert_aborted(test_case, returncode):
    if sys.platform == "win32":
        test_case.assertNotEqual(returncode, 0)
    else:
        test_case.assertIn(returncode, [-signal.SIGABRT, -signal.SIGILL, -signal.SIGTRAP])


@wp.kernel
def shared_tile_oob_kernel(out: wp.array[int]):
    i = wp.tid()

    t = wp.tile_zeros(shape=1, dtype=int, storage="shared")
    out[0] = wp.tile_extract(t, i)


@wp.kernel
def shared_tile_oob_tiled_kernel(out: wp.array[int]):
    _, lane = wp.tid()

    t = wp.tile_zeros(shape=1, dtype=int, storage="shared")
    out[0] = wp.tile_extract(t, lane)


@wp.kernel
def shared_tile_negative_oob_kernel(out: wp.array[int]):
    t = wp.tile_zeros(shape=1, dtype=int, storage="shared")
    out[0] = wp.tile_extract(t, -1)


@wp.kernel
def shared_tile_2d_oob_kernel(out: wp.array[int]):
    t = wp.tile_zeros(shape=(2, 2), dtype=int, storage="shared")
    out[0] = wp.tile_extract(t, 0, 2)


# Public wp.tile_extract() retags tile arguments to shared storage during type
# resolution, so use a native snippet to exercise register layout bounds directly.
@wp.func_native(
    """
    auto val = tile_extract(t, i);
    out[0] = val;
    """
)
def native_register_tile_extract(t: wp.tile[int, 1], i: int, out: wp.array[int]): ...


@wp.kernel
def register_tile_oob_kernel(out: wp.array[int]):
    i = wp.tid()

    t = wp.tile_zeros(shape=1, dtype=int, storage="register")
    native_register_tile_extract(t, i, out)


def _trigger_shared_tile_oob(device):
    wp.config.quiet = True

    with wp.ScopedDevice(device):
        wp.launch(shared_tile_oob_kernel, dim=2, inputs=[wp.zeros(1, dtype=int, device=device)], device=device)


def _trigger_shared_tile_oob_cuda(device):
    wp.config.quiet = True

    with wp.ScopedDevice(device):
        out = wp.zeros(1, dtype=int, device=device)
        wp.launch_tiled(shared_tile_oob_tiled_kernel, dim=[1], inputs=[out], block_dim=2, device=device)
        wp.synchronize_device(device)


def _trigger_shared_tile_negative_oob(device):
    wp.config.quiet = True

    with wp.ScopedDevice(device):
        wp.launch(shared_tile_negative_oob_kernel, dim=1, inputs=[wp.zeros(1, dtype=int, device=device)], device=device)


def _trigger_shared_tile_2d_oob(device):
    wp.config.quiet = True

    with wp.ScopedDevice(device):
        wp.launch(shared_tile_2d_oob_kernel, dim=1, inputs=[wp.zeros(1, dtype=int, device=device)], device=device)


def _trigger_register_tile_oob(device):
    wp.config.quiet = True

    with wp.ScopedDevice(device):
        wp.launch(register_tile_oob_kernel, dim=2, inputs=[wp.zeros(1, dtype=int, device=device)], device=device)


def test_cuda_shared_tile_oob_reports_tile_index(test, device):
    # Warp reports this CUDA error as output in the subprocess, so assert
    # on the diagnostic text instead of the process exit status.
    _returncode, stdout, stderr = _run_in_subprocess("_trigger_shared_tile_oob_cuda", device)

    output = stdout + stderr
    test.assertRegex(output, r"Warp tile index out of bounds in shared tile")
    test.assertRegex(output, r"coordinate dimension 0 has index 1, outside valid range \[0, 1\)")
    test.assertRegex(output, r"device-side assert triggered")


# The CPU OOB checks abort the process, so run them as fixed-device tests
# against "cpu" rather than parameterizing over all test devices.
@unittest.skipIf(
    sys.platform == "win32",
    "Tile OOB tests intentionally trigger device-side asserts and host aborts. On the "
    "Windows release-qualification rig (debug driver with bsod-on-release-assert and a "
    "short TDR) the CUDA device-side assert escalates to a kernel bugcheck "
    "(0x116 VIDEO_TDR_FAILURE), crashing the machine. Covered on Linux.",
)
class TestTileOOB(unittest.TestCase):
    def test_shared_tile_oob_reports_tile_index(self):
        returncode, stdout, stderr = _run_in_subprocess("_trigger_shared_tile_oob", "cpu")
        _assert_aborted(self, returncode)

        output = stdout + stderr
        self.assertRegex(output, r"Warp tile index out of bounds in shared tile")
        self.assertRegex(output, r"coordinate dimension 0 has index 1, outside valid range \[0, 1\)")

    def test_shared_tile_negative_oob_reports_tile_index(self):
        returncode, stdout, stderr = _run_in_subprocess("_trigger_shared_tile_negative_oob", "cpu")
        _assert_aborted(self, returncode)

        output = stdout + stderr
        self.assertRegex(output, r"Warp tile index out of bounds in shared tile")
        self.assertRegex(output, r"coordinate dimension 0 has index -1, outside valid range \[0, 1\)")

    def test_register_tile_oob_reports_tile_index(self):
        returncode, stdout, stderr = _run_in_subprocess("_trigger_register_tile_oob", "cpu")
        _assert_aborted(self, returncode)

        output = stdout + stderr
        self.assertRegex(output, r"Warp tile index out of bounds in register tile")
        self.assertRegex(output, r"coordinate dimension 0 has index 1, outside valid range \[0, 1\)")

    def test_shared_tile_2d_oob_reports_dimension(self):
        returncode, stdout, stderr = _run_in_subprocess("_trigger_shared_tile_2d_oob", "cpu")
        _assert_aborted(self, returncode)

        output = stdout + stderr
        self.assertRegex(output, r"Warp tile index out of bounds in shared tile")
        self.assertRegex(output, r"coordinate dimension 1 has index 2, outside valid range \[0, 2\)")


cuda_devices = get_cuda_test_devices()

add_function_test(
    TestTileOOB,
    "test_cuda_shared_tile_oob_reports_tile_index",
    test_cuda_shared_tile_oob_reports_tile_index,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
