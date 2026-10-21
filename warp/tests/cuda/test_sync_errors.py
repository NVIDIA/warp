# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, run_python_subprocess


# Delay the trap so the host can enqueue follow-up work, such as an event
# record, before the CUDA context becomes unusable.
@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    const long long start = clock64();
    while (clock64() - start < 100000000) {
    }
    __trap();
#endif
    """
)
def delayed_trap(): ...


@wp.kernel
def trap_kernel():
    delayed_trap()


def _trigger_sync_error(case: str, device_alias: str):
    """Launch a trapping kernel and print the exception that ``case`` raises."""
    device = wp.get_device(device_alias)
    event = wp.Event(device)

    if case == "verify_cuda":
        wp.config.verify_cuda = True

    try:
        if case == "scoped_timer":
            with wp.ScopedTimer("trap", print=False, synchronize=True):
                wp.launch(trap_kernel, dim=1, device=device)
        elif case == "scoped_timer_body_error":
            with wp.ScopedTimer("trap", print=False, synchronize=True):
                wp.launch(trap_kernel, dim=1, device=device)
                raise ValueError("error raised inside the timed block")
        elif case == "timing_end":
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
            wp.launch(trap_kernel, dim=1, device=device)
            try:
                wp.timing_end()
            finally:
                pending = wp._src.context.runtime.core.wp_cuda_timing_get_result_count()
                print(f"pending timing results: {pending}", flush=True)
        else:
            wp.launch(trap_kernel, dim=1, device=device)

            if case == "synchronize":
                wp.synchronize()
            elif case == "synchronize_device":
                wp.synchronize_device(device)
            elif case == "synchronize_stream":
                wp.synchronize_stream(device.stream)
            elif case == "synchronize_event":
                wp.record_event(event)
                wp.synchronize_event(event)
            elif case != "verify_cuda":
                raise ValueError(f"Unknown case '{case}'")
    except Exception as error:
        print(f"{type(error).__name__}: {error}", flush=True)
    else:
        print("no error", flush=True)


def _run_sync_error_subprocess(case: str, device):
    """Run a sync-error case in a child process, since the trap leaves the CUDA context unusable."""
    result = run_python_subprocess(
        "import sys; from warp.tests.cuda.test_sync_errors import _trigger_sync_error; "
        "_trigger_sync_error(sys.argv[1], sys.argv[2])",
        case,
        device.alias,
        timeout=120,
    )
    return result.stdout


def make_test_sync_api_raises(case: str):
    def test_sync_api_raises(test, device):
        stdout = _run_sync_error_subprocess(case, device)
        test.assertRegex(stdout, r"RuntimeError: .*Warp CUDA error 719")

    return test_sync_api_raises


def test_scoped_timer_preserves_body_error(test, device):
    stdout = _run_sync_error_subprocess("scoped_timer_body_error", device)
    test.assertRegex(stdout, r"ValueError: error raised inside the timed block")


def test_timing_end_ends_timing_on_error(test, device):
    stdout = _run_sync_error_subprocess("timing_end", device)
    test.assertRegex(stdout, r"RuntimeError: .*Warp CUDA error 719")
    test.assertIn("pending timing results: 0", stdout)


@unittest.skipIf(
    sys.platform == "win32",
    "The intentional CUDA trap may destabilize Windows release-qualification systems.",
)
class TestSyncErrors(unittest.TestCase):
    pass


devices = get_cuda_test_devices(mode="basic")

for case in (
    "synchronize",
    "synchronize_device",
    "synchronize_stream",
    "synchronize_event",
    "scoped_timer",
    "verify_cuda",
):
    add_function_test(TestSyncErrors, f"test_{case}_raises", make_test_sync_api_raises(case), devices=devices)

add_function_test(
    TestSyncErrors,
    "test_scoped_timer_preserves_body_error",
    test_scoped_timer_preserves_body_error,
    devices=devices,
)
add_function_test(
    TestSyncErrors,
    "test_timing_end_ends_timing_on_error",
    test_timing_end_ends_timing_on_error,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
