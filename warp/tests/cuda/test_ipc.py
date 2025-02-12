# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import multiprocessing as mp
import unittest

import warp as wp
from warp.tests.unittest_utils import *


def test_ipc_get_memory_handle(test, device):
    if device.is_ipc_supported is False:
        test.skipTest(f"IPC is not supported on {device}")

    with wp.ScopedMempool(device, False):
        test_array = wp.full(10, value=42.0, dtype=wp.float32, device=device)
        ipc_handle = test_array.ipc_handle()

    test.assertNotEqual(ipc_handle, bytes(64), "IPC memory handle appears to be invalid")


def test_ipc_get_event_handle(test, device):
    if device.is_ipc_supported is False:
        test.skipTest(f"IPC is not supported on {device}")

    e1 = wp.Event(device, interprocess=True)

    ipc_handle = e1.ipc_handle()

    test.assertNotEqual(ipc_handle, bytes(64), "IPC event handle appears to be invalid")


def test_ipc_event_missing_interprocess_flag(test, device):
    if device.is_ipc_supported is False:
        test.skipTest(f"IPC is not supported on {device}")

    e1 = wp.Event(device, interprocess=False)

    try:
        capture = StdOutCapture()
        capture.begin()
        ipc_handle = e1.ipc_handle()
    finally:
        output = capture.end()

    # Older Windows C runtimes have a bug where stdout sometimes does not get properly flushed.
    if sys.platform != "win32":
        test.assertRegex(output, r"Warp UserWarning: IPC event handle appears to be invalid.")


@wp.kernel
def multiply_by_two(a: wp.array(dtype=wp.float32)):
    i = wp.tid()
    a[i] = 2.0 * a[i]


def child_task(array_handle, dtype, shape, device, event_handle):
    with wp.ScopedDevice(device):
        ipc_array = wp.from_ipc_handle(array_handle, dtype, shape, device=device)
        ipc_event = wp.event_from_ipc_handle(event_handle, device=device)
        stream = wp.get_stream()
        wp.launch(multiply_by_two, ipc_array.shape, inputs=[ipc_array])
        stream.record_event(ipc_event)
        stream.wait_event(ipc_event)
        wp.synchronize_device()


def test_ipc_multiprocess_write(test, device):
    if device.is_ipc_supported is False:
        test.skipTest(f"IPC is not supported on {device}")

    stream = wp.get_stream(device)
    e1 = wp.Event(device, interprocess=True)

    with wp.ScopedMempool(device, False):
        test_array = wp.full(1024, value=42.0, dtype=wp.float32, device=device)
        ipc_handle = test_array.ipc_handle()

    wp.launch(multiply_by_two, test_array.shape, inputs=[test_array], device=device)

    ctx = mp.get_context("spawn")

    process = ctx.Process(
        target=child_task, args=(ipc_handle, test_array.dtype, test_array.shape, str(device), e1.ipc_handle())
    )

    process.start()
    process.join()

    assert_np_equal(test_array.numpy(), np.full(test_array.shape, 168.0, dtype=np.float32))


cuda_devices = get_cuda_test_devices()


class TestIpc(unittest.TestCase):
    pass


add_function_test(TestIpc, "test_ipc_get_memory_handle", test_ipc_get_memory_handle, devices=cuda_devices)
add_function_test(TestIpc, "test_ipc_get_event_handle", test_ipc_get_event_handle, devices=cuda_devices)
add_function_test(
    TestIpc, "test_ipc_event_missing_interprocess_flag", test_ipc_event_missing_interprocess_flag, devices=cuda_devices
)
add_function_test(
    TestIpc, "test_ipc_multiprocess_write", test_ipc_multiprocess_write, devices=cuda_devices, check_output=False
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
