# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1024


@wp.kernel
def _fill_kernel(arr: wp.array[int], value: int):
    arr[wp.tid()] = value


def _run_fill_on_exec_place(device: wp.Device, exec_place, value: int) -> np.ndarray:
    with wp.ScopedDevice(device):
        result = wp.zeros(N, dtype=wp.int32, device=device)
        graph = wp_stf.task_graph()
        ctx = graph.context
        token = ctx.token()

        with graph:
            with ctx.task(token.write(), exec_place=exec_place, symbol="fill_on_exec_place") as (stream,):
                assert stream.device.ordinal == device.ordinal
                wp.launch(_fill_kernel, dim=N, inputs=[result, value], stream=stream)

        graph.launch()
        graph.finalize()

        return result.numpy()


@unittest.skipUnless(wp.is_cuda_available() and wp_stf.is_available(), "CUDASTF is not available")
class TestSTFExperimentalExecPlace(unittest.TestCase):
    def test_raw_exec_place(self):
        stf = importlib.import_module("cuda.stf._experimental")

        wp.init()
        device = wp.get_device("cuda:0")
        result = _run_fill_on_exec_place(device, stf.exec_place.device(device.ordinal), 17)
        np.testing.assert_array_equal(result, np.full(N, 17, dtype=np.int32))

    def test_warp_device_exec_place(self):
        wp.init()
        device = wp.get_device("cuda:0")
        result = _run_fill_on_exec_place(device, device, 23)
        np.testing.assert_array_equal(result, np.full(N, 23, dtype=np.int32))

    def test_invalid_exec_place_inputs(self):
        stf = importlib.import_module("cuda.stf._experimental")

        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            ctx = wp_stf.context()
            token = ctx.token()

            with self.assertRaises(TypeError):
                with ctx.task(token.write(), exec_place=0):
                    pass

            # Warp device alias strings are coerced to wp.Device.
            with ctx.task(token.write(), exec_place="cuda:0") as (stream,):
                self.assertEqual(stream.device, device)

            with self.assertRaises(TypeError):
                with ctx.task(token.write(), exec_place=stf.exec_place.host()):
                    pass

            with self.assertRaises(TypeError):
                with ctx.task(token.write(), exec_place=stf.exec_place_grid.from_devices([0])):
                    pass

            ctx.finalize()


@unittest.skipUnless(
    wp.is_cuda_available() and wp_stf.is_available() and wp.get_cuda_device_count() >= 2,
    "CUDASTF or multiple CUDA devices are not available",
)
class TestSTFExperimentalExecPlaceMultiGpu(unittest.TestCase):
    def test_warp_device_exec_place_cuda_1(self):
        wp.init()
        device = wp.get_device("cuda:1")
        result = _run_fill_on_exec_place(device, device, 31)
        np.testing.assert_array_equal(result, np.full(N, 31, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
