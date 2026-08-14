# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1 << 12


@wp.kernel
def _fill_kernel(arr: wp.array[int], value: int):
    arr[wp.tid()] = value


@wp.kernel
def _add_kernel(out: wp.array[int], a: wp.array[int], b: wp.array[int]):
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel
def _scale_kernel(arr: wp.array[int], factor: int):
    i = wp.tid()
    arr[i] = arr[i] * factor


def _run_stf_inner_local(device) -> np.ndarray:
    stream = wp.Stream(device)
    a = wp.zeros(N, dtype=wp.int32, device=device)
    b = wp.zeros(N, dtype=wp.int32, device=device)
    c = wp.zeros(N, dtype=wp.int32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(
        device=device, stream=stream, capture_mode=wp.CaptureMode.RELAXED, force_module_load=False
    ) as capture:
        with wp_stf.context(stream=stream) as ctx:
            tok_a = ctx.token()
            tok_b = ctx.token()
            tok_c = ctx.token()

            with ctx.task(tok_a.write()) as (s,):
                wp.launch(_fill_kernel, dim=N, inputs=[a, 3], stream=s)

            with ctx.task(tok_b.write()) as (s,):
                wp.launch(_fill_kernel, dim=N, inputs=[b, 4], stream=s)

            with ctx.task(tok_a.read(), tok_b.read(), tok_c.write()) as (s,):
                wp.launch(_add_kernel, dim=N, inputs=[c, a, b], stream=s)

            with ctx.task(tok_c.rw()) as (s,):
                wp.launch(_scale_kernel, dim=N, inputs=[c, 2], stream=s)

    wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    return c.numpy()


@unittest.skipUnless(wp.is_cuda_available() and wp_stf.is_available(), "CUDASTF is not available")
class TestSTFExperimentalInnerLocal(unittest.TestCase):
    def test_inner_stream_context_in_capture(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            result = _run_stf_inner_local(device)

        np.testing.assert_array_equal(result, np.full(N, 14, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
