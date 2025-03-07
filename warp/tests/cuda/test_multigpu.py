# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *
from warp.utils import check_p2p


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


@wp.kernel
def arange(start: int, step: int, a: wp.array(dtype=int)):
    tid = wp.tid()
    a[tid] = start + step * tid


class TestMultiGPU(unittest.TestCase):
    @unittest.skipUnless(len(wp.get_cuda_devices()) > 1, "Requires at least two CUDA devices")
    def test_multigpu_set_device(self):
        # save default device
        saved_device = wp.get_device()

        n = 32

        wp.set_device("cuda:0")
        a0 = wp.empty(n, dtype=int)
        wp.launch(arange, dim=a0.size, inputs=[0, 1, a0])

        wp.set_device("cuda:1")
        a1 = wp.empty(n, dtype=int)
        wp.launch(arange, dim=a1.size, inputs=[0, 1, a1])

        # restore default device
        wp.set_device(saved_device)

        assert a0.device == "cuda:0"
        assert a1.device == "cuda:1"

        expected = np.arange(n, dtype=int)

        assert_np_equal(a0.numpy(), expected)
        assert_np_equal(a1.numpy(), expected)

    @unittest.skipUnless(len(wp.get_cuda_devices()) > 1, "Requires at least two CUDA devices")
    def test_multigpu_scoped_device(self):
        n = 32

        with wp.ScopedDevice("cuda:0"):
            a0 = wp.empty(n, dtype=int)
            wp.launch(arange, dim=a0.size, inputs=[0, 1, a0])

        with wp.ScopedDevice("cuda:1"):
            a1 = wp.empty(n, dtype=int)
            wp.launch(arange, dim=a1.size, inputs=[0, 1, a1])

        assert a0.device == "cuda:0"
        assert a1.device == "cuda:1"

        expected = np.arange(n, dtype=int)

        assert_np_equal(a0.numpy(), expected)
        assert_np_equal(a1.numpy(), expected)

    @unittest.skipUnless(len(wp.get_cuda_devices()) > 1, "Requires at least two CUDA devices")
    def test_multigpu_nesting(self):
        initial_device = wp.get_device()
        initial_cuda_device = wp.get_cuda_device()

        with wp.ScopedDevice("cuda:1"):
            assert wp.get_device() == "cuda:1"
            assert wp.get_cuda_device() == "cuda:1"

            with wp.ScopedDevice("cuda:0"):
                assert wp.get_device() == "cuda:0"
                assert wp.get_cuda_device() == "cuda:0"

                with wp.ScopedDevice("cpu"):
                    assert wp.get_device() == "cpu"
                    assert wp.get_cuda_device() == "cuda:0"

                    wp.set_device("cuda:1")

                    assert wp.get_device() == "cuda:1"
                    assert wp.get_cuda_device() == "cuda:1"

                assert wp.get_device() == "cuda:0"
                assert wp.get_cuda_device() == "cuda:0"

            assert wp.get_device() == "cuda:1"
            assert wp.get_cuda_device() == "cuda:1"

        assert wp.get_device() == initial_device
        assert wp.get_cuda_device() == initial_cuda_device

    @unittest.skipUnless(len(wp.get_cuda_devices()) > 1, "Requires at least two CUDA devices")
    @unittest.skipUnless(check_p2p(), "Peer-to-Peer transfers not supported")
    def test_multigpu_pingpong(self):
        n = 1024 * 1024

        a0 = wp.zeros(n, dtype=float, device="cuda:0")
        a1 = wp.zeros(n, dtype=float, device="cuda:1")

        iters = 10

        for _ in range(iters):
            wp.launch(inc, dim=a0.size, inputs=[a0], device=a0.device)
            wp.synchronize_device(a0.device)
            wp.copy(a1, a0)

            wp.launch(inc, dim=a1.size, inputs=[a1], device=a1.device)
            wp.synchronize_device(a1.device)
            wp.copy(a0, a1)

        expected = np.full(n, iters * 2, dtype=np.float32)

        assert_np_equal(a0.numpy(), expected)
        assert_np_equal(a1.numpy(), expected)

    @unittest.skipUnless(len(wp.get_cuda_devices()) > 1, "Requires at least two CUDA devices")
    @unittest.skipUnless(check_p2p(), "Peer-to-Peer transfers not supported")
    def test_multigpu_pingpong_streams(self):
        n = 1024 * 1024

        a0 = wp.zeros(n, dtype=float, device="cuda:0")
        a1 = wp.zeros(n, dtype=float, device="cuda:1")

        stream0 = wp.get_stream("cuda:0")
        stream1 = wp.get_stream("cuda:1")

        iters = 10

        for _ in range(iters):
            wp.launch(inc, dim=a0.size, inputs=[a0], stream=stream0)
            stream1.wait_stream(stream0)
            wp.copy(a1, a0, stream=stream1)

            wp.launch(inc, dim=a1.size, inputs=[a1], stream=stream1)
            stream0.wait_stream(stream1)
            wp.copy(a0, a1, stream=stream0)

        expected = np.full(n, iters * 2, dtype=np.float32)

        assert_np_equal(a0.numpy(), expected)
        assert_np_equal(a1.numpy(), expected)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
