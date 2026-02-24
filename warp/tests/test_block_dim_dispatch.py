# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test that kernel dispatch uses correct block_dim for each device.

This test reproduces a bug where shared mutable state (Module.options["block_dim"])
can cause a module compiled for one block dimension to be incorrectly used for
execution with a different block dimension.

The bug manifests when:
1. A kernel is launched on CPU (which sets block_dim=1)
2. The same kernel is later launched on CUDA (which should use block_dim=256)
3. If Module.load() is called without explicit block_dim, it uses the stale value
4. This causes out-of-bounds shared memory access in tile infrastructure
"""

import unittest

import warp as wp
from warp.tests.unittest_utils import *


def test_block_dim_cpu_then_cuda(test, device):
    """Test that CUDA execution works correctly after CPU execution.

    This test specifically checks that the kernel dispatch mechanism
    uses the correct block_dim for each device, even when the module's
    options["block_dim"] may have been modified by a previous launch.
    """

    @wp.kernel
    def simple_conditional(x: float, result: wp.array(dtype=wp.int32)):
        # Use a conditional expression that requires both branches to be executable
        # This will fail if there's memory corruption from incorrect block_dim
        wp.atomic_add(result, 0, 1) if x > 0.0 else wp.atomic_add(result, 1, 1)

    # First, launch on CPU (sets module.options["block_dim"] = 1)
    result_cpu = wp.zeros(2, dtype=wp.int32, device="cpu")
    wp.launch(simple_conditional, dim=1, inputs=[1.0, result_cpu], device="cpu")

    # Verify CPU result
    values_cpu = result_cpu.numpy()
    test.assertEqual(values_cpu[0], 1, "CPU: First branch should execute")
    test.assertEqual(values_cpu[1], 0, "CPU: Second branch should not execute")

    # Now launch on CUDA - this should use block_dim=256, not the stale value of 1
    # If the bug exists, Launch.__init__ will call module.load(device) without
    # passing block_dim, causing it to use the stale value of 1 from the CPU launch
    result_cuda = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(simple_conditional, dim=1, inputs=[1.0, result_cuda], device=device)

    # Verify CUDA result - this will fail if block_dim mismatch causes memory corruption
    values_cuda = result_cuda.numpy()
    test.assertEqual(values_cuda[0], 1, "CUDA: First branch should execute")
    test.assertEqual(values_cuda[1], 0, "CUDA: Second branch should not execute")


class TestBlockDimDispatch(unittest.TestCase):
    pass


devices = get_test_devices()

# Only test on CUDA devices (CPU would pass trivially)
cuda_devices = [d for d in devices if d != "cpu"]

add_function_test(
    TestBlockDimDispatch,
    "test_block_dim_cpu_then_cuda",
    test_block_dim_cpu_then_cuda,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
