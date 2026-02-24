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
    """Test that CUDA execution uses correct block_dim after CPU execution.

    This test verifies that the kernel dispatch mechanism loads separate
    module executables for CPU (block_dim=1) and CUDA (block_dim=256),
    rather than incorrectly reusing the CPU module for CUDA.

    We check this by verifying that module.execs contains distinct entries
    for the different (device.context, block_dim) pairs.
    """

    @wp.kernel
    def simple_conditional(x: float, result: wp.array(dtype=wp.int32)):
        wp.atomic_add(result, 0, 1) if x > 0.0 else wp.atomic_add(result, 1, 1)

    module = simple_conditional.module
    cpu_device = wp.get_device("cpu")

    # First, launch on CPU (should load module with block_dim=1)
    result_cpu = wp.zeros(2, dtype=wp.int32, device="cpu")
    wp.launch(simple_conditional, dim=1, inputs=[1.0, result_cpu], device="cpu")

    # Verify CPU module exec was loaded with block_dim=1
    cpu_exec_key = (cpu_device.context, 1)
    test.assertIn(cpu_exec_key, module.execs, "CPU module exec should be loaded with block_dim=1")

    # Now launch on CUDA (should load separate module with block_dim=256)
    result_cuda = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(simple_conditional, dim=1, inputs=[1.0, result_cuda], device=device)

    # Verify CUDA module exec was loaded with block_dim=256
    cuda_exec_key = (device.context, 256)
    test.assertIn(cuda_exec_key, module.execs, "CUDA module exec should be loaded with block_dim=256")

    # Verify that the two execs are distinct (not the same object)
    cpu_exec = module.execs[cpu_exec_key]
    cuda_exec = module.execs[cuda_exec_key]
    test.assertIsNot(cpu_exec, cuda_exec, "CPU and CUDA should use separate module executables")

    # Verify functional correctness as a sanity check
    test.assertEqual(result_cpu.numpy()[0], 1)
    test.assertEqual(result_cuda.numpy()[0], 1)


class TestBlockDimDispatch(unittest.TestCase):
    pass


def test_block_dim_record_cmd_cpu(test, device):
    """Test that CPU command recording uses correct block_dim after CUDA launch.

    This test specifically checks the CPU record_cmd path to ensure it passes
    block_dim to Launch.__init__ correctly, even when module.options["block_dim"]
    has been set to 256 by a previous CUDA launch.
    """

    @wp.kernel
    def simple_conditional(x: float, result: wp.array(dtype=wp.int32)):
        wp.atomic_add(result, 0, 1) if x > 0.0 else wp.atomic_add(result, 1, 1)

    # First, launch on CUDA (sets module.options["block_dim"] = 256)
    result_cuda = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(simple_conditional, dim=1, inputs=[1.0, result_cuda], device=device)
    test.assertEqual(result_cuda.numpy()[0], 1)

    # Now record a command on CPU - this should use block_dim=1, not the stale value of 256
    # Without the fix, the record_cmd path would use block_dim=256 and fail
    result_cpu = wp.zeros(2, dtype=wp.int32, device="cpu")
    cmd = wp.launch(simple_conditional, dim=1, inputs=[1.0, result_cpu], device="cpu", record_cmd=True)

    # Execute the recorded command
    cmd.launch()

    # Verify CPU result - this will fail if block_dim mismatch causes memory corruption
    values_cpu = result_cpu.numpy()
    test.assertEqual(values_cpu[0], 1, "CPU: First branch should execute")
    test.assertEqual(values_cpu[1], 0, "CPU: Second branch should not execute")


devices = get_test_devices()

# Only test on CUDA devices (CPU would pass trivially)
cuda_devices = [d for d in devices if d != "cpu"]

add_function_test(
    TestBlockDimDispatch,
    "test_block_dim_cpu_then_cuda",
    test_block_dim_cpu_then_cuda,
    devices=cuda_devices,
)

add_function_test(
    TestBlockDimDispatch,
    "test_block_dim_record_cmd_cpu",
    test_block_dim_record_cmd_cpu,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
