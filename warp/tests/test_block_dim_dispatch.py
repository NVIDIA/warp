# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test block_dim variant selection across devices and explicit loads.

These tests cover regressions where per-launch block_dim choices, such as
CPU's forced block_dim=1 or a tiled launch's explicit block_dim, must stay
scoped to the loaded executable variant instead of retargeting the
module-level default used by later loads.

The covered flows include:
1. A CPU launch loads a block_dim=1 variant.
2. A later CUDA launch or load uses the CUDA-default block_dim=256 variant.
3. Explicit non-default tiled loads and captures keep their requested variant.
"""

import unittest

import warp as wp
from warp.tests.unittest_utils import *

# Kernels are declared once at module scope so the @wp.kernel decorator (and its
# module hashing) runs a single time at import, not on every device-parameterized
# test invocation. Each kernel uses module="unique" because these tests assert on
# a module's per-(context, block_dim) execs, including negative assertNotIn checks
# (see test_load_module_no_cross_device_block_dim_leak and
# test_force_load_preserves_loaded_block_dim). A single shared module would not
# work: test_block_dim_cpu_then_cuda deliberately compiles the CUDA default
# block_dim=256 variant, which would then appear in the shared execs and break the
# force_load test's assertNotIn((ctx, 256)). module="unique" gives each distinct
# kernel its own module; identical kernels are deduplicated by hash, so the two
# conditional-atomic tests share one module and the two tile tests share another.


@wp.kernel(module="unique")
def conditional_atomic(x: float, result: wp.array[wp.int32]):
    wp.atomic_add(result, 0, 1) if x > 0.0 else wp.atomic_add(result, 1, 1)


@wp.kernel(module="unique")
def single_atomic(out: wp.array[wp.int32]):
    wp.atomic_add(out, 0, 1)


@wp.kernel(module="unique")
def tile_zeros_64(out: wp.array[float]):
    t = wp.tile_zeros(shape=64, dtype=float)
    out[wp.tid()] = t[0]


def test_block_dim_cpu_then_cuda(test, device):
    """Verify CUDA execution uses the correct block_dim after CPU execution.

    This test verifies that the kernel dispatch mechanism loads separate
    module executables for CPU (block_dim=1) and CUDA (block_dim=256),
    rather than incorrectly reusing the CPU module for CUDA.

    We check this by verifying that module.execs contains distinct entries
    for the different (device.context, block_dim) pairs.
    """

    module = conditional_atomic.module
    cpu_device = wp.get_device("cpu")

    # First, launch on CPU (should load module with block_dim=1)
    result_cpu = wp.zeros(2, dtype=wp.int32, device="cpu")
    wp.launch(conditional_atomic, dim=1, inputs=[1.0, result_cpu], device="cpu")

    # Verify CPU module exec was loaded with block_dim=1
    cpu_exec_key = (cpu_device.context, 1)
    test.assertIn(cpu_exec_key, module.execs, "CPU module exec should be loaded with block_dim=1")

    # Now launch on CUDA (should load separate module with block_dim=256)
    result_cuda = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(conditional_atomic, dim=1, inputs=[1.0, result_cuda], device=device)

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
    """Verify CPU command recording uses the correct block_dim after CUDA launch.

    This test specifically checks that the CPU record_cmd path carries the
    launch-normalized block_dim=1 into Launch.__init__ after a previous CUDA
    launch loaded the CUDA-default variant.
    """

    # First, launch on CUDA to load the CUDA-default block_dim=256 variant.
    result_cuda = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(conditional_atomic, dim=1, inputs=[1.0, result_cuda], device=device)
    test.assertEqual(result_cuda.numpy()[0], 1)

    # Now record a command on CPU. The launch path should normalize the
    # recorded block_dim to 1 instead of carrying the prior CUDA launch's 256.
    result_cpu = wp.zeros(2, dtype=wp.int32, device="cpu")
    cmd = wp.launch(conditional_atomic, dim=1, inputs=[1.0, result_cpu], device="cpu", record_cmd=True)

    # Execute the recorded command
    cmd.launch()

    # Verify CPU result - this will fail if block_dim mismatch causes memory corruption
    values_cpu = result_cpu.numpy()
    test.assertEqual(values_cpu[0], 1, "CPU: First branch should execute")
    test.assertEqual(values_cpu[1], 0, "CPU: Second branch should not execute")


def test_load_module_no_cross_device_block_dim_leak(test, device):
    """Verify wp.load_module on CUDA ignores a prior CPU launch's block_dim variant.

    The CPU fallback for the tile API loads modules with a forced block_dim=1.
    If that value leaks into the module-level default, a later GPU load picks up
    the nonsense block_dim=1 instead of the intended threads-per-block, forcing an
    unnecessary recompilation when the GPU kernel is actually launched. This test
    pins the fix: after a CPU launch loads a block_dim=1 variant,
    wp.load_module(device='cuda:0') without an explicit block_dim must compile the
    CUDA-default block_dim=256 variant and must not produce a block_dim=1 variant
    on CUDA.
    """

    module = single_atomic.module
    cpu_device = wp.get_device("cpu")

    # CPU launch first. The CPU path loads a block_dim=1 variant.
    result_cpu = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(single_atomic, dim=1, inputs=[result_cpu], device="cpu")
    test.assertIn((cpu_device.context, 1), module.execs)

    # wp.load_module(device='cuda:0') without explicit block_dim must
    # compile the CUDA-default variant, not reuse the CPU variant.
    wp.load_module(module=module, device=device)
    test.assertIn(
        (device.context, 256),
        module.execs,
        "wp.load_module should compile the CUDA default block_dim=256 variant",
    )
    test.assertNotIn(
        (device.context, 1),
        module.execs,
        "wp.load_module must not have compiled a block_dim=1 variant on CUDA",
    )


def test_load_module_tiled_block_dim(test, device):
    """Verify a tiled launch compiles the variant for its own block_dim.

    A tiled launch with block_dim != 256 must compile the variant for
    the launch's block_dim, with the source template's WP_TILE_BLOCK_DIM
    matching. Verifies that Module.resolve_options() honours the per-load
    block_dim, not the module-level default when they differ.
    """

    module = tile_zeros_64.module

    out = wp.zeros(64, dtype=float, device=device)
    wp.launch_tiled(tile_zeros_64, dim=64, inputs=[out], block_dim=64, device=device)

    test.assertIn(
        (device.context, 64),
        module.execs,
        "tiled launch should produce a block_dim=64 variant",
    )

    # The resolved options for the block_dim=64 variant must carry
    # block_dim=64, not the module-level default. Source-gen reads this
    # value to substitute WP_TILE_BLOCK_DIM in the .cu template.
    test.assertEqual(module.resolved_options[64]["block_dim"], 64)

    wp.synchronize_device(device)


def test_force_load_preserves_loaded_block_dim(test, device):
    """Verify force_load reuses an already-loaded block_dim variant.

    force_load(device) with no explicit block_dim must reuse the variant
    already loaded on the device, not compile a fresh module-level-default
    (256) variant. capture_begin() routes through force_load() on driver
    < 12.3, so a spurious default-block_dim compile there breaks graph
    capture for kernels whose tile shapes are tied to a specific block_dim
    (e.g. examples/tile/example_tile_mlp.py).
    """

    module = tile_zeros_64.module

    out = wp.zeros(64, dtype=float, device=device)
    wp.launch_tiled(tile_zeros_64, dim=64, inputs=[out], block_dim=64, device=device)
    test.assertIn((device.context, 64), module.execs)

    # force_load without an explicit block_dim must reuse the loaded 64
    # variant and must not compile the module-level default (256).
    wp.force_load(device=device, modules=[module])
    test.assertIn((device.context, 64), module.execs)
    test.assertNotIn(
        (device.context, 256),
        module.execs,
        "force_load must not compile the default block_dim=256 variant when a "
        "non-default variant is already loaded on the device",
    )

    wp.synchronize_device(device)


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

add_function_test(
    TestBlockDimDispatch,
    "test_load_module_no_cross_device_block_dim_leak",
    test_load_module_no_cross_device_block_dim_leak,
    devices=cuda_devices,
)

add_function_test(
    TestBlockDimDispatch,
    "test_load_module_tiled_block_dim",
    test_load_module_tiled_block_dim,
    devices=cuda_devices,
)

add_function_test(
    TestBlockDimDispatch,
    "test_force_load_preserves_loaded_block_dim",
    test_force_load_preserves_loaded_block_dim,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
