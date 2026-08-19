# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test CUDA debug stack sizing for 16-bit floating-point types.

This test uses a standalone module because Warp compiles all kernels in a
module together, and unrelated tile kernels could alter the reproducer.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_cuda_test_devices

TILE_M = wp.constant(8)
TILE_DIM = 64
bfloat16_vec3 = wp.types.vector(3, dtype=wp.bfloat16)


@wp.struct
class TileMapStruct:
    x: wp.float32
    y: wp.vec3


@wp.struct
class Float16VectorStruct:
    value: wp.vec3h


@wp.struct
class Bfloat16VectorStruct:
    value: bfloat16_vec3


@wp.func
def tile_map_struct_scale(value: TileMapStruct) -> TileMapStruct:
    result = TileMapStruct()
    result.x = value.x + wp.float32(1.0)
    result.y = value.y * wp.float32(2.0)
    return result


@wp.func
def tile_map_struct_add(a: TileMapStruct, b: TileMapStruct) -> TileMapStruct:
    result = TileMapStruct()
    result.x = a.x + b.x
    result.y = a.y + b.y
    return result


@wp.func
def tile_map_struct_sum(value: TileMapStruct) -> float:
    return value.x + value.y[0] + value.y[1] + value.y[2]


@wp.kernel(enable_backward=False)
def tile_float16_vector_kernel(input: wp.array[Float16VectorStruct], output: wp.array[Float16VectorStruct]):
    values = wp.tile_load(input, shape=TILE_M)
    wp.tile_store(output, values)


@wp.kernel(enable_backward=False)
def tile_bfloat16_vector_kernel(input: wp.array[Bfloat16VectorStruct], output: wp.array[Bfloat16VectorStruct]):
    values = wp.tile_load(input, shape=TILE_M)
    wp.tile_store(output, values)


@wp.kernel
def tile_map_struct_grad_kernel(input: wp.array[TileMapStruct], loss: wp.array[float]):
    i = wp.tid()
    values = wp.tile_load(input, shape=TILE_M, offset=i * TILE_M)
    scaled = wp.tile_map(tile_map_struct_scale, values)

    bias = TileMapStruct()
    bias.x = wp.float32(10.0)
    bias.y = wp.vec3(1.0, 2.0, 3.0)
    biased = wp.tile_map(tile_map_struct_add, scaled, bias)

    components = wp.tile_map(tile_map_struct_sum, biased)
    wp.tile_store(loss, wp.tile_sum(components), offset=i)


def test_tile_low_precision_module_stack(test, device):
    """Verify float16 and bfloat16 code does not disrupt another kernel's stack sizing.

    Warp compiles every kernel in a module together. Neither
    ``tile_float16_vector_kernel`` nor ``tile_bfloat16_vector_kernel`` is
    launched; merely compiling their 16-bit vector code reproduces the
    module-wide interaction that previously caused CUDA error 719 when
    launching the struct kernel's adjoint in debug mode.

    Args:
        test: The test case used for assertions.
        device: The CUDA device on which to run the test.
    """
    data = []
    for i in range(TILE_M):
        value = TileMapStruct()
        value.x = float(i)
        value.y = wp.vec3(float(i), float(i + 1), float(i + 2))
        data.append(value)

    input_wp = wp.array(data, dtype=TileMapStruct, requires_grad=True, device=device)
    loss_wp = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_map_struct_grad_kernel,
            dim=[1],
            inputs=[input_wp],
            outputs=[loss_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    indices = np.arange(TILE_M, dtype=np.float32)
    expected_y = np.stack((2.0 * indices + 1.0, 2.0 * indices + 4.0, 2.0 * indices + 7.0), axis=1)
    expected_loss = np.sum(indices + 11.0 + np.sum(expected_y, axis=1))
    test.assertAlmostEqual(loss_wp.numpy()[0], expected_loss, places=5)

    tape.backward(loss_wp)
    input_grad = input_wp.grad.numpy()
    assert_np_equal(input_grad["x"], np.ones(TILE_M, dtype=np.float32))
    assert_np_equal(input_grad["y"], np.full((TILE_M, 3), 2.0, dtype=np.float32))


cuda_devices = get_cuda_test_devices()


class TestTileLowPrecisionStack(unittest.TestCase):
    pass


add_function_test(
    TestTileLowPrecisionStack,
    "test_tile_low_precision_module_stack",
    test_tile_low_precision_module_stack,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
