# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 64


@wp.kernel
def tile_sum_kernel(input: wp.array3d(dtype=float), output: wp.array(dtype=float)):    
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], 0, 0, m=TILE_M, n=TILE_N)
    s = wp.tile_sum(a) * 0.5

    wp.tile_store(output, i, 0, s)


def test_tile_reduce_sum(test, device):
    batch_count = 56

    M = TILE_M
    N = TILE_N

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, M, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_sum_kernel, dim=[batch_count], inputs=[input_wp, output_wp], block_dim=TILE_DIM, device=device)

    sum_wp = output_wp.numpy()
    for i in range(batch_count):
        sum_np = np.sum(input[i]) * 0.5
        test.assertAlmostEqual(sum_wp[i], sum_np, places=5)

    output_wp.grad.fill_(1.0)

    tape.backward()

    assert_np_equal(input_wp.grad.numpy(), np.ones_like(input) * 0.5)


@wp.kernel
def tile_reduce_simt_kernel(output: wp.array(dtype=int)):
    
    # thread index
    i = wp.tid()

    t = wp.tile(i)  # convert to block wide tile
    s = wp.tile_sum(t)  # sum over block

    # update global sum
    wp.tile_atomic_add(output, 0, 0, s)


def test_tile_reduce_simt(test, device):
    
    # use an unaligned grid dimension
    N = TILE_DIM*4 + 5

    output = wp.zeros(shape=1, dtype=int, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(tile_reduce_simt_kernel, dim=N, inputs=[output], block_dim=TILE_DIM, device=device)

    test.assertEqual(output.numpy()[0], np.sum(np.arange(N)))


@wp.kernel
def tile_ones_kernel(out: wp.array(dtype=float)):
    i = wp.tid()

    t = wp.tile_ones(dtype=float, m=16, n=16)
    s = wp.tile_sum(t)

    wp.tile_store(out, 0, 0, s)


def test_tile_ones(test, device):
    output = wp.zeros(shape=1, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_ones_kernel, dim=[1], inputs=[output], block_dim=TILE_DIM, device=device)

    test.assertAlmostEqual(output.numpy()[0], 256.0)


@wp.kernel
def tile_arange_kernel(out: wp.array2d(dtype=int)):
    i = wp.tid()

    a = wp.tile_arange(17, dtype=int)
    b = wp.tile_arange(5, 23, dtype=int)
    c = wp.tile_arange(0, 34, 2, dtype=int)

    wp.tile_store(out, 0, 0, a)
    wp.tile_store(out, 1, 0, b)
    wp.tile_store(out, 2, 0, c)


def test_tile_arange(test, device):
    N = 17

    output = wp.zeros(shape=(3, N), dtype=int, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_arange_kernel, dim=[1], inputs=[output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy()[0], np.arange(17))
    assert_np_equal(output.numpy()[1], np.arange(5, 22))
    assert_np_equal(output.numpy()[2], np.arange(0, 34, 2))


devices = get_cuda_test_devices()


class TestTileReduce(unittest.TestCase):
    pass


add_function_test(TestTileReduce, "test_tile_reduce_sum", test_tile_reduce_sum, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_simt", test_tile_reduce_simt, devices=devices)
add_function_test(TestTileReduce, "test_tile_ones", test_tile_ones, devices=devices)
add_function_test(TestTileReduce, "test_tile_arange", test_tile_arange, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
