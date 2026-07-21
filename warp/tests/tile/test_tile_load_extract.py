# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests tile element extraction with 512-thread blocks.

Covers one- through four-dimensional element access and its adjoint behavior.
Add tests here when they read individual elements from tiles or validate the
corresponding gradients.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_SIZE = 4


@wp.kernel
def tile_extract_1d_kernel(input: wp.array1d[float], output: wp.array1d[float]):
    i = wp.tid()

    t = wp.tile_load(input, shape=TILE_SIZE)

    output[i] = t[i]


@wp.kernel
def tile_extract_2d_kernel(input: wp.array2d[float], output: wp.array2d[float]):
    i, j = wp.tid()

    t = wp.tile_load(input, shape=(TILE_SIZE, TILE_SIZE))

    output[i, j] = t[i, j]


@wp.kernel
def tile_extract_3d_kernel(input: wp.array3d[float], output: wp.array3d[float]):
    i, j, k = wp.tid()

    t = wp.tile_load(input, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE))

    output[i, j, k] = t[i, j, k]


@wp.kernel
def tile_extract_4d_kernel(input: wp.array4d[float], output: wp.array4d[float]):
    i, j, k, l = wp.tid()

    t = wp.tile_load(input, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE))

    output[i, j, k, l] = t[i, j, k, l]


def test_tile_extract(kernel, ndim):
    shape = (TILE_SIZE,) * ndim

    def test_run(test, device):
        rng = np.random.default_rng(42)

        input = wp.array(rng.random(shape), dtype=float, requires_grad=True, device=device)
        output = wp.zeros_like(input)

        with wp.Tape() as tape:
            wp.launch(
                kernel,
                dim=shape,
                inputs=[input, output],
                # Debug CUDA backward kernels may be resource-limited below 1024 threads.
                block_dim=512,
                device=device,
            )

        assert_np_equal(output.numpy(), input.numpy())

        output.grad = wp.ones_like(output)
        tape.backward()

        assert_np_equal(input.grad.numpy(), np.ones_like(input.numpy()))

    return test_run


devices = get_test_devices()


class TestTileLoadExtract(unittest.TestCase):
    pass


add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_1d",
    test_tile_extract(tile_extract_1d_kernel, 1),
    devices=devices,
)
add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_2d",
    test_tile_extract(tile_extract_2d_kernel, 2),
    devices=devices,
)
add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_3d",
    test_tile_extract(tile_extract_3d_kernel, 3),
    devices=devices,
)
add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_4d",
    test_tile_extract(tile_extract_4d_kernel, 4),
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
