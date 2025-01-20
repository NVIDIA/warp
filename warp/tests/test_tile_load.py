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

TILE_DIM = 64

TILE_M = wp.constant(16)
TILE_N = wp.constant(8)
TILE_O = wp.constant(8)
TILE_P = wp.constant(6)

OFFSET = 5


@wp.kernel
def tile_load_1d_padded(
    input: wp.array1d(dtype=float), output_full: wp.array1d(dtype=float), output_partial: wp.array1d(dtype=float)
):
    i = wp.tid()

    # read OFFSET elements past bounds, zfill remainder
    t = wp.tile_load(input, 0, TILE_M)

    u = t * 2.0 + wp.tile_ones(TILE_M, dtype=float)

    # store the full tile
    wp.tile_store(output_full, 0, u)
    # store a partial tile
    wp.tile_store(output_partial, 0, u)


def test_tile_load_1d_padded(test, device):
    TILE_FULL = TILE_M
    TILE_PARTIAL = TILE_M - OFFSET

    rng = np.random.default_rng(42)

    input = wp.array(rng.random(TILE_PARTIAL), dtype=float, requires_grad=True, device=device)
    output_full = wp.zeros(TILE_FULL, dtype=float, device=device)
    output_partial = wp.zeros(TILE_PARTIAL, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_load_1d_padded,
            dim=[1],
            inputs=[input, output_full, output_partial],
            block_dim=TILE_DIM,
            device=device,
        )

    ref_full = np.pad(input.numpy(), (0, OFFSET))
    ref_partial = input.numpy()

    assert_np_equal(output_full.numpy(), ref_full * 2.0 + 1.0)
    assert_np_equal(output_partial.numpy(), ref_partial * 2.0 + 1.0)

    output_full.grad = wp.ones_like(output_full)
    tape.backward()

    assert_np_equal(input.grad.numpy(), np.ones_like(input.grad.numpy()) * 2.0)


# ----------------------------------------------------------------------------------------


@wp.kernel
def tile_load_2d_padded(
    input: wp.array2d(dtype=float), output_full: wp.array2d(dtype=float), output_partial: wp.array2d(dtype=float)
):
    i = wp.tid()

    # read OFFSET elements past bounds, zfill remainder
    t = wp.tile_load(input, 0, 0, TILE_M, TILE_N)

    u = t * 2.0 + wp.tile_ones(TILE_M, TILE_N, dtype=float)

    # store the full tile
    wp.tile_store(output_full, 0, 0, u)
    # store a partial tile
    wp.tile_store(output_partial, 0, 0, u)


def test_tile_load_2d_padded(test, device):
    TILE_FULL = (TILE_M, TILE_N)
    TILE_PARTIAL = (TILE_M - OFFSET, TILE_N - OFFSET)

    rng = np.random.default_rng(42)

    input = wp.array(rng.random(TILE_PARTIAL), dtype=float, requires_grad=True, device=device)
    output_full = wp.zeros(TILE_FULL, dtype=float, device=device)
    output_partial = wp.zeros(TILE_PARTIAL, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_load_2d_padded,
            dim=[1],
            inputs=[input, output_full, output_partial],
            block_dim=TILE_DIM,
            device=device,
        )

    ref_full = np.pad(input.numpy(), [(0, OFFSET), (0, OFFSET)])
    ref_partial = input.numpy()

    assert_np_equal(output_full.numpy(), ref_full * 2.0 + 1.0)
    assert_np_equal(output_partial.numpy(), ref_partial * 2.0 + 1.0)

    output_full.grad = wp.ones_like(output_full)
    tape.backward()

    assert_np_equal(input.grad.numpy(), np.ones_like(input.grad.numpy()) * 2.0)


# ----------------------------------------------------------------------------------------


@wp.kernel
def tile_load_3d_padded(
    input: wp.array3d(dtype=float), output_full: wp.array3d(dtype=float), output_partial: wp.array3d(dtype=float)
):
    i = wp.tid()

    # read OFFSET elements past bounds, zfill remainder
    t = wp.tile_load(input, 0, 0, 0, TILE_M, TILE_N, TILE_O)

    u = t * 2.0 + wp.tile_ones(TILE_M, TILE_N, TILE_O, dtype=float)

    # store the full tile
    wp.tile_store(output_full, 0, 0, 0, u)
    # store a partial tile
    wp.tile_store(output_partial, 0, 0, 0, u)


def test_tile_load_3d_padded(test, device):
    TILE_FULL = (TILE_M, TILE_N, TILE_O)
    TILE_PARTIAL = (TILE_M - OFFSET, TILE_N - OFFSET, TILE_O - OFFSET)

    rng = np.random.default_rng(42)

    input = wp.array(rng.random(TILE_PARTIAL), dtype=float, requires_grad=True, device=device)
    output_full = wp.zeros(TILE_FULL, dtype=float, device=device)
    output_partial = wp.zeros(TILE_PARTIAL, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_load_3d_padded,
            dim=[1],
            inputs=[input, output_full, output_partial],
            block_dim=TILE_DIM,
            device=device,
        )

    ref_full = np.pad(input.numpy(), [(0, OFFSET), (0, OFFSET), (0, OFFSET)])
    ref_partial = input.numpy()

    assert_np_equal(output_full.numpy(), ref_full * 2.0 + 1.0)
    assert_np_equal(output_partial.numpy(), ref_partial * 2.0 + 1.0)

    output_full.grad = wp.ones_like(output_full)
    tape.backward()

    assert_np_equal(input.grad.numpy(), np.ones_like(input.grad.numpy()) * 2.0)


# ----------------------------------------------------------------------------------------


@wp.kernel
def tile_load_4d_padded(
    input: wp.array4d(dtype=float), output_full: wp.array4d(dtype=float), output_partial: wp.array4d(dtype=float)
):
    i = wp.tid()

    # read OFFSET elements past bounds, zfill remainder
    t = wp.tile_load(input, 0, 0, 0, 0, TILE_M, TILE_N, TILE_O, TILE_P)

    u = t * 2.0 + wp.tile_ones(TILE_M, TILE_N, TILE_O, TILE_P, dtype=float)

    # store the full tile
    wp.tile_store(output_full, 0, 0, 0, 0, u)
    # store a partial tile
    wp.tile_store(output_partial, 0, 0, 0, 0, u)


def test_tile_load_4d_padded(test, device):
    TILE_FULL = (TILE_M, TILE_N, TILE_O, TILE_P)
    TILE_PARTIAL = (TILE_M - OFFSET, TILE_N - OFFSET, TILE_O - OFFSET, TILE_P - OFFSET)

    rng = np.random.default_rng(42)

    input = wp.array(rng.random(TILE_PARTIAL), dtype=float, requires_grad=True, device=device)
    output_full = wp.zeros(TILE_FULL, dtype=float, device=device)
    output_partial = wp.zeros(TILE_PARTIAL, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_load_4d_padded,
            dim=[1],
            inputs=[input, output_full, output_partial],
            block_dim=TILE_DIM,
            device=device,
        )

    ref_full = np.pad(input.numpy(), [(0, OFFSET), (0, OFFSET), (0, OFFSET), (0, OFFSET)])
    ref_partial = input.numpy()

    assert_np_equal(output_full.numpy(), ref_full * 2.0 + 1.0)
    assert_np_equal(output_partial.numpy(), ref_partial * 2.0 + 1.0)

    output_full.grad = wp.ones_like(output_full)
    tape.backward()

    assert_np_equal(input.grad.numpy(), np.ones_like(input.grad.numpy()) * 2.0)


# ----------------------------------------------------------------------------------------

TILE_SIZE = 4


@wp.kernel
def tile_load_1d_extract_kernel(input: wp.array1d(dtype=float), output: wp.array1d(dtype=float)):
    i = wp.tid()

    t = wp.tile_load(input, 0, TILE_SIZE)

    output[i] = t[i]


@wp.kernel
def tile_load_2d_extract_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
    i, j = wp.tid()

    t = wp.tile_load(input, 0, 0, TILE_SIZE, TILE_SIZE)

    output[i, j] = t[i, j]


@wp.kernel
def tile_load_3d_extract_kernel(input: wp.array3d(dtype=float), output: wp.array3d(dtype=float)):
    i, j, k = wp.tid()

    t = wp.tile_load(input, 0, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE)

    output[i, j, k] = t[i, j, k]


@wp.kernel
def tile_load_4d_extract_kernel(input: wp.array4d(dtype=float), output: wp.array4d(dtype=float)):
    i, j, k, l = wp.tid()

    t = wp.tile_load(input, 0, 0, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE)

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
                block_dim=1024,
                device=device,
            )

        assert_np_equal(output.numpy(), input.numpy())

        output.grad = wp.ones_like(output)
        tape.backward()

        assert_np_equal(input.grad.numpy(), np.ones_like(input.numpy()))

    return test_run


# ----------------------------------------------------------------------------------------

TILE_SIZE = 4


@wp.kernel
def tile_load_1d_assign_kernel(input: wp.array1d(dtype=float), output: wp.array1d(dtype=float)):
    i = wp.tid()

    t = wp.tile_zeros(TILE_SIZE, dtype=float)

    # assign to tile
    t[i] = input[i] * 2.0

    output[i] = t[i]


@wp.kernel
def tile_load_2d_assign_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
    i, j = wp.tid()

    t = wp.tile_zeros(TILE_SIZE, TILE_SIZE, dtype=float)

    # assign to tile
    t[i, j] = input[i, j] * 2.0

    output[i, j] = t[i, j]


@wp.kernel
def tile_load_3d_assign_kernel(input: wp.array3d(dtype=float), output: wp.array3d(dtype=float)):
    i, j, k = wp.tid()

    t = wp.tile_zeros(TILE_SIZE, TILE_SIZE, TILE_SIZE, dtype=float)

    # assign to tile
    t[i, j, k] = input[i, j, k] * 2.0

    output[i, j, k] = t[i, j, k]


@wp.kernel
def tile_load_4d_assign_kernel(input: wp.array4d(dtype=float), output: wp.array4d(dtype=float)):
    i, j, k, l = wp.tid()

    t = wp.tile_zeros(TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, dtype=float)

    # assign to tile
    t[i, j, k, l] = input[i, j, k, l] * 2.0

    output[i, j, k, l] = t[i, j, k, l]


def test_tile_assign(kernel, ndim):
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
                block_dim=1024,
                device=device,
            )

        assert_np_equal(output.numpy(), input.numpy() * 2.0)

    return test_run


# ----------------------------------------------------------------------------------------


@wp.kernel
def tile_load_fortran_kernel(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float)):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(A, i, j, m=TILE_M, n=TILE_N)
    wp.tile_store(B, i, j, a)


def test_tile_load_fortran(test, device):
    rng = np.random.default_rng(42)

    M = TILE_M * 7
    N = TILE_N * 5

    A = rng.random((M, N), dtype=np.float32)
    B = rng.random((M, N), dtype=np.float32)

    # convert to column major layout
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_load_fortran_kernel,
            dim=[int(M / TILE_M), int(N / TILE_N)],
            inputs=[A_wp, B_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    # verify forward pass
    assert_array_equal(B_wp, A_wp)

    # verify backward pass
    B_wp.grad = wp.ones_like(B_wp, device=device)
    tape.backward()

    assert_array_equal(B_wp.grad, A_wp.grad)


devices = get_cuda_test_devices()


class TestTileLoad(unittest.TestCase):
    pass


add_function_test(TestTileLoad, "test_tile_load_1d_padded", test_tile_load_1d_padded, devices=devices)
add_function_test(TestTileLoad, "test_tile_load_2d_padded", test_tile_load_2d_padded, devices=devices)
add_function_test(TestTileLoad, "test_tile_load_3d_padded", test_tile_load_3d_padded, devices=devices)
add_function_test(TestTileLoad, "test_tile_load_4d_padded", test_tile_load_4d_padded, devices=devices)

add_function_test(
    TestTileLoad, "test_tile_extract_1d", test_tile_extract(tile_load_1d_extract_kernel, 1), devices=devices
)
add_function_test(
    TestTileLoad, "test_tile_extract_2d", test_tile_extract(tile_load_2d_extract_kernel, 2), devices=devices
)
add_function_test(
    TestTileLoad, "test_tile_extract_3d", test_tile_extract(tile_load_3d_extract_kernel, 3), devices=devices
)
add_function_test(
    TestTileLoad, "test_tile_extract_4d", test_tile_extract(tile_load_4d_extract_kernel, 4), devices=devices
)

add_function_test(TestTileLoad, "test_tile_assign_1d", test_tile_assign(tile_load_1d_assign_kernel, 1), devices=devices)
add_function_test(TestTileLoad, "test_tile_assign_2d", test_tile_assign(tile_load_2d_assign_kernel, 2), devices=devices)
add_function_test(TestTileLoad, "test_tile_assign_3d", test_tile_assign(tile_load_3d_assign_kernel, 3), devices=devices)
add_function_test(TestTileLoad, "test_tile_assign_4d", test_tile_assign(tile_load_4d_assign_kernel, 4), devices=devices)

add_function_test(TestTileLoad, "test_tile_load_fortran", test_tile_load_fortran, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
