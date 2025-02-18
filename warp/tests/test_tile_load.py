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

TILE_OFFSET = 5


@wp.kernel
def tile_load_1d_kernel(
    input: wp.array1d(dtype=float),
    out_full: wp.array1d(dtype=float),
    out_padded: wp.array1d(dtype=float),
    out_offset: wp.array1d(dtype=float),
):
    full0 = wp.tile_load(input, TILE_M)
    full1 = wp.tile_load(input, shape=TILE_M)
    full2 = wp.tile_load(input, shape=(TILE_M,))

    padded0 = wp.tile_load(input, TILE_M, TILE_OFFSET)
    padded1 = wp.tile_load(input, shape=TILE_M, offset=TILE_OFFSET)
    padded2 = wp.tile_load(input, shape=(TILE_M,), offset=(TILE_OFFSET,))

    wp.tile_store(out_full, full0)
    wp.tile_store(out_padded, padded0)
    wp.tile_store(out_offset, full0, offset=(TILE_OFFSET,))


@wp.kernel
def tile_load_2d_kernel(
    input: wp.array2d(dtype=float),
    out_full: wp.array2d(dtype=float),
    out_padded: wp.array2d(dtype=float),
    out_offset: wp.array2d(dtype=float),
):
    full0 = wp.tile_load(input, shape=(TILE_M, TILE_N))
    padded0 = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(TILE_OFFSET, TILE_OFFSET))

    wp.tile_store(out_full, full0)
    wp.tile_store(out_padded, padded0)
    wp.tile_store(out_offset, full0, offset=(TILE_OFFSET, TILE_OFFSET))


@wp.kernel
def tile_load_3d_kernel(
    input: wp.array3d(dtype=float),
    out_full: wp.array3d(dtype=float),
    out_padded: wp.array3d(dtype=float),
    out_offset: wp.array3d(dtype=float),
):
    full0 = wp.tile_load(input, shape=(TILE_M, TILE_N, TILE_O))
    padded0 = wp.tile_load(input, shape=(TILE_M, TILE_N, TILE_O), offset=(TILE_OFFSET, TILE_OFFSET, TILE_OFFSET))

    wp.tile_store(out_full, full0)
    wp.tile_store(out_padded, padded0)
    wp.tile_store(out_offset, full0, offset=(TILE_OFFSET, TILE_OFFSET, TILE_OFFSET))


@wp.kernel
def tile_load_4d_kernel(
    input: wp.array4d(dtype=float),
    out_full: wp.array4d(dtype=float),
    out_padded: wp.array4d(dtype=float),
    out_offset: wp.array4d(dtype=float),
):
    full0 = wp.tile_load(input, shape=(TILE_M, TILE_N, TILE_O, TILE_P))
    padded0 = wp.tile_load(
        input, shape=(TILE_M, TILE_N, TILE_O, TILE_P), offset=(TILE_OFFSET, TILE_OFFSET, TILE_OFFSET, TILE_OFFSET)
    )

    wp.tile_store(out_full, full0)
    wp.tile_store(out_padded, padded0)
    wp.tile_store(out_offset, full0, offset=(TILE_OFFSET, TILE_OFFSET, TILE_OFFSET, TILE_OFFSET))


def test_tile_load(kernel, ndim):
    def test(test, device):
        rng = np.random.default_rng(42)

        shape = [TILE_M, TILE_N, TILE_O, TILE_P]
        shape = shape[0:ndim]

        input = wp.array(rng.random(shape), dtype=float, requires_grad=True, device=device)
        output_full = wp.zeros(shape, dtype=float, device=device)
        output_padded = wp.zeros(shape, dtype=float, device=device)
        output_offset = wp.zeros(shape, dtype=float, device=device)

        with wp.Tape() as tape:
            wp.launch_tiled(
                kernel,
                dim=[1],
                inputs=[input, output_full, output_padded, output_offset],
                block_dim=TILE_DIM,
                device=device,
            )

        # construct a slice for the offset portion of the source/dest arrays
        src_slice = tuple(slice(TILE_OFFSET, dim) for dim in shape)
        dest_slice = tuple(slice(None, dim - TILE_OFFSET) for dim in shape)

        ref_full = input.numpy()
        ref_padded = np.zeros_like(ref_full)
        ref_padded[dest_slice] = ref_full[src_slice]

        ref_offset = np.zeros_like(ref_full)
        ref_offset[src_slice] = ref_full[dest_slice]

        assert_np_equal(output_full.numpy(), ref_full)
        assert_np_equal(output_padded.numpy(), ref_padded)
        assert_np_equal(output_offset.numpy(), ref_offset)

        output_full.grad = wp.ones_like(output_full)
        tape.backward()

        assert_np_equal(input.grad.numpy(), np.ones_like(input.grad.numpy()))

    return test


@wp.kernel
def tile_load_unaligned_kernel(
    input: wp.array2d(dtype=float),
    output: wp.array2d(dtype=float),
):
    t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(1, 1), storage="shared")
    wp.tile_store(output, t, offset=(1, 1))


def test_tile_load_unaligned(test, device):
    rng = np.random.default_rng(42)

    shape = [TILE_M + 1, TILE_N + 1]

    input = wp.array(rng.random(shape), dtype=float, requires_grad=True, device=device)
    output = wp.zeros(shape, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_load_unaligned_kernel,
            dim=[1],
            inputs=[input, output],
            block_dim=TILE_DIM,
            device=device,
        )

    # first row and column should be zero
    assert_np_equal(output.numpy()[0, :], np.zeros(TILE_N + 1))
    assert_np_equal(output.numpy()[:, 0], np.zeros(TILE_M + 1))

    # check output elements
    assert_np_equal(output.numpy()[1:, 1:], input.numpy()[1:, 1:])

    output.grad = wp.ones_like(output)
    tape.backward()

    expected_grad = np.ones_like(input.grad.numpy())
    expected_grad[0, :] = 0.0
    expected_grad[:, 0] = 0.0

    assert_np_equal(input.grad.numpy(), expected_grad)


# ----------------------------------------------------------------------------------------

TILE_SIZE = 4


@wp.kernel
def tile_extract_1d_kernel(input: wp.array1d(dtype=float), output: wp.array1d(dtype=float)):
    i = wp.tid()

    t = wp.tile_load(input, shape=TILE_SIZE)

    output[i] = t[i]


@wp.kernel
def tile_extract_2d_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
    i, j = wp.tid()

    t = wp.tile_load(input, shape=(TILE_SIZE, TILE_SIZE))

    output[i, j] = t[i, j]


@wp.kernel
def tile_extract_3d_kernel(input: wp.array3d(dtype=float), output: wp.array3d(dtype=float)):
    i, j, k = wp.tid()

    t = wp.tile_load(input, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE))

    output[i, j, k] = t[i, j, k]


@wp.kernel
def tile_extract_4d_kernel(input: wp.array4d(dtype=float), output: wp.array4d(dtype=float)):
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
def tile_assign_1d_kernel(input: wp.array1d(dtype=float), output: wp.array1d(dtype=float)):
    i = wp.tid()

    t = wp.tile_zeros(shape=(TILE_SIZE,), dtype=float)

    # assign to tile
    t[i] = input[i] * 2.0

    output[i] = t[i]


@wp.kernel
def tile_assign_2d_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
    i, j = wp.tid()

    t = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE), dtype=float)

    # assign to tile
    t[i, j] = input[i, j] * 2.0

    output[i, j] = t[i, j]


@wp.kernel
def tile_assign_3d_kernel(input: wp.array3d(dtype=float), output: wp.array3d(dtype=float)):
    i, j, k = wp.tid()

    t = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE), dtype=float)

    # assign to tile
    t[i, j, k] = input[i, j, k] * 2.0

    output[i, j, k] = t[i, j, k]


@wp.kernel
def tile_assign_4d_kernel(input: wp.array4d(dtype=float), output: wp.array4d(dtype=float)):
    i, j, k, l = wp.tid()

    t = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE), dtype=float)

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

    a = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(B, t=a, offset=(i * TILE_M, j * TILE_N))


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


add_function_test(TestTileLoad, "test_tile_load_1d", test_tile_load(tile_load_1d_kernel, 1), devices=devices)
add_function_test(TestTileLoad, "test_tile_load_2d", test_tile_load(tile_load_2d_kernel, 2), devices=devices)
add_function_test(TestTileLoad, "test_tile_load_3d", test_tile_load(tile_load_3d_kernel, 3), devices=devices)
add_function_test(TestTileLoad, "test_tile_load_4d", test_tile_load(tile_load_4d_kernel, 4), devices=devices)
add_function_test(TestTileLoad, "test_tile_load_unaligned", test_tile_load_unaligned, devices=devices)

add_function_test(TestTileLoad, "test_tile_extract_1d", test_tile_extract(tile_extract_1d_kernel, 1), devices=devices)
add_function_test(TestTileLoad, "test_tile_extract_2d", test_tile_extract(tile_extract_2d_kernel, 2), devices=devices)
add_function_test(TestTileLoad, "test_tile_extract_3d", test_tile_extract(tile_extract_3d_kernel, 3), devices=devices)
add_function_test(TestTileLoad, "test_tile_extract_4d", test_tile_extract(tile_extract_4d_kernel, 4), devices=devices)

add_function_test(TestTileLoad, "test_tile_assign_1d", test_tile_assign(tile_assign_1d_kernel, 1), devices=devices)
add_function_test(TestTileLoad, "test_tile_assign_2d", test_tile_assign(tile_assign_2d_kernel, 2), devices=devices)
add_function_test(TestTileLoad, "test_tile_assign_3d", test_tile_assign(tile_assign_3d_kernel, 3), devices=devices)
add_function_test(TestTileLoad, "test_tile_assign_4d", test_tile_assign(tile_assign_4d_kernel, 4), devices=devices)

add_function_test(TestTileLoad, "test_tile_load_fortran", test_tile_load_fortran, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
