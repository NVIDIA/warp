# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests general tile load and store behavior with 64-thread blocks.

Covers sliced, offset, unaligned, Fortran-layout, and scoped shared-memory
loads and stores in one through four dimensions. The module includes forward
and backward tests. Add a test here when it exercises general tile loading at
the standard 64-thread block size rather than an indexed, vectorized,
extraction, or assignment-specific path.
"""

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

# Non-power-of-two tile size — forces the scalar fallback path for all dimensions
# (not divisible by float4, so the vectorized path is skipped)
TILE_NPOT = wp.constant(17)
TILE_NPOT_3D = wp.constant(7)


@wp.kernel(enable_backward=False)
def tile_load_1d_sliced_scalar_kernel(
    input: wp.array1d[float],
    out_full: wp.array1d[float],
    out_sliced: wp.array1d[float],
):
    full = wp.tile_load(input, shape=TILE_NPOT, storage="shared")
    sliced = wp.tile_load(input[::2], shape=TILE_NPOT, storage="shared")

    wp.tile_store(out_full, full)
    wp.tile_store(out_sliced, sliced)


@wp.kernel(enable_backward=False)
def tile_load_2d_sliced_scalar_kernel(
    input: wp.array2d[float],
    out_full: wp.array2d[float],
    out_sliced: wp.array2d[float],
):
    full = wp.tile_load(input, shape=(TILE_NPOT, TILE_NPOT), storage="shared")
    sliced = wp.tile_load(input[::2, ::2], shape=(TILE_NPOT, TILE_NPOT), storage="shared")

    wp.tile_store(out_full, full)
    wp.tile_store(out_sliced, sliced)


@wp.kernel(enable_backward=False)
def tile_load_3d_sliced_scalar_kernel(
    input: wp.array3d[float],
    out_full: wp.array3d[float],
    out_sliced: wp.array3d[float],
):
    full = wp.tile_load(input, shape=(TILE_NPOT_3D, TILE_NPOT_3D, TILE_NPOT_3D), storage="shared")
    sliced = wp.tile_load(input[::2, ::2, ::2], shape=(TILE_NPOT_3D, TILE_NPOT_3D, TILE_NPOT_3D), storage="shared")

    wp.tile_store(out_full, full)
    wp.tile_store(out_sliced, sliced)


@wp.kernel(enable_backward=False)
def tile_store_1d_sliced_scalar_kernel(
    input: wp.array1d[float],
    output: wp.array1d[float],
):
    t = wp.tile_load(input, shape=TILE_NPOT, storage="shared")
    wp.tile_store(output[::2], t)


@wp.kernel(enable_backward=False)
def tile_store_2d_sliced_scalar_kernel(
    input: wp.array2d[float],
    output: wp.array2d[float],
):
    t = wp.tile_load(input, shape=(TILE_NPOT, TILE_NPOT), storage="shared")
    wp.tile_store(output[::2, ::2], t)


@wp.kernel
def tile_load_1d_kernel(
    input: wp.array1d[float],
    out_full: wp.array1d[float],
    out_padded: wp.array1d[float],
    out_sliced: wp.array1d[float],
    out_offset: wp.array1d[float],
):
    full0 = wp.tile_load(input, TILE_M)
    full1 = wp.tile_load(input, shape=TILE_M)
    full2 = wp.tile_load(input, shape=(TILE_M,))

    padded0 = wp.tile_load(input, TILE_M, TILE_OFFSET)
    padded1 = wp.tile_load(input, shape=TILE_M, offset=TILE_OFFSET)
    padded2 = wp.tile_load(input, shape=(TILE_M,), offset=(TILE_OFFSET,))

    sliced0 = wp.tile_load(input[::2], TILE_M)
    sliced1 = wp.tile_load(input[::2], shape=TILE_M)
    sliced2 = wp.tile_load(input[::2], shape=(TILE_M,))

    wp.tile_store(out_full, full0)
    wp.tile_store(out_padded, padded0)
    wp.tile_store(out_sliced, sliced0)
    wp.tile_store(out_offset, full0, offset=(TILE_OFFSET,))


@wp.kernel
def tile_load_2d_kernel(
    input: wp.array2d[float],
    out_full: wp.array2d[float],
    out_padded: wp.array2d[float],
    out_sliced: wp.array2d[float],
    out_offset: wp.array2d[float],
):
    full0 = wp.tile_load(input, shape=(TILE_M, TILE_N))
    padded0 = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(TILE_OFFSET, TILE_OFFSET))
    sliced0 = wp.tile_load(input[::2, ::2], shape=(TILE_M, TILE_N))

    wp.tile_store(out_full, full0)
    wp.tile_store(out_padded, padded0)
    wp.tile_store(out_sliced, sliced0)
    wp.tile_store(out_offset, full0, offset=(TILE_OFFSET, TILE_OFFSET))


@wp.kernel
def tile_load_3d_kernel(
    input: wp.array3d[float],
    out_full: wp.array3d[float],
    out_padded: wp.array3d[float],
    out_sliced: wp.array3d[float],
    out_offset: wp.array3d[float],
):
    full0 = wp.tile_load(input, shape=(TILE_M, TILE_N, TILE_O))
    padded0 = wp.tile_load(input, shape=(TILE_M, TILE_N, TILE_O), offset=(TILE_OFFSET, TILE_OFFSET, TILE_OFFSET))
    sliced0 = wp.tile_load(input[::2, ::2, ::2], shape=(TILE_M, TILE_N, TILE_O))

    wp.tile_store(out_full, full0)
    wp.tile_store(out_padded, padded0)
    wp.tile_store(out_sliced, sliced0)
    wp.tile_store(out_offset, full0, offset=(TILE_OFFSET, TILE_OFFSET, TILE_OFFSET))


@wp.kernel
def tile_load_4d_kernel(
    input: wp.array4d[float],
    out_full: wp.array4d[float],
    out_padded: wp.array4d[float],
    out_sliced: wp.array4d[float],
    out_offset: wp.array4d[float],
):
    full0 = wp.tile_load(input, shape=(TILE_M, TILE_N, TILE_O, TILE_P))
    padded0 = wp.tile_load(
        input, shape=(TILE_M, TILE_N, TILE_O, TILE_P), offset=(TILE_OFFSET, TILE_OFFSET, TILE_OFFSET, TILE_OFFSET)
    )
    sliced0 = wp.tile_load(input[::2, ::2, ::2, ::2], shape=(TILE_M, TILE_N, TILE_O, TILE_P))

    wp.tile_store(out_full, full0)
    wp.tile_store(out_padded, padded0)
    wp.tile_store(out_sliced, sliced0)
    wp.tile_store(out_offset, full0, offset=(TILE_OFFSET, TILE_OFFSET, TILE_OFFSET, TILE_OFFSET))


def test_tile_load(kernel, ndim):
    def test(test, device):
        rng = np.random.default_rng(42)

        shape = [TILE_M, TILE_N, TILE_O, TILE_P]
        shape = shape[0:ndim]

        input = wp.array(rng.random(shape), dtype=float, requires_grad=True, device=device)
        output_full = wp.zeros(shape, dtype=float, device=device)
        output_padded = wp.zeros(shape, dtype=float, device=device)
        output_sliced = wp.zeros(shape, dtype=float, device=device)
        output_offset = wp.zeros(shape, dtype=float, device=device)

        with wp.Tape() as tape:
            wp.launch_tiled(
                kernel,
                dim=[1],
                inputs=[input, output_full, output_padded, output_sliced, output_offset],
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

        # construct a slice for the source/dest sliced arrays
        src_slice = tuple(slice(0, dim, 2) for dim in shape)
        dest_slice = tuple(slice(0, (dim + 1) // 2) for dim in shape)

        ref_sliced = np.zeros_like(ref_full)
        ref_sliced[dest_slice] = ref_full[src_slice]

        assert_np_equal(output_full.numpy(), ref_full)
        assert_np_equal(output_padded.numpy(), ref_padded)
        assert_np_equal(output_sliced.numpy(), ref_sliced)
        assert_np_equal(output_offset.numpy(), ref_offset)

        output_full.grad = wp.ones_like(output_full)
        tape.backward()

        assert_np_equal(input.grad.numpy(), np.ones_like(input.grad.numpy()))

    return test


@wp.kernel
def tile_load_unaligned_kernel(
    input: wp.array2d[float],
    output: wp.array2d[float],
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


@wp.kernel(enable_backward=False)
def tile_load_aligned_small_kernel(
    input: wp.array2d[float],
    output: wp.array2d[float],
):
    t = wp.tile_load(input, shape=(3, 3), offset=(0, 0), storage="shared")
    wp.tile_store(output, t, offset=(0, 0))


# regression test for tiles that are smaller than sizeof(float4) in that last
# dimension but are aligned to float4. Did trigger the fast float4 path by accident.
def test_tile_load_aligned_small(test, device):
    rng = np.random.default_rng(42)

    shape = [TILE_M, TILE_N]

    input = wp.array(rng.random(shape), dtype=float, requires_grad=True, device=device)
    output = wp.zeros(shape, dtype=float, device=device)

    wp.launch_tiled(
        tile_load_aligned_small_kernel,
        dim=[1],
        inputs=[input, output],
        block_dim=TILE_DIM,
        device=device,
    )

    # zeros except for the 3x3 tile at 0, 0
    assert_np_equal(output.numpy()[3:, :], np.zeros((TILE_M - 3, TILE_N)))
    assert_np_equal(output.numpy()[:, 3:], np.zeros((TILE_M, TILE_N - 3)))

    # check output elements
    assert_np_equal(output.numpy()[:3, :3], input.numpy()[:3, :3])


TILE_WIDTH = 5
TILE_OFFSET_X = 0
TILE_OFFSET_Y = 8


@wp.kernel(enable_backward=False)
def test_tile_load_aligned_offset_unaligned_size_kernel(
    input: wp.array2d[float],
    output: wp.array2d[float],
):
    # Load a 5x5 tile from the input array starting at offset (0,8)
    # and store it in shared memory
    tile = wp.tile_load(input, shape=(TILE_WIDTH, TILE_WIDTH), offset=(TILE_OFFSET_X, TILE_OFFSET_Y), storage="shared")

    # Store the loaded tile back to the output array at the same offset
    wp.tile_store(output, tile, offset=(TILE_OFFSET_X, TILE_OFFSET_Y))


def test_tile_load_aligned_offset_unaligned_size(test, device):
    """Test loading a tile with aligned offset but unaligned size."""

    rng = np.random.default_rng(42)
    array_shape = [TILE_N, TILE_M]

    input_array = wp.array(rng.random(array_shape), dtype=float, requires_grad=True, device=device)
    output_array = wp.zeros(array_shape, dtype=float, device=device)

    wp.launch_tiled(
        test_tile_load_aligned_offset_unaligned_size_kernel,
        dim=[1],
        inputs=[input_array, output_array],
        block_dim=TILE_DIM,
        device=device,
    )

    # Region before the tile offset should be zeros
    assert_np_equal(output_array.numpy()[:TILE_WIDTH, :TILE_OFFSET_Y], np.zeros((TILE_WIDTH, TILE_OFFSET_Y)))

    # Region where the tile was loaded/stored should match input
    assert_np_equal(
        output_array.numpy()[:TILE_WIDTH, TILE_OFFSET_Y : TILE_OFFSET_Y + TILE_WIDTH],
        input_array.numpy()[:TILE_WIDTH, TILE_OFFSET_Y : TILE_OFFSET_Y + TILE_WIDTH],
    )

    # Region after the tile should be zeros
    remaining_width = TILE_M - (TILE_OFFSET_Y + TILE_WIDTH)
    assert_np_equal(
        output_array.numpy()[:TILE_WIDTH, TILE_OFFSET_Y + TILE_WIDTH :], np.zeros((TILE_WIDTH, remaining_width))
    )

    # Rows below the tile should all be zeros
    remaining_height = TILE_N - TILE_WIDTH
    assert_np_equal(output_array.numpy()[TILE_WIDTH:, :], np.zeros((remaining_height, TILE_M)))


@wp.kernel(enable_backward=False)
def test_tile_load_stride_unaligned_kernel(input: wp.array2d[wp.float32], output: wp.array2d[wp.float32]):
    tile = wp.tile_load(input, shape=(4, 4))
    wp.tile_store(output, tile)


# regression test for float4 aligned tiles that load from a source array with an incommensurate stride
def test_tile_load_stride_unaligned(test, device):
    DIM = 5
    input_np = np.eye(DIM) * 2.0
    input_array = wp.array(input_np, dtype=wp.float32, device=device)
    output_array = wp.zeros_like(input_array)

    wp.launch_tiled(
        test_tile_load_stride_unaligned_kernel,
        dim=(1, 1),
        inputs=[input_array],
        outputs=[output_array],
        block_dim=TILE_DIM,
        device=device,
    )

    input_np[DIM - 1, DIM - 1] = 0.0
    assert_np_equal(output_array.numpy(), input_np)


@wp.kernel
def tile_load_fortran_kernel(A: wp.array2d[float], B: wp.array2d[float]):
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

    assert_np_equal(A_wp.grad.numpy(), np.ones((M, N), dtype=float))


# ----------------------------------------------------------------------------------------


@wp.func
def test_tile_load_scoped_func(A: wp.array2d[float]):
    A_tile = wp.tile_load(A, shape=(TILE_DIM, TILE_DIM), offset=(0, 0), storage="shared")


@wp.kernel(enable_backward=False)
def test_tile_load_scoped_kernel(A: wp.array2d[float], B: wp.array2d[float]):
    test_tile_load_scoped_func(A)
    B_tile = wp.tile_load(B, shape=(TILE_DIM, TILE_DIM), offset=(0, 0), storage="shared")


def test_tile_load_scoped(test, device):
    """Regression test for 2D shared tiles that are allocated in a function then deallocated when they go out of scope

    Passes if it runs without errors
    """
    A = wp.ones((TILE_DIM, TILE_DIM), dtype=float, device=device)
    B = wp.ones((TILE_DIM, TILE_DIM), dtype=float, device=device)

    wp.launch_tiled(test_tile_load_scoped_kernel, dim=1, inputs=[A, B], block_dim=TILE_DIM, device=device)


def test_tile_load_sliced_scalar(kernel, ndim, tile_size=TILE_NPOT):
    """Test tile load from sliced (strided) arrays on the scalar path."""

    def test(test, device):
        rng = np.random.default_rng(42)

        shape = [tile_size] * ndim

        input = wp.array(rng.random(shape), dtype=float, device=device)
        output_full = wp.zeros(shape, dtype=float, device=device)
        output_sliced = wp.zeros(shape, dtype=float, device=device)

        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[input, output_full, output_sliced],
            block_dim=TILE_DIM,
            device=device,
        )

        ref_full = input.numpy()

        src_slice = tuple(slice(0, dim, 2) for dim in shape)
        dest_slice = tuple(slice(0, (dim + 1) // 2) for dim in shape)
        ref_sliced = np.zeros_like(ref_full)
        ref_sliced[dest_slice] = ref_full[src_slice]

        assert_np_equal(output_full.numpy(), ref_full)
        assert_np_equal(output_sliced.numpy(), ref_sliced)

    return test


def test_tile_store_sliced_scalar(kernel, ndim, tile_size=TILE_NPOT):
    """Test tile store to sliced (strided) destination arrays on the scalar path."""

    def test(test, device):
        rng = np.random.default_rng(42)

        shape = [tile_size] * ndim

        input = wp.array(rng.random(shape), dtype=float, device=device)
        # Output array is 2x the tile size per dim so output[::2] has room for the tile
        out_shape = [tile_size * 2] * ndim
        output = wp.zeros(out_shape, dtype=float, device=device)

        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[input, output],
            block_dim=TILE_DIM,
            device=device,
        )

        ref_input = input.numpy()
        ref_output = np.zeros(out_shape, dtype=np.float64)
        dest_slice = tuple(slice(0, tile_size * 2, 2) for _ in range(ndim))
        ref_output[dest_slice] = ref_input

        assert_np_equal(output.numpy(), ref_output)

    return test


devices = get_test_devices()


class TestTileLoad(unittest.TestCase):
    pass


add_function_test(
    TestTileLoad,
    "test_tile_load_1d_sliced_scalar",
    test_tile_load_sliced_scalar(tile_load_1d_sliced_scalar_kernel, 1),
    devices=devices,
)
add_function_test(
    TestTileLoad,
    "test_tile_load_2d_sliced_scalar",
    test_tile_load_sliced_scalar(tile_load_2d_sliced_scalar_kernel, 2),
    devices=devices,
)
add_function_test(
    TestTileLoad,
    "test_tile_load_3d_sliced_scalar",
    test_tile_load_sliced_scalar(tile_load_3d_sliced_scalar_kernel, 3, tile_size=TILE_NPOT_3D),
    devices=devices,
)
add_function_test(
    TestTileLoad,
    "test_tile_store_1d_sliced_scalar",
    test_tile_store_sliced_scalar(tile_store_1d_sliced_scalar_kernel, 1),
    devices=devices,
)
add_function_test(
    TestTileLoad,
    "test_tile_store_2d_sliced_scalar",
    test_tile_store_sliced_scalar(tile_store_2d_sliced_scalar_kernel, 2),
    devices=devices,
)
add_function_test(TestTileLoad, "test_tile_load_1d", test_tile_load(tile_load_1d_kernel, 1), devices=devices)
add_function_test(TestTileLoad, "test_tile_load_2d", test_tile_load(tile_load_2d_kernel, 2), devices=devices)
add_function_test(TestTileLoad, "test_tile_load_3d", test_tile_load(tile_load_3d_kernel, 3), devices=devices)
add_function_test(TestTileLoad, "test_tile_load_4d", test_tile_load(tile_load_4d_kernel, 4), devices=devices)
add_function_test(TestTileLoad, "test_tile_load_unaligned", test_tile_load_unaligned, devices=devices)
add_function_test(TestTileLoad, "test_tile_load_aligned_small", test_tile_load_aligned_small, devices=devices)
add_function_test(
    TestTileLoad,
    "test_tile_load_aligned_offset_unaligned_size",
    test_tile_load_aligned_offset_unaligned_size,
    devices=devices,
)
add_function_test(TestTileLoad, "test_tile_load_stride_unaligned", test_tile_load_stride_unaligned, devices=devices)
add_function_test(TestTileLoad, "test_tile_load_fortran", test_tile_load_fortran, devices=devices)

add_function_test(TestTileLoad, "test_tile_load_scoped", test_tile_load_scoped, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
