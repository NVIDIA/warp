# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

MIB = 1024**2
GIB = 1024**3
INT32_MAX = 2**31 - 1
INT32_OVERFLOW_BYTE_OFFSET = INT32_MAX + 1
BACKING_BYTES = 2 * GIB + 256 * MIB
BACKING_SHAPE = (GIB + 128 * MIB, 2)
BACKING_SENTINEL_VALUE = 11

SHARED_TILE_SIZE = 3
SHARED_BYTE_STRIDE = 4096
SHARED_LOAD_OFFSET = INT32_OVERFLOW_BYTE_OFFSET // SHARED_BYTE_STRIDE

DENSE_LOAD_BYTE_OFFSET = INT32_OVERFLOW_BYTE_OFFSET + 32 * MIB
DENSE_STORE_BYTE_OFFSET = INT32_OVERFLOW_BYTE_OFFSET + 64 * MIB
ALIGNED_STORE_BYTE_OFFSET = INT32_OVERFLOW_BYTE_OFFSET + 96 * MIB
MAT33_BYTE_OFFSET = INT32_OVERFLOW_BYTE_OFFSET + 128 * MIB

TILE_BLOCK_DIM = 32


def shared_expected_byte_offsets():
    return tuple((SHARED_LOAD_OFFSET + i) * SHARED_BYTE_STRIDE for i in range(SHARED_TILE_SIZE))


def shared_expected_values():
    return np.array([22, 33, 99], dtype=np.uint8)


def shared_store_values():
    return np.array([122, 133, 144], dtype=np.uint8)


def shared_2d_store_values():
    return np.array([[155], [166], [177]], dtype=np.uint8)


def indexed_axis1_store_values():
    return np.array([[188, 199, 211]], dtype=np.uint8)


def indexed_atomic_base_values():
    return np.array([10, 20, 30], dtype=np.int32)


def indexed_atomic_add_values():
    return np.array([1, 2, 3], dtype=np.int32)


def align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def dense_float32_case():
    values = np.array([[1.25, 2.5, 3.75, 5.0]], dtype=np.float32)
    row = DENSE_LOAD_BYTE_OFFSET // values.nbytes
    return row, (row + 1, values.shape[1]), row * values.nbytes, values


def dense_float32_store_case():
    values = np.array([[6.25, 7.5, 8.75, 10.0]], dtype=np.float32)
    row = DENSE_STORE_BYTE_OFFSET // values.nbytes
    return row, (row + 1, values.shape[1]), row * values.nbytes, values


def dense_float32_aligned_store_case():
    values = np.array([[12.25, 13.5, 14.75, 16.0]], dtype=np.float32)
    row = ALIGNED_STORE_BYTE_OFFSET // values.nbytes
    return row, (row + 1, values.shape[1]), row * values.nbytes, values


def mat33_case():
    element_size = wp.types.type_size_in_bytes(wp.mat33f)
    byte_offset = align_up(MAT33_BYTE_OFFSET, element_size)
    row = byte_offset // element_size
    return row, (row + 1, 1), row * element_size


def mat33_store_values():
    return np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) + np.float32(0.5)


def launch_tile(kernel, device, inputs):
    wp.launch_tiled(kernel, dim=[1], inputs=inputs, block_dim=TILE_BLOCK_DIM, device=device)


@wp.kernel(enable_backward=False)
def load_shared_1d_stride_step_wrap(
    src: wp.array[wp.uint8],
    dst: wp.array[wp.uint8],
    offset: int,
):
    tile = wp.tile_load(src, shape=SHARED_TILE_SIZE, offset=offset, storage="shared")
    wp.tile_store(dst, tile)


@wp.kernel(enable_backward=False)
def load_shared_2d_scalar_iterator(
    src: wp.array2d[wp.uint8],
    dst: wp.array2d[wp.uint8],
    row: int,
):
    tile = wp.tile_load(src, shape=(3, 1), offset=(row, 0), storage="shared")
    wp.tile_store(dst, tile)


@wp.kernel(enable_backward=False)
def load_shared_2d_vectorized_dense(
    src: wp.array2d[wp.float32],
    dst: wp.array2d[wp.float32],
    row: int,
):
    tile = wp.tile_load(src, shape=(1, 4), offset=(row, 0), storage="shared")
    wp.tile_store(dst, tile)


@wp.kernel(enable_backward=False)
def load_shared_2d_vectorized_dense_aligned(
    src: wp.array2d[wp.float32],
    dst: wp.array2d[wp.float32],
    row: int,
):
    tile = wp.tile_load(src, shape=(1, 4), offset=(row, 0), storage="shared", aligned=True)
    wp.tile_store(dst, tile)


@wp.kernel(enable_backward=False)
def load_shared_2d_coalesced_large_element(
    src: wp.array2d[wp.mat33f],
    dst: wp.array2d[wp.mat33f],
    row: int,
):
    tile = wp.tile_load(src, shape=(1, 1), offset=(row, 0), storage="shared")
    wp.tile_store(dst, tile)


@wp.kernel(enable_backward=False)
def load_indexed_axis0_large_stride(
    src: wp.array2d[wp.uint8],
    dst: wp.array2d[wp.uint8],
    row: int,
):
    indices = wp.tile_arange(3, dtype=int, storage="shared")
    tile = wp.tile_load_indexed(src, indices=indices, shape=(3, 1), offset=(row, 0), axis=0, storage="shared")
    wp.tile_store(dst, tile)


@wp.kernel(enable_backward=False)
def load_indexed_axis1_large_base(
    src: wp.array2d[wp.uint8],
    dst: wp.array2d[wp.uint8],
    row: int,
):
    indices = wp.tile_arange(3, dtype=int, storage="shared")
    tile = wp.tile_load_indexed(src, indices=indices, shape=(1, 3), offset=(0, row), axis=1, storage="shared")
    wp.tile_store(dst, tile)


@wp.kernel(enable_backward=False)
def store_shared_1d_stride_step_wrap(
    src: wp.array[wp.uint8],
    dst: wp.array[wp.uint8],
    offset: int,
):
    tile = wp.tile_load(src, shape=SHARED_TILE_SIZE, storage="shared")
    wp.tile_store(dst, tile, offset=offset)


@wp.kernel(enable_backward=False)
def store_shared_2d_scalar_iterator(
    src: wp.array2d[wp.uint8],
    dst: wp.array2d[wp.uint8],
    row: int,
):
    tile = wp.tile_load(src, shape=(3, 1), storage="shared")
    wp.tile_store(dst, tile, offset=(row, 0))


@wp.kernel(enable_backward=False)
def store_shared_2d_vectorized_dense(
    src: wp.array2d[wp.float32],
    dst: wp.array2d[wp.float32],
    row: int,
):
    tile = wp.tile_load(src, shape=(1, 4), storage="shared")
    wp.tile_store(dst, tile, offset=(row, 0))


@wp.kernel(enable_backward=False)
def store_shared_2d_vectorized_dense_aligned(
    src: wp.array2d[wp.float32],
    dst: wp.array2d[wp.float32],
    row: int,
):
    tile = wp.tile_load(src, shape=(1, 4), storage="shared")
    wp.tile_store(dst, tile, offset=(row, 0), aligned=True)


@wp.kernel(enable_backward=False)
def store_shared_2d_coalesced_large_element(
    src: wp.array2d[wp.mat33f],
    dst: wp.array2d[wp.mat33f],
    row: int,
):
    tile = wp.tile_load(src, shape=(1, 1), storage="shared")
    wp.tile_store(dst, tile, offset=(row, 0))


@wp.kernel(enable_backward=False)
def store_indexed_axis0_large_stride(
    src: wp.array2d[wp.uint8],
    dst: wp.array2d[wp.uint8],
    row: int,
):
    indices = wp.tile_arange(3, dtype=int, storage="shared")
    tile = wp.tile_load(src, shape=(3, 1), storage="shared")
    wp.tile_store_indexed(dst, indices=indices, t=tile, offset=(row, 0), axis=0)


@wp.kernel(enable_backward=False)
def store_indexed_axis1_large_stride(
    src: wp.array2d[wp.uint8],
    dst: wp.array2d[wp.uint8],
    row: int,
):
    indices = wp.tile_arange(3, dtype=int, storage="shared")
    tile = wp.tile_load(src, shape=(1, 3), storage="shared")
    wp.tile_store_indexed(dst, indices=indices, t=tile, offset=(0, row), axis=1)


@wp.kernel(enable_backward=False)
def atomic_add_indexed_axis0_large_stride(
    src: wp.array2d[wp.int32],
    dst: wp.array2d[wp.int32],
    row: int,
):
    indices = wp.tile_arange(3, dtype=int, storage="shared")
    tile = wp.tile_load(src, shape=(3, 1), storage="shared")
    wp.tile_atomic_add_indexed(dst, indices=indices, t=tile, offset=(row, 0), axis=0)


@wp.kernel(enable_backward=False)
def atomic_add_indexed_axis1_large_stride(
    src: wp.array2d[wp.int32],
    dst: wp.array2d[wp.int32],
    row: int,
):
    indices = wp.tile_arange(3, dtype=int, storage="shared")
    tile = wp.tile_load(src, shape=(1, 3), storage="shared")
    wp.tile_atomic_add_indexed(dst, indices=indices, t=tile, offset=(0, row), axis=1)


def make_backing_view(device, backing, dtype, shape, strides=None):
    kwargs = {"ptr": backing.ptr, "dtype": dtype, "shape": shape, "device": device}
    if strides is not None:
        kwargs["strides"] = strides
    return wp.array(**kwargs)


def make_shared_1d_stride_step_view(device, backing):
    return make_backing_view(device, backing, wp.uint8, (SHARED_LOAD_OFFSET + SHARED_TILE_SIZE,), (SHARED_BYTE_STRIDE,))


def make_shared_2d_scalar_iterator_view(device, backing):
    return make_backing_view(
        device, backing, wp.uint8, (SHARED_LOAD_OFFSET + SHARED_TILE_SIZE, 1), (SHARED_BYTE_STRIDE, 1)
    )


def make_shared_2d_axis1_indexed_view(device, backing):
    return make_backing_view(
        device, backing, wp.uint8, (1, SHARED_LOAD_OFFSET + SHARED_TILE_SIZE), (1, SHARED_BYTE_STRIDE)
    )


def make_vectorized_dense_float32_view(device, backing):
    _, shape, _, _ = dense_float32_aligned_store_case()
    return make_backing_view(device, backing, wp.float32, shape)


def make_coalesced_mat33_view(device, backing):
    _, shape, _ = mat33_case()
    return make_backing_view(device, backing, wp.mat33f, shape)


def make_indexed_axis0_int32_view(device, backing):
    return make_backing_view(
        device, backing, wp.int32, (SHARED_LOAD_OFFSET + SHARED_TILE_SIZE, 1), (SHARED_BYTE_STRIDE, 4)
    )


def make_indexed_axis1_int32_view(device, backing):
    return make_backing_view(
        device, backing, wp.int32, (1, SHARED_LOAD_OFFSET + SHARED_TILE_SIZE), (4, SHARED_BYTE_STRIDE)
    )


def write_cuda_byte(device, ptr, byte_offset, value):
    src = wp.array(np.array([value], dtype=np.uint8), dtype=wp.uint8, device=device)
    dst = wp.array(ptr=ptr + byte_offset, dtype=wp.uint8, shape=(1,), device=device)
    wp.copy(dst, src)


def write_cuda_float32_tile(device, ptr, byte_offset, values):
    src = wp.array(values, dtype=wp.float32, device=device)
    dst = wp.array(ptr=ptr + byte_offset, dtype=wp.float32, shape=values.shape, device=device)
    wp.copy(dst, src)


def write_cuda_int32(device, ptr, byte_offset, value):
    src = wp.array(np.array([value], dtype=np.int32), dtype=wp.int32, device=device)
    dst = wp.array(ptr=ptr + byte_offset, dtype=wp.int32, shape=(1,), device=device)
    wp.copy(dst, src)


def read_cuda_byte(device, ptr, byte_offset):
    src = wp.array(ptr=ptr + byte_offset, dtype=wp.uint8, shape=(1,), device=device)
    return int(src.numpy()[0])


def read_cuda_int32(device, ptr, byte_offset):
    src = wp.array(ptr=ptr + byte_offset, dtype=wp.int32, shape=(1,), device=device)
    return int(src.numpy()[0])


def read_cuda_float32_tile(device, ptr, byte_offset, shape):
    src = wp.array(ptr=ptr + byte_offset, dtype=wp.float32, shape=shape, device=device)
    return src.numpy()


def read_cuda_mat33_tile(device, ptr, byte_offset):
    src = wp.array(ptr=ptr + byte_offset, dtype=wp.mat33f, shape=(1, 1), device=device)
    return src.numpy()


def seed_gpu_large_offset_backing(device, backing):
    _, _, dense_byte_offset, dense_values = dense_float32_case()
    _, _, mat33_byte_offset = mat33_case()

    write_cuda_byte(device, backing.ptr, 0, BACKING_SENTINEL_VALUE)

    values = dict(zip(shared_expected_byte_offsets(), shared_expected_values(), strict=True))
    values[mat33_byte_offset] = 66
    for byte_offset, value in values.items():
        write_cuda_byte(device, backing.ptr, byte_offset, value)

    write_cuda_float32_tile(device, backing.ptr, dense_byte_offset, dense_values)


def seed_indexed_atomic_values(device, backing, base_values):
    for byte_offset, value in zip(shared_expected_byte_offsets(), base_values, strict=True):
        write_cuda_int32(device, backing.ptr, byte_offset, int(value))


def read_indexed_atomic_values(device, backing):
    return np.array(
        [read_cuda_int32(device, backing.ptr, byte_offset) for byte_offset in shared_expected_byte_offsets()],
        dtype=np.int32,
    )


def read_shared_u8_values(device, backing):
    return np.array(
        [read_cuda_byte(device, backing.ptr, byte_offset) for byte_offset in shared_expected_byte_offsets()],
        dtype=np.uint8,
    )


def assert_gpu_backing_sentinels(test, device, backing):
    _, _, mat33_byte_offset = mat33_case()

    test.assertEqual(read_cuda_byte(device, backing.ptr, 0), BACKING_SENTINEL_VALUE)
    for byte_offset, value in zip(shared_expected_byte_offsets(), shared_expected_values(), strict=True):
        test.assertEqual(read_cuda_byte(device, backing.ptr, byte_offset), int(value))
    test.assertEqual(read_cuda_byte(device, backing.ptr, mat33_byte_offset), 66)


def make_gpu_large_offset_backing(device):
    try:
        return wp.empty(BACKING_SHAPE, dtype=wp.uint8, device=device)
    except RuntimeError as exc:
        raise unittest.SkipTest(f"Failed to allocate {BACKING_BYTES} bytes on {device}") from exc


def test_shared_large_offset_loads_and_stores(test, device):
    dense_row, _, dense_byte_offset, dense_values = dense_float32_case()
    dense_store_row, _, dense_store_byte_offset, dense_store_values = dense_float32_store_case()
    aligned_store_row, _, aligned_store_byte_offset, aligned_store_values = dense_float32_aligned_store_case()
    mat33_row, _, mat33_byte_offset = mat33_case()
    mat33_size = wp.types.type_size_in_bytes(wp.mat33f)
    expected_values = shared_expected_values()
    store_values = shared_store_values()
    store_2d_values = shared_2d_store_values()
    store_axis1_values = indexed_axis1_store_values()
    atomic_base_values = indexed_atomic_base_values()
    atomic_add_values = indexed_atomic_add_values()
    mat33_values = mat33_store_values()

    test.assertEqual(BACKING_BYTES, 2 * GIB + 256 * MIB)
    test.assertEqual(BACKING_SHAPE[0] * BACKING_SHAPE[1], BACKING_BYTES)
    for byte_offset in shared_expected_byte_offsets():
        test.assertGreater(byte_offset, INT32_MAX)
        test.assertLess(byte_offset, BACKING_BYTES)
    test.assertGreater(dense_byte_offset, INT32_MAX)
    test.assertLessEqual(dense_byte_offset + dense_values.nbytes, BACKING_BYTES)
    test.assertEqual(dense_byte_offset % 16, 0)
    test.assertGreater(dense_store_byte_offset, INT32_MAX)
    test.assertLessEqual(dense_store_byte_offset + dense_store_values.nbytes, BACKING_BYTES)
    test.assertGreater(aligned_store_byte_offset, INT32_MAX)
    test.assertLessEqual(aligned_store_byte_offset + aligned_store_values.nbytes, BACKING_BYTES)
    test.assertEqual(aligned_store_byte_offset % 16, 0)
    test.assertGreater(mat33_byte_offset, INT32_MAX)
    test.assertLessEqual(mat33_byte_offset + mat33_size, BACKING_BYTES)

    backing = make_gpu_large_offset_backing(device)
    try:
        seed_gpu_large_offset_backing(device, backing)
        assert_gpu_backing_sentinels(test, device, backing)

        dst_1d = wp.empty((3,), dtype=wp.uint8, device=device)
        launch_tile(
            load_shared_1d_stride_step_wrap,
            device,
            [make_shared_1d_stride_step_view(device, backing), dst_1d, SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(dst_1d.numpy(), expected_values)

        dst_axis0 = wp.empty((3, 1), dtype=wp.uint8, device=device)
        launch_tile(
            load_shared_2d_scalar_iterator,
            device,
            [make_shared_2d_scalar_iterator_view(device, backing), dst_axis0, SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(dst_axis0.numpy(), expected_values.reshape(3, 1))

        dst_dense = wp.empty((1, 4), dtype=wp.float32, device=device)
        launch_tile(
            load_shared_2d_vectorized_dense,
            device,
            [make_vectorized_dense_float32_view(device, backing), dst_dense, dense_row],
        )
        np.testing.assert_allclose(dst_dense.numpy(), dense_values)

        dst_aligned_dense = wp.empty((1, 4), dtype=wp.float32, device=device)
        launch_tile(
            load_shared_2d_vectorized_dense_aligned,
            device,
            [make_vectorized_dense_float32_view(device, backing), dst_aligned_dense, dense_row],
        )
        np.testing.assert_allclose(dst_aligned_dense.numpy(), dense_values)

        dst_mat33 = wp.empty((1, 1), dtype=wp.mat33f, device=device)
        launch_tile(
            load_shared_2d_coalesced_large_element,
            device,
            [make_coalesced_mat33_view(device, backing), dst_mat33, mat33_row],
        )
        test.assertEqual(read_cuda_byte(device, dst_mat33.ptr, 0), 66)

        dst_indexed_axis0 = wp.empty((3, 1), dtype=wp.uint8, device=device)
        launch_tile(
            load_indexed_axis0_large_stride,
            device,
            [make_shared_2d_scalar_iterator_view(device, backing), dst_indexed_axis0, SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(dst_indexed_axis0.numpy(), expected_values.reshape(3, 1))

        dst_indexed_axis1 = wp.empty((1, 3), dtype=wp.uint8, device=device)
        launch_tile(
            load_indexed_axis1_large_base,
            device,
            [make_shared_2d_axis1_indexed_view(device, backing), dst_indexed_axis1, SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(dst_indexed_axis1.numpy(), expected_values.reshape(1, 3))

        src_store_1d = wp.array(store_values, dtype=wp.uint8, device=device)
        launch_tile(
            store_shared_1d_stride_step_wrap,
            device,
            [src_store_1d, make_shared_1d_stride_step_view(device, backing), SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(read_shared_u8_values(device, backing), store_values)
        test.assertEqual(read_cuda_byte(device, backing.ptr, 0), BACKING_SENTINEL_VALUE)

        src_store_2d = wp.array(store_2d_values, dtype=wp.uint8, device=device)
        launch_tile(
            store_shared_2d_scalar_iterator,
            device,
            [src_store_2d, make_shared_2d_scalar_iterator_view(device, backing), SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(read_shared_u8_values(device, backing), store_2d_values.reshape(3))
        test.assertEqual(read_cuda_byte(device, backing.ptr, 0), BACKING_SENTINEL_VALUE)

        src_dense_store = wp.array(dense_store_values, dtype=wp.float32, device=device)
        launch_tile(
            store_shared_2d_vectorized_dense,
            device,
            [src_dense_store, make_vectorized_dense_float32_view(device, backing), dense_store_row],
        )
        np.testing.assert_allclose(
            read_cuda_float32_tile(device, backing.ptr, dense_store_byte_offset, dense_store_values.shape),
            dense_store_values,
        )

        src_aligned_store = wp.array(aligned_store_values, dtype=wp.float32, device=device)
        launch_tile(
            store_shared_2d_vectorized_dense_aligned,
            device,
            [src_aligned_store, make_vectorized_dense_float32_view(device, backing), aligned_store_row],
        )
        np.testing.assert_allclose(
            read_cuda_float32_tile(device, backing.ptr, aligned_store_byte_offset, aligned_store_values.shape),
            aligned_store_values,
        )

        src_mat33_store = wp.array(mat33_values, dtype=wp.mat33f, device=device)
        launch_tile(
            store_shared_2d_coalesced_large_element,
            device,
            [src_mat33_store, make_coalesced_mat33_view(device, backing), mat33_row],
        )
        np.testing.assert_allclose(read_cuda_mat33_tile(device, backing.ptr, mat33_byte_offset), mat33_values)

        src_store_axis0 = wp.array(store_2d_values, dtype=wp.uint8, device=device)
        launch_tile(
            store_indexed_axis0_large_stride,
            device,
            [src_store_axis0, make_shared_2d_scalar_iterator_view(device, backing), SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(read_shared_u8_values(device, backing), store_2d_values.reshape(3))
        test.assertEqual(read_cuda_byte(device, backing.ptr, 0), BACKING_SENTINEL_VALUE)

        src_store_axis1 = wp.array(store_axis1_values, dtype=wp.uint8, device=device)
        launch_tile(
            store_indexed_axis1_large_stride,
            device,
            [src_store_axis1, make_shared_2d_axis1_indexed_view(device, backing), SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(read_shared_u8_values(device, backing), store_axis1_values.reshape(3))
        test.assertEqual(read_cuda_byte(device, backing.ptr, 0), BACKING_SENTINEL_VALUE)

        seed_indexed_atomic_values(device, backing, atomic_base_values)
        src_atomic_axis0 = wp.array(atomic_add_values.reshape(3, 1), dtype=wp.int32, device=device)
        launch_tile(
            atomic_add_indexed_axis0_large_stride,
            device,
            [src_atomic_axis0, make_indexed_axis0_int32_view(device, backing), SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(
            read_indexed_atomic_values(device, backing), atomic_base_values + atomic_add_values
        )
        test.assertEqual(read_cuda_byte(device, backing.ptr, 0), BACKING_SENTINEL_VALUE)

        seed_indexed_atomic_values(device, backing, atomic_base_values)
        src_atomic_axis1 = wp.array(atomic_add_values.reshape(1, 3), dtype=wp.int32, device=device)
        launch_tile(
            atomic_add_indexed_axis1_large_stride,
            device,
            [src_atomic_axis1, make_indexed_axis1_int32_view(device, backing), SHARED_LOAD_OFFSET],
        )
        np.testing.assert_array_equal(
            read_indexed_atomic_values(device, backing), atomic_base_values + atomic_add_values
        )
        test.assertEqual(read_cuda_byte(device, backing.ptr, 0), BACKING_SENTINEL_VALUE)
    finally:
        try:
            wp.synchronize_device(device)
        finally:
            del backing
            gc.collect()


cuda_devices = get_selected_cuda_test_devices(mode="basic")


class TestTileLargeOffsets(unittest.TestCase):
    pass


add_function_test(
    TestTileLargeOffsets,
    "test_shared_large_offset_loads_and_stores",
    test_shared_large_offset_loads_and_stores,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
