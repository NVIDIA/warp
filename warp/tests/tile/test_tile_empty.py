# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# num threads per-tile
TILE_DIM = 64

# tile_empty allocates a tile of uninitialized data. The contract matches
# np.empty: contents are undefined, the user must overwrite every element
# before any read. These tests use only safe patterns (full-tile overwrite
# immediately after empty) and never assert on uninitialized contents.

TILE_EMPTY_M = wp.constant(16)


@wp.kernel
def tile_empty_register_float_1d_kernel(src: wp.array[float], dst: wp.array[float]):
    a = wp.tile_empty(shape=(TILE_EMPTY_M,), dtype=float, storage="register")
    a = wp.tile_load(src, shape=TILE_EMPTY_M)
    wp.tile_store(dst, a)


def test_tile_empty_register_float_1d(test, device):
    src_data = np.arange(TILE_EMPTY_M, dtype=np.float32)
    src = wp.array(src_data, device=device)
    dst = wp.zeros(TILE_EMPTY_M, dtype=wp.float32, device=device)

    wp.launch_tiled(
        tile_empty_register_float_1d_kernel,
        dim=[1],
        inputs=[src, dst],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(dst.numpy(), src_data)


@wp.kernel
def tile_empty_shared_float_1d_kernel(src: wp.array[float], dst: wp.array[float]):
    a = wp.tile_empty(shape=(TILE_EMPTY_M,), dtype=float, storage="shared")
    a = wp.tile_load(src, shape=TILE_EMPTY_M)
    wp.tile_store(dst, a)


def test_tile_empty_shared_float_1d(test, device):
    src_data = np.arange(TILE_EMPTY_M, dtype=np.float32)
    src = wp.array(src_data, device=device)
    dst = wp.zeros(TILE_EMPTY_M, dtype=wp.float32, device=device)

    wp.launch_tiled(
        tile_empty_shared_float_1d_kernel,
        dim=[1],
        inputs=[src, dst],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(dst.numpy(), src_data)


@wp.kernel
def tile_empty_scalar_shape_kernel(src: wp.array[float], dst: wp.array[float]):
    a = wp.tile_empty(shape=TILE_EMPTY_M, dtype=float)  # scalar shape, default storage
    a = wp.tile_load(src, shape=TILE_EMPTY_M)
    wp.tile_store(dst, a)


def test_tile_empty_scalar_shape(test, device):
    src_data = np.arange(TILE_EMPTY_M, dtype=np.float32)
    src = wp.array(src_data, device=device)
    dst = wp.zeros(TILE_EMPTY_M, dtype=wp.float32, device=device)

    wp.launch_tiled(
        tile_empty_scalar_shape_kernel,
        dim=[1],
        inputs=[src, dst],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(dst.numpy(), src_data)


TILE_EMPTY_2D_M = wp.constant(8)
TILE_EMPTY_2D_N = wp.constant(8)


@wp.kernel
def tile_empty_register_float_2d_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    a = wp.tile_empty(shape=(TILE_EMPTY_2D_M, TILE_EMPTY_2D_N), dtype=float, storage="register")
    a = wp.tile_load(src, shape=(TILE_EMPTY_2D_M, TILE_EMPTY_2D_N))
    wp.tile_store(dst, a)


def test_tile_empty_register_float_2d(test, device):
    src_data = np.arange(TILE_EMPTY_2D_M * TILE_EMPTY_2D_N, dtype=np.float32).reshape(TILE_EMPTY_2D_M, TILE_EMPTY_2D_N)
    src = wp.array(src_data, device=device)
    dst = wp.zeros((TILE_EMPTY_2D_M, TILE_EMPTY_2D_N), dtype=wp.float32, device=device)

    wp.launch_tiled(
        tile_empty_register_float_2d_kernel,
        dim=[1],
        inputs=[src, dst],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(dst.numpy(), src_data)


@wp.kernel
def tile_empty_register_int_kernel(src: wp.array[int], dst: wp.array[int]):
    a = wp.tile_empty(shape=(TILE_EMPTY_M,), dtype=int, storage="register")
    a = wp.tile_load(src, shape=TILE_EMPTY_M)
    wp.tile_store(dst, a)


def test_tile_empty_int(test, device):
    src_data = np.arange(TILE_EMPTY_M, dtype=np.int32)
    src = wp.array(src_data, device=device)
    dst = wp.zeros(TILE_EMPTY_M, dtype=wp.int32, device=device)

    wp.launch_tiled(
        tile_empty_register_int_kernel,
        dim=[1],
        inputs=[src, dst],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(dst.numpy(), src_data)


@wp.kernel
def tile_empty_register_vec3_kernel(src: wp.array[wp.vec3], dst: wp.array[wp.vec3]):
    a = wp.tile_empty(shape=(TILE_EMPTY_M,), dtype=wp.vec3, storage="register")
    a = wp.tile_load(src, shape=TILE_EMPTY_M)
    wp.tile_store(dst, a)


def test_tile_empty_vec3(test, device):
    src_data = np.arange(TILE_EMPTY_M * 3, dtype=np.float32).reshape(TILE_EMPTY_M, 3)
    src = wp.array(src_data, dtype=wp.vec3, device=device)
    dst = wp.zeros(TILE_EMPTY_M, dtype=wp.vec3, device=device)

    wp.launch_tiled(
        tile_empty_register_vec3_kernel,
        dim=[1],
        inputs=[src, dst],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(dst.numpy(), src_data)


@wp.kernel
def tile_empty_defaults_kernel(src: wp.array[float], dst: wp.array[float]):
    a = wp.tile_empty(shape=(TILE_EMPTY_M,))  # all defaults: dtype=float, storage="register"
    a = wp.tile_load(src, shape=TILE_EMPTY_M)
    wp.tile_store(dst, a)


def test_tile_empty_defaults(test, device):
    src_data = np.arange(TILE_EMPTY_M, dtype=np.float32)
    src = wp.array(src_data, device=device)
    dst = wp.zeros(TILE_EMPTY_M, dtype=wp.float32, device=device)

    wp.launch_tiled(
        tile_empty_defaults_kernel,
        dim=[1],
        inputs=[src, dst],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(dst.numpy(), src_data)


# tile_empty followed by full overwrite, with backward pass.
@wp.kernel
def tile_empty_register_backward_kernel(src: wp.array[float], dst: wp.array[float]):
    a = wp.tile_empty(shape=(TILE_EMPTY_M,), dtype=float, storage="register")
    a = wp.tile_load(src, shape=TILE_EMPTY_M)
    b = a * 2.0
    wp.tile_store(dst, b)


def test_tile_empty_register_backward(test, device):
    src_data = np.arange(TILE_EMPTY_M, dtype=np.float32)
    src = wp.array(src_data, requires_grad=True, device=device)
    dst = wp.zeros(TILE_EMPTY_M, dtype=wp.float32, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_empty_register_backward_kernel,
            dim=[1],
            inputs=[src, dst],
            block_dim=TILE_DIM,
            device=device,
        )

    assert_np_equal(dst.numpy(), src_data * 2.0)

    dst.grad = wp.ones_like(dst, device=device)
    tape.backward()

    assert_np_equal(src.grad.numpy(), np.full(TILE_EMPTY_M, 2.0, dtype=np.float32))


# tile_empty inside a runtime-bounded for-loop. The bound comes from a kernel
# argument so the loop is emitted as a dynamic begin_for/end_for rather than
# being statically unrolled, exercising the per-iteration adjoint reset of
# tile-typed locals.
TILE_EMPTY_LOOP_ITERS = 4


@wp.kernel
def tile_empty_in_loop_kernel(iters: int, src: wp.array[float], dst: wp.array[float]):
    for i in range(iters):
        a = wp.tile_empty(shape=(TILE_EMPTY_M,), dtype=float)
        a = wp.tile_load(src, shape=TILE_EMPTY_M, offset=i * TILE_EMPTY_M)
        wp.tile_store(dst, a, offset=i * TILE_EMPTY_M)


def test_tile_empty_in_loop(test, device):
    n = TILE_EMPTY_LOOP_ITERS * TILE_EMPTY_M
    src_data = np.arange(n, dtype=np.float32)
    src = wp.array(src_data, requires_grad=True, device=device)
    dst = wp.zeros(n, dtype=wp.float32, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_empty_in_loop_kernel,
            dim=[1],
            inputs=[TILE_EMPTY_LOOP_ITERS, src, dst],
            block_dim=TILE_DIM,
            device=device,
        )

    assert_np_equal(dst.numpy(), src_data)

    dst.grad = wp.ones_like(dst, device=device)
    tape.backward()

    assert_np_equal(src.grad.numpy(), np.ones(n, dtype=np.float32))


# Regression: tile_zeros / tile_ones / tile_full must accept their default
# dtype argument when invoked without explicit dtype=. The default is wrapped
# as Var(constant=...) by add_call; the dispatch_func must unwrap it.
@wp.kernel
def tile_zeros_default_dtype_kernel(src: wp.array[float], dst: wp.array[float]):
    a = wp.tile_zeros(shape=(TILE_EMPTY_M,))
    b = wp.tile_load(src, shape=TILE_EMPTY_M)
    c = a + b
    wp.tile_store(dst, c)


def test_tile_zeros_default_dtype(test, device):
    src_data = np.arange(TILE_EMPTY_M, dtype=np.float32)
    src = wp.array(src_data, requires_grad=True, device=device)
    dst = wp.zeros(TILE_EMPTY_M, dtype=wp.float32, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_zeros_default_dtype_kernel,
            dim=[1],
            inputs=[src, dst],
            block_dim=TILE_DIM,
            device=device,
        )

    assert_np_equal(dst.numpy(), src_data)

    dst.grad = wp.ones_like(dst, device=device)
    tape.backward()

    assert_np_equal(src.grad.numpy(), np.ones(TILE_EMPTY_M, dtype=np.float32))


devices = get_test_devices()


class TestTileEmpty(unittest.TestCase):
    pass


add_function_test(
    TestTileEmpty, "test_tile_empty_register_float_1d", test_tile_empty_register_float_1d, devices=devices
)
add_function_test(TestTileEmpty, "test_tile_empty_shared_float_1d", test_tile_empty_shared_float_1d, devices=devices)
add_function_test(TestTileEmpty, "test_tile_empty_scalar_shape", test_tile_empty_scalar_shape, devices=devices)
add_function_test(
    TestTileEmpty, "test_tile_empty_register_float_2d", test_tile_empty_register_float_2d, devices=devices
)
add_function_test(TestTileEmpty, "test_tile_empty_int", test_tile_empty_int, devices=devices)
add_function_test(TestTileEmpty, "test_tile_empty_vec3", test_tile_empty_vec3, devices=devices)
add_function_test(TestTileEmpty, "test_tile_empty_defaults", test_tile_empty_defaults, devices=devices)
add_function_test(
    TestTileEmpty, "test_tile_empty_register_backward", test_tile_empty_register_backward, devices=devices
)
add_function_test(TestTileEmpty, "test_tile_empty_in_loop", test_tile_empty_in_loop, devices=devices)
add_function_test(TestTileEmpty, "test_tile_zeros_default_dtype", test_tile_zeros_default_dtype, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
