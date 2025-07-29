# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 64


@wp.kernel
def tile_copy_1d_kernel(A: wp.array(dtype=float), B: wp.array(dtype=float)):
    # tile index
    i = wp.tid()

    a = wp.tile_load(A, shape=TILE_N, offset=i * TILE_N)
    wp.tile_store(B, a, offset=i * TILE_N)


def test_tile_copy_1d(test, device):
    rng = np.random.default_rng(42)

    N = TILE_N * 5

    A = rng.random((N), dtype=np.float32)
    B = rng.random((N), dtype=np.float32)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_copy_1d_kernel,
            dim=[int(N / TILE_N)],
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


@wp.kernel
def tile_copy_2d_kernel(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float)):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(B, a, offset=(i * TILE_M, j * TILE_N))


def test_tile_copy_2d(test, device):
    rng = np.random.default_rng(42)

    M = TILE_M * 7
    N = TILE_N * 5

    A = rng.random((M, N), dtype=np.float32)
    B = rng.random((M, N), dtype=np.float32)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_copy_2d_kernel,
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


@wp.func
def unary_func(x: wp.float32):
    return wp.sin(x)


@wp.func
def unary_func(x: wp.float64):
    return wp.sin(x)


@wp.kernel
def tile_unary_map_user_func(input: wp.array2d(dtype=Any), output: wp.array2d(dtype=Any)):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    sa = wp.tile_map(unary_func, a)

    wp.tile_store(output, sa, offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def tile_unary_map_builtin_func(input: wp.array2d(dtype=Any), output: wp.array2d(dtype=Any)):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    sa = wp.tile_map(wp.sin, a)

    wp.tile_store(output, sa, offset=(i * TILE_M, j * TILE_N))


def test_tile_unary_map(test, device):
    rng = np.random.default_rng(42)

    M = TILE_M * 7
    N = TILE_N * 5

    def run(kernel, dtype):
        A = rng.random((M, N), dtype=dtype)
        B = np.sin(A)

        A_grad = np.cos(A)

        A_wp = wp.array(A, requires_grad=True, device=device)
        B_wp = wp.zeros_like(A_wp, requires_grad=True, device=device)

        with wp.Tape() as tape:
            wp.launch_tiled(
                kernel,
                dim=[int(M / TILE_M), int(N / TILE_N)],
                inputs=[A_wp, B_wp],
                block_dim=TILE_DIM,
                device=device,
            )

        tol = 1.0e-6 if dtype == np.float64 else 1.0e-4

        # verify forward pass
        assert_np_equal(B_wp.numpy(), B, tol=tol)

        # verify backward pass
        B_wp.grad = wp.ones_like(B_wp, device=device)
        tape.backward()

        assert_np_equal(A_wp.grad.numpy(), A_grad, tol=tol)

    dtypes = [np.float32, np.float64]

    for dtype in dtypes:
        run(tile_unary_map_user_func, dtype)
        run(tile_unary_map_builtin_func, dtype)


@wp.func
def unary_func_mixed_types(x: int) -> float:
    return wp.sin(float(x))


@wp.kernel
def tile_unary_map_mixed_types(input: wp.array2d(dtype=int), output: wp.array2d(dtype=float)):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    sa = wp.tile_map(unary_func_mixed_types, a)

    wp.tile_store(output, sa, offset=(i * TILE_M, j * TILE_N))


def test_tile_unary_map_mixed_types(test, device):
    rng = np.random.default_rng(42)

    M = TILE_M * 7
    N = TILE_N * 5

    A = rng.integers(0, 100, size=(M, N), dtype=np.int32)
    B = np.sin(A.astype(np.float32))

    A_grad = np.cos(A.astype(np.float32))

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.zeros((M, N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_unary_map_mixed_types,
            dim=[int(M / TILE_M), int(N / TILE_N)],
            inputs=[A_wp, B_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    # verify forward pass
    assert_np_equal(B_wp.numpy(), B, tol=1.0e-4)

    # verify backward pass
    B_wp.grad = wp.ones_like(B_wp, device=device)
    tape.backward()

    # The a gradients are now stored as ints and can't capture the correct values
    # assert_np_equal(A_wp.grad.numpy(), A_grad, tol=1.0e-6)


@wp.func
def binary_func(x: wp.float32, y: wp.float32):
    return x + y


@wp.func
def binary_func(x: wp.float64, y: wp.float64):
    return x + y


@wp.kernel
def tile_binary_map_user_func(
    input_a: wp.array2d(dtype=Any), input_b: wp.array2d(dtype=Any), output: wp.array2d(dtype=Any)
):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    sa = wp.tile_map(binary_func, a, b)

    wp.tile_store(output, sa, offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def tile_binary_map_builtin_func(
    input_a: wp.array2d(dtype=Any), input_b: wp.array2d(dtype=Any), output: wp.array2d(dtype=Any)
):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    sa = wp.tile_map(wp.add, a, b)

    wp.tile_store(output, sa, offset=(i * TILE_M, j * TILE_N))


def test_tile_binary_map(test, device):
    rng = np.random.default_rng(42)

    M = TILE_M * 7
    N = TILE_N * 5

    def run(kernel, dtype):
        A = rng.random((M, N), dtype=dtype)
        B = rng.random((M, N), dtype=dtype)
        C = A + B

        A_grad = np.ones_like(A)
        B_grad = np.ones_like(B)

        A_wp = wp.array(A, requires_grad=True, device=device)
        B_wp = wp.array(B, requires_grad=True, device=device)
        C_wp = wp.zeros_like(A_wp, requires_grad=True, device=device)

        with wp.Tape() as tape:
            wp.launch_tiled(
                kernel,
                dim=[int(M / TILE_M), int(N / TILE_N)],
                inputs=[A_wp, B_wp, C_wp],
                block_dim=TILE_DIM,
                device=device,
            )

        tol = 1.0e-6 if dtype == np.float64 else 1.0e-4

        # verify forward pass
        assert_np_equal(C_wp.numpy(), C, tol=tol)

        # verify backward pass
        C_wp.grad = wp.ones_like(C_wp, device=device)
        tape.backward()

        assert_np_equal(A_wp.grad.numpy(), A_grad, tol=tol)
        assert_np_equal(B_wp.grad.numpy(), B_grad, tol=tol)

    dtypes = [np.float32, np.float64]

    for dtype in dtypes:
        run(tile_binary_map_builtin_func, dtype)
        run(tile_binary_map_user_func, dtype)


@wp.func
def binary_func_mixed_types(x: int, y: float) -> float:
    return wp.sin(float(x)) + y


@wp.kernel
def tile_binary_map_mixed_types(
    input_a: wp.array2d(dtype=int), input_b: wp.array2d(dtype=float), output: wp.array2d(dtype=float)
):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    sa = wp.tile_map(binary_func_mixed_types, a, b)

    wp.tile_store(output, sa, offset=(i * TILE_M, j * TILE_N))


def test_tile_binary_map_mixed_types(test, device):
    rng = np.random.default_rng(42)

    M = TILE_M * 7
    N = TILE_N * 5

    A = rng.integers(0, 100, size=(M, N), dtype=np.int32)
    B = rng.random((M, N), dtype=np.float32)
    C = np.sin(A.astype(np.float32)) + B

    A_grad = np.cos(A.astype(np.float32))
    B_grad = np.ones_like(B)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros_like(B_wp, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_binary_map_mixed_types,
            dim=[int(M / TILE_M), int(N / TILE_N)],
            inputs=[A_wp, B_wp, C_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    # verify forward pass
    assert_np_equal(C_wp.numpy(), C, tol=1.0e-6)

    # verify backward pass
    C_wp.grad = wp.ones_like(C_wp, device=device)
    tape.backward()

    # The a gradiens are now stored as ints and can't capture the correct values
    # assert_np_equal(A_wp.grad.numpy(), A_grad, tol=1.0e-6)
    assert_np_equal(B_wp.grad.numpy(), B_grad)


@wp.kernel
def tile_operators(input: wp.array3d(dtype=float), output: wp.array3d(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=(TILE_M, TILE_N))

    # neg
    b = -a

    # right scalar multiply
    c = b * 0.5

    # left scalar multiply
    d = 0.5 * c

    # add tiles
    e = a + d

    wp.tile_store(output[i], e)


def test_tile_operators(test, device):
    batch_count = 56

    M = TILE_M
    N = TILE_N

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, M, N), dtype=np.float32)
    output = input * 0.75

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros_like(input_wp, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_operators, dim=[batch_count], inputs=[input_wp, output_wp], block_dim=TILE_DIM, device=device
        )

    assert_np_equal(output_wp.numpy(), output)

    output_wp.grad.fill_(1.0)

    tape.backward()

    assert_np_equal(input_wp.grad.numpy(), np.ones_like(input) * 0.75)


@wp.kernel
def test_tile_tile_preserve_type_kernel(x: wp.array(dtype=Any), y: wp.array(dtype=Any)):
    a = x[0]
    t = wp.tile(a, preserve_type=True)
    wp.tile_store(y, t)


wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array(dtype=float), "y": wp.array(dtype=float)})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array(dtype=wp.vec3), "y": wp.array(dtype=wp.vec3)})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array(dtype=wp.quat), "y": wp.array(dtype=wp.quat)})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array(dtype=wp.mat33), "y": wp.array(dtype=wp.mat33)})


@wp.kernel
def test_tile_tile_scalar_expansion_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


@wp.kernel
def test_tile_tile_vec_expansion_kernel(x: wp.array(dtype=wp.vec3), y: wp.array2d(dtype=float)):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


@wp.kernel
def test_tile_tile_mat_expansion_kernel(x: wp.array(dtype=wp.mat33), y: wp.array3d(dtype=float)):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


def test_tile_tile(test, device):
    # preserve type
    def test_func_preserve_type(type: Any):
        x = wp.ones(1, dtype=type, requires_grad=True, device=device)
        y = wp.zeros((TILE_DIM), dtype=type, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_tile_tile_preserve_type_kernel,
                dim=[TILE_DIM],
                inputs=[x],
                outputs=[y],
                block_dim=TILE_DIM,
                device=device,
            )

        y.grad = wp.ones_like(y)

        tape.backward()

        assert_np_equal(y.numpy(), wp.full((TILE_DIM), type(1.0), dtype=type, device="cpu").numpy())
        assert_np_equal(x.grad.numpy(), wp.full((1,), type(TILE_DIM), dtype=type, device="cpu").numpy())

    test_func_preserve_type(float)
    test_func_preserve_type(wp.vec3)
    test_func_preserve_type(wp.quat)
    test_func_preserve_type(wp.mat33)

    # scalar expansion
    x = wp.ones(1, dtype=float, requires_grad=True, device=device)
    y = wp.zeros((TILE_DIM), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            test_tile_tile_scalar_expansion_kernel,
            dim=[TILE_DIM],
            inputs=[x],
            outputs=[y],
            block_dim=TILE_DIM,
            device=device,
        )

    y.grad = wp.ones_like(y)

    tape.backward()

    assert_np_equal(y.numpy(), wp.full((TILE_DIM), 1.0, dtype=float, device="cpu").numpy())
    assert_np_equal(x.grad.numpy(), wp.full((1,), wp.float32(TILE_DIM), dtype=float, device="cpu").numpy())

    # vec expansion
    x = wp.ones(1, dtype=wp.vec3, requires_grad=True, device=device)
    y = wp.zeros((3, TILE_DIM), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            test_tile_tile_vec_expansion_kernel,
            dim=[TILE_DIM],
            inputs=[x],
            outputs=[y],
            block_dim=TILE_DIM,
            device=device,
        )

    y.grad = wp.ones_like(y)

    tape.backward()

    assert_np_equal(y.numpy(), wp.full((3, TILE_DIM), 1.0, dtype=float, device="cpu").numpy())
    assert_np_equal(x.grad.numpy(), wp.full((1,), wp.float32(TILE_DIM), dtype=wp.vec3, device="cpu").numpy())

    # mat expansion
    x = wp.ones(1, dtype=wp.mat33, requires_grad=True, device=device)
    y = wp.zeros((3, 3, TILE_DIM), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            test_tile_tile_mat_expansion_kernel,
            dim=[TILE_DIM],
            inputs=[x],
            outputs=[y],
            block_dim=TILE_DIM,
            device=device,
        )

    y.grad = wp.ones_like(y)

    tape.backward()

    assert_np_equal(y.numpy(), wp.full((3, 3, TILE_DIM), 1.0, dtype=float, device="cpu").numpy())
    assert_np_equal(x.grad.numpy(), wp.full((1,), wp.float32(TILE_DIM), dtype=wp.mat33, device="cpu").numpy())


@wp.kernel
def test_tile_untile_preserve_type_kernel(x: wp.array(dtype=Any), y: wp.array(dtype=Any)):
    i = wp.tid()
    a = x[i]
    t = wp.tile(a, preserve_type=True)
    b = wp.untile(t)
    y[i] = b


wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array(dtype=float), "y": wp.array(dtype=float)})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array(dtype=wp.vec3), "y": wp.array(dtype=wp.vec3)})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array(dtype=wp.quat), "y": wp.array(dtype=wp.quat)})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array(dtype=wp.mat33), "y": wp.array(dtype=wp.mat33)})


@wp.kernel
def test_tile_untile_kernel(x: wp.array(dtype=Any), y: wp.array(dtype=Any)):
    i = wp.tid()
    a = x[i]
    t = wp.tile(a)
    b = wp.untile(t)
    y[i] = b


wp.overload(test_tile_untile_kernel, {"x": wp.array(dtype=float), "y": wp.array(dtype=float)})
wp.overload(test_tile_untile_kernel, {"x": wp.array(dtype=wp.vec3), "y": wp.array(dtype=wp.vec3)})
wp.overload(test_tile_untile_kernel, {"x": wp.array(dtype=wp.mat33), "y": wp.array(dtype=wp.mat33)})


def test_tile_untile(test, device):
    def test_func_preserve_type(type: Any):
        x = wp.ones(TILE_DIM, dtype=type, requires_grad=True, device=device)
        y = wp.zeros_like(x)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_tile_untile_preserve_type_kernel,
                dim=TILE_DIM,
                inputs=[x],
                outputs=[y],
                block_dim=TILE_DIM,
                device=device,
            )

        y.grad = wp.ones_like(y)

        tape.backward()

        assert_np_equal(y.numpy(), x.numpy())
        assert_np_equal(x.grad.numpy(), y.grad.numpy())

    test_func_preserve_type(float)
    test_func_preserve_type(wp.vec3)
    test_func_preserve_type(wp.quat)
    test_func_preserve_type(wp.mat33)

    def test_func(type: Any):
        x = wp.ones(TILE_DIM, dtype=type, requires_grad=True, device=device)
        y = wp.zeros_like(x)

        tape = wp.Tape()
        with tape:
            wp.launch(test_tile_untile_kernel, dim=TILE_DIM, inputs=[x], outputs=[y], block_dim=TILE_DIM, device=device)

        y.grad = wp.ones_like(y)

        tape.backward()

        assert_np_equal(y.numpy(), x.numpy())
        assert_np_equal(x.grad.numpy(), y.grad.numpy())

    test_func(float)
    test_func(wp.vec3)
    test_func(wp.mat33)


@wp.func
def tile_sum_func(a: wp.tile(dtype=float, shape=(TILE_M, TILE_N))):
    return wp.tile_sum(a) * 0.5


@wp.kernel
def tile_sum_kernel(input: wp.array3d(dtype=float), output: wp.array(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=(TILE_M, TILE_N))
    s = tile_sum_func(a)

    wp.tile_store(output, s, offset=i)


def test_tile_sum(test, device):
    batch_count = 56

    M = TILE_M
    N = TILE_N

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, M, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_sum_kernel,
            dim=[batch_count],
            inputs=[input_wp, output_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    sum_wp = output_wp.numpy()

    for i in range(batch_count):
        sum_np = np.sum(input[i]) * 0.5
        test.assertAlmostEqual(sum_wp[i], sum_np, places=5)

    output_wp.grad.fill_(1.0)

    tape.backward()

    assert_np_equal(input_wp.grad.numpy(), np.ones_like(input) * 0.5)


def test_tile_sum_launch(test, device):
    batch_count = 56

    M = TILE_M
    N = TILE_N

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, M, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, requires_grad=True, device=device)

    cmd = wp.launch_tiled(
        tile_sum_kernel,
        dim=[batch_count],
        inputs=[input_wp, output_wp],
        block_dim=TILE_DIM,
        device=device,
        record_cmd=True,
    )
    cmd.launch()

    sum_wp = output_wp.numpy()

    for i in range(batch_count):
        sum_np = np.sum(input[i]) * 0.5
        test.assertAlmostEqual(sum_wp[i], sum_np, places=5)

    output_wp.grad.fill_(1.0)

    wp.launch_tiled(
        tile_sum_kernel,
        dim=[batch_count],
        inputs=[input_wp, output_wp],
        adj_inputs=[input_wp.grad, output_wp.grad],
        block_dim=TILE_DIM,
        device=device,
        adjoint=True,
    )

    assert_np_equal(input_wp.grad.numpy(), np.ones_like(input) * 0.5)


@wp.kernel(module="unique")
def test_tile_extract_kernel(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):
    i, j, x, y = wp.tid()

    tile = wp.tile_load(a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    # compute sum of array sub tile
    wp.atomic_add(b, i, j, wp.tile_extract(tile, x, y))


def test_tile_extract(test, device):
    block_dim = 16

    input = np.arange(TILE_M * TILE_N * 4).reshape((TILE_M * 2, TILE_N * 2))

    a = wp.array(input, dtype=float, requires_grad=True, device=device)
    b = wp.zeros((2, 2), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(
            test_tile_extract_kernel, dim=[2, 2, TILE_M, TILE_N], inputs=[a, b], block_dim=block_dim, device=device
        )

    # compute sum of each sub-block
    sums = input.reshape(2, input.shape[0] // 2, 2, input.shape[1] // 2).sum(axis=(1, 3))

    assert_np_equal(b.numpy(), sums)

    b.grad.fill_(1.0)

    tape.backward()

    expected_grad = np.ones_like(input)
    assert_np_equal(a.grad.numpy(), expected_grad)


@wp.kernel(module="unique")
def test_tile_extract_repeated_kernel(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):
    i, j, x, y = wp.tid()

    tile = wp.tile_load(a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    # each thread extracts the first element of the sub-tile
    # and accumulates the value onto the output
    wp.atomic_add(b, i, j, wp.tile_extract(tile, 0, 0))


def test_tile_extract_repeated(test, device):
    block_dim = 16

    input = np.arange(TILE_M * TILE_N * 4).reshape((TILE_M * 2, TILE_N * 2))

    a = wp.array(input, dtype=float, requires_grad=True, device=device)
    b = wp.zeros((2, 2), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(
            test_tile_extract_repeated_kernel,
            dim=[2, 2, TILE_M, TILE_N],
            inputs=[a, b],
            block_dim=block_dim,
            device=device,
        )

    # each thread adds the first element to the output
    scale = TILE_M * TILE_N
    sums = np.array([[input[0, 0], input[0, TILE_N]], [input[TILE_M, 0], input[TILE_M, TILE_N]]]) * scale

    assert_np_equal(b.numpy(), sums)

    b.grad.fill_(1.0)

    tape.backward()

    expected_grad = np.zeros_like(input)
    expected_grad[0, 0] = scale
    expected_grad[0, TILE_N] = scale
    expected_grad[TILE_M, 0] = scale
    expected_grad[TILE_M, TILE_N] = scale

    assert_np_equal(a.grad.numpy(), expected_grad)


@wp.kernel
def test_tile_assign_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i, j = wp.tid()

    a = wp.tile_zeros(shape=(TILE_M,), dtype=float)

    a[j] = x[j]

    wp.tile_atomic_add(y, a, offset=(0,))


def test_tile_assign(test, device):
    x = wp.full(TILE_M, 2.0, dtype=float, device=device, requires_grad=True)
    y = wp.zeros(TILE_M, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(test_tile_assign_kernel, dim=[1, TILE_M], inputs=[x], outputs=[y], block_dim=TILE_DIM, device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.full(TILE_M, 2.0, dtype=np.float32))
    assert_np_equal(x.grad.numpy(), np.full(TILE_M, 1.0, dtype=np.float32))


@wp.kernel
def test_tile_transpose_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
    x = wp.tile_load(input, shape=(TILE_M, TILE_N))
    y = wp.tile_transpose(x)

    wp.tile_store(output, y)


def test_tile_transpose(test, device):
    rng = np.random.default_rng(42)
    input = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), device=device)
    output = wp.zeros_like(input.transpose(), device=device)

    wp.launch_tiled(test_tile_transpose_kernel, dim=[1], inputs=[input, output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy(), input.numpy().T)


@wp.kernel
def test_tile_broadcast_add_1d_kernel(
    input_a: wp.array(dtype=float), input_b: wp.array(dtype=float), output: wp.array(dtype=float)
):
    a = wp.tile_load(input_a, shape=(10,))
    b = wp.tile_load(input_b, shape=(1,))

    c = wp.tile_broadcast(b, shape=(10,))
    d = a + c

    wp.tile_store(output, d)


def test_tile_broadcast_add_1d(test, device):
    N = 10

    # implicit 1-dim ([1], 1)
    a = wp.array(np.arange(0, N, dtype=np.float32), device=device)
    b = wp.array(np.ones(1, dtype=np.float32), device=device)
    out = wp.zeros((N,), dtype=float, device=device)

    wp.launch_tiled(test_tile_broadcast_add_1d_kernel, dim=[1], inputs=[a, b, out], block_dim=TILE_DIM, device=device)

    assert_np_equal(out.numpy(), a.numpy() + b.numpy())


@wp.kernel
def test_tile_broadcast_add_2d_kernel(
    input_a: wp.array2d(dtype=float), input_b: wp.array(dtype=float), output: wp.array2d(dtype=float)
):
    # implicit 1-dim ([1], 10)
    a = wp.tile_load(input_a, shape=(10, 10))
    b = wp.tile_load(input_b, shape=10)

    c = wp.tile_broadcast(b, shape=(10, 10))
    d = a + c

    wp.tile_store(output, d)


def test_tile_broadcast_add_2d(test, device):
    M = 10
    N = 10

    a = wp.array(np.ones((M, N), dtype=np.float32), device=device)
    b = wp.array(np.arange(0, N, dtype=np.float32), device=device)
    out = wp.zeros((M, N), dtype=float, device=device)

    wp.launch_tiled(test_tile_broadcast_add_2d_kernel, dim=[1], inputs=[a, b, out], block_dim=TILE_DIM, device=device)

    assert_np_equal(out.numpy(), a.numpy() + b.numpy())


@wp.kernel
def test_tile_broadcast_add_3d_kernel(
    input_a: wp.array3d(dtype=float), input_b: wp.array3d(dtype=float), output: wp.array3d(dtype=float)
):
    a = wp.tile_load(input_a, shape=(4, 10, 12))
    b = wp.tile_load(input_b, shape=(4, 10, 1))

    c = wp.tile_broadcast(b, shape=(4, 10, 12))
    d = a + c

    wp.tile_store(output, d)


def test_tile_broadcast_add_3d(test, device):
    M = 4
    N = 10
    O = 12

    # explicit 1-dim (M, N, 1) to (M, N, O)
    a = wp.array(np.ones((M, N, O), dtype=np.float32), device=device)
    b = wp.array(np.arange(0, M * N, dtype=np.float32).reshape((M, N, 1)), device=device)
    out = wp.zeros((M, N, O), dtype=float, device=device)

    wp.launch_tiled(test_tile_broadcast_add_3d_kernel, dim=[1], inputs=[a, b, out], block_dim=TILE_DIM, device=device)
    assert_np_equal(out.numpy(), a.numpy() + b.numpy())


@wp.kernel
def test_tile_broadcast_add_4d_kernel(
    input_a: wp.array4d(dtype=float), input_b: wp.array4d(dtype=float), output: wp.array4d(dtype=float)
):
    a = wp.tile_load(input_a, shape=(4, 10, 5, 6))
    b = wp.tile_load(input_b, shape=(4, 1, 5, 1))
    c = wp.tile_broadcast(b, shape=(4, 10, 5, 6))
    d = a + c

    wp.tile_store(output, d)


def test_tile_broadcast_add_4d(test, device):
    M = 4
    N = 10
    O = 5
    P = 6

    # explicit 1-dims (M, 1, O, 1) to (M, N, O, P)
    a = wp.array(np.ones((M, N, O, P), dtype=np.float32), device=device)
    b = wp.array(np.arange(0, M * O, dtype=np.float32).reshape((M, 1, O, 1)), device=device)
    out = wp.zeros((M, N, O, P), dtype=float, device=device)

    wp.launch_tiled(test_tile_broadcast_add_4d_kernel, dim=[1], inputs=[a, b, out], block_dim=TILE_DIM, device=device)

    assert_np_equal(out.numpy(), a.numpy() + b.numpy())


@wp.kernel
def test_tile_broadcast_grad_kernel(a: wp.array(dtype=float), b: wp.array2d(dtype=float)):
    x = wp.tile_load(a, shape=5)
    y = wp.tile_broadcast(x, shape=(5, 5))

    w = wp.tile_ones(dtype=float, shape=(5, 5))
    z = w + y

    wp.tile_store(b, z)


def test_tile_broadcast_grad(test, device):
    a = wp.array(np.arange(0, 5, dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.ones((5, 5), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_broadcast_grad_kernel, dim=[1], inputs=[a, b], block_dim=TILE_DIM, device=device)

    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(b.numpy(), a.numpy() + np.ones((5, 5)))
    assert_np_equal(a.grad.numpy(), np.ones(5) * 5.0)


@wp.kernel
def test_tile_squeeze_kernel(x: wp.array3d(dtype=float), y: wp.array(dtype=float)):
    a = wp.tile_load(x, shape=(1, TILE_M, 1), offset=(0, 0, 0))
    b = wp.tile_squeeze(a, axis=(2,))
    c = wp.tile_squeeze(b)

    wp.tile_store(y, c, offset=(0,))


def test_tile_squeeze(test, device):
    x = wp.ones((1, TILE_M, 1), dtype=float, device=device, requires_grad=True)
    y = wp.zeros((TILE_M,), dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch_tiled(test_tile_squeeze_kernel, dim=1, inputs=[x], outputs=[y], block_dim=TILE_DIM, device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.ones((TILE_M,), dtype=np.float32))
    assert_np_equal(x.grad.numpy(), np.ones((1, TILE_M, 1), dtype=np.float32))


@wp.kernel
def test_tile_reshape_kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(0, 0))
    b = wp.tile_reshape(a, shape=(wp.static(TILE_M * TILE_N), 1))
    c = wp.tile_reshape(b, shape=(-1, 1))

    wp.tile_store(y, c, offset=(0, 0))


def test_tile_reshape(test, device):
    x = wp.ones((TILE_M, TILE_N), dtype=float, device=device, requires_grad=True)
    y = wp.zeros((TILE_M * TILE_N, 1), dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch_tiled(test_tile_reshape_kernel, dim=1, inputs=[x], outputs=[y], block_dim=TILE_DIM, device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.ones((TILE_M * TILE_N, 1), dtype=np.float32))
    assert_np_equal(x.grad.numpy(), np.ones((TILE_M, TILE_N), dtype=np.float32))


@wp.kernel
def test_tile_astype_kernel(x: wp.array2d(dtype=Any), y: wp.array2d(dtype=wp.float32)):
    a = wp.tile_load(x, shape=(TILE_M, TILE_N))
    b = wp.tile_astype(a, dtype=wp.float32)
    wp.tile_store(y, b)


def test_tile_astype(test, device):
    x_np = np.arange(TILE_M * TILE_N, dtype=np.int32).reshape((TILE_M, TILE_N))
    x = wp.array(x_np, dtype=wp.int32, device=device)
    y = wp.zeros((TILE_M, TILE_N), dtype=wp.float32, device=device)

    wp.launch_tiled(test_tile_astype_kernel, dim=1, inputs=[x], outputs=[y], block_dim=TILE_DIM, device=device)

    assert_np_equal(y.numpy(), x_np.astype(np.float32))

    x_np = np.arange(TILE_M * TILE_N, dtype=np.float64).reshape((TILE_M, TILE_N))
    x = wp.array(x_np, dtype=wp.float64, requires_grad=True, device=device)
    y = wp.zeros((TILE_M, TILE_N), dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch_tiled(test_tile_astype_kernel, dim=1, inputs=[x], outputs=[y], block_dim=TILE_DIM, device=device)

    y.grad = wp.ones_like(y)

    tape.backward()

    assert_np_equal(y.numpy(), x_np.astype(np.float32))
    assert_np_equal(x.grad.numpy(), np.ones_like(x_np))


@wp.func
def test_tile_func_return_func(tile: Any):
    return tile


@wp.kernel
def test_tile_func_return_kernel(x: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.float32)):
    a = wp.tile_load(x, shape=(TILE_M, 1))
    b = wp.tile_broadcast(a, shape=(TILE_M, TILE_K))
    c = test_tile_func_return_func(b)
    wp.tile_store(y, c)


def test_tile_func_return(test, device):
    x = wp.ones(shape=(TILE_M, 1), dtype=wp.float32, requires_grad=True, device=device)
    y = wp.zeros(shape=(TILE_M, TILE_K), dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch_tiled(
            test_tile_func_return_kernel, dim=[1, 1], inputs=[x], outputs=[y], block_dim=TILE_DIM, device=device
        )

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.ones((TILE_M, TILE_K), dtype=np.float32))
    assert_np_equal(x.grad.numpy(), np.ones((TILE_M, 1), dtype=np.float32) * TILE_K)


@wp.kernel
def tile_len_kernel(
    a: wp.array(dtype=float, ndim=2),
    out: wp.array(dtype=int),
):
    x = wp.tile_load(a, shape=(TILE_M, TILE_N))

    length = wp.static(len(x))
    wp.expect_eq(wp.static(len(x)), TILE_M)
    out[0] = wp.static(len(x))


def test_tile_len(test, device):
    a = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)
    out = wp.empty(1, dtype=int, device=device)
    wp.launch_tiled(tile_len_kernel, dim=(1,), inputs=(a,), outputs=(out,), block_dim=TILE_DIM, device=device)

    test.assertEqual(out.numpy()[0], TILE_M)


@wp.kernel
def test_tile_print_kernel():
    # shared tile
    a = wp.tile_ones(shape=(4, 3), dtype=float, storage="shared")
    # register tile
    b = wp.tile_ones(shape=(4, 3), dtype=float)

    print(a)
    print(b)


def test_tile_print(test, device):
    wp.launch_tiled(test_tile_print_kernel, dim=1, inputs=[], block_dim=64, device=device)
    wp.synchronize()


@wp.kernel
def test_tile_add_inplace_kernel(
    input_a: wp.array2d(dtype=float),
    input_b: wp.array2d(dtype=float),
    output_reg: wp.array2d(dtype=float),
    output_shared: wp.array2d(dtype=float),
):
    i, j = wp.tid()

    a_reg = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="register")
    b_reg = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="register")
    a_shared = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="shared")
    b_shared = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="shared")

    a_reg += b_reg
    a_reg += b_shared
    a_shared += b_reg
    a_shared += b_shared

    wp.tile_store(output_reg, a_reg, offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(output_shared, a_shared, offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def test_tile_sub_inplace_kernel(
    input_a: wp.array2d(dtype=float),
    input_b: wp.array2d(dtype=float),
    output_reg: wp.array2d(dtype=float),
    output_shared: wp.array2d(dtype=float),
):
    i, j = wp.tid()

    a_reg = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="register")
    b_reg = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="register")
    a_shared = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="shared")
    b_shared = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N), storage="shared")

    a_reg -= b_reg
    a_reg -= b_shared
    a_shared -= b_reg
    a_shared -= b_shared

    wp.tile_store(output_reg, a_reg, offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(output_shared, a_shared, offset=(i * TILE_M, j * TILE_N))


def test_tile_inplace(test, device):
    M = TILE_M * 2
    N = TILE_N * 2

    a = wp.zeros((M, N), requires_grad=True, device=device)
    b = wp.ones_like(a, requires_grad=True, device=device)
    c = wp.zeros_like(a, requires_grad=True, device=device)
    d = wp.zeros_like(a, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            test_tile_add_inplace_kernel,
            dim=[int(M / TILE_M), int(N / TILE_N)],
            inputs=[a, b, c, d],
            block_dim=TILE_DIM,
            device=device,
        )

    assert_np_equal(a.numpy(), np.zeros((M, N)))
    assert_np_equal(b.numpy(), np.ones((M, N)))
    assert_np_equal(c.numpy(), 2.0 * np.ones((M, N)))
    assert_np_equal(d.numpy(), 2.0 * np.ones((M, N)))

    c.grad = wp.ones_like(c, device=device)
    d.grad = wp.ones_like(d, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), 2.0 * np.ones((M, N)))
    assert_np_equal(b.grad.numpy(), 4.0 * np.ones((M, N)))

    tape.zero()

    a.zero_()
    b.fill_(1.0)
    c.zero_()
    d.zero_()

    with wp.Tape() as tape:
        wp.launch_tiled(
            test_tile_sub_inplace_kernel,
            dim=[int(M / TILE_M), int(N / TILE_N)],
            inputs=[a, b, c, d],
            block_dim=TILE_DIM,
            device=device,
        )

    assert_np_equal(a.numpy(), np.zeros((M, N)))
    assert_np_equal(b.numpy(), np.ones((M, N)))
    assert_np_equal(c.numpy(), -2.0 * np.ones((M, N)))
    assert_np_equal(d.numpy(), -2.0 * np.ones((M, N)))

    c.grad = wp.ones_like(c, device=device)
    d.grad = wp.ones_like(d, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), 2.0 * np.ones((M, N)))
    assert_np_equal(b.grad.numpy(), -4.0 * np.ones((M, N)))


devices = get_test_devices()


class TestTile(unittest.TestCase):
    pass


add_function_test(TestTile, "test_tile_copy_1d", test_tile_copy_1d, devices=devices)
add_function_test(TestTile, "test_tile_copy_2d", test_tile_copy_2d, devices=devices)
add_function_test(TestTile, "test_tile_unary_map", test_tile_unary_map, devices=devices)
add_function_test(TestTile, "test_tile_unary_map_mixed_types", test_tile_unary_map_mixed_types, devices=devices)
add_function_test(TestTile, "test_tile_binary_map", test_tile_binary_map, devices=devices)
add_function_test(TestTile, "test_tile_binary_map_mixed_types", test_tile_binary_map_mixed_types, devices=devices)
add_function_test(TestTile, "test_tile_transpose", test_tile_transpose, devices=devices)
add_function_test(TestTile, "test_tile_operators", test_tile_operators, devices=devices)
add_function_test(TestTile, "test_tile_tile", test_tile_tile, devices=get_cuda_test_devices())
add_function_test(TestTile, "test_tile_untile", test_tile_untile, devices=devices)
add_function_test(TestTile, "test_tile_sum", test_tile_sum, devices=devices, check_output=False)
add_function_test(TestTile, "test_tile_sum_launch", test_tile_sum_launch, devices=devices)
add_function_test(TestTile, "test_tile_extract", test_tile_extract, devices=devices)
add_function_test(TestTile, "test_tile_extract_repeated", test_tile_extract_repeated, devices=devices)
add_function_test(TestTile, "test_tile_assign", test_tile_assign, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_add_1d", test_tile_broadcast_add_1d, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_add_2d", test_tile_broadcast_add_2d, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_add_3d", test_tile_broadcast_add_3d, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_add_4d", test_tile_broadcast_add_4d, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_grad", test_tile_broadcast_grad, devices=devices)
add_function_test(TestTile, "test_tile_squeeze", test_tile_squeeze, devices=devices)
add_function_test(TestTile, "test_tile_reshape", test_tile_reshape, devices=devices)
add_function_test(TestTile, "test_tile_len", test_tile_len, devices=devices)
# add_function_test(TestTile, "test_tile_print", test_tile_print, devices=devices, check_output=False)
# add_function_test(TestTile, "test_tile_inplace", test_tile_inplace, devices=devices)
# add_function_test(TestTile, "test_tile_astype", test_tile_astype, devices=devices)
# add_function_test(TestTile, "test_tile_func_return", test_tile_func_return, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
