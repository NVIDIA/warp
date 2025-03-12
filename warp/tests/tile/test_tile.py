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
def unary_func(x: float):
    return wp.sin(x)


@wp.kernel
def tile_unary_map(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    sa = wp.tile_map(wp.sin, a)

    wp.tile_store(output, sa, offset=(i * TILE_M, j * TILE_N))


def test_tile_unary_map(test, device):
    rng = np.random.default_rng(42)

    M = TILE_M * 7
    N = TILE_N * 5

    A = rng.random((M, N), dtype=np.float32)
    B = np.sin(A)

    A_grad = np.cos(A)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.zeros_like(A_wp, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_unary_map,
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

    assert_np_equal(A_wp.grad.numpy(), A_grad, tol=1.0e-6)


@wp.func
def binary_func(x: float, y: float):
    return wp.sin(x) + y


@wp.kernel
def tile_binary_map(
    input_a: wp.array2d(dtype=float), input_b: wp.array2d(dtype=float), output: wp.array2d(dtype=float)
):
    # tile index
    i, j = wp.tid()

    a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))

    sa = wp.tile_map(binary_func, a, b)

    wp.tile_store(output, sa, offset=(i * TILE_M, j * TILE_N))


def test_tile_binary_map(test, device):
    rng = np.random.default_rng(42)

    M = TILE_M * 7
    N = TILE_N * 5

    A = rng.random((M, N), dtype=np.float32)
    B = rng.random((M, N), dtype=np.float32)
    C = np.sin(A) + B

    A_grad = np.cos(A)
    B_grad = np.ones_like(B)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros_like(A_wp, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_binary_map,
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

    assert_np_equal(A_wp.grad.numpy(), A_grad, tol=1.0e-6)
    assert_np_equal(B_wp.grad.numpy(), B_grad)


def test_tile_grouped_gemm(test, device):
    @wp.kernel
    def tile_grouped_gemm(A: wp.array3d(dtype=float), B: wp.array3d(dtype=float), C: wp.array3d(dtype=float)):
        # output tile index
        i = wp.tid()

        a = wp.tile_load(A[i], shape=(TILE_M, TILE_K))
        b = wp.tile_load(B[i], shape=(TILE_K, TILE_N))

        sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        wp.tile_matmul(a, b, sum)

        wp.tile_store(C[i], sum)

    batch_count = 56

    M = TILE_M
    N = TILE_N
    K = TILE_K

    rng = np.random.default_rng(42)
    A = rng.random((batch_count, M, K), dtype=np.float32)
    B = rng.random((batch_count, K, N), dtype=np.float32)
    C = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((batch_count, TILE_M, TILE_N), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_grouped_gemm, dim=[batch_count], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

    # TODO: 32 mismatched elements
    assert_np_equal(C_wp.numpy(), C, 1e-6)


def test_tile_gemm(dtype):
    def test(test, device):
        @wp.kernel
        def tile_gemm(A: wp.array2d(dtype=dtype), B: wp.array2d(dtype=dtype), C: wp.array2d(dtype=dtype)):
            # output tile index
            i, j = wp.tid()

            sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=dtype)

            M = A.shape[0]
            N = B.shape[1]
            K = A.shape[1]

            count = int(K / TILE_K)

            for k in range(0, count):
                a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
                b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

                # sum += a*b
                wp.tile_matmul(a, b, sum)

            wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))

        M = TILE_M * 7
        K = TILE_K * 6
        N = TILE_N * 5

        rng = np.random.default_rng(42)
        A = rng.random((M, K), dtype=float).astype(wp.dtype_to_numpy(dtype))
        B = rng.random((K, N), dtype=float).astype(wp.dtype_to_numpy(dtype))
        C = np.zeros((M, N), dtype=float).astype(wp.dtype_to_numpy(dtype))

        A_wp = wp.array(A, requires_grad=True, device=device)
        B_wp = wp.array(B, requires_grad=True, device=device)
        C_wp = wp.array(C, requires_grad=True, device=device)

        with wp.Tape() as tape:
            wp.launch_tiled(
                tile_gemm,
                dim=(int(M / TILE_M), int(N / TILE_N)),
                inputs=[A_wp, B_wp, C_wp],
                block_dim=TILE_DIM,
                device=device,
            )

        assert_np_equal(C_wp.numpy(), A @ B, tol=1.0e-1)

        adj_C = np.ones_like(C)

        tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

        assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1.0e-1)
        assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, 1.0e-1)

    return test


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
def tile_sum_kernel(input: wp.array3d(dtype=float), output: wp.array(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=(TILE_M, TILE_N))
    s = wp.tile_sum(a) * 0.5

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


@wp.kernel
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


@wp.kernel
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
def test_tile_transpose_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
    x = wp.tile_load(input, shape=(TILE_M, TILE_N))
    y = wp.tile_transpose(x)

    wp.tile_store(output, y)


def test_tile_transpose(test, device):
    rng = np.random.default_rng(42)
    input = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), device=device)
    output = wp.zeros_like(input.transpose(), device=device)

    wp.launch_tiled(test_tile_transpose_kernel, dim=[1], inputs=[input, output], block_dim=32, device=device)

    assert_np_equal(output.numpy(), input.numpy().T)


def test_tile_transpose_matmul(test, device):
    @wp.kernel
    def test_tile_transpose_matmul_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
        x = wp.tile_load(input, shape=(TILE_M, TILE_N))
        y = wp.tile_transpose(x)

        z = wp.tile_zeros(dtype=float, shape=(TILE_N, TILE_N))
        wp.tile_matmul(y, x, z)

        wp.tile_store(output, z)

    rng = np.random.default_rng(42)
    input = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), device=device)
    output = wp.zeros((TILE_N, TILE_N), dtype=float, device=device)

    wp.launch_tiled(test_tile_transpose_matmul_kernel, dim=[1], inputs=[input, output], block_dim=32, device=device)

    assert_np_equal(output.numpy(), input.numpy().T @ input.numpy())


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

    wp.launch_tiled(test_tile_broadcast_add_1d_kernel, dim=[1], inputs=[a, b, out], block_dim=32, device=device)

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

    wp.launch_tiled(test_tile_broadcast_add_2d_kernel, dim=[1], inputs=[a, b, out], block_dim=32, device=device)

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

    wp.launch_tiled(test_tile_broadcast_add_3d_kernel, dim=[1], inputs=[a, b, out], block_dim=32, device=device)
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

    wp.launch_tiled(test_tile_broadcast_add_4d_kernel, dim=[1], inputs=[a, b, out], block_dim=32, device=device)

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
        wp.launch_tiled(test_tile_broadcast_grad_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(b.numpy(), a.numpy() + np.ones((5, 5)))
    assert_np_equal(a.grad.numpy(), np.ones(5) * 5.0)


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
    wp.launch_tiled(
        tile_len_kernel,
        dim=(1,),
        inputs=(a,),
        outputs=(out,),
        block_dim=32,
        device=device,
    )

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


devices = get_test_devices()


class TestTile(unittest.TestCase):
    pass


add_function_test(TestTile, "test_tile_copy_1d", test_tile_copy_1d, devices=devices)
add_function_test(TestTile, "test_tile_copy_2d", test_tile_copy_2d, devices=devices)
add_function_test(TestTile, "test_tile_unary_map", test_tile_unary_map, devices=devices)
add_function_test(TestTile, "test_tile_binary_map", test_tile_binary_map, devices=devices)
add_function_test(TestTile, "test_tile_grouped_gemm", test_tile_grouped_gemm, devices=devices)
add_function_test(TestTile, "test_tile_gemm_fp16", test_tile_gemm(wp.float16), devices=devices)
add_function_test(TestTile, "test_tile_gemm_fp32", test_tile_gemm(wp.float32), devices=devices)
add_function_test(TestTile, "test_tile_gemm_fp64", test_tile_gemm(wp.float64), devices=devices)
add_function_test(TestTile, "test_tile_transpose", test_tile_transpose, devices=devices)
add_function_test(TestTile, "test_tile_transpose_matmul", test_tile_transpose_matmul, devices=devices)
add_function_test(TestTile, "test_tile_operators", test_tile_operators, devices=devices)
add_function_test(TestTile, "test_tile_sum", test_tile_sum, devices=devices, check_output=False)
add_function_test(TestTile, "test_tile_sum_launch", test_tile_sum_launch, devices=devices)
add_function_test(TestTile, "test_tile_extract", test_tile_extract, devices=devices)
add_function_test(TestTile, "test_tile_extract_repeated", test_tile_extract_repeated, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_add_1d", test_tile_broadcast_add_1d, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_add_2d", test_tile_broadcast_add_2d, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_add_3d", test_tile_broadcast_add_3d, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_add_4d", test_tile_broadcast_add_4d, devices=devices)
add_function_test(TestTile, "test_tile_broadcast_grad", test_tile_broadcast_grad, devices=devices)
add_function_test(TestTile, "test_tile_len", test_tile_len, devices=devices)
add_function_test(TestTile, "test_tile_print", test_tile_print, devices=devices, check_output=False)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
