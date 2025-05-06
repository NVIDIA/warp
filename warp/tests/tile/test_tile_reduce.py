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
def tile_sum_kernel(input: wp.array2d(dtype=float), output: wp.array(dtype=float)):
    # output tile index
    i = wp.tid()

    n = input.shape[1]
    count = int(n / TILE_DIM)

    s = wp.tile_zeros(shape=1, dtype=float)

    for j in range(count):
        a = wp.tile_load(input[i], shape=TILE_DIM, offset=j * TILE_DIM)
        s += wp.tile_sum(a) * 0.5

    wp.tile_store(output, s, offset=i)


def test_tile_reduce_sum(test, device):
    batch_count = 56

    N = TILE_DIM * 3

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_sum_kernel, dim=[batch_count], inputs=[input_wp, output_wp], block_dim=TILE_DIM, device=device
        )

    sum_wp = output_wp.numpy()
    for i in range(batch_count):
        sum_np = np.sum(input[i]) * 0.5
        test.assertAlmostEqual(sum_wp[i], sum_np, places=4)

    output_wp.grad.fill_(1.0)

    tape.backward()

    assert_np_equal(input_wp.grad.numpy(), np.ones_like(input) * 0.5, tol=1.0e-4)


@wp.kernel
def tile_min_kernel(input: wp.array2d(dtype=float), output: wp.array(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=TILE_DIM)
    m = wp.tile_min(a)

    wp.tile_store(output, m, offset=i)


def test_tile_reduce_min(test, device):
    batch_count = 56

    N = TILE_DIM

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_min_kernel, dim=[batch_count], inputs=[input_wp, output_wp], block_dim=TILE_DIM, device=device
        )

    min_wp = output_wp.numpy()
    for i in range(batch_count):
        min_np = np.min(input[i])
        test.assertAlmostEqual(min_wp[i], min_np, places=4)


@wp.kernel
def tile_argmin_kernel(input: wp.array2d(dtype=float), output: wp.array(dtype=int)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=TILE_DIM)
    m = wp.tile_argmin(a)

    wp.tile_store(output, m, offset=i)


def test_tile_reduce_argmin(test, device):
    batch_count = 56

    N = TILE_DIM

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, dtype=wp.int32, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_argmin_kernel, dim=[batch_count], inputs=[input_wp, output_wp], block_dim=TILE_DIM, device=device
        )

    argmin_wp = output_wp.numpy()
    for i in range(batch_count):
        argmin_np = np.argmin(input[i])
        test.assertAlmostEqual(argmin_wp[i], argmin_np, places=4)


@wp.kernel
def tile_max_kernel(input: wp.array2d(dtype=float), output: wp.array(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=TILE_DIM)
    m = wp.tile_max(a)

    wp.tile_store(output, m, offset=i)


def test_tile_reduce_max(test, device):
    batch_count = 56

    N = TILE_DIM

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_max_kernel, dim=[batch_count], inputs=[input_wp, output_wp], block_dim=TILE_DIM, device=device
        )

    max_wp = output_wp.numpy()
    for i in range(batch_count):
        max_np = np.max(input[i])
        test.assertAlmostEqual(max_wp[i], max_np, places=4)


@wp.kernel
def tile_argmax_kernel(input: wp.array2d(dtype=float), output: wp.array(dtype=int)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=TILE_DIM)
    m = wp.tile_argmax(a)

    wp.tile_store(output, m, offset=i)


def test_tile_reduce_argmax(test, device):
    batch_count = 56

    N = TILE_DIM

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, dtype=wp.int32, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_argmax_kernel, dim=[batch_count], inputs=[input_wp, output_wp], block_dim=TILE_DIM, device=device
        )

    argmax_wp = output_wp.numpy()
    for i in range(batch_count):
        argmax_np = np.argmax(input[i])
        test.assertAlmostEqual(argmax_wp[i], argmax_np, places=4)


@wp.kernel
def tile_reduce_custom_kernel(input: wp.array2d(dtype=float), output: wp.array(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=TILE_DIM)
    m = wp.tile_reduce(wp.mul, a)

    wp.tile_store(output, m, offset=i)


def test_tile_reduce_custom(test, device):
    batch_count = 56

    N = TILE_DIM

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_reduce_custom_kernel,
            dim=[batch_count],
            inputs=[input_wp, output_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    prod_wp = output_wp.numpy()
    for i in range(batch_count):
        prod_np = np.prod(input[i])
        test.assertAlmostEqual(prod_wp[i], prod_np, places=4)


@wp.struct
class KeyValue:
    key: wp.int32
    value: wp.float32


@wp.func
def kv_max(a: KeyValue, b: KeyValue) -> KeyValue:
    return wp.where(a.value < b.value, b, a)


@wp.kernel
def initialize_key_value(values: wp.array2d(dtype=wp.float32), keyvalues: wp.array2d(dtype=KeyValue)):
    batch, idx = wp.tid()
    keyvalues[batch, idx] = KeyValue(idx, values[batch, idx])


@wp.kernel(enable_backward=False)
def tile_reduce_custom_struct_kernel(values: wp.array2d(dtype=KeyValue), res: wp.array(dtype=KeyValue)):
    # output tile index
    i = wp.tid()

    t = wp.tile_load(values, shape=(1, TILE_DIM), offset=(i, 0))

    max_el = wp.tile_reduce(kv_max, t)
    wp.tile_store(res, max_el, offset=i)


def test_tile_reduce_custom_struct(test, device):
    batch_count = 56

    N = TILE_DIM

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, N), dtype=np.float32)

    input_wp = wp.array(input, dtype=wp.float32, device=device)
    keyvalues_wp = wp.empty(input_wp.shape, dtype=KeyValue, device=device)

    wp.launch(initialize_key_value, dim=[batch_count, N], inputs=[input_wp], outputs=[keyvalues_wp], device=device)

    output_wp = wp.empty(batch_count, dtype=KeyValue, device=device)

    wp.launch_tiled(
        tile_reduce_custom_struct_kernel,
        dim=[batch_count],
        inputs=[keyvalues_wp],
        outputs=[output_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    prod_wp = np.array([k for k, v in output_wp.numpy()])
    expected = np.argmax(input, axis=1)

    assert_np_equal(prod_wp, expected)


@wp.kernel
def tile_grouped_sum_kernel(input: wp.array3d(dtype=float), output: wp.array(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(input[i], shape=(TILE_M, TILE_N))
    s = wp.tile_sum(a) * 0.5

    wp.tile_store(output, s, offset=i)


def test_tile_reduce_grouped_sum(test, device):
    batch_count = 56

    M = TILE_M
    N = TILE_N

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, M, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True, device=device)
    output_wp = wp.zeros(batch_count, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_sum_kernel, dim=[batch_count], inputs=[input_wp, output_wp], block_dim=TILE_DIM, device=device
        )

    sum_wp = output_wp.numpy()
    for i in range(batch_count):
        sum_np = np.sum(input[i]) * 0.5
        test.assertAlmostEqual(sum_wp[i], sum_np, places=4)

    output_wp.grad.fill_(1.0)

    tape.backward()

    assert_np_equal(input_wp.grad.numpy(), np.ones_like(input) * 0.5, tol=1.0e-4)


@wp.kernel
def tile_reduce_simt_kernel(output: wp.array(dtype=int)):
    # thread index
    i = wp.tid()

    t = wp.tile(i)  # convert to block wide tile
    s = wp.tile_sum(t)  # sum over block

    # update global sum
    wp.tile_atomic_add(output, s)


def test_tile_reduce_simt(test, device):
    # use an unaligned grid dimension
    N = TILE_DIM * 4 + 5

    output = wp.zeros(shape=1, dtype=int, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(tile_reduce_simt_kernel, dim=N, inputs=[output], block_dim=TILE_DIM, device=device)

    test.assertEqual(output.numpy()[0], np.sum(np.arange(N)))


@wp.kernel
def tile_untile_kernel(output: wp.array(dtype=int)):
    # thread index
    i = wp.tid()

    # convert to block wide tile
    t = wp.tile(i) * 2
    s = wp.untile(t)

    output[i] = s


def test_tile_untile(test, device):
    # use an unaligned grid dimension
    N = TILE_DIM * 4 + 5

    output = wp.zeros(shape=N, dtype=int, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(tile_untile_kernel, dim=N, inputs=[output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy(), np.arange(N) * 2)


@wp.kernel
def tile_untile_scalar_kernel(output: wp.array(dtype=int)):
    # thread index
    i = wp.tid()

    # convert to block wide tile
    t = wp.tile(i) * 2
    s = wp.untile(t)

    output[i] = s


def test_tile_untile_scalar(test, device):
    # use an unaligned grid dimension
    N = TILE_DIM * 4 + 5

    output = wp.zeros(shape=N, dtype=int, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(tile_untile_kernel, dim=N, inputs=[output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy(), np.arange(N) * 2)


@wp.kernel
def test_untile_vector_kernel(input: wp.array(dtype=wp.vec3), output: wp.array(dtype=wp.vec3)):
    i = wp.tid()

    v = input[i] * 0.5

    t = wp.tile(v)
    u = wp.untile(t)

    output[i] = u * 2.0


def test_tile_untile_vector(test, device):
    input = wp.full(16, wp.vec3(1.0, 2.0, 3.0), requires_grad=True, device=device)
    output = wp.zeros_like(input, device=device)

    with wp.Tape() as tape:
        wp.launch(test_untile_vector_kernel, dim=16, inputs=[input, output], block_dim=16, device=device)

    output.grad = wp.ones_like(output, device=device)
    tape.backward()

    assert_np_equal(output.numpy(), input.numpy())
    assert_np_equal(input.grad.numpy(), np.ones((16, 3)))


@wp.kernel
def tile_ones_kernel(out: wp.array(dtype=float)):
    i = wp.tid()

    t = wp.tile_ones(dtype=float, shape=(16, 16))
    s = wp.tile_sum(t)

    wp.tile_store(out, s)


def test_tile_ones(test, device):
    output = wp.zeros(1, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_ones_kernel, dim=[1], inputs=[output], block_dim=TILE_DIM, device=device)

    test.assertAlmostEqual(output.numpy()[0], 256.0)


@wp.kernel
def tile_arange_kernel(out: wp.array2d(dtype=int)):
    i = wp.tid()

    a = wp.tile_arange(17, dtype=int)
    b = wp.tile_arange(5, 23, dtype=int)
    c = wp.tile_arange(0, 34, 2, dtype=int)
    d = wp.tile_arange(-1, 16, dtype=int)
    e = wp.tile_arange(17, 0, -1, dtype=int)

    wp.tile_store(out[0], a)
    wp.tile_store(out[1], b)
    wp.tile_store(out[2], c)
    wp.tile_store(out[3], d)
    wp.tile_store(out[4], e)


def test_tile_arange(test, device):
    N = 17

    output = wp.zeros(shape=(5, N), dtype=int, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_arange_kernel, dim=[1], inputs=[output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy()[0], np.arange(17))
    assert_np_equal(output.numpy()[1], np.arange(5, 22))
    assert_np_equal(output.numpy()[2], np.arange(0, 34, 2))
    assert_np_equal(output.numpy()[3], np.arange(-1, 16))
    assert_np_equal(output.numpy()[4], np.arange(17, 0, -1))


@wp.kernel
def tile_strided_loop_kernel(arr: wp.array(dtype=float), max_val: wp.array(dtype=float)):
    tid, lane = wp.tid()

    num_threads = wp.block_dim()

    thread_max = wp.float32(-wp.inf)

    length = arr.shape[0]
    upper = ((length + num_threads - 1) // num_threads) * num_threads
    for el_id in range(lane, upper, num_threads):
        if el_id < length:
            val = arr[el_id]
        else:
            val = wp.float32(-wp.inf)

        t = wp.tile(val)
        local_max = wp.tile_max(t)

        thread_max = wp.max(thread_max, local_max[0])

    if lane == 0:
        max_val[0] = thread_max


def test_tile_strided_loop(test, device):
    N = 5  # Length of array

    rng = np.random.default_rng(42)
    input = rng.random(N, dtype=np.float32)

    input_wp = wp.array(input, device=device)
    output_wp = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch_tiled(
        tile_strided_loop_kernel,
        dim=[1],
        inputs=[input_wp, output_wp],
        device=device,
        block_dim=128,
    )

    max_wp = output_wp.numpy()
    max_np = np.max(input)
    test.assertAlmostEqual(max_wp[0], max_np, places=4)


devices = get_test_devices()


class TestTileReduce(unittest.TestCase):
    pass


add_function_test(TestTileReduce, "test_tile_reduce_sum", test_tile_reduce_sum, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_min", test_tile_reduce_min, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_max", test_tile_reduce_max, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_argmin", test_tile_reduce_argmin, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_argmax", test_tile_reduce_argmax, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_custom", test_tile_reduce_custom, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_custom_struct", test_tile_reduce_custom_struct, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_grouped_sum", test_tile_reduce_sum, devices=devices)
add_function_test(TestTileReduce, "test_tile_reduce_simt", test_tile_reduce_simt, devices=devices)
add_function_test(TestTileReduce, "test_tile_ones", test_tile_ones, devices=devices)
add_function_test(TestTileReduce, "test_tile_arange", test_tile_arange, devices=devices)
add_function_test(TestTileReduce, "test_tile_untile_scalar", test_tile_untile_scalar, devices=devices)
add_function_test(TestTileReduce, "test_tile_untile_vector", test_tile_untile_vector, devices=devices)
add_function_test(TestTileReduce, "test_tile_strided_loop", test_tile_strided_loop, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
