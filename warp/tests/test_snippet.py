# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def test_basic(test, device):
    snippet = """
    out[tid] = a * x[tid] + y[tid];
    """
    adj_snippet = """
    adj_a += x[tid] * adj_out[tid];
    adj_x[tid] += a * adj_out[tid];
    adj_y[tid] += adj_out[tid];
    """

    @wp.func_native(snippet, adj_snippet)
    def saxpy(
        a: wp.float32,
        x: wp.array(dtype=wp.float32),
        y: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32),
        tid: int,
    ):  # fmt: skip
        ...

    @wp.kernel
    def saxpy_cu(
        a: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)
    ):
        tid = wp.tid()
        saxpy(a, x, y, out, tid)

    @wp.kernel
    def saxpy_py(
        a: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)
    ):
        tid = wp.tid()
        out[tid] = a * x[tid] + y[tid]

    N = 128

    a1 = 2.0
    x1 = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device, requires_grad=True)
    y1 = wp.zeros_like(x1)
    out1 = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device)
    adj_out1 = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, device=device)

    a2 = 2.0
    x2 = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device, requires_grad=True)
    y2 = wp.zeros_like(x2)
    out2 = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device)
    adj_out2 = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, device=device)

    tape = wp.Tape()

    with tape:
        wp.launch(kernel=saxpy_cu, dim=N, inputs=[a1, x1, y1], outputs=[out1], device=device)
        wp.launch(kernel=saxpy_py, dim=N, inputs=[a2, x2, y2], outputs=[out2], device=device)

    tape.backward(grads={out1: adj_out1, out2: adj_out2})

    # test forward snippet
    assert_np_equal(out1.numpy(), out2.numpy())

    # test backward snippet
    assert_np_equal(x1.grad.numpy(), a1 * np.ones(N, dtype=np.float32))
    assert_np_equal(x1.grad.numpy(), x2.grad.numpy())

    assert_np_equal(y1.grad.numpy(), np.ones(N, dtype=np.float32))
    assert_np_equal(y1.grad.numpy(), y2.grad.numpy())


def test_shared_memory(test, device):
    snippet = """
        __shared__ int s[128];

        s[tid] = d[tid];
        __syncthreads();
        d[tid] = s[N - tid - 1];
        """

    @wp.func_native(snippet)
    def reverse(d: wp.array(dtype=int), N: int, tid: int):
        """Reverse the array d in place using shared memory."""
        return

    @wp.kernel
    def reverse_kernel(d: wp.array(dtype=int), N: int):
        tid = wp.tid()
        reverse(d, N, tid)

    N = 128
    x = wp.array(np.arange(N, dtype=int), dtype=int, device=device)
    y = np.arange(127, -1, -1, dtype=int)

    wp.launch(kernel=reverse_kernel, dim=N, inputs=[x, N], device=device)

    assert_np_equal(x.numpy(), y)
    assert reverse.__doc__ == "Reverse the array d in place using shared memory."


def test_cpu_snippet(test, device):
    snippet = """
    int inc = 1;
    out[tid] = x[tid] + inc;
    """

    @wp.func_native(snippet)
    def increment_snippet(
        x: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.int32),
        tid: int,
    ):  # fmt: skip
        ...

    @wp.kernel
    def increment(x: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        increment_snippet(x, out, tid)

    N = 128
    x = wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32, device=device)
    out = wp.zeros(N, dtype=wp.int32, device=device)

    wp.launch(kernel=increment, dim=N, inputs=[x], outputs=[out], device=device)

    assert_np_equal(out.numpy(), np.arange(1, N + 1, 1, dtype=np.int32))


def test_custom_replay_grad(test, device):
    num_threads = 16
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    thread_ids = wp.zeros(num_threads, dtype=wp.int32, device=device)
    inputs = wp.array(np.arange(num_threads, dtype=np.float32), device=device, requires_grad=True)
    outputs = wp.zeros_like(inputs)

    snippet = """
        int next_index = atomicAdd(counter, 1);
        thread_values[tid] = next_index;
        """
    replay_snippet = ""

    @wp.func_native(snippet, replay_snippet=replay_snippet)
    def reversible_increment(counter: wp.array(dtype=int), thread_values: wp.array(dtype=int), tid: int):  # fmt: skip
        ...

    @wp.kernel
    def run_atomic_add(
        input: wp.array(dtype=float),
        counter: wp.array(dtype=int),
        thread_values: wp.array(dtype=int),
        output: wp.array(dtype=float),
    ):
        tid = wp.tid()
        reversible_increment(counter, thread_values, tid)
        idx = thread_values[tid]
        output[idx] = input[idx] ** 2.0

    tape = wp.Tape()
    with tape:
        wp.launch(
            run_atomic_add, dim=num_threads, inputs=[inputs, counter, thread_ids], outputs=[outputs], device=device
        )

    tape.backward(grads={outputs: wp.array(np.ones(num_threads, dtype=np.float32), device=device)})
    assert_np_equal(inputs.grad.numpy(), 2.0 * inputs.numpy(), tol=1e-4)


def test_replay_simplification(test, device):
    num_threads = 8
    x = wp.array(1.0 + np.arange(num_threads, dtype=np.float32), device=device, requires_grad=True)
    y = wp.zeros_like(x)
    z = wp.zeros_like(x)

    snippet = "y[tid] = powf(x[tid], 2.0);"
    replay_snippet = "y[tid] = x[tid];"
    adj_snippet = "adj_x[tid] += 2.0 * adj_y[tid];"

    @wp.func_native(snippet, adj_snippet=adj_snippet, replay_snippet=replay_snippet)
    def square(x: wp.array(dtype=float), y: wp.array(dtype=float), tid: int):  # fmt: skip
        ...

    @wp.kernel
    def log_square_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float)):
        tid = wp.tid()
        square(x, y, tid)
        z[tid] = wp.log(y[tid])

    tape = wp.Tape()
    with tape:
        wp.launch(log_square_kernel, dim=num_threads, inputs=[x, y], outputs=[z], device=device)

    tape.backward(grads={z: wp.array(np.ones(num_threads, dtype=np.float32), device=device)})
    assert_np_equal(x.grad.numpy(), 2.0 / (1.0 + np.arange(num_threads)), tol=1e-6)


def test_recompile_snippet(test, device):
    snippet = """
    int inc = 1;
    out[tid] = x[tid] + inc;
    """

    @wp.func_native(snippet)
    def increment_snippet(
        x: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.int32),
        tid: int,
    ):  # fmt: skip
        ...

    @wp.kernel
    def increment(x: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        increment_snippet(x, out, tid)

    N = 128
    x = wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32, device=device)
    out = wp.zeros(N, dtype=wp.int32, device=device)

    wp.launch(kernel=increment, dim=N, inputs=[x], outputs=[out], device=device)

    assert_np_equal(out.numpy(), np.arange(1, N + 1, 1, dtype=np.int32))

    snippet = """
    int inc = 2;
    out[tid] = x[tid] + inc;
    """

    @wp.func_native(snippet)
    def increment_snippet(
        x: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.int32),
        tid: int,
    ):  # fmt: skip
        ...

    wp.launch(kernel=increment, dim=N, inputs=[x], outputs=[out], device=device)

    assert_np_equal(out.numpy(), 1 + np.arange(1, N + 1, 1, dtype=np.int32))


def test_return_type(test, device):
    snippet = """
        float sq = x * x;
        return sq;
        """
    adj_snippet = """
        adj_x += 2 * x * adj_ret;
        """

    # check python built-in return type compilation
    @wp.func_native(snippet, adj_snippet)
    def square(x: float) -> float: ...

    # check warp built-in return type compilation
    @wp.func_native(snippet, adj_snippet)
    def square(x: wp.float32) -> wp.float32: ...

    @wp.kernel
    def square_kernel(i: wp.array(dtype=float), o: wp.array(dtype=float)):
        tid = wp.tid()
        x = i[tid]
        o[tid] = square(x)

    N = 5
    x = wp.array(np.arange(N, dtype=float), dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=square_kernel, dim=N, inputs=[x, y], device=device)

    y.grad = wp.ones(N, dtype=float, device=device)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([0.0, 1.0, 4.0, 9.0, 16.0]))
    assert_np_equal(x.grad.numpy(), np.array([0.0, 2.0, 4.0, 6.0, 8.0]))


class TestSnippets(unittest.TestCase):
    pass


add_function_test(TestSnippets, "test_basic", test_basic, devices=get_selected_cuda_test_devices())
add_function_test(TestSnippets, "test_shared_memory", test_shared_memory, devices=get_selected_cuda_test_devices())
add_function_test(TestSnippets, "test_cpu_snippet", test_cpu_snippet, devices=["cpu"])
add_function_test(
    TestSnippets, "test_custom_replay_grad", test_custom_replay_grad, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestSnippets, "test_replay_simplification", test_replay_simplification, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestSnippets, "test_recompile_snippet", test_recompile_snippet, devices=get_selected_cuda_test_devices()
)
add_function_test(TestSnippets, "test_return_type", test_return_type, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
