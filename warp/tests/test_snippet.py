# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp._src.codegen import WarpCodegenError
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

    @wp.kernel(module="unique")
    def saxpy_cu(
        a: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)
    ):
        tid = wp.tid()
        saxpy(a, x, y, out, tid)

    @wp.kernel(module="unique")
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

    @wp.kernel(module="unique")
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

    @wp.kernel(module="unique")
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

    @wp.kernel(module="unique")
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

    @wp.kernel(module="unique")
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

    @wp.kernel(module="unique")
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


def test_return_vector_matrix(test, device):
    """Test that func_native correctly handles vector and matrix return types."""
    vec_snippet = """
        return wp::vec_t<3, wp::float32>(x, x, x);
        """

    @wp.func_native(vec_snippet)
    def make_vec3_generic(x: wp.float32) -> wp.types.vector(length=3, dtype=wp.float32): ...

    @wp.func_native(vec_snippet)
    def make_vec3_named(x: wp.float32) -> wp.vec3f: ...

    mat_snippet = """
        return wp::mat_t<2, 2, wp::float32>(x, 0.0f, 0.0f, x);
        """

    @wp.func_native(mat_snippet)
    def make_mat22_generic(x: wp.float32) -> wp.types.matrix(shape=(2, 2), dtype=wp.float32): ...

    @wp.func_native(mat_snippet)
    def make_mat22_named(x: wp.float32) -> wp.mat22f: ...

    @wp.kernel
    def vec_kernel_generic(input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.vec3f)):
        tid = wp.tid()
        output[tid] = make_vec3_generic(input[tid])

    @wp.kernel
    def vec_kernel_named(input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.vec3f)):
        tid = wp.tid()
        output[tid] = make_vec3_named(input[tid])

    @wp.kernel
    def mat_kernel_generic(input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.mat22f)):
        tid = wp.tid()
        output[tid] = make_mat22_generic(input[tid])

    @wp.kernel
    def mat_kernel_named(input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.mat22f)):
        tid = wp.tid()
        output[tid] = make_mat22_named(input[tid])

    N = 3
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    expected_vec = [[float(i + 1)] * 3 for i in range(N)]

    y_vec = wp.zeros(N, dtype=wp.vec3f, device=device)
    wp.launch(vec_kernel_generic, dim=N, inputs=[x, y_vec], device=device)
    for i in range(N):
        np.testing.assert_allclose(y_vec.numpy()[i], expected_vec[i])

    y_vec_named = wp.zeros(N, dtype=wp.vec3f, device=device)
    wp.launch(vec_kernel_named, dim=N, inputs=[x, y_vec_named], device=device)
    for i in range(N):
        np.testing.assert_allclose(y_vec_named.numpy()[i], expected_vec[i])

    y_mat = wp.zeros(N, dtype=wp.mat22f, device=device)
    wp.launch(mat_kernel_generic, dim=N, inputs=[x, y_mat], device=device)
    for i in range(N):
        expected = np.array([[float(i + 1), 0.0], [0.0, float(i + 1)]])
        np.testing.assert_allclose(y_mat.numpy()[i], expected)

    y_mat_named = wp.zeros(N, dtype=wp.mat22f, device=device)
    wp.launch(mat_kernel_named, dim=N, inputs=[x, y_mat_named], device=device)
    for i in range(N):
        expected = np.array([[float(i + 1), 0.0], [0.0, float(i + 1)]])
        np.testing.assert_allclose(y_mat_named.numpy()[i], expected)


def test_return_array(test, device):
    """Test that func_native correctly handles array return types."""
    snippet = """
        return arr;
        """

    @wp.func_native(snippet)
    def passthrough(arr: wp.array(dtype=wp.float32)) -> wp.array(dtype=wp.float32): ...

    @wp.kernel
    def kernel(input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        a = passthrough(input)
        output[tid] = a[tid]

    N = 3
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)
    wp.launch(kernel, dim=N, inputs=[x, y], device=device)
    np.testing.assert_allclose(y.numpy(), [1.0, 2.0, 3.0])


def test_return_fixedarray(test, device):
    """Test that func_native correctly handles fixedarray return types."""
    snippet = """
        wp::fixedarray_t<3, wp::float32> result;
        result[0] = x;
        result[1] = x + 1.0f;
        result[2] = x + 2.0f;
        return result;
        """

    @wp.func_native(snippet)
    def make_fixed(x: wp.float32) -> wp.fixedarray(dtype=wp.float32, shape=3): ...

    @wp.kernel
    def kernel(
        input: wp.array(dtype=wp.float32),
        out0: wp.array(dtype=wp.float32),
        out1: wp.array(dtype=wp.float32),
        out2: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        f = make_fixed(input[tid])
        out0[tid] = f[0]
        out1[tid] = f[1]
        out2[tid] = f[2]

    N = 3
    x = wp.array([10.0, 20.0, 30.0], dtype=wp.float32, device=device)
    o0 = wp.zeros(N, dtype=wp.float32, device=device)
    o1 = wp.zeros(N, dtype=wp.float32, device=device)
    o2 = wp.zeros(N, dtype=wp.float32, device=device)
    wp.launch(kernel, dim=N, inputs=[x, o0, o1, o2], device=device)
    np.testing.assert_allclose(o0.numpy(), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(o1.numpy(), [11.0, 21.0, 31.0])
    np.testing.assert_allclose(o2.numpy(), [12.0, 22.0, 32.0])


def test_return_struct_unsupported(test, device):
    """Test that func_native rejects struct return types with a clear error."""

    @wp.struct
    class Pair:
        x: wp.float32
        y: wp.float32

    snippet = "return {};"

    @wp.func_native(snippet)
    def make_pair(a: wp.float32) -> Pair: ...

    @wp.kernel
    def kernel(input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        p = make_pair(input[tid])
        output[tid] = p.x

    x = wp.array([1.0], dtype=wp.float32, device=device)
    y = wp.zeros(1, dtype=wp.float32, device=device)
    with test.assertRaisesRegex(WarpCodegenError, "unsupported return type"):
        wp.launch(kernel, dim=1, inputs=[x, y], device=device)


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
add_function_test(
    TestSnippets,
    "test_return_vector_matrix",
    test_return_vector_matrix,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestSnippets,
    "test_return_array",
    test_return_array,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestSnippets,
    "test_return_fixedarray",
    test_return_fixedarray,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestSnippets,
    "test_return_struct_unsupported",
    test_return_struct_unsupported,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
