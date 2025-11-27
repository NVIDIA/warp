# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from warp.tests.aux_test_grad_customs import aux_custom_fn
from warp.tests.unittest_utils import *


# atomic add function that memorizes which thread incremented the counter
# so that the correct counter value per thread can be used in the replay
# phase of the backward pass
@wp.func
def reversible_increment(
    counter: wp.array(dtype=int), counter_index: int, value: int, thread_values: wp.array(dtype=int), tid: int
):
    """This is a docstring"""
    next_index = wp.atomic_add(counter, counter_index, value)
    thread_values[tid] = next_index
    return next_index


@wp.func_replay(reversible_increment)
def replay_reversible_increment(
    counter: wp.array(dtype=int), counter_index: int, value: int, thread_values: wp.array(dtype=int), tid: int
):
    """This is a docstring"""
    return thread_values[tid]


@wp.kernel
def run_atomic_add(
    input: wp.array(dtype=float),
    counter: wp.array(dtype=int),
    thread_values: wp.array(dtype=int),
    output: wp.array(dtype=float),
):
    tid = wp.tid()
    idx = reversible_increment(counter, 0, 1, thread_values, tid)
    output[idx] = input[idx] ** 2.0


def test_custom_replay_grad(test, device):
    num_threads = 128
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    thread_ids = wp.zeros(num_threads, dtype=wp.int32, device=device)
    inputs = wp.array(np.arange(num_threads, dtype=np.float32), device=device, requires_grad=True)
    outputs = wp.zeros_like(inputs)

    tape = wp.Tape()
    with tape:
        wp.launch(
            run_atomic_add, dim=num_threads, inputs=[inputs, counter, thread_ids], outputs=[outputs], device=device
        )

    tape.backward(grads={outputs: wp.ones(num_threads, dtype=wp.float32, device=device)})
    assert_np_equal(inputs.grad.numpy(), 2.0 * inputs.numpy(), tol=1e-4)


@wp.func
def overload_fn(x: float, y: float):
    """This is a docstring"""
    return x * 3.0 + y / 3.0, y**2.5


@wp.func_grad(overload_fn)
def overload_fn_grad(x: float, y: float, adj_ret0: float, adj_ret1: float):
    """This is a docstring"""
    wp.adjoint[x] += x * adj_ret0 * 42.0 + y * adj_ret1 * 10.0
    wp.adjoint[y] += y * adj_ret1 * 3.0


@wp.struct
class MyStruct:
    """This is a docstring"""

    scalar: float
    vec: wp.vec3


@wp.func
def overload_fn(x: MyStruct):
    """This is a docstring"""
    return x.vec[0] * x.vec[1] * x.vec[2] * 4.0, wp.length(x.vec), x.scalar**0.5


@wp.func_grad(overload_fn)
def overload_fn_grad(x: MyStruct, adj_ret0: float, adj_ret1: float, adj_ret2: float):
    """This is a docstring"""
    wp.adjoint[x.scalar] += x.scalar * adj_ret0 * 10.0
    wp.adjoint[x.vec][0] += adj_ret0 * x.vec[1] * x.vec[2] * 20.0
    wp.adjoint[x.vec][1] += adj_ret1 * x.vec[0] * x.vec[2] * 30.0
    wp.adjoint[x.vec][2] += adj_ret2 * x.vec[0] * x.vec[1] * 40.0


@wp.kernel
def run_overload_float_fn(
    xs: wp.array(dtype=float), ys: wp.array(dtype=float), output0: wp.array(dtype=float), output1: wp.array(dtype=float)
):
    """This is a docstring"""
    i = wp.tid()
    out0, out1 = overload_fn(xs[i], ys[i])
    output0[i] = out0
    output1[i] = out1


@wp.kernel
def run_overload_struct_fn(xs: wp.array(dtype=MyStruct), output: wp.array(dtype=float)):
    i = wp.tid()
    out0, out1, out2 = overload_fn(xs[i])
    output[i] = out0 + out1 + out2


def test_custom_overload_grad(test, device):
    dim = 3
    xs_float = wp.array(np.arange(1.0, dim + 1.0), dtype=wp.float32, requires_grad=True, device=device)
    ys_float = wp.array(np.arange(10.0, dim + 10.0), dtype=wp.float32, requires_grad=True, device=device)
    out0_float = wp.zeros(dim, device=device)
    out1_float = wp.zeros(dim, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            run_overload_float_fn, dim=dim, inputs=[xs_float, ys_float], outputs=[out0_float, out1_float], device=device
        )
    tape.backward(
        grads={
            out0_float: wp.ones(dim, dtype=wp.float32, device=device),
            out1_float: wp.ones(dim, dtype=wp.float32, device=device),
        }
    )
    assert_np_equal(xs_float.grad.numpy(), xs_float.numpy() * 42.0 + ys_float.numpy() * 10.0)
    assert_np_equal(ys_float.grad.numpy(), ys_float.numpy() * 3.0)

    x0 = MyStruct()
    x0.vec = wp.vec3(1.0, 2.0, 3.0)
    x0.scalar = 4.0
    x1 = MyStruct()
    x1.vec = wp.vec3(5.0, 6.0, 7.0)
    x1.scalar = -1.0
    x2 = MyStruct()
    x2.vec = wp.vec3(8.0, 9.0, 10.0)
    x2.scalar = 19.0
    xs_struct = wp.array([x0, x1, x2], dtype=MyStruct, requires_grad=True, device=device)
    out_struct = wp.zeros(dim, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(run_overload_struct_fn, dim=dim, inputs=[xs_struct], outputs=[out_struct], device=device)
    tape.backward(grads={out_struct: wp.ones(dim, dtype=wp.float32, device=device)})
    xs_struct_np = xs_struct.numpy()
    struct_grads = xs_struct.grad.numpy()
    # fmt: off
    assert_np_equal(
        np.array([g[0] for g in struct_grads]),
        np.array([g[0] * 10.0 for g in xs_struct_np]))
    assert_np_equal(
        np.array([g[1][0] for g in struct_grads]),
        np.array([g[1][1] * g[1][2] * 20.0 for g in xs_struct_np]))
    assert_np_equal(
        np.array([g[1][1] for g in struct_grads]),
        np.array([g[1][0] * g[1][2] * 30.0 for g in xs_struct_np]))
    assert_np_equal(
        np.array([g[1][2] for g in struct_grads]),
        np.array([g[1][0] * g[1][1] * 40.0 for g in xs_struct_np]))
    # fmt: on


@wp.kernel
def run_defined_float_fn(
    xs: wp.array(dtype=float), ys: wp.array(dtype=float), output0: wp.array(dtype=float), output1: wp.array(dtype=float)
):
    i = wp.tid()
    out0, out1 = aux_custom_fn(xs[i], ys[i])
    output0[i] = out0
    output1[i] = out1


def test_custom_import_grad(test, device):
    dim = 3
    xs_float = wp.array(np.arange(1.0, dim + 1.0), dtype=wp.float32, requires_grad=True, device=device)
    ys_float = wp.array(np.arange(10.0, dim + 10.0), dtype=wp.float32, requires_grad=True, device=device)
    out0_float = wp.zeros(dim, device=device)
    out1_float = wp.zeros(dim, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            run_defined_float_fn, dim=dim, inputs=[xs_float, ys_float], outputs=[out0_float, out1_float], device=device
        )
    tape.backward(
        grads={
            out0_float: wp.ones(dim, dtype=wp.float32, device=device),
            out1_float: wp.ones(dim, dtype=wp.float32, device=device),
        }
    )
    assert_np_equal(xs_float.grad.numpy(), xs_float.numpy() * 42.0 + ys_float.numpy() * 10.0)
    assert_np_equal(ys_float.grad.numpy(), ys_float.numpy() * 3.0)


@wp.func
def sigmoid(x: float):
    return 1.0 / (1.0 + wp.exp(-x))


@wp.func_grad(sigmoid)
def adj_sigmoid(x: float, adj: float):
    # unused function to test that we don't run into infinite recursion when calling
    # the forward function from within the gradient function
    wp.adjoint[x] += adj * sigmoid(x) * (1.0 - sigmoid(x))


@wp.func
def sigmoid_no_return(i: int, xs: wp.array(dtype=float), ys: wp.array(dtype=float)):
    # test function that does not return anything
    ys[i] = sigmoid(xs[i])


@wp.func_grad(sigmoid_no_return)
def adj_sigmoid_no_return(i: int, xs: wp.array(dtype=float), ys: wp.array(dtype=float)):
    wp.adjoint[xs][i] += ys[i] * (1.0 - ys[i])


@wp.kernel
def eval_sigmoid(xs: wp.array(dtype=float), ys: wp.array(dtype=float)):
    i = wp.tid()
    sigmoid_no_return(i, xs, ys)


def test_custom_grad_no_return(test, device):
    xs = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, requires_grad=True, device=device)
    ys = wp.zeros_like(xs, device=device)
    ys.grad.fill_(1.0)

    tape = wp.Tape()
    with tape:
        wp.launch(eval_sigmoid, dim=len(xs), inputs=[xs], outputs=[ys], device=device)
    tape.backward()

    sigmoids = ys.numpy()
    grad = xs.grad.numpy()
    assert_np_equal(grad, sigmoids * (1.0 - sigmoids))


@wp.func
def dense_gemm(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    # outputs
    C: wp.array(dtype=float),
):
    # this function doesn't get called but it is an important test for code generation
    # multiply a `m x p` matrix A by a `p x n` matrix B to produce a `m x n` matrix C
    for i in range(m):
        for j in range(n):
            sum = float(0.0)
            for k in range(p):
                if transpose_A:
                    a_i = k * m + i
                else:
                    a_i = i * p + k
                if transpose_B:
                    b_j = j * p + k
                else:
                    b_j = k * n + j
                sum += A[a_i] * B[b_j]

            if add_to_C:
                C[i * n + j] += sum
            else:
                C[i * n + j] = sum


@wp.func_grad(dense_gemm)
def adj_dense_gemm(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    # outputs
    C: wp.array(dtype=float),
):
    # code generation would break here if we didn't defer building the custom grad
    # function until after the forward functions + kernels of the module have been built
    add_to_C = True
    if transpose_A:
        dense_gemm(p, m, n, False, True, add_to_C, B, wp.adjoint[C], wp.adjoint[A])
        dense_gemm(p, n, m, False, False, add_to_C, A, wp.adjoint[C], wp.adjoint[B])
    else:
        dense_gemm(m, p, n, False, not transpose_B, add_to_C, wp.adjoint[C], B, wp.adjoint[A])
        dense_gemm(p, n, m, True, False, add_to_C, A, wp.adjoint[C], wp.adjoint[B])


# Test for nested function calls with custom gradients
# This tests that custom gradient functions are generated in the correct order
# to avoid undefined reference errors during compilation.
@wp.func
def custom_norm(v: wp.vec3):
    """Compute vector length with custom gradient."""
    return wp.length(v)


@wp.func_grad(custom_norm)
def adj_custom_norm(v: wp.vec3, adj_ret: float):
    """Custom gradient that normalizes the adjoint."""
    # Use normalized gradient instead of the automatic one
    wp.adjoint[v] += wp.normalize(v) * adj_ret


@wp.func
def nested_norm(v: wp.vec3):
    """Function that calls another function with custom gradient."""
    # This call will generate an adjoint that references adj_custom_norm
    return custom_norm(v)


@wp.kernel
def test_nested_custom_grad_kernel(vectors: wp.array(dtype=wp.vec3), norms: wp.array(dtype=float)):
    i = wp.tid()
    norms[i] = nested_norm(vectors[i])


def test_nested_custom_grad(test, device):
    """Test that nested functions with custom gradients generate correct code.

    This test ensures that when a function with a custom gradient (custom_norm) is
    called by another function (nested_norm), the custom gradient function is
    generated before the auto-generated adjoint that references it. Without proper
    code generation ordering, this would cause a compilation error.
    """
    # Test vectors: chosen to make normalization easy to verify
    # [0, 2, 0] has length 2, normalizes to [0, 1, 0]
    # [3, 0, 0] has length 3, normalizes to [1, 0, 0]
    # [0, 0, -1] has length 1, normalizes to [0, 0, -1]
    vecs_np = np.array(
        [[0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )

    # Input: 3 vectors (vec3), Output: 3 scalars (float)
    vecs = wp.array(vecs_np, dtype=wp.vec3, requires_grad=True, device=device)
    norms = wp.zeros(vecs.shape[0], dtype=wp.float32, device=device)

    # Forward pass: compute length of each vector
    tape = wp.Tape()
    with tape:
        wp.launch(test_nested_custom_grad_kernel, dim=vecs.shape[0], inputs=[vecs], outputs=[norms], device=device)

    expected_norms = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    assert_np_equal(norms.numpy(), expected_norms, tol=1e-4)

    # Backward pass: seed with scalar adjoints (one per output)
    norms_grad = wp.ones(vecs.shape[0], dtype=wp.float32, device=device)  # [1.0, 1.0, 1.0]
    tape.backward(grads={norms: norms_grad})

    # Gradient of length is the normalized vector: ∂||v||/∂v = v/||v||
    # [0, 2, 0] → [0, 1, 0], [3, 0, 0] → [1, 0, 0], [0, 0, -1] → [0, 0, -1]
    expected_grad = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    assert_np_equal(vecs.grad.numpy(), expected_grad, tol=1e-4)


# Test for correct ordering when custom gradient functions depend on regular functions
# This test catches a bug where function ordering doesn't preserve insertion order,
# causing forward functions with custom grads to be placed before helper functions they call.
@wp.func
def helper_multiply(x: float):
    """Regular helper function used by custom gradient function."""
    return x * 2.0


@wp.func
def custom_transform(x: float):
    """Function with custom gradient that depends on helper_multiply."""
    # This function calls a regular helper - important for testing ordering!
    return helper_multiply(x) + 1.0


@wp.func_grad(custom_transform)
def adj_custom_transform(x: float, adj_ret: float):
    """Custom gradient for custom_transform."""
    # Custom gradient: derivative is 2.0 (from helper_multiply)
    wp.adjoint[x] += 2.0 * adj_ret


@wp.func
def outer_transform(x: float):
    """Function that calls custom_transform."""
    return custom_transform(x) * 3.0


@wp.kernel
def test_custom_grad_with_helper_kernel(inputs: wp.array(dtype=float), outputs: wp.array(dtype=float)):
    i = wp.tid()
    outputs[i] = outer_transform(inputs[i])


def test_custom_grad_with_helper_dependency(test, device):
    """Test that custom gradient functions can depend on regular helper functions.

    This test ensures the code generation ordering preserves the original insertion order
    for regular functions, even when custom gradient functions are involved. Without
    proper ordering, the forward function with custom gradient (custom_transform) might
    be placed before the helper function it calls (helper_multiply), causing compilation errors.

    The dependency chain is:
    1. helper_multiply (regular function)
    2. custom_transform (calls helper_multiply, has custom gradient)
    3. adj_custom_transform (custom gradient)
    4. outer_transform (calls custom_transform, generates auto-adjoint)
    """
    n = 3
    inputs_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    inputs = wp.array(inputs_np, dtype=wp.float32, requires_grad=True, device=device)
    outputs = wp.zeros(n, dtype=wp.float32, device=device)

    # Forward pass: output[i] = ((input[i] * 2.0) + 1.0) * 3.0
    tape = wp.Tape()
    with tape:
        wp.launch(test_custom_grad_with_helper_kernel, dim=n, inputs=[inputs], outputs=[outputs], device=device)

    # Expected: [1*2+1]*3=9, [2*2+1]*3=15, [3*2+1]*3=21
    expected_outputs = np.array([9.0, 15.0, 21.0], dtype=np.float32)
    assert_np_equal(outputs.numpy(), expected_outputs, tol=1e-4)

    # Backward pass
    outputs_grad = wp.ones(n, dtype=wp.float32, device=device)
    tape.backward(grads={outputs: outputs_grad})

    # Gradient: d/dx[(2x+1)*3] = 2*3 = 6 (from custom gradient)
    expected_grad = np.array([6.0, 6.0, 6.0], dtype=np.float32)
    assert_np_equal(inputs.grad.numpy(), expected_grad, tol=1e-4)


# Native snippet called by function with custom gradient
# This tests that native snippets are available when generating forward code
# for functions that have custom gradients

add_two_snippet = """
    return a + 2.0f;
"""

add_two_adj_snippet = """
    // derivative of (a + 2) w.r.t. a is 1
    adj_a += adj_ret;
"""


@wp.func_native(add_two_snippet, adj_snippet=add_two_adj_snippet)
def add_two_native(a: float) -> float:
    """Native snippet that adds 2 to input."""
    ...


@wp.func
def func_with_native_and_custom_grad(x: float):
    """Function that calls native snippet and has custom gradient."""
    # Forward pass calls native snippet
    y = add_two_native(x)
    return y * 3.0


@wp.func_grad(func_with_native_and_custom_grad)
def adj_func_with_native_and_custom_grad(x: float, adj_ret: float):
    """Custom gradient that provides derivative: d/dx[(x+2)*3] = 3."""
    wp.adjoint[x] += 3.0 * adj_ret


@wp.kernel
def test_native_snippet_in_forward_kernel(inputs: wp.array(dtype=float), outputs: wp.array(dtype=float)):
    i = wp.tid()
    outputs[i] = func_with_native_and_custom_grad(inputs[i])


def test_native_snippet_in_forward_with_custom_grad(test, device):
    """Test that functions with custom gradients can call native snippets in forward pass.

    This covers the case where:
    1. A native snippet function exists (add_two_native)
    2. A function calls it in forward pass (func_with_native_and_custom_grad)
    3. That function has a custom gradient

    Tests two-pass codegen where native snippets must be in Pass 1.
    """
    n = 3
    inputs_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = wp.array(inputs_np, dtype=wp.float32, requires_grad=True, device=device)
    outputs = wp.zeros(n, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(test_native_snippet_in_forward_kernel, dim=n, inputs=[inputs], outputs=[outputs], device=device)

    # Forward: (x+2)*3 = [9.0, 12.0, 15.0]
    expected_outputs = np.array([9.0, 12.0, 15.0], dtype=np.float32)
    assert_np_equal(outputs.numpy(), expected_outputs, tol=1e-4)

    # Backward pass
    outputs_grad = wp.ones(n, dtype=wp.float32, device=device)
    tape.backward(grads={outputs: outputs_grad})

    # Gradient from custom gradient: 3
    expected_grad = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    assert_np_equal(inputs.grad.numpy(), expected_grad, tol=1e-4)


# Native snippet called by custom gradient function
# This tests that native snippets are available when generating custom gradient code

multiply_by_two_snippet = """
    return a * 2.0f;
"""

multiply_by_two_adj_snippet = """
    // derivative of (a * 2) w.r.t. a is 2
    adj_a += 2.0f * adj_ret;
"""


@wp.func_native(multiply_by_two_snippet, adj_snippet=multiply_by_two_adj_snippet)
def multiply_by_two_native(a: float) -> float:
    """Native snippet that multiplies input by 2."""
    ...


@wp.func
def func_with_custom_grad_calling_native(x: float):
    """Regular function with custom gradient that calls native snippet."""
    return x + 1.0


@wp.func_grad(func_with_custom_grad_calling_native)
def adj_func_with_custom_grad_calling_native(x: float, adj_ret: float):
    """Custom gradient that calls a native snippet."""
    # Custom gradient computes: derivative = 2 (by calling native snippet)
    factor = multiply_by_two_native(1.0)
    wp.adjoint[x] += factor * adj_ret


@wp.kernel
def test_native_snippet_in_custom_grad_kernel(inputs: wp.array(dtype=float), outputs: wp.array(dtype=float)):
    i = wp.tid()
    outputs[i] = func_with_custom_grad_calling_native(inputs[i])


def test_native_snippet_in_custom_grad(test, device):
    """Test that custom gradient functions can call native snippets.

    This covers the case where:
    1. A native snippet function exists (multiply_by_two_native)
    2. A custom gradient function calls it (adj_func_with_custom_grad_calling_native)

    Tests two-pass codegen where native snippets in Pass 1 are available to
    custom gradients in Pass 2.
    """
    n = 3
    inputs_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = wp.array(inputs_np, dtype=wp.float32, requires_grad=True, device=device)
    outputs = wp.zeros(n, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(test_native_snippet_in_custom_grad_kernel, dim=n, inputs=[inputs], outputs=[outputs], device=device)

    # Forward: x+1 = [2.0, 3.0, 4.0]
    expected_outputs = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    assert_np_equal(outputs.numpy(), expected_outputs, tol=1e-4)

    # Backward pass
    outputs_grad = wp.ones(n, dtype=wp.float32, device=device)
    tape.backward(grads={outputs: outputs_grad})

    # Gradient from custom gradient: multiply_by_two_native(1.0) = 2.0
    expected_grad = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    assert_np_equal(inputs.grad.numpy(), expected_grad, tol=1e-4)


devices = get_test_devices()


class TestGradCustoms(unittest.TestCase):
    def test_wrapped_docstring(self):
        self.assertTrue("This is a docstring" in reversible_increment.__doc__)
        self.assertTrue("This is a docstring" in replay_reversible_increment.__doc__)
        self.assertTrue("This is a docstring" in overload_fn.__doc__)
        self.assertTrue("This is a docstring" in overload_fn_grad.__doc__)
        self.assertTrue("This is a docstring" in run_overload_float_fn.__doc__)
        self.assertTrue("This is a docstring" in MyStruct.__doc__)


add_function_test(TestGradCustoms, "test_custom_replay_grad", test_custom_replay_grad, devices=devices)
add_function_test(TestGradCustoms, "test_custom_overload_grad", test_custom_overload_grad, devices=devices)
add_function_test(TestGradCustoms, "test_custom_import_grad", test_custom_import_grad, devices=devices)
add_function_test(TestGradCustoms, "test_custom_grad_no_return", test_custom_grad_no_return, devices=devices)
add_function_test(TestGradCustoms, "test_nested_custom_grad", test_nested_custom_grad, devices=devices)
add_function_test(
    TestGradCustoms, "test_custom_grad_with_helper_dependency", test_custom_grad_with_helper_dependency, devices=devices
)
add_function_test(
    TestGradCustoms,
    "test_native_snippet_in_forward_with_custom_grad",
    test_native_snippet_in_forward_with_custom_grad,
    devices=devices,
)
add_function_test(
    TestGradCustoms, "test_native_snippet_in_custom_grad", test_native_snippet_in_custom_grad, devices=devices
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
