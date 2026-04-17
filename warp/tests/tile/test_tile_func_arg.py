# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for passing tiles of various storage types to @wp.func."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

BLOCK_DIM = 64
TILE_M = 4
TILE_N = 4


# ---- Helper functions used by tests ----


@wp.func
def scale_tile(t: wp.tile(dtype=float, shape=(TILE_M, TILE_N))):
    """Multiply every element of a tile by 2."""
    return t * 2.0


@wp.func
def pass_through_tile(t: wp.tile(dtype=float, shape=(TILE_M, TILE_N))):
    """Return a tile unchanged (tests direct return of parameter)."""
    return t


@wp.func
def add_tiles(
    a: wp.tile(dtype=float, shape=(TILE_M, TILE_N)),
    b: wp.tile(dtype=float, shape=(TILE_M, TILE_N)),
):
    """Add two tiles element-wise."""
    return a + b


@wp.func
def scale_then_pass_through(t: wp.tile(dtype=float, shape=(TILE_M, TILE_N))):
    """Chain two @wp.func calls: scale then pass through."""
    s = scale_tile(t)
    return pass_through_tile(s)


@wp.func
def inplace_add_func(
    a: wp.tile(dtype=float, shape=(TILE_M, TILE_N)),
    b: wp.tile(dtype=float, shape=(TILE_M, TILE_N)),
):
    """Modify a in place (a += b) and return a."""
    a += b
    return a


@wp.func
def modify_tile_no_return(t: wp.tile(dtype=float, shape=(TILE_M, TILE_N))):
    """Write into a tile in place without returning it. Used for both register and shared tiles."""
    t += wp.tile_ones(dtype=float, shape=(TILE_M, TILE_N), storage="register") * 5.0


# ---- Forward tests ----


def test_shared_tile_func_arg(test, device):
    """Pass a shared tile into a @wp.func, get a scaled copy back."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = scale_tile(t)
        wp.tile_store(out, r)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 6.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_register_tile_func_arg(test, device):
    """Pass a register tile into a @wp.func, get a scaled copy back."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        r = scale_tile(t)
        wp.tile_store(out, r)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 5.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 10.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_pass_through_shared_tile(test, device):
    """Return a shared tile directly from a @wp.func (tests cross-storage assignment)."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = pass_through_tile(t)
        wp.tile_store(out, r)

    inp_data = np.arange(TILE_M * TILE_N, dtype=np.float32).reshape(TILE_M, TILE_N)
    inp = wp.array(inp_data, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_array_equal(out.numpy(), inp_data)


def test_mixed_storage_func_args(test, device):
    """Pass one shared tile and one register tile to the same @wp.func."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        input_a: wp.array2d(dtype=float),
        input_b: wp.array2d(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        i = wp.tid()
        a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        r = add_tiles(a, b)
        wp.tile_store(out, r)

    inp_a = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 2.0, device=device)
    inp_b = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp_a, inp_b, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 5.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_nested_func_calls(test, device):
    """Chain two @wp.func calls with tile args (tests template propagation)."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = scale_then_pass_through(t)
        wp.tile_store(out, r)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 4.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 8.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_shared_tile_inplace_no_return(test, device):
    """Modify a shared tile inside a @wp.func without returning it.

    The caller should see the mutation because shared tiles live in shared memory.
    """

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        modify_tile_no_return(t)
        wp.tile_store(out, t)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 8.0  # 3.0 + 5.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_register_tile_inplace_no_return(test, device):
    """Modify a register tile inside a @wp.func without returning it.

    Register tiles are now passed by reference (previously by value), so the
    caller should see the in-place mutation.
    """

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        modify_tile_no_return(t)
        wp.tile_store(out, t)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    expected = np.ones((TILE_M, TILE_N), dtype=np.float32) * 8.0  # 3.0 + 5.0
    np.testing.assert_array_equal(out.numpy(), expected)


# ---- Gradient tests ----


def test_nested_func_calls_grad(test, device):
    """Backward pass through chained @wp.func calls with tile args.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d(dtype=float), loss: wp.array1d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = scale_then_pass_through(t)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp = wp.array(np.full((TILE_M, TILE_N), 4.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    np.testing.assert_allclose(inp.grad.numpy(), np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), rtol=1e-5)


def test_shared_tile_func_grad(test, device):
    """Backward pass through a @wp.func with a shared tile argument.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d(dtype=float), loss: wp.array1d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = scale_tile(t)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp = wp.array(np.full((TILE_M, TILE_N), 3.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    np.testing.assert_allclose(inp.grad.numpy(), np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), rtol=1e-5)


def test_register_tile_func_grad(test, device):
    """Backward pass through a @wp.func with a register tile argument.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d(dtype=float), loss: wp.array1d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        r = scale_tile(t)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp = wp.array(np.full((TILE_M, TILE_N), 3.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    np.testing.assert_allclose(inp.grad.numpy(), np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), rtol=1e-5)


def test_mixed_storage_func_grad(test, device):
    """Backward pass through a @wp.func with mixed shared + register tile args.

    f(a, b) = sum(a + b), df/da = df/db = 1.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(
        input_a: wp.array2d(dtype=float),
        input_b: wp.array2d(dtype=float),
        loss: wp.array1d(dtype=float),
    ):
        i = wp.tid()
        a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        r = add_tiles(a, b)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp_a = wp.array(np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), device=device, requires_grad=True)
    inp_b = wp.array(np.full((TILE_M, TILE_N), 3.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp_a, inp_b, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    expected_grad = np.ones((TILE_M, TILE_N), dtype=np.float32)
    np.testing.assert_allclose(inp_a.grad.numpy(), expected_grad, rtol=1e-5)
    np.testing.assert_allclose(inp_b.grad.numpy(), expected_grad, rtol=1e-5)


def test_inplace_modify_func_grad(test, device):
    """Backward pass through a @wp.func that modifies a tile in place (+=).

    f(a, b) = sum(a + b) computed via a += b; return a.
    df/da = df/db = 1.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(
        input_a: wp.array2d(dtype=float),
        input_b: wp.array2d(dtype=float),
        loss: wp.array1d(dtype=float),
    ):
        i = wp.tid()
        a = wp.tile_load(input_a, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        b = wp.tile_load(input_b, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        r = inplace_add_func(a, b)
        s = wp.tile_sum(r)
        wp.tile_store(loss, s)

    inp_a = wp.array(np.full((TILE_M, TILE_N), 2.0, dtype=np.float32), device=device, requires_grad=True)
    inp_b = wp.array(np.full((TILE_M, TILE_N), 3.0, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[inp_a, inp_b, loss], block_dim=BLOCK_DIM, device=device)

    tape.backward(loss=loss)
    expected_grad = np.ones((TILE_M, TILE_N), dtype=np.float32)
    np.testing.assert_allclose(inp_a.grad.numpy(), expected_grad, rtol=1e-5)
    np.testing.assert_allclose(inp_b.grad.numpy(), expected_grad, rtol=1e-5)


# ---- @wp.func_native tests ----


def test_native_func_shared_tile(test, device):
    """Pass a shared tile to a @wp.func_native and read from it."""

    snippet = """
    // Extract element [0,0] of the tile and write to out[0]
    auto val = tile_extract(t, 0, 0);
    if (threadIdx.x == 0) {
        out[0] = val;
    }
    """

    @wp.func_native(snippet)
    def read_tile_native(t: wp.tile(dtype=float, shape=(TILE_M, TILE_N)), out: wp.array(dtype=float)): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        read_tile_native(t, out)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 7.0, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(out.numpy()[0], 7.0)


def test_native_func_register_tile(test, device):
    """Pass a register tile to a @wp.func_native and read from it."""

    snippet = """
    auto val = tile_extract(t, 0, 0);
    if (threadIdx.x == 0) {
        out[0] = val;
    }
    """

    @wp.func_native(snippet)
    def read_tile_native(t: wp.tile(dtype=float, shape=(TILE_M, TILE_N)), out: wp.array(dtype=float)): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="register")
        read_tile_native(t, out)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 3.0, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(out.numpy()[0], 3.0)


def test_native_func_inplace_modify_tile(test, device):
    """Modify a shared tile in-place inside a @wp.func_native via reference."""

    snippet = """
    // Add 10.0 to element [0,0] of the tile via shared memory pointer
    auto& data = t.data;
    if (threadIdx.x == 0) {
        data.ptr[0] += wp::float32(10.0);
    }
    __syncthreads();
    """

    @wp.func_native(snippet)
    def modify_tile_native(t: wp.tile(dtype=float, shape=(TILE_M, TILE_N))): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        modify_tile_native(t)
        wp.tile_store(out, t)

    inp = wp.array(np.ones((TILE_M, TILE_N), dtype=np.float32) * 5.0, device=device)
    out = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(compute, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)

    result = out.numpy()
    test.assertAlmostEqual(result[0, 0], 15.0)  # 5.0 + 10.0
    test.assertAlmostEqual(result[0, 1], 5.0)  # unchanged


devices = get_test_devices()


class TestTileFuncArg(unittest.TestCase):
    pass


add_function_test(TestTileFuncArg, "test_shared_tile_func_arg", test_shared_tile_func_arg, devices=devices)
add_function_test(TestTileFuncArg, "test_register_tile_func_arg", test_register_tile_func_arg, devices=devices)
add_function_test(TestTileFuncArg, "test_pass_through_shared_tile", test_pass_through_shared_tile, devices=devices)
add_function_test(TestTileFuncArg, "test_mixed_storage_func_args", test_mixed_storage_func_args, devices=devices)
add_function_test(TestTileFuncArg, "test_nested_func_calls", test_nested_func_calls, devices=devices)
add_function_test(
    TestTileFuncArg, "test_shared_tile_inplace_no_return", test_shared_tile_inplace_no_return, devices=devices
)
add_function_test(
    TestTileFuncArg, "test_register_tile_inplace_no_return", test_register_tile_inplace_no_return, devices=devices
)
add_function_test(TestTileFuncArg, "test_shared_tile_func_grad", test_shared_tile_func_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_register_tile_func_grad", test_register_tile_func_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_mixed_storage_func_grad", test_mixed_storage_func_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_nested_func_calls_grad", test_nested_func_calls_grad, devices=devices)
add_function_test(TestTileFuncArg, "test_inplace_modify_func_grad", test_inplace_modify_func_grad, devices=devices)
add_function_test(
    TestTileFuncArg, "test_native_func_shared_tile", test_native_func_shared_tile, devices=get_cuda_test_devices()
)
add_function_test(
    TestTileFuncArg, "test_native_func_register_tile", test_native_func_register_tile, devices=get_cuda_test_devices()
)
add_function_test(
    TestTileFuncArg,
    "test_native_func_inplace_modify_tile",
    test_native_func_inplace_modify_tile,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
