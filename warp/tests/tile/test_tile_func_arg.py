# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for passing tiles of various storage types to @wp.func."""

import functools
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

BLOCK_DIM = 64
TILE_M = 4
TILE_N = 4


# ---- Helper functions used by tests ----


@wp.func
def scale_tile(t: wp.tile[float, TILE_M, TILE_N]):
    """Multiply every element of a tile by 2."""
    return t * 2.0


@wp.func
def pass_through_tile(t: wp.tile[float, TILE_M, TILE_N]):
    """Return a tile unchanged (tests direct return of parameter)."""
    return t


@wp.func
def add_tiles(
    a: wp.tile[float, TILE_M, TILE_N],
    b: wp.tile[float, TILE_M, TILE_N],
):
    """Add two tiles element-wise."""
    return a + b


@wp.func
def scale_then_pass_through(t: wp.tile[float, TILE_M, TILE_N]):
    """Chain two @wp.func calls: scale then pass through."""
    s = scale_tile(t)
    return pass_through_tile(s)


@wp.func
def inplace_add_func(
    a: wp.tile[float, TILE_M, TILE_N],
    b: wp.tile[float, TILE_M, TILE_N],
):
    """Modify a in place (a += b) and return a."""
    a += b
    return a


@wp.func
def modify_tile_no_return(t: wp.tile[float, TILE_M, TILE_N]):
    """Write into a tile in place without returning it. Used for both register and shared tiles."""
    t += wp.tile_ones(dtype=float, shape=(TILE_M, TILE_N), storage="register") * 5.0


@wp.func
def read_elements_func(t: wp.tile[float, TILE_M, TILE_N]):
    """Combine two elements of a tile passed to a helper."""
    return t[1, 2] * 2.0 + t[3, 0]


# ---- Forward tests ----


def test_shared_tile_func_arg(test, device):
    """Pass a shared tile into a @wp.func, get a scaled copy back."""

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
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
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
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
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
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
        input_a: wp.array2d[float],
        input_b: wp.array2d[float],
        out: wp.array2d[float],
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
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
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
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
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
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
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
    """Test gradient propagation through chained tile functions.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d[float], loss: wp.array1d[float]):
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
    """Test gradient propagation through a function with a shared tile argument.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d[float], loss: wp.array1d[float]):
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
    """Test gradient propagation through a function with a register tile argument.

    f(x) = sum(2*x), df/dx = 2.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(input: wp.array2d[float], loss: wp.array1d[float]):
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
    """Test gradient propagation through mixed tile storage.

    f(a, b) = sum(a + b), df/da = df/db = 1.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(
        input_a: wp.array2d[float],
        input_b: wp.array2d[float],
        loss: wp.array1d[float],
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
    """Test gradient propagation through an in-place tile update.

    f(a, b) = sum(a + b) computed via a += b; return a.
    df/da = df/db = 1.0 for every element.
    """

    @wp.kernel(module="unique")
    def compute(
        input_a: wp.array2d[float],
        input_b: wp.array2d[float],
        loss: wp.array1d[float],
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
    def read_tile_native(t: wp.tile[float, TILE_M, TILE_N], out: wp.array[float]): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array[float]):
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
    def read_tile_native(t: wp.tile[float, TILE_M, TILE_N], out: wp.array[float]): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array[float]):
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
    def modify_tile_native(t: wp.tile[float, TILE_M, TILE_N]): ...

    @wp.kernel(enable_backward=False, module="unique")
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
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


def test_tile_func_return_composition(test, device, kind, backward):
    """Read elements of returned tiles across different helper compositions."""

    @wp.func
    def return_tile(t: wp.tile[float, TILE_M, TILE_N], early: bool):
        if wp.static(kind == "nested"):
            return pass_through_tile(t)
        elif wp.static(kind == "local"):
            read_elements_func(t)
            return t
        elif wp.static(kind == "conditional"):
            if early:
                return t
            read_elements_func(t)
            return t
        elif wp.static(kind == "allocated"):
            return t * 2.0
        else:
            return t

    @wp.func
    def load_tile(x: wp.array2d[float]):
        return wp.tile_load(x, shape=(TILE_M, TILE_N), storage="register")

    @wp.kernel(module="unique", enable_backward=backward)
    def compute(x: wp.array2d[float], out: wp.array[float], early: bool):
        _, lane = wp.tid()
        t = wp.tile_load(x, shape=(TILE_M, TILE_N), storage="register")
        if wp.static(kind == "view"):
            t = wp.tile_transpose(t)
        elif wp.static(kind == "broadcast"):
            t = wp.tile_broadcast(t[0:1, :], shape=(TILE_M, TILE_N))
        if wp.static(kind == "load"):
            u = load_tile(x)
        else:
            u = return_tile(t, early)
        s = read_elements_func(u)
        if lane == 0:
            out[0] = s

    x_np = np.arange(TILE_M * TILE_N, dtype=np.float32).reshape(TILE_M, TILE_N)
    scale = 2.0 if kind == "allocated" else 1.0
    for early in (False, True) if kind == "conditional" else (False,):
        with test.subTest(early=early):
            x = wp.array(x_np, device=device, requires_grad=backward)
            out = wp.zeros(1, device=device, requires_grad=backward)
            with wp.Tape() as tape:
                wp.launch_tiled(compute, dim=1, inputs=[x, out, early], block_dim=BLOCK_DIM, device=device)
            expected_value = {"view": 21.0, "broadcast": 4.0}.get(kind, 24.0 * scale)
            np.testing.assert_allclose(out.numpy(), [expected_value])
            if backward:
                tape.backward(loss=out)
                expected = np.zeros_like(x_np)
                expected[1, 2] = 2.0 * scale
                expected[3, 0] = scale
                if kind == "view":
                    expected = expected.T
                elif kind == "broadcast":
                    expected = np.zeros_like(x_np)
                    expected[0, 2] = 2.0
                    expected[0, 0] = 1.0
                np.testing.assert_allclose(x.grad.numpy(), expected)


def test_tile_func_multiple_return_composition(test, device, backward):
    """Read one returned tile while preserving other results and repeated calls."""

    @wp.func
    def pair(t: wp.tile[float, TILE_M, TILE_N]):
        return t, t, 3.0

    @wp.kernel(module="unique", enable_backward=backward)
    def compute(x: wp.array2d[float], out: wp.array[float], saved: wp.array2d[float]):
        _, lane = wp.tid()
        t = wp.tile_load(x, shape=(TILE_M, TILE_N), storage="register")
        before = pass_through_tile(t)
        u, v, factor = pair(t)
        s = read_elements_func(u)
        after = pass_through_tile(t)
        wp.tile_store(saved, before + v + after)
        if lane == 0:
            out[0] = factor * s

    x_np = np.arange(TILE_M * TILE_N, dtype=np.float32).reshape(TILE_M, TILE_N)
    x = wp.array(x_np, device=device, requires_grad=backward)
    out = wp.zeros(1, device=device, requires_grad=backward)
    saved = wp.zeros_like(x)
    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=1, inputs=[x, out, saved], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_allclose(out.numpy(), [72.0])
    np.testing.assert_allclose(saved.numpy(), 3.0 * x_np)
    if backward:
        tape.backward(grads={out: wp.ones_like(out), saved: wp.ones_like(saved)})
        expected = np.full_like(x_np, 3.0)
        expected[1, 2] += 6.0
        expected[3, 0] += 3.0
        np.testing.assert_allclose(x.grad.numpy(), expected)


def test_tile_func_borrowed_parameter_return(test, device):
    """Preserve factory annotations that explicitly return a borrowed shared view."""
    # Factory-style compatibility: storage and ownership are not subscript parameters.
    view_type = wp.tile(dtype=float, shape=(TILE_M, TILE_N), storage="shared", owner=False)

    @wp.func
    def return_view(t: view_type):
        return t

    @wp.kernel(module="unique", enable_backward=False)
    def compute(out: wp.array2d[float]):
        t = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float, storage="shared")
        u = return_view(t)
        u[0, 0] = 123.0
        wp.tile_store(out, t)

    out = wp.empty((TILE_M, TILE_N), device=device)
    wp.launch_tiled(compute, dim=1, outputs=[out], block_dim=BLOCK_DIM, device=device)
    expected = np.zeros((TILE_M, TILE_N), dtype=np.float32)
    expected[0, 0] = 123.0
    np.testing.assert_array_equal(out.numpy(), expected)


def test_register_tile_func_where_grad(test, device):
    """Preserve selection gradients when one operand is also passed to a helper."""

    @wp.kernel(module="unique")
    def compute(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float], choose_a: bool):
        x = wp.tile_load(a, shape=(TILE_M, TILE_N), storage="register")
        y = wp.tile_load(b, shape=(TILE_M, TILE_N), storage="register")
        selected = wp.where(choose_a, x, y)
        # Read one operand through a helper before storing the selection.
        read_elements_func(x)
        wp.tile_store(out, selected)

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((TILE_M, TILE_N)).astype(np.float32)
    b_np = rng.standard_normal((TILE_M, TILE_N)).astype(np.float32)
    seed = rng.standard_normal((TILE_M, TILE_N)).astype(np.float32)
    for choose_a in (False, True):
        with test.subTest(choose_a=choose_a):
            a = wp.array(a_np, device=device, requires_grad=True)
            b = wp.array(b_np, device=device, requires_grad=True)
            out = wp.zeros_like(a)
            with wp.Tape() as tape:
                wp.launch_tiled(compute, dim=[1], inputs=[a, b, out, choose_a], block_dim=BLOCK_DIM, device=device)
            np.testing.assert_allclose(out.numpy(), a_np if choose_a else b_np)
            tape.backward(grads={out: wp.array(seed, device=device)})
            np.testing.assert_allclose(a.grad.numpy(), seed if choose_a else np.zeros_like(seed))
            np.testing.assert_allclose(b.grad.numpy(), np.zeros_like(seed) if choose_a else seed)


def test_tile_func_thread_conversions(test, device):
    """Preserve thread/tile conversion gradients across an unused helper read."""

    def make_kernel(block_dim, from_threads, read):
        @wp.func
        def read_first(t: wp.tile[float, block_dim]):
            return t[0]

        @wp.kernel(module="unique")
        def compute(x: wp.array[float], out: wp.array[float]):
            lane = wp.tid()
            if wp.static(from_threads):
                value = x[lane]
                t = wp.tile(value)
            else:
                t = wp.tile_load(x, shape=block_dim, storage="register")
            if wp.static(read):
                read_first(t)
            if wp.static(from_threads):
                wp.tile_store(out, t)
            else:
                out[lane] = wp.untile(t)

        return compute

    for block_dim in (32, 128):
        for from_threads in (False, True):
            for read in (False, True):
                with test.subTest(block_dim=block_dim, from_threads=from_threads, read=read):
                    compute = make_kernel(block_dim, from_threads, read)
                    rng = np.random.default_rng(814)
                    values = rng.standard_normal(block_dim).astype(np.float32)
                    seeds = rng.standard_normal(block_dim).astype(np.float32)
                    x = wp.array(values, device=device, requires_grad=True)
                    out = wp.zeros_like(x)
                    with wp.Tape() as tape:
                        wp.launch(compute, dim=block_dim, inputs=[x], outputs=[out], block_dim=block_dim, device=device)
                    tape.backward(grads={out: wp.array(seeds, device=device)})
                    np.testing.assert_array_equal(out.numpy(), values)
                    np.testing.assert_array_equal(x.grad.numpy(), seeds)


def test_register_tile_func_large_read(test, device, backward):
    """Read a register tile too large to stage entirely in CUDA shared memory."""

    @wp.func
    def read_element(t: wp.tile[float, 32768]):
        return t[1234]

    @wp.kernel(module="unique", enable_backward=backward)
    def compute(x: wp.array[float], out: wp.array[float]):
        _, lane = wp.tid()
        t = wp.tile_load(x, shape=32768, storage="register")
        value = read_element(t)
        if lane == 0:
            out[0] = value

    x = wp.array(np.arange(32768, dtype=np.float32), device=device, requires_grad=backward)
    out = wp.zeros(1, device=device, requires_grad=backward)
    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=1, inputs=[x], outputs=[out], block_dim=128, device=device)
    np.testing.assert_array_equal(out.numpy(), [1234.0])
    if backward:
        tape.backward(grads={out: wp.full_like(out, 3.0)})
        expected = np.zeros(32768, dtype=np.float32)
        expected[1234] = 3.0
        np.testing.assert_array_equal(x.grad.numpy(), expected)


def test_register_tile_func_matrix_read(test, device, all_lanes):
    """Keep matrix-valued register reads within CUDA shared-memory limits."""

    @wp.func
    def read(t: wp.tile[wp.mat44d, 1]):
        return t[0]

    @wp.kernel(module="unique")
    def compute(x: wp.array[wp.mat44d], out: wp.array[wp.mat44d]):
        _, lane = wp.tid()
        t = wp.tile_load(x, shape=1, storage="register")
        value = read(t)
        if wp.static(all_lanes) or lane == 0:
            out[lane] = value

    # Enough lanes that one matrix per lane exceeds the 48 KB static shared-memory
    # limit, without exceeding the register file for the backward kernel.
    block_dim = 384
    lanes = block_dim if all_lanes else 1
    rng = np.random.default_rng(764)
    values = rng.standard_normal((1, 4, 4))
    seeds = rng.standard_normal((lanes, 4, 4))
    x = wp.array(values, dtype=wp.mat44d, device=device, requires_grad=True)
    out = wp.zeros(lanes, dtype=wp.mat44d, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=1, inputs=[x], outputs=[out], block_dim=block_dim, device=device)
    tape.backward(grads={out: wp.array(seeds, dtype=wp.mat44d, device=device)})
    np.testing.assert_array_equal(out.numpy(), np.broadcast_to(values, seeds.shape))
    np.testing.assert_allclose(x.grad.numpy(), seeds.sum(axis=0, keepdims=True), rtol=1e-12, atol=1e-12)


def test_register_tile_func_element_read_grad(test, device, dtype, block_dim):
    """Accumulate repeated element-read gradients from all lanes into register owners."""

    @wp.func
    def read(t: wp.tile[dtype, 8, 8], index: int):
        return t[index // 8, index % 8]

    @wp.kernel(module="unique")
    def compute(x: wp.array2d[dtype], a: wp.array[dtype], b: wp.array[dtype]):
        _, lane = wp.tid()
        t = wp.tile_load(x, shape=(8, 8), storage="register")
        index = 6
        a[lane] = read(t, index)
        b[lane] = read(t, (index + 17) % 64)

    component_shape = {wp.float32: (), wp.float64: (), wp.vec3: (3,), wp.mat22: (2, 2), wp.mat33d: (3, 3)}[dtype]
    np_dtype = np.float64 if dtype in (wp.float64, wp.mat33d) else np.float32
    rng = np.random.default_rng(578)
    values = rng.standard_normal((8, 8, *component_shape)).astype(np_dtype)
    lanes = block_dim if device.is_cuda or wp.config.enable_cpu_blocks else 1
    seeds_a = rng.standard_normal((lanes, *component_shape)).astype(np_dtype)
    seeds_b = rng.standard_normal((lanes, *component_shape)).astype(np_dtype)
    x = wp.array(values, dtype=dtype, device=device, requires_grad=True)
    a = wp.zeros(lanes, dtype=dtype, device=device, requires_grad=True)
    b = wp.zeros_like(a)
    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=1, inputs=[x], outputs=[a, b], block_dim=block_dim, device=device)
    tape.backward(
        grads={a: wp.array(seeds_a, dtype=dtype, device=device), b: wp.array(seeds_b, dtype=dtype, device=device)}
    )
    indices = np.full(lanes, 6)
    flat = values.reshape(64, *component_shape)
    np.testing.assert_allclose(a.numpy(), flat[indices])
    np.testing.assert_allclose(b.numpy(), flat[(indices + 17) % 64])
    expected = np.zeros_like(flat)
    np.add.at(expected, indices, seeds_a)
    np.add.at(expected, (indices + 17) % 64, seeds_b)
    np.testing.assert_allclose(x.grad.numpy().reshape(expected.shape), expected, rtol=2e-5, atol=2e-5)


def test_register_tile_func_element_read(test, device):
    """Read elements of a register tile in a user function and check the gradient."""

    @wp.kernel(module="unique")
    def compute(x: wp.array2d[float], out: wp.array1d[float]):
        _, lane = wp.tid()
        t = wp.tile_load(x, shape=(TILE_M, TILE_N), storage="register")
        s = read_elements_func(t)
        # Every thread holds the result; store it from one thread so that only one passes a gradient back
        if lane == 0:
            out[0] = s

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((TILE_M, TILE_N)).astype(np.float32)

    x = wp.array(x_np, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[x], outputs=[out], block_dim=BLOCK_DIM, device=device)

    np.testing.assert_allclose(out.numpy(), [x_np[1, 2] * 2.0 + x_np[3, 0]], rtol=1e-6)

    tape.backward(loss=out)
    expected_grad = np.zeros_like(x_np)
    expected_grad[1, 2] = 2.0
    expected_grad[3, 0] = 1.0
    np.testing.assert_allclose(x.grad.numpy(), expected_grad)


devices = get_test_devices()


class TestTileFuncArg(unittest.TestCase):
    pass


for kind in ("identity", "nested", "local", "conditional", "allocated", "load", "view", "broadcast"):
    add_function_test(
        TestTileFuncArg,
        f"test_tile_func_return_composition_{kind}_forward",
        functools.partial(test_tile_func_return_composition, kind=kind, backward=False),
        devices=devices,
    )
add_function_test(
    TestTileFuncArg,
    "test_tile_func_multiple_return_composition_forward",
    functools.partial(test_tile_func_multiple_return_composition, backward=False),
    devices=devices,
)
add_function_test(
    TestTileFuncArg,
    "test_tile_func_borrowed_parameter_return",
    test_tile_func_borrowed_parameter_return,
    devices=devices,
)
add_function_test(
    TestTileFuncArg, "test_register_tile_func_where_grad", test_register_tile_func_where_grad, devices=devices
)
add_function_test(
    TestTileFuncArg,
    "test_tile_func_thread_conversions",
    test_tile_func_thread_conversions,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestTileFuncArg,
    "test_tile_func_thread_conversions_cpu_blocks",
    test_tile_func_thread_conversions,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileFuncArg,
    "test_register_tile_func_large_read_forward",
    functools.partial(test_register_tile_func_large_read, backward=False),
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestTileFuncArg,
    "test_register_tile_func_matrix_read_owner",
    functools.partial(test_register_tile_func_matrix_read, all_lanes=False),
    devices=get_cuda_test_devices(),
)


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


for dtype in (wp.float32, wp.float64, wp.vec3, wp.mat22, wp.mat33d):
    # one partial warp, a partial last warp, and whole warps
    for block_dim in (24, 40, 128):
        name = f"test_register_tile_func_element_read_grad_{dtype.__name__}_{block_dim}"
        func = functools.partial(test_register_tile_func_element_read_grad, dtype=dtype, block_dim=block_dim)
        add_function_test(TestTileFuncArg, name, func, devices=devices)
        add_function_test(
            TestTileFuncArg, name + "_cpu_blocks", func, devices=get_cpu_test_devices(), enable_cpu_blocks=True
        )

add_function_test(
    TestTileFuncArg,
    "test_register_tile_func_matrix_read_all_lanes",
    functools.partial(test_register_tile_func_matrix_read, all_lanes=True),
    devices=get_cuda_test_devices(),
)

add_function_test(
    TestTileFuncArg, "test_register_tile_func_element_read", test_register_tile_func_element_read, devices=devices
)
add_function_test(
    TestTileFuncArg,
    "test_register_tile_func_element_read_cpu_blocks",
    test_register_tile_func_element_read,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileFuncArg,
    "test_register_tile_func_large_read_grad",
    functools.partial(test_register_tile_func_large_read, backward=True),
    devices=get_cuda_test_devices(),
)
for kind in ("identity", "nested", "local", "conditional", "allocated", "load", "view", "broadcast"):
    add_function_test(
        TestTileFuncArg,
        f"test_tile_func_return_composition_{kind}_grad",
        functools.partial(test_tile_func_return_composition, kind=kind, backward=True),
        devices=devices,
    )
add_function_test(
    TestTileFuncArg,
    "test_tile_func_multiple_return_composition_grad",
    functools.partial(test_tile_func_multiple_return_composition, backward=True),
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
