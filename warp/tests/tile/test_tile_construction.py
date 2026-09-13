# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests tile construction and per-thread tile conversion with 64-thread blocks.

Covers ``tile()``, ``untile()``, ``tile_zeros()``, ``tile_ones()``,
``tile_arange()``, and ``tile_full()``. Keep kernels in this module on the
standard 64-thread block size so it does not acquire extra block-dimension
variants.
"""

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_DIM = 64
TILE_M = wp.constant(8)


@wp.kernel
def test_tile_tile_preserve_type_kernel(x: wp.array[Any], y: wp.array[Any]):
    a = x[0]
    t = wp.tile(a, preserve_type=True)
    wp.tile_store(y, t)


wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array[float], "y": wp.array[float]})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array[wp.vec3], "y": wp.array[wp.vec3]})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array[wp.quat], "y": wp.array[wp.quat]})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array[wp.mat33], "y": wp.array[wp.mat33]})


@wp.kernel
def test_tile_tile_scalar_expansion_kernel(x: wp.array[float], y: wp.array[float]):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


@wp.kernel
def test_tile_tile_vec_expansion_kernel(x: wp.array[wp.vec3], y: wp.array2d[float]):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


@wp.kernel
def test_tile_tile_mat_expansion_kernel(x: wp.array[wp.mat33], y: wp.array3d[float]):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


def test_tile_preserves_and_expands_value_types(test, device):
    """Preserve and expand value types through ``tile()`` operations."""

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
def test_tile_untile_preserve_type_kernel(x: wp.array[Any], y: wp.array[Any]):
    i = wp.tid()
    a = x[i]
    t = wp.tile(a, preserve_type=True)
    b = wp.untile(t)
    y[i] = b


wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array[float], "y": wp.array[float]})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array[wp.vec3], "y": wp.array[wp.vec3]})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array[wp.quat], "y": wp.array[wp.quat]})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array[wp.mat33], "y": wp.array[wp.mat33]})


@wp.kernel
def test_tile_untile_kernel(x: wp.array[Any], y: wp.array[Any]):
    i = wp.tid()
    a = x[i]
    t = wp.tile(a)
    b = wp.untile(t)
    y[i] = b


wp.overload(test_tile_untile_kernel, {"x": wp.array[float], "y": wp.array[float]})
wp.overload(test_tile_untile_kernel, {"x": wp.array[wp.vec3], "y": wp.array[wp.vec3]})
wp.overload(test_tile_untile_kernel, {"x": wp.array[wp.mat33], "y": wp.array[wp.mat33]})


def test_tile_untile(test, device):
    """Preserve values, types, and gradients through ``tile()`` and ``untile()``."""

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
        assert_np_equal(x.grad.numpy(), wp.ones_like(x).numpy())

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
        assert_np_equal(x.grad.numpy(), wp.ones_like(x).numpy())

    test_func(float)
    test_func(wp.vec3)
    test_func(wp.mat33)


@wp.kernel
def tile_untile_scalar_kernel(output: wp.array[int]):
    i = wp.tid()
    t = wp.tile(i) * 2
    s = wp.untile(t)
    output[i] = s


def test_tile_untile_scalar(test, device):
    """Convert scalar thread indices through ``tile()`` and ``untile()`` on an unaligned grid."""

    # use an unaligned grid dimension
    N = TILE_DIM * 4 + 5

    output = wp.zeros(shape=N, dtype=int, requires_grad=True, device=device)

    with wp.Tape():
        wp.launch(tile_untile_scalar_kernel, dim=N, inputs=[output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy(), np.arange(N) * 2)


@wp.kernel
def test_untile_vector_kernel(input: wp.array[wp.vec3], output: wp.array[wp.vec3]):
    i = wp.tid()

    v = input[i] * 0.5

    t = wp.tile(v)
    u = wp.untile(t)

    output[i] = u * 2.0


def test_tile_untile_vector(test, device):
    """Preserve vector values and gradients through ``tile()`` and ``untile()``."""

    input = wp.full(TILE_DIM, wp.vec3(1.0, 2.0, 3.0), requires_grad=True, device=device)
    output = wp.zeros_like(input, device=device)

    with wp.Tape() as tape:
        wp.launch(test_untile_vector_kernel, dim=TILE_DIM, inputs=[input, output], block_dim=TILE_DIM, device=device)

    output.grad = wp.ones_like(output, device=device)
    tape.backward()

    assert_np_equal(output.numpy(), input.numpy())
    assert_np_equal(input.grad.numpy(), np.ones((TILE_DIM, 3)))


@wp.struct
class TestStruct:
    x: wp.float32
    y: wp.vec3


@wp.struct
class TestStructWithArray:
    """Struct with array field for testing tile_zeros with complex types."""

    x: wp.array[wp.float64]


@wp.kernel
def test_tile_construction_kernel(
    out_zeros: wp.array[float],
    out_ones: wp.array[float],
    out_arange: wp.array[float],
    out_full_twos: wp.array[float],
    out_full_vecs: wp.array[wp.vec3],
    out_full_mats: wp.array[wp.mat33],
    out_full_structs_register: wp.array[TestStruct],
    out_full_structs_shared: wp.array[TestStruct],
    out_zeros_struct_with_array: wp.array[TestStructWithArray],
):
    zeros = wp.tile_zeros(TILE_M, dtype=float)
    ones = wp.tile_ones(TILE_M, dtype=float)
    arange = wp.tile_arange(TILE_M, dtype=float)
    full_twos = wp.tile_full(TILE_M, value=2.0, dtype=float)
    full_vecs = wp.tile_full(TILE_M, value=wp.vec3(1.0), dtype=wp.vec3)
    full_mats = wp.tile_full(TILE_M, value=wp.mat33(1.0), dtype=wp.mat33)

    ts = TestStruct()
    ts.x = wp.float32(2.0)
    ts.y = wp.vec3(1.0)
    full_structs_register = wp.tile_full(TILE_M, value=ts, dtype=TestStruct, storage="register")
    full_structs_shared = wp.tile_full(TILE_M, value=ts, dtype=TestStruct, storage="shared")

    zeros_struct_with_array = wp.tile_zeros(TILE_M, dtype=TestStructWithArray)

    wp.tile_store(out_zeros, zeros)
    wp.tile_store(out_ones, ones)
    wp.tile_store(out_arange, arange)
    wp.tile_store(out_full_twos, full_twos)
    wp.tile_store(out_full_vecs, full_vecs)
    wp.tile_store(out_full_mats, full_mats)
    wp.tile_store(out_full_structs_register, full_structs_register)
    wp.tile_store(out_full_structs_shared, full_structs_shared)
    wp.tile_store(out_zeros_struct_with_array, zeros_struct_with_array)


def test_tile_construction(test, device):
    """Construct tiles with initialization helpers and composite types."""

    zeros = wp.empty(TILE_M, dtype=float, device=device)
    ones = wp.empty(TILE_M, dtype=float, device=device)
    arange = wp.empty(TILE_M, dtype=float, device=device)
    full_twos = wp.empty(TILE_M, dtype=float, device=device)
    full_vecs = wp.empty(TILE_M, dtype=wp.vec3, device=device)
    full_mats = wp.empty(TILE_M, dtype=wp.mat33, device=device)
    full_structs_register = wp.empty(TILE_M, dtype=TestStruct, device=device)
    full_structs_shared = wp.empty(TILE_M, dtype=TestStruct, device=device)
    zeros_struct_with_array = wp.empty(TILE_M, dtype=TestStructWithArray, device=device)

    wp.launch_tiled(
        test_tile_construction_kernel,
        dim=1,
        inputs=[],
        outputs=[
            zeros,
            ones,
            arange,
            full_twos,
            full_vecs,
            full_mats,
            full_structs_register,
            full_structs_shared,
            zeros_struct_with_array,
        ],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(zeros.numpy(), np.zeros(TILE_M, dtype=float))
    assert_np_equal(ones.numpy(), np.ones(TILE_M, dtype=float))
    assert_np_equal(full_twos.numpy(), np.full(TILE_M, 2.0, dtype=float))
    assert_np_equal(full_vecs.numpy(), np.ones((TILE_M, 3), dtype=float))
    assert_np_equal(full_mats.numpy(), np.ones((TILE_M, 3, 3), dtype=float))
    assert_np_equal(full_structs_register.numpy()["x"], np.full(TILE_M, 2.0, dtype=float))
    assert_np_equal(full_structs_register.numpy()["y"], np.ones((TILE_M, 3), dtype=float))
    assert_np_equal(full_structs_shared.numpy()["x"], np.full(TILE_M, 2.0, dtype=float))
    assert_np_equal(full_structs_shared.numpy()["y"], np.ones((TILE_M, 3), dtype=float))
    assert_np_equal(arange.numpy(), np.arange(TILE_M, dtype=float))

    # Verify struct with array field is zero-initialized
    # The array field is an array_t with (data, grad, shape, strides, ndim) - all should be zero
    struct_arr_np = zeros_struct_with_array.numpy()
    test.assertTrue(np.all(struct_arr_np["x"]["data"] == 0))
    test.assertTrue(np.all(struct_arr_np["x"]["grad"] == 0))
    test.assertTrue(np.all(struct_arr_np["x"]["ndim"] == 0))


@wp.kernel
def tile_ones_kernel(out: wp.array[float]):
    i = wp.tid()

    t = wp.tile_ones(dtype=float, shape=(16, 16))
    s = wp.tile_sum(t)

    wp.tile_store(out, s)


def test_tile_ones(test, device):
    """Fill a tile with ones and reduce it to the expected sum."""

    output = wp.zeros(1, dtype=float, device=device)

    with wp.Tape():
        wp.launch_tiled(tile_ones_kernel, dim=[1], inputs=[output], block_dim=TILE_DIM, device=device)

    test.assertAlmostEqual(output.numpy()[0], 256.0)


@wp.kernel(enable_backward=False)
def tile_arange_partial_step_int_kernel(out: wp.array2d[int]):
    a = wp.tile_arange(0, 10, 3, dtype=int)
    b = wp.tile_arange(0, 5, 2, dtype=int)
    c = wp.tile_arange(10, 0, -3, dtype=int)
    # exact multiple of the step: the count must not grow
    d = wp.tile_arange(0, 9, 3, dtype=int)

    wp.tile_store(out[0], a)
    wp.tile_store(out[1], b)
    wp.tile_store(out[2], c)
    wp.tile_store(out[3], d)


@wp.kernel(enable_backward=False)
def tile_arange_partial_step_float_kernel(out: wp.array2d[float]):
    a = wp.tile_arange(0.0, 0.3, 0.1, dtype=float)
    b = wp.tile_arange(0.0, 2.0, 0.3, dtype=float)
    # one- and two-argument forms take an implicit step of 1, so a span below 1 is
    # a single element rather than an empty range
    c = wp.tile_arange(0.0, 0.3, dtype=float)
    d = wp.tile_arange(0.3, dtype=float)

    wp.tile_store(out[0], a)
    wp.tile_store(out[1], b)
    wp.tile_store(out[2], c)
    wp.tile_store(out[3], d)


def test_tile_arange_partial_step(test, device):
    """Verify ``tile_arange()`` keeps the final element of a non-multiple range.

    NumPy defines the expected half-open range, so the element count is the ceiling of the
    span divided by the step. This covers integer and floating-point ranges, the one- and
    two-argument forms, and an exact-multiple control.
    """
    int_expected = [np.arange(0, 10, 3), np.arange(0, 5, 2), np.arange(10, 0, -3), np.arange(0, 9, 3)]
    float_expected = [np.arange(0.0, 0.3, 0.1), np.arange(0.0, 2.0, 0.3), np.arange(0.0, 0.3), np.arange(0.3)]

    # Each destination row is longer than its range and pre-filled with a sentinel: a short
    # tile leaves a sentinel where an element belongs, and an over-long tile overwrites one.
    int_sentinel = -1
    float_sentinel = -1.0
    int_width = max(len(expected) for expected in int_expected) + 2
    float_width = max(len(expected) for expected in float_expected) + 2

    int_out = wp.array(np.full((len(int_expected), int_width), int_sentinel, dtype=np.int32), device=device)
    float_out = wp.array(np.full((len(float_expected), float_width), float_sentinel, dtype=np.float32), device=device)

    wp.launch_tiled(tile_arange_partial_step_int_kernel, dim=[1], inputs=[int_out], block_dim=TILE_DIM, device=device)
    wp.launch_tiled(
        tile_arange_partial_step_float_kernel, dim=[1], inputs=[float_out], block_dim=TILE_DIM, device=device
    )

    int_result = int_out.numpy()
    for row, expected in enumerate(int_expected):
        n = len(expected)
        assert_np_equal(int_result[row, :n], expected.astype(np.int32))
        assert_np_equal(int_result[row, n:], np.full(int_width - n, int_sentinel, dtype=np.int32))

    float_result = float_out.numpy()
    for row, expected in enumerate(float_expected):
        n = len(expected)
        np.testing.assert_allclose(float_result[row, :n], expected, rtol=1.0e-6)
        assert_np_equal(float_result[row, n:], np.full(float_width - n, float_sentinel, dtype=np.float32))


@wp.kernel(enable_backward=False)
def tile_arange_exact_integer_length_kernel(out: wp.array[wp.int64], boundary: wp.array[wp.int64]):
    # span 2**53 + 1 with step 2**53: the quotient is not representable as a double
    t = wp.tile_arange(0, 9007199254740993, 9007199254740992, dtype=wp.int64)
    # bounds just past the int32 range, which is the type a bare integer literal is inferred as
    tile_arange_int32_boundary = wp.tile_arange(0, 2147483650, 2147483649, dtype=wp.int64)

    wp.tile_store(out, t)
    wp.tile_store(boundary, tile_arange_int32_boundary)


def test_tile_arange_exact_integer_length(test, device):
    """Verify all-integer ranges preserve exact lengths and values beyond double precision.

    NumPy evaluates the same length in double precision and returns one element. Both ``0``
    and ``2**53`` lie below the stop value of ``2**53 + 1``, so exact integer arithmetic
    produces two exact ``int64`` values.

    The values matter as much as the count. A bare integer literal is inferred as ``int32``, so
    a range that does not fit that type has to reach the native tile_arange() at the requested
    ``dtype`` instead, or the second element wraps to a negative number while the tile still has
    the right length.
    """
    sentinel = -1
    out = wp.array(np.full(4, sentinel, dtype=np.int64), device=device)
    boundary = wp.array(np.full(4, sentinel, dtype=np.int64), device=device)

    wp.launch_tiled(
        tile_arange_exact_integer_length_kernel,
        dim=[1],
        inputs=[out, boundary],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(out.numpy(), np.array([0, 9007199254740992, sentinel, sentinel], dtype=np.int64))
    assert_np_equal(boundary.numpy(), np.array([0, 2147483649, sentinel, sentinel], dtype=np.int64))


TILE_ARANGE_I32_START = wp.constant(wp.int32(0))
TILE_ARANGE_I32_STOP = wp.constant(wp.int32(10))
TILE_ARANGE_I32_STEP = wp.constant(wp.int32(3))
TILE_ARANGE_I64_START = wp.constant(wp.int64(0))
TILE_ARANGE_I64_STOP = wp.constant(wp.int64(9007199254740993))
TILE_ARANGE_I64_STEP = wp.constant(wp.int64(9007199254740992))


@wp.kernel(enable_backward=False)
def tile_arange_integer_constant_kernel(out32: wp.array[wp.int32], out64: wp.array[wp.int64]):
    t32 = wp.tile_arange(
        TILE_ARANGE_I32_START,
        TILE_ARANGE_I32_STOP,
        TILE_ARANGE_I32_STEP,
        dtype=wp.int32,
    )
    t64 = wp.tile_arange(
        TILE_ARANGE_I64_START,
        TILE_ARANGE_I64_STOP,
        TILE_ARANGE_I64_STEP,
        dtype=wp.int64,
    )

    wp.tile_store(out32, t32)
    wp.tile_store(out64, t64)


def test_tile_arange_integer_constants(test, device):
    """Verify ``tile_arange()`` preserves integer values wrapped by ``wp.constant()``.

    A wrapped value must be counted as exactly as a bare literal, so the ``int64`` range keeps
    the element that a double-precision quotient would round away.
    """
    out32 = wp.array(np.full(5, -1, dtype=np.int32), device=device)
    out64 = wp.array(np.full(3, -1, dtype=np.int64), device=device)

    wp.launch_tiled(
        tile_arange_integer_constant_kernel,
        dim=[1],
        inputs=[out32, out64],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(out32.numpy(), np.array([0, 3, 6, 9, -1], dtype=np.int32))
    assert_np_equal(out64.numpy(), np.array([0, 9007199254740992, -1], dtype=np.int64))


@wp.kernel(enable_backward=False)
def tile_arange_local_constant_reuse_kernel(
    values: wp.array[wp.float32],
    loaded: wp.array[wp.float32],
    count: wp.array[wp.int32],
):
    stop = 3
    t = wp.tile_arange(stop, dtype=wp.float32)
    wp.tile_store(values, t)

    # `stop` must survive the call unchanged: still readable as a value, and still usable as a
    # compile-time integer wherever an `int` literal is, such as a tile shape or a slice bound.
    wp.tile_store(loaded, wp.tile_load(loaded, shape=stop))
    vector = wp.vec3f(4.0, 5.0, 6.0)
    sliced = vector[0:stop]
    count[0] = stop
    count[1] = wp.int32(sliced[2])


def test_tile_arange_local_constant_reuse(test, device):
    """Verify a local constant is unchanged by its use as a ``tile_arange()`` bound.

    The dispatch casts the bound to the output element type. That must not rewrite the caller's
    own constant, which stays readable as a value and usable as a compile-time integer in a
    later tile shape or slice bound.
    """
    values = wp.empty(3, dtype=wp.float32, device=device)
    loaded = wp.array(np.array([7.0, 8.0, 9.0], dtype=np.float32), device=device)
    count = wp.empty(2, dtype=wp.int32, device=device)

    wp.launch_tiled(
        tile_arange_local_constant_reuse_kernel,
        dim=[1],
        outputs=[values, loaded, count],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(values.numpy(), np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert_np_equal(loaded.numpy(), np.array([7.0, 8.0, 9.0], dtype=np.float32))
    assert_np_equal(count.numpy(), np.array([3, 6], dtype=np.int32))


def test_tile_arange_zero_step_rejected(test, device):
    """Reject a zero step with a clear ``tile_arange()`` error.

    A zero step has no element count to compute and must be reported against the call rather
    than escaping as a bare ``ZeroDivisionError`` from code generation.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_arange(0, 10, 0, dtype=int)

    with test.assertRaisesRegex((RuntimeError, ValueError), r"tile_arange.*step cannot be zero"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=TILE_DIM, device=device)


def test_tile_arange_empty_range_rejected(test, device):
    """Reject empty and inverted ranges before native compilation.

    A range spanning no elements would need a zero- or negative-length tile dimension, which
    has no native tile type. Report it against the call rather than letting it reach a backend
    that fails on CUDA and silently accepts a zero length on CPU.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def empty_kernel():
        t = wp.tile_arange(5, 5, dtype=int)

    @wp.kernel(module="unique", enable_backward=False)
    def inverted_kernel():
        t = wp.tile_arange(0, -5, dtype=int)

    @wp.kernel(module="unique", enable_backward=False)
    def negative_step_kernel():
        t = wp.tile_arange(0, 10, -1, dtype=int)

    for kernel_fn in (empty_kernel, inverted_kernel, negative_step_kernel):
        with test.assertRaisesRegex((RuntimeError, ValueError), r"non-empty range|zero-length"):
            wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=TILE_DIM, device=device)


def test_tile_arange_argument_count_rejected(test, device):
    """Reject more than three positional arguments before interpreting the range.

    The extra argument is the caller's mistake, so it has to be reported even when the three
    arguments the value function would otherwise use are themselves an invalid range. Checking
    the arity only in the dispatch reports the range instead, which is not what went wrong.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        values = wp.tile_arange(5, 5, 1, 99, dtype=int)

    with test.assertRaisesRegex(
        (RuntimeError, TypeError),
        r"tile_arange\(\) accepts at most 3 positional arguments, got 4",
    ):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=TILE_DIM, device=device)


def test_tile_arange_non_constant_rejected(test, device):
    """Reject a range argument that is only known at runtime.

    The tile shape is resolved during code generation, so a kernel parameter cannot bound the
    range. It must be named as such rather than reaching the element-count arithmetic and
    surfacing as an internal type error.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn(stop: int):
        t = wp.tile_arange(0, stop, 1, dtype=int)

    with test.assertRaisesRegex(
        (RuntimeError, TypeError),
        r"tile_arange\(\) arguments must be compile time constants, but stop is not",
    ):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[10], block_dim=TILE_DIM, device=device)


TILE_ARANGE_F32_START = wp.constant(wp.float32(0.0))
TILE_ARANGE_F32_STOP = wp.constant(wp.float32(3.0000001))
TILE_ARANGE_F32_STEP = wp.constant(wp.float32(0.99999999))
TILE_ARANGE_F32_UNDERFLOW = wp.constant(wp.float32(1.0e-46))
TILE_ARANGE_F32_MIN = wp.constant(wp.float32(1.401298464324817e-45))
TILE_ARANGE_I64_TO_F32_START = wp.constant(wp.int64(2**54))
TILE_ARANGE_I64_TO_F32_STOP = wp.constant(wp.int64(2**54 + 3 * 2**30 - 1))
TILE_ARANGE_I64_TO_F32_STEP = wp.constant(wp.int64(2**31))


@wp.kernel(enable_backward=False)
def tile_arange_float32_quantized_kernel(
    literal_out: wp.array[wp.float32],
    constant_out: wp.array[wp.float32],
):
    literal_values = wp.tile_arange(0.0, 3.0000001, 0.99999999, dtype=wp.float32)
    constant_values = wp.tile_arange(
        TILE_ARANGE_F32_START,
        TILE_ARANGE_F32_STOP,
        TILE_ARANGE_F32_STEP,
        dtype=wp.float32,
    )

    wp.tile_store(literal_out, literal_values)
    wp.tile_store(constant_out, constant_values)


def test_tile_arange_float32_quantized(test, device):
    """Size and fill a float32 range from the values float32 can hold.

    Rounded to float32 the range is exactly ``[0, 3)`` with a step of ``1``, so it holds three
    elements. Counting in double precision instead would see a stop just above ``3`` and a step
    just below ``1``, and would add a fourth element the native fill puts at or past the stop.
    A literal and a ``wp.constant()`` bound must agree.
    """
    literal_out = wp.full(4, -1.0, dtype=wp.float32, device=device)
    constant_out = wp.full(4, -1.0, dtype=wp.float32, device=device)

    wp.launch_tiled(
        tile_arange_float32_quantized_kernel,
        dim=[1],
        inputs=[literal_out, constant_out],
        block_dim=TILE_DIM,
        device=device,
    )

    expected = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
    assert_np_equal(literal_out.numpy(), expected)
    assert_np_equal(constant_out.numpy(), expected)


@wp.kernel(enable_backward=False)
def tile_arange_float32_substep_range_kernel(out: wp.array[wp.float32]):
    values = wp.tile_arange(0.0, TILE_ARANGE_F32_MIN, 2.0, dtype=wp.float32)
    wp.tile_store(out, values)


def test_tile_arange_float32_substep_range(test, device):
    """Keep the start of a nonempty range when its quotient underflows."""
    out = wp.full(1, -1.0, dtype=wp.float32, device=device)

    wp.launch_tiled(tile_arange_float32_substep_range_kernel, dim=[1], outputs=[out], block_dim=TILE_DIM, device=device)

    assert_np_equal(out.numpy(), np.array([0.0], dtype=np.float32))


@wp.kernel(enable_backward=False)
def tile_arange_int64_to_float32_kernel(out: wp.array[wp.float32]):
    values = wp.tile_arange(
        TILE_ARANGE_I64_TO_F32_START,
        TILE_ARANGE_I64_TO_F32_STOP,
        TILE_ARANGE_I64_TO_F32_STEP,
        dtype=wp.float32,
    )
    wp.tile_store(out, values)


def test_tile_arange_int64_to_float32(test, device):
    """Round integer bounds directly to the floating-point output type."""
    out = wp.full(3, -1.0, dtype=wp.float32, device=device)

    wp.launch_tiled(tile_arange_int64_to_float32_kernel, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)

    assert_np_equal(out.numpy(), np.array([float(2**54), -1.0, -1.0], dtype=np.float32))


@wp.kernel(enable_backward=False)
def tile_arange_float32_finite_span_kernel(out: wp.array[wp.float32]):
    values = wp.tile_arange(-3.0e38, 3.0e38, 3.0e38, dtype=wp.float32)
    wp.tile_store(out, values)


def test_tile_arange_float32_finite_span(test, device):
    """Count a float32 range spanning the whole representable interval.

    The endpoints sit at opposite ends of the float32 range, so the span is the widest one an
    element count is ever derived from.
    """
    out = wp.full(3, -1.0, dtype=wp.float32, device=device)

    wp.launch_tiled(tile_arange_float32_finite_span_kernel, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)

    assert_np_equal(out.numpy(), np.array([-3.0e38, 0.0, -1.0], dtype=np.float32))


# A bound past the widest integer type, paired with a step large enough to keep the element count
# storable. float32 represents both exactly, so the range itself is perfectly valid.
TILE_ARANGE_WIDE_STOP = wp.constant(wp.float32(2.0**70))
TILE_ARANGE_WIDE_STEP = wp.constant(wp.float32(2.0**64))


@wp.kernel(enable_backward=False)
def tile_arange_wide_float_bounds_kernel(out: wp.array[wp.float32]):
    values = wp.tile_arange(0.0, TILE_ARANGE_WIDE_STOP, TILE_ARANGE_WIDE_STEP, dtype=wp.float32)

    wp.tile_store(out, values)


def test_tile_arange_wide_integer_bound_rejected(test, device):
    """Reject an integer bound too wide to carry, and point at a spelling that works.

    Rounding a bound exactly needs an integer type at least as wide as the value, and code
    generation could not emit a literal that wide either, so a bound past ``uint64`` has to be
    refused. What runs out is the integer, not the output type: float32 holds ``2**70`` exactly.
    The error therefore must not blame the output type, and the floating-point spelling it
    recommends has to build the very range the rejected call asked for.

    ``2**70`` over a step of ``2**64`` is the smallest pair that is both past the integer limit
    and small enough to store. A bound just past ``uint64`` with a unit step would be equally
    rejected, but describes about 1.8e19 elements, so it could not be checked against a result.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def rejected_kernel(out: wp.array[wp.float32]):
        # 2**70 and 2**64, which no Warp integer type can hold
        values = wp.tile_arange(0, 1180591620717411303424, 18446744073709551616, dtype=wp.float32)
        wp.tile_store(out, values)

    expected = np.array([k * float(2**64) for k in range(64)], dtype=np.float32)
    out = wp.full(len(expected), -1.0, dtype=wp.float32, device=device)

    with test.assertRaisesRegex(
        (RuntimeError, ValueError),
        r"tile_arange\(\) stop=1180591620717411303424 is too wide to use as an integer bound",
    ):
        wp.launch_tiled(rejected_kernel, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)

    # The spelling the error recommends builds the range the rejected call described.
    wp.launch_tiled(tile_arange_wide_float_bounds_kernel, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)

    assert_np_equal(out.numpy(), expected)


def test_tile_arange_non_numeric_dtype_rejected(test, device):
    """Reject an output type that no linear range can describe.

    A bool tile would silently hold the truth of each element rather than the element itself, and
    a vector type has no native tile_arange() at all, so both must be reported against the call.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def bool_kernel(out: wp.array[wp.bool]):
        t = wp.tile_arange(0, 6, dtype=wp.bool)
        wp.tile_store(out, t)

    @wp.kernel(module="unique", enable_backward=False)
    def vector_kernel(out: wp.array[wp.vec3f]):
        t = wp.tile_arange(0, 4, dtype=wp.vec3f)
        wp.tile_store(out, t)

    for kernel_fn, dtype in ((bool_kernel, wp.bool), (vector_kernel, wp.vec3f)):
        out = wp.empty(6, dtype=dtype, device=device)
        with test.assertRaisesRegex((RuntimeError, TypeError), r"tile_arange\(\) requires a numeric scalar dtype"):
            wp.launch_tiled(kernel_fn, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)


@wp.kernel(enable_backward=False)
def tile_arange_integral_float_bounds_kernel(out: wp.array[wp.int32]):
    # integral float bounds carry an exact integer value, so an integer output type keeps them
    t = wp.tile_arange(0.0, 10.0, dtype=wp.int32)

    wp.tile_store(out, t)


def test_tile_arange_integral_float_bounds(test, device):
    """Accept float bounds that hold an exact integer with an integer output type."""
    out = wp.full(12, -1, dtype=wp.int32, device=device)

    wp.launch_tiled(tile_arange_integral_float_bounds_kernel, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)

    assert_np_equal(out.numpy(), np.array([*range(10), -1, -1], dtype=np.int32))


def test_tile_arange_float32_underflow_step_rejected(test, device):
    """Reject a float32 step that underflows to zero."""

    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        values = wp.tile_arange(
            TILE_ARANGE_F32_START,
            TILE_ARANGE_F32_UNDERFLOW,
            TILE_ARANGE_F32_UNDERFLOW,
            dtype=wp.float32,
        )

    with test.assertRaisesRegex((RuntimeError, ValueError), r"tile_arange.*step cannot be zero"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=TILE_DIM, device=device)


def test_tile_arange_unrepresentable_rejected(test, device):
    """Reject range bounds the output element type cannot represent.

    ``ctypes`` wraps an out-of-range integer and truncates a fractional one, so an unchecked
    bound would silently describe a range the caller never asked for.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def out_of_range_kernel():
        t = wp.tile_arange(0, 1099511627776, dtype=wp.int32)  # 2**40

    @wp.kernel(module="unique", enable_backward=False)
    def fractional_kernel():
        t = wp.tile_arange(0.0, 3.5, dtype=wp.int32)

    @wp.kernel(module="unique", enable_backward=False)
    def overflowing_float_kernel():
        t = wp.tile_arange(0.0, 1.0e300, dtype=wp.float32)

    with test.assertRaisesRegex((RuntimeError, ValueError), r"stop=1099511627776 is out of range"):
        wp.launch_tiled(out_of_range_kernel, dim=[1], inputs=[], block_dim=TILE_DIM, device=device)

    with test.assertRaisesRegex((RuntimeError, ValueError), r"stop=3.5 has a fractional part"):
        wp.launch_tiled(fractional_kernel, dim=[1], inputs=[], block_dim=TILE_DIM, device=device)

    with test.assertRaisesRegex((RuntimeError, ValueError), r"stop=1e\+300 is out of range"):
        wp.launch_tiled(overflowing_float_kernel, dim=[1], inputs=[], block_dim=TILE_DIM, device=device)


def test_tile_arange_uncountable_range_rejected(test, device):
    """Reject a range whose element count overflows the quotient.

    A subnormal step is not zero, so it clears the zero-step guard, but dividing by it overflows
    to infinity. That has to be reported against the call rather than escaping as an
    ``OverflowError`` from the count.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_arange(0.0, 1.0, 5.0e-324, dtype=wp.float64)

    with test.assertRaisesRegex((RuntimeError, ValueError), r"spans more elements than can be counted"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=TILE_DIM, device=device)


@wp.kernel
def tile_arange_kernel(out: wp.array2d[int]):
    i = wp.tid()

    a = wp.tile_arange(17, dtype=int)
    b = wp.tile_arange(5, 22, dtype=int)
    c = wp.tile_arange(0, 34, 2, dtype=int)
    d = wp.tile_arange(-1, 16, dtype=int)
    e = wp.tile_arange(17, 0, -1, dtype=int)

    wp.tile_store(out[0], a)
    wp.tile_store(out[1], b)
    wp.tile_store(out[2], c)
    wp.tile_store(out[3], d)
    wp.tile_store(out[4], e)


def test_tile_arange_start_stop_step_forms(test, device):
    """Construct integer tile ranges with supported start, stop, and step forms."""

    N = 17

    output = wp.zeros(shape=(5, N), dtype=int, device=device)

    with wp.Tape():
        wp.launch_tiled(tile_arange_kernel, dim=[1], inputs=[output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy()[0], np.arange(17))
    assert_np_equal(output.numpy()[1], np.arange(5, 22))
    assert_np_equal(output.numpy()[2], np.arange(0, 34, 2))
    assert_np_equal(output.numpy()[3], np.arange(-1, 16))
    assert_np_equal(output.numpy()[4], np.arange(17, 0, -1))


class TestTileConstruction(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(
    TestTileConstruction,
    "test_tile_preserves_and_expands_value_types",
    test_tile_preserves_and_expands_value_types,
    devices=get_cuda_test_devices(),
)
add_function_test(TestTileConstruction, "test_tile_untile", test_tile_untile, devices=devices)
add_function_test(TestTileConstruction, "test_tile_untile_scalar", test_tile_untile_scalar, devices=devices)
add_function_test(TestTileConstruction, "test_tile_untile_vector", test_tile_untile_vector, devices=devices)
add_function_test(TestTileConstruction, "test_tile_construction", test_tile_construction, devices=devices)
add_function_test(TestTileConstruction, "test_tile_ones", test_tile_ones, devices=devices)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_start_stop_step_forms",
    test_tile_arange_start_stop_step_forms,
    devices=devices,
)
add_function_test(TestTileConstruction, "test_tile_arange_partial_step", test_tile_arange_partial_step, devices=devices)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_exact_integer_length",
    test_tile_arange_exact_integer_length,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_integer_constants",
    test_tile_arange_integer_constants,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_local_constant_reuse",
    test_tile_arange_local_constant_reuse,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_zero_step_rejected",
    test_tile_arange_zero_step_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_empty_range_rejected",
    test_tile_arange_empty_range_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_argument_count_rejected",
    test_tile_arange_argument_count_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_non_constant_rejected",
    test_tile_arange_non_constant_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_float32_quantized",
    test_tile_arange_float32_quantized,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_float32_substep_range",
    test_tile_arange_float32_substep_range,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_int64_to_float32",
    test_tile_arange_int64_to_float32,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_float32_finite_span",
    test_tile_arange_float32_finite_span,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_integral_float_bounds",
    test_tile_arange_integral_float_bounds,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_wide_integer_bound_rejected",
    test_tile_arange_wide_integer_bound_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_non_numeric_dtype_rejected",
    test_tile_arange_non_numeric_dtype_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_float32_underflow_step_rejected",
    test_tile_arange_float32_underflow_step_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_unrepresentable_rejected",
    test_tile_arange_unrepresentable_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_uncountable_range_rejected",
    test_tile_arange_uncountable_range_rejected,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_untile_cpu_blocks",
    test_tile_untile,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileConstruction,
    "test_tile_construction_cpu_blocks",
    test_tile_construction,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
