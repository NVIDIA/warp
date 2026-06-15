# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

kernel_cache = {}


def getkernel(func, suffix=""):
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def test_infinity_scalar(test, device, dtype, register_kernels=False):
    def check_infinity(outputs: wp.array[dtype], bool_outputs: wp.array[wp.bool]):
        outputs[0] = dtype(wp.inf)
        outputs[1] = dtype(-wp.inf)
        outputs[2] = dtype(2.0 * wp.inf)
        outputs[3] = dtype(-2.0 * wp.inf)
        outputs[4] = dtype(2.0 / 0.0)
        outputs[5] = dtype(-2.0 / 0.0)
        outputs[6] = wp.log(dtype(0))
        outputs[7] = wp.exp(dtype(800))

        # Fill out bool outputs
        bool_outputs[0] = wp.isinf(dtype(wp.inf))
        bool_outputs[1] = wp.isfinite(dtype(wp.inf))
        bool_outputs[2] = wp.isinf(dtype(-wp.inf))
        bool_outputs[3] = wp.isfinite(dtype(-wp.inf))
        bool_outputs[4] = wp.isinf(dtype(0))
        bool_outputs[5] = wp.isinf(wp.exp(dtype(800)))

    kernel = getkernel(check_infinity, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs = wp.empty(8, dtype=dtype, device=device)
    outputs_bool = wp.empty(6, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[], outputs=[outputs, outputs_bool], device=device)

    outputs_cpu = outputs.to("cpu").list()

    test.assertEqual(outputs_cpu[0], math.inf)
    test.assertEqual(outputs_cpu[1], -math.inf)
    test.assertEqual(outputs_cpu[2], math.inf)
    test.assertEqual(outputs_cpu[3], -math.inf)
    test.assertEqual(outputs_cpu[4], math.inf)
    test.assertEqual(outputs_cpu[5], -math.inf)
    test.assertEqual(outputs_cpu[6], -math.inf)
    test.assertEqual(outputs_cpu[7], math.inf)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isinf(wp.inf) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isfinite(wp.inf) is not False")
    test.assertTrue(outputs_bool_cpu[2], "wp.isinf(-wp.inf) is not True")
    test.assertFalse(outputs_bool_cpu[3], "wp.isfinite(-wp.inf) is not False")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(0) is not False")
    test.assertTrue(outputs_bool_cpu[5], "wp.isinf(wp.exp(800)) is not True")


def test_nan_scalar(test, device, dtype, register_kernels=False):
    def check_nan(outputs: wp.array[dtype], bool_outputs: wp.array[wp.bool]):
        outputs[0] = dtype(wp.nan)
        outputs[1] = dtype(-wp.nan)
        outputs[2] = dtype(2.0 * wp.nan)
        outputs[3] = dtype(2.0 + wp.nan)
        outputs[4] = dtype(0.0 / 0.0)
        outputs[5] = wp.sqrt(dtype(-1))
        outputs[6] = wp.log(dtype(-1))
        outputs[7] = dtype(wp.inf) - dtype(wp.inf)

        # Fill out bool outputs
        bool_outputs[0] = dtype(wp.nan) == dtype(wp.nan)
        bool_outputs[1] = dtype(wp.nan) != dtype(wp.nan)
        bool_outputs[2] = dtype(wp.nan) == dtype(1)
        bool_outputs[3] = dtype(wp.nan) != dtype(1)
        bool_outputs[4] = wp.isnan(wp.nan)
        bool_outputs[5] = wp.isnan(dtype(0.0))
        bool_outputs[6] = wp.isnan(dtype(wp.inf))
        bool_outputs[7] = dtype(wp.nan) > dtype(1)
        bool_outputs[8] = dtype(wp.nan) >= dtype(1)
        bool_outputs[9] = dtype(wp.nan) < dtype(1)
        bool_outputs[10] = dtype(wp.nan) <= dtype(1)
        bool_outputs[11] = dtype(wp.nan) > dtype(wp.nan)
        bool_outputs[12] = dtype(wp.nan) >= dtype(wp.nan)
        bool_outputs[13] = dtype(wp.nan) < dtype(wp.nan)
        bool_outputs[14] = dtype(wp.nan) <= dtype(wp.nan)
        bool_outputs[15] = wp.isfinite(dtype(wp.nan))
        bool_outputs[16] = wp.isinf(dtype(wp.nan))

    kernel = getkernel(check_nan, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs = wp.empty(8, dtype=dtype, device=device)
    outputs_bool = wp.empty(17, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[], outputs=[outputs, outputs_bool], device=device)

    outputs_cpu = outputs.to("cpu").list()

    test.assertTrue(math.isnan(outputs_cpu[0]), "wp.nan is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[1]), "-wp.nan is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[2]), "2.0*wp.nan is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[3]), "2.0+wp.nan is not NaN ")
    test.assertTrue(math.isnan(outputs_cpu[4]), "0.0/0.0 is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[5]), "Sqrt of a negative number is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[6]), "Log of a negative number is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[7]), "Subtracting infinity from infinity is not NaN")

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertFalse(outputs_bool_cpu[0], "wp.nan == wp.nan is not False")
    test.assertTrue(outputs_bool_cpu[1], "wp.nan != wp.nan is not True")
    test.assertFalse(outputs_bool_cpu[2], "wp.nan == 1 is not False")
    test.assertTrue(outputs_bool_cpu[3], "wp.nan != 1 is not True")
    test.assertTrue(outputs_bool_cpu[4], "isnan(wp.nan) is not True")
    test.assertFalse(outputs_bool_cpu[5], "isnan(0.0) is not False")
    test.assertFalse(outputs_bool_cpu[6], "isnan(wp.inf) is not False")
    test.assertFalse(outputs_bool_cpu[7], "wp.nan > 1 is not False")
    test.assertFalse(outputs_bool_cpu[8], "wp.nan >= 1 is not False")
    test.assertFalse(outputs_bool_cpu[9], "wp.nan < 1 is not False")
    test.assertFalse(outputs_bool_cpu[10], "wp.nan <= 1 is not False")
    test.assertFalse(outputs_bool_cpu[11], "wp.nan > wp.nan is not False")
    test.assertFalse(outputs_bool_cpu[12], "wp.nan >= wp.nan is not False")
    test.assertFalse(outputs_bool_cpu[13], "wp.nan < wp.nan is not False")
    test.assertFalse(outputs_bool_cpu[14], "wp.nan <= wp.nan is not False")
    test.assertFalse(outputs_bool_cpu[15], "wp.isfinite(wp.nan) is not False")
    test.assertFalse(outputs_bool_cpu[16], "wp.isinf(wp.nan) is not False")


def test_is_special_vec(test, device, dtype, register_kernels=False):
    vector_type = wp.types.vector(5, dtype)

    def check_special_vec(bool_outputs: wp.array[wp.bool]):
        zeros_vector = vector_type()
        bool_outputs[0] = wp.isfinite(zeros_vector)
        bool_outputs[1] = wp.isinf(zeros_vector)
        bool_outputs[2] = wp.isnan(zeros_vector)

        nan_vector = vector_type()
        nan_vector[0] = dtype(wp.NAN)
        bool_outputs[3] = wp.isfinite(nan_vector)
        bool_outputs[4] = wp.isinf(nan_vector)
        bool_outputs[5] = wp.isnan(nan_vector)

        inf_vector = vector_type()
        inf_vector[0] = dtype(wp.inf)
        bool_outputs[6] = wp.isfinite(inf_vector)
        bool_outputs[7] = wp.isinf(inf_vector)
        bool_outputs[8] = wp.isnan(inf_vector)

    kernel = getkernel(check_special_vec, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs_bool = wp.empty(9, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[outputs_bool], device=device)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isfinite(zeros_vector) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isinf(zeros_vector) is not False")
    test.assertFalse(outputs_bool_cpu[2], "wp.isnan(zeros_vector) is not False")

    test.assertFalse(outputs_bool_cpu[3], "wp.isfinite(nan_vector) is not False")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(nan_vector) is not False")
    test.assertTrue(outputs_bool_cpu[5], "wp.isnan(nan_vector) is not True")

    test.assertFalse(outputs_bool_cpu[6], "wp.isfinite(inf_vector) is not False")
    test.assertTrue(outputs_bool_cpu[7], "wp.isinf(inf_vector) is not True")
    test.assertFalse(outputs_bool_cpu[8], "wp.isnan(inf_vector) is not False")


def test_is_special_mat(test, device, dtype, register_kernels=False):
    mat_type = wp.types.matrix((5, 5), dtype)

    def check_special_mat(bool_outputs: wp.array[wp.bool]):
        zeros_mat = mat_type()
        bool_outputs[0] = wp.isfinite(zeros_mat)
        bool_outputs[1] = wp.isinf(zeros_mat)
        bool_outputs[2] = wp.isnan(zeros_mat)

        nan_mat = mat_type()
        nan_mat[0, 0] = dtype(wp.NAN)
        bool_outputs[3] = wp.isfinite(nan_mat)
        bool_outputs[4] = wp.isinf(nan_mat)
        bool_outputs[5] = wp.isnan(nan_mat)

        inf_mat = mat_type()
        inf_mat[0, 0] = dtype(wp.inf)
        bool_outputs[6] = wp.isfinite(inf_mat)
        bool_outputs[7] = wp.isinf(inf_mat)
        bool_outputs[8] = wp.isnan(inf_mat)

    kernel = getkernel(check_special_mat, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs_bool = wp.empty(9, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[outputs_bool], device=device)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isfinite(zeros_mat) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isinf(zeros_mat) is not False")
    test.assertFalse(outputs_bool_cpu[2], "wp.isnan(zeros_mat) is not False")

    test.assertFalse(outputs_bool_cpu[3], "wp.isfinite(nan_mat) is not False")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(nan_mat) is not False")
    test.assertTrue(outputs_bool_cpu[5], "wp.isnan(nan_mat) is not True")

    test.assertFalse(outputs_bool_cpu[6], "wp.isfinite(inf_mat) is not False")
    test.assertTrue(outputs_bool_cpu[7], "wp.isinf(inf_mat) is not True")
    test.assertFalse(outputs_bool_cpu[8], "wp.isnan(inf_mat) is not False")


def assert_float_eq(test, actual, expected, msg):
    """Compare two floats, treating NaN as equal to NaN and distinguishing signed zeros."""
    actual_f = float(actual)
    expected_f = float(expected)
    test.assertEqual(
        math.isnan(actual_f),
        math.isnan(expected_f),
        f"{msg}: NaN mismatch (actual={actual_f}, expected={expected_f})",
    )
    if not math.isnan(expected_f):
        test.assertEqual(actual_f, expected_f, f"{msg}: value mismatch (actual={actual_f}, expected={expected_f})")
        if expected_f == 0.0:
            test.assertEqual(
                math.copysign(1.0, actual_f),
                math.copysign(1.0, expected_f),
                f"{msg}: signed-zero mismatch (actual={actual_f}, expected={expected_f})",
            )


def test_minmax_scalar(test, device, dtype, register_kernels=False):
    # wp.min / wp.max / wp.clamp follow C fmin/fmax semantics on float types
    # (NaN-as-missing, symmetric). NumPy's np.fmin / np.fmax match this
    # exactly for the NaN cases.

    def check_minmax(
        a: wp.array[dtype],
        b: wp.array[dtype],
        mn: wp.array[dtype],
        mx: wp.array[dtype],
        cmn: wp.array[dtype],
        cmx: wp.array[dtype],
    ):
        i = wp.tid()
        mn[i] = wp.min(a[i], b[i])
        mx[i] = wp.max(a[i], b[i])
        # clamp uses min/max internally and inherits the same NaN semantics.
        cmn[i] = wp.clamp(a[i], dtype(-1.0), dtype(1.0))
        cmx[i] = wp.clamp(b[i], dtype(-1.0), dtype(1.0))

    kernel = getkernel(check_minmax, suffix="minmax_" + dtype.__name__)

    if register_kernels:
        return

    nan = float("nan")
    # NaN-only probes -- C99 leaves signed-zero ties implementation-defined,
    # so don't include ±0 cases here.
    inputs_a = [-1.0, nan, nan, -1.0, nan, 2.0]
    inputs_b = [nan, -1.0, nan, 2.0, 2.0, nan]
    # np.fmin / np.fmax are exact NumPy oracles for C fmin/fmax NaN handling.
    expected_min = [float(np.fmin(np.float64(a), np.float64(b))) for a, b in zip(inputs_a, inputs_b, strict=True)]
    expected_max = [float(np.fmax(np.float64(a), np.float64(b))) for a, b in zip(inputs_a, inputs_b, strict=True)]
    # clamp(x, -1, 1) with NaN-as-missing semantics resolves NaN inputs to a
    # finite bound; finite inputs clamp normally.
    expected_clamp_a = [float(np.fmin(np.fmax(np.float64(a), -1.0), 1.0)) for a in inputs_a]
    expected_clamp_b = [float(np.fmin(np.fmax(np.float64(b), -1.0), 1.0)) for b in inputs_b]

    n = len(inputs_a)
    # The module hash caches the resolved options; flipping a global flag
    # mid-session requires explicit invalidation so the new flag value is
    # picked up at the next launch.
    a = wp.array(inputs_a, dtype=dtype, device=device)
    b = wp.array(inputs_b, dtype=dtype, device=device)
    mn = wp.empty(n, dtype=dtype, device=device)
    mx = wp.empty(n, dtype=dtype, device=device)
    cmn = wp.empty(n, dtype=dtype, device=device)
    cmx = wp.empty(n, dtype=dtype, device=device)
    wp.launch(kernel, dim=n, inputs=[a, b], outputs=[mn, mx, cmn, cmx], device=device)
    actual_min = mn.to("cpu").list()
    actual_max = mx.to("cpu").list()
    actual_clamp_a = cmn.to("cpu").list()
    actual_clamp_b = cmx.to("cpu").list()
    for i in range(n):
        assert_float_eq(test, actual_min[i], expected_min[i], f"wp.min({inputs_a[i]}, {inputs_b[i]})")
        assert_float_eq(test, actual_max[i], expected_max[i], f"wp.max({inputs_a[i]}, {inputs_b[i]})")
        assert_float_eq(test, actual_clamp_a[i], expected_clamp_a[i], f"wp.clamp({inputs_a[i]}, -1, 1)")
        assert_float_eq(test, actual_clamp_b[i], expected_clamp_b[i], f"wp.clamp({inputs_b[i]}, -1, 1)")


def test_minmax_vec(test, device, dtype, register_kernels=False):
    # Element-wise vec wp.min / wp.max applies fmin/fmax per component.
    # Vector reduction folds with fmin/fmax so any non-NaN value wins
    # regardless of position.
    vec3_t = wp.types.vector(3, dtype)

    def check_vec_elementwise(
        a: wp.array[vec3_t],
        b: wp.array[vec3_t],
        mn: wp.array[vec3_t],
        mx: wp.array[vec3_t],
    ):
        i = wp.tid()
        mn[i] = wp.min(a[i], b[i])
        mx[i] = wp.max(a[i], b[i])

    def check_vec_reduce(
        a: wp.array[vec3_t],
        red_mn: wp.array[dtype],
        red_mx: wp.array[dtype],
    ):
        i = wp.tid()
        red_mn[i] = wp.min(a[i])
        red_mx[i] = wp.max(a[i])

    kernel_elem = getkernel(check_vec_elementwise, suffix="standard_" + dtype.__name__)
    kernel_red = getkernel(check_vec_reduce, suffix="standard_" + dtype.__name__)

    if register_kernels:
        return

    nan = float("nan")

    # Element-wise inputs: focus on NaN behavior. C99 leaves signed-zero ties
    # implementation-defined for fmin/fmax, so don't include ±0 cases.
    elem_a = [
        [-1.0, nan, nan],
        [-1.0, 2.0, 3.0],
    ]
    elem_b = [
        [nan, -1.0, nan],
        [2.0, -1.0, 1.0],
    ]
    elem_a_np = np.array(elem_a, dtype=np.float64)
    elem_b_np = np.array(elem_b, dtype=np.float64)
    expected_elem_mn = np.fmin(elem_a_np, elem_b_np)
    expected_elem_mx = np.fmax(elem_a_np, elem_b_np)

    # Reduction inputs: NaN never wins unless every element is NaN.
    red_a = [
        [nan, -1.0, 2.0],  # NaN first -> -1 (min) / 2 (max) (NaN ignored)
        [-1.0, nan, 2.0],
        [-1.0, 2.0, nan],
        [-1.0, 2.0, 0.5],
    ]
    expected_red_mn = [-1.0, -1.0, -1.0, -1.0]
    expected_red_mx = [2.0, 2.0, 2.0, 2.0]

    n_elem = len(elem_a)
    a_arr = wp.array(elem_a, dtype=vec3_t, device=device)
    b_arr = wp.array(elem_b, dtype=vec3_t, device=device)
    mn = wp.empty(n_elem, dtype=vec3_t, device=device)
    mx = wp.empty(n_elem, dtype=vec3_t, device=device)
    wp.launch(kernel_elem, dim=n_elem, inputs=[a_arr, b_arr], outputs=[mn, mx], device=device)
    actual_mn = mn.numpy()
    actual_mx = mx.numpy()

    n_red = len(red_a)
    red_arr = wp.array(red_a, dtype=vec3_t, device=device)
    red_mn = wp.empty(n_red, dtype=dtype, device=device)
    red_mx = wp.empty(n_red, dtype=dtype, device=device)
    wp.launch(kernel_red, dim=n_red, inputs=[red_arr], outputs=[red_mn, red_mx], device=device)
    actual_red_mn = red_mn.to("cpu").list()
    actual_red_mx = red_mx.to("cpu").list()
    for i in range(n_elem):
        for j in range(3):
            assert_float_eq(test, actual_mn[i][j], float(expected_elem_mn[i][j]), f"wp.min(vec3, vec3)[{i}][{j}]")
            assert_float_eq(test, actual_mx[i][j], float(expected_elem_mx[i][j]), f"wp.max(vec3, vec3)[{i}][{j}]")
    for i in range(n_red):
        assert_float_eq(test, actual_red_mn[i], expected_red_mn[i], f"wp.min(vec3)[row {i}]")
        assert_float_eq(test, actual_red_mx[i], expected_red_mx[i], f"wp.max(vec3)[row {i}]")


def test_clamp_adjoint(test, device, dtype, register_kernels=False):
    # adj_clamp follows the chain rule of min(max(a, x), b): when an input
    # is NaN, the gradient routes to whichever of x/a/b the forward output
    # depended on (rather than being silently dropped). E.g. clamp(NaN, a, b)
    # returns min(a, b), so the gradient flows to that surviving bound.

    def kern(
        x: wp.array[dtype],
        a: wp.array[dtype],
        b: wp.array[dtype],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        out[i] = wp.clamp(x[i], a[i], b[i])

    kernel = getkernel(kern, suffix="clamp_adj_" + dtype.__name__)

    if register_kernels:
        return

    nan = float("nan")
    # (x, a, b, expected_grad_x, expected_grad_a, expected_grad_b).
    rows = [
        (nan, -1.0, 1.0, 0.0, 1.0, 0.0),  # forward = min(-1, 1) = -1; depends on a
        (nan, 1.0, -1.0, 0.0, 0.0, 1.0),  # forward = min(1, -1) = -1; depends on b
        (0.5, nan, 1.0, 1.0, 0.0, 0.0),  # forward = min(max(NaN, 0.5), 1) = 0.5; depends on x
        (2.0, nan, 1.0, 0.0, 0.0, 1.0),  # forward = min(2, 1) = 1; depends on b
        (-2.0, -1.0, nan, 0.0, 1.0, 0.0),  # forward = max(-1, -2) = -1; depends on a
        (0.5, -1.0, 1.0, 1.0, 0.0, 0.0),  # forward = 0.5; depends on x (no NaN)
    ]

    n = len(rows)
    x = wp.array([r[0] for r in rows], dtype=dtype, device=device, requires_grad=True)
    a = wp.array([r[1] for r in rows], dtype=dtype, device=device, requires_grad=True)
    b = wp.array([r[2] for r in rows], dtype=dtype, device=device, requires_grad=True)
    out = wp.zeros(n, dtype=dtype, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=n, inputs=[x, a, b], outputs=[out], device=device)
    out.grad.fill_(dtype(1.0))
    tape.backward()
    actual_grad_x = x.grad.numpy()
    actual_grad_a = a.grad.numpy()
    actual_grad_b = b.grad.numpy()
    for i, (xi, ai, bi, gx, ga, gb) in enumerate(rows):
        assert_float_eq(test, actual_grad_x[i], gx, f"adj_clamp x[{i}] (x={xi}, a={ai}, b={bi})")
        assert_float_eq(test, actual_grad_a[i], ga, f"adj_clamp a[{i}] (x={xi}, a={ai}, b={bi})")
        assert_float_eq(test, actual_grad_b[i], gb, f"adj_clamp b[{i}] (x={xi}, a={ai}, b={bi})")


def test_minmax_reduction_adjoint(test, device, dtype, register_kernels=False):
    # The reduction adjoint adj_min(vec) / adj_max(vec) must route the
    # gradient to the index the forward picked. Critically, when v[0] is NaN,
    # the forward fmin reduction skips NaN and picks the first non-NaN
    # extremum; the adjoint must do the same (i.e. NOT route to slot 0).
    vec3_t = wp.types.vector(3, dtype)

    def reduce_kern(
        v: wp.array[vec3_t],
        out_min: wp.array[dtype],
        out_max: wp.array[dtype],
    ):
        i = wp.tid()
        out_min[i] = wp.min(v[i])
        out_max[i] = wp.max(v[i])

    kernel = getkernel(reduce_kern, suffix="reduce_adj_" + dtype.__name__)

    if register_kernels:
        return

    nan = float("nan")
    # The forward fmin reduction returns -1 (slot 2 for min) / 2 (slot 1 for
    # max). The gradient must route there, NOT to slot 0 (the NaN slot).
    rows = [
        [nan, 2.0, -1.0],
    ]
    expected_grad_min = [
        [0.0, 0.0, 1.0],  # min picks v[2]=-1
    ]
    expected_grad_max = [
        [0.0, 1.0, 0.0],  # max picks v[1]=2
    ]

    n = len(rows)
    v = wp.array(rows, dtype=vec3_t, device=device, requires_grad=True)
    out_min = wp.zeros(n, dtype=dtype, device=device, requires_grad=True)
    out_max = wp.zeros(n, dtype=dtype, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=n, inputs=[v], outputs=[out_min, out_max], device=device)
    # Use a sum-of-outputs loss so each row contributes adj_ret = 1 to its
    # own min and max output.
    out_min.grad.fill_(dtype(1.0))
    out_max.grad.fill_(dtype(1.0))
    tape.backward()
    actual_grad = v.grad.numpy()
    # Combined expected gradient: min and max contribute independently.
    for i in range(len(rows)):
        for j in range(3):
            expected = expected_grad_min[i][j] + expected_grad_max[i][j]
            assert_float_eq(
                test,
                actual_grad[i][j],
                expected,
                f"adj_min+adj_max v[{i}][{j}]",
            )


def test_atomic_minmax_adjoint(test, device, dtype, register_kernels=False):
    # adj_atomic_min / adj_atomic_max accumulate the gradient onto `value`
    # whenever the forward op committed `value`'s payload to the slot --
    # including when both operands were NaN. The standard `value == *addr`
    # check returns false for NaN==NaN, so adj_atomic_minmax in builtin.h
    # has an explicit both-NaN branch to catch that case.

    def kern(
        slot: wp.array[dtype],
        value: wp.array[dtype],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        out[i] = wp.atomic_min(slot, i, value[i])

    kernel = getkernel(kern, suffix="atomic_min_adj_" + dtype.__name__)

    if register_kernels:
        return

    nan = float("nan")
    # (slot, value, expected_grad_value)
    # - slot=NaN, value=NaN: forward writes value's payload (slot ends up NaN).
    #   Without the both-NaN branch the adjoint would drop the gradient.
    # - slot=NaN, value=2: forward writes 2 (slot was NaN). value won; full grad.
    # - slot=2,   value=NaN: forward keeps slot=2. value lost; zero grad.
    # - slot=5,   value=2:   forward writes 2. value won; full grad.
    # - slot=2,   value=5:   forward keeps slot=2. value lost; zero grad.
    rows = [
        (nan, nan, 1.0),
        (nan, 2.0, 1.0),
        (2.0, nan, 0.0),
        (5.0, 2.0, 1.0),
        (2.0, 5.0, 0.0),
    ]

    n = len(rows)
    slot = wp.array([r[0] for r in rows], dtype=dtype, device=device, requires_grad=True)
    value = wp.array([r[1] for r in rows], dtype=dtype, device=device, requires_grad=True)
    out = wp.zeros(n, dtype=dtype, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=n, inputs=[slot, value], outputs=[out], device=device)
    slot.grad.fill_(dtype(1.0))
    tape.backward()
    actual_grad_value = value.grad.numpy()
    for i, (s, v, expected) in enumerate(rows):
        assert_float_eq(test, actual_grad_value[i], expected, f"adj_atomic_min grad value[{i}] (slot={s}, value={v})")


def test_atomic_minmax(test, device, dtype, register_kernels=False):
    # wp.atomic_min / wp.atomic_max behave like their non-atomic counterparts:
    # a NaN already in the array is overwritten by a finite value, and a NaN
    # value leaves the array unchanged when the slot holds a finite number.
    # atomic_min / atomic_max do not support float16, so this test runs on
    # float32 / float64 only.

    def kern(
        a: wp.array[dtype],
        b: wp.array[dtype],
        in_min: wp.array[dtype],
        in_max: wp.array[dtype],
        ret_min: wp.array[dtype],
        ret_max: wp.array[dtype],
    ):
        i = wp.tid()
        # a[i] is the array slot, b[i] is the value passed to atomic_min/max.
        # Copy a into in_min/in_max so each test row has its own slot.
        in_min[i] = a[i]
        in_max[i] = a[i]
        ret_min[i] = wp.atomic_min(in_min, i, b[i])
        ret_max[i] = wp.atomic_max(in_max, i, b[i])

    kernel = getkernel(kern, suffix="atomic_" + dtype.__name__)

    if register_kernels:
        return

    nan = float("nan")
    # Each row pairs (a, b): a is the initial array slot, b is the value passed
    # to atomic_min/atomic_max. The post-call slot should equal np.fmin(a, b)
    # / np.fmax(a, b).
    rows_a = [-1.0, nan, nan, -1.0, nan, 2.0]
    rows_b = [nan, -1.0, nan, 2.0, 2.0, nan]
    expected_min = [float(np.fmin(np.float64(a), np.float64(b))) for a, b in zip(rows_a, rows_b, strict=True)]
    expected_max = [float(np.fmax(np.float64(a), np.float64(b))) for a, b in zip(rows_a, rows_b, strict=True)]

    n = len(rows_a)
    a = wp.array(rows_a, dtype=dtype, device=device)
    b = wp.array(rows_b, dtype=dtype, device=device)
    in_min = wp.empty(n, dtype=dtype, device=device)
    in_max = wp.empty(n, dtype=dtype, device=device)
    ret_min = wp.empty(n, dtype=dtype, device=device)
    ret_max = wp.empty(n, dtype=dtype, device=device)
    wp.launch(kernel, dim=n, inputs=[a, b], outputs=[in_min, in_max, ret_min, ret_max], device=device)
    # Post-call array slot is what atomic_min/max wrote.
    actual_post_min = in_min.to("cpu").list()
    actual_post_max = in_max.to("cpu").list()
    # Returned value is the *old* slot (= rows_a[i]).
    actual_ret_min = ret_min.to("cpu").list()
    actual_ret_max = ret_max.to("cpu").list()
    for i in range(n):
        # Final slot should match non-atomic min<nan_as_missing>(a, b).
        assert_float_eq(test, actual_post_min[i], expected_min[i], f"atomic_min slot[{i}]")
        assert_float_eq(test, actual_post_max[i], expected_max[i], f"atomic_max slot[{i}]")
        # Returned value must be the prior slot value (a[i]).
        assert_float_eq(test, actual_ret_min[i], rows_a[i], f"atomic_min ret[{i}]")
        assert_float_eq(test, actual_ret_max[i], rows_a[i], f"atomic_max ret[{i}]")


def test_is_special_quat(test, device, dtype, register_kernels=False):
    quat_type = wp.types.quaternion(dtype)

    def check_special_quat(bool_outputs: wp.array[wp.bool]):
        zeros_quat = quat_type()
        bool_outputs[0] = wp.isfinite(zeros_quat)
        bool_outputs[1] = wp.isinf(zeros_quat)
        bool_outputs[2] = wp.isnan(zeros_quat)

        nan_quat = quat_type(dtype(wp.NAN), dtype(0), dtype(0), dtype(0))
        bool_outputs[3] = wp.isfinite(nan_quat)
        bool_outputs[4] = wp.isinf(nan_quat)
        bool_outputs[5] = wp.isnan(nan_quat)

        inf_quat = quat_type(dtype(wp.INF), dtype(0), dtype(0), dtype(0))
        bool_outputs[6] = wp.isfinite(inf_quat)
        bool_outputs[7] = wp.isinf(inf_quat)
        bool_outputs[8] = wp.isnan(inf_quat)

    kernel = getkernel(check_special_quat, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs_bool = wp.empty(9, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[outputs_bool], device=device)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isfinite(zeros_quat) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isinf(zeros_quat) is not False")
    test.assertFalse(outputs_bool_cpu[2], "wp.isnan(zeros_quat) is not False")

    test.assertFalse(outputs_bool_cpu[3], "wp.isfinite(nan_quat) is not False")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(nan_quat) is not False")
    test.assertTrue(outputs_bool_cpu[5], "wp.isnan(nan_quat) is not True")

    test.assertFalse(outputs_bool_cpu[6], "wp.isfinite(inf_quat) is not False")
    test.assertTrue(outputs_bool_cpu[7], "wp.isinf(inf_quat) is not True")
    test.assertFalse(outputs_bool_cpu[8], "wp.isnan(inf_quat) is not False")


def test_copysign(test, device, dtype, register_kernels=False):
    # Forward: wp.copysign(x, y) returns x with the sign bit of y. Covers
    # finite inputs, signed zeros, and NaN x (whose magnitude is preserved
    # but whose sign is replaced by y's). NaN y is platform-dependent for the
    # CPU JIT (no signbit access without bit twiddling on that path), so it's
    # not exercised here.

    def kern(
        x: wp.array[dtype],
        y: wp.array[dtype],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        out[i] = wp.copysign(x[i], y[i])

    kernel = getkernel(kern, suffix="copysign_" + dtype.__name__)

    if register_kernels:
        return

    nan = float("nan")
    # (x, y, expected). Use np.copysign as the oracle (matches IEEE semantics).
    rows = [
        (3.0, 1.0, 3.0),
        (3.0, -1.0, -3.0),
        (-3.0, 1.0, 3.0),
        (-3.0, -1.0, -3.0),
        (0.0, -1.0, -0.0),
        (-0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, -0.0, -1.0),
        (nan, 1.0, nan),
        (nan, -1.0, nan),
    ]

    n = len(rows)
    x = wp.array([r[0] for r in rows], dtype=dtype, device=device)
    y = wp.array([r[1] for r in rows], dtype=dtype, device=device)
    out = wp.empty(n, dtype=dtype, device=device)
    wp.launch(kernel, dim=n, inputs=[x, y], outputs=[out], device=device)
    actual = out.to("cpu").list()

    for i, (xi, yi, expected) in enumerate(rows):
        # NaN sign-bit on the CPU JIT defaults to positive in the reference path,
        # so just check NaN-ness for NaN x.
        if math.isnan(expected):
            test.assertTrue(math.isnan(actual[i]), f"copysign({xi}, {yi}) expected NaN, got {actual[i]}")
        else:
            assert_float_eq(test, actual[i], expected, f"copysign({xi}, {yi})")


def test_copysign_adjoint(test, device, dtype, register_kernels=False):
    # adj_copysign: d/dx is +1 when signs of x and y agree, -1 otherwise. d/dy
    # is 0 almost everywhere (result depends on y only through its sign).

    def kern(
        x: wp.array[dtype],
        y: wp.array[dtype],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        out[i] = wp.copysign(x[i], y[i])

    kernel = getkernel(kern, suffix="copysign_adj_" + dtype.__name__)

    if register_kernels:
        return

    # (x, y, expected_grad_x). Gradient on y is always 0. Signed-zero inputs
    # are classified by their sign bit (not by `< 0`), matching the forward.
    rows = [
        (3.0, 1.0, 1.0),  # signs agree -> +1
        (3.0, -1.0, -1.0),  # signs differ -> -1
        (-3.0, 1.0, -1.0),  # signs differ -> -1
        (-3.0, -1.0, 1.0),  # signs agree -> +1
        (-0.0, -1.0, 1.0),  # both negative (sign bit) -> +1
        (-0.0, 1.0, -1.0),  # signs differ -> -1
        (0.0, -1.0, -1.0),  # signs differ -> -1
        (0.0, 1.0, 1.0),  # signs agree -> +1
    ]

    n = len(rows)
    x = wp.array([r[0] for r in rows], dtype=dtype, device=device, requires_grad=True)
    y = wp.array([r[1] for r in rows], dtype=dtype, device=device, requires_grad=True)
    out = wp.zeros(n, dtype=dtype, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=n, inputs=[x, y], outputs=[out], device=device)
    out.grad.fill_(dtype(1.0))
    tape.backward()
    actual_grad_x = x.grad.numpy()
    actual_grad_y = y.grad.numpy()

    for i, (xi, yi, expected_x) in enumerate(rows):
        assert_float_eq(test, actual_grad_x[i], expected_x, f"adj_copysign d/dx ({xi}, {yi})")
        assert_float_eq(test, actual_grad_y[i], 0.0, f"adj_copysign d/dy ({xi}, {yi})")


devices = get_test_devices()


class TestSpecialValues(unittest.TestCase):
    pass


for dtype in [wp.float16, wp.float32, wp.float64]:
    add_function_test_register_kernel(
        TestSpecialValues, f"test_infinity_{dtype.__name__}", test_infinity_scalar, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues, f"test_nan_{dtype.__name__}", test_nan_scalar, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues, f"test_is_special_vec_{dtype.__name__}", test_is_special_vec, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues, f"test_is_special_mat_{dtype.__name__}", test_is_special_mat, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues, f"test_is_special_quat_{dtype.__name__}", test_is_special_quat, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues,
        f"test_minmax_scalar_{dtype.__name__}",
        test_minmax_scalar,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpecialValues,
        f"test_minmax_vec_{dtype.__name__}",
        test_minmax_vec,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpecialValues,
        f"test_minmax_reduction_adjoint_{dtype.__name__}",
        test_minmax_reduction_adjoint,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpecialValues,
        f"test_clamp_adjoint_{dtype.__name__}",
        test_clamp_adjoint,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpecialValues,
        f"test_copysign_{dtype.__name__}",
        test_copysign,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpecialValues,
        f"test_copysign_adjoint_{dtype.__name__}",
        test_copysign_adjoint,
        devices=devices,
        dtype=dtype,
    )

# atomic_min / atomic_max do not support float16 -- skip it here.
for dtype in [wp.float32, wp.float64]:
    add_function_test_register_kernel(
        TestSpecialValues,
        f"test_atomic_minmax_{dtype.__name__}",
        test_atomic_minmax,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpecialValues,
        f"test_atomic_minmax_adjoint_{dtype.__name__}",
        test_atomic_minmax_adjoint,
        devices=devices,
        dtype=dtype,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
