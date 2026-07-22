# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from functools import cache, partial
from typing import Any
from unittest import mock

import numpy as np

import warp as wp
from warp._src.jax import get_jax_device
from warp.tests.unittest_utils import *

# Prevent JAX from preallocating GPU memory before any module-level version checks import it.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# default array size for tests
ARRAY_SIZE = 1024 * 1024
TILE_STORE_SIZE = 64


# basic kernel with one input and output
@wp.kernel
def triple_kernel(inp: wp.array[float], output: wp.array[float]):
    tid = wp.tid()
    output[tid] = 3.0 * inp[tid]


# generic kernel with one scalar input and output
@wp.kernel
def triple_kernel_scalar(inp: wp.array[Any], output: wp.array[Any]):
    tid = wp.tid()
    output[tid] = inp.dtype(3) * inp[tid]


# generic kernel with one vector/matrix input and output
@wp.kernel
def triple_kernel_vecmat(inp: wp.array[Any], output: wp.array[Any]):
    tid = wp.tid()
    output[tid] = inp.dtype.dtype(3) * inp[tid]


@wp.kernel
def inc_1d_kernel(x: wp.array[float], y: wp.array[float]):
    tid = wp.tid()
    y[tid] = x[tid] + 1.0


@wp.kernel
def inc_2d_kernel(x: wp.array2d[float], y: wp.array2d[float]):
    i, j = wp.tid()
    y[i, j] = x[i, j] + 1.0


@wp.kernel
def shaped_tile_store_kernel(output: wp.array[float]):
    tile = wp.tile_ones(dtype=float, shape=TILE_STORE_SIZE)
    wp.tile_store(output, tile)


# kernel with multiple inputs and outputs
@wp.kernel
def multiarg_kernel(
    # inputs
    a: wp.array[float],
    b: wp.array[float],
    c: wp.array[float],
    # outputs
    ab: wp.array[float],
    bc: wp.array[float],
):
    tid = wp.tid()
    ab[tid] = a[tid] + b[tid]
    bc[tid] = b[tid] + c[tid]


# various types for testing
scalar_types = wp._src.types.scalar_types
vector_types = []
matrix_types = []
for dim in [2, 3, 4]:
    for T in scalar_types:
        vector_types.append(wp.types.vector(dim, T))
        matrix_types.append(wp.types.matrix((dim, dim), T))

# explicitly overload generic kernels to avoid module reloading during tests
for T in scalar_types:
    wp.overload(triple_kernel_scalar, [wp.array[T], wp.array[T]])
for T in [*vector_types, *matrix_types]:
    wp.overload(triple_kernel_vecmat, [wp.array[T], wp.array[T]])


def _jax_version():
    try:
        jax = _import_jax()
    except Exception:
        return (0, 0, 0)

    return jax.__version_info__


def _import_jax():
    import jax  # noqa: PLC0415

    return jax


def _import_jax_numpy():
    import jax.numpy as jp  # noqa: PLC0415

    return jp


class _RecordingFfiModule:
    """Minimal Warp module stand-in that records ``load()`` calls and returns a canned result."""

    def __init__(self, load_error=None, load_result=mock.sentinel.module_exec):
        self.name = "recording_module"
        self.loaded_devices = []
        self.load_error = load_error
        self.load_result = load_result

    def load(self, device, _block_dim=None):
        self.loaded_devices.append(device)
        if self.load_error is not None:
            raise self.load_error
        return self.load_result


_JAX_NAMESPACE_MODULES = ("warp.jax", "warp.jax_experimental")


def _clear_jax_namespace_modules():
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in _JAX_NAMESPACE_MODULES):
            del sys.modules[module_name]


def _clear_jax_experimental_warning_cache():
    wp._src.logger._warnings_seen = {
        entry
        for entry in wp._src.logger._warnings_seen
        if not (entry[0] is DeprecationWarning and isinstance(entry[1], str) and "warp.jax_experimental" in entry[1])
    }


def _import_deprecated_jax_namespace(module_name):
    _clear_jax_experimental_warning_cache()
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()) as stderr:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)
    return module, stderr.getvalue()


def _get_experimental_custom_call_jax_kernel():
    _clear_jax_experimental_warning_cache()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module("warp.jax_experimental.custom_call")
        return module.jax_kernel
    finally:
        _clear_jax_experimental_warning_cache()


def _get_experimental_register_ffi_callback():
    _clear_jax_experimental_warning_cache()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module("warp.jax_experimental.ffi")
        return module.register_ffi_callback
    finally:
        _clear_jax_experimental_warning_cache()


def test_jax_experimental_import_deprecation(test, device):
    _clear_jax_namespace_modules()

    module, output = _import_deprecated_jax_namespace("warp.jax_experimental")

    expected = (
        "Warp DeprecationWarning: The `warp.jax_experimental` namespace is deprecated "
        "and will be removed in Warp 1.18. Use top-level `warp` JAX APIs instead.\n"
    )
    test.assertEqual(output, expected)
    test.assertIs(module.jax_kernel, wp.jax_kernel)
    test.assertIsNot(module.jax_callable, wp.jax_callable)
    test.assertIs(module.GraphMode, wp.JaxCallableGraphMode)
    test.assertIs(module.ModulePreloadMode, wp.JaxModulePreloadMode)
    test.assertTrue(callable(module.register_ffi_callback))


def test_jax_experimental_ffi_import_deprecation(test, device):
    _clear_jax_namespace_modules()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        importlib.import_module("warp.jax_experimental")
    sys.modules.pop("warp.jax_experimental.ffi", None)

    module, output = _import_deprecated_jax_namespace("warp.jax_experimental.ffi")
    ffi_module = importlib.import_module("warp._src.jax.ffi")

    expected = (
        "Warp DeprecationWarning: The `warp.jax_experimental.ffi` namespace is deprecated "
        "and will be removed in Warp 1.18. Use top-level `warp` JAX APIs instead.\n"
    )
    test.assertEqual(output, expected)
    test.assertIs(module.jax_kernel, ffi_module.jax_kernel)
    test.assertIsNot(module.jax_callable, ffi_module.jax_callable)
    test.assertIs(module.register_ffi_callback, ffi_module.register_ffi_callback)
    test.assertIs(module.GraphMode, ffi_module.JaxCallableGraphMode)
    test.assertIs(module.ModulePreloadMode, ffi_module.JaxModulePreloadMode)


def test_jax_experimental_custom_call_import_deprecation(test, device):
    _clear_jax_namespace_modules()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        importlib.import_module("warp.jax_experimental")
    sys.modules.pop("warp.jax_experimental.custom_call", None)

    module, output = _import_deprecated_jax_namespace("warp.jax_experimental.custom_call")
    custom_call_module = importlib.import_module("warp._src.jax.custom_call")

    expected = (
        "Warp DeprecationWarning: The `warp.jax_experimental.custom_call` namespace is deprecated "
        "and will be removed in Warp 1.18. Use `warp.jax_kernel()` instead.\n"
    )
    test.assertEqual(output, expected)
    test.assertIs(module.jax_kernel, custom_call_module.jax_kernel)


def test_dtype_from_jax(test, device):
    jp = _import_jax_numpy()

    def test_conversions(jax_type, warp_type):
        test.assertEqual(wp.dtype_from_jax(jax_type), warp_type)
        test.assertEqual(wp.dtype_from_jax(jp.dtype(jax_type)), warp_type)

    test_conversions(jp.float16, wp.float16)
    test_conversions(jp.float32, wp.float32)
    test_conversions(jp.float64, wp.float64)
    test_conversions(jp.int8, wp.int8)
    test_conversions(jp.int16, wp.int16)
    test_conversions(jp.int32, wp.int32)
    test_conversions(jp.int64, wp.int64)
    test_conversions(jp.uint8, wp.uint8)
    test_conversions(jp.uint16, wp.uint16)
    test_conversions(jp.uint32, wp.uint32)
    test_conversions(jp.uint64, wp.uint64)
    test_conversions(jp.bool_, wp.bool)


def test_dtype_to_jax(test, device):
    jp = _import_jax_numpy()

    def test_conversions(warp_type, jax_type):
        test.assertEqual(wp.dtype_to_jax(warp_type), jax_type)

    test_conversions(wp.float16, jp.float16)
    test_conversions(wp.float32, jp.float32)
    test_conversions(wp.float64, jp.float64)
    test_conversions(wp.int8, jp.int8)
    test_conversions(wp.int16, jp.int16)
    test_conversions(wp.int32, jp.int32)
    test_conversions(wp.int64, jp.int64)
    test_conversions(wp.uint8, jp.uint8)
    test_conversions(wp.uint16, jp.uint16)
    test_conversions(wp.uint32, jp.uint32)
    test_conversions(wp.uint64, jp.uint64)
    test_conversions(wp.bool, jp.bool_)


def test_device_conversion(test, device):
    jax_device = wp.device_to_jax(device)
    warp_device = wp.device_from_jax(jax_device)
    test.assertEqual(warp_device, device)


def test_jax_kernel_basic(test, device, use_ffi=False):
    jp = _import_jax_numpy()

    if use_ffi:
        jax_kernel = wp.jax_kernel

        jax_triple = jax_kernel(triple_kernel)
    else:
        jax_kernel = _get_experimental_custom_call_jax_kernel()

        jax_triple = jax_kernel(triple_kernel, quiet=True)  # suppress deprecation warnings

    n = ARRAY_SIZE

    @jax.jit
    def f():
        x = jp.arange(n, dtype=jp.float32)
        return jax_triple(x)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        y = f()

    jax.block_until_ready(y)

    result = np.asarray(y).reshape((n,))
    expected = 3 * np.arange(n, dtype=np.float32)

    assert_np_equal(result, expected)


def test_jax_kernel_scalar(test, device, use_ffi=False):
    jp = _import_jax_numpy()

    if use_ffi:
        jax_kernel = wp.jax_kernel

        kwargs = {}
    else:
        jax_kernel = _get_experimental_custom_call_jax_kernel()

        kwargs = {"quiet": True}

    # use a smallish size to ensure arange * 3 doesn't overflow
    n = 64

    for T in scalar_types:
        jp_dtype = wp.dtype_to_jax(T)
        np_dtype = wp.dtype_to_numpy(T)

        with test.subTest(msg=T.__name__):
            # get the concrete overload
            kernel_instance = triple_kernel_scalar.add_overload([wp.array[T], wp.array[T]])

            jax_triple = jax_kernel(kernel_instance, **kwargs)

            @jax.jit
            def f(jax_triple=jax_triple, jp_dtype=jp_dtype):
                x = jp.arange(n, dtype=jp_dtype)
                return jax_triple(x)

            # run on the given device
            with jax.default_device(wp.device_to_jax(device)):
                y = f()

            jax.block_until_ready(y)

            result = np.asarray(y).reshape((n,))
            expected = 3 * np.arange(n, dtype=np_dtype)

            assert_np_equal(result, expected)


def test_jax_kernel_vecmat(test, device, use_ffi=False):
    jp = _import_jax_numpy()

    if use_ffi:
        jax_kernel = wp.jax_kernel

        kwargs = {}
    else:
        jax_kernel = _get_experimental_custom_call_jax_kernel()

        kwargs = {"quiet": True}

    for T in [*vector_types, *matrix_types]:
        jp_dtype = wp.dtype_to_jax(T._wp_scalar_type_)
        np_dtype = wp.dtype_to_numpy(T._wp_scalar_type_)

        # use a smallish size to ensure arange * 3 doesn't overflow
        n = 64 // T._length_
        scalar_shape = (n, *T._shape_)
        scalar_len = n * T._length_

        with test.subTest(msg=T.__name__):
            # get the concrete overload
            kernel_instance = triple_kernel_vecmat.add_overload([wp.array[T], wp.array[T]])

            jax_triple = jax_kernel(kernel_instance, **kwargs)

            @jax.jit
            def f(jax_triple=jax_triple, jp_dtype=jp_dtype, scalar_len=scalar_len, scalar_shape=scalar_shape):
                x = jp.arange(scalar_len, dtype=jp_dtype).reshape(scalar_shape)
                return jax_triple(x)

            # run on the given device
            with jax.default_device(wp.device_to_jax(device)):
                y = f()

            jax.block_until_ready(y)

            result = np.asarray(y).reshape(scalar_shape)
            expected = 3 * np.arange(scalar_len, dtype=np_dtype).reshape(scalar_shape)

            assert_np_equal(result, expected)


def test_jax_kernel_multiarg(test, device, use_ffi=False):
    jp = _import_jax_numpy()

    if use_ffi:
        jax_kernel = wp.jax_kernel

        jax_multiarg = jax_kernel(multiarg_kernel, num_outputs=2)
    else:
        jax_kernel = _get_experimental_custom_call_jax_kernel()

        jax_multiarg = jax_kernel(multiarg_kernel, quiet=True)

    n = ARRAY_SIZE

    @jax.jit
    def f():
        a = jp.full(n, 1, dtype=jp.float32)
        b = jp.full(n, 2, dtype=jp.float32)
        c = jp.full(n, 3, dtype=jp.float32)
        return jax_multiarg(a, b, c)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        x, y = f()

    jax.block_until_ready([x, y])

    result_x, result_y = np.asarray(x), np.asarray(y)
    expected_x = np.full(n, 3, dtype=np.float32)
    expected_y = np.full(n, 5, dtype=np.float32)

    assert_np_equal(result_x, expected_x)
    assert_np_equal(result_y, expected_y)


def test_jax_kernel_launch_dims(test, device, use_ffi=False):
    jp = _import_jax_numpy()

    if use_ffi:
        jax_kernel = wp.jax_kernel

        kwargs = {}
    else:
        jax_kernel = _get_experimental_custom_call_jax_kernel()

        kwargs = {"quiet": True}

    n = 64
    m = 32

    # Test with 1D launch dims
    jax_inc_1d = jax_kernel(
        inc_1d_kernel, launch_dims=(n - 2,), **kwargs
    )  # Intentionally not the same as the first dimension of the input

    @jax.jit
    def f_1d():
        x = jp.arange(n, dtype=jp.float32)
        return jax_inc_1d(x)

    # Test with 2D launch dims
    jax_inc_2d = jax_kernel(
        inc_2d_kernel, launch_dims=(n - 2, m - 2), **kwargs
    )  # Intentionally not the same as the first dimension of the input

    @jax.jit
    def f_2d():
        x = jp.zeros((n, m), dtype=jp.float32) + 3.0
        return jax_inc_2d(x)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        y_1d = f_1d()
        y_2d = f_2d()

    jax.block_until_ready([y_1d, y_2d])

    result_1d = np.asarray(y_1d).reshape((n - 2,))
    expected_1d = np.arange(n - 2, dtype=np.float32) + 1.0

    result_2d = np.asarray(y_2d).reshape((n - 2, m - 2))
    expected_2d = np.full((n - 2, m - 2), 4.0, dtype=np.float32)

    assert_np_equal(result_1d, expected_1d)
    assert_np_equal(result_2d, expected_2d)


# =========================================================================================================
# JAX FFI
# =========================================================================================================


@wp.kernel
def add_kernel(a: wp.array[float], b: wp.array[float], output: wp.array[float]):
    tid = wp.tid()
    output[tid] = a[tid] + b[tid]


@wp.kernel
def add2d_kernel(a: wp.array2d[float], b: wp.array2d[float], output: wp.array2d[float]):
    i, j = wp.tid()
    output[i, j] = a[i, j] + b[i, j]


@wp.kernel
def axpy_kernel(x: wp.array[float], y: wp.array[float], alpha: float, out: wp.array[float]):
    tid = wp.tid()
    out[tid] = alpha * x[tid] + y[tid]


@wp.kernel
def sincos_kernel(angle: wp.array[float], sin_out: wp.array[float], cos_out: wp.array[float]):
    tid = wp.tid()
    sin_out[tid] = wp.sin(angle[tid])
    cos_out[tid] = wp.cos(angle[tid])


@wp.kernel
def diagonal_kernel(output: wp.array[wp.mat33]):
    tid = wp.tid()
    d = float(tid + 1)
    output[tid] = wp.mat33(d, 0.0, 0.0, 0.0, d * 2.0, 0.0, 0.0, 0.0, d * 3.0)


@wp.kernel
def scale_kernel(a: wp.array[float], s: float, output: wp.array[float]):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def scale_vec_kernel(a: wp.array[wp.vec2], s: float, output: wp.array[wp.vec2]):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def accum_kernel(a: wp.array[float], b: wp.array[float]):
    tid = wp.tid()
    b[tid] += a[tid]


@wp.kernel
def matmul_kernel(
    a: wp.array2d[float],  # NxK
    b: wp.array2d[float],  # KxM
    c: wp.array2d[float],  # NxM
):
    # launch dims should be (N, M)
    i, j = wp.tid()
    N = a.shape[0]
    K = a.shape[1]
    M = b.shape[1]
    if i < N and j < M:
        s = wp.float32(0)
        for k in range(K):
            s += a[i, k] * b[k, j]
        c[i, j] = s


@wp.kernel
def in_out_kernel(
    a: wp.array[float],  # input only
    b: wp.array[float],  # input and output
    c: wp.array[float],  # output only
):
    tid = wp.tid()
    b[tid] += a[tid]
    c[tid] = 2.0 * a[tid]


@wp.kernel
def multi_out_kernel(a: wp.array[float], b: wp.array[float], s: float, c: wp.array[float], d: wp.array[float]):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]
    d[tid] = s * a[tid]


@wp.kernel
def multi_out_kernel_v2(a: wp.array[float], b: wp.array[float], s: float, c: wp.array[float], d: wp.array[float]):
    tid = wp.tid()
    c[tid] = a[tid] * a[tid]
    d[tid] = a[tid] * b[tid] * s


@wp.kernel
def multi_out_kernel_v3(a: wp.array[float], b: wp.array[float], s: float, c: wp.array[float], d: wp.array[float]):
    tid = wp.tid()
    c[tid] = a[tid] ** 2.0
    d[tid] = a[tid] * b[tid] * s


@wp.kernel
def scale_sum_square_kernel(a: wp.array[float], b: wp.array[float], s: float, c: wp.array[float]):
    tid = wp.tid()
    c[tid] = (a[tid] * s + b[tid]) ** 2.0


# Kernels using subscript-style type hints (wp.array[dtype] syntax)
@wp.kernel
def add_kernel_subscript(a: wp.array[float], b: wp.array[float], output: wp.array[float]):
    tid = wp.tid()
    output[tid] = a[tid] + b[tid]


@wp.kernel
def scale_vec_kernel_subscript(a: wp.array[wp.vec2], s: float, output: wp.array[wp.vec2]):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def scale_sum_square_kernel_subscript(a: wp.array[float], b: wp.array[float], s: float, c: wp.array[float]):
    tid = wp.tid()
    c[tid] = (a[tid] * s + b[tid]) ** 2.0


# The Python function to call.
# Note the argument annotations, just like Warp kernels.
def scale_func(
    # inputs
    a: wp.array[float],
    b: wp.array[wp.vec2],
    s: float,
    # outputs
    c: wp.array[float],
    d: wp.array[wp.vec2],
):
    wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])
    wp.launch(scale_vec_kernel, dim=b.shape, inputs=[b, s], outputs=[d])


def in_out_func(
    a: wp.array[float],  # input only
    b: wp.array[float],  # input and output
    c: wp.array[float],  # output only
):
    wp.launch(scale_kernel, dim=a.size, inputs=[a, 2.0], outputs=[c])
    wp.launch(accum_kernel, dim=a.size, inputs=[a, b])  # modifies `b`


def double_func(
    # inputs
    a: wp.array[float],
    # outputs
    b: wp.array[float],
):
    wp.launch(scale_kernel, dim=a.shape, inputs=[a, 2.0], outputs=[b])


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_add(test, device):
    # two inputs and one output
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_add = jax_kernel(add_kernel)

    @jax.jit
    def f():
        n = ARRAY_SIZE
        a = jp.arange(n, dtype=jp.float32)
        b = jp.ones(n, dtype=jp.float32)
        return jax_add(a, b)

    with jax.default_device(wp.device_to_jax(device)):
        (y,) = f()

    jax.block_until_ready(y)

    result = np.asarray(y)
    expected = np.arange(1, ARRAY_SIZE + 1, dtype=np.float32)

    assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_sincos(test, device):
    # one input and two outputs
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_sincos = jax_kernel(sincos_kernel, num_outputs=2)

    n = ARRAY_SIZE

    @jax.jit
    def f():
        a = jp.linspace(0, 2 * jp.pi, n, dtype=jp.float32)
        return jax_sincos(a)

    with jax.default_device(wp.device_to_jax(device)):
        s, c = f()

    jax.block_until_ready([s, c])

    result_s = np.asarray(s)
    result_c = np.asarray(c)

    a = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    expected_s = np.sin(a)
    expected_c = np.cos(a)

    assert_np_equal(result_s, expected_s, tol=1e-4)
    assert_np_equal(result_c, expected_c, tol=1e-4)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_diagonal(test, device):
    # no inputs and one output
    jax_kernel = wp.jax_kernel

    jax_diagonal = jax_kernel(diagonal_kernel)

    @jax.jit
    def f():
        # launch dimensions determine output size
        return jax_diagonal(launch_dims=4)

    with jax.default_device(wp.device_to_jax(device)):
        (d,) = f()

    jax.block_until_ready(d)

    result = np.asarray(d)
    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]],
            [[3.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]],
            [[4.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
        ],
        dtype=np.float32,
    )

    assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_in_out(test, device):
    # in-out args
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_func = jax_kernel(in_out_kernel, num_outputs=2, in_out_argnames=["b"])

    f = jax.jit(jax_func)

    with jax.default_device(wp.device_to_jax(device)):
        a = jp.ones(ARRAY_SIZE, dtype=jp.float32)
        b = jp.arange(ARRAY_SIZE, dtype=jp.float32)
        b, c = f(a, b)

    jax.block_until_ready([b, c])

    assert_np_equal(b, np.arange(1, ARRAY_SIZE + 1, dtype=np.float32))
    assert_np_equal(c, np.full(ARRAY_SIZE, 2, dtype=np.float32))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_scale_vec_constant(test, device):
    # multiply vectors by scalar (constant)
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_scale_vec = jax_kernel(scale_vec_kernel)

    @jax.jit
    def f():
        a = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))  # array of vec2
        s = 2.0
        return jax_scale_vec(a, s)

    with jax.default_device(wp.device_to_jax(device)):
        (b,) = f()

    jax.block_until_ready(b)

    expected = 2 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(b, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_scale_vec_static(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit() returns zeros instead of scaled values")

    # multiply vectors by scalar (static arg)
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_scale_vec = jax_kernel(scale_vec_kernel)

    # NOTE: scalar arguments must be static compile-time constants
    @partial(jax.jit, static_argnames=["s"])
    def f(a, s):
        return jax_scale_vec(a, s)

    a = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))  # array of vec2
    s = 3.0

    with jax.default_device(wp.device_to_jax(device)):
        (b,) = f(a, s)

    jax.block_until_ready(b)

    expected = 3 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(b, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_launch_dims_default(test, device):
    # specify default launch dims
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    N, M, K = 3, 4, 2

    jax_matmul = jax_kernel(matmul_kernel, launch_dims=(N, M))

    @jax.jit
    def f():
        a = jp.full((N, K), 2, dtype=jp.float32)
        b = jp.full((K, M), 3, dtype=jp.float32)

        # use default launch dims
        return jax_matmul(a, b)

    with jax.default_device(wp.device_to_jax(device)):
        (result,) = f()

    jax.block_until_ready(result)

    expected = np.full((3, 4), 12, dtype=np.float32)

    test.assertEqual(result.shape, expected.shape)
    assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_launch_dims_custom(test, device):
    # specify custom launch dims per call
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_matmul = jax_kernel(matmul_kernel)

    @jax.jit
    def f():
        N1, M1, K1 = 3, 4, 2
        a1 = jp.full((N1, K1), 2, dtype=jp.float32)
        b1 = jp.full((K1, M1), 3, dtype=jp.float32)

        # use custom launch dims
        result1 = jax_matmul(a1, b1, launch_dims=(N1, M1))

        N2, M2, K2 = 4, 3, 2
        a2 = jp.full((N2, K2), 2, dtype=jp.float32)
        b2 = jp.full((K2, M2), 3, dtype=jp.float32)

        # use different custom launch dims
        result2 = jax_matmul(a2, b2, launch_dims=(N2, M2))

        return result1[0], result2[0]

    with jax.default_device(wp.device_to_jax(device)):
        result1, result2 = f()

    jax.block_until_ready([result1, result2])

    expected1 = np.full((3, 4), 12, dtype=np.float32)
    expected2 = np.full((4, 3), 12, dtype=np.float32)

    test.assertEqual(result1.shape, expected1.shape)
    test.assertEqual(result2.shape, expected2.shape)
    assert_np_equal(result1, expected1)
    assert_np_equal(result2, expected2)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_callable_scale_constant(test, device):
    # scale two arrays using a constant
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    jax_func = jax_callable(scale_func, num_outputs=2)

    @jax.jit
    def f():
        # inputs
        a = jp.arange(ARRAY_SIZE, dtype=jp.float32)
        b = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))  # wp.vec2
        s = 2.0

        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    with jax.default_device(wp.device_to_jax(device)):
        result1, result2 = f()

    jax.block_until_ready([result1, result2])

    expected1 = 2 * np.arange(ARRAY_SIZE, dtype=np.float32)
    expected2 = 2 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(result1, expected1)
    assert_np_equal(result2, expected2)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_callable_scale_static(test, device):
    # scale two arrays using a static arg
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    jax_func = jax_callable(scale_func, num_outputs=2)

    # NOTE: scalar arguments must be static compile-time constants
    @partial(jax.jit, static_argnames=["s"])
    def f(a, b, s):
        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    with jax.default_device(wp.device_to_jax(device)):
        # inputs
        a = jp.arange(ARRAY_SIZE, dtype=jp.float32)
        b = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))  # wp.vec2
        s = 3.0
        result1, result2 = f(a, b, s)

    jax.block_until_ready([result1, result2])

    expected1 = 3 * np.arange(ARRAY_SIZE, dtype=np.float32)
    expected2 = 3 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(result1, expected1)
    assert_np_equal(result2, expected2)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_callable_in_out(test, device):
    # in-out arguments
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    jax_func = jax_callable(in_out_func, num_outputs=2, in_out_argnames=["b"])

    f = jax.jit(jax_func)

    with jax.default_device(wp.device_to_jax(device)):
        a = jp.ones(ARRAY_SIZE, dtype=jp.float32)
        b = jp.arange(ARRAY_SIZE, dtype=jp.float32)
        b, c = f(a, b)

    jax.block_until_ready([b, c])

    assert_np_equal(b, np.arange(1, ARRAY_SIZE + 1, dtype=np.float32))
    assert_np_equal(c, np.full(ARRAY_SIZE, 2, dtype=np.float32))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_callable_graph_cache(test, device):
    # test graph caching limits
    jax = _import_jax()
    jp = _import_jax_numpy()

    ffi_module = importlib.import_module("warp._src.jax.ffi")
    default_graph_cache_max = ffi_module.JAX_CALLABLE_DEFAULT_GRAPH_CACHE_MAX

    with contextlib.redirect_stderr(io.StringIO()):
        experimental_ffi = importlib.import_module("warp.jax_experimental.ffi")

    _clear_jax_experimental_warning_cache()
    wp.load_module(module=scale_kernel.module, device=device)
    old_force_module_load = wp.config.enable_graph_capture_module_load_by_default
    wp.config.enable_graph_capture_module_load_by_default = False
    try:
        JaxCallableGraphMode = wp.JaxCallableGraphMode
        clear_jax_callable_graph_cache = wp.clear_jax_callable_graph_cache
        jax_callable = wp.jax_callable

        # --- test with default cache settings ---

        jax_double = jax_callable(double_func, graph_mode=JaxCallableGraphMode.WARP)
        f = jax.jit(jax_double)
        arrays = []

        test.assertEqual(jax_double.graph_cache_max, default_graph_cache_max)

        with jax.default_device(wp.device_to_jax(device)):
            for i in range(10):
                n = 10 + i
                a = jp.arange(n, dtype=jp.float32)
                (b,) = f(a)

                assert_np_equal(b, 2 * np.arange(n, dtype=np.float32))

                # ensure graph cache is always growing
                test.assertEqual(jax_double.graph_cache_size, i + 1)

                # keep JAX array alive to prevent the memory from being reused, thus forcing a new graph capture each time
                arrays.append(a)

        jax_double_none = jax_callable(double_func, graph_mode=JaxCallableGraphMode.WARP, graph_cache_max=None)
        test.assertIsNone(jax_double_none.graph_cache_max)

        # --- test clearing one callable's cache ---

        clear_jax_callable_graph_cache(jax_double)

        test.assertEqual(jax_double.graph_cache_size, 0)

        # --- test with a custom cache limit ---

        graph_cache_max = 5
        jax_double = jax_callable(double_func, graph_mode=JaxCallableGraphMode.WARP, graph_cache_max=graph_cache_max)
        f = jax.jit(jax_double)
        arrays = []

        test.assertEqual(jax_double.graph_cache_max, graph_cache_max)

        with jax.default_device(wp.device_to_jax(device)):
            for i in range(10):
                n = 10 + i
                a = jp.arange(n, dtype=jp.float32)
                (b,) = f(a)

                assert_np_equal(b, 2 * np.arange(n, dtype=np.float32))

                # ensure graph cache size is capped
                test.assertEqual(jax_double.graph_cache_size, min(i + 1, graph_cache_max))

                # keep JAX array alive to prevent the memory from being reused, thus forcing a new graph capture
                arrays.append(a)

        # --- test clearing all callables' caches ---

        clear_jax_callable_graph_cache()

        with wp._src.jax.ffi._FFI_REGISTRY_LOCK:
            for c in wp._src.jax.ffi._FFI_CALLABLE_REGISTRY.values():
                test.assertEqual(c.graph_cache_size, 0)

        # --- test with a custom default cache limit ---

        saved_max = ffi_module.get_jax_callable_default_graph_cache_max()
        try:
            ffi_module.set_jax_callable_default_graph_cache_max(5)
            jax_double = jax_callable(double_func, graph_mode=JaxCallableGraphMode.WARP)
            test.assertEqual(jax_double.graph_cache_max, default_graph_cache_max)
            jax_double_none = jax_callable(double_func, graph_mode=JaxCallableGraphMode.WARP, graph_cache_max=None)
            test.assertIsNone(jax_double_none.graph_cache_max)

            # Deprecated namespace preserves the runtime-default behavior for compatibility.
            jax_double = experimental_ffi.jax_callable(double_func, graph_mode=JaxCallableGraphMode.WARP)
            f = jax.jit(jax_double)
            arrays = []

            test.assertEqual(jax_double.graph_cache_max, ffi_module.get_jax_callable_default_graph_cache_max())
            jax_double_none = experimental_ffi.jax_callable(
                double_func, graph_mode=JaxCallableGraphMode.WARP, graph_cache_max=None
            )
            test.assertEqual(jax_double_none.graph_cache_max, ffi_module.get_jax_callable_default_graph_cache_max())

            with jax.default_device(wp.device_to_jax(device)):
                for i in range(10):
                    n = 10 + i
                    a = jp.arange(n, dtype=jp.float32)
                    (b,) = f(a)

                    assert_np_equal(b, 2 * np.arange(n, dtype=np.float32))

                    # ensure graph cache size is capped
                    test.assertEqual(
                        jax_double.graph_cache_size,
                        min(i + 1, ffi_module.get_jax_callable_default_graph_cache_max()),
                    )

                    # keep JAX array alive to prevent the memory from being reused, thus forcing a new graph capture
                    arrays.append(a)

            clear_jax_callable_graph_cache()

        finally:
            ffi_module.set_jax_callable_default_graph_cache_max(saved_max)

    finally:
        wp.config.enable_graph_capture_module_load_by_default = old_force_module_load
        _clear_jax_experimental_warning_cache()


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
def test_ffi_jax_callable_graph_replay_skips_module_load(test, device):
    """Test that FFI graph capture loads no extra modules and graph replay loads none.

    Emulates drivers that cannot compile modules during graph capture by pinning the reported
    driver version below 12.3. With the required module loaded up front, capture must not
    force-load every registered module, and replaying the cached graph must not reload the
    module.
    """
    jax = _import_jax()
    jp = _import_jax_numpy()

    module = wp.get_module(double_func.__module__)
    wp.load_module(module=module, device=device)

    with (
        mock.patch.object(wp._src.context.runtime, "driver_version", (12, 2)),
        mock.patch.object(wp.config, "enable_graph_capture_module_load_by_default", False),
        mock.patch.object(wp._src.context, "force_load", side_effect=AssertionError("capture loaded modules")),
    ):
        for graph_mode in (wp.JaxCallableGraphMode.WARP_STAGED, wp.JaxCallableGraphMode.WARP_STAGED_EX):
            with test.subTest(graph_mode=graph_mode):
                jax_double = wp.jax_callable(
                    double_func,
                    graph_mode=graph_mode,
                    module_preload_mode=wp.JaxModulePreloadMode.NONE,
                )
                run = jax.jit(lambda value, jax_double=jax_double: jax_double(value)[0])

                with jax.default_device(wp.device_to_jax(device)):
                    x = jp.arange(32, dtype=jp.float32)
                    y = run(x)
                    jax.block_until_ready(y)

                    with mock.patch.object(module, "load", side_effect=AssertionError("replay reloaded the module")):
                        y = run(x)
                        jax.block_until_ready(y)

                assert_np_equal(np.asarray(y), 2.0 * np.arange(32, dtype=np.float32))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
def test_ffi_jax_mixed_devices(test, device):
    """Test dispatching the same jitted FFI wrappers alternately to CPU and CUDA inputs."""
    jax = _import_jax()
    test.assertTrue(device.is_cuda)

    cpu = wp.get_device("cpu")
    jax_triple = wp.jax_kernel(triple_kernel)
    jax_double = wp.jax_callable(double_func)

    @jax.jit
    def run(x):
        (tripled,) = jax_triple(x)
        (doubled,) = jax_double(x)
        return tripled, doubled

    expected = np.arange(32, dtype=np.float32)
    for target in (cpu, device, cpu, device):
        with test.subTest(target=target):
            x = jax.device_put(expected, wp.device_to_jax(target))
            tripled, doubled = run(x)
            jax.block_until_ready((tripled, doubled))
            assert_np_equal(np.asarray(tripled), 3.0 * expected)
            assert_np_equal(np.asarray(doubled), 2.0 * expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
def test_ffi_jax_cuda_requires_cuda_support(test, device):
    """Test that FFI calls on CUDA arrays report a clear error when Warp lacks CUDA support."""
    jax = _import_jax()
    test.assertTrue(device.is_cuda)

    x = jax.device_put(np.arange(32, dtype=np.float32), wp.device_to_jax(device))
    wrappers = (
        ("jax_kernel", lambda: wp.jax_kernel(triple_kernel)),
        ("jax_callable", lambda: wp.jax_callable(double_func)),
    )

    for name, make_wrapper in wrappers:
        with test.subTest(wrapper=name):
            with mock.patch.object(wp._src.context.runtime, "is_cuda_enabled", False):
                wrapper = make_wrapper()
                run = jax.jit(lambda value, wrapper=wrapper: wrapper(value)[0])
                with test.assertRaisesRegex(Exception, "does not include CUDA support"):
                    jax.block_until_ready(run(x))


def _run_ffi_jax_cpu_subprocess(test):
    """Run the auxiliary CPU FFI script in a CPU-only subprocess and assert that it succeeds."""
    script = os.path.join(os.path.dirname(__file__), "aux_test_jax_cpu_ffi.py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

    with tempfile.TemporaryDirectory(prefix="warp-jax-cpu-ffi-") as cache_dir:
        env["WARP_CACHE_PATH"] = cache_dir
        result = subprocess.run(
            [sys.executable, script],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=300,
        )

    test.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
@unittest.skip(
    "Flaky: race condition in multi-device JAX pmap with FFI - second device output occasionally returns zeros"
)
def test_ffi_jax_callable_pmap_mul(test, device):
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    j = jax_callable(double_func, num_outputs=1)

    ndev = jax.local_device_count()
    per_device = max(ARRAY_SIZE // ndev, 64)
    x = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))

    def per_device_func(v):
        (y,) = j(v)
        return y

    y = jax.pmap(per_device_func)(x)

    jax.block_until_ready(y)

    assert_np_equal(np.asarray(y), 2 * np.asarray(x))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
@unittest.skip(
    "Flaky: race condition in multi-device JAX pmap with FFI - second device output occasionally returns zeros"
)
def test_ffi_jax_callable_pmap_multi_output(test, device):
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    def multi_out_py(
        a: wp.array[float],
        b: wp.array[float],
        s: float,
        c: wp.array[float],
        d: wp.array[float],
    ):
        wp.launch(multi_out_kernel, dim=a.shape, inputs=[a, b, s], outputs=[c, d])

    j = jax_callable(multi_out_py, num_outputs=2)

    ndev = jax.local_device_count()
    per_device = max(ARRAY_SIZE // ndev, 64)
    a = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    b = jp.ones((ndev, per_device), dtype=jp.float32)
    s = 3.0

    def per_device_func(aa, bb):
        c, d = j(aa, bb, s)
        return c + d  # simple combine to exercise both outputs

    out = jax.pmap(per_device_func)(a, b)

    jax.block_until_ready(out)

    a_np = np.arange(ndev * per_device, dtype=np.float32).reshape((ndev, per_device))
    b_np = np.ones((ndev, per_device), dtype=np.float32)
    ref = (a_np + b_np) + s * a_np
    assert_np_equal(np.asarray(out), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
@unittest.skip(
    "Flaky: race condition in multi-device JAX pmap with FFI - second device output occasionally returns zeros"
)
def test_ffi_jax_callable_pmap_multi_stage(test, device):
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    def multi_stage_py(
        a: wp.array[float],
        b: wp.array[float],
        alpha: float,
        tmp: wp.array[float],
        out: wp.array[float],
    ):
        wp.launch(add_kernel, dim=a.shape, inputs=[a, b], outputs=[tmp])
        wp.launch(axpy_kernel, dim=a.shape, inputs=[tmp, b, alpha], outputs=[out])

    j = jax_callable(multi_stage_py, num_outputs=2)

    ndev = jax.local_device_count()
    per_device = max(ARRAY_SIZE // ndev, 64)
    a = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    b = jp.ones((ndev, per_device), dtype=jp.float32)
    alpha = 2.5

    def per_device_func(aa, bb):
        tmp, out = j(aa, bb, alpha)
        return tmp + out

    combined = jax.pmap(per_device_func)(a, b)

    jax.block_until_ready(combined)

    a_np = np.arange(ndev * per_device, dtype=np.float32).reshape((ndev, per_device))
    b_np = np.ones((ndev, per_device), dtype=np.float32)
    tmp_ref = a_np + b_np
    out_ref = alpha * (a_np + b_np) + b_np
    ref = tmp_ref + out_ref
    assert_np_equal(np.asarray(combined), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_callback(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI segfaults intermittently")

    # in-out arguments
    jp = _import_jax_numpy()

    register_ffi_callback = _get_experimental_register_ffi_callback()

    # the Python function to call
    def warp_func(inputs, outputs, attrs, ctx):
        # input arrays
        a = inputs[0]
        b = inputs[1]

        # scalar attributes
        s = attrs["scale"]

        # output arrays
        c = outputs[0]
        d = outputs[1]

        device = wp.device_from_jax(get_jax_device())
        stream = wp.Stream(device, cuda_stream=ctx.stream)

        with wp.ScopedStream(stream):
            # launch with arrays of scalars
            wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])

            # launch with arrays of vec2
            # NOTE: the input shapes are from JAX arrays, we need to strip the inner dimension for vec2 arrays
            wp.launch(scale_vec_kernel, dim=b.shape[0], inputs=[b, s], outputs=[d])

    # register callback
    register_ffi_callback("warp_func", warp_func)

    n = ARRAY_SIZE

    with jax.default_device(wp.device_to_jax(device)):
        # inputs
        a = jp.arange(n, dtype=jp.float32)
        b = jp.arange(n, dtype=jp.float32).reshape((n // 2, 2))  # array of wp.vec2
        s = 2.0

        # set up call
        out_types = [
            jax.ShapeDtypeStruct(a.shape, jp.float32),
            jax.ShapeDtypeStruct(b.shape, jp.float32),  # array of wp.vec2
        ]
        call = jax.ffi.ffi_call("warp_func", out_types)

        # call it
        c, d = call(a, b, scale=s)

    jax.block_until_ready([c, d])

    assert_np_equal(c, 2 * np.arange(ARRAY_SIZE, dtype=np.float32))
    assert_np_equal(d, 2 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2)))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_simple(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_func = jax_kernel(
        scale_sum_square_kernel,
        num_outputs=1,
        enable_backward=True,
    )

    @partial(jax.jit, static_argnames=["s"])
    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.grad(loss, argnums=(0, 1))(a, b, s)

    jax.block_until_ready([da, db])

    # reference gradients
    # d/da sum((a*s + b)^2) = sum(2*(a*s + b) * s)
    # d/db sum((a*s + b)^2) = sum(2*(a*s + b))
    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_jit_of_grad_simple(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_func = jax_kernel(scale_sum_square_kernel, num_outputs=1, enable_backward=True)

    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    grad_fn = jax.grad(loss, argnums=(0, 1))

    # more typical: jit(grad(...)) with static scalar
    jitted_grad = jax.jit(grad_fn, static_argnames=("s",))

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jitted_grad(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_multi_output(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_func = jax_kernel(multi_out_kernel_v3, num_outputs=2, enable_backward=True)

    def caller(fn, a, b, s):
        c, d = fn(a, b, s)
        return jp.sum(c + d)

    @jax.jit
    def grads(a, b, s):
        # mark s as static in the inner call via partial to avoid hashing
        def _inner(a, b, s):
            return caller(jax_func, a, b, s)

        return jax.grad(lambda a, b: _inner(a, b, 2.0), argnums=(0, 1))(a, b)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = grads(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    # d/da sum(c+d) = 2*a + b*s
    ref_da = 2.0 * a_np + b_np * s
    # d/db sum(c+d) = a*s
    ref_db = a_np * s

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_jit_of_grad_multi_output(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_func = jax_kernel(multi_out_kernel_v3, num_outputs=2, enable_backward=True)

    def loss(a, b, s):
        c, d = jax_func(a, b, s)
        return jp.sum(c + d)

    grad_fn = jax.grad(loss, argnums=(0, 1))
    jitted_grad = jax.jit(grad_fn, static_argnames=("s",))

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jitted_grad(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * a_np + b_np * s
    ref_db = a_np * s

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_2d(test, device):
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_func = jax_kernel(inc_2d_kernel, num_outputs=1, enable_backward=True)

    @jax.jit
    def loss(a):
        out = jax_func(a)[0]
        return jp.sum(out)

    n, m = 8, 6
    a = jp.arange(n * m, dtype=jp.float32).reshape((n, m))

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.grad(loss, argnums=(0,))(a)

    jax.block_until_ready(da)

    ref = np.ones((n, m), dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_vec2(test, device):
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_func = jax_kernel(scale_vec_kernel, num_outputs=1, enable_backward=True)

    @partial(jax.jit, static_argnames=("s",))
    def loss(a, s):
        out = jax_func(a, s)[0]
        return jp.sum(out)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32).reshape((n // 2, 2))
    s = 3.0

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.grad(loss, argnums=(0,))(a, s)

    jax.block_until_ready(da)

    # d/da sum(a*s) = s
    ref = np.full_like(np.asarray(a), s)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_mat22(test, device):
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    @wp.kernel
    def scale_mat_kernel(a: wp.array[wp.mat22], s: float, out: wp.array[wp.mat22]):
        tid = wp.tid()
        out[tid] = a[tid] * s

    jax_func = jax_kernel(scale_mat_kernel, num_outputs=1, enable_backward=True)

    @partial(jax.jit, static_argnames=("s",))
    def loss(a, s):
        out = jax_func(a, s)[0]
        return jp.sum(out)

    n = 12  # must be divisible by 4 for 2x2 matrices
    a = jp.arange(n, dtype=jp.float32).reshape((n // 4, 2, 2))
    s = 2.5

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.grad(loss, argnums=(0,))(a, s)

    jax.block_until_ready(da)

    ref = np.full((n // 4, 2, 2), s, dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_static_required(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    # Require explicit static_argnames for scalar s
    jax_func = jax_kernel(scale_sum_square_kernel, num_outputs=1, enable_backward=True)

    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 1.5

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.grad(loss, argnums=(0, 1))(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_pmap_triple(test, device):
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_mul = jax_kernel(triple_kernel, num_outputs=1, enable_backward=True)

    ndev = jax.local_device_count()
    per_device = ARRAY_SIZE // ndev
    x = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))

    def per_device_loss(x):
        y = jax_mul(x)[0]
        return jp.sum(y)

    grads = jax.pmap(jax.grad(per_device_loss))(x)

    jax.block_until_ready(grads)

    assert_np_equal(np.asarray(grads), np.full((ndev, per_device), 3.0, dtype=np.float32))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
@unittest.skip(
    "Flaky: race condition in multi-device JAX pmap with FFI - second device output occasionally returns zeros"
)
def test_ffi_jax_kernel_autodiff_pmap_multi_output(test, device):
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_mo = jax_kernel(multi_out_kernel_v2, num_outputs=2, enable_backward=True)

    ndev = jax.local_device_count()
    per_device = ARRAY_SIZE // ndev
    a = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    b = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    s = 2.0

    def per_dev_loss(aa, bb):
        c, d = jax_mo(aa, bb, s)
        return jp.sum(c + d)

    da, db = jax.pmap(jax.grad(per_dev_loss, argnums=(0, 1)))(a, b)

    jax.block_until_ready([da, db])

    a_np = np.arange(ndev * per_device, dtype=np.float32).reshape((ndev, per_device))
    b_np = np.arange(ndev * per_device, dtype=np.float32).reshape((ndev, per_device))
    ref_da = 2.0 * a_np + b_np * s
    ref_db = a_np * s
    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


# --- launch_dims + enable_backward=True tests --------------------


@wp.kernel
def scale_extra_axis_kernel(
    a: wp.array4d[wp.float32],
    b: wp.array4d[wp.float32],
):
    """3-D tid but 4-D array; outer axis iterated inside the kernel body."""
    i, j, k = wp.tid()
    for m in range(a.shape[0]):
        b[m, i, j, k] = a[m, i, j, k] * 2.0


@wp.kernel
def scale_outer_2d_kernel(
    a: wp.array3d[wp.float32],
    b: wp.array3d[wp.float32],
):
    """2-D tid with an outer axis iterated inside; used for the vmap test."""
    i, j = wp.tid()
    for m in range(a.shape[0]):
        b[m, i, j] = a[m, i, j] * 2.0


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_launch_dims_autodiff_basic(test, device):
    """launch_dims is accepted with enable_backward=True."""
    jax = _import_jax()
    jnp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    N = 8
    with jax.default_device(wp.device_to_jax(device)):
        jax_func = jax_kernel(
            scale_extra_axis_kernel,
            num_outputs=1,
            launch_dims=(N, N, N),
            enable_backward=True,
        )

        a = jnp.ones((4, N, N, N), dtype=jnp.float32)
        out = jax_func(a)
        if isinstance(out, (list, tuple)):
            out = out[0]

        expected = 2.0 * np.ones((4, N, N, N), dtype=np.float32)
        assert_np_equal(np.asarray(out), expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_launch_dims_autodiff_gradient(test, device):
    """Gradient matches the analytical value when launch_dims is explicit.

    Without this fix, auto-inference returns the full 4-D shape and the
    adjoint kernel over-accumulates by a factor equal to the outer axis
    size via atomic_add. Explicit launch_dims, shared between forward and
    adjoint via the enclosing closure, prevents this.
    """
    jax = _import_jax()
    jnp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    BATCH, N = 4, 8
    with jax.default_device(wp.device_to_jax(device)):
        jax_func = jax_kernel(
            scale_extra_axis_kernel,
            num_outputs=1,
            launch_dims=(N, N, N),
            enable_backward=True,
        )

        a = jnp.ones((BATCH, N, N, N), dtype=jnp.float32)

        def loss(x):
            """Sum of the kernel's output; analytical gradient is 2.0 everywhere."""
            y = jax_func(x)
            if isinstance(y, (list, tuple)):
                y = y[0]
            return jnp.sum(y)

        grad = jax.grad(loss)(a)

        expected = 2.0 * np.ones((BATCH, N, N, N), dtype=np.float32)
        assert_np_equal(np.asarray(grad), expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_launch_dims_autodiff_separate_cache(test, device):
    """Wrapping the same kernel with different launch_dims must not share
    wrappers.

    Regression guard for the _FFI_DIFF_KERNEL_REGISTRY cache key: without
    launch_dims in the key, the second wrapper silently reuses the first
    wrapper's closure, so the launch_dims passed to the second call is
    ignored at runtime.
    """
    jax = _import_jax()
    jnp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    with jax.default_device(wp.device_to_jax(device)):
        ffi_small = jax_kernel(
            scale_extra_axis_kernel,
            num_outputs=1,
            launch_dims=(4, 4, 4),
            enable_backward=True,
        )
        ffi_large = jax_kernel(
            scale_extra_axis_kernel,
            num_outputs=1,
            launch_dims=(8, 8, 8),
            enable_backward=True,
        )

        a_small = jnp.ones((2, 4, 4, 4), dtype=jnp.float32)
        a_large = jnp.ones((2, 8, 8, 8), dtype=jnp.float32)

        out_small = ffi_small(a_small)
        out_large = ffi_large(a_large)
        if isinstance(out_small, (list, tuple)):
            out_small = out_small[0]
        if isinstance(out_large, (list, tuple)):
            out_large = out_large[0]

        test.assertEqual(out_small.shape, (2, 4, 4, 4))
        test.assertEqual(out_large.shape, (2, 8, 8, 8))
        assert_np_equal(
            np.asarray(out_small),
            2.0 * np.ones((2, 4, 4, 4), dtype=np.float32),
        )
        assert_np_equal(
            np.asarray(out_large),
            2.0 * np.ones((2, 8, 8, 8), dtype=np.float32),
        )

        # Also exercise the adjoint path: if the cache key were missing
        # launch_dims, the second wrapper would silently reuse the first
        # wrapper's closure and the gradient would be computed with the
        # wrong launch dimensions (atomic_add over-accumulation).
        def loss_small(x):
            """Sum; analytical gradient is 2.0 everywhere."""
            y = ffi_small(x)
            if isinstance(y, (list, tuple)):
                y = y[0]
            return jnp.sum(y)

        grad_small = jax.grad(loss_small)(a_small)
        assert_np_equal(
            np.asarray(grad_small),
            2.0 * np.ones((2, 4, 4, 4), dtype=np.float32),
        )

        def loss_large(x):
            """Sum; analytical gradient is 2.0 everywhere."""
            y = ffi_large(x)
            if isinstance(y, (list, tuple)):
                y = y[0]
            return jnp.sum(y)

        grad_large = jax.grad(loss_large)(a_large)
        assert_np_equal(
            np.asarray(grad_large),
            2.0 * np.ones((2, 8, 8, 8), dtype=np.float32),
        )


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_per_call_override_rejected(test, device):
    """Passing FfiKernel-style per-call kwargs to a differentiable wrapper raises TypeError."""
    jnp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    @wp.kernel
    def noop(a: wp.array1d[wp.float32], b: wp.array1d[wp.float32]):
        i = wp.tid()
        b[i] = a[i]

    N = 8
    jax_func = jax_kernel(noop, num_outputs=1, launch_dims=(N,), enable_backward=True)
    a = jnp.ones((N,), dtype=jnp.float32)

    with test.assertRaisesRegex(TypeError, "launch_dims cannot be overridden per-call"):
        jax_func(a, launch_dims=(N,))

    with test.assertRaisesRegex(TypeError, "output_dims is not supported"):
        jax_func(a, output_dims={"b": (N,)})

    with test.assertRaisesRegex(TypeError, "vmap_method cannot be overridden per-call"):
        jax_func(a, vmap_method="sequential")


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_output_dims_autodiff_still_blocked(test, device):
    """output_dims with enable_backward=True remains a follow-up (still blocked)."""
    jax_kernel = wp.jax_kernel

    @wp.kernel
    def noop(a: wp.array1d[wp.float32], b: wp.array1d[wp.float32]):
        """Identity copy; used only to trigger the output_dims guard path."""
        i = wp.tid()
        b[i] = a[i]

    with test.assertRaises(NotImplementedError):
        jax_kernel(
            noop,
            num_outputs=1,
            output_dims={"b": (8,)},
            enable_backward=True,
        )


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_launch_dims_autodiff_vmap(test, device):
    """launch_dims + enable_backward=True composes with jax.vmap.

    The user-supplied launch_dims fixes the inner (kernel tid) iteration
    space, and jax.vmap prefixes an additional outer axis which the FFI
    layer handles via collapse_batch_dims/compute_batch_size. The two
    operate on disjoint axes, so forward values and gradients remain
    correct under vmap.
    """
    jax = _import_jax()
    jnp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    OUTER, INNER, N = 3, 4, 8
    with jax.default_device(wp.device_to_jax(device)):
        jax_func = jax_kernel(
            scale_outer_2d_kernel,
            num_outputs=1,
            launch_dims=(N, N),
            enable_backward=True,
        )
        batched = jax.vmap(jax_func)

        a = jnp.ones((OUTER, INNER, N, N), dtype=jnp.float32)
        out = batched(a)
        if isinstance(out, (list, tuple)):
            out = out[0]

        expected = 2.0 * np.ones((OUTER, INNER, N, N), dtype=np.float32)
        assert_np_equal(np.asarray(out), expected)

        def loss(x):
            """Sum over all axes; analytical gradient is 2.0 everywhere."""
            y = batched(x)
            if isinstance(y, (list, tuple)):
                y = y[0]
            return jnp.sum(y)

        grad = jax.grad(loss)(a)
        assert_np_equal(np.asarray(grad), expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_vmap_add(test, device, vmap_method):
    """Test basic batching over different input and output axes."""
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    jax_kernel = wp.jax_kernel

    # jax reference implementation
    def jax_add(a, b):
        return a + b

    # warp callable 1d
    def warp_add1d(a: wp.array[float], b: wp.array[float], output: wp.array[float]):
        wp.launch(add_kernel, dim=a.shape, inputs=[a, b, output])

    # warp callable 2d
    def warp_add2d(a: wp.array2d[float], b: wp.array2d[float], output: wp.array2d[float]):
        wp.launch(add2d_kernel, dim=a.shape, inputs=[a, b, output])

    jk_add1d = jax_kernel(add_kernel, vmap_method=vmap_method)
    jk_add2d = jax_kernel(add2d_kernel, vmap_method=vmap_method)
    jc_add1d = jax_callable(warp_add1d, vmap_method=vmap_method)
    jc_add2d = jax_callable(warp_add2d, vmap_method=vmap_method)

    with jax.default_device(wp.device_to_jax(device)):
        # test 1d batching
        a = jp.arange(3 * 4, dtype=jp.float32).reshape((3, 4))
        b = jp.ones(3 * 4, dtype=jp.float32).reshape((3, 4))
        for in_axis in range(2):
            for out_axis in range(2):
                expected = jax.jit(jax.vmap(jax_add, in_axes=in_axis, out_axes=out_axis))(a, b)

                # test jax_kernel()
                (output,) = jax.jit(jax.vmap(jk_add1d, in_axes=in_axis, out_axes=out_axis))(a, b)
                test.assertEqual(output.shape, expected.shape)
                assert_np_equal(np.asarray(output), np.asarray(expected))

                # test jax_callable()
                (output,) = jax.jit(jax.vmap(jc_add1d, in_axes=in_axis, out_axes=out_axis))(a, b)
                test.assertEqual(output.shape, expected.shape)
                assert_np_equal(np.asarray(output), np.asarray(expected))

        # test 2d batching
        a = jp.arange(2 * 3 * 4, dtype=jp.float32).reshape((2, 3, 4))
        b = jp.ones(2 * 3 * 4, dtype=jp.float32).reshape((2, 3, 4))
        for in_axis in range(3):
            for out_axis in range(3):
                expected = jax.jit(jax.vmap(jax_add, in_axes=in_axis, out_axes=out_axis))(a, b)

                # test jax_kernel()
                (output,) = jax.jit(jax.vmap(jk_add2d, in_axes=in_axis, out_axes=out_axis))(a, b)
                test.assertEqual(output.shape, expected.shape)
                assert_np_equal(np.asarray(output), np.asarray(expected))

                # test jax_callable()
                (output,) = jax.jit(jax.vmap(jc_add2d, in_axes=in_axis, out_axes=out_axis))(a, b)
                test.assertEqual(output.shape, expected.shape)
                assert_np_equal(np.asarray(output), np.asarray(expected))


@wp.kernel
def rowsum_kernel(matrix: wp.array2d[float], sums: wp.array1d[float]):
    i, j = wp.tid()
    wp.atomic_add(sums, i, matrix[i, j])


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_vmap_rowsum(test, device, vmap_method):
    """Test in-out arguments with vmap."""
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    jax_kernel = wp.jax_kernel

    # jax reference implementation
    def jax_rowsum(matrix):
        return jp.sum(matrix, axis=1)

    # warp callable
    def warp_rowsum(matrix: wp.array2d[float], sums: wp.array1d[float]):
        wp.launch(rowsum_kernel, dim=matrix.shape, inputs=[matrix, sums])

    jk_rowsum = jax_kernel(rowsum_kernel, in_out_argnames=["sums"], vmap_method=vmap_method)
    jc_rowsum = jax_callable(warp_rowsum, in_out_argnames=["sums"], vmap_method=vmap_method)

    with jax.default_device(wp.device_to_jax(device)):
        # batched input with shape (2, 3, 4)
        matrices = jp.arange(2 * 3 * 4, dtype=jp.float32).reshape((2, 3, 4))

        # NOTE: need to pass zeroed sum arrays whose shape depends on the input batch dimension.

        # --------------------------------------------------------------
        # batch dim 0: 2 matrices with shape (3, 4), output shape (2, 3)
        expected = jax.jit(jax.vmap(jax_rowsum, in_axes=0))(matrices)
        sums = jp.zeros((2, 3), dtype=jp.float32)

        # test jax_kernel()
        (output,) = jax.jit(jax.vmap(jk_rowsum, in_axes=(0, 0)))(matrices, sums)
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))

        # test jax_callable()
        (output,) = jax.jit(jax.vmap(jc_rowsum, in_axes=(0, 0)))(matrices, sums)
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))

        # --------------------------------------------------------------
        # batch dim 1: 3 matrices with shape (2, 4), output shape (3, 2)
        expected = jax.jit(jax.vmap(jax_rowsum, in_axes=1))(matrices)
        sums = jp.zeros((3, 2), dtype=jp.float32)

        # test jax_kernel()
        (output,) = jax.jit(jax.vmap(jk_rowsum, in_axes=(1, 0)))(matrices, sums)
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))

        # test jax_callable()
        (output,) = jax.jit(jax.vmap(jc_rowsum, in_axes=(1, 0)))(matrices, sums)
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))

        # --------------------------------------------------------------
        # batch dim 2: 4 matrices with shape (2, 3), output shape (4, 2)
        expected = jax.jit(jax.vmap(jax_rowsum, in_axes=2))(matrices)
        sums = jp.zeros((4, 2), dtype=jp.float32)

        # test jax_kernel()
        (output,) = jax.jit(jax.vmap(jk_rowsum, in_axes=(2, 0)))(matrices, sums)
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))

        # test jax_callable()
        (output,) = jax.jit(jax.vmap(jc_rowsum, in_axes=(2, 0)))(matrices, sums)
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))


@wp.kernel
def lookup_kernel(table: wp.array[float], indices: wp.array[int], output: wp.array[float]):
    i = wp.tid()
    output[i] = table[indices[i]]


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_vmap_lookup(test, device, vmap_method):
    """
    Test the following vmap features:
    - Unbatched inputs (lookup table).
    - Custom launch and output dimensions for kernels (not inferred from argument shape).
    - Custom output dimensions for callables.
    """
    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_callable = wp.jax_callable

    jax_kernel = wp.jax_kernel

    # jax reference implementation
    def jax_lookup(a, indices):
        return a[indices]

    def warp_lookup(table: wp.array[float], indices: wp.array[int], output: wp.array[float]):
        wp.launch(lookup_kernel, dim=indices.shape, inputs=[table, indices, output])

    jk_lookup = jax_kernel(lookup_kernel, vmap_method=vmap_method)
    jc_lookup = jax_callable(warp_lookup, vmap_method=vmap_method)

    with jax.default_device(wp.device_to_jax(device)):
        # lookup table (not batched)
        N = 100
        table = jp.arange(N, dtype=jp.float32)

        # batched indices to look up
        key = jax.random.key(42)
        indices = jax.random.randint(key, (3, 5), 0, N, dtype=jp.int32)

        # NOTE: use functools.partial() to pass output_dims (will be batched)

        # ----------------------------------------------------------
        # batch dim 0: 3 sets of 5 indices each, output shape (3, 5)
        expected = jax.jit(jax.vmap(jax_lookup, in_axes=(None, 0)))(table, indices)

        # test jax_kernel()
        (output,) = jax.jit(jax.vmap(partial(jk_lookup, launch_dims=indices.shape[1]), in_axes=(None, 0)))(
            table, indices
        )
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))

        # test jax_callable()
        (output,) = jax.jit(jax.vmap(partial(jc_lookup, output_dims=indices.shape[1]), in_axes=(None, 0)))(
            table, indices
        )
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))

        # ----------------------------------------------------------
        # batch dim 1: 5 sets of 3 indices each, output shape (5, 3)
        expected = jax.jit(jax.vmap(jax_lookup, in_axes=(None, 1)))(table, indices)

        # test jax_kernel()
        (output,) = jax.jit(jax.vmap(partial(jk_lookup, launch_dims=indices.shape[0]), in_axes=(None, 1)))(
            table, indices
        )
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))

        # test jax_callable()
        (output,) = jax.jit(jax.vmap(partial(jc_lookup, output_dims=indices.shape[0]), in_axes=(None, 1)))(
            table, indices
        )
        test.assertEqual(output.shape, expected.shape)
        assert_np_equal(np.asarray(output), np.asarray(expected))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_subscript_scalar(test, device):
    """Test jax_kernel with wp.array[float] subscript syntax for scalar dtypes."""
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_add = jax_kernel(add_kernel_subscript)

    @jax.jit
    def f():
        n = ARRAY_SIZE
        a = jp.arange(n, dtype=jp.float32)
        b = jp.ones(n, dtype=jp.float32)
        return jax_add(a, b)

    with jax.default_device(wp.device_to_jax(device)):
        (y,) = f()

    jax.block_until_ready(y)

    result = np.asarray(y)
    expected = np.arange(1, ARRAY_SIZE + 1, dtype=np.float32)

    assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_subscript_vec(test, device):
    """Test jax_kernel with wp.array[wp.vec2] subscript syntax for vector dtypes."""
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_scale_vec = jax_kernel(scale_vec_kernel_subscript)

    @jax.jit
    def f():
        a = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))
        s = 2.0
        return jax_scale_vec(a, s)

    with jax.default_device(wp.device_to_jax(device)):
        (b,) = f()

    jax.block_until_ready(b)

    expected = 2 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(b, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_subscript_autodiff(test, device):
    """Test jax_kernel with subscript syntax and enable_backward=True."""
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    jax = _import_jax()
    jp = _import_jax_numpy()

    jax_kernel = wp.jax_kernel

    jax_func = jax_kernel(
        scale_sum_square_kernel_subscript,
        num_outputs=1,
        enable_backward=True,
    )

    @partial(jax.jit, static_argnames=["s"])
    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.grad(loss, argnums=(0, 1))(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


def test_bf16_interop_jax(test, device):
    jnp = _import_jax_numpy()

    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    wp_arr = wp.array(input_data, dtype=wp.bfloat16, device=device)
    jax_arr = wp.to_jax(wp_arr)
    test.assertEqual(jax_arr.dtype, jnp.bfloat16)

    # Verify values survived the conversion
    np.testing.assert_allclose(np.asarray(jax_arr, dtype=np.float32), input_data, rtol=1e-2)

    # Reverse direction: JAX bfloat16 -> Warp bfloat16
    jax_bf16 = jnp.array(input_data, dtype=jnp.bfloat16)
    wp_from_jax = wp.from_jax(jax_bf16, dtype=wp.bfloat16)
    test.assertEqual(wp_from_jax.dtype, wp.bfloat16)
    np.testing.assert_allclose(wp_from_jax.numpy().astype(np.float32), input_data, rtol=1e-2)


class TestJax(unittest.TestCase):
    def _get_jax_cpu_device(self):
        jax = _import_jax()
        try:
            return jax.devices("cpu")[0]
        except RuntimeError as e:
            self.skipTest(f"JAX CPU backend is unavailable: {e}")

    @unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
    def test_ffi_jax_kernel_cpu_shaped_tile_store(self):
        """Test jax_kernel on CPU with a kernel that stores a fixed-shape tile."""
        jax = _import_jax()
        jax_cpu = self._get_jax_cpu_device()
        jax_tile_store = wp.jax_kernel(
            shaped_tile_store_kernel,
            launch_dims=1,
            output_dims=TILE_STORE_SIZE,
        )

        with jax.default_device(jax_cpu):
            (result,) = jax.jit(jax_tile_store)()

        jax.block_until_ready(result)
        assert_np_equal(np.asarray(result), np.ones(TILE_STORE_SIZE, dtype=np.float32))

    @unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
    def test_ffi_jax_callable_cpu_graph_modes(self):
        """Test that the NONE and JAX graph modes run on CPU while CUDA-only graph modes raise."""
        jax = _import_jax()
        jp = _import_jax_numpy()
        jax_cpu = self._get_jax_cpu_device()

        for graph_mode in (wp.JaxCallableGraphMode.NONE, wp.JaxCallableGraphMode.JAX):
            with self.subTest(graph_mode=graph_mode), jax.default_device(jax_cpu):
                jax_double = wp.jax_callable(double_func, graph_mode=graph_mode)
                x = jp.arange(32, dtype=jp.float32)
                (y,) = jax.jit(jax_double)(x)
                jax.block_until_ready(y)
                assert_np_equal(np.asarray(y), 2.0 * np.arange(32, dtype=np.float32))

        cuda_only_modes = (
            wp.JaxCallableGraphMode.WARP,
            wp.JaxCallableGraphMode.WARP_STAGED,
            wp.JaxCallableGraphMode.WARP_STAGED_EX,
        )
        for graph_mode in cuda_only_modes:
            with self.subTest(graph_mode=graph_mode), jax.default_device(jax_cpu):
                jax_double = wp.jax_callable(double_func, graph_mode=graph_mode)
                x = jp.arange(32, dtype=jp.float32)
                with self.assertRaisesRegex(Exception, rf"{graph_mode.name}.*CPU"):
                    y = jax.jit(jax_double)(x)
                    jax.block_until_ready(y)

    @unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
    def test_ffi_jax_callable_integer_graph_mode(self):
        """Test that an integer graph_mode argument is converted to JaxCallableGraphMode."""

        def func(x: wp.array[float], y: wp.array[float]):
            wp.launch(double_kernel, dim=x.shape, inputs=[x], outputs=[y])

        jax_func = wp.jax_callable(
            func,
            graph_mode=2,
            module_preload_mode=wp.JaxModulePreloadMode.NONE,
        )

        self.assertIs(jax_func.graph_mode, wp.JaxCallableGraphMode.WARP)

    @unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
    def test_ffi_jax_module_load_failure(self):
        """Test that a failed Warp module load surfaces as a clear FFI error."""
        jax = _import_jax()
        jax_cpu = self._get_jax_cpu_device()
        device = wp.get_device("cpu")
        module = triple_kernel.module
        self.assertIsNotNone(module.load(device, 1))

        wrappers = (
            (
                "jax_kernel",
                wp.jax_kernel(triple_kernel, module_preload_mode=wp.JaxModulePreloadMode.NONE),
            ),
            (
                "jax_callable",
                wp.jax_callable(double_func, module_preload_mode=wp.JaxModulePreloadMode.NONE),
            ),
        )
        x = jax.device_put(np.arange(8, dtype=np.float32), jax_cpu)

        for name, wrapper in wrappers:
            with self.subTest(wrapper=name):
                run = jax.jit(lambda value, wrapper=wrapper: wrapper(value)[0])
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    mock.patch.object(module, "load", return_value=None),
                    self.assertRaisesRegex(Exception, "Failed to load Warp module"),
                ):
                    y = run(x)
                    jax.block_until_ready(y)

    @unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
    def test_ffi_jax_cpu_requires_cpu_support(self):
        """Test that FFI calls on CPU arrays report a clear error when Warp lacks CPU support."""
        jax = _import_jax()
        jax_cpu = self._get_jax_cpu_device()
        x = jax.device_put(np.arange(8, dtype=np.float32), jax_cpu)
        wrappers = (
            (
                "jax_kernel",
                wp.jax_kernel(triple_kernel, module_preload_mode=wp.JaxModulePreloadMode.NONE),
            ),
            (
                "jax_callable",
                wp.jax_callable(double_func, module_preload_mode=wp.JaxModulePreloadMode.NONE),
            ),
        )

        for name, wrapper in wrappers:
            with self.subTest(wrapper=name):
                run = jax.jit(lambda value, wrapper=wrapper: wrapper(value)[0])
                with (
                    mock.patch.object(wp, "is_cpu_available", return_value=False),
                    self.assertRaisesRegex(Exception, "does not include CPU support"),
                ):
                    y = run(x)
                    jax.block_until_ready(y)

    @unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
    def test_ffi_module_preload_modes(self):
        """Test _preload_ffi_module device selection for each JaxModulePreloadMode.

        NONE must not load anything, CURRENT_DEVICE loads the Warp device mapped from the
        default JAX device and skips devices Warp cannot map, and ALL_DEVICES queries the CPU
        and CUDA backends while tolerating backends that are unavailable.
        """
        from warp._src.jax import ffi as ffi_module  # noqa: PLC0415

        with self.subTest(mode="none"):
            module = _RecordingFfiModule()
            ffi_module._preload_ffi_module(module, wp.JaxModulePreloadMode.NONE)
            self.assertEqual(module.loaded_devices, [])

        jax_device = object()
        warp_device = mock.Mock(is_cpu=False)
        module = _RecordingFfiModule()
        with self.subTest(mode="current_device"):
            with (
                mock.patch.object(ffi_module, "get_jax_device", return_value=jax_device),
                mock.patch.object(ffi_module.wp, "device_from_jax", return_value=warp_device),
            ):
                ffi_module._preload_ffi_module(module, wp.JaxModulePreloadMode.CURRENT_DEVICE)
            self.assertEqual(module.loaded_devices, [warp_device])

        with self.subTest(mode="current_device_unavailable"):
            module = _RecordingFfiModule()
            with (
                mock.patch.object(ffi_module, "get_jax_device", return_value=jax_device),
                mock.patch.object(ffi_module.wp, "device_from_jax", side_effect=RuntimeError("Device unavailable")),
            ):
                ffi_module._preload_ffi_module(module, wp.JaxModulePreloadMode.CURRENT_DEVICE)
            self.assertEqual(module.loaded_devices, [])

        cpu_0 = object()
        cpu_1 = object()
        requested_backends = []

        class FakeJax:
            @staticmethod
            def local_devices(process_index=None, backend=None, host_id=None):
                requested_backends.append(backend)
                if backend == "cpu":
                    return [cpu_0, cpu_1]
                if backend == "cuda":
                    raise RuntimeError("CUDA backend is unavailable")
                raise AssertionError(f"Unexpected backend: {backend}")

        warp_cpu = mock.Mock(is_cpu=True)
        module = _RecordingFfiModule()
        with self.subTest(mode="all_devices"):
            with (
                mock.patch.object(ffi_module, "_get_jax", return_value=FakeJax()),
                mock.patch.object(ffi_module.wp, "device_from_jax", return_value=warp_cpu),
            ):
                ffi_module._preload_ffi_module(module, wp.JaxModulePreloadMode.ALL_DEVICES)
            self.assertEqual(module.loaded_devices, [warp_cpu])
            self.assertIn("cuda", requested_backends)

    @unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
    def test_ffi_module_preload_load_error(self):
        """Test that module preloading propagates load errors and reports previously failed builds.

        A load that raises must propagate to the caller, and a load that returns no executable
        module (a cached build failure) must raise a RuntimeError for both the CURRENT_DEVICE
        and ALL_DEVICES preload modes.
        """
        from warp._src.jax import ffi as ffi_module  # noqa: PLC0415

        jax_device = object()
        warp_device = mock.Mock(is_cpu=False)
        module = _RecordingFfiModule(load_error=ValueError("module load failed"))
        with (
            mock.patch.object(ffi_module, "get_jax_device", return_value=jax_device),
            mock.patch.object(ffi_module.wp, "device_from_jax", return_value=warp_device),
            self.assertRaisesRegex(ValueError, "module load failed"),
        ):
            ffi_module._preload_ffi_module(module, wp.JaxModulePreloadMode.CURRENT_DEVICE)

        self.assertEqual(module.loaded_devices, [warp_device])

        with self.subTest(cached_failure="current_device"):
            module = _RecordingFfiModule(load_result=None)
            with (
                mock.patch.object(ffi_module, "get_jax_device", return_value=jax_device),
                mock.patch.object(ffi_module.wp, "device_from_jax", return_value=warp_device),
                self.assertRaisesRegex(RuntimeError, "previous build failure"),
            ):
                ffi_module._preload_ffi_module(module, wp.JaxModulePreloadMode.CURRENT_DEVICE)

        class FakeJax:
            @staticmethod
            def local_devices(process_index=None, backend=None, host_id=None):
                return [jax_device] if backend == "cpu" else []

        with self.subTest(cached_failure="all_devices"):
            module = _RecordingFfiModule(load_result=None)
            with (
                mock.patch.object(ffi_module, "_get_jax", return_value=FakeJax()),
                mock.patch.object(ffi_module.wp, "device_from_jax", return_value=warp_device),
                self.assertRaisesRegex(RuntimeError, "previous build failure"),
            ):
                ffi_module._preload_ffi_module(module, wp.JaxModulePreloadMode.ALL_DEVICES)

    @unittest.skipUnless(_jax_version() >= (0, 5, 0), "JAX version too old")
    def test_ffi_jax_cpu_subprocess(self):
        """Test FFI wrappers on a CPU-only JAX runtime configured with two host platform devices.

        JAX_PLATFORMS and XLA_FLAGS must be set before JAX initializes, so the checks run in a
        separate process via ``aux_test_jax_cpu_ffi.py``.
        """
        _run_ffi_jax_cpu_subprocess(self)


# try adding Jax tests if Jax is installed correctly
try:
    import jax
except Exception as error:
    print(f"Skipping JAX tests due to exception: {error}")
else:
    # NOTE: we must enable 64-bit types in Jax to test the full gamut of types
    jax.config.update("jax_enable_x64", True)

    jax_candidate_devices = get_test_devices()
    jax_cuda_candidate_devices = [device for device in jax_candidate_devices if device.is_cuda]
    jax_cpu_candidate_devices = [device for device in jax_candidate_devices if device.is_cpu]

    # pmap tests dispatch across all local JAX devices; register them when those map onto
    # candidate test devices and defer the JAX availability probe to run time.
    try:
        pmap_warp_devices = [wp.device_from_jax(d) for d in jax.local_devices()]
    except (IndexError, RuntimeError):
        pmap_warp_devices = []
    pmap_devices_are_candidates = bool(pmap_warp_devices) and all(d in jax_candidate_devices for d in pmap_warp_devices)

    @cache
    def _jax_device_error(device_alias):
        device = wp.get_device(device_alias)
        try:
            with jax.default_device(wp.device_to_jax(device)):
                array = jax.numpy.arange(10, dtype=jax.numpy.float32)
                array += 1
            jax.block_until_ready(array)
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return None

    def _check_jax_device(test, device):
        device = wp.get_device(device)
        error = _jax_device_error(device.alias)
        if error is not None:
            test.skipTest(f"JAX is unavailable on Warp device '{device}': {error}")

    def _check_all_jax_cuda_devices(test, _device):
        for device in jax_cuda_candidate_devices:
            _check_jax_device(test, device)

    def _check_pmap_devices(test, _device):
        for device in pmap_warp_devices:
            _check_jax_device(test, device)

    add_function_test(
        TestJax,
        "test_jax_experimental_import_deprecation",
        test_jax_experimental_import_deprecation,
        devices=None,
    )
    add_function_test(
        TestJax,
        "test_jax_experimental_ffi_import_deprecation",
        test_jax_experimental_ffi_import_deprecation,
        devices=None,
    )
    add_function_test(
        TestJax,
        "test_jax_experimental_custom_call_import_deprecation",
        test_jax_experimental_custom_call_import_deprecation,
        devices=None,
    )
    add_function_test(TestJax, "test_dtype_from_jax", test_dtype_from_jax, devices=None)
    add_function_test(TestJax, "test_dtype_to_jax", test_dtype_to_jax, devices=None)
    if jax_candidate_devices:
        add_function_test(
            TestJax,
            "test_device_conversion",
            test_device_conversion,
            devices=jax_candidate_devices,
            device_check=_check_jax_device,
        )

    if jax.__version_info__ >= (0, 5, 0):
        if jax_candidate_devices:
            # FFI-based tests run on any JAX-compatible device (CPU or CUDA)
            jax_kernel_ffi_tests = (
                test_jax_kernel_basic,
                test_jax_kernel_scalar,
                test_jax_kernel_vecmat,
                test_jax_kernel_multiarg,
                test_jax_kernel_launch_dims,
            )
            backend_neutral_ffi_tests = (
                *jax_kernel_ffi_tests,
                # direct FFI kernels
                test_ffi_jax_kernel_add,
                test_ffi_jax_kernel_sincos,
                test_ffi_jax_kernel_diagonal,
                test_ffi_jax_kernel_in_out,
                test_ffi_jax_kernel_scale_vec_constant,
                test_ffi_jax_kernel_scale_vec_static,
                test_ffi_jax_kernel_launch_dims_default,
                test_ffi_jax_kernel_launch_dims_custom,
                # callables
                test_ffi_jax_callable_scale_constant,
                test_ffi_jax_callable_scale_static,
                test_ffi_jax_callable_in_out,
                # autodiff and subscript annotations
                test_ffi_jax_kernel_autodiff_simple,
                test_ffi_jax_kernel_autodiff_jit_of_grad_simple,
                test_ffi_jax_kernel_autodiff_multi_output,
                test_ffi_jax_kernel_autodiff_jit_of_grad_multi_output,
                test_ffi_jax_kernel_autodiff_2d,
                test_ffi_jax_kernel_autodiff_vec2,
                test_ffi_jax_kernel_autodiff_mat22,
                test_ffi_jax_kernel_autodiff_static_required,
                test_ffi_jax_kernel_launch_dims_autodiff_basic,
                test_ffi_jax_kernel_launch_dims_autodiff_gradient,
                test_ffi_jax_kernel_launch_dims_autodiff_separate_cache,
                test_ffi_jax_kernel_autodiff_per_call_override_rejected,
                test_ffi_jax_kernel_output_dims_autodiff_still_blocked,
                test_ffi_jax_kernel_launch_dims_autodiff_vmap,
                test_ffi_jax_kernel_subscript_scalar,
                test_ffi_jax_kernel_subscript_vec,
                test_ffi_jax_kernel_subscript_autodiff,
            )

            for test_func in backend_neutral_ffi_tests:
                test_name = test_func.__name__
                test_kwargs = {}
                if test_func in jax_kernel_ffi_tests:
                    test_name = f"{test_name}_ffi"
                    test_kwargs["use_ffi"] = True
                add_function_test(
                    TestJax,
                    test_name,
                    test_func,
                    devices=jax_candidate_devices,
                    device_check=_check_jax_device,
                    **test_kwargs,
                )

            for vmap_method in ["broadcast_all", "sequential"]:
                add_function_test(
                    TestJax,
                    f"test_ffi_vmap_add_{vmap_method}",
                    partial(test_ffi_vmap_add, vmap_method=vmap_method),
                    devices=jax_candidate_devices,
                    device_check=_check_jax_device,
                )
                add_function_test(
                    TestJax,
                    f"test_ffi_vmap_rowsum_{vmap_method}",
                    partial(test_ffi_vmap_rowsum, vmap_method=vmap_method),
                    devices=jax_candidate_devices,
                    device_check=_check_jax_device,
                )
                add_function_test(
                    TestJax,
                    f"test_ffi_vmap_lookup_{vmap_method}",
                    partial(test_ffi_vmap_lookup, vmap_method=vmap_method),
                    devices=jax_candidate_devices,
                    device_check=_check_jax_device,
                )

        if pmap_devices_are_candidates:
            pmap_tests = (
                test_ffi_jax_callable_pmap_multi_output,
                test_ffi_jax_callable_pmap_mul,
                test_ffi_jax_callable_pmap_multi_stage,
                test_ffi_jax_kernel_autodiff_pmap_triple,
                test_ffi_jax_kernel_autodiff_pmap_multi_output,
            )
            for test_func in pmap_tests:
                add_function_test(
                    TestJax, test_func.__name__, test_func, devices=None, device_check=_check_pmap_devices
                )

    if jax_cuda_candidate_devices:
        if (0, 4, 25) <= jax.__version_info__ < (0, 8, 0):
            # legacy custom_call path is CUDA-only
            legacy_custom_call_tests = (
                test_jax_kernel_basic,
                test_jax_kernel_scalar,
                test_jax_kernel_vecmat,
                test_jax_kernel_multiarg,
                test_jax_kernel_launch_dims,
            )
            for test_func in legacy_custom_call_tests:
                add_function_test(
                    TestJax,
                    f"{test_func.__name__}_cc",
                    test_func,
                    devices=jax_cuda_candidate_devices,
                    device_check=_check_jax_device,
                    use_ffi=False,
                )

        if jax.__version_info__ >= (0, 5, 0):
            cuda_only_jax_tests = (
                test_ffi_jax_callable_graph_cache,
                test_ffi_jax_callable_graph_replay_skips_module_load,
                test_ffi_jax_cuda_requires_cuda_support,
                test_ffi_callback,
            )
            for test_func in cuda_only_jax_tests:
                add_function_test(
                    TestJax,
                    test_func.__name__,
                    test_func,
                    devices=jax_cuda_candidate_devices,
                    device_check=_check_jax_device,
                )

            if jax_cpu_candidate_devices:
                add_function_test(
                    TestJax,
                    "test_ffi_jax_mixed_devices",
                    test_ffi_jax_mixed_devices,
                    devices=jax_cuda_candidate_devices,
                    device_check=_check_jax_device,
                )

    # bfloat16 tests require arch >= 80
    bf16_jax_devices = [
        device for device in jax_candidate_devices if device.is_cpu or (device.is_cuda and device.arch >= 80)
    ]
    if bf16_jax_devices:
        add_function_test(
            TestJax,
            "test_bf16_interop_jax",
            test_bf16_interop_jax,
            devices=bf16_jax_devices,
            device_check=_check_jax_device,
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
