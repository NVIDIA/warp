# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for wp.ref[T] pass-by-reference and wp.address_of()."""

import unittest

import numpy as np

import warp as wp
from warp._src.codegen import WarpCodegenError, WarpCodegenTypeError
from warp._src.types import Reference
from warp.tests.unittest_utils import add_function_test, get_test_devices


class TestRef(unittest.TestCase):
    """Tests for wp.ref[T] and wp.address_of()."""

    def test_ref_subscript_creates_type(self):
        r = wp.ref[wp.int32]
        self.assertIsInstance(r, Reference)
        self.assertEqual(r.dtype, wp.int32)
        self.assertEqual(r.value_type, wp.int32)
        self.assertEqual(repr(r), "wp.ref[int32]")

    def test_ref_subscript_different_dtypes(self):
        self.assertEqual(wp.ref[wp.float32].dtype, wp.float32)
        self.assertEqual(wp.ref[wp.float64].dtype, wp.float64)
        self.assertEqual(wp.ref[wp.vec3].dtype, wp.vec3)

    def test_ref_distinct_instances(self):
        r1 = wp.ref[wp.int32]
        r2 = wp.ref[wp.float32]
        self.assertNotEqual(r1.dtype, r2.dtype)

    def test_ref_same_dtype_equal(self):
        r1 = wp.ref[wp.int32]
        r2 = wp.ref[wp.int32]
        self.assertEqual(r1.dtype, r2.dtype)
        self.assertEqual(r1, r2)
        self.assertEqual(hash(r1), hash(r2))

    def test_ref_rejects_literal_argument(self):
        """Passing a literal to a wp.ref[T] parameter must raise WarpCodegenError."""

        @wp.func
        def takes_ref(x: wp.ref[wp.int32]):
            x += 1

        with self.assertRaises(WarpCodegenError):

            @wp.kernel(module="unique")
            def bad_literal_kernel():
                takes_ref(wp.int32(5))  # literal - not addressable

            wp.launch(bad_literal_kernel, dim=1)

    def test_ref_rejects_function_return_argument(self):
        """Passing a function-call result to a wp.ref[T] parameter must fail."""

        @wp.func
        def make_val() -> wp.int32:
            return wp.int32(1)

        @wp.func
        def takes_ref2(x: wp.ref[wp.int32]):
            x += 1

        with self.assertRaises(WarpCodegenError):

            @wp.kernel(module="unique")
            def bad_return_kernel(arr: wp.array[wp.int32]):
                takes_ref2(make_val())  # temporary - not addressable

            wp.launch(bad_return_kernel, dim=1, inputs=[wp.zeros(1, dtype=wp.int32)])

    def test_ref_rejects_function_return_struct_field_argument(self):
        """Passing a field of a function-call result to wp.ref[T] must fail."""

        with self.assertRaises(WarpCodegenError):

            @wp.kernel(module="unique", enable_backward=False)
            def bad_return_field_kernel(arr: wp.array[wp.float32]):
                set_ref_float(make_ref_box().value, wp.float32(2.0))
                arr[0] = wp.float32(0.0)

            wp.launch(bad_return_field_kernel, dim=1, outputs=[wp.zeros(1, dtype=wp.float32)])

    def test_ref_rejects_constant_local_argument(self):
        """Passing a compile-time constant local to wp.ref[T] must fail."""

        @wp.func
        def takes_int_ref(x: wp.ref[wp.int32], value: wp.int32):
            x = value

        @wp.func
        def takes_float_ref(x: wp.ref[wp.float32], value: wp.float32):
            x = value

        with self.assertRaisesRegex(WarpCodegenError, r"x = wp\.int32\(0\)"):

            @wp.kernel(module="unique", enable_backward=False)
            def bad_int_constant_ref_kernel():
                x = 0
                takes_int_ref(x, wp.int32(2))

            wp.launch(bad_int_constant_ref_kernel, dim=1)

        with self.assertRaisesRegex(WarpCodegenError, r"x = wp\.float32\(1\.0\)"):

            @wp.kernel(module="unique", enable_backward=False)
            def bad_float_constant_ref_kernel():
                x = 1.0
                takes_float_ref(x, wp.float32(2.0))

            wp.launch(bad_float_constant_ref_kernel, dim=1)

    def test_ref_backward_enabled_kernel_raises(self):
        """Calling a wp.ref[T] @wp.func from a backward-enabled kernel must raise."""

        @wp.func
        def ref_func(x: wp.ref[wp.float32]):
            x += wp.float32(1.0)

        with self.assertRaises(WarpCodegenError):

            @wp.kernel(module="unique")
            def bad_backward_kernel(arr: wp.array[wp.float32]):
                i = wp.tid()
                ref_func(arr[i])

            tape = wp.Tape()
            arr = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            with tape:
                wp.launch(bad_backward_kernel, dim=1, inputs=[arr])
            tape.backward()

    def test_native_ref_without_adj_snippet_backward_raises(self):
        """Calling a wp.ref[T] @wp.func_native without adj_snippet from backward must raise."""

        @wp.func_native("x = x + 1.0f;")
        def native_no_adj(x: wp.ref[wp.float32]): ...

        with self.assertRaises(WarpCodegenError):

            @wp.kernel(module="unique")
            def bad_native_backward_kernel(arr: wp.array[wp.float32]):
                i = wp.tid()
                native_no_adj(arr[i])

            tape = wp.Tape()
            arr = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            with tape:
                wp.launch(bad_native_backward_kernel, dim=1, inputs=[arr])
            tape.backward()

    def test_ref_overload_differing_only_by_refness_rejected(self):
        """Warp function overloads cannot differ only by wp.ref[T]."""

        @wp.func
        def refness_only_overload(x: wp.int32) -> wp.int32:
            return x

        with self.assertRaisesRegex(RuntimeError, "differ only"):

            @wp.func
            def refness_only_overload(x: wp.ref[wp.int32]):
                x += 1

    def test_address_of_rejects_arithmetic_expression(self):
        """wp.address_of() must reject value temporaries."""

        with self.assertRaises(WarpCodegenError):

            @wp.kernel(module="unique", enable_backward=False)
            def bad_address_of_expression_kernel(result: wp.array[wp.uint64]):
                x = wp.int32(1)
                y = wp.int32(2)
                result[0] = wp.address_of(x + y)

            wp.launch(bad_address_of_expression_kernel, dim=1, inputs=[wp.zeros(1, dtype=wp.uint64)])

    def test_address_of_rejects_temporary_vector_component(self):
        """wp.address_of() must reject component extracts from temporaries."""

        with self.assertRaises(WarpCodegenError):

            @wp.kernel(module="unique", enable_backward=False)
            def bad_address_of_component_kernel(result: wp.array[wp.uint64]):
                result[0] = wp.address_of(wp.vec3f(1.0, 2.0, 3.0).x)

            wp.launch(bad_address_of_component_kernel, dim=1, inputs=[wp.zeros(1, dtype=wp.uint64)])

    def test_address_of_rejects_function_return_struct_field(self):
        """wp.address_of() must reject fields of function-call results."""

        with self.assertRaises(WarpCodegenError):

            @wp.kernel(module="unique", enable_backward=False)
            def bad_address_of_return_field_kernel(result: wp.array[wp.uint64]):
                result[0] = wp.address_of(make_ref_box().value)

            wp.launch(bad_address_of_return_field_kernel, dim=1, inputs=[wp.zeros(1, dtype=wp.uint64)])

    def test_address_of_rejects_constant_local(self):
        """wp.address_of() must reject compile-time constant locals."""

        with self.assertRaisesRegex(WarpCodegenError, r"x = wp\.int32\(0\)"):

            @wp.kernel(module="unique", enable_backward=False)
            def bad_int_constant_address_of_kernel(result: wp.array[wp.uint64]):
                x = 0
                result[0] = wp.address_of(x)

            wp.launch(bad_int_constant_address_of_kernel, dim=1, inputs=[wp.zeros(1, dtype=wp.uint64)])

        with self.assertRaisesRegex(WarpCodegenError, r"x = wp\.float32\(1\.0\)"):

            @wp.kernel(module="unique", enable_backward=False)
            def bad_float_constant_address_of_kernel(result: wp.array[wp.uint64]):
                x = 1.0
                result[0] = wp.address_of(x)

            wp.launch(bad_float_constant_address_of_kernel, dim=1, inputs=[wp.zeros(1, dtype=wp.uint64)])

    def test_by_value_unchanged(self):
        """By-value functions still copy - the original is not mutated."""
        result = wp.zeros(2, dtype=wp.int32, device="cpu")
        wp.launch(kernel_by_value_unchanged, dim=1, inputs=[result], device="cpu")
        np.testing.assert_array_equal(result.numpy(), [10, 11])

    def test_ref_augassign_user_operator_result_type_checked(self):
        """User-defined += results must match the wp.ref[T] value type."""

        @wp.func
        def add(x: wp.float32, y: wp.float32) -> wp.int32:
            return wp.int32(1)

        @wp.func
        def bad_ref_augassign(x: wp.ref[wp.float32]):
            x += wp.float32(1.0)

        missing_add = object()
        previous_add = bad_ref_augassign.func.__globals__.get("add", missing_add)
        bad_ref_augassign.func.__globals__["add"] = add
        try:
            with self.assertRaisesRegex(WarpCodegenTypeError, "augmented assignment to ref parameter"):

                @wp.kernel(module="unique", enable_backward=False)
                def bad_ref_augassign_user_operator_type_kernel(out: wp.array[wp.float32]):
                    x = wp.float32(0.0)
                    bad_ref_augassign(x)
                    out[0] = x

                wp.launch(
                    bad_ref_augassign_user_operator_type_kernel,
                    dim=1,
                    outputs=[wp.zeros(1, dtype=wp.float32)],
                )
        finally:
            if previous_add is missing_add:
                del bad_ref_augassign.func.__globals__["add"]
            else:
                bad_ref_augassign.func.__globals__["add"] = previous_add


# -----------------------------------------------------------------------
# Kernel-level tests (require compilation)
# -----------------------------------------------------------------------


@wp.func
def increment_ref(x: wp.ref[wp.int32]):
    x += 1


@wp.kernel(enable_backward=False)
def kernel_increment_local(result: wp.array[wp.int32]):
    val = wp.int32(0)
    increment_ref(val)
    result[0] = val


def test_ref_mutates_local(test, device):
    """Basic end-to-end: kernel calls @wp.func with wp.ref[T], mutation is visible."""
    result = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(kernel_increment_local, dim=1, inputs=[result], device=device)
    np.testing.assert_array_equal(result.numpy(), [1])


@wp.func
def add_to_ref(x: wp.ref[wp.float32], delta: wp.float32):
    x += delta


@wp.kernel(enable_backward=False)
def kernel_add_to_array_element(arr: wp.array[wp.float32]):
    i = wp.tid()
    add_to_ref(arr[i], wp.float32(10.0))


def test_ref_mutates_array_element(test, device):
    """Passing an array element by reference and mutating it."""
    arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    wp.launch(kernel_add_to_array_element, dim=3, inputs=[arr], device=device)
    np.testing.assert_allclose(arr.numpy(), [11.0, 12.0, 13.0])


@wp.struct
class Counter:
    value: wp.int32


@wp.func
def increment_counter(c: wp.ref[Counter]):
    c.value += 1


@wp.kernel(enable_backward=False)
def kernel_increment_struct_field(result: wp.array[wp.int32]):
    c = Counter()
    c.value = wp.int32(5)
    increment_counter(c)
    result[0] = c.value


def test_ref_mutates_struct_field(test, device):
    """Passing a struct by reference and mutating a field."""
    result = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(kernel_increment_struct_field, dim=1, inputs=[result], device=device)
    np.testing.assert_array_equal(result.numpy(), [6])


@wp.func
def double_ref(x: wp.ref[wp.float32]):
    x = x * wp.float32(2.0)


@wp.kernel(enable_backward=False)
def kernel_double_via_ref(result: wp.array[wp.float32]):
    val = wp.float32(3.0)
    double_ref(val)
    result[0] = val


def test_ref_simple_assignment_mutates(test, device):
    """Simple assignment (x = expr) to a ref param mutates the original storage."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(kernel_double_via_ref, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy(), [6.0])


@wp.func
def forward_ref(x: wp.ref[wp.int32]):
    x += 1


@wp.func
def outer_ref(x: wp.ref[wp.int32]):
    forward_ref(x)  # forwarding a ref param to another ref param


@wp.kernel(enable_backward=False)
def kernel_ref_forwarding(result: wp.array[wp.int32]):
    val = wp.int32(0)
    outer_ref(val)
    result[0] = val


def test_ref_forwarding(test, device):
    """A wp.ref[T] parameter can be forwarded to another wp.ref[T] parameter."""
    result = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(kernel_ref_forwarding, dim=1, inputs=[result], device=device)
    np.testing.assert_array_equal(result.numpy(), [1])


@wp.func
def store_pair_ref(x: wp.ref[wp.float32], y: wp.ref[wp.float32], a: wp.float32, b: wp.float32):
    x, y = a, b  # tuple-unpack assignment to ref params must mutate caller storage (GH-1581)


@wp.func
def swap_pair_ref(x: wp.ref[wp.float32], y: wp.ref[wp.float32]):
    x, y = y, x  # RHS values must be loaded before either ref target is stored


@wp.func
def ref_tuple_local_first(x: wp.ref[wp.float32]) -> wp.float32:
    old, x = x, wp.float32(2.0)
    return old


@wp.func
def ref_tuple_ref_first(x: wp.ref[wp.float32]) -> wp.float32:
    x, old = wp.float32(1.0), x
    return old


@wp.func
def ref_tuple_existing_local(x: wp.ref[wp.float32]) -> wp.float32:
    old = wp.float32(0.0)
    old, x = x, wp.float32(2.0)
    return old


@wp.kernel(enable_backward=False)
def kernel_ref_tuple_unpack(out: wp.array[wp.float32]):
    x = wp.float32(0.0)
    y = wp.float32(0.0)
    store_pair_ref(x, y, wp.float32(3.0), wp.float32(4.0))
    out[0] = x
    out[1] = y


@wp.kernel(enable_backward=False)
def kernel_ref_tuple_unpack_swap(out: wp.array[wp.float32]):
    x = wp.float32(3.0)
    y = wp.float32(4.0)
    swap_pair_ref(x, y)
    out[0] = x
    out[1] = y


@wp.kernel(enable_backward=False)
def kernel_ref_tuple_unpack_mixed_targets(out: wp.array[wp.float32]):
    x = wp.float32(9.0)
    out[0] = ref_tuple_local_first(x)
    out[1] = x

    y = wp.float32(7.0)
    out[2] = ref_tuple_ref_first(y)
    out[3] = y

    z = wp.float32(9.0)
    out[4] = ref_tuple_existing_local(z)
    out[5] = z


def test_ref_tuple_unpack_assignment(test, device):
    """Tuple-unpack assignment to wp.ref[T] parameters mutates caller storage (GH-1581)."""
    out = wp.zeros(2, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_tuple_unpack, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [3.0, 4.0])


def test_ref_tuple_unpack_assignment_swaps(test, device):
    """Tuple-unpack assignment to wp.ref[T] parameters preserves RHS value order."""
    out = wp.zeros(2, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_tuple_unpack_swap, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [4.0, 3.0])


def test_ref_tuple_unpack_assignment_mixed_targets(test, device):
    """Tuple-unpack assignment snapshots ref RHS values before binding any target."""
    out = wp.zeros(6, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_tuple_unpack_mixed_targets, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [9.0, 2.0, 7.0, 1.0, 9.0, 2.0])


@wp.kernel
def kernel_array_tuple_unpack_adjoint(
    a: wp.array[wp.float32],
    b: wp.array[wp.float32],
    out: wp.array[wp.float32],
):
    tid = wp.tid()
    ai, bi = a[tid], b[tid]  # ordinary array-element unpack / must keep ref semantics for adjoint
    out[tid] = ai * bi


def test_array_tuple_unpack_preserves_adjoint(test, device):
    """Ordinary array-element tuple unpack must preserve reference semantics for autodiff.

    The tuple-unpack snapshot must NOT load array-element RHS values into
    plain copies when no target is a wp.ref[T] parameter, or the adjoint
    graph is severed and gradients are silently wrong.
    """
    a = wp.array([2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    b = wp.array([4.0, 5.0], dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros(2, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_array_tuple_unpack_adjoint, dim=2, inputs=[a, b], outputs=[out], device=device)

    out_grad = wp.ones(2, dtype=wp.float32, device=device)
    tape.backward(grads={out: out_grad})
    # d(a[i]*b[i])/da[i] = b[i], d/db[i] = a[i]
    np.testing.assert_allclose(a.grad.numpy(), b.numpy())
    np.testing.assert_allclose(b.grad.numpy(), a.numpy())


@wp.func_native(
    "x = x + 5;",
)
def native_add_to_ref(x: wp.ref[wp.int32]): ...


@wp.kernel(enable_backward=False)
def kernel_native_ref_param(result: wp.array[wp.int32]):
    local = wp.int32(10)
    native_add_to_ref(local)
    result[0] = local

    native_add_to_ref(result[1])


def test_native_ref_param_alias(test, device):
    """A func_native wp.ref[T] parameter is snippet-visible as a C++ reference."""
    result = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(kernel_native_ref_param, dim=1, inputs=[result], device=device)
    np.testing.assert_array_equal(result.numpy(), [15, 5])


@wp.func_native(
    "x = y * 2.0f;",
    adj_snippet="""
    adj_y += 2.0f * adj_x;
    adj_x = 0.0f;
    """,
)
def native_store_double_ref(x: wp.ref[wp.float32], y: wp.float32): ...


@wp.kernel
def kernel_native_ref_adjoint_local(x: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    tmp = wp.float32(0.0)
    native_store_double_ref(tmp, x[i])
    out[i] = tmp


@wp.kernel
def kernel_native_ref_adjoint_array(x: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    native_store_double_ref(out[i], x[i])


@wp.kernel
def kernel_native_ref_adjoint_array_2d_direct(
    x: wp.array[wp.float32],
    dst: wp.array2d[wp.float32],
    out: wp.array[wp.float32],
):
    i = wp.tid()
    native_store_double_ref(dst[i, 0], x[i])
    out[i] = dst[i, 0]


@wp.kernel
def kernel_native_ref_adjoint_array_slice(
    x: wp.array[wp.float32],
    dst: wp.array2d[wp.float32],
    out: wp.array[wp.float32],
):
    i = wp.tid()
    row = dst[i]
    native_store_double_ref(row[0], x[i])
    out[i] = row[0]


@wp.kernel
def kernel_native_ref_adjoint_array_nested_slice(
    x: wp.array[wp.float32],
    dst: wp.array2d[wp.float32],
    out: wp.array[wp.float32],
):
    i = wp.tid()
    row = dst[i]
    window = row[0:1]
    native_store_double_ref(window[0], x[i])
    out[i] = window[0]


def test_native_ref_param_adjoint_local(test, device):
    """A func_native ref parameter with adj_snippet can receive a local adjoint."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros(3, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_native_ref_adjoint_local, dim=3, inputs=[x], outputs=[out], device=device)

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])


def test_native_ref_param_adjoint_array(test, device):
    """A func_native ref parameter with adj_snippet can receive an array-element adjoint."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros(3, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_native_ref_adjoint_array, dim=3, inputs=[x], outputs=[out], device=device)

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])


def test_native_ref_param_adjoint_array_2d_direct(test, device):
    """A native ref adjoint can target a direct element of a 2D array."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    dst = wp.zeros((3, 1), dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros(3, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel_native_ref_adjoint_array_2d_direct,
            dim=3,
            inputs=[x],
            outputs=[dst, out],
            device=device,
        )

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])
    np.testing.assert_allclose(dst.grad.numpy(), np.zeros((3, 1), dtype=np.float32))


def test_native_ref_param_adjoint_array_slice(test, device):
    """A native ref adjoint can target an element through an array slice/view."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    dst = wp.zeros((3, 1), dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros(3, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel_native_ref_adjoint_array_slice,
            dim=3,
            inputs=[x],
            outputs=[dst, out],
            device=device,
        )

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])
    np.testing.assert_allclose(dst.grad.numpy(), np.zeros((3, 1), dtype=np.float32))


def test_native_ref_param_adjoint_array_nested_slice(test, device):
    """A native ref adjoint can target an element through nested array views."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    dst = wp.zeros((3, 1), dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros(3, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel_native_ref_adjoint_array_nested_slice,
            dim=3,
            inputs=[x],
            outputs=[dst, out],
            device=device,
        )

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])
    np.testing.assert_allclose(dst.grad.numpy(), np.zeros((3, 1), dtype=np.float32))


@wp.struct
class RefBox:
    value: wp.float32


@wp.func
def make_ref_box() -> RefBox:
    box = RefBox()
    box.value = wp.float32(1.0)
    return box


@wp.struct
class RefInner:
    value: wp.float32


@wp.struct
class RefOuter:
    inner: RefInner


@wp.struct
class RefCompositeBox:
    inner: RefInner
    vec: wp.vec3
    mat: wp.mat33
    quat: wp.quatf
    xform: wp.transformf


@wp.func
def set_ref_float(x: wp.ref[wp.float32], value: wp.float32):
    x = value


@wp.func
def set_ref_vec3(x: wp.ref[wp.vec3], value: wp.vec3):
    x = value


@wp.func
def bump_ref_vec_subscript(v: wp.ref[wp.vec3]):
    v[0] += 1.0


@wp.func
def bump_ref_mat_subscript(m: wp.ref[wp.mat33]):
    m[0, 0] += 1.0


@wp.kernel
def kernel_native_ref_adjoint_struct_field(x: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    box = RefBox()
    box.value = wp.float32(0.0)
    native_store_double_ref(box.value, x[i])
    out[i] = box.value


def test_native_ref_param_adjoint_struct_field(test, device):
    """A func_native ref parameter with adj_snippet can receive a struct-field adjoint."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros(3, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_native_ref_adjoint_struct_field, dim=3, inputs=[x], outputs=[out], device=device)

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_struct_array_field(boxes: wp.array[RefBox], out: wp.array[wp.float32]):
    i = wp.tid()
    set_ref_float(boxes[i].value, wp.float32(i) + wp.float32(10.0))
    out[i] = boxes[i].value


def test_ref_mutates_struct_array_field(test, device):
    """A field of a struct array element can be passed to wp.ref[T]."""
    boxes = wp.zeros(3, dtype=RefBox, device=device)
    out = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_mutates_struct_array_field, dim=3, inputs=[boxes], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [10.0, 11.0, 12.0])


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_nested_struct_array_field(outers: wp.array[RefOuter], out: wp.array[wp.float32]):
    i = wp.tid()
    set_ref_float(outers[i].inner.value, wp.float32(i) + wp.float32(20.0))
    out[i] = outers[i].inner.value


def test_ref_mutates_nested_struct_array_field(test, device):
    """Nested fields of struct array elements can be passed to wp.ref[T]."""
    outers = wp.zeros(3, dtype=RefOuter, device=device)
    out = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_mutates_nested_struct_array_field, dim=3, inputs=[outers], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [20.0, 21.0, 22.0])


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_composite_struct_array_fields(boxes: wp.array[RefCompositeBox], out: wp.array[wp.float32]):
    i = wp.tid()
    set_ref_float(boxes[i].vec.y, wp.float32(i) + wp.float32(30.0))
    set_ref_float(boxes[i].mat[1, 2], wp.float32(i) + wp.float32(40.0))
    out[2 * i] = boxes[i].vec.y
    out[2 * i + 1] = boxes[i].mat[1, 2]


def test_ref_mutates_composite_struct_array_fields(test, device):
    """Vector and matrix slots in struct array elements can be passed to wp.ref[T]."""
    boxes = wp.zeros(3, dtype=RefCompositeBox, device=device)
    out = wp.zeros(6, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_mutates_composite_struct_array_fields, dim=3, inputs=[boxes], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [30.0, 40.0, 31.0, 41.0, 32.0, 42.0])


@wp.kernel(enable_backward=False)
def kernel_ref_augassign_composite_subscripts(out: wp.array[wp.float32]):
    v = wp.vec3(1.0, 2.0, 3.0)
    m = wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    bump_ref_vec_subscript(v)
    bump_ref_mat_subscript(m)
    out[0] = v[0]
    out[1] = v[1]
    out[2] = m[0, 0]
    out[3] = m[0, 1]


def test_ref_augassign_composite_subscripts(test, device):
    """Subscripted += through a wp.ref[T] vector or matrix mutates caller storage."""
    out = wp.zeros(4, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_augassign_composite_subscripts, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [2.0, 2.0, 2.0, 2.0])


@wp.func
def forward_ref_outer_inner_value(outer: wp.ref[RefOuter], value: wp.float32):
    set_ref_float(outer.inner.value, value)


@wp.kernel(enable_backward=False)
def kernel_ref_parameter_nested_field_forwarding(x: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    outer = RefOuter()
    forward_ref_outer_inner_value(outer, x[i])
    out[i] = outer.inner.value


def test_ref_parameter_nested_field_forwarding(test, device):
    """A nested field of a ref parameter can be forwarded to another ref parameter."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    out = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_parameter_nested_field_forwarding, dim=3, inputs=[x], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [1.0, 2.0, 3.0])


@wp.kernel(enable_backward=False)
def kernel_ref_copy_from_component_is_local(out: wp.array[wp.float32]):
    v = wp.vec3(1.0, 2.0, 3.0)
    y = v.x
    set_ref_float(y, wp.float32(9.0))
    out[0] = v.x
    out[1] = y


def test_ref_copy_from_component_is_local(test, device):
    """A local copy of a component is addressable without aliasing the source component."""
    out = wp.zeros(2, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_copy_from_component_is_local, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [1.0, 9.0])


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_negative_subscripts(out: wp.array[wp.float32]):
    v = wp.vec3(1.0, 2.0, 3.0)
    m = wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    set_ref_float(v[-1], wp.float32(10.0))
    set_ref_vec3(m[-1], wp.vec3(30.0, 31.0, 32.0))
    set_ref_float(m[-1, -1], wp.float32(20.0))
    out[0] = v.z
    out[1] = m[2, 0]
    out[2] = m[2, 1]
    out[3] = m[2, 2]


def test_ref_mutates_negative_subscripts(test, device):
    """Negative vector and matrix subscripts use native wp::index semantics."""
    out = wp.zeros(4, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_mutates_negative_subscripts, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [10.0, 30.0, 31.0, 20.0])


@wp.kernel
def kernel_native_ref_adjoint_components(x: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    v = wp.vec3(0.0, 0.0, 0.0)
    m = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q = wp.quatf(0.0, 0.0, 0.0, 0.0)
    t = wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))

    native_store_double_ref(v.y, x[i])
    native_store_double_ref(m[1, 2], x[i])
    native_store_double_ref(q.w, x[i])
    native_store_double_ref(t.p.x, x[i])

    j = 4 * i
    out[j] = v.y
    out[j + 1] = m[1, 2]
    out[j + 2] = q.w
    out[j + 3] = t.p.x


def test_native_ref_param_adjoint_components(test, device):
    """A native ref adjoint can target local composite components."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros(12, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_native_ref_adjoint_components, dim=3, inputs=[x], outputs=[out], device=device)

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), np.repeat([2.0, 4.0, 6.0], 4))
    np.testing.assert_allclose(x.grad.numpy(), [8.0, 8.0, 8.0])


@wp.kernel
def kernel_native_ref_adjoint_struct_array_components(
    x: wp.array[wp.float32],
    boxes: wp.array[RefCompositeBox],
    out: wp.array[wp.float32],
):
    i = wp.tid()
    native_store_double_ref(boxes[i].quat.w, x[i])
    native_store_double_ref(boxes[i].xform.p.x, x[i])

    j = 2 * i
    out[j] = boxes[i].quat.w
    out[j + 1] = boxes[i].xform.p.x


def test_native_ref_param_adjoint_struct_array_components(test, device):
    """A native ref adjoint can target components inside struct array elements."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    boxes = wp.zeros(3, dtype=RefCompositeBox, device=device, requires_grad=True)
    out = wp.zeros(6, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel_native_ref_adjoint_struct_array_components,
            dim=3,
            inputs=[x, boxes],
            outputs=[out],
            device=device,
        )

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), np.repeat([2.0, 4.0, 6.0], 2))
    np.testing.assert_allclose(x.grad.numpy(), [4.0, 4.0, 4.0])


@wp.kernel
def kernel_native_ref_adjoint_nested_struct_array_field(
    x: wp.array[wp.float32],
    outers: wp.array[RefOuter],
    out: wp.array[wp.float32],
):
    i = wp.tid()
    native_store_double_ref(outers[i].inner.value, x[i])
    out[i] = outers[i].inner.value


def test_native_ref_param_adjoint_nested_struct_array_field(test, device):
    """A native ref adjoint can target a nested field inside a struct array element."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
    outers = wp.zeros(3, dtype=RefOuter, device=device, requires_grad=True)
    out = wp.zeros(3, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel_native_ref_adjoint_nested_struct_array_field, dim=3, inputs=[x, outers], outputs=[out], device=device
        )

    tape.backward(grads={out: wp.ones_like(out)})

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])


@wp.func
def forward_ref_box_value(box: wp.ref[RefBox], y: wp.float32):
    native_store_double_ref(box.value, y)


@wp.kernel(enable_backward=False)
def kernel_ref_parameter_struct_field_forwarding(x: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    box = RefBox()
    box.value = wp.float32(0.0)
    forward_ref_box_value(box, x[i])
    out[i] = box.value


def test_ref_parameter_struct_field_forwarding(test, device):
    """A ref parameter's field can be forwarded to another ref parameter."""
    x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    out = wp.zeros(3, dtype=wp.float32, device=device)

    wp.launch(kernel_ref_parameter_struct_field_forwarding, dim=3, inputs=[x], outputs=[out], device=device)

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])


# -----------------------------------------------------------------------
# wp.address_of() tests
# -----------------------------------------------------------------------


@wp.func_native(
    "*(int32_t*)ptr = 42;",
)
def set_via_ptr(ptr: wp.uint64): ...


@wp.kernel(enable_backward=False)
def kernel_address_of_local(result: wp.array[wp.int32]):
    val = wp.int32(0)
    set_via_ptr(wp.address_of(val))
    result[0] = val


def test_address_of_local(test, device):
    """wp.address_of() on a local variable passes its address to a native snippet."""
    result = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(kernel_address_of_local, dim=1, inputs=[result], device=device)
    np.testing.assert_array_equal(result.numpy(), [42])


@wp.func_native(
    "*(float*)ptr = *(float*)ptr + 1.0f;",
)
def increment_via_ptr(ptr: wp.uint64): ...


@wp.kernel(enable_backward=False)
def kernel_address_of_array_element(arr: wp.array[wp.float32]):
    i = wp.tid()
    increment_via_ptr(wp.address_of(arr[i]))


def test_address_of_array_element(test, device):
    """wp.address_of(arr[i]) passes element address to a native snippet."""
    arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    wp.launch(kernel_address_of_array_element, dim=3, inputs=[arr], device=device)
    np.testing.assert_allclose(arr.numpy(), [2.0, 3.0, 4.0])


@wp.kernel(enable_backward=False)
def kernel_address_of_struct_field(result: wp.array[wp.float32]):
    box = RefBox()
    box.value = wp.float32(4.0)
    increment_via_ptr(wp.address_of(box.value))
    result[0] = box.value


def test_address_of_struct_field(test, device):
    """wp.address_of(s.field) passes the address of struct field storage."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(kernel_address_of_struct_field, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy(), [5.0])


@wp.kernel(enable_backward=False)
def kernel_address_of_vector_component(result: wp.array[wp.float32]):
    v = wp.vec3(1.0, 2.0, 3.0)
    increment_via_ptr(wp.address_of(v.y))
    result[0] = v.y


def test_address_of_vector_component(test, device):
    """wp.address_of(v.y) passes the address of vector component storage."""
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(kernel_address_of_vector_component, dim=1, outputs=[result], device=device)
    np.testing.assert_allclose(result.numpy(), [3.0])


@wp.kernel(enable_backward=False)
def kernel_address_of_nested_struct_array_field(outers: wp.array[RefOuter], result: wp.array[wp.float32]):
    i = wp.tid()
    set_ref_float(outers[i].inner.value, wp.float32(5.0))
    increment_via_ptr(wp.address_of(outers[i].inner.value))
    result[i] = outers[i].inner.value


def test_address_of_nested_struct_array_field(test, device):
    """wp.address_of() supports nested lvalues rooted at struct array elements."""
    outers = wp.zeros(3, dtype=RefOuter, device=device)
    result = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(kernel_address_of_nested_struct_array_field, dim=3, inputs=[outers], outputs=[result], device=device)
    np.testing.assert_allclose(result.numpy(), [6.0, 6.0, 6.0])


# -----------------------------------------------------------------------
# Regression: by-value semantics still work
# -----------------------------------------------------------------------


@wp.func
def by_value_increment(x: wp.int32) -> wp.int32:
    x += 1
    return x


@wp.kernel
def kernel_by_value_unchanged(result: wp.array[wp.int32]):
    val = wp.int32(10)
    out = by_value_increment(val)
    result[0] = val  # original must be unchanged
    result[1] = out  # returned value must be incremented


# -----------------------------------------------------------------------
# Test registration
# -----------------------------------------------------------------------

devices = get_test_devices()


add_function_test(TestRef, func=test_ref_mutates_local, name="test_ref_mutates_local", devices=devices)
add_function_test(TestRef, func=test_ref_mutates_array_element, name="test_ref_mutates_array_element", devices=devices)
add_function_test(TestRef, func=test_ref_mutates_struct_field, name="test_ref_mutates_struct_field", devices=devices)
add_function_test(
    TestRef, func=test_ref_simple_assignment_mutates, name="test_ref_simple_assignment_mutates", devices=devices
)
add_function_test(TestRef, func=test_ref_forwarding, name="test_ref_forwarding", devices=devices)
add_function_test(
    TestRef, func=test_ref_tuple_unpack_assignment, name="test_ref_tuple_unpack_assignment", devices=devices
)
add_function_test(
    TestRef, func=test_ref_tuple_unpack_assignment_swaps, name="test_ref_tuple_unpack_assignment_swaps", devices=devices
)
add_function_test(
    TestRef,
    func=test_ref_tuple_unpack_assignment_mixed_targets,
    name="test_ref_tuple_unpack_assignment_mixed_targets",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_array_tuple_unpack_preserves_adjoint,
    name="test_array_tuple_unpack_preserves_adjoint",
    devices=devices,
)
add_function_test(TestRef, func=test_native_ref_param_alias, name="test_native_ref_param_alias", devices=["cpu"])
add_function_test(
    TestRef, func=test_native_ref_param_adjoint_local, name="test_native_ref_param_adjoint_local", devices=devices
)
add_function_test(
    TestRef, func=test_native_ref_param_adjoint_array, name="test_native_ref_param_adjoint_array", devices=devices
)
add_function_test(
    TestRef,
    func=test_native_ref_param_adjoint_array_2d_direct,
    name="test_native_ref_param_adjoint_array_2d_direct",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_native_ref_param_adjoint_array_slice,
    name="test_native_ref_param_adjoint_array_slice",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_native_ref_param_adjoint_array_nested_slice,
    name="test_native_ref_param_adjoint_array_nested_slice",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_native_ref_param_adjoint_struct_field,
    name="test_native_ref_param_adjoint_struct_field",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_mutates_struct_array_field,
    name="test_ref_mutates_struct_array_field",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_mutates_nested_struct_array_field,
    name="test_ref_mutates_nested_struct_array_field",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_mutates_composite_struct_array_fields,
    name="test_ref_mutates_composite_struct_array_fields",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_augassign_composite_subscripts,
    name="test_ref_augassign_composite_subscripts",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_parameter_nested_field_forwarding,
    name="test_ref_parameter_nested_field_forwarding",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_copy_from_component_is_local,
    name="test_ref_copy_from_component_is_local",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_mutates_negative_subscripts,
    name="test_ref_mutates_negative_subscripts",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_native_ref_param_adjoint_components,
    name="test_native_ref_param_adjoint_components",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_native_ref_param_adjoint_struct_array_components,
    name="test_native_ref_param_adjoint_struct_array_components",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_native_ref_param_adjoint_nested_struct_array_field,
    name="test_native_ref_param_adjoint_nested_struct_array_field",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_parameter_struct_field_forwarding,
    name="test_ref_parameter_struct_field_forwarding",
    devices=devices,
)
add_function_test(TestRef, func=test_address_of_local, name="test_address_of_local", devices=["cpu"])
add_function_test(TestRef, func=test_address_of_array_element, name="test_address_of_array_element", devices=["cpu"])
add_function_test(TestRef, func=test_address_of_struct_field, name="test_address_of_struct_field", devices=["cpu"])
add_function_test(
    TestRef, func=test_address_of_vector_component, name="test_address_of_vector_component", devices=["cpu"]
)
add_function_test(
    TestRef,
    func=test_address_of_nested_struct_array_field,
    name="test_address_of_nested_struct_array_field",
    devices=["cpu"],
)


# -----------------------------------------------------------------------
# Quaternion and transform component pass-by-reference
# -----------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_quat_component(out: wp.array[wp.float32]):
    q = wp.quatf(1.0, 2.0, 3.0, 4.0)
    set_ref_float(q.y, wp.float32(9.0))
    out[0] = q.x
    out[1] = q.y
    out[2] = q.z
    out[3] = q.w


def test_ref_mutates_quat_component(test, device):
    """A quaternion component can be passed to wp.ref[T] and mutated."""
    out = wp.zeros(4, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_mutates_quat_component, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [1.0, 9.0, 3.0, 4.0])


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_struct_array_quat_component(boxes: wp.array[RefCompositeBox], out: wp.array[wp.float32]):
    i = wp.tid()
    set_ref_float(boxes[i].quat.w, wp.float32(i) + wp.float32(10.0))
    out[i] = boxes[i].quat.w


def test_ref_mutates_struct_array_quat_component(test, device):
    """A quaternion component in a struct array element can be passed to wp.ref[T]."""
    boxes = wp.zeros(3, dtype=RefCompositeBox, device=device)
    out = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_mutates_struct_array_quat_component, dim=3, inputs=[boxes], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [10.0, 11.0, 12.0])


@wp.kernel(enable_backward=False)
def kernel_ref_copy_from_quat_component_is_local(out: wp.array[wp.float32]):
    q = wp.quatf(1.0, 2.0, 3.0, 4.0)
    w = q.w
    set_ref_float(w, wp.float32(9.0))
    out[0] = q.w
    out[1] = w


def test_ref_copy_from_quat_component_is_local(test, device):
    """A local copy of a quat component is addressable without aliasing the source."""
    out = wp.zeros(2, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_copy_from_quat_component_is_local, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [4.0, 9.0])


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_quat_subscript(out: wp.array[wp.float32]):
    q = wp.quatf(1.0, 2.0, 3.0, 4.0)
    set_ref_float(q[2], wp.float32(9.0))
    out[0] = q.x
    out[1] = q.y
    out[2] = q.z
    out[3] = q.w


def test_ref_mutates_quat_subscript(test, device):
    """A quaternion subscript can be passed to wp.ref[T] and mutated."""
    out = wp.zeros(4, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_mutates_quat_subscript, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [1.0, 2.0, 9.0, 4.0])


add_function_test(
    TestRef, func=test_ref_mutates_quat_component, name="test_ref_mutates_quat_component", devices=devices
)
add_function_test(
    TestRef,
    func=test_ref_mutates_struct_array_quat_component,
    name="test_ref_mutates_struct_array_quat_component",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_copy_from_quat_component_is_local,
    name="test_ref_copy_from_quat_component_is_local",
    devices=devices,
)
add_function_test(
    TestRef, func=test_ref_mutates_quat_subscript, name="test_ref_mutates_quat_subscript", devices=devices
)


# -----------------------------------------------------------------------
# Transform component pass-by-reference
# -----------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_transform_position_component(out: wp.array[wp.float32]):
    t = wp.transformf(wp.vec3f(1.0, 2.0, 3.0), wp.quatf(0.0, 0.0, 0.0, 1.0))
    set_ref_float(t.p.x, wp.float32(9.0))
    out[0] = t.p.x
    out[1] = t.p.y
    out[2] = t.p.z


def test_ref_mutates_transform_position_component(test, device):
    """A transform position component can be passed to wp.ref[T] and mutated."""
    out = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_mutates_transform_position_component, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [9.0, 2.0, 3.0])


@wp.kernel(enable_backward=False)
def kernel_ref_mutates_struct_array_transform_position(boxes: wp.array[RefCompositeBox], out: wp.array[wp.float32]):
    i = wp.tid()
    set_ref_float(boxes[i].xform.p.x, wp.float32(i) + wp.float32(10.0))
    out[i] = boxes[i].xform.p.x


def test_ref_mutates_struct_array_transform_position(test, device):
    """A transform position component in a struct array element can be passed to wp.ref[T]."""
    boxes = wp.zeros(3, dtype=RefCompositeBox, device=device)
    out = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(
        kernel_ref_mutates_struct_array_transform_position,
        dim=3,
        inputs=[boxes],
        outputs=[out],
        device=device,
    )
    np.testing.assert_allclose(out.numpy(), [10.0, 11.0, 12.0])


@wp.kernel(enable_backward=False)
def kernel_ref_copy_from_transform_position_is_local(out: wp.array[wp.float32]):
    t = wp.transformf(wp.vec3f(1.0, 2.0, 3.0), wp.quatf(0.0, 0.0, 0.0, 1.0))
    x = t.p.x
    set_ref_float(x, wp.float32(9.0))
    out[0] = t.p.x
    out[1] = x


def test_ref_copy_from_transform_position_is_local(test, device):
    """A local copy of a transform position component is addressable without aliasing the source."""
    out = wp.zeros(2, dtype=wp.float32, device=device)
    wp.launch(kernel_ref_copy_from_transform_position_is_local, dim=1, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [1.0, 9.0])


add_function_test(
    TestRef,
    func=test_ref_mutates_transform_position_component,
    name="test_ref_mutates_transform_position_component",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_mutates_struct_array_transform_position,
    name="test_ref_mutates_struct_array_transform_position",
    devices=devices,
)
add_function_test(
    TestRef,
    func=test_ref_copy_from_transform_position_is_local,
    name="test_ref_copy_from_transform_position_is_local",
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
