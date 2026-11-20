# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for passing user-defined subclasses of the Warp array classes to Warp APIs (GH-1723).

A subclass instance is treated as its native array family (``wp.array``, ``wp.indexedarray``,
``wp.fabricarray`` or ``wp.indexedfabricarray``) wherever the parent class is accepted, while the
original object is retained at runtime.
"""

import collections
import ctypes
import gc
import json
import os
import tempfile
import unittest
import weakref
from collections.abc import Sequence
from typing import Any, Literal
from unittest import mock

import numpy as np

import warp as wp
import warp._src.codegen as codegen
import warp._src.context as context
import warp._src.types as wptypes
import warp.autograd as autograd
import warp.fem as fem
from warp.tests.test_fabricarray import _create_fabric_array_interface
from warp.tests.unittest_utils import (
    add_function_test,
    assert_np_equal,
    get_selected_cuda_test_devices,
    get_test_devices,
    run_python_subprocess,
)

# ---------------------------------------------------------------------------------------------
# Application subclasses.
# The constructors require a keyword that Warp cannot know, and they record how often they run,
# so a library call that constructs a subclass instance is detected.
# ---------------------------------------------------------------------------------------------

constructor_calls = collections.Counter()


class TaggedArray(wp.array):
    def __init__(self, *args, tag, **kwargs):
        constructor_calls[type(self).__name__] += 1
        self.tag = tag
        super().__init__(*args, **kwargs)


class SecondTaggedArray(TaggedArray):
    pass


class ThirdLevelArray(SecondTaggedArray):
    pass


class Mixin:
    """An ordinary mixin without any Warp base class."""

    def hello(self):
        return "hello"


class MixinArray(Mixin, TaggedArray):
    pass


class LeftArray(TaggedArray):
    pass


class RightArray(TaggedArray):
    pass


class DiamondArray(LeftArray, RightArray):
    pass


class TaggedIndexedArray(wp.indexedarray):
    def __init__(self, *args, tag, **kwargs):
        constructor_calls[type(self).__name__] += 1
        self.tag = tag
        super().__init__(*args, **kwargs)


class IndexedChild(TaggedIndexedArray):
    pass


class TaggedFabricArray(wp.fabricarray):
    def __init__(self, *args, tag, **kwargs):
        constructor_calls[type(self).__name__] += 1
        self.tag = tag
        super().__init__(*args, **kwargs)


class FabricChild(TaggedFabricArray):
    pass


class TaggedIndexedFabricArray(wp.indexedfabricarray):
    def __init__(self, *args, tag, **kwargs):
        constructor_calls[type(self).__name__] += 1
        self.tag = tag
        super().__init__(*args, **kwargs)


class IndexedFabricChild(TaggedIndexedFabricArray):
    pass


class CrossFamilyArray(wp.array, wp.indexedarray):
    """Invalid: derives from two unrelated native array families."""


class CrossFamilyReversed(wp.indexedarray, wp.array):
    """Invalid: the same families in the opposite base order."""


class FixedChild(wp.fixedarray):
    pass


FAMILIES = (wp.array, wp.indexedarray, wp.fabricarray, wp.indexedfabricarray)
TYPE_CODE_PREFIXES = ("a", "ia", "fa", "ifa")


# ---------------------------------------------------------------------------------------------
# Kernels and functions
# ---------------------------------------------------------------------------------------------


@wp.kernel
def read_regular(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel
def read_indexed(a: wp.indexedarray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_fabric(a: wp.fabricarray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_indexed_fabric(a: wp.indexedfabricarray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_any(a: Any, out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_generic_regular(a: wp.array[Any], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_generic_indexed(a: wp.indexedarray[Any], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_generic_fabric(a: wp.fabricarray[Any], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel(enable_backward=False)
def read_generic_indexed_fabric(a: wp.indexedfabricarray[Any], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


GENERIC_KERNEL_BY_FAMILY = {
    wp.array: read_generic_regular,
    wp.indexedarray: read_generic_indexed,
    wp.fabricarray: read_generic_fabric,
    wp.indexedfabricarray: read_generic_indexed_fabric,
}


@wp.kernel
def square(values: wp.array[Any], result: wp.array[Any]):
    i = wp.tid()
    result[i] = values[i] * values[i]


@wp.kernel
def square_float(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * a[i]


@wp.kernel
def double_in_place(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] * 2.0


@wp.kernel
def sum_values(a: wp.array[float], loss: wp.array[float]):
    wp.atomic_add(loss, 0, a[wp.tid()])


@wp.kernel
def multiply_two(a: wp.array[float], out: wp.array[float]):
    out[wp.tid()] = 2.0 * a[wp.tid()]


@wp.kernel
def multiply_three(a: wp.array[float], out: wp.array[float]):
    out[wp.tid()] = 3.0 * a[wp.tid()]


@wp.struct
class Record:
    v: wp.vec3
    w: float


@wp.kernel
def read_record(a: wp.array[Record], out: wp.array[float]):
    record = a[wp.tid()]
    out[wp.tid()] = record.v[0] + record.v[1] + record.v[2] + record.w


@wp.kernel
def read_bfloat16(a: wp.array[wp.bfloat16], out: wp.array[float]):
    out[wp.tid()] = float(a[wp.tid()])


@wp.kernel
def read_or_default(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    if a.shape[0] > 0:
        out[i] = a[i]
    else:
        out[i] = -1.0


@wp.kernel
def read_vec3(a: wp.array[wp.vec3], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i][0] + a[i][1] + a[i][2]


@wp.kernel
def read_2d(a: wp.array2d[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i, 0]


@wp.kernel
def copy_generic(a: wp.array[Any], out: wp.array[Any]):
    i = wp.tid()
    out[i] = a[i]


wp.overload(copy_generic, [wp.array[wp.float32], wp.array[wp.float32]])


@wp.func
def square_element(a: float) -> float:
    return a * a


@wp.func
def split_element(a: float):
    return a + 1.0, a * 2.0


@wp.func
def add_elements(a: float, b: float):
    return a + b


@wp.func
def identify_array(a: wp.array[float]) -> wp.array[float]:
    return a


@wp.func
def identify_array(a: wp.array[wp.int32]) -> wp.array[wp.int32]:
    return a


@wp.func
def identify_generic_array(a: wp.array[Any]) -> wp.array[Any]:
    return a


@wp.func
def identify_generic_array(a: int) -> int:
    return a


@wp.func
def identify_indexed(a: wp.indexedarray[float]) -> wp.indexedarray[float]:
    return a


@wp.func
def identify_indexed(a: int) -> int:
    return a


@wp.struct
class ArrayBundle:
    a: wp.array[float]
    ia: wp.indexedarray[float]
    fa: wp.fabricarray[float]
    ifa: wp.indexedfabricarray[float]


@wp.kernel(enable_backward=False)
def read_bundle(bundle: ArrayBundle, out: wp.array[float]):
    i = wp.tid()
    out[i] = bundle.a[i] + bundle.ia[i] + bundle.fa[i] + bundle.ifa[i]


@wp.struct
class InnerStruct:
    arr: wp.array[float]


@wp.struct
class OuterStruct:
    inner: InnerStruct
    scale: float


@wp.kernel
def read_nested(o: OuterStruct, out: wp.array[float]):
    i = wp.tid()
    out[i] = o.inner.arr[i] * o.scale


@wp.func
def scaled_read(a: wp.array[float], i: int, scale: float = 3.0):
    return a[i] * scale


@wp.kernel
def read_with_default_argument(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = scaled_read(a, i)


@wp.kernel
def row_view_kernel(a: wp.array2d[Any], out: wp.array[Any]):
    i = wp.tid()
    row = a[i]  # creates a view, which reconstructs an array type from the argument type
    out[i] = row[0]


@fem.integrand
def subclass_sample_value(s: fem.Sample):
    return float(s.qp_index) + 1.0


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def make_families(device, classes=(TaggedArray, TaggedIndexedArray, TaggedFabricArray, TaggedIndexedFabricArray)):
    """Return ``(array, expected values read by the kernels, typed kernel)`` for each native family."""
    cls_a, cls_ia, cls_fa, cls_ifa = classes
    values = np.arange(1, 5, dtype=np.float32)
    a = cls_a(tag="regular", data=values, dtype=float, device=device)
    indices = wp.array([3, 1, 0, 2], dtype=wp.int32, device=device)
    ia = cls_ia(tag="indexed", data=a, indices=indices)
    fa = cls_fa(tag="fabric", data=_create_fabric_array_interface(a, "a", [2, 2]), attrib="a")
    ifa = cls_ifa(tag="indexed fabric", fa=fa, indices=indices)
    permuted = values[[3, 1, 0, 2]]
    return [
        (a, values, read_regular),
        (ia, permuted, read_indexed),
        (fa, values, read_fabric),
        (ifa, permuted, read_indexed_fabric),
    ]


def make_deep_families(device):
    return make_families(device, (ThirdLevelArray, IndexedChild, FabricChild, IndexedFabricChild))


# ---------------------------------------------------------------------------------------------
# CPU-only tests
# ---------------------------------------------------------------------------------------------


class TestArraySubclass(unittest.TestCase):
    def setUp(self):
        self.cpu = wp.get_device("cpu")

    def test_family_resolution(self):
        for (instance, _, _), family in zip(make_deep_families(self.cpu), FAMILIES, strict=True):
            with self.subTest(family=family.__name__):
                self.assertTrue(wptypes.is_array(instance))
                self.assertIs(wptypes.concrete_array_type(instance), family)

        for cls in (MixinArray, DiamondArray, SecondTaggedArray):
            instance = cls(tag="x", data=[1.0, 2.0], dtype=float, device=self.cpu)
            self.assertIs(wptypes.concrete_array_type(instance), wp.array)
        self.assertEqual(MixinArray(tag="m", shape=1, dtype=float, device=self.cpu).hello(), "hello")

        # exact classes and annotations keep reporting their own family
        self.assertIs(wptypes.concrete_array_type(wp.zeros(2, device=self.cpu)), wp.array)
        for family in FAMILIES:
            self.assertIs(wptypes.concrete_array_type(family[float]), family)
        self.assertIs(wptypes.concrete_array_type(wp.array(dtype=float)), wp.array)

    def test_non_arrays_keep_their_type(self):
        self.assertIs(wptypes.concrete_array_type(3.0), float)
        self.assertIs(wptypes.concrete_array_type(wp.float32), type)
        # class objects are not arrays, neither is a NumPy array
        self.assertIs(wptypes.concrete_array_type(TaggedArray), type)
        self.assertIs(wptypes.concrete_array_type(np.zeros(2)), np.ndarray)
        self.assertFalse(wptypes.is_array(TaggedArray))

    def test_mock_and_weak_proxies_are_not_array_subclasses(self):
        owner = TaggedArray([1.0, 2.0], tag="proxy owner", dtype=float, device=self.cpu)
        for value in (mock.Mock(spec=wp.array), weakref.proxy(owner)):
            with self.subTest(type=type(value)):
                self.assertIsInstance(value, wp.array)
                self.assertIsNone(wptypes._resolve_array_family(value))
                self.assertIs(wptypes.concrete_array_type(value), type(value))

    def test_cross_family_inheritance_is_rejected(self):
        for cls in (CrossFamilyArray, CrossFamilyReversed):
            with self.subTest(cls=cls.__name__):
                # an uninitialized instance is enough: the class decides
                instance = cls.__new__(cls)
                self.assertTrue(wptypes.is_array(instance))
                with self.assertRaisesRegex(TypeError, "native array family"):
                    wptypes.concrete_array_type(instance)
                with self.assertRaisesRegex(TypeError, "native array family"):
                    wptypes.infer_argument_types([instance], [wp.array[Any]])
                with self.assertRaisesRegex(RuntimeError, "native array family"):
                    context.pack_arg(read_regular, wp.array[float], "a", instance, self.cpu)
                with self.assertRaisesRegex(RuntimeError, "native array family"):
                    context.pack_arg(read_indexed, wp.indexedarray[float], "a", instance, self.cpu, adjoint=True)
                with self.assertRaisesRegex(TypeError, "native array family"):
                    wptypes.array_type_id(instance)
                with self.assertRaisesRegex(TypeError, "native array family"):
                    ArrayBundle().a = instance
                # Diagnostic rendering must remain usable even for ambiguous objects.
                instance.dtype = wp.float32
                instance.ndim = 1
                instance.shape = (1,)
                self.assertIn(cls.__name__, wptypes.type_repr(instance))

    def test_generic_launch_skips_descriptor_normalization(self):
        @wp.kernel(module="unique", enable_backward=False)
        def add_vectors(a: wp.vec3, b: wp.vec3, c: wp.vec3, x: wp.array[Any], y: wp.array[Any]):
            i = wp.tid()
            y[i] = x[i] + a[0] + b[0] + c[0]

        x = TaggedArray([1.0, 2.0], tag="generic", dtype=float, device=self.cpu)
        y = wp.empty_like(x)
        args = [wp.vec3(1.0), wp.vec3(2.0), wp.vec3(3.0), x, y]
        with mock.patch.object(
            wptypes, "_normalize_array_argument_type", side_effect=AssertionError("Redundant normalization")
        ):
            # Cover both creation and reuse of a canonical specialization.
            for _ in range(2):
                wp.launch(add_vectors, 2, inputs=args, device=self.cpu)
                np.testing.assert_array_equal(y.numpy(), [7.0, 8.0])
        self.assertEqual(len(add_vectors.overloads), 1)

    def test_fixedarray_host_values_leave_overloads_unchanged(self):
        @wp.kernel(module="unique", enable_backward=False)
        def read(a: wp.array[Any], out: wp.array[float]):
            out[0] = a[0]

        out = wp.empty(1, dtype=float, device=self.cpu)
        for specialized in (False, True):
            if specialized:
                wp.launch(read, 1, inputs=[wp.ones_like(out)], outputs=[out], device=self.cpu)
                np.testing.assert_array_equal(out.numpy(), [1.0])
            overloads = dict(read.overloads)
            module_hash = read.module.hash_module()
            # A regression must fail without compiling an incompatible descriptor.
            with mock.patch.object(
                read.module, "load", side_effect=AssertionError("Invalid host value reached compilation")
            ):
                for cls in (wp.fixedarray, FixedChild):
                    value = cls(dtype=float, shape=1)
                    with self.subTest(specialized=specialized, cls=cls.__name__):
                        with self.assertRaisesRegex(TypeError, "fixedarray"):
                            wp.launch(read, 1, inputs=[value], outputs=[out], device=self.cpu)
                        for descriptors in ([value, wp.array[float]], {"a": value, "out": wp.array[float]}):
                            with self.assertRaisesRegex(TypeError, "fixedarray"):
                                wp.overload(read, descriptors)
                        self.assertEqual(read.overloads, overloads)
                        self.assertEqual(read.module.hash_module(), module_hash)

    def test_fixedarray_remains_a_distinct_compiler_type(self):
        class RaisingFixedArray(wp.fixedarray):
            def __init__(self, *args, **kwargs):
                raise AssertionError("Warp called an application fixedarray constructor")

        for shape in ((0,), (3,), (2, 3), (2, 1, 3)):
            for cls in (wp.fixedarray, FixedChild, RaisingFixedArray):
                with self.subTest(shape=shape, cls=cls.__name__):
                    value = cls.__new__(cls)
                    wp.fixedarray.__init__(value, dtype=wp.float32, shape=shape)
                    template = wp.array[wp.float32, Literal[len(shape)]]
                    self.assertIs(wptypes.concrete_array_type(value), wp.fixedarray)
                    self.assertFalse(wptypes.types_equal(value, template))
                    self.assertEqual(wptypes.get_type_code(value), f"a{len(shape)}f4")
                    self.assertTrue(wptypes.type_matches_template(value, template))
                    # A compiler template must retain its fixed extents without
                    # calling an application constructor. It is not a host value.
                    inferred = wptypes.infer_argument_types([None], [value])[0]
                    self.assertIs(type(inferred), wp.fixedarray)
                    self.assertIs(inferred.dtype, wp.float32)
                    self.assertEqual(inferred.ndim, len(shape))
                    self.assertEqual(inferred.shape, shape)
                    self.assertEqual(inferred.size, value.size)
                    self.assertEqual(
                        codegen.Var.type_to_ctype(inferred), f"wp::fixedarray_t<{value.size}, wp::float32>"
                    )
                    for template in (value, Any, wp.array[Any]):
                        with self.assertRaisesRegex(TypeError, "fixedarray"):
                            wptypes.infer_argument_types([value], [template])

    def test_fixedarray_subclass_source_annotations_are_rejected(self):
        annotation = FixedChild(dtype=float, shape=4)

        def invalid_kernel(a: annotation):
            pass

        def invalid_function(a: annotation):
            return a

        def invalid_return() -> annotation:
            return wp.zeros(shape=4, dtype=float)

        class InvalidStruct:
            arr: annotation

        for decorator, definition in (
            (wp.kernel, invalid_kernel),
            (wp.func, invalid_function),
            (wp.func, invalid_return),
            (wp.struct, InvalidStruct),
        ):
            with self.subTest(decorator=decorator.__name__, definition=definition.__name__):
                module = wp.get_module(__name__)
                module_hash = module.hash_module()
                with self.assertRaisesRegex(TypeError, "fixedarray subclass instance as an annotation"):
                    decorator(definition)
                self.assertEqual(module.hash_module(), module_hash)

    def test_subclass_annotations_remain_unsupported(self):
        for cls in (TaggedArray, TaggedIndexedArray, TaggedFabricArray, TaggedIndexedFabricArray):
            with self.subTest(cls=cls.__name__):
                self.assertFalse(wptypes.type_is_array(cls))
                with self.assertRaisesRegex(TypeError, "Subscript syntax not supported"):
                    cls[wp.float32]
                with self.assertRaisesRegex(TypeError, "Unrecognized type"):
                    wptypes.get_type_code(cls)

    def test_inference_describes_without_constructing(self):
        templates = (
            wp.array[Any, Literal[1]],
            wp.indexedarray[Any, Literal[1]],
            wp.fabricarray[Any, Literal[1]],
            wp.indexedfabricarray[Any, Literal[1]],
        )
        for (instance, _, _), template in zip(make_deep_families(self.cpu), templates, strict=True):
            with self.subTest(cls=type(instance).__name__):
                before = constructor_calls.copy()
                (inferred,) = wptypes.infer_argument_types([instance], [template])
                self.assertEqual(constructor_calls, before)
                self.assertIs(type(inferred), type(template))
                self.assertIs(inferred.dtype, wp.float32)
                self.assertEqual(inferred.ndim, 1)

        # a subclass instance used as a factory-style template must not be constructed either
        template = TaggedArray(tag="template", shape=(2, 2), dtype=float, device=self.cpu)
        before = constructor_calls.copy()
        (inferred,) = wptypes.infer_argument_types([None], [template])
        self.assertEqual(constructor_calls, before)
        self.assertEqual(inferred, wp.array2d[wp.float32])

    def test_parent_and_subclass_share_one_type(self):
        parent = wp.zeros(3, dtype=float, device=self.cpu)
        child = TaggedArray(tag="child", shape=3, dtype=float, device=self.cpu)
        sibling = SecondTaggedArray(tag="sibling", shape=3, dtype=float, device=self.cpu)
        inferred = [wptypes.infer_argument_types([a], [wp.array[Any]])[0] for a in (parent, child, sibling)]
        self.assertEqual(inferred[0], inferred[1])
        self.assertEqual(inferred[0], inferred[2])
        self.assertEqual(len({wptypes.get_signature([t]) for t in inferred}), 1)
        self.assertTrue(wptypes.types_equal(child, sibling))
        self.assertTrue(wptypes.types_equal(child, wp.array[float]))
        self.assertFalse(wptypes.types_equal(child, wp.array[wp.float64]))
        self.assertTrue(wptypes.type_matches_template(child, wp.array[float]))
        self.assertTrue(wptypes.type_matches_template(child, wp.array[Any]))
        self.assertFalse(wptypes.type_matches_template(child, wp.array[wp.float64]))
        self.assertFalse(wptypes.type_matches_template(child, wp.array2d[float]))
        self.assertFalse(wptypes.type_matches_template(child, wp.indexedarray[Any]))

    def test_families_stay_distinct(self):
        values = [instance for instance, _, _ in make_deep_families(self.cpu)]
        for i, lhs in enumerate(values):
            self.assertEqual(wptypes.get_type_code(lhs), f"{TYPE_CODE_PREFIXES[i]}1f4")
            for j, rhs in enumerate(values):
                self.assertEqual(wptypes.types_equal(lhs, rhs), i == j)

    def test_annotation_discovery_keeps_runtime_values(self):
        # General annotation discovery is deliberately unchanged. Runtime call
        # boundaries normalize values before placing types in an overload cache.
        for instance, _, _ in make_deep_families(self.cpu):
            self.assertIs(codegen.get_arg_type(instance), instance)
        annotation = wp.array[float]
        self.assertIs(codegen.get_arg_type(annotation), annotation)
        # C++ types use native families; runtime diagnostics keep Python names.
        for (instance, _, _), family in zip(make_deep_families(self.cpu), FAMILIES, strict=True):
            self.assertEqual(codegen.Var.type_to_ctype(instance), f"wp::{family.__name__}_t<wp::float32>")
            self.assertIn(type(instance).__name__, wptypes.type_repr(instance))

    def test_annotations_are_not_runtime_values(self):
        for family in FAMILIES:
            for annotation in (family[float], family[Any], family[float, Literal[2]], family[Any, Literal[2]]):
                with self.subTest(annotation=annotation):
                    with self.assertRaisesRegex(TypeError, "Unable to infer"):
                        wptypes.infer_argument_types([annotation], [Any])

    def test_annotation_values_are_not_packed(self):
        # Pack directly: an invalid descriptor must never reach a native launch.
        for (_, _, kernel), family in zip(make_families(self.cpu), FAMILIES, strict=True):
            for annotation in (family[float], family[Any], family[float, Literal[2]], family[Any, Literal[2]]):
                for mode in wp.config.LaunchArrayAccessMode:
                    for adjoint in (False, True):
                        with self.subTest(annotation=annotation, mode=mode, adjoint=adjoint):
                            prefix = "adjoint " if adjoint else ""
                            message = (
                                f"{prefix}argument 'a' expects an array of type .*"
                                f"passed value has type {type(annotation).__name__}"
                            )
                            with mock.patch.object(wp.config, "launch_array_access_mode", mode):
                                with self.assertRaisesRegex(RuntimeError, message):
                                    context.pack_arg(kernel, family[float], "a", annotation, self.cpu, adjoint=adjoint)

    def test_annotation_values_do_not_modify_struct_fields(self):
        families = make_families(self.cpu)
        families[0][0].requires_grad = True
        fields = ("a", "ia", "fa", "ifa")
        for family, field in zip(FAMILIES, fields, strict=True):
            for annotation in (family[float], family[Any], family[float, Literal[2]], family[Any, Literal[2]]):
                with self.subTest(field=field, annotation=annotation):
                    bundle = ArrayBundle()
                    for name, (array, _, _) in zip(fields, families, strict=True):
                        setattr(bundle, name, array)
                    descriptor = bytes(bundle.__ctype__())
                    original = getattr(bundle, field)
                    grad_refs = (bundle._a_grad, bundle._ia_grad)
                    message = f"Struct field '{field}' expects a Warp .*array, got {type(annotation).__name__}"
                    try:
                        with self.assertRaisesRegex(TypeError, message):
                            setattr(bundle, field, annotation)
                    finally:
                        # Rejection must preserve both the Python owners and the native storage.
                        self.assertIs(getattr(bundle, field), original)
                        self.assertEqual(bytes(bundle.__ctype__()), descriptor)
                        self.assertIs(bundle._a_grad, grad_refs[0])
                        self.assertIs(bundle._ia_grad, grad_refs[1])

    def test_annotation_values_are_not_recorded_command_parameters(self):
        for (array, expected, kernel), family in zip(make_families(self.cpu), FAMILIES, strict=True):
            annotation = family[float]
            out = wp.empty(4, dtype=float, device=self.cpu)
            command = wp.launch(kernel, 4, inputs=[array], outputs=[out], device=self.cpu, record_cmd=True)
            message = f"argument 'a' expects an array of type .*passed value has type {type(annotation).__name__}"
            for parameter in ("name", "index"):
                with self.subTest(family=family.__name__, parameter=parameter):
                    with self.assertRaisesRegex(RuntimeError, message):
                        if parameter == "name":
                            command.set_param_by_name("a", annotation)
                        else:
                            command.set_param_at_index(0, annotation)
                    # Only launch after replacement was rejected, so the descriptor remains valid.
                    command.launch()
                    np.testing.assert_allclose(out.numpy(), expected * 2.0)
            with self.subTest(family=family.__name__, parameter="creation"):
                with self.assertRaisesRegex(RuntimeError, message):
                    wp.launch(kernel, 4, inputs=[annotation], outputs=[out], device=self.cpu, record_cmd=True)

    def test_subclass_instance_parameter_annotations_are_rejected(self):
        for annotation, _, _ in make_families(self.cpu):
            with self.subTest(family=type(annotation).__name__):

                def invalid_kernel(a: annotation):
                    pass

                def invalid_function(a: annotation):
                    return a

                class InvalidStruct:
                    arr: annotation

                for decorator, definition in (
                    (wp.kernel, invalid_kernel),
                    (wp.func, invalid_function),
                    (wp.struct, InvalidStruct),
                ):
                    with self.subTest(decorator=decorator.__name__):
                        with self.assertRaisesRegex(TypeError, "annotation"):
                            decorator(definition)

    def test_family_resolution_does_not_retain_dynamic_classes(self):
        def resolve():
            dynamic_cls = type("TemporaryArray", (TaggedArray,), {})
            array = dynamic_cls(tag="temporary", shape=1, dtype=float, device=self.cpu)
            self.assertIs(wptypes.concrete_array_type(array), wp.array)
            self.assertEqual(wptypes.infer_argument_types([array], [Any]), [wp.array[wp.float32]])
            return weakref.ref(dynamic_cls), weakref.ref(array)

        class_ref, array_ref = resolve()
        gc.collect()
        self.assertIsNone(array_ref())
        self.assertIsNone(class_ref())

    def test_custom_metaclasses_use_real_family_identity(self):
        class UnhashableMeta(type):
            __hash__ = None

        class EqualToArrayMeta(type):
            def __eq__(cls, other):
                return cls is other or other is wp.array

            def __hash__(cls):
                return hash(wp.array)

        for meta in (UnhashableMeta, EqualToArrayMeta):
            with self.subTest(metaclass=meta.__name__):
                classes = tuple(
                    meta(f"Meta{base.__name__}", (base,), {})
                    for base in (TaggedArray, TaggedIndexedArray, TaggedFabricArray, TaggedIndexedFabricArray)
                )
                for family, (array, _, kernel), field in zip(
                    FAMILIES, make_families(self.cpu, classes), ("a", "ia", "fa", "ifa"), strict=True
                ):
                    with self.subTest(family=family.__name__):
                        self.assertIsInstance(hash(array), int)
                        self.assertIs(wptypes.concrete_array_type(array), family)
                        before = constructor_calls.copy()
                        self.assertEqual(wptypes.infer_argument_types([array], [Any]), [family[wp.float32]])
                        self.assertEqual(constructor_calls, before)
                        self.assertEqual(
                            bytes(context.pack_arg(kernel, family[float], "a", array, self.cpu)),
                            bytes(array.__ctype__()),
                        )
                        bundle = ArrayBundle()
                        setattr(bundle, field, array)
                        self.assertIs(getattr(bundle, field), array)

                unrelated_cls = meta("Unrelated", (), {})
                unrelated = unrelated_cls()
                self.assertIs(wptypes.concrete_array_type(unrelated), unrelated_cls)
                self.assertFalse(wptypes.is_array(unrelated))
                with self.assertRaises(TypeError):
                    wptypes.infer_argument_types([unrelated], [Any])
                with self.assertRaisesRegex(RuntimeError, "expects an array of type"):
                    context.pack_arg(read_regular, wp.array[float], "a", unrelated, self.cpu)

    def test_exact_launch_packing_keeps_fast_path(self):
        array = wp.ones(2, dtype=float, device=self.cpu)
        # Count calls instead of making a timing assertion on a CI worker.
        with mock.patch.object(wptypes, "_resolve_array_family", wraps=wptypes._resolve_array_family) as resolve:
            self.assertIs(context.pack_arg(read_regular, wp.array[float], "a", array, self.cpu), array.__ctype__())
            resolve.assert_not_called()

    def test_exact_struct_fields_keep_fast_path(self):
        regular = wp.array([1.0, 2.0], dtype=float, device=self.cpu)
        indices = wp.array([1, 0], dtype=int, device=self.cpu)
        indexed = wp.indexedarray(regular, indices)
        fabric = wp.fabricarray(_create_fabric_array_interface(regular, "a", [2]), "a")
        indexed_fabric = wp.indexedfabricarray(fabric, indices)
        bundle = ArrayBundle()
        with (
            mock.patch.object(codegen, "_resolve_array_family", wraps=wptypes._resolve_array_family) as resolve,
            mock.patch.object(
                codegen, "_validate_array_subclass_descriptor", wraps=wptypes._validate_array_subclass_descriptor
            ) as validate,
        ):
            for field, array in zip(("a", "ia", "fa", "ifa"), (regular, indexed, fabric, indexed_fabric), strict=True):
                setattr(bundle, field, array)
                self.assertIs(getattr(bundle, field), array)
            resolve.assert_not_called()
            validate.assert_not_called()

    def test_scalar_function_calls_keep_fast_path(self):
        with mock.patch.object(wptypes, "_resolve_array_family", wraps=wptypes._resolve_array_family) as resolve:
            for value in (1, 2.0, True, int, float, wp.float32):
                self.assertIs(wptypes._normalize_array_argument_type(value), value)
            self.assertEqual(
                wptypes.infer_argument_types([1, 2.0, True], [Any, Any, Any]), [wp.int32, wp.float32, wp.bool]
            )
            resolve.assert_not_called()
        self.assertEqual(square_element(3.0), 9.0)
        self.assertEqual(identify_generic_array(3), 3)

    def test_cpu_apic_retains_subclass_storage(self):
        def record():
            data = TaggedArray([1.0, 2.0, 3.0, 4.0], tag="storage", dtype=float, device=self.cpu)
            indices = TaggedArray([3, 1, 0, 2], tag="indices", dtype=wp.int32, device=self.cpu)
            indexed = TaggedIndexedArray(tag="view", data=data, indices=indices)
            out = TaggedArray(tag="out", shape=4, dtype=float, device=self.cpu)
            refs = tuple(weakref.ref(array) for array in (data, indices, out))
            wp.launch(read_indexed, 4, inputs=[indexed], outputs=[out], device=self.cpu)
            with wp.ScopedCapture(device=self.cpu, apic=True, force_module_load=False) as capture:
                wp.launch(read_indexed, 4, inputs=[indexed], outputs=[out], device=self.cpu)
            return capture.graph, refs

        graph, refs = record()
        gc.collect()
        for ref in refs:
            self.assertIsNotNone(ref())
        refs[2]().zero_()
        wp.capture_launch(graph)
        np.testing.assert_allclose(refs[2]().numpy(), [8.0, 4.0, 2.0, 6.0])


# ---------------------------------------------------------------------------------------------
# Tests running on every test device
# ---------------------------------------------------------------------------------------------


def test_launch_concrete_and_generic_kernels(test, device):
    for array, expected, typed_kernel in make_deep_families(device):
        family = wptypes.concrete_array_type(array)
        for kernel in (typed_kernel, read_any, GENERIC_KERNEL_BY_FAMILY[family]):
            with test.subTest(cls=type(array).__name__, kernel=kernel.key):
                out = wp.empty(4, dtype=float, device=device)
                before = constructor_calls.copy()
                wp.launch(kernel, 4, inputs=[array], outputs=[out], device=device)
                assert_np_equal(out.numpy(), expected * 2.0)
                test.assertEqual(constructor_calls, before)


def test_launch_with_subclass_outputs(test, device):
    values = TaggedArray([2.0, 3.0], tag="sensor", dtype=wp.float32, device=device)
    result = TaggedArray(tag="result", shape=2, dtype=wp.float32, device=device)
    before = constructor_calls.copy()
    wp.launch(square, dim=values.size, inputs=[values], outputs=[result], device=device)
    test.assertEqual(constructor_calls, before)
    assert_np_equal(result.numpy(), np.array([4.0, 9.0]))
    test.assertEqual(result.tag, "result")

    # Allocation helpers accept subclass instances without invoking their constructors.
    fresh = wp.empty_like(values)
    test.assertIsInstance(fresh, wp.array)
    wp.launch(square, dim=values.size, inputs=[values], outputs=[fresh], device=device)
    assert_np_equal(fresh.numpy(), np.array([4.0, 9.0]))


def test_specialization_is_shared_with_parent(test, device):
    parent = wp.array([1.0, 2.0], dtype=float, device=device)
    child = TaggedArray([1.0, 2.0], tag="child", dtype=float, device=device)
    sibling = SecondTaggedArray([1.0, 2.0], tag="sibling", dtype=float, device=device)
    out = wp.empty(2, dtype=float, device=device)

    wp.launch(square, 2, inputs=[parent], outputs=[out], device=device)
    overload_count = len(square.overloads)
    wp.launch(square, 2, inputs=[child], outputs=[out], device=device)
    wp.launch(square, 2, inputs=[sibling], outputs=[out], device=device)
    test.assertEqual(len(square.overloads), overload_count)

    sig_parent = wptypes.get_signature(square.infer_argument_types([parent, out]))
    sig_child = wptypes.get_signature(square.infer_argument_types([child, out]))
    test.assertEqual(sig_parent, sig_child)


def test_explicit_overload_is_shared(test, device):
    overload = wp.overload(copy_generic, [wp.array[wp.float32], wp.array[wp.float32]])
    src = TaggedArray([1.0, 2.0], tag="src", dtype=float, device=device)
    dst = TaggedArray(tag="dst", shape=2, dtype=float, device=device)
    test.assertIs(copy_generic.add_overload(copy_generic.infer_argument_types([src, dst])), overload)
    wp.launch(copy_generic, 2, inputs=[src], outputs=[dst], device=device)
    assert_np_equal(dst.numpy(), np.array([1.0, 2.0]))


def test_validation_errors(test, device):
    ints = TaggedArray([1, 2], tag="int", dtype=wp.int32, device=device)
    twod = TaggedArray(np.ones((2, 2), np.float32), tag="2d", dtype=float, device=device)
    out = wp.zeros(2, dtype=float, device=device)

    with test.assertRaisesRegex(RuntimeError, "dtype"):
        wp.launch(read_regular, 2, inputs=[ints], outputs=[out], device=device)
    with test.assertRaisesRegex(RuntimeError, "dimension"):
        wp.launch(read_regular, 2, inputs=[twod], outputs=[out], device=device)
    with test.assertRaisesRegex(RuntimeError, "dimension"):
        context.pack_arg(read_regular, wp.array[float], "a", wp.clone(twod), device)

    families = make_families(device)
    for array, _, _ in families[1:]:
        with test.subTest(cls=type(array).__name__):
            with test.assertRaisesRegex(RuntimeError, "expects an array of type"):
                context.pack_arg(read_regular, wp.array[float], "a", array, device)
    with test.assertRaisesRegex(RuntimeError, "expects an array of type"):
        context.pack_arg(read_indexed, wp.indexedarray[float], "a", families[0][0], device)
    with test.assertRaisesRegex(RuntimeError, "expects an array of type"):
        context.pack_arg(read_indexed_fabric, wp.indexedfabricarray[float], "a", families[2][0], device)

    # a generic launch rejects a mismatching dimensionality at specialization time
    with test.assertRaisesRegex(TypeError, "type mismatch"):
        wp.launch(read_generic_regular, 2, inputs=[twod], outputs=[out], device=device)


def test_external_array_interface_objects(test, device):
    class NumPyLike:
        def __init__(self, array):
            self._array = array

        @property
        def __array_interface__(self):
            return self._array.__array_interface__

    values = np.arange(1, 5, dtype=np.float32)
    out = wp.empty(4, dtype=float, device=device)

    if device.is_cpu:
        wp.launch(read_regular, 4, inputs=[NumPyLike(values)], outputs=[out], device=device)
        assert_np_equal(out.numpy(), values * 2.0)
        wp.launch(read_regular, 4, inputs=[values], outputs=[out], device=device)
        assert_np_equal(out.numpy(), values * 2.0)
    else:
        source = wp.array(values, dtype=float, device=device)

        class CudaLike:
            @property
            def __cuda_array_interface__(self):
                return source.__cuda_array_interface__

        wp.launch(read_regular, 4, inputs=[CudaLike()], outputs=[out], device=device)
        assert_np_equal(out.numpy(), values * 2.0)

    class ClassSpoof:
        @property
        def __class__(self):
            return wp.array

        @property
        def __array_interface__(self):
            return values.__array_interface__

        @property
        def __cuda_array_interface__(self):
            return source.__cuda_array_interface__

        def __ctype__(self):
            raise AssertionError("An external interface provider was treated as a Warp array")

    spoof = ClassSpoof()
    test.assertIsInstance(spoof, wp.array)
    wp.launch(read_regular, 4, inputs=[spoof], outputs=[out], device=device)
    assert_np_equal(out.numpy(), values * 2.0)


def test_null_arrays_and_descriptors(test, device):
    out = wp.zeros(2, dtype=float, device=device)
    wp.launch(read_or_default, 2, inputs=[None], outputs=[out], device=device)
    assert_np_equal(out.numpy(), np.array([-1.0, -1.0]))

    source = TaggedArray([5.0, 6.0], tag="descriptor", dtype=float, device=device)
    descriptor = source.__ctype__()
    wp.launch(read_or_default, 2, inputs=[descriptor], outputs=[out], device=device)
    assert_np_equal(out.numpy(), np.array([5.0, 6.0]))


def test_gradients_and_tape(test, device):
    a = TaggedArray([1.0, 2.0, 3.0], tag="in", dtype=float, device=device, requires_grad=True)
    b = TaggedArray(tag="out", shape=3, dtype=float, device=device, requires_grad=True)

    # the native descriptor keeps the gradient pointer of the original array
    packed = context.pack_arg(square_float, wp.array[float], "a", a, device)
    test.assertEqual(packed.grad, a.grad.ptr)
    test.assertEqual(packed.ndim, 1)

    with wp.Tape() as tape:
        wp.launch(square_float, 3, inputs=[a], outputs=[b], device=device)
    test.assertIs(tape.launches[0][3][0], a)
    tape.backward(grads={b: wp.ones_like(b)})
    assert_np_equal(a.grad.numpy(), np.array([2.0, 4.0, 6.0]))
    test.assertIs(tape.gradients[a], a.grad)
    tape.zero()
    assert_np_equal(a.grad.numpy(), np.zeros(3))

    # Generic kernels are differentiable as well.
    a.grad.zero_()
    b.grad.zero_()
    with wp.Tape() as tape:
        wp.launch(square, 3, inputs=[a], outputs=[b], device=device)
    tape.backward(grads={b: wp.ones_like(b)})
    assert_np_equal(a.grad.numpy(), np.array([2.0, 4.0, 6.0]))


def test_overwrite_gradient_parity(test, device):
    def make_array(cls, **kwargs):
        if cls is wp.array:
            return cls(dtype=float, device=device, requires_grad=True, **kwargs)
        return cls(tag="gradient", dtype=float, device=device, requires_grad=True, **kwargs)

    def chain(cls, retain_grad):
        a = make_array(cls, data=[1.0, 1.0], retain_grad=retain_grad)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        packed = context.pack_arg(double_in_place, wp.array[float], "a", a, device)
        test.assertEqual(packed.grad, a.grad.ptr)
        test.assertEqual(packed.flags, a.__ctype__().flags)
        with wp.Tape() as tape:
            wp.launch(double_in_place, 2, inputs=[a], device=device)
            wp.launch(double_in_place, 2, inputs=[a], device=device)
            wp.launch(sum_values, 2, inputs=[a, loss], device=device)
        tape.backward(loss=loss)
        return a.grad.numpy()

    for retain_grad, expected in ((False, 4.0), (True, 9.0)):
        with test.subTest(retain_grad=retain_grad):
            parent = chain(wp.array, retain_grad)
            np.testing.assert_allclose(parent, expected)
            np.testing.assert_allclose(chain(TaggedArray, retain_grad), parent)

    for cls in (wp.array, TaggedArray):
        with test.subTest(cls=cls.__name__):
            first = make_array(cls, data=[1.0, 1.0])
            second = make_array(cls, data=[1.0, 1.0])
            result = make_array(cls, shape=2)
            with wp.Tape() as tape:
                wp.launch(multiply_two, 2, inputs=[first], outputs=[result], device=device)
                wp.launch(multiply_three, 2, inputs=[second], outputs=[result], device=device)
            tape.backward(grads={result: wp.ones_like(result)})
            for array, expected in ((first, 0.0), (second, 3.0), (result, 0.0)):
                np.testing.assert_allclose(array.grad.numpy(), expected)


def test_gradient_arrays_may_be_subclasses(test, device):
    a = TaggedArray([1.0, 2.0, 3.0], tag="in", dtype=float, device=device, requires_grad=True)
    b = TaggedArray(tag="out", shape=3, dtype=float, device=device, requires_grad=True)
    a.grad = TaggedArray([0.0, 0.0, 0.0], tag="in grad", dtype=float, device=device)
    b.grad = TaggedArray([1.0, 1.0, 1.0], tag="out grad", dtype=float, device=device)
    wp.launch(square_float, 3, inputs=[a], outputs=[b], device=device)
    wp.launch(
        square_float, 3, inputs=[a], outputs=[b], adj_inputs=[a.grad], adj_outputs=[b.grad], adjoint=True, device=device
    )
    assert_np_equal(a.grad.numpy(), np.array([2.0, 4.0, 6.0]))

    # an indexed array's adjoint is a regular array, whatever the family of the forward argument
    adjoint = TaggedArray([0.0] * 4, tag="adjoint", dtype=float, device=device)
    packed = context.pack_arg(read_indexed, wp.indexedarray[float], "a", adjoint, device, adjoint=True)
    test.assertEqual(packed.data, adjoint.ptr)
    test.assertEqual(tuple(packed.shape)[: packed.ndim], adjoint.shape)


def test_gradcheck_and_jacobian(test, device):
    source = TaggedArray([1.0, 2.0, 3.0], tag="gradcheck input", dtype=float, device=device, requires_grad=True)
    output = SecondTaggedArray(tag="gradcheck output", shape=3, dtype=float, device=device, requires_grad=True)
    test.assertTrue(
        autograd.gradcheck(
            square_float,
            dim=3,
            inputs=[source],
            outputs=[output],
            eps=1.0e-3,
            show_summary=False,
            device=device,
        )
    )
    jacobians = autograd.jacobian(square_float, dim=3, inputs=[source], outputs=[output], device=device)
    np.testing.assert_allclose(jacobians[0, 0].numpy(), np.diag([2.0, 4.0, 6.0]))


def test_array_operators(test, device):
    for index in range(len(FAMILIES)):
        # The family views share storage; use a fresh owner before each in-place operation.
        families = make_families(device)
        array, expected, _ = families[index]
        with test.subTest(family=type(array).__name__):
            before = constructor_calls.copy()
            np.testing.assert_allclose((array - 2.0).numpy(), expected - 2.0)
            np.testing.assert_allclose((12.0 / array).numpy(), 12.0 / expected)
            np.testing.assert_allclose((-array).numpy(), -expected)
            original = array
            array += 2.0
            test.assertIs(array, original)
            np.testing.assert_allclose(array.numpy(), expected + 2.0)
            test.assertEqual(constructor_calls, before)


def test_fem_interpolate_cache(test, device):
    domain = fem.Cells(fem.Grid2D(res=(2, 2)))
    destinations = (
        wp.empty(4, dtype=float, device=device),
        LeftArray(tag="left destination", shape=4, dtype=float, device=device),
        RightArray(tag="right destination", shape=4, dtype=float, device=device),
    )
    kernels = []
    for destination in destinations:
        fem.interpolate(subclass_sample_value, dest=destination, at=domain, dim=4, device=device)
        np.testing.assert_allclose(destination.numpy(), [1.0, 2.0, 3.0, 4.0])
        test.assertEqual(len(subclass_sample_value.cached_kernels), 1)
        kernels.append(next(iter(subclass_sample_value.cached_kernels.values()))[0])
    test.assertIs(kernels[0], kernels[1])
    test.assertIs(kernels[0], kernels[2])


def test_map_inputs_and_outputs(test, device):
    for array, expected, _ in make_families(device):
        with test.subTest(cls=type(array).__name__):
            before = constructor_calls.copy()
            result = wp.map(square_element, array)
            test.assertIsInstance(result, wp.array)
            assert_np_equal(result.numpy(), expected**2)

            explicit = TaggedArray(tag="output", shape=4, dtype=float, device=device)
            test.assertIs(wp.map(square_element, array, out=explicit), explicit)
            test.assertEqual(constructor_calls, before + collections.Counter({"TaggedArray": 1}))
            assert_np_equal(explicit.numpy(), expected**2)

    for output, _, _ in make_families(device):
        with test.subTest(output=type(output).__name__):
            source = wp.array([1.0, 2.0, 3.0, 4.0], dtype=float, device=device)
            test.assertIs(wp.map(square_element, source, out=output), output)
            assert_np_equal(output.numpy(), np.array([1.0, 4.0, 9.0, 16.0]))

    # broadcasting, tuple outputs and the operators that are implemented with map
    a = TaggedArray(np.arange(6, dtype=np.float32).reshape(2, 3), tag="a", dtype=float, device=device)
    b = SecondTaggedArray(np.array([[10.0], [20.0]], dtype=np.float32), tag="b", dtype=float, device=device)
    total = wp.map(add_elements, a, b)
    assert_np_equal(total.numpy(), np.arange(6).reshape(2, 3) + np.array([[10.0], [20.0]]))

    first = TaggedArray(tag="first", shape=(2, 3), dtype=float, device=device)
    second = SecondTaggedArray(tag="second", shape=(2, 3), dtype=float, device=device)
    results = wp.map(split_element, a, out=[first, second])
    test.assertIs(results[0], first)
    test.assertIs(results[1], second)
    assert_np_equal(first.numpy(), a.numpy() + 1.0)
    assert_np_equal(second.numpy(), a.numpy() * 2.0)
    assert_np_equal((a + a).numpy(), a.numpy() * 2.0)


def test_map_kernels_are_cached_by_native_family(test, device):
    a = TaggedArray([1.0, 2.0], tag="first", dtype=float, device=device)
    b = SecondTaggedArray([3.0, 4.0], tag="second", dtype=float, device=device)
    parent = wp.clone(a)
    tagged_out = TaggedArray(tag="out", shape=2, dtype=float, device=device)
    first = wp.map(square_element, a, return_kernel=True)
    test.assertIs(wp.map(square_element, b, out=tagged_out, return_kernel=True), first)
    test.assertIs(wp.map(square_element, parent, out=parent, return_kernel=True), first)

    # unrelated families remain distinct even when the Python classes carry the same name
    same_name_regular = type("Same", (TaggedArray,), {})(tag="a", data=[1.0, 2.0, 3.0, 4.0], dtype=float, device=device)
    same_name_indexed = type("Same", (TaggedIndexedArray,), {})(
        tag="i", data=same_name_regular, indices=wp.array([3, 1, 0, 2], dtype=wp.int32, device=device)
    )
    regular_kernel = wp.map(square_element, same_name_regular, return_kernel=True)
    indexed_kernel = wp.map(square_element, same_name_indexed, return_kernel=True)
    test.assertIsNot(regular_kernel, indexed_kernel)
    assert_np_equal(wp.map(square_element, same_name_indexed).numpy(), np.array([16.0, 4.0, 1.0, 9.0]))
    assert_np_equal(wp.map(square_element, same_name_regular).numpy(), np.array([1.0, 4.0, 9.0, 16.0]))


def test_map_literal_names(test, device):
    @wp.func
    def Literal(value: float):
        return value + 1.0

    @wp.func
    def named_argument(Literal: float):
        return Literal * 2.0

    source = TaggedArray(np.arange(6, dtype=np.float32).reshape(2, 3), tag="rank two", dtype=float, device=device)
    np.testing.assert_allclose(wp.map(Literal, source).numpy(), source.numpy() + 1.0)
    indexed, expected, _ = make_families(device)[1]
    np.testing.assert_allclose(wp.map(named_argument, indexed).numpy(), expected * 2.0)


def test_python_function_overloads(test, device):
    a = TaggedArray([1.0, 2.0], tag="first", dtype=float, device=device)
    b = SecondTaggedArray([3.0, 4.0], tag="second", dtype=float, device=device)
    parent = wp.clone(a)
    ints = TaggedArray([1, 2], tag="int overload", dtype=wp.int32, device=device)

    # explicit overloads select on the native type and return the original object
    for array in (a, b, parent, ints):
        test.assertIs(identify_array(array), array)

    # inferred overloads are shared between the parent and its subclasses
    test.assertIs(identify_generic_array(a), a)
    count = len(identify_generic_array.user_overloads)
    test.assertIs(identify_generic_array(b), b)
    test.assertIs(identify_generic_array(parent), parent)
    test.assertEqual(len(identify_generic_array.user_overloads), count)
    # one overload for float32 vectors, whose cached type is a lightweight annotation and not an array object
    array_overloads = [
        overload
        for overload in identify_generic_array.user_overloads.values()
        if wptypes.is_array(overload.input_types["a"]) and overload.input_types["a"] == wp.array[wp.float32]
    ]
    test.assertEqual(len(array_overloads), 1)
    test.assertIsInstance(array_overloads[0].input_types["a"], wptypes._ArrayAnnotationBase)

    indexed = make_families(device)[1][0]
    test.assertIs(identify_indexed(indexed), indexed)


def test_sequence_array_overloads(test, device):
    class SequenceArray(TaggedArray, Sequence):
        def __getitem__(self, key):
            raise AssertionError("Overload resolution must not iterate a Warp array")

    array = SequenceArray(tag="sequence", data=[1.0, 2.0], dtype=float, device=device)
    test.assertIsInstance(array, Sequence)
    test.assertIs(identify_array(array), array)
    test.assertIs(identify_generic_array(array), array)


def test_struct_fields(test, device):
    families = make_families(device)
    bundle = ArrayBundle()
    for name, (array, _, _) in zip(("a", "ia", "fa", "ifa"), families, strict=True):
        setattr(bundle, name, array)
        test.assertIs(getattr(bundle, name), array)
    out = wp.empty(4, dtype=float, device=device)
    wp.launch(read_bundle, 4, inputs=[bundle], outputs=[out], device=device)
    assert_np_equal(out.numpy(), sum(expected for _, expected, _ in families))

    arr = TaggedArray([1.0, 2.0], tag="nested", dtype=float, device=device)
    inner = InnerStruct()
    inner.arr = arr
    outer = OuterStruct()
    outer.inner = inner
    outer.scale = 3.0
    test.assertIs(outer.inner.arr, arr)
    out = wp.empty(2, dtype=float, device=device)
    wp.launch(read_nested, 2, inputs=[outer], outputs=[out], device=device)
    assert_np_equal(out.numpy(), np.array([3.0, 6.0]))

    with test.assertRaises(TypeError):
        bundle.a = families[1][0]


def test_struct_descriptor_errors_preserve_field(test, device):
    class BrokenDescriptor(TaggedArray):
        def __ctype__(self):
            raise KeyError("descriptor failure")

    original = TaggedArray([1.0, 2.0], tag="original", dtype=float, device=device)
    bad = BrokenDescriptor([3.0, 4.0], tag="bad", dtype=float, device=device)
    bundle = ArrayBundle()
    bundle.a = original
    previous = bytes(bundle.__ctype__())
    with test.assertRaisesRegex(TypeError, "Struct field 'a'.*descriptor failure"):
        bundle.a = bad
    test.assertIs(bundle.a, original)
    test.assertEqual(bytes(bundle.__ctype__()), previous)
    owner = weakref.ref(original)
    del original
    gc.collect()
    test.assertIs(bundle.a, owner())


def test_subclass_descriptor_validation(test, device):
    def descriptor_subclass(family, base, mode):
        def ctype(self):
            self.descriptor_calls += 1
            native = family.__ctype__(self)
            descriptor_cls = type(native)
            if mode == "wrong_class":
                return ctypes.c_uint64(0)
            if mode in ("wrong_class_equal_size", "spoofed_class_equal_size"):
                attributes = {"_fields_": [("data", ctypes.c_byte * ctypes.sizeof(native))]}
                if mode == "spoofed_class_equal_size":
                    attributes["__class__"] = property(lambda _: descriptor_cls)
                wrong_class = type(
                    "UnrelatedDescriptor",
                    (ctypes.Structure,),
                    attributes,
                )
                return wrong_class.from_buffer_copy(native)
            if mode == "wrong_size":
                extra = type("ExtendedDescriptor", (descriptor_cls,), {"_fields_": [("extra", ctypes.c_uint64)]})
                return extra.__new__(extra)
            if mode == "wrong_ndim":
                descriptor = descriptor_cls.from_buffer_copy(native)
                if family is wp.indexedarray:
                    descriptor.data.ndim = 2
                else:
                    descriptor.ndim = 2
                return descriptor
            if mode == "hidden_wrong_ndim":
                if family is wp.array:
                    native.ndim = 2
                    hidden = type("HiddenRankDescriptor", (descriptor_cls,), {"ndim": property(lambda _: self.ndim)})
                else:
                    native.data.ndim = 2
                    data_cls = type(native.data)
                    hidden_data = type("HiddenRankData", (data_cls,), {"ndim": property(lambda _: self.ndim)})
                    hidden = type(
                        "HiddenRankDescriptor",
                        (descriptor_cls,),
                        {"data": property(hidden_data.from_buffer)},
                    )
                return hidden.from_buffer_copy(native)
            compatible = type("CompatibleDescriptor", (descriptor_cls,), {})
            return compatible.from_buffer_copy(native)

        return type("DescriptorArray", (base,), {"__ctype__": ctype, "descriptor_calls": 0})

    bases = (TaggedArray, TaggedIndexedArray, TaggedFabricArray, TaggedIndexedFabricArray)
    expected = np.arange(1, 5, dtype=np.float32)
    storage = wp.array(expected, dtype=float, device=device)
    indices = wp.array([3, 1, 0, 2], dtype=wp.int32, device=device)
    interface = _create_fabric_array_interface(storage, "a", [2, 2])
    fabric_storage = wp.fabricarray(interface, "a")
    for mode in ("wrong_class", "wrong_class_equal_size", "spoofed_class_equal_size", "wrong_size", "compatible"):
        classes = tuple(descriptor_subclass(family, base, mode) for family, base in zip(FAMILIES, bases, strict=True))
        # Each invalid descriptor has valid backing storage. In particular an
        # invalid Fabric descriptor must not be embedded during construction of
        # the separate indexed Fabric case.
        values = (
            (classes[0](tag="descriptor", data=expected, dtype=float, device=device), expected, read_regular),
            (classes[1](tag="descriptor", data=storage, indices=indices), expected[[3, 1, 0, 2]], read_indexed),
            (classes[2](tag="descriptor", data=interface, attrib="a"), expected, read_fabric),
            (
                classes[3](tag="descriptor", fa=fabric_storage, indices=indices),
                expected[[3, 1, 0, 2]],
                read_indexed_fabric,
            ),
        )
        for family, (array, expected_values, kernel), field in zip(
            FAMILIES, values, ("a", "ia", "fa", "ifa"), strict=True
        ):
            with test.subTest(family=family.__name__, descriptor=mode):
                if mode == "compatible":
                    descriptor = context.pack_arg(kernel, family[float], "a", array, device)
                    test.assertEqual(array.descriptor_calls, 1)
                    test.assertIs(type(descriptor), type(family.__ctype__(array)))
                    test.assertEqual(ctypes.sizeof(descriptor), ctypes.sizeof(family.__ctype__(array)))
                    bundle = ArrayBundle()
                    setattr(bundle, field, array)
                    test.assertEqual(array.descriptor_calls, 2)
                    test.assertIs(getattr(bundle, field), array)
                    out = wp.empty(4, dtype=float, device=device)
                    wp.launch(kernel, 4, inputs=[array], outputs=[out], device=device)
                    np.testing.assert_allclose(out.numpy(), expected_values * 2.0)
                    cache_size = len(kernel._invoke_cache)
                    for _ in range(3):
                        wp.launch(kernel, 4, inputs=[array], outputs=[out], device=device)
                    np.testing.assert_allclose(out.numpy(), expected_values * 2.0)
                    test.assertEqual(len(kernel._invoke_cache), cache_size)
                else:
                    with test.assertRaisesRegex(RuntimeError, "__ctype__|incompatible"):
                        context.pack_arg(kernel, family[float], "a", array, device)
                    test.assertEqual(array.descriptor_calls, 1)
                    with test.assertRaisesRegex(TypeError, f"Struct field '{field}'"):
                        setattr(ArrayBundle(), field, array)

    wrong_ndim_cls = descriptor_subclass(wp.array, TaggedArray, "wrong_ndim")
    wrong_ndim = wrong_ndim_cls(tag="invalid ndim", shape=4, dtype=float, device=device)
    for annotation, adjoint in ((wp.array[float], False), (wp.indexedarray[float], True)):
        with test.subTest(adjoint=adjoint):
            with test.assertRaisesRegex(RuntimeError, "descriptor.*dimension"):
                context.pack_arg(read_regular, annotation, "a", wrong_ndim, device, adjoint=adjoint)
    with test.assertRaisesRegex(TypeError, "Struct field 'a'"):
        ArrayBundle().a = wrong_ndim

    wrong_indexed_cls = descriptor_subclass(wp.indexedarray, TaggedIndexedArray, "wrong_ndim")
    wrong_indexed = wrong_indexed_cls(tag="invalid embedded ndim", data=storage, indices=indices)
    malformed_backing = TaggedIndexedArray(tag="invalid backing ndim", data=wrong_ndim, indices=indices)
    valid_indexed = TaggedIndexedArray(tag="valid", data=storage, indices=indices)
    out = wp.empty(4, dtype=float, device=device)
    for array, family, field, kernel, valid in (
        (wrong_ndim, wp.array, "a", read_regular, storage),
        (wrong_indexed, wp.indexedarray, "ia", read_indexed, valid_indexed),
        (malformed_backing, wp.indexedarray, "ia", read_indexed, valid_indexed),
    ):
        with test.subTest(family=family.__name__, malformed_backing=array is malformed_backing):
            with test.assertRaisesRegex(RuntimeError, "argument 'a'.*descriptor.*dimension"):
                context.pack_arg(kernel, family[float], "a", array, device)
            with test.assertRaisesRegex(TypeError, f"Struct field '{field}'.*descriptor.*dimension"):
                setattr(ArrayBundle(), field, array)
            command = wp.launch(kernel, 4, inputs=[valid], outputs=[out], device=device, record_cmd=True)
            with test.assertRaisesRegex(RuntimeError, "argument 'a'.*descriptor.*dimension"):
                command.set_param_by_name("a", array)

    for family, base, kernel in (
        (wp.array, TaggedArray, read_regular),
        (wp.indexedarray, TaggedIndexedArray, read_indexed),
    ):
        hidden_cls = descriptor_subclass(family, base, "hidden_wrong_ndim")
        kwargs = (
            {"data": storage, "indices": indices}
            if family is wp.indexedarray
            else {"shape": 4, "dtype": float, "device": device}
        )
        hidden_rank = hidden_cls(tag="hidden native rank", **kwargs)
        with test.subTest(family=family.__name__, descriptor="hidden_wrong_ndim"):
            with test.assertRaisesRegex(RuntimeError, "descriptor.*dimension"):
                context.pack_arg(kernel, family[float], "a", hidden_rank, device)


def test_wrong_family_does_not_use_external_interface(test, device):
    class IndexedWithInterface(TaggedIndexedArray):
        @property
        def __array_interface__(self):
            return self.data.__array_interface__

        @property
        def __cuda_array_interface__(self):
            return self.data.__cuda_array_interface__

    values = TaggedArray([1.0, 2.0], tag="backing", dtype=float, device=device)
    indexed = IndexedWithInterface(tag="indexed", data=values, indices=wp.array([1, 0], dtype=wp.int32, device=device))
    with test.assertRaisesRegex(RuntimeError, "expects an array of type"):
        context.pack_arg(read_regular, wp.array[float], "a", indexed, device)


def test_function_defaults(test, device):
    a = TaggedArray([1.0, 2.0], tag="defaults", dtype=float, device=device)
    out = wp.empty(2, dtype=float, device=device)
    wp.launch(read_with_default_argument, 2, inputs=[a], outputs=[out], device=device)
    assert_np_equal(out.numpy(), np.array([3.0, 6.0]))


def test_array_views_in_kernels(test, device):
    values = TaggedArray(np.arange(6, dtype=np.float32).reshape(3, 2), tag="views", dtype=float, device=device)
    out = TaggedArray(tag="views out", shape=3, dtype=float, device=device)
    before = constructor_calls.copy()
    # Explicit live descriptors must be normalized before view inference too.
    overload = wp.overload(row_view_kernel, [values, out])
    test.assertIsInstance(overload.adj.args[0].type, wptypes._ArrayAnnotationBase)
    wp.launch(row_view_kernel, 3, inputs=[values], outputs=[out], device=device)
    test.assertEqual(constructor_calls, before)
    assert_np_equal(out.numpy(), np.array([0.0, 2.0, 4.0]))


def test_vector_dtypes_and_dimensions(test, device):
    vectors = TaggedArray(np.ones((3, 3), np.float32), tag="vec", dtype=wp.vec3, device=device)
    out = wp.empty(3, dtype=float, device=device)
    wp.launch(read_vec3, 3, inputs=[vectors], outputs=[out], device=device)
    assert_np_equal(out.numpy(), np.array([3.0, 3.0, 3.0]))

    matrix = TaggedArray(np.arange(6, dtype=np.float32).reshape(3, 2), tag="2d", dtype=float, device=device)
    wp.launch(read_2d, 3, inputs=[matrix], outputs=[out], device=device)
    assert_np_equal(out.numpy(), np.array([0.0, 2.0, 4.0]))

    vector_out = TaggedArray(tag="generic out", shape=3, dtype=wp.vec3, device=device)
    generic_in = TaggedArray(np.ones((3, 3), np.float32), tag="generic in", dtype=wp.vec3, device=device)
    wp.launch(copy_generic, 3, inputs=[generic_in], outputs=[vector_out], device=device)
    assert_np_equal(vector_out.numpy(), np.ones((3, 3)))

    # Inherited interop interfaces can reinterpret these layouts. A recognized
    # Warp subclass must validate its actual dtype instead of taking that path.
    scalar_matrix = TaggedArray(np.arange(6, dtype=np.float32).reshape(2, 3), tag="scalar", dtype=float, device=device)
    vector_array = TaggedArray(np.arange(6, dtype=np.float32).reshape(2, 3), tag="vector", dtype=wp.vec3, device=device)
    for annotation, array in ((wp.array[wp.vec3], scalar_matrix), (wp.array2d[float], vector_array)):
        with test.subTest(dtype=array.dtype):
            with test.assertRaisesRegex(RuntimeError, "dtype"):
                context.pack_arg(read_regular, annotation, "a", array, device)


def test_struct_and_bfloat16_dtypes(test, device):
    host = np.zeros(2, dtype=Record.numpy_dtype())
    host["v"] = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
    host["w"] = [4.0, 5.0]
    records = TaggedArray(host, tag="records", dtype=Record, device=device)
    out = wp.empty(2, dtype=float, device=device)
    wp.launch(read_record, 2, inputs=[records], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [10.0, 14.0])

    values = TaggedArray(tag="bfloat16", shape=2, dtype=wp.bfloat16, device=device)
    bits = np.array([1.0, 2.0], dtype=np.float32).view(np.uint32)
    src = wp.array((bits >> 16).astype(np.uint16), dtype=wp.uint16, device=device)
    wp.copy(values.view(wp.uint16), src)
    wp.launch(read_bfloat16, 2, inputs=[values], outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), [1.0, 2.0])


def test_host_operations_keep_identity(test, device):
    families = make_families(device)
    array = families[0][0]
    for source, expected, _ in families:
        destination = TaggedArray(tag="copy", shape=4, dtype=float, device=device)
        wp.copy(destination, source)
        assert_np_equal(destination.numpy(), expected)
        for result in (wp.clone(source), wp.empty_like(source), wp.zeros_like(source)):
            test.assertIsInstance(result, wp.array)

    view_owner = TaggedArray([1.0, 2.0, 3.0, 4.0], tag="view owner", dtype=float, device=device)
    owner_ref = weakref.ref(view_owner)
    view = view_owner.reshape((2, 2))
    strided = view_owner[::2]
    test.assertIsInstance(view, wp.array)
    test.assertIsInstance(strided, wp.array)
    del view_owner
    gc.collect()
    test.assertIsNotNone(owner_ref())
    assert_np_equal(view.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert_np_equal(strided.numpy(), np.array([1.0, 3.0]))
    del view, strided
    gc.collect()
    test.assertIsNone(owner_ref())

    # existing no-op paths keep returning the original object
    test.assertIs(array.to(device), array)
    test.assertIs(array.contiguous(), array)

    array.fill_(7.0)
    assert_np_equal(array.numpy(), np.full(4, 7.0))
    array.assign(np.arange(4, dtype=np.float32))
    test.assertEqual(array.list(), [0.0, 1.0, 2.0, 3.0])

    source = TaggedArray([1.0, 2.0, 3.0], tag="scan", dtype=float, device=device)
    scanned = TaggedArray(tag="scan out", shape=3, dtype=float, device=device)
    wp.utils.array_scan(source, scanned)
    assert_np_equal(scanned.numpy(), np.array([1.0, 3.0, 6.0]))
    converted = TaggedArray(tag="cast out", shape=3, dtype=wp.int32, device=device)
    wp.utils.array_cast(source, converted)
    assert_np_equal(converted.numpy(), np.array([1, 2, 3]))


def test_strided_empty_and_high_rank_arrays(test, device):
    base = wp.array(np.arange(1, 9, dtype=np.float32), dtype=float, device=device)
    view = base[::2]
    strided = TaggedArray(
        ptr=view.ptr,
        dtype=float,
        shape=view.shape,
        strides=view.strides,
        capacity=base.capacity,
        device=device,
        tag="strided",
    )
    test.assertFalse(strided.is_contiguous)
    out = wp.zeros(4, dtype=float, device=device)
    wp.launch(read_regular, 4, inputs=[strided], outputs=[out], device=device)
    assert_np_equal(out.numpy(), view.numpy() * 2.0)
    wp.launch(square, 4, inputs=[strided], outputs=[out], device=device)
    assert_np_equal(out.numpy(), view.numpy() ** 2)
    destination = TaggedArray(tag="destination", shape=4, dtype=float, device=device)
    wp.copy(destination, strided)
    assert_np_equal(destination.numpy(), view.numpy())

    empty = TaggedArray(tag="empty", shape=0, dtype=float, device=device)
    wp.launch(read_regular, 0, inputs=[empty], outputs=[empty], device=device)
    wp.launch(square, 0, inputs=[empty], outputs=[empty], device=device)
    test.assertEqual(wp.map(square_element, empty).shape, (0,))

    four_d = TaggedArray(np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2), tag="4d", dtype=float, device=device)
    assert_np_equal(wp.map(square_element, four_d).numpy(), np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) ** 2)


def test_launch_command_parameter_replacement(test, device):
    a = TaggedArray([1.0, 2.0], tag="a", dtype=float, device=device)
    b = SecondTaggedArray([10.0, 20.0], tag="b", dtype=float, device=device)
    out = TaggedArray(tag="out", shape=2, dtype=float, device=device)
    command = wp.launch(read_regular, 2, inputs=[a], outputs=[out], device=device, record_cmd=True)
    source_ref = weakref.ref(a)
    del a
    gc.collect()
    test.assertIsNotNone(source_ref())
    command.launch()
    assert_np_equal(out.numpy(), np.array([2.0, 4.0]))
    command.set_param_by_name("a", b)
    command.launch()
    assert_np_equal(out.numpy(), np.array([20.0, 40.0]))
    with test.assertRaisesRegex(RuntimeError, "expects an array of type"):
        command.set_param_at_index(0, make_families(device)[1][0])


def test_library_caches_do_not_retain_arrays(test, device):
    out = wp.empty(2, dtype=float, device=device)

    def check(make_array, use):
        array = make_array()
        reference = weakref.ref(array)
        use(array)
        del array
        gc.collect()
        test.assertIsNone(reference())

    check(
        lambda: TaggedArray([1.0, 2.0], tag="r", dtype=float, device=device),
        lambda a: wp.launch(square, 2, inputs=[a], outputs=[out], device=device),
    )
    check(
        lambda: TaggedArray([1.0, 2.0], tag="r", dtype=float, device=device),
        lambda a: (identify_generic_array(a), identify_array(a)),
    )
    check(
        lambda: SecondTaggedArray([1.0, 2.0], tag="r", dtype=float, device=device),
        lambda a: wp.map(square_element, a),
    )
    check(
        lambda: make_families(device)[1][0],
        lambda a: wp.map(square_element, a),
    )
    check(
        lambda: TaggedArray([1.0, 2.0], tag="live overload", dtype=float, device=device),
        lambda a: wp.overload(copy_generic, [a, out]),
    )


def test_sgd_live_overload_descriptors(test, device):
    from warp.optim import SGD  # noqa: PLC0415

    parameter = TaggedArray([1.0, 2.0], tag="parameter", dtype=float, device=device)
    gradient = TaggedArray([0.5, 1.0], tag="gradient", dtype=float, device=device)
    parameter_ref = weakref.ref(parameter)
    optimizer = SGD([parameter], lr=0.2)
    test.assertIs(optimizer.params[0], parameter)
    optimizer.step([gradient])
    np.testing.assert_allclose(parameter.numpy(), [0.9, 1.8])
    del parameter, optimizer
    gc.collect()
    test.assertIsNone(parameter_ref())


def test_class_reassignment_uses_real_family(test, device):
    def forbidden_constructor(self, *args, **kwargs):
        raise AssertionError("Warp called the constructor of a reassigned array subclass")

    families = make_families(device)
    families[0][0].requires_grad = True
    families[0][0].retain_grad = True
    for family, (source, expected, kernel) in zip(FAMILIES, families, strict=True):
        source.__class__ = family
        source.metadata = object()
        metadata = source.metadata
        descriptor = bytes(source.__ctype__())
        cls = type(f"Reassigned{family.__name__}", (family,), {"__init__": forbidden_constructor})
        for state, target in (("promoted", cls), ("restored", family)):
            with test.subTest(family=family.__name__, state=state):
                source.__class__ = target
                test.assertIs(wptypes.concrete_array_type(source), family)
                destination = wp.empty(4, dtype=float, device=device)
                for reader in (kernel, GENERIC_KERNEL_BY_FAMILY[family]):
                    wp.launch(reader, 4, inputs=[source], outputs=[destination], device=device)
                    np.testing.assert_allclose(destination.numpy(), expected * 2.0)
                np.testing.assert_allclose(wp.map(square_element, source).numpy(), expected**2)
                test.assertIs(source.metadata, metadata)
                test.assertEqual(bytes(context.pack_arg(kernel, family[float], "a", source, device)), descriptor)


def test_subprocess_specialization_cache(test, device):
    source_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with tempfile.TemporaryDirectory(prefix="wp_array_subclass_") as cache_dir:
        records = []
        for cache_state in ("cold", "warm"):
            result = run_python_subprocess(
                "import sys; sys.path.insert(0, sys.argv.pop(1)); "
                "from warp.tests.aux_array_subclass_cache import main; main(*sys.argv[1:])",
                source_root,
                device.alias,
                cache_state,
                timeout=180,
                hide_gpu=device.is_cpu,
                env_updates={"WARP_CACHE_PATH": cache_dir, "PYTHONFAULTHANDLER": "1"},
            )
            test.assertEqual(
                result.returncode,
                0,
                f"{cache_state.capitalize()} cache subprocess failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            marker = "ARRAY_SUBCLASS_CACHE_RESULT="
            lines = [line.removeprefix(marker) for line in result.stdout.splitlines() if line.startswith(marker)]
            test.assertEqual(len(lines), 1, result.stdout)
            records.append(json.loads(lines[0]))

        test.assertGreater(records[0]["native_builds"], 0)
        test.assertEqual(records[1]["native_builds"], 0)
        test.assertEqual(records[0]["signatures"], records[1]["signatures"])
        test.assertEqual(records[0]["module_hash"], records[1]["module_hash"])
        test.assertEqual(records[0]["wrong_family_rejections"], 3)
        test.assertEqual(records[1]["wrong_family_rejections"], 3)


def test_no_constructor_calls(test, device):
    class StrictArray(TaggedArray):
        allowed = True

        def __init__(self, *args, **kwargs):
            if not StrictArray.allowed:
                raise AssertionError("Warp called the constructor of a user subclass")
            super().__init__(*args, **kwargs)

    values = StrictArray(tag="in", data=[1.0, 2.0], dtype=float, device=device)
    result = StrictArray(tag="out", shape=2, dtype=float, device=device)
    StrictArray.allowed = False
    try:
        wp.launch(square, 2, inputs=[values], outputs=[result], device=device)
        wp.launch(read_any, 2, inputs=[values], outputs=[wp.empty(2, dtype=float, device=device)], device=device)
        wp.map(square_element, values)
        wp.map(square_element, values, out=result)
        identify_generic_array(values)
        identify_array(values)
        with wp.Tape():
            wp.launch(square_float, 2, inputs=[values], outputs=[result], device=device)
        context.pack_arg(read_regular, wp.array[float], "a", values, device)
        wp.copy(result, values)
        wp.clone(values)
        wp.empty_like(values)
        values.numpy()
        _ = values + values
    finally:
        StrictArray.allowed = True


def test_exact_arrays_are_unchanged(test, device):
    a = wp.array([1.0, 2.0, 3.0, 4.0], dtype=float, device=device)
    out = wp.empty(4, dtype=float, device=device)
    for kernel in (read_regular, read_any, read_generic_regular):
        wp.launch(kernel, 4, inputs=[a], outputs=[out], device=device)
        assert_np_equal(out.numpy(), a.numpy() * 2.0)
    assert_np_equal(wp.map(square_element, a).numpy(), a.numpy() ** 2)
    test.assertIs(identify_array(a), a)
    with test.assertRaisesRegex(RuntimeError, "dtype"):
        wp.launch(read_regular, 2, inputs=[wp.zeros(2, dtype=wp.int32, device=device)], outputs=[out], device=device)


def test_cross_device_access_modes(test, device):
    for source_device, target_device in ((device, wp.get_device("cpu")), (wp.get_device("cpu"), device)):
        storage = wp.array([1.0, 2.0], dtype=float, device=source_device)
        tagged = TaggedArray([1.0, 2.0], tag="off-device", dtype=float, device=source_device)
        indices = wp.array([1, 0], dtype=int, device=source_device)
        families = (
            (storage, tagged, wp.array, read_regular),
            (
                wp.indexedarray(storage, indices),
                TaggedIndexedArray(tag="off-device indexed", data=tagged, indices=indices),
                wp.indexedarray,
                read_indexed,
            ),
        )
        for mode in (wp.config.LaunchArrayAccessMode.CHECKED, wp.config.LaunchArrayAccessMode.STRICT):
            for parent, subclass, family, kernel in families:
                with test.subTest(source=source_device, target=target_device, family=family.__name__, mode=mode):
                    with mock.patch.object(wp.config, "launch_array_access_mode", mode):
                        try:
                            context.pack_arg(kernel, family[float], "a", parent, target_device)
                        except RuntimeError as error:
                            with test.assertRaises(RuntimeError) as actual:
                                context.pack_arg(kernel, family[float], "a", subclass, target_device)
                            test.assertEqual(str(actual.exception), str(error))
                            test.assertIn("argument 'a'", str(error))
                        else:
                            # Some CUDA platforms can access unpinned host allocations.
                            context.pack_arg(kernel, family[float], "a", subclass, target_device)

    if device.is_uva:
        pinned = TaggedArray([1.0, 2.0], tag="pinned host", dtype=float, device="cpu", pinned=True)
        out = wp.empty(2, dtype=float, device=device)
        with mock.patch.object(wp.config, "launch_array_access_mode", wp.config.LaunchArrayAccessMode.CHECKED):
            wp.launch(read_regular, 2, inputs=[pinned], outputs=[out], device=device)
            np.testing.assert_allclose(out.numpy(), [2.0, 4.0])
        with mock.patch.object(wp.config, "launch_array_access_mode", wp.config.LaunchArrayAccessMode.STRICT):
            with test.assertRaisesRegex(RuntimeError, "STRICT.*argument 'a'"):
                wp.launch(read_regular, 2, inputs=[pinned], outputs=[out], device=device)


def test_apic_rejects_redirected_descriptors(test, device):
    class RedirectedArray(TaggedArray):
        redirected_field = "data"

        def __ctype__(self):
            descriptor = super().__ctype__()
            redirected = type(descriptor).from_buffer_copy(descriptor)
            setattr(redirected, self.redirected_field, self.redirect.ptr)
            return redirected

    class RedirectedGradArray(RedirectedArray):
        redirected_field = "grad"

    out = wp.empty(2, dtype=float, device=device)
    for cls, expected in ((RedirectedArray, [6.0, 8.0]), (RedirectedGradArray, [2.0, 4.0])):
        with test.subTest(cls=cls.__name__):
            source = cls([1.0, 2.0], tag="own storage", dtype=float, device=device, requires_grad=True)
            source.redirect = wp.array([3.0, 4.0], dtype=float, device=device)
            wp.launch(read_regular, 2, inputs=[source], outputs=[out], device=device)
            np.testing.assert_allclose(out.numpy(), expected)
            with test.assertRaisesRegex(RuntimeError, "APIC capture.*own storage"):
                with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
                    wp.launch(read_regular, 2, inputs=[source], outputs=[out], device=device)


def test_graph_capture_replay(test, device):
    families = make_families(device)
    storage = families[0][0]
    out = TaggedArray(tag="captured out", shape=4, dtype=float, device=device)
    for family, (source, expected, typed) in zip(FAMILIES, families, strict=True):
        for reader in (typed, GENERIC_KERNEL_BY_FAMILY[family]):
            with test.subTest(family=family.__name__, kernel=reader.key):
                storage.assign(np.arange(1, 5, dtype=np.float32))
                # Specialize and load each module before beginning CUDA capture.
                wp.launch(reader, 4, inputs=[source], outputs=[out], device=device)
                with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                    wp.launch(reader, 4, inputs=[source], outputs=[out], device=device)
                storage.assign(np.arange(1, 5, dtype=np.float32) + 10.0)
                out.zero_()
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(out.numpy(), (expected + 10.0) * 2.0)


def test_interoperability_aliases(test, device):
    a = TaggedArray([1.0, 2.0, 3.0], tag="interop", dtype=float, device=device)
    restored = wp.from_dlpack(wp.to_dlpack(a))
    test.assertEqual(restored.ptr, a.ptr)
    test.assertIsInstance(restored, wp.array)
    if device.is_cpu:
        view = np.asarray(a)
        test.assertEqual(view.ctypes.data, a.ptr)


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

add_function_test(TestArraySubclass, "test_launch_kernels", test_launch_concrete_and_generic_kernels, devices=devices)
add_function_test(TestArraySubclass, "test_launch_outputs", test_launch_with_subclass_outputs, devices=devices)
add_function_test(
    TestArraySubclass, "test_shared_specialization", test_specialization_is_shared_with_parent, devices=devices
)
add_function_test(TestArraySubclass, "test_shared_explicit_overload", test_explicit_overload_is_shared, devices=devices)
add_function_test(TestArraySubclass, "test_validation", test_validation_errors, devices=devices)
add_function_test(TestArraySubclass, "test_external_interfaces", test_external_array_interface_objects, devices=devices)
add_function_test(TestArraySubclass, "test_null_arrays", test_null_arrays_and_descriptors, devices=devices)
add_function_test(TestArraySubclass, "test_gradients_tape", test_gradients_and_tape, devices=devices)
add_function_test(TestArraySubclass, "test_overwrite_gradient_parity", test_overwrite_gradient_parity, devices=devices)
add_function_test(
    TestArraySubclass, "test_gradient_subclasses", test_gradient_arrays_may_be_subclasses, devices=devices
)
add_function_test(TestArraySubclass, "test_gradcheck", test_gradcheck_and_jacobian, devices=devices)
add_function_test(TestArraySubclass, "test_operators", test_array_operators, devices=devices)
add_function_test(TestArraySubclass, "test_fem_cache", test_fem_interpolate_cache, devices=devices)
add_function_test(TestArraySubclass, "test_map", test_map_inputs_and_outputs, devices=devices)
add_function_test(TestArraySubclass, "test_map_cache", test_map_kernels_are_cached_by_native_family, devices=devices)
add_function_test(TestArraySubclass, "test_map_literal_names", test_map_literal_names, devices=devices)
add_function_test(TestArraySubclass, "test_function_overloads", test_python_function_overloads, devices=devices)
add_function_test(TestArraySubclass, "test_sequence_overloads", test_sequence_array_overloads, devices=devices)
add_function_test(TestArraySubclass, "test_structs", test_struct_fields, devices=devices)
add_function_test(
    TestArraySubclass, "test_struct_descriptor_errors", test_struct_descriptor_errors_preserve_field, devices=devices
)
add_function_test(TestArraySubclass, "test_descriptor_validation", test_subclass_descriptor_validation, devices=devices)
add_function_test(
    TestArraySubclass, "test_wrong_family_interface", test_wrong_family_does_not_use_external_interface, devices=devices
)
add_function_test(TestArraySubclass, "test_defaults", test_function_defaults, devices=devices)
add_function_test(TestArraySubclass, "test_views", test_array_views_in_kernels, devices=devices)
add_function_test(TestArraySubclass, "test_dtypes", test_vector_dtypes_and_dimensions, devices=devices)
add_function_test(TestArraySubclass, "test_struct_bfloat16", test_struct_and_bfloat16_dtypes, devices=devices)
add_function_test(TestArraySubclass, "test_host_operations", test_host_operations_keep_identity, devices=devices)
add_function_test(TestArraySubclass, "test_layouts", test_strided_empty_and_high_rank_arrays, devices=devices)
add_function_test(TestArraySubclass, "test_launch_command", test_launch_command_parameter_replacement, devices=devices)
add_function_test(TestArraySubclass, "test_retention", test_library_caches_do_not_retain_arrays, devices=devices)
add_function_test(TestArraySubclass, "test_sgd_overloads", test_sgd_live_overload_descriptors, devices=devices)
add_function_test(
    TestArraySubclass, "test_class_reassignment", test_class_reassignment_uses_real_family, devices=devices
)
add_function_test(TestArraySubclass, "test_subprocess_cache", test_subprocess_specialization_cache, devices=devices)
add_function_test(TestArraySubclass, "test_constructors", test_no_constructor_calls, devices=devices)
add_function_test(TestArraySubclass, "test_exact_arrays", test_exact_arrays_are_unchanged, devices=devices)
add_function_test(TestArraySubclass, "test_cross_device", test_cross_device_access_modes, devices=cuda_devices)
add_function_test(TestArraySubclass, "test_apic_descriptors", test_apic_rejects_redirected_descriptors, devices=devices)
add_function_test(TestArraySubclass, "test_capture", test_graph_capture_replay, devices=cuda_devices)
add_function_test(TestArraySubclass, "test_interop", test_interoperability_aliases, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
