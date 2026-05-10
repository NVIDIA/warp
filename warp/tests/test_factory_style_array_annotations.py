# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility tests for factory-style Warp array annotations."""

import unittest
from typing import Any, Literal

import numpy as np

import warp as wp
from warp._src.types import (
    ARRAY_TYPE_FABRIC,
    ARRAY_TYPE_FABRIC_INDEXED,
    ARRAY_TYPE_INDEXED,
    ARRAY_TYPE_REGULAR,
    array_type_id,
    get_type_code,
)
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.struct
class FactoryStyleArrayStruct:
    values: wp.array(dtype=float)


@wp.func
def factory_style_read_func(values: wp.array(dtype=Any), index: int):
    return values[index]


@wp.kernel
def factory_style_ndim_kernel(
    arr: wp.array(dtype=float, ndim=2),
    arr1d: wp.array1d(dtype=float),
    arr2d: wp.array2d(dtype=float),
    arr3d: wp.array3d(dtype=float),
    arr4d: wp.array4d(dtype=float),
    out: wp.array(dtype=float),
):
    out[0] = arr[1, 1] + arr1d[1] + arr2d[1, 1] + arr3d[0, 1, 1] + arr4d[0, 0, 1, 1]


@wp.kernel
def factory_style_composite_kernel(
    vecs: wp.array(dtype=wp.vec3),
    mats: wp.array(dtype=wp.mat22),
    out: wp.array(dtype=float),
):
    v = vecs[0]
    m = mats[0]
    out[0] = v[0] + v[1] + v[2] + m[0, 0] + m[1, 1]


@wp.kernel
def factory_style_matrix_array2d_kernel(mats: wp.array2d(dtype=wp.mat22), out: wp.array2d(dtype=float)):
    m = mats[0, 0]
    out[0, 0] = m[0, 0] + m[1, 1]


@wp.kernel
def factory_style_generic_scale_kernel(values: wp.array(dtype=Any), scale: Any):
    i = wp.tid()
    values[i] = values[i] * scale


@wp.kernel
def factory_style_vec3_component_sum_kernel(values: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    v = values[i]
    out[i] = v[0] + v[1] + v[2]


@wp.kernel
def factory_style_func_kernel(values: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = factory_style_read_func(values, i) * 2.0


@wp.kernel
def factory_style_struct_kernel(data: FactoryStyleArrayStruct, out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = data.values[i] + 1.0


@wp.kernel
def factory_style_indexedarray_kernel(values: wp.indexedarray(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = values[i] * 2.0


wp.overload(factory_style_generic_scale_kernel, [wp.array(dtype=wp.float32), wp.float32])
wp.overload(factory_style_generic_scale_kernel, [wp.array(dtype=wp.int32), wp.int32])
wp.overload(factory_style_vec3_component_sum_kernel, [wp.array(dtype=wp.vec3h), wp.array(dtype=wp.float16)])
wp.overload(factory_style_vec3_component_sum_kernel, [wp.array(dtype=wp.vec3f), wp.array(dtype=wp.float32)])
wp.overload(factory_style_vec3_component_sum_kernel, [wp.array(dtype=wp.vec3d), wp.array(dtype=wp.float64)])


def create_factory_style_string_annotation_kernel(dtype):
    def fn(
        out: "wp.array(dtype=dtype)",
    ):
        i = wp.tid()
        out[i] = dtype(4.56)

    return wp.Kernel(func=fn)


def test_factory_style_ndim_annotations(test, device):
    arr = wp.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), device=device)
    arr1d = wp.array(np.array([10.0, 20.0], dtype=np.float32), device=device)
    arr2d = wp.array(np.array([[30.0, 40.0], [50.0, 60.0]], dtype=np.float32), device=device)
    arr3d = wp.array(np.arange(8, dtype=np.float32).reshape(2, 2, 2), device=device)
    arr4d = wp.array(np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2), device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch(factory_style_ndim_kernel, dim=1, inputs=[arr, arr1d, arr2d, arr3d, arr4d, out], device=device)

    np.testing.assert_allclose(out.numpy(), [4.0 + 20.0 + 60.0 + 3.0 + 3.0])


def test_factory_style_composite_annotations(test, device):
    vecs = wp.array([wp.vec3(1.0, 2.0, 3.0)], dtype=wp.vec3, device=device)
    mats = wp.array([wp.mat22(4.0, 5.0, 6.0, 7.0)], dtype=wp.mat22, device=device)
    out = wp.zeros(1, dtype=float, device=device)

    wp.launch(factory_style_composite_kernel, dim=1, inputs=[vecs, mats, out], device=device)

    np.testing.assert_allclose(out.numpy(), [1.0 + 2.0 + 3.0 + 4.0 + 7.0])


def test_factory_style_matrix_array2d_annotations(test, device):
    mats = wp.array(np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32), dtype=wp.mat22, device=device)
    out = wp.zeros((1, 1), dtype=float, device=device)

    wp.launch(factory_style_matrix_array2d_kernel, dim=1, inputs=[mats, out], device=device)

    np.testing.assert_allclose(out.numpy(), [[5.0]])


def test_factory_style_generic_overload_annotations(test, device):
    float_values = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    int_values = wp.array([1, 2, 3], dtype=wp.int32, device=device)

    wp.launch(factory_style_generic_scale_kernel, dim=3, inputs=[float_values, wp.float32(2.0)], device=device)
    wp.launch(factory_style_generic_scale_kernel, dim=3, inputs=[int_values, wp.int32(3)], device=device)

    np.testing.assert_allclose(float_values.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(int_values.numpy(), [3, 6, 9])


def test_factory_style_vec3_overload_annotations(test, device):
    test_cases = (
        (wp.vec3h, wp.float16, np.float16),
        (wp.vec3f, wp.float32, np.float32),
        (wp.vec3d, wp.float64, np.float64),
    )

    for vec_dtype, scalar_dtype, np_dtype in test_cases:
        with test.subTest(vec_dtype=vec_dtype):
            values = wp.array(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np_dtype),
                dtype=vec_dtype,
                device=device,
            )
            out = wp.zeros(2, dtype=scalar_dtype, device=device)

            wp.launch(factory_style_vec3_component_sum_kernel, dim=2, inputs=[values, out], device=device)

            np.testing.assert_allclose(out.numpy(), [6.0, 15.0])


def test_factory_style_func_annotations(test, device):
    values = wp.array([1.0, 2.0, 3.0], dtype=float, device=device)
    out = wp.zeros(3, dtype=float, device=device)

    wp.launch(factory_style_func_kernel, dim=3, inputs=[values, out], device=device)

    np.testing.assert_allclose(out.numpy(), [2.0, 4.0, 6.0])


def test_factory_style_struct_annotations(test, device):
    data = FactoryStyleArrayStruct()
    data.values = wp.array([1.0, 2.0, 3.0], dtype=float, device=device)
    out = wp.zeros(3, dtype=float, device=device)

    wp.launch(factory_style_struct_kernel, dim=3, inputs=[data, out], device=device)

    np.testing.assert_allclose(out.numpy(), [2.0, 3.0, 4.0])


def test_factory_style_indexedarray_annotations(test, device):
    base = wp.array(np.arange(10, dtype=np.float32), device=device)
    indices = wp.array([1, 3, 5], dtype=int, device=device)
    values = wp.indexedarray1d(base, [indices])
    out = wp.zeros(3, dtype=float, device=device)

    wp.launch(factory_style_indexedarray_kernel, dim=3, inputs=[values, out], device=device)

    np.testing.assert_allclose(out.numpy(), [2.0, 6.0, 10.0])


def test_factory_style_local_dtype_unique_kernel(test, device):
    dtype = wp.float16

    @wp.kernel(module="unique")
    def local_dtype_kernel(values: wp.array2d(dtype=dtype), out: wp.array(dtype=dtype)):
        out[0] = values[0, 1] + dtype(1.0)

    values = wp.array(np.array([[1.0, 2.0]], dtype=np.float16), dtype=dtype, device=device)
    out = wp.zeros(1, dtype=dtype, device=device)

    wp.launch(local_dtype_kernel, dim=1, inputs=[values, out], device=device)

    np.testing.assert_allclose(out.numpy(), [3.0])


def test_factory_style_string_annotation_scope(test, device):
    kernel = create_factory_style_string_annotation_kernel(wp.float32)
    out = wp.empty(1, dtype=wp.float32, device=device)

    wp.launch(kernel, dim=out.shape, outputs=[out], device=device)

    test.assertAlmostEqual(float(out.numpy()[0]), 4.56, places=5)


class TestFactoryStyleArrayAnnotations(unittest.TestCase):
    """Tests for backward-compatible factory-style array annotations."""

    def test_factory_style_annotation_metadata(self):
        arr = wp.array(dtype=wp.float64, ndim=2)
        self.assertEqual(arr.dtype, wp.float64)
        self.assertEqual(arr.ndim, 2)

        self.assertEqual(wp.array1d(dtype=float).ndim, 1)
        self.assertEqual(wp.array2d(dtype=float).ndim, 2)
        self.assertEqual(wp.array3d(dtype=float).ndim, 3)
        self.assertEqual(wp.array4d(dtype=float).ndim, 4)

        generic = wp.array(dtype=Any)
        self.assertIs(generic.dtype, Any)
        self.assertEqual(generic.ndim, 1)

    def test_factory_style_any_metadata(self):
        generic_4d = wp.array(dtype=Any, ndim=4)
        self.assertIs(generic_4d.dtype, Any)
        self.assertEqual(generic_4d.ndim, 4)
        self.assertEqual(array_type_id(generic_4d), ARRAY_TYPE_REGULAR)
        self.assertEqual(get_type_code(generic_4d), get_type_code(wp.array4d[Any]))

        default_generic = wp.array()
        self.assertIs(default_generic.dtype, Any)
        self.assertEqual(default_generic.ndim, 1)

    def test_factory_style_texture_array_metadata(self):
        annotations = (
            (wp.array(dtype=wp.Texture1D), wp.array[wp.Texture1D]),
            (wp.array(dtype=wp.Texture2D), wp.array[wp.Texture2D]),
            (wp.array(dtype=wp.Texture3D), wp.array[wp.Texture3D]),
        )

        for factory_style, subscript_style in annotations:
            with self.subTest(factory_style=factory_style):
                self.assertEqual(factory_style.dtype, subscript_style.dtype)
                self.assertEqual(factory_style.ndim, subscript_style.ndim)
                self.assertEqual(array_type_id(factory_style), ARRAY_TYPE_REGULAR)
                self.assertEqual(get_type_code(factory_style), get_type_code(subscript_style))

    def test_factory_style_geometry_array_metadata(self):
        annotations = (
            (wp.array(dtype=wp.vec3h), wp.array[wp.vec3h]),
            (wp.array(dtype=wp.vec3f), wp.array[wp.vec3f]),
            (wp.array(dtype=wp.vec3d), wp.array[wp.vec3d]),
            (wp.array2d(dtype=wp.int32), wp.array2d[wp.int32]),
            (wp.array(dtype=wp.int32, ndim=2), wp.array2d[wp.int32]),
            (wp.array2d(dtype=wp.mat22), wp.array2d[wp.mat22]),
            (wp.array(dtype=wp.mat22, ndim=2), wp.array2d[wp.mat22]),
        )

        for factory_style, subscript_style in annotations:
            with self.subTest(factory_style=factory_style):
                self.assertEqual(factory_style.dtype, subscript_style.dtype)
                self.assertEqual(factory_style.ndim, subscript_style.ndim)
                self.assertEqual(array_type_id(factory_style), ARRAY_TYPE_REGULAR)
                self.assertEqual(get_type_code(factory_style), get_type_code(subscript_style))

    def test_factory_style_noncontiguous_array_metadata(self):
        annotations = (
            (wp.indexedarray(dtype=wp.float64, ndim=3), wp.indexedarray[wp.float64, Literal[3]], ARRAY_TYPE_INDEXED),
            (wp.fabricarray(dtype=wp.float32, ndim=2), wp.fabricarray[wp.float32, 2], ARRAY_TYPE_FABRIC),
            (
                wp.indexedfabricarray(dtype=wp.float64, ndim=2),
                wp.indexedfabricarray[wp.float64, 2],
                ARRAY_TYPE_FABRIC_INDEXED,
            ),
        )

        for factory_style, subscript_style, expected_array_type in annotations:
            with self.subTest(factory_style=factory_style):
                self.assertEqual(factory_style.dtype, subscript_style.dtype)
                self.assertEqual(factory_style.ndim, subscript_style.ndim)
                self.assertEqual(array_type_id(factory_style), expected_array_type)
                self.assertEqual(get_type_code(factory_style), get_type_code(subscript_style))


devices = get_test_devices()

add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_ndim_annotations",
    test_factory_style_ndim_annotations,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_composite_annotations",
    test_factory_style_composite_annotations,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_matrix_array2d_annotations",
    test_factory_style_matrix_array2d_annotations,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_generic_overload_annotations",
    test_factory_style_generic_overload_annotations,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_vec3_overload_annotations",
    test_factory_style_vec3_overload_annotations,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_func_annotations",
    test_factory_style_func_annotations,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_struct_annotations",
    test_factory_style_struct_annotations,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_indexedarray_annotations",
    test_factory_style_indexedarray_annotations,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_local_dtype_unique_kernel",
    test_factory_style_local_dtype_unique_kernel,
    devices=devices,
)
add_function_test(
    TestFactoryStyleArrayAnnotations,
    "test_factory_style_string_annotation_scope",
    test_factory_style_string_annotation_scope,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
