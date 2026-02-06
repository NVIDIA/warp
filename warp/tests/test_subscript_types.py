# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for subscript-style type annotations (wp.array[float], wp.types.Vector[float, Literal[3]], etc.)."""

import unittest
from typing import Any, Literal, TypeVar, get_origin

import numpy as np

import warp as wp
from warp._src.types import (
    ARRAY_TYPE_FABRIC,
    ARRAY_TYPE_FABRIC_INDEXED,
    ARRAY_TYPE_INDEXED,
    ARRAY_TYPE_REGULAR,
    _ArrayAnnotation,
    _ArrayAnnotationBase,
    _IndexedArrayAnnotation,
    array_type_id,
    concrete_array_type,
    get_type_code,
    is_array,
    is_tile,
    matches_array_class,
    type_is_matrix,
    type_is_vector,
)
from warp.tests.unittest_utils import add_function_test, get_test_devices


def test_subscript_kernel_actually_runs(test, device):
    """Test that kernels with subscript annotations compile and run correctly."""

    @wp.kernel
    def add_kernel(a: wp.array[float], b: wp.array[float], c: wp.array[float]):
        i = wp.tid()
        c[i] = a[i] + b[i]

    n = 10
    a = wp.full(n, 1.0, dtype=float, device=device)
    b = wp.full(n, 2.0, dtype=float, device=device)
    c = wp.zeros(n, dtype=float, device=device)

    wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)
    test.assertEqual(c.numpy()[0], 3.0)


def test_subscript_kernel_with_vectors(test, device):
    """Test kernels with Vector subscript annotations."""

    @wp.kernel
    def vector_kernel(
        positions: wp.array[wp.types.Vector[wp.float32, Literal[3]]],
        output: wp.array[float],
    ):
        i = wp.tid()
        output[i] = wp.length(positions[i])

    n = 5
    positions = wp.zeros(n, dtype=wp.vec3f, device=device)
    output = wp.zeros(n, dtype=float, device=device)

    wp.launch(vector_kernel, dim=n, inputs=[positions, output], device=device)


# Constants for tile tests - defined at module level for kernel compilation
TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_THREADS = 64


@wp.func
def tile_passthrough(
    t: wp.tile[float, Literal[8], Literal[4]],
) -> wp.tile[float, Literal[8], Literal[4]]:
    return t


@wp.kernel
def tile_func_kernel(
    input: wp.array2d[float],
    output: wp.array2d[float],
):
    i = wp.tid()
    t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    t = tile_passthrough(t)
    wp.tile_store(output, t, offset=(i * TILE_M, 0))


def test_tile_subscript_in_func(test, device):
    """Test that tile subscript annotations work in @wp.func called from kernel."""
    n_tiles = 4
    n_rows = n_tiles * TILE_M
    n_cols = TILE_N
    input_data = wp.ones((n_rows, n_cols), dtype=float, device=device)
    output_data = wp.zeros((n_rows, n_cols), dtype=float, device=device)

    wp.launch_tiled(
        tile_func_kernel,
        dim=[n_tiles],
        inputs=[input_data, output_data],
        block_dim=TILE_THREADS,
        device=device,
    )

    result = output_data.numpy()
    expected = np.ones((n_rows, n_cols), dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


@wp.func
def tile_identity(
    t: wp.tile[float],
) -> wp.tile[float]:
    return t


@wp.kernel
def tile_dtype_only_kernel(
    input: wp.array2d[float],
    output: wp.array2d[float],
):
    i = wp.tid()
    t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    t = tile_identity(t)
    wp.tile_store(output, t, offset=(i * TILE_M, 0))


def test_tile_subscript_dtype_only_in_func(test, device):
    """Test that dtype-only tile subscript annotations (shape=Any) work in @wp.func."""
    n_tiles = 4
    n_rows = n_tiles * TILE_M
    n_cols = TILE_N
    input_data = wp.full((n_rows, n_cols), 2.0, dtype=float, device=device)
    output_data = wp.zeros((n_rows, n_cols), dtype=float, device=device)

    wp.launch_tiled(
        tile_dtype_only_kernel,
        dim=[n_tiles],
        inputs=[input_data, output_data],
        block_dim=TILE_THREADS,
        device=device,
    )

    expected = np.full((n_rows, n_cols), 2.0, dtype=np.float32)
    np.testing.assert_array_almost_equal(output_data.numpy(), expected)


@wp.kernel
def generic_scale_subscript(x: wp.array[Any], s: Any):
    i = wp.tid()
    x[i] = s * x[i]


wp.overload(generic_scale_subscript, [wp.array(dtype=wp.float32), wp.float32])
wp.overload(generic_scale_subscript, [wp.array(dtype=wp.float64), wp.float64])


def test_subscript_generic_kernel(test, device):
    """Test that subscript-style annotations work for generic kernels (like () style)."""
    n = 5
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # float32
    x32 = wp.array(data, dtype=wp.float32, device=device)
    wp.launch(generic_scale_subscript, dim=n, inputs=[x32, wp.float32(3)], device=device)
    np.testing.assert_allclose(x32.numpy(), [3.0, 6.0, 9.0, 12.0, 15.0])

    # float64
    x64 = wp.array(data, dtype=wp.float64, device=device)
    wp.launch(generic_scale_subscript, dim=n, inputs=[x64, wp.float64(2)], device=device)
    np.testing.assert_allclose(x64.numpy(), [2.0, 4.0, 6.0, 8.0, 10.0])


def test_subscript_indexedarray_kernel(test, device):
    """Test that kernels with indexedarray subscript annotations compile and run correctly."""

    @wp.kernel
    def indexed_add(a: wp.indexedarray[float], b: wp.array[float]):
        i = wp.tid()
        b[i] = a[i] + 1.0

    n = 5
    values = wp.array(np.arange(10, dtype=np.float32), device=device)
    indices = wp.array([0, 2, 4, 6, 8], dtype=int, device=device)
    iarr = wp.indexedarray1d(values, [indices])
    out = wp.zeros(n, dtype=float, device=device)

    wp.launch(indexed_add, dim=n, inputs=[iarr, out], device=device)
    np.testing.assert_allclose(out.numpy(), [1.0, 3.0, 5.0, 7.0, 9.0])


def test_subscript_dtype_mismatch(test, device):
    """Test that launching a kernel with mismatched dtypes raises RuntimeError."""

    @wp.kernel
    def float_kernel(a: wp.array[float], b: wp.array[float]):
        i = wp.tid()
        b[i] = a[i]

    n = 4
    a_float = wp.zeros(n, dtype=float, device=device)
    b_float = wp.zeros(n, dtype=float, device=device)
    a_int = wp.zeros(n, dtype=int, device=device)

    wp.launch(float_kernel, dim=n, inputs=[a_float, b_float], device=device)

    with test.assertRaises(RuntimeError):
        wp.launch(float_kernel, dim=n, inputs=[a_int, b_float], device=device)


def test_subscript_ndim_mismatch(test, device):
    """Test that launching a kernel with mismatched ndim raises RuntimeError."""

    @wp.kernel
    def kernel_2d(a: wp.array2d[float]):
        pass

    a_1d = wp.zeros(4, dtype=float, device=device)
    a_2d = wp.zeros((2, 2), dtype=float, device=device)

    wp.launch(kernel_2d, dim=1, inputs=[a_2d], device=device)

    with test.assertRaises(RuntimeError):
        wp.launch(kernel_2d, dim=1, inputs=[a_1d], device=device)


class TestSubscriptTypes(unittest.TestCase):
    """Tests for subscript-style type hint syntax."""

    def test_array_subscript_syntax(self):
        """Test array subscript syntax with dtype and optional ndim."""
        ann = wp.array[float]
        self.assertEqual(ann.dtype, wp.float32)
        self.assertEqual(ann.ndim, 1)

        ann2 = wp.array[wp.float64, Literal[2]]
        self.assertEqual(ann2.dtype, wp.float64)
        self.assertEqual(ann2.ndim, 2)

        # Bare integer (runtime only, not type-checker compatible)
        ann3 = wp.array[wp.float32, 3]
        self.assertEqual(ann3.ndim, 3)

    def test_arrayNd_subscript_syntax(self):
        """Test array2d/3d/4d subscript syntax."""
        self.assertEqual(wp.array2d[float].ndim, 2)
        self.assertEqual(wp.array3d[float].ndim, 3)
        self.assertEqual(wp.array4d[float].ndim, 4)

    def test_indexedarray_subscript_syntax(self):
        """Test indexedarray subscript syntax."""
        ann = wp.indexedarray[float]
        self.assertEqual(ann.dtype, wp.float32)
        self.assertEqual(ann.ndim, 1)

        ann2 = wp.indexedarray[wp.float64, Literal[2]]
        self.assertEqual(ann2.dtype, wp.float64)
        self.assertEqual(ann2.ndim, 2)

    def test_fabricarray_subscript_syntax(self):
        """Test fabricarray subscript syntax."""
        ann = wp.fabricarray[float]
        self.assertEqual(ann.dtype, wp.float32)
        self.assertEqual(ann.ndim, 1)

        ann2 = wp.fabricarray[wp.float64, Literal[2]]
        self.assertEqual(ann2.dtype, wp.float64)
        self.assertEqual(ann2.ndim, 2)

        # Bare integer (runtime only)
        ann3 = wp.fabricarray[wp.float32, 2]
        self.assertEqual(ann3.ndim, 2)

    def test_indexedfabricarray_subscript_syntax(self):
        """Test indexedfabricarray subscript syntax."""
        ann = wp.indexedfabricarray[float]
        self.assertEqual(ann.dtype, wp.float32)
        self.assertEqual(ann.ndim, 1)

        ann2 = wp.indexedfabricarray[wp.float64, Literal[2]]
        self.assertEqual(ann2.dtype, wp.float64)
        self.assertEqual(ann2.ndim, 2)

        # Bare integer (runtime only)
        ann3 = wp.indexedfabricarray[wp.float32, 2]
        self.assertEqual(ann3.ndim, 2)

    def test_vector_subscript_syntax(self):
        """Test Vector subscript syntax."""
        vec_type = wp.types.Vector[float, Literal[3]]
        self.assertEqual(vec_type._length_, 3)
        self.assertIs(vec_type, wp.types.vector(3, wp.float32))

        # Bare integer (runtime only)
        vec_type2 = wp.types.Vector[wp.float64, 4]
        self.assertEqual(vec_type2._length_, 4)

    def test_matrix_subscript_syntax(self):
        """Test Matrix subscript syntax."""
        mat_type = wp.types.Matrix[float, Literal[3], Literal[3]]
        self.assertEqual(mat_type._shape_, (3, 3))
        self.assertIs(mat_type, wp.types.matrix((3, 3), wp.float32))

        # Non-square matrix
        mat_type2 = wp.types.Matrix[wp.float64, Literal[2], Literal[4]]
        self.assertEqual(mat_type2._shape_, (2, 4))

    def test_quaternion_subscript_syntax(self):
        """Test Quaternion subscript syntax."""
        quat_type = wp.types.Quaternion[wp.float32]
        self.assertIs(quat_type, wp.types.quaternion(wp.float32))

        quat_type2 = wp.types.Quaternion[wp.float64]
        self.assertIs(quat_type2, wp.types.quaternion(wp.float64))

        # Verify caching
        self.assertIs(wp.types.Quaternion[wp.float32], wp.types.Quaternion[wp.float32])

    def test_transformation_subscript_syntax(self):
        """Test Transformation subscript syntax."""
        xform_type = wp.types.Transformation[wp.float32]
        self.assertIs(xform_type, wp.types.transformation(wp.float32))

        xform_type2 = wp.types.Transformation[wp.float64]
        self.assertIs(xform_type2, wp.types.transformation(wp.float64))

        # Verify caching
        self.assertIs(wp.types.Transformation[wp.float32], wp.types.Transformation[wp.float32])

    def test_tile_subscript_syntax(self):
        """Test tile subscript annotation syntax."""
        # wp.tile[dtype] -- dtype only, shape unspecified
        t0 = wp.tile[float]
        self.assertTrue(is_tile(t0))
        self.assertEqual(t0.dtype, wp.float32)
        self.assertEqual(t0.shape, Any)
        self.assertEqual(t0.storage, "register")

        # wp.tile[dtype, M, N] -- with Literal shape dimensions
        t1 = wp.tile[float, Literal[8], Literal[4]]
        self.assertTrue(is_tile(t1))
        self.assertEqual(t1.shape, (8, 4))
        self.assertEqual(t1.storage, "register")

        # wp.tile[dtype, M, N] -- with bare integers
        t2 = wp.tile[wp.float64, 16, 8]
        self.assertEqual(t2.shape, (16, 8))

        # wp.tile[dtype, (M, N)] -- tuple syntax
        t3 = wp.tile[float, (4, 4)]
        self.assertEqual(t3.shape, (4, 4))

    def test_subscript_identity(self):
        """Test that subscript and factory functions return identical cached types."""
        # Literal and bare int return same cached type
        self.assertIs(wp.types.Vector[wp.float32, Literal[3]], wp.types.Vector[wp.float32, 3])
        self.assertIs(wp.types.Vector[wp.float32, Literal[3]], wp.types.vector(3, wp.float32))

        # Matrix identity
        self.assertIs(
            wp.types.Matrix[wp.float32, Literal[3], Literal[3]],
            wp.types.matrix((3, 3), wp.float32),
        )

    def test_subscript_error_cases(self):
        """Test error handling for invalid subscript parameters."""
        # Vector errors
        with self.assertRaisesRegex(TypeError, r"requires 2 parameters"):
            wp.types.Vector[float]

        with self.assertRaisesRegex(TypeError, r"positive integer"):
            wp.types.Vector[float, -1]

        with self.assertRaisesRegex(TypeError, r"positive integer"):
            wp.types.Vector[float, 0]

        # Matrix errors
        with self.assertRaisesRegex(TypeError, r"requires 3 parameters"):
            wp.types.Matrix[float, 3]

        with self.assertRaisesRegex(TypeError, r"positive integer"):
            wp.types.Matrix[float, -1, 3]

        # Array errors
        with self.assertRaisesRegex(ValueError, r"ndim must be between"):
            wp.array[float, 5]  # ARRAY_MAX_DIMS is 4

        # arrayNd errors — only accept a single dtype, not tuple params
        with self.assertRaisesRegex(TypeError, r"single type parameter"):
            wp.array2d[float, Literal[2]]

        # Tile errors
        with self.assertRaisesRegex(TypeError, r"requires at least a dtype"):
            wp.tile[()]

        with self.assertRaisesRegex(TypeError, r"positive integer"):
            wp.tile[float, -1, 4]

    def test_codegen_compatibility(self):
        """Test that subscript results have required codegen attributes."""
        # Array annotation has required attributes
        arr_ann = wp.array[float]
        self.assertTrue(is_array(arr_ann))
        self.assertEqual(arr_ann.dtype, wp.float32)
        self.assertEqual(arr_ann.ndim, 1)
        self.assertTrue(hasattr(arr_ann, "vars"))
        self.assertTrue(callable(getattr(arr_ann, "__ctype__", None)))

        # Fabric array annotations are recognized by is_array()
        self.assertTrue(is_array(wp.fabricarray[float]))
        self.assertTrue(is_array(wp.indexedfabricarray[float]))

        # Vector type has required attributes
        vec_type = wp.types.Vector[float, Literal[3]]
        self.assertTrue(type_is_vector(vec_type))
        self.assertTrue(hasattr(vec_type, "_wp_generic_type_hint_"))
        self.assertTrue(hasattr(vec_type, "_wp_type_params_"))
        self.assertTrue(hasattr(vec_type, "_wp_scalar_type_"))

        # Matrix type has required attributes
        mat_type = wp.types.Matrix[float, Literal[3], Literal[3]]
        self.assertTrue(type_is_matrix(mat_type))
        self.assertTrue(hasattr(mat_type, "_wp_generic_type_hint_"))

    def test_annotation_is_lightweight(self):
        """Subscript annotations are lightweight objects, not full array instances."""
        arr_ann = wp.array[float]
        self.assertIsInstance(arr_ann, _ArrayAnnotation)
        self.assertIsInstance(arr_ann, _ArrayAnnotationBase)
        self.assertNotIsInstance(arr_ann, wp.array)

        ia_ann = wp.indexedarray[float]
        self.assertIsInstance(ia_ann, _IndexedArrayAnnotation)
        self.assertIsInstance(ia_ann, _ArrayAnnotationBase)
        self.assertNotIsInstance(ia_ann, wp.indexedarray)

        # arrayNd subscripts also return lightweight annotations
        self.assertIsInstance(wp.array2d[float], _ArrayAnnotation)
        self.assertIsInstance(wp.array3d[float], _ArrayAnnotation)
        self.assertIsInstance(wp.array4d[float], _ArrayAnnotation)

        # fabricarray and indexedfabricarray subscripts return lightweight annotations
        self.assertNotIsInstance(wp.fabricarray[float], wp.fabricarray)
        self.assertNotIsInstance(wp.indexedfabricarray[float], wp.indexedfabricarray)

        # Annotations use __slots__ — no instance __dict__
        self.assertFalse(hasattr(arr_ann, "__dict__"))
        self.assertFalse(hasattr(ia_ann, "__dict__"))

    def test_annotation_vars_caching(self):
        """Verify that annotation .vars property is cached per instance."""
        arr_ann = wp.array[float]
        vars1 = arr_ann.vars
        vars2 = arr_ann.vars
        self.assertIs(vars1, vars2)

    def test_annotation_helpers(self):
        """Test matches_array_class and concrete_array_type helpers."""
        arr_ann = wp.array[float]
        ia_ann = wp.indexedarray[float]
        fa_ann = wp.fabricarray[float]
        ifa_ann = wp.indexedfabricarray[float]

        self.assertTrue(matches_array_class(arr_ann, wp.array))
        self.assertFalse(matches_array_class(arr_ann, wp.indexedarray))
        self.assertTrue(matches_array_class(ia_ann, wp.indexedarray))
        self.assertFalse(matches_array_class(ia_ann, wp.array))
        self.assertTrue(matches_array_class(fa_ann, wp.fabricarray))
        self.assertFalse(matches_array_class(fa_ann, wp.array))
        self.assertTrue(matches_array_class(ifa_ann, wp.indexedfabricarray))
        self.assertFalse(matches_array_class(ifa_ann, wp.fabricarray))

        self.assertIs(concrete_array_type(arr_ann), wp.array)
        self.assertIs(concrete_array_type(ia_ann), wp.indexedarray)
        self.assertIs(concrete_array_type(fa_ann), wp.fabricarray)
        self.assertIs(concrete_array_type(ifa_ann), wp.indexedfabricarray)

    def test_annotation_equality_and_hashing(self):
        """Annotations with same parameters are equal and hashable."""
        a1 = wp.array[float]
        a2 = wp.array[wp.float32]
        self.assertEqual(a1, a2)
        self.assertEqual(hash(a1), hash(a2))

        # Different ndim → not equal
        a3 = wp.array[float, Literal[2]]
        self.assertNotEqual(a1, a3)

        # Different concrete class → not equal
        ia = wp.indexedarray[float]
        self.assertNotEqual(a1, ia)

    def test_typevar_delegation(self):
        """Verify that TypeVar parameters delegate to Generic (preserves static typing)."""
        T = TypeVar("T")  # dtype
        N = TypeVar("N", bound=int)  # length

        # Should return a _GenericAlias, not a vec_t class
        # Note: dtype-first order (T=Scalar, N=Length)
        result = wp.types.Vector[T, N]
        self.assertTrue(hasattr(result, "__origin__"))
        self.assertEqual(get_origin(result), wp.types.Vector)

    def test_array_type_id_with_annotations(self):
        """array_type_id() returns correct constants for annotation objects."""
        self.assertEqual(array_type_id(wp.array[float]), ARRAY_TYPE_REGULAR)
        self.assertEqual(array_type_id(wp.array[wp.float64, Literal[2]]), ARRAY_TYPE_REGULAR)
        self.assertEqual(array_type_id(wp.indexedarray[float]), ARRAY_TYPE_INDEXED)
        self.assertEqual(array_type_id(wp.indexedarray[wp.float64, Literal[3]]), ARRAY_TYPE_INDEXED)
        self.assertEqual(array_type_id(wp.fabricarray[float]), ARRAY_TYPE_FABRIC)
        self.assertEqual(array_type_id(wp.fabricarray[wp.float64, Literal[2]]), ARRAY_TYPE_FABRIC)
        self.assertEqual(array_type_id(wp.indexedfabricarray[float]), ARRAY_TYPE_FABRIC_INDEXED)
        self.assertEqual(array_type_id(wp.indexedfabricarray[wp.float64, Literal[2]]), ARRAY_TYPE_FABRIC_INDEXED)

    def test_get_type_code_with_annotations(self):
        """get_type_code() returns identical codes for annotations and concrete instances."""
        # array annotation should match concrete array instance
        arr_ann = wp.array[wp.float32, Literal[2]]
        arr_inst = wp.array(dtype=wp.float32, ndim=2)
        self.assertEqual(get_type_code(arr_ann), get_type_code(arr_inst))

        # indexedarray annotation should match concrete indexedarray instance
        ia_ann = wp.indexedarray[wp.float64]
        ia_inst = wp.indexedarray(dtype=wp.float64)
        self.assertEqual(get_type_code(ia_ann), get_type_code(ia_inst))

        # fabricarray annotation should match concrete fabricarray instance
        fa_ann = wp.fabricarray[wp.float32]
        fa_inst = wp.fabricarray(dtype=wp.float32)
        self.assertEqual(get_type_code(fa_ann), get_type_code(fa_inst))

        # indexedfabricarray annotation should match concrete indexedfabricarray instance
        ifa_ann = wp.indexedfabricarray[wp.float64]
        ifa_inst = wp.indexedfabricarray(dtype=wp.float64)
        self.assertEqual(get_type_code(ifa_ann), get_type_code(ifa_inst))

        # tile annotation with dtype only (shape=Any) should produce a valid code
        tile_ann = wp.tile[float]
        code = get_type_code(tile_ann)
        self.assertIn("?", code)  # shape is unknown

        # tile annotation with concrete shape should produce a valid code
        tile_concrete = wp.tile[float, 8, 4]
        code2 = get_type_code(tile_concrete)
        self.assertNotEqual(code, code2)

    def test_unrecognized_dtype_passthrough(self):
        """Verify unrecognized dtypes are passed through without immediate error."""

        class CustomType:
            pass

        # Array annotations should not raise - dtype validation is deferred to codegen
        ann = wp.array[CustomType]
        self.assertEqual(ann.dtype, CustomType)

        # Note: Vector/Matrix factory functions have stricter type requirements
        # (they try to access dtype._type_ for ctypes.Array creation), so they
        # fail immediately for truly unknown types. This is expected behavior.

    def test_any_ndim(self):
        """wp.array[dtype, Any] means any dimensionality."""
        ann = wp.array[float, Any]
        self.assertEqual(ann.dtype, wp.float32)
        self.assertIs(ann.ndim, Any)

        # Same for all array variants
        ia = wp.indexedarray[float, Any]
        self.assertIs(ia.ndim, Any)

        fa = wp.fabricarray[wp.float64, Any]
        self.assertIs(fa.ndim, Any)
        self.assertEqual(fa.dtype, wp.float64)

        ifa = wp.indexedfabricarray[wp.float32, Any]
        self.assertIs(ifa.ndim, Any)

    def test_any_dtype(self):
        """wp.array[Any] means any dtype, 1D."""
        ann = wp.array[Any]
        self.assertIs(ann.dtype, Any)
        self.assertEqual(ann.ndim, 1)

        # Any dtype with explicit ndim
        ann2 = wp.array[Any, Literal[3]]
        self.assertIs(ann2.dtype, Any)
        self.assertEqual(ann2.ndim, 3)

        # Same for other variants
        self.assertIs(wp.indexedarray[Any].dtype, Any)
        self.assertIs(wp.fabricarray[Any].dtype, Any)
        self.assertIs(wp.indexedfabricarray[Any].dtype, Any)


devices = get_test_devices()

add_function_test(
    TestSubscriptTypes, "test_subscript_kernel_actually_runs", test_subscript_kernel_actually_runs, devices=devices
)
add_function_test(
    TestSubscriptTypes, "test_subscript_kernel_with_vectors", test_subscript_kernel_with_vectors, devices=devices
)
add_function_test(TestSubscriptTypes, "test_tile_subscript_in_func", test_tile_subscript_in_func, devices=devices)
add_function_test(
    TestSubscriptTypes,
    "test_tile_subscript_dtype_only_in_func",
    test_tile_subscript_dtype_only_in_func,
    devices=devices,
)
add_function_test(TestSubscriptTypes, "test_subscript_generic_kernel", test_subscript_generic_kernel, devices=devices)
add_function_test(TestSubscriptTypes, "test_subscript_dtype_mismatch", test_subscript_dtype_mismatch, devices=devices)
add_function_test(TestSubscriptTypes, "test_subscript_ndim_mismatch", test_subscript_ndim_mismatch, devices=devices)
add_function_test(
    TestSubscriptTypes,
    "test_subscript_indexedarray_kernel",
    test_subscript_indexedarray_kernel,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
