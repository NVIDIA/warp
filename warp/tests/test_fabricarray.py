# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# types to test fabric arrays
_fabric_types = [
    *wp.types.scalar_types,
    *[wp.types.vector(2, T) for T in wp.types.scalar_types],
    *[wp.types.vector(3, T) for T in wp.types.scalar_types],
    *[wp.types.vector(4, T) for T in wp.types.scalar_types],
    *[wp.types.matrix((2, 2), T) for T in wp.types.scalar_types],
    *[wp.types.matrix((3, 3), T) for T in wp.types.scalar_types],
    *[wp.types.matrix((4, 4), T) for T in wp.types.scalar_types],
    *[wp.types.quaternion(T) for T in wp.types.float_types],
]


def _warp_type_to_fabric(dtype, is_array=False):
    scalar_map = {
        wp.bool: "b",
        wp.int8: "i1",
        wp.int16: "i2",
        wp.int32: "i4",
        wp.int64: "i8",
        wp.uint8: "u1",
        wp.uint16: "u2",
        wp.uint32: "u4",
        wp.uint64: "u8",
        wp.float16: "f2",
        wp.float32: "f4",
        wp.float64: "f8",
    }

    if hasattr(dtype, "_wp_scalar_type_"):
        type_str = scalar_map[dtype._wp_scalar_type_]
        if len(dtype._shape_) == 1:
            role = "vector"
        else:
            role = "matrix"
    else:
        type_str = scalar_map[dtype]
        role = ""

    if is_array:
        array_depth = 1
    else:
        array_depth = 0

    return (True, type_str, dtype._length_, array_depth, role)


# returns a fabric array interface constructed from a regular array
def _create_fabric_array_interface(data: wp.array, attrib: str, bucket_sizes: list[int] | None = None, copy=False):
    assert isinstance(data, wp.array)
    assert data.ndim == 1

    assert isinstance(attrib, str)

    if copy:
        data = wp.clone(data)

    if bucket_sizes is not None:
        assert hasattr(bucket_sizes, "__len__")

        # verify total size
        total_size = 0
        for bucket_size in bucket_sizes:
            total_size += bucket_size

        if total_size != data.size:
            raise RuntimeError("Bucket sizes don't add up to the size of data array")

    elif data.size > 0:
        rng = np.random.default_rng(123)

        # generate random bucket sizes
        bucket_min = 1
        bucket_max = math.ceil(0.5 * data.size)
        total_size = data.size
        size_remaining = total_size

        bucket_sizes = []
        while size_remaining >= bucket_max:
            bucket_size = rng.integers(bucket_min, high=bucket_max, dtype=int)
            bucket_sizes.append(bucket_size)
            size_remaining -= bucket_size

        if size_remaining > 0:
            bucket_sizes.append(size_remaining)

    else:
        # empty data array
        bucket_sizes = []

    dtype_size = wp.types.type_size_in_bytes(data.dtype)
    p = int(data.ptr) if data.ptr else 0
    pointers = []
    counts = []
    for bucket_size in bucket_sizes:
        pointers.append(p)
        counts.append(bucket_size)
        p += bucket_size * dtype_size

    attrib_info = {}

    attrib_info["type"] = _warp_type_to_fabric(data.dtype)
    attrib_info["access"] = 2  # ReadWrite
    attrib_info["pointers"] = pointers
    attrib_info["counts"] = counts

    iface = {}
    iface["version"] = 1
    iface["device"] = str(data.device)
    iface["attribs"] = {attrib: attrib_info}
    iface["_ref"] = data  # backref to keep the array alive

    return iface


# returns a fabric array array interface constructed from a list of regular arrays
def _create_fabric_array_array_interface(data: list, attrib: str, bucket_sizes: list[int] | None = None):
    # data should be a list of arrays
    assert isinstance(data, list)

    num_arrays = len(data)
    assert num_arrays > 0

    device = data[0].device
    dtype = data[0].dtype

    assert isinstance(attrib, str)

    if bucket_sizes is not None:
        assert hasattr(bucket_sizes, "__len__")

        # verify total size
        total_size = 0
        for bucket_size in bucket_sizes:
            total_size += bucket_size

        if total_size != num_arrays:
            raise RuntimeError("Bucket sizes don't add up to the number of given arrays")

    else:
        rng = np.random.default_rng(123)

        # generate random bucket sizes
        bucket_min = 1
        bucket_max = math.ceil(0.5 * num_arrays)
        total_size = num_arrays
        size_remaining = total_size

        bucket_sizes = []
        while size_remaining >= bucket_max:
            bucket_size = rng.integers(bucket_min, high=bucket_max, dtype=int)
            bucket_sizes.append(bucket_size)
            size_remaining -= bucket_size

        if size_remaining > 0:
            bucket_sizes.append(size_remaining)

    # initialize array of pointers to arrays and their lengths
    _array_pointers = []
    _array_lengths = []
    for i in range(num_arrays):
        _array_pointers.append(data[i].ptr)
        _array_lengths.append(data[i].size)

    array_pointers = wp.array(_array_pointers, dtype=wp.uint64, device=device)
    pointer_size = wp.types.type_size_in_bytes(array_pointers.dtype)

    lengths = wp.array(_array_lengths, dtype=wp.uint64, device=device)
    length_size = wp.types.type_size_in_bytes(lengths.dtype)

    p_pointers = int(array_pointers.ptr)
    p_lengths = int(lengths.ptr)
    pointers = []
    counts = []
    array_lengths = []
    for bucket_size in bucket_sizes:
        pointers.append(p_pointers)
        counts.append(bucket_size)
        array_lengths.append(p_lengths)
        p_pointers += bucket_size * pointer_size
        p_lengths += bucket_size * length_size

    attrib_info = {}

    attrib_info["type"] = _warp_type_to_fabric(dtype, is_array=True)
    attrib_info["access"] = 2  # ReadWrite
    attrib_info["pointers"] = pointers
    attrib_info["counts"] = counts
    attrib_info["array_lengths"] = array_lengths

    iface = {}
    iface["version"] = 1
    iface["device"] = str(device)
    iface["attribs"] = {attrib: attrib_info}
    iface["_ref"] = data  # backref to keep the data arrays alive
    iface["_ref_pointers"] = array_pointers  # backref to keep the array pointers alive
    iface["_ref_lengths"] = lengths  # backref to keep the lengths array alive

    return iface


@wp.kernel
def fa_kernel(a: wp.fabricarray(dtype=float), expected: wp.array(dtype=float)):
    i = wp.tid()

    wp.expect_eq(a[i], expected[i])

    a[i] = 2.0 * a[i]

    wp.atomic_add(a, i, 1.0)

    wp.expect_eq(a[i], 2.0 * expected[i] + 1.0)


@wp.kernel
def fa_kernel_indexed(a: wp.indexedfabricarray(dtype=float), expected: wp.indexedarray(dtype=float)):
    i = wp.tid()

    wp.expect_eq(a[i], expected[i])

    a[i] = 2.0 * a[i]

    wp.atomic_add(a, i, 1.0)

    wp.expect_eq(a[i], 2.0 * expected[i] + 1.0)


def test_fabricarray_kernel(test, device):
    data = wp.array(data=np.arange(100, dtype=np.float32), device=device)
    iface = _create_fabric_array_interface(data, "foo", copy=True)
    fa = wp.fabricarray(data=iface, attrib="foo")

    test.assertEqual(fa.dtype, data.dtype)
    test.assertEqual(fa.ndim, 1)
    test.assertEqual(fa.shape, data.shape)
    test.assertEqual(fa.size, data.size)

    wp.launch(fa_kernel, dim=fa.size, inputs=[fa, data], device=device)

    # reset data
    wp.copy(fa, data)

    # test indexed
    indices = wp.array(data=np.arange(1, data.size, 2, dtype=np.int32), device=device)
    ifa = fa[indices]
    idata = data[indices]

    test.assertEqual(ifa.dtype, idata.dtype)
    test.assertEqual(ifa.ndim, 1)
    test.assertEqual(ifa.shape, idata.shape)
    test.assertEqual(ifa.size, idata.size)

    wp.launch(fa_kernel_indexed, dim=ifa.size, inputs=[ifa, idata], device=device)

    wp.synchronize_device(device)


@wp.kernel
def fa_generic_dtype_kernel(a: wp.fabricarray(dtype=Any), b: wp.fabricarray(dtype=Any)):
    i = wp.tid()
    b[i] = a[i] + a[i]


@wp.kernel
def fa_generic_dtype_kernel_indexed(a: wp.indexedfabricarray(dtype=Any), b: wp.indexedfabricarray(dtype=Any)):
    i = wp.tid()
    b[i] = a[i] + a[i]


def test_fabricarray_generic_dtype(test, device):
    for T in _fabric_types:
        if hasattr(T, "_wp_scalar_type_"):
            nptype = wp.types.warp_type_to_np_dtype[T._wp_scalar_type_]
        else:
            nptype = wp.types.warp_type_to_np_dtype[T]

        data = wp.array(data=np.arange(10, dtype=nptype), device=device)
        data_iface = _create_fabric_array_interface(data, "foo", copy=True)
        fa = wp.fabricarray(data=data_iface, attrib="foo")

        result = wp.zeros_like(data)
        result_iface = _create_fabric_array_interface(result, "foo", copy=True)
        fb = wp.fabricarray(data=result_iface, attrib="foo")

        test.assertEqual(fa.dtype, fb.dtype)
        test.assertEqual(fa.ndim, fb.ndim)
        test.assertEqual(fa.shape, fb.shape)
        test.assertEqual(fa.size, fb.size)

        wp.launch(fa_generic_dtype_kernel, dim=fa.size, inputs=[fa, fb], device=device)

        assert_np_equal(fb.numpy(), 2 * fa.numpy())

        # reset data
        wp.copy(fa, data)
        wp.copy(fb, result)

        # test indexed
        indices = wp.array(data=np.arange(1, data.size, 2, dtype=np.int32), device=device)
        ifa = fa[indices]
        ifb = fb[indices]

        test.assertEqual(ifa.dtype, ifb.dtype)
        test.assertEqual(ifa.ndim, ifb.ndim)
        test.assertEqual(ifa.shape, ifb.shape)
        test.assertEqual(ifa.size, ifb.size)

        wp.launch(fa_generic_dtype_kernel_indexed, dim=ifa.size, inputs=[ifa, ifb], device=device)

        assert_np_equal(ifb.numpy(), 2 * ifa.numpy())


@wp.kernel
def fa_generic_array_kernel(a: Any, b: Any):
    i = wp.tid()
    b[i] = a[i] + a[i]


def test_fabricarray_generic_array(test, device):
    for T in _fabric_types:
        if hasattr(T, "_wp_scalar_type_"):
            nptype = wp.types.warp_type_to_np_dtype[T._wp_scalar_type_]
        else:
            nptype = wp.types.warp_type_to_np_dtype[T]

        data = wp.array(data=np.arange(100, dtype=nptype), device=device)
        data_iface = _create_fabric_array_interface(data, "foo", copy=True)
        fa = wp.fabricarray(data=data_iface, attrib="foo")

        result = wp.zeros_like(data)
        result_iface = _create_fabric_array_interface(result, "foo", copy=True)
        fb = wp.fabricarray(data=result_iface, attrib="foo")

        test.assertEqual(fa.dtype, fb.dtype)
        test.assertEqual(fa.ndim, fb.ndim)
        test.assertEqual(fa.shape, fb.shape)
        test.assertEqual(fa.size, fb.size)

        wp.launch(fa_generic_array_kernel, dim=fa.size, inputs=[fa, fb], device=device)

        assert_np_equal(fb.numpy(), 2 * fa.numpy())

        # reset data
        wp.copy(fa, data)
        wp.copy(fb, result)

        # test indexed
        indices = wp.array(data=np.arange(1, data.size, 2, dtype=np.int32), device=device)
        ifa = fa[indices]
        ifb = fb[indices]

        test.assertEqual(ifa.dtype, ifb.dtype)
        test.assertEqual(ifa.ndim, ifb.ndim)
        test.assertEqual(ifa.shape, ifb.shape)
        test.assertEqual(ifa.size, ifb.size)

        wp.launch(fa_generic_array_kernel, dim=ifa.size, inputs=[ifa, ifb], device=device)

        assert_np_equal(ifb.numpy(), 2 * ifa.numpy())


def test_fabricarray_empty(test, device):
    # Test whether common operations work with empty (zero-sized) indexed arrays
    # without throwing exceptions.

    def test_empty_ops(nrows, ncols, wptype, nptype):
        # scalar, vector, or matrix
        if ncols > 0:
            if nrows > 0:
                wptype = wp.types.matrix((nrows, ncols), wptype)
            else:
                wptype = wp.types.vector(ncols, wptype)
            dtype_shape = wptype._shape_
        else:
            dtype_shape = ()

        fill_value = wptype(42)

        # create an empty data array
        data = wp.empty(0, dtype=wptype, device=device)
        iface = _create_fabric_array_interface(data, "foo", copy=True)
        fa = wp.fabricarray(data=iface, attrib="foo")

        test.assertEqual(fa.size, 0)
        test.assertEqual(fa.shape, (0,))

        # all of these methods should succeed with zero-sized arrays
        fa.zero_()
        fa.fill_(fill_value)
        fb = fa.contiguous()

        fb = wp.empty_like(fa)
        fb = wp.zeros_like(fa)
        fb = wp.full_like(fa, fill_value)
        fb = wp.clone(fa)

        wp.copy(fa, fb)
        fa.assign(fb)

        na = fa.numpy()
        test.assertEqual(na.size, 0)
        test.assertEqual(na.shape, (0, *dtype_shape))
        test.assertEqual(na.dtype, nptype)

        test.assertEqual(fa.list(), [])

        # test indexed

        # create a zero-sized array of indices
        indices = wp.empty(0, dtype=int, device=device)

        ifa = fa[indices]

        test.assertEqual(ifa.size, 0)
        test.assertEqual(ifa.shape, (0,))

        # all of these methods should succeed with zero-sized arrays
        ifa.zero_()
        ifa.fill_(fill_value)
        ifb = ifa.contiguous()

        ifb = wp.empty_like(ifa)
        ifb = wp.zeros_like(ifa)
        ifb = wp.full_like(ifa, fill_value)
        ifb = wp.clone(ifa)

        wp.copy(ifa, ifb)
        ifa.assign(ifb)

        na = ifa.numpy()
        test.assertEqual(na.size, 0)
        test.assertEqual(na.shape, (0, *dtype_shape))
        test.assertEqual(na.dtype, nptype)

        test.assertEqual(ifa.list(), [])

    # test with scalars, vectors, and matrices
    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        # scalars
        test_empty_ops(0, 0, wptype, nptype)

        for ncols in [2, 3, 4, 5]:
            # vectors
            test_empty_ops(0, ncols, wptype, nptype)
            # square matrices (the Fabric interface only supports square matrices right now)
            test_empty_ops(ncols, ncols, wptype, nptype)


def test_fabricarray_fill_scalar(test, device):
    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        # create a data array
        data = wp.zeros(100, dtype=wptype, device=device)
        iface = _create_fabric_array_interface(data, "foo", copy=True)
        fa = wp.fabricarray(data=iface, attrib="foo")

        assert_np_equal(fa.numpy(), np.zeros(fa.shape, dtype=nptype))

        # fill with int value
        fill_value = 42
        fa.fill_(fill_value)
        assert_np_equal(fa.numpy(), np.full(fa.shape, fill_value, dtype=nptype))

        fa.zero_()
        assert_np_equal(fa.numpy(), np.zeros(fa.shape, dtype=nptype))

        if wptype in wp.types.float_types:
            # fill with float value
            fill_value = 13.37
            fa.fill_(fill_value)
            assert_np_equal(fa.numpy(), np.full(fa.shape, fill_value, dtype=nptype))

        # fill with Warp scalar value
        fill_value = wptype(17)
        fa.fill_(fill_value)
        assert_np_equal(fa.numpy(), np.full(fa.shape, fill_value.value, dtype=nptype))

        # reset data
        wp.copy(fa, data)

        # test indexed
        indices1 = wp.array(data=np.arange(1, data.size, 2, dtype=np.int32), device=device)
        ifa = fa[indices1]

        # ensure that the other indices remain unchanged
        indices2 = wp.array(data=np.arange(0, data.size, 2, dtype=np.int32), device=device)
        ifb = fa[indices2]

        assert_np_equal(ifa.numpy(), np.zeros(ifa.shape, dtype=nptype))
        assert_np_equal(ifb.numpy(), np.zeros(ifb.shape, dtype=nptype))

        # fill with int value
        fill_value = 42
        ifa.fill_(fill_value)
        assert_np_equal(ifa.numpy(), np.full(ifa.shape, fill_value, dtype=nptype))
        assert_np_equal(ifb.numpy(), np.zeros(ifb.shape, dtype=nptype))

        ifa.zero_()
        assert_np_equal(ifa.numpy(), np.zeros(ifa.shape, dtype=nptype))
        assert_np_equal(ifb.numpy(), np.zeros(ifb.shape, dtype=nptype))

        if wptype in wp.types.float_types:
            # fill with float value
            fill_value = 13.37
            ifa.fill_(fill_value)
            assert_np_equal(ifa.numpy(), np.full(ifa.shape, fill_value, dtype=nptype))
            assert_np_equal(ifb.numpy(), np.zeros(ifb.shape, dtype=nptype))

        # fill with Warp scalar value
        fill_value = wptype(17)
        ifa.fill_(fill_value)
        assert_np_equal(ifa.numpy(), np.full(ifa.shape, fill_value.value, dtype=nptype))
        assert_np_equal(ifb.numpy(), np.zeros(ifb.shape, dtype=nptype))


def test_fabricarray_fill_vector(test, device):
    # test filling a vector array with scalar or vector values (vec_type, list, or numpy array)

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        # vector types
        vector_types = [
            wp.types.vector(2, wptype),
            wp.types.vector(3, wptype),
            wp.types.vector(4, wptype),
            wp.types.vector(5, wptype),
        ]

        for vec_type in vector_types:
            vec_len = vec_type._length_

            data = wp.zeros(100, dtype=vec_type, device=device)
            iface = _create_fabric_array_interface(data, "foo", copy=True)
            fa = wp.fabricarray(data=iface, attrib="foo")

            assert_np_equal(fa.numpy(), np.zeros((*fa.shape, vec_len), dtype=nptype))

            # fill with int scalar
            fill_value = 42
            fa.fill_(fill_value)
            assert_np_equal(fa.numpy(), np.full((*fa.shape, vec_len), fill_value, dtype=nptype))

            # test zeroing
            fa.zero_()
            assert_np_equal(fa.numpy(), np.zeros((*fa.shape, vec_len), dtype=nptype))

            # vector values can be passed as a list, numpy array, or Warp vector instance
            fill_list = [17, 42, 99, 101, 127][:vec_len]
            fill_arr = np.array(fill_list, dtype=nptype)
            fill_vec = vec_type(fill_list)

            expected = np.tile(fill_arr, fa.size).reshape((*fa.shape, vec_len))

            # fill with list of vector length
            fa.fill_(fill_list)
            assert_np_equal(fa.numpy(), expected)

            # clear
            fa.zero_()

            # fill with numpy array of vector length
            fa.fill_(fill_arr)
            assert_np_equal(fa.numpy(), expected)

            # clear
            fa.zero_()

            # fill with vec instance
            fa.fill_(fill_vec)
            assert_np_equal(fa.numpy(), expected)

            if wptype in wp.types.float_types:
                # fill with float scalar
                fill_value = 13.37
                fa.fill_(fill_value)
                assert_np_equal(fa.numpy(), np.full((*fa.shape, vec_len), fill_value, dtype=nptype))

                # fill with float list of vector length
                fill_list = [-2.5, -1.25, 1.25, 2.5, 5.0][:vec_len]

                fa.fill_(fill_list)

                expected = np.tile(np.array(fill_list, dtype=nptype), fa.size).reshape((*fa.shape, vec_len))

                assert_np_equal(fa.numpy(), expected)

            # reset data
            wp.copy(fa, data)

            # test indexed
            indices1 = wp.array(data=np.arange(1, data.size, 2, dtype=np.int32), device=device)
            ifa = fa[indices1]

            # ensure that the other indices remain unchanged
            indices2 = wp.array(data=np.arange(0, data.size, 2, dtype=np.int32), device=device)
            ifb = fa[indices2]

            assert_np_equal(ifa.numpy(), np.zeros((*ifa.shape, vec_len), dtype=nptype))
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, vec_len), dtype=nptype))

            # fill with int scalar
            fill_value = 42
            ifa.fill_(fill_value)
            assert_np_equal(ifa.numpy(), np.full((*ifa.shape, vec_len), fill_value, dtype=nptype))
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, vec_len), dtype=nptype))

            # test zeroing
            ifa.zero_()
            assert_np_equal(ifa.numpy(), np.zeros((*ifa.shape, vec_len), dtype=nptype))
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, vec_len), dtype=nptype))

            # vector values can be passed as a list, numpy array, or Warp vector instance
            fill_list = [17, 42, 99, 101, 127][:vec_len]
            fill_arr = np.array(fill_list, dtype=nptype)
            fill_vec = vec_type(fill_list)

            expected = np.tile(fill_arr, ifa.size).reshape((*ifa.shape, vec_len))

            # fill with list of vector length
            ifa.fill_(fill_list)
            assert_np_equal(ifa.numpy(), expected)
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, vec_len), dtype=nptype))

            # clear
            ifa.zero_()

            # fill with numpy array of vector length
            ifa.fill_(fill_arr)
            assert_np_equal(ifa.numpy(), expected)
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, vec_len), dtype=nptype))

            # clear
            ifa.zero_()

            # fill with vec instance
            ifa.fill_(fill_vec)
            assert_np_equal(ifa.numpy(), expected)
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, vec_len), dtype=nptype))

            if wptype in wp.types.float_types:
                # fill with float scalar
                fill_value = 13.37
                ifa.fill_(fill_value)
                assert_np_equal(ifa.numpy(), np.full((*ifa.shape, vec_len), fill_value, dtype=nptype))
                assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, vec_len), dtype=nptype))

                # fill with float list of vector length
                fill_list = [-2.5, -1.25, 1.25, 2.5, 5.0][:vec_len]

                ifa.fill_(fill_list)

                expected = np.tile(np.array(fill_list, dtype=nptype), ifa.size).reshape((*ifa.shape, vec_len))

                assert_np_equal(ifa.numpy(), expected)
                assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, vec_len), dtype=nptype))


def test_fabricarray_fill_matrix(test, device):
    # test filling a matrix array with scalar or matrix values (mat_type, nested list, or 2d numpy array)

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        # matrix types
        matrix_types = [
            # square matrices only
            wp.types.matrix((2, 2), wptype),
            wp.types.matrix((3, 3), wptype),
            wp.types.matrix((4, 4), wptype),
            wp.types.matrix((5, 5), wptype),
        ]

        for mat_type in matrix_types:
            mat_len = mat_type._length_
            mat_shape = mat_type._shape_

            data = wp.zeros(100, dtype=mat_type, device=device)
            iface = _create_fabric_array_interface(data, "foo", copy=True)
            fa = wp.fabricarray(data=iface, attrib="foo")

            assert_np_equal(fa.numpy(), np.zeros((*fa.shape, *mat_shape), dtype=nptype))

            # fill with scalar
            fill_value = 42
            fa.fill_(fill_value)
            assert_np_equal(fa.numpy(), np.full((*fa.shape, *mat_shape), fill_value, dtype=nptype))

            # test zeroing
            fa.zero_()
            assert_np_equal(fa.numpy(), np.zeros((*fa.shape, *mat_shape), dtype=nptype))

            # matrix values can be passed as a 1d numpy array, 2d numpy array, flat list, nested list, or Warp matrix instance
            if wptype != wp.bool:
                fill_arr1 = np.arange(mat_len, dtype=nptype)
            else:
                fill_arr1 = np.ones(mat_len, dtype=nptype)

            fill_arr2 = fill_arr1.reshape(mat_shape)
            fill_list1 = list(fill_arr1)
            fill_list2 = [list(row) for row in fill_arr2]
            fill_mat = mat_type(fill_arr1)

            expected = np.tile(fill_arr1, fa.size).reshape((*fa.shape, *mat_shape))

            # fill with 1d numpy array
            fa.fill_(fill_arr1)
            assert_np_equal(fa.numpy(), expected)

            # clear
            fa.zero_()

            # fill with 2d numpy array
            fa.fill_(fill_arr2)
            assert_np_equal(fa.numpy(), expected)

            # clear
            fa.zero_()

            # fill with flat list
            fa.fill_(fill_list1)
            assert_np_equal(fa.numpy(), expected)

            # clear
            fa.zero_()

            # fill with nested list
            fa.fill_(fill_list2)
            assert_np_equal(fa.numpy(), expected)

            # clear
            fa.zero_()

            # fill with mat instance
            fa.fill_(fill_mat)
            assert_np_equal(fa.numpy(), expected)

            # reset data
            wp.copy(fa, data)

            # test indexed
            indices1 = wp.array(data=np.arange(1, data.size, 2, dtype=np.int32), device=device)
            ifa = fa[indices1]

            # ensure that the other indices remain unchanged
            indices2 = wp.array(data=np.arange(0, data.size, 2, dtype=np.int32), device=device)
            ifb = fa[indices2]

            assert_np_equal(ifa.numpy(), np.zeros((*ifa.shape, *mat_shape), dtype=nptype))
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, *mat_shape), dtype=nptype))

            # fill with scalar
            fill_value = 42
            ifa.fill_(fill_value)
            assert_np_equal(ifa.numpy(), np.full((*ifa.shape, *mat_shape), fill_value, dtype=nptype))
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, *mat_shape), dtype=nptype))

            # test zeroing
            ifa.zero_()
            assert_np_equal(ifa.numpy(), np.zeros((*ifa.shape, *mat_shape), dtype=nptype))
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, *mat_shape), dtype=nptype))

            # matrix values can be passed as a 1d numpy array, 2d numpy array, flat list, nested list, or Warp matrix instance
            if wptype != wp.bool:
                fill_arr1 = np.arange(mat_len, dtype=nptype)
            else:
                fill_arr1 = np.ones(mat_len, dtype=nptype)
            fill_arr2 = fill_arr1.reshape(mat_shape)
            fill_list1 = list(fill_arr1)
            fill_list2 = [list(row) for row in fill_arr2]
            fill_mat = mat_type(fill_arr1)

            expected = np.tile(fill_arr1, ifa.size).reshape((*ifa.shape, *mat_shape))

            # fill with 1d numpy array
            ifa.fill_(fill_arr1)
            assert_np_equal(ifa.numpy(), expected)
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, *mat_shape), dtype=nptype))

            # clear
            ifa.zero_()

            # fill with 2d numpy array
            ifa.fill_(fill_arr2)
            assert_np_equal(ifa.numpy(), expected)
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, *mat_shape), dtype=nptype))

            # clear
            ifa.zero_()

            # fill with flat list
            ifa.fill_(fill_list1)
            assert_np_equal(ifa.numpy(), expected)
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, *mat_shape), dtype=nptype))

            # clear
            ifa.zero_()

            # fill with nested list
            ifa.fill_(fill_list2)
            assert_np_equal(ifa.numpy(), expected)
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, *mat_shape), dtype=nptype))

            # clear
            ifa.zero_()

            # fill with mat instance
            ifa.fill_(fill_mat)
            assert_np_equal(ifa.numpy(), expected)
            assert_np_equal(ifb.numpy(), np.zeros((*ifb.shape, *mat_shape), dtype=nptype))


@wp.kernel
def fa_kernel_indexing_types(
    a: wp.fabricarray(dtype=wp.int32),
):
    x = a[wp.uint8(0)]
    y = a[wp.int16(1)]
    z = a[wp.uint32(2)]
    w = a[wp.int64(3)]

    a[wp.uint8(0)] = 123
    a[wp.int16(1)] = 123
    a[wp.uint32(2)] = 123
    a[wp.int64(3)] = 123

    wp.atomic_add(a, wp.uint8(0), 123)
    wp.atomic_sub(a, wp.int16(1), 123)
    # wp.atomic_min(a, wp.uint32(2), 123)
    # wp.atomic_max(a, wp.int64(3), 123)


def test_fabricarray_indexing_types(test, device):
    data = wp.zeros(shape=(4,), dtype=wp.int32, device=device)
    iface = _create_fabric_array_interface(data, "foo", copy=True)
    fa = wp.fabricarray(data=iface, attrib="foo")
    wp.launch(
        kernel=fa_kernel_indexing_types,
        dim=1,
        inputs=(fa,),
        device=device,
    )


@wp.kernel
def fa_generic_sums_kernel(a: wp.fabricarrayarray(dtype=Any), sums: wp.array(dtype=Any)):
    i = wp.tid()

    # get sub-array using wp::view()
    row = a[i]

    # get sub-array length
    count = row.shape[0]

    # compute sub-array sum
    for j in range(count):
        sums[i] = sums[i] + row[j]


@wp.kernel
def fa_generic_sums_kernel_indexed(a: wp.indexedfabricarrayarray(dtype=Any), sums: wp.array(dtype=Any)):
    i = wp.tid()

    # get sub-array using wp::view()
    row = a[i]

    # get sub-array length
    count = row.shape[0]

    # compute sub-array sum
    for j in range(count):
        sums[i] = sums[i] + row[j]


def test_fabricarrayarray(test, device):
    for T in _fabric_types:
        if hasattr(T, "_wp_scalar_type_"):
            nptype = wp.types.warp_type_to_np_dtype[T._wp_scalar_type_]
        else:
            nptype = wp.types.warp_type_to_np_dtype[T]

        n = 100

        min_length = 1
        max_length = 10
        arrays = []
        expected_sums = []
        expected_sums_indexed = []

        # generate data arrays
        length = min_length
        for i in range(n):
            if length > max_length:
                length = min_length

            na = np.arange(1, length + 1, dtype=nptype)

            arrays.append(wp.array(data=na, device=device))
            expected_sums.append(na.sum())

            # every second index
            if i % 2 == 0:
                expected_sums_indexed.append(na.sum())

            length += 1

        data_iface = _create_fabric_array_array_interface(arrays, "foo")
        fa = wp.fabricarrayarray(data=data_iface, attrib="foo")

        sums = wp.zeros_like(fa)

        test.assertEqual(fa.dtype, sums.dtype)
        test.assertEqual(fa.ndim, 2)
        test.assertEqual(sums.ndim, 1)
        test.assertEqual(fa.shape, sums.shape)
        test.assertEqual(fa.size, sums.size)

        wp.launch(fa_generic_sums_kernel, dim=fa.size, inputs=[fa, sums], device=device)

        assert_np_equal(sums.numpy(), np.array(expected_sums, dtype=nptype))

        # test indexed
        indices = wp.array(data=np.arange(0, n, 2, dtype=np.int32), device=device)
        ifa = fa[indices]

        sums = wp.zeros_like(ifa)

        test.assertEqual(ifa.dtype, sums.dtype)
        test.assertEqual(ifa.ndim, 2)
        test.assertEqual(sums.ndim, 1)
        test.assertEqual(ifa.shape, sums.shape)
        test.assertEqual(ifa.size, sums.size)

        wp.launch(fa_generic_sums_kernel_indexed, dim=ifa.size, inputs=[ifa, sums], device=device)

        assert_np_equal(sums.numpy(), np.array(expected_sums_indexed, dtype=nptype))


# explicit kernel overloads
for T in _fabric_types:
    wp.overload(fa_generic_dtype_kernel, [wp.fabricarray(dtype=T), wp.fabricarray(dtype=T)])
    wp.overload(fa_generic_dtype_kernel_indexed, [wp.indexedfabricarray(dtype=T), wp.indexedfabricarray(dtype=T)])

    wp.overload(fa_generic_array_kernel, [wp.fabricarray(dtype=T), wp.fabricarray(dtype=T)])
    wp.overload(fa_generic_array_kernel, [wp.indexedfabricarray(dtype=T), wp.indexedfabricarray(dtype=T)])

    wp.overload(fa_generic_sums_kernel, [wp.fabricarrayarray(dtype=T), wp.array(dtype=T)])
    wp.overload(fa_generic_sums_kernel_indexed, [wp.indexedfabricarrayarray(dtype=T), wp.array(dtype=T)])


devices = get_test_devices()


class TestFabricArray(unittest.TestCase):
    def test_fabricarray_new_del(self):
        # test the scenario in which a fabricarray is created but not initialized before gc
        instance = wp.fabricarray.__new__(wp.fabricarray)
        instance.__del__()


# fabric arrays
add_function_test(TestFabricArray, "test_fabricarray_kernel", test_fabricarray_kernel, devices=devices)
add_function_test(TestFabricArray, "test_fabricarray_empty", test_fabricarray_empty, devices=devices)
add_function_test(TestFabricArray, "test_fabricarray_generic_dtype", test_fabricarray_generic_dtype, devices=devices)
add_function_test(TestFabricArray, "test_fabricarray_generic_array", test_fabricarray_generic_array, devices=devices)
add_function_test(TestFabricArray, "test_fabricarray_fill_scalar", test_fabricarray_fill_scalar, devices=devices)
add_function_test(TestFabricArray, "test_fabricarray_fill_vector", test_fabricarray_fill_vector, devices=devices)
add_function_test(TestFabricArray, "test_fabricarray_fill_matrix", test_fabricarray_fill_matrix, devices=devices)
add_function_test(TestFabricArray, "test_fabricarray_indexing_types", test_fabricarray_indexing_types, devices=devices)

# fabric arrays of arrays
add_function_test(TestFabricArray, "test_fabricarrayarray", test_fabricarrayarray, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
