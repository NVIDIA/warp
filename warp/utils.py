# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cProfile
import ctypes
import os
import sys
import time
import warnings
from types import ModuleType
from typing import Any, Callable

import numpy as np

import warp as wp
import warp.context
import warp.types
from warp.context import Devicelike
from warp.types import Array, DType, type_repr, types_equal

warnings_seen = set()


def warp_showwarning(message, category, filename, lineno, file=None, line=None):
    """Version of warnings.showwarning that always prints to sys.stdout."""

    if warp.config.verbose_warnings:
        s = f"Warp {category.__name__}: {message} ({filename}:{lineno})\n"

        if line is None:
            try:
                import linecache

                line = linecache.getline(filename, lineno)
            except Exception:
                # When a warning is logged during Python shutdown, linecache
                # and the import machinery don't work anymore
                line = None
                linecache = None

        if line:
            line = line.strip()
            s += f"  {line}\n"
    else:
        # simple warning
        s = f"Warp {category.__name__}: {message}\n"

    sys.stdout.write(s)


def warn(message, category=None, stacklevel=1):
    if (category, message) in warnings_seen:
        return

    with warnings.catch_warnings():
        warnings.simplefilter("default")  # Change the filter in this process
        warnings.showwarning = warp_showwarning
        warnings.warn(
            message,
            category,
            stacklevel=stacklevel + 1,  # Increment stacklevel by 1 since we are in a wrapper
        )

    if category is DeprecationWarning:
        warnings_seen.add((category, message))


# expand a 7-vec to a tuple of arrays
def transform_expand(t):
    return wp.transform(np.array(t[0:3]), np.array(t[3:7]))


@wp.func
def quat_between_vectors(a: wp.vec3, b: wp.vec3) -> wp.quat:
    """
    Compute the quaternion that rotates vector a to vector b
    """
    a = wp.normalize(a)
    b = wp.normalize(b)
    c = wp.cross(a, b)
    d = wp.dot(a, b)
    q = wp.quat(c[0], c[1], c[2], 1.0 + d)
    return wp.normalize(q)


def array_scan(in_array, out_array, inclusive=True):
    """Perform a scan (prefix sum) operation on an array.

    This function computes the inclusive or exclusive scan of the input array and stores the result in the output array.
    The scan operation computes a running sum of elements in the array.

    Args:
        in_array (wp.array): Input array to scan. Must be of type int32 or float32.
        out_array (wp.array): Output array to store scan results. Must match input array type and size.
        inclusive (bool, optional): If True, performs an inclusive scan (includes current element in sum).
                                  If False, performs an exclusive scan (excludes current element). Defaults to True.

    Raises:
        RuntimeError: If array storage devices don't match, if storage size is insufficient, or if data types are unsupported.
    """

    if in_array.device != out_array.device:
        raise RuntimeError(f"In and out array storage devices do not match ({in_array.device} vs {out_array.device})")

    if in_array.size != out_array.size:
        raise RuntimeError(f"In and out array storage sizes do not match ({in_array.size} vs {out_array.size})")

    if not types_equal(in_array.dtype, out_array.dtype):
        raise RuntimeError(
            f"In and out array data types do not match ({type_repr(in_array.dtype)} vs {type_repr(out_array.dtype)})"
        )

    if in_array.size == 0:
        return

    from warp.context import runtime

    if in_array.device.is_cpu:
        if in_array.dtype == wp.int32:
            runtime.core.wp_array_scan_int_host(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        elif in_array.dtype == wp.float32:
            runtime.core.wp_array_scan_float_host(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(in_array.dtype)}")
    elif in_array.device.is_cuda:
        if in_array.dtype == wp.int32:
            runtime.core.wp_array_scan_int_device(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        elif in_array.dtype == wp.float32:
            runtime.core.wp_array_scan_float_device(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(in_array.dtype)}")


def radix_sort_pairs(keys, values, count: int):
    """Sort key-value pairs using radix sort.

    This function sorts pairs of arrays based on the keys array, maintaining the key-value
    relationship. The sort is stable and operates in linear time.
    The `keys` and `values` arrays must be large enough to accommodate 2*`count` elements.

    Args:
        keys (wp.array): Array of keys to sort. Must be of type int32, float32, or int64.
        values (wp.array): Array of values to sort along with keys. Must be of type int32.
        count (int): Number of elements to sort.

    Raises:
        RuntimeError: If array storage devices don't match, if storage size is insufficient, or if data types are unsupported.
    """
    if keys.device != values.device:
        raise RuntimeError(f"Keys and values array storage devices do not match ({keys.device} vs {values.device})")

    if count == 0:
        return

    if keys.size < 2 * count or values.size < 2 * count:
        raise RuntimeError("Keys and values array storage must be large enough to contain 2*count elements")

    from warp.context import runtime

    if keys.device.is_cpu:
        if keys.dtype == wp.int32 and values.dtype == wp.int32:
            runtime.core.wp_radix_sort_pairs_int_host(keys.ptr, values.ptr, count)
        elif keys.dtype == wp.float32 and values.dtype == wp.int32:
            runtime.core.wp_radix_sort_pairs_float_host(keys.ptr, values.ptr, count)
        elif keys.dtype == wp.int64 and values.dtype == wp.int32:
            runtime.core.wp_radix_sort_pairs_int64_host(keys.ptr, values.ptr, count)
        else:
            raise RuntimeError(
                f"Unsupported keys and values data types: {type_repr(keys.dtype)}, {type_repr(values.dtype)}"
            )
    elif keys.device.is_cuda:
        if keys.dtype == wp.int32 and values.dtype == wp.int32:
            runtime.core.wp_radix_sort_pairs_int_device(keys.ptr, values.ptr, count)
        elif keys.dtype == wp.float32 and values.dtype == wp.int32:
            runtime.core.wp_radix_sort_pairs_float_device(keys.ptr, values.ptr, count)
        elif keys.dtype == wp.int64 and values.dtype == wp.int32:
            runtime.core.wp_radix_sort_pairs_int64_device(keys.ptr, values.ptr, count)
        else:
            raise RuntimeError(
                f"Unsupported keys and values data types: {type_repr(keys.dtype)}, {type_repr(values.dtype)}"
            )


def segmented_sort_pairs(
    keys,
    values,
    count: int,
    segment_start_indices: wp.array(dtype=wp.int32),
    segment_end_indices: wp.array(dtype=wp.int32) = None,
):
    """Sort key-value pairs within segments.

    This function performs a segmented sort of key-value pairs, where the sorting is done independently within each segment.
    The segments are defined by their start and optionally end indices.
    The `keys` and `values` arrays must be large enough to accommodate 2*`count` elements.

    Args:
        keys: Array of keys to sort. Must be of type int32 or float32.
        values: Array of values to sort along with keys. Must be of type int32.
        count: Number of elements to sort.
        segment_start_indices: Array containing start index of each segment. Must be of type int32.
            If segment_end_indices is None, this array must have length at least num_segments + 1,
            and segment_end_indices will be inferred as segment_start_indices[1:].
            If segment_end_indices is provided, this array must have length at least num_segments.
        segment_end_indices: Optional array containing end index of each segment. Must be of type int32 if provided.
            If None, segment_end_indices will be inferred from segment_start_indices[1:].
            If provided, must have length at least num_segments.

    Raises:
        RuntimeError: If array storage devices don't match, if storage size is insufficient,
                     if segment_start_indices is not of type int32, or if data types are unsupported.
    """
    if keys.device != values.device:
        raise RuntimeError(f"Array storage devices do not match ({keys.device} vs {values.device})")

    if count == 0:
        return

    if keys.size < 2 * count or values.size < 2 * count:
        raise RuntimeError("Array storage must be large enough to contain 2*count elements")

    from warp.context import runtime

    if segment_start_indices.dtype != wp.int32:
        raise RuntimeError("segment_start_indices array must be of type int32")

    # Handle case where segment_end_indices is not provided
    if segment_end_indices is None:
        num_segments = max(0, segment_start_indices.size - 1)

        segment_end_indices = segment_start_indices[1:]
        segment_end_indices_ptr = segment_end_indices.ptr
        segment_start_indices_ptr = segment_start_indices.ptr
    else:
        if segment_end_indices.dtype != wp.int32:
            raise RuntimeError("segment_end_indices array must be of type int32")

        num_segments = segment_start_indices.size

        segment_end_indices_ptr = segment_end_indices.ptr
        segment_start_indices_ptr = segment_start_indices.ptr

    if keys.device.is_cpu:
        if keys.dtype == wp.int32 and values.dtype == wp.int32:
            runtime.core.wp_segmented_sort_pairs_int_host(
                keys.ptr,
                values.ptr,
                count,
                segment_start_indices_ptr,
                segment_end_indices_ptr,
                num_segments,
            )
        elif keys.dtype == wp.float32 and values.dtype == wp.int32:
            runtime.core.wp_segmented_sort_pairs_float_host(
                keys.ptr,
                values.ptr,
                count,
                segment_start_indices_ptr,
                segment_end_indices_ptr,
                num_segments,
            )
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(keys.dtype)}")
    elif keys.device.is_cuda:
        if keys.dtype == wp.int32 and values.dtype == wp.int32:
            runtime.core.wp_segmented_sort_pairs_int_device(
                keys.ptr,
                values.ptr,
                count,
                segment_start_indices_ptr,
                segment_end_indices_ptr,
                num_segments,
            )
        elif keys.dtype == wp.float32 and values.dtype == wp.int32:
            runtime.core.wp_segmented_sort_pairs_float_device(
                keys.ptr,
                values.ptr,
                count,
                segment_start_indices_ptr,
                segment_end_indices_ptr,
                num_segments,
            )
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(keys.dtype)}")


def runlength_encode(values, run_values, run_lengths, run_count=None, value_count=None):
    """Perform run-length encoding on an array.

    This function compresses an array by replacing consecutive identical values with a single value
    and its count. For example, [1,1,1,2,2,3] becomes values=[1,2,3] and lengths=[3,2,1].

    Args:
        values (wp.array): Input array to encode. Must be of type int32.
        run_values (wp.array): Output array to store unique values. Must be at least value_count in size.
        run_lengths (wp.array): Output array to store run lengths. Must be at least value_count in size.
        run_count (wp.array, optional): Optional output array to store the number of runs.
                                       If None, returns the count as an integer.
        value_count (int, optional): Number of values to process. If None, processes entire array.

    Returns:
        int or wp.array: Number of runs if run_count is None, otherwise returns run_count array.

    Raises:
        RuntimeError: If array storage devices don't match, if storage size is insufficient, or if data types are unsupported.
    """
    if run_values.device != values.device or run_lengths.device != values.device:
        raise RuntimeError("run_values, run_lengths and values storage devices do not match")

    if value_count is None:
        value_count = values.size

    if run_values.size < value_count or run_lengths.size < value_count:
        raise RuntimeError(f"Output array storage sizes must be at least equal to value_count ({value_count})")

    if not types_equal(values.dtype, run_values.dtype):
        raise RuntimeError(
            f"values and run_values data types do not match ({type_repr(values.dtype)} vs {type_repr(run_values.dtype)})"
        )

    if run_lengths.dtype != wp.int32:
        raise RuntimeError("run_lengths array must be of type int32")

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if run_count is None:
        if value_count == 0:
            return 0
        run_count = wp.empty(shape=(1,), dtype=int, device=values.device)
        host_return = True
    else:
        if run_count.device != values.device:
            raise RuntimeError("run_count storage device does not match other arrays")
        if run_count.dtype != wp.int32:
            raise RuntimeError("run_count array must be of type int32")
        if value_count == 0:
            run_count.zero_()
            return run_count
        host_return = False

    from warp.context import runtime

    if values.device.is_cpu:
        if values.dtype == wp.int32:
            runtime.core.wp_runlength_encode_int_host(
                values.ptr, run_values.ptr, run_lengths.ptr, run_count.ptr, value_count
            )
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(values.dtype)}")
    elif values.device.is_cuda:
        if values.dtype == wp.int32:
            runtime.core.wp_runlength_encode_int_device(
                values.ptr, run_values.ptr, run_lengths.ptr, run_count.ptr, value_count
            )
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(values.dtype)}")

    if host_return:
        return int(run_count.numpy()[0])
    return run_count


def array_sum(values, out=None, value_count=None, axis=None):
    """Compute the sum of array elements.

    This function computes the sum of array elements, optionally along a specified axis.
    The operation can be performed on the entire array or along a specific dimension.

    Args:
        values (wp.array): Input array to sum. Must be of type float32 or float64.
        out (wp.array, optional): Output array to store results. If None, a new array is created.
        value_count (int, optional): Number of elements to process. If None, processes entire array.
        axis (int, optional): Axis along which to compute sum. If None, computes sum of all elements.

    Returns:
        wp.array or float: The sum result. Returns a float if axis is None and out is None,
                           otherwise returns the output array.

    Raises:
        RuntimeError: If output array storage device or data type is incompatible with input array.
    """
    if value_count is None:
        if axis is None:
            value_count = values.size
        else:
            value_count = values.shape[axis]

    if axis is None:
        output_shape = (1,)
    else:

        def output_dim(ax, dim):
            return 1 if ax == axis else dim

        output_shape = tuple(output_dim(ax, dim) for ax, dim in enumerate(values.shape))

    type_size = wp.types.type_size(values.dtype)
    scalar_type = wp.types.type_scalar_type(values.dtype)

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if out is None:
        host_return = True
        out = wp.empty(shape=output_shape, dtype=values.dtype, device=values.device)
    else:
        host_return = False
        if out.device != values.device:
            raise RuntimeError("out storage device should match values array")
        if out.dtype != values.dtype:
            raise RuntimeError(f"out array should have type {values.dtype.__name__}")
        if out.shape != output_shape:
            raise RuntimeError(f"out array should have shape {output_shape}")

    if value_count == 0:
        out.zero_()
        if axis is None and host_return:
            return out.numpy()[0]
        return out

    from warp.context import runtime

    if values.device.is_cpu:
        if scalar_type == wp.float32:
            native_func = runtime.core.wp_array_sum_float_host
        elif scalar_type == wp.float64:
            native_func = runtime.core.wp_array_sum_double_host
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(values.dtype)}")
    elif values.device.is_cuda:
        if scalar_type == wp.float32:
            native_func = runtime.core.wp_array_sum_float_device
        elif scalar_type == wp.float64:
            native_func = runtime.core.wp_array_sum_double_device
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(values.dtype)}")

    if axis is None:
        stride = wp.types.type_size_in_bytes(values.dtype)
        native_func(values.ptr, out.ptr, value_count, stride, type_size)

        if host_return:
            return out.numpy()[0]
        return out

    stride = values.strides[axis]
    for idx in np.ndindex(output_shape):
        out_offset = sum(i * s for i, s in zip(idx, out.strides))
        val_offset = sum(i * s for i, s in zip(idx, values.strides))

        native_func(
            values.ptr + val_offset,
            out.ptr + out_offset,
            value_count,
            stride,
            type_size,
        )

    return out


def array_inner(a, b, out=None, count=None, axis=None):
    """Compute the inner product of two arrays.

    This function computes the dot product between two arrays, optionally along a specified axis.
    The operation can be performed on the entire arrays or along a specific dimension.

    Args:
        a (wp.array): First input array.
        b (wp.array): Second input array. Must match shape and type of a.
        out (wp.array, optional): Output array to store results. If None, a new array is created.
        count (int, optional): Number of elements to process. If None, processes entire arrays.
        axis (int, optional): Axis along which to compute inner product. If None, computes on flattened arrays.

    Returns:
        wp.array or float: The inner product result. Returns a float if axis is None and out is None,
                           otherwise returns the output array.

    Raises:
        RuntimeError: If array storage devices, sizes, or data types are incompatible.
    """
    if a.size != b.size:
        raise RuntimeError(f"A and b array storage sizes do not match ({a.size} vs {b.size})")

    if a.device != b.device:
        raise RuntimeError(f"A and b array storage devices do not match ({a.device} vs {b.device})")

    if not types_equal(a.dtype, b.dtype):
        raise RuntimeError(f"A and b array data types do not match ({type_repr(a.dtype)} vs {type_repr(b.dtype)})")

    if count is None:
        if axis is None:
            count = a.size
        else:
            count = a.shape[axis]

    if axis is None:
        output_shape = (1,)
    else:

        def output_dim(ax, dim):
            return 1 if ax == axis else dim

        output_shape = tuple(output_dim(ax, dim) for ax, dim in enumerate(a.shape))

    type_size = wp.types.type_size(a.dtype)
    scalar_type = wp.types.type_scalar_type(a.dtype)

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if out is None:
        host_return = True
        out = wp.empty(shape=output_shape, dtype=scalar_type, device=a.device)
    else:
        host_return = False
        if out.device != a.device:
            raise RuntimeError("out storage device should match values array")
        if out.dtype != scalar_type:
            raise RuntimeError(f"out array should have type {scalar_type.__name__}")
        if out.shape != output_shape:
            raise RuntimeError(f"out array should have shape {output_shape}")

    if count == 0:
        if axis is None and host_return:
            return 0.0
        out.zero_()
        return out

    from warp.context import runtime

    if a.device.is_cpu:
        if scalar_type == wp.float32:
            native_func = runtime.core.wp_array_inner_float_host
        elif scalar_type == wp.float64:
            native_func = runtime.core.wp_array_inner_double_host
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(a.dtype)}")
    elif a.device.is_cuda:
        if scalar_type == wp.float32:
            native_func = runtime.core.wp_array_inner_float_device
        elif scalar_type == wp.float64:
            native_func = runtime.core.wp_array_inner_double_device
        else:
            raise RuntimeError(f"Unsupported data type: {type_repr(a.dtype)}")

    if axis is None:
        stride_a = wp.types.type_size_in_bytes(a.dtype)
        stride_b = wp.types.type_size_in_bytes(b.dtype)
        native_func(a.ptr, b.ptr, out.ptr, count, stride_a, stride_b, type_size)

        if host_return:
            return out.numpy()[0]
        return out

    stride_a = a.strides[axis]
    stride_b = b.strides[axis]

    for idx in np.ndindex(output_shape):
        out_offset = sum(i * s for i, s in zip(idx, out.strides))
        a_offset = sum(i * s for i, s in zip(idx, a.strides))
        b_offset = sum(i * s for i, s in zip(idx, b.strides))

        native_func(
            a.ptr + a_offset,
            b.ptr + b_offset,
            out.ptr + out_offset,
            count,
            stride_a,
            stride_b,
            type_size,
        )

    return out


@wp.kernel
def _array_cast_kernel(
    dest: Any,
    src: Any,
):
    i = wp.tid()
    dest[i] = dest.dtype(src[i])


def array_cast(in_array, out_array, count=None):
    """Cast elements from one array to another array with a different data type.

    This function performs element-wise casting from the input array to the output array.
    The arrays must have the same number of dimensions and data type shapes. If they don't match,
    the arrays will be flattened and casting will be performed at the scalar level.

    Args:
        in_array (wp.array): Input array to cast from.
        out_array (wp.array): Output array to cast to. Must have the same device as in_array.
        count (int, optional): Number of elements to process. If None, processes entire array.
                             For multi-dimensional arrays, partial casting is not supported.

    Raises:
        RuntimeError: If arrays have different devices or if attempting partial casting
                     on multi-dimensional arrays.

    Note:
        If the input and output arrays have the same data type, this function will
        simply copy the data without any conversion.
    """
    if in_array.device != out_array.device:
        raise RuntimeError(f"Array storage devices do not match ({in_array.device} vs {out_array.device})")

    in_array_data_shape = getattr(in_array.dtype, "_shape_", ())
    out_array_data_shape = getattr(out_array.dtype, "_shape_", ())

    if in_array.ndim != out_array.ndim or in_array_data_shape != out_array_data_shape:
        # Number of dimensions or data type shape do not match.
        # Flatten arrays and do cast at the scalar level
        in_array = in_array.flatten()
        out_array = out_array.flatten()

        in_array_data_length = warp.types.type_size(in_array.dtype)
        out_array_data_length = warp.types.type_size(out_array.dtype)
        in_array_scalar_type = wp.types.type_scalar_type(in_array.dtype)
        out_array_scalar_type = wp.types.type_scalar_type(out_array.dtype)

        in_array = wp.array(
            data=None,
            ptr=in_array.ptr,
            capacity=in_array.capacity,
            device=in_array.device,
            dtype=in_array_scalar_type,
            shape=in_array.shape[0] * in_array_data_length,
        )

        out_array = wp.array(
            data=None,
            ptr=out_array.ptr,
            capacity=out_array.capacity,
            device=out_array.device,
            dtype=out_array_scalar_type,
            shape=out_array.shape[0] * out_array_data_length,
        )

        if count is not None:
            count *= in_array_data_length

    if count is None:
        count = in_array.size

    if in_array.ndim == 1:
        dim = count
    elif count < in_array.size:
        raise RuntimeError("Partial cast is not supported for arrays with more than one dimension")
    else:
        dim = in_array.shape

    if in_array.dtype == out_array.dtype:
        # Same data type, can simply copy
        wp.copy(dest=out_array, src=in_array, count=count)
    else:
        wp.launch(kernel=_array_cast_kernel, dim=dim, inputs=[out_array, in_array], device=out_array.device)


def create_warp_function(func: Callable) -> tuple[wp.Function, warp.context.Module]:
    """Create a Warp function from a Python function.

    Args:
        func (Callable): A Python function to be converted to a Warp function.

    Returns:
        wp.Function: A Warp function created from the input function.
    """

    from .codegen import Adjoint, get_full_arg_spec

    def unique_name(code: str):
        return "func_" + hex(hash(code))[-8:]

    # Create a Warp function from the input function
    source = None
    argspec = get_full_arg_spec(func)
    key = getattr(func, "__name__", None)
    if key is None:
        source, _ = Adjoint.extract_function_source(func)
        key = unique_name(source)
    elif key == "<lambda>":
        body = Adjoint.extract_lambda_source(func, only_body=True)
        if body is None:
            raise ValueError("Could not extract lambda source code")
        key = unique_name(body)
        source = f"def {key}({', '.join(argspec.args)}):\n  return {body}"
    else:
        # use the qualname of the function as the key
        key = getattr(func, "__qualname__", key)
        key = key.replace(".", "_").replace(" ", "_").replace("<", "").replace(">", "_")

    module = warp.context.get_module(f"map_{key}")
    func = wp.Function(
        func,
        namespace="",
        module=module,
        key=key,
        source=source,
        overloaded_annotations=dict.fromkeys(argspec.args, Any),
    )
    return func, module


def broadcast_shapes(shapes: list[tuple[int]]) -> tuple[int]:
    """Broadcast a list of shapes to a common shape.

    Following the broadcasting rules of NumPy, two shapes are compatible when:
    starting from the trailing dimension,
        1. the two dimensions are equal, or
        2. one of the dimensions is 1.

    Example:
        >>> broadcast_shapes([(3, 1, 4), (5, 4)])
        (3, 5, 4)

    Returns:
        tuple[int]: The broadcasted shape.

    Raises:
        ValueError: If the shapes are not broadcastable.
    """
    ref = shapes[0]
    for shape in shapes[1:]:
        broad = []
        for j in range(1, max(len(ref), len(shape)) + 1):
            if j <= len(ref) and j <= len(shape):
                s = shape[-j]
                r = ref[-j]
                if s == r:
                    broad.append(s)
                elif s == 1 or r == 1:
                    broad.append(max(s, r))
                else:
                    raise ValueError(f"Shapes {ref} and {shape} are not broadcastable")
            elif j <= len(ref):
                broad.append(ref[-j])
            else:
                broad.append(shape[-j])
        ref = tuple(reversed(broad))
    return ref


def map(
    func: Callable | wp.Function,
    *inputs: Array[DType] | Any,
    out: Array[DType] | list[Array[DType]] | None = None,
    return_kernel: bool = False,
    block_dim=256,
    device: Devicelike = None,
) -> Array[DType] | list[Array[DType]] | wp.Kernel:
    """
    Map a function over the elements of one or more arrays.

    You can use a Warp function, a regular Python function, or a lambda expression to map it to a set of arrays.

    .. testcode::

        a = wp.array([1, 2, 3], dtype=wp.float32)
        b = wp.array([4, 5, 6], dtype=wp.float32)
        c = wp.array([7, 8, 9], dtype=wp.float32)
        result = wp.map(lambda x, y, z: x + 2.0 * y - z, a, b, c)
        print(result)

    .. testoutput::

        [2. 4. 6.]

    Clamp values in an array in place:

    .. testcode::

        xs = wp.array([-1.0, 0.0, 1.0], dtype=wp.float32)
        wp.map(wp.clamp, xs, -0.5, 0.5, out=xs)
        print(xs)

    .. testoutput::

        [-0.5  0.   0.5]

    Note that only one of the inputs must be a Warp array. For example, it is possible
    vectorize the function :func:`warp.transform_point` over a collection of points
    with a given input transform as follows:

    .. code-block:: python

        tf = wp.transform((1.0, 2.0, 3.0), wp.quat_rpy(0.2, -0.6, 0.1))
        points = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=wp.vec3)
        transformed = wp.map(wp.transform_point, tf, points)

    Besides regular Warp arrays, other array types, such as the ``indexedarray``, are supported as well:

    .. testcode::

        arr = wp.array(data=np.arange(10, dtype=np.float32))
        indices = wp.array([1, 3, 5, 7, 9], dtype=int)
        iarr = wp.indexedarray1d(arr, [indices])
        out = wp.map(lambda x: x * 10.0, iarr)
        print(out)

    .. testoutput::

        [10. 30. 50. 70. 90.]

    If multiple arrays are provided, the
    `NumPy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
    are applied to determine the shape of the output array.
    Two shapes are compatible when:
    starting from the trailing dimension,

    1. the two dimensions are equal, or
    2. one of the dimensions is 1.

    For example, given arrays of shapes ``(3, 1, 4)`` and ``(5, 4)``, the broadcasted
    shape is ``(3, 5, 4)``.

    If no array(s) are provided to the ``out`` argument, the output array(s) are created automatically.
    The data type(s) of the output array(s) are determined by the type of the return value(s) of
    the function. The ``requires_grad`` flag for an automatically created output array is set to ``True``
    if any of the input arrays have it set to ``True`` and the respective output array's ``dtype`` is a type that
    supports differentiation.

    Args:
        func (Callable | Function): The function to map over the arrays.
        *inputs (array | Any): The input arrays or values to pass to the function.
        out (array | list[array] | None): Optional output array(s) to store the result(s). If None, the output array(s) will be created automatically.
        return_kernel (bool): If True, only return the generated kernel without performing the mapping operation.
        block_dim (int): The block dimension for the kernel launch.
        device (Devicelike): The device on which to run the kernel.

    Returns:
        array | list[array] | Kernel:
            The resulting array(s) of the mapping. If ``return_kernel`` is True, only returns the kernel used for mapping.
    """

    import builtins

    from .codegen import Adjoint, Struct, StructInstance
    from .types import (
        is_array,
        type_is_matrix,
        type_is_quaternion,
        type_is_transformation,
        type_is_vector,
        type_repr,
        type_to_warp,
        types_equal,
    )

    # mapping from struct name to its Python definition
    referenced_modules: dict[str, ModuleType] = {}

    def type_to_code(wp_type) -> str:
        """Returns the string representation of a given Warp type."""
        if is_array(wp_type):
            return f"warp.array(ndim={wp_type.ndim}, dtype={type_to_code(wp_type.dtype)})"
        if isinstance(wp_type, Struct):
            key = f"{wp_type.__module__}.{wp_type.key}"
            module = sys.modules.get(wp_type.__module__, None)
            if module is not None:
                referenced_modules[wp_type.__module__] = module
            return key
        if type_is_transformation(wp_type):
            return f"warp.types.transformation(dtype={type_to_code(wp_type._wp_scalar_type_)})"
        if type_is_quaternion(wp_type):
            return f"warp.types.quaternion(dtype={type_to_code(wp_type._wp_scalar_type_)})"
        if type_is_vector(wp_type):
            return f"warp.types.vector(length={wp_type._shape_[0]}, dtype={type_to_code(wp_type._wp_scalar_type_)})"
        if type_is_matrix(wp_type):
            return f"warp.types.matrix(shape=({wp_type._shape_[0]}, {wp_type._shape_[1]}), dtype={type_to_code(wp_type._wp_scalar_type_)})"
        if wp_type == builtins.bool:
            return "bool"
        if wp_type == builtins.float:
            return "float"
        if wp_type == builtins.int:
            return "int"

        name = getattr(wp_type, "__name__", None)
        if name is None:
            return type_repr(wp_type)
        name = getattr(wp_type, "__qualname__", name)
        module = getattr(wp_type, "__module__", None)
        if module is not None:
            referenced_modules[wp_type.__module__] = module
        return wp_type.__module__ + "." + name

    def get_warp_type(value):
        dtype = type(value)
        if issubclass(dtype, StructInstance):
            # a struct
            return value._cls
        return type_to_warp(dtype)

    # gather the arrays in the inputs
    array_shapes = [a.shape for a in inputs if is_array(a)]
    if len(array_shapes) == 0:
        raise ValueError("map requires at least one warp.array input")
    # broadcast the shapes of the arrays
    out_shape = broadcast_shapes(array_shapes)

    module = None
    out_dtypes = None
    if isinstance(func, wp.Function):
        func_name = func.key
        wp_func = func
    else:
        # check if op is a callable function
        if not callable(func):
            raise TypeError("func must be a callable function or a warp.Function")
        wp_func, module = create_warp_function(func)
        func_name = wp_func.key
    if module is None:
        module = warp.context.get_module(f"map_{func_name}")

    arg_names = list(wp_func.input_types.keys())

    if len(inputs) != len(arg_names):
        raise TypeError(
            f"Number of input arguments ({len(inputs)}) does not match expected number of function arguments ({len(arg_names)})"
        )

    # determine output dtype
    arg_types = {}
    arg_values = {}
    for i, arg_name in enumerate(arg_names):
        if is_array(inputs[i]):
            # we will pass an element of the array to the function
            arg_types[arg_name] = inputs[i].dtype
            if device is None:
                device = inputs[i].device
        else:
            # we pass the input value directly to the function
            arg_types[arg_name] = get_warp_type(inputs[i])
    func_or_none = wp_func.get_overload(list(arg_types.values()), {})
    if func_or_none is None:
        raise TypeError(
            f"Function {func_name} does not support the provided argument types {', '.join(type_repr(t) for t in arg_types.values())}"
        )
    func = func_or_none

    if func.value_type is not None:
        out_dtype = func.value_type
    elif func.value_func is not None:
        out_dtype = func.value_func(arg_types, arg_values)
    else:
        func.build(None)
        out_dtype = func.value_func(arg_types, arg_values)

    if out_dtype is None:
        raise TypeError("The provided function must return a value")

    if isinstance(out_dtype, tuple) or isinstance(out_dtype, list):
        out_dtypes = out_dtype
    else:
        out_dtypes = (out_dtype,)

    if out is None:
        requires_grad = any(getattr(a, "requires_grad", False) for a in inputs if is_array(a))
        outputs = []
        for dtype in out_dtypes:
            rg = requires_grad and Adjoint.is_differentiable_value_type(dtype)
            outputs.append(wp.empty(out_shape, dtype=dtype, requires_grad=rg, device=device))
    elif len(out_dtypes) == 1 and is_array(out):
        if not types_equal(out.dtype, out_dtypes[0]):
            raise TypeError(
                f"Output array dtype {type_repr(out.dtype)} does not match expected dtype {type_repr(out_dtypes[0])}"
            )
        if out.shape != out_shape:
            raise TypeError(f"Output array shape {out.shape} does not match expected shape {out_shape}")
        outputs = [out]
    elif len(out_dtypes) > 1:
        if isinstance(out, tuple) or isinstance(out, list):
            if len(out) != len(out_dtypes):
                raise TypeError(
                    f"Number of provided output arrays ({len(out)}) does not match expected number of function outputs ({len(out_dtypes)})"
                )
            for i, a in enumerate(out):
                if not types_equal(a.dtype, out_dtypes[i]):
                    raise TypeError(
                        f"Output array {i} dtype {type_repr(a.dtype)} does not match expected dtype {type_repr(out_dtypes[i])}"
                    )
                if a.shape != out_shape:
                    raise TypeError(f"Output array {i} shape {a.shape} does not match expected shape {out_shape}")
            outputs = list(out)
        else:
            raise TypeError(
                f"Invalid output provided, expected {len(out_dtypes)} Warp arrays with shape {out_shape} and dtypes ({', '.join(type_repr(t) for t in out_dtypes)})"
            )

    # create code for a kernel
    code = """def map_kernel({kernel_args}):
    {tids} = wp.tid()
    {load_args}
    """
    if len(outputs) == 1:
        code += "__out_0[{tids}] = {func_name}({arg_names})"
    else:
        code += ", ".join(f"__o_{i}" for i in range(len(outputs)))
        code += " = {func_name}({arg_names})\n"
        for i in range(len(outputs)):
            code += f"    __out_{i}" + "[{tids}]" + f" = __o_{i}\n"

    tids = [f"__tid_{i}" for i in range(len(out_shape))]

    load_args = []
    kernel_args = []
    for arg_name, input in zip(arg_names, inputs):
        if is_array(input):
            arr_name = f"{arg_name}_array"
            array_type_name = type(input).__name__
            kernel_args.append(
                f"{arr_name}: wp.{array_type_name}(dtype={type_to_code(input.dtype)}, ndim={input.ndim})"
            )
            shape = input.shape
            indices = []
            for i in range(1, len(shape) + 1):
                if shape[-i] == 1:
                    indices.append("0")
                else:
                    indices.append(tids[-i])

            load_args.append(f"{arg_name} = {arr_name}[{', '.join(reversed(indices))}]")
        else:
            kernel_args.append(f"{arg_name}: {type_to_code(type(input))}")
    for i, o in enumerate(outputs):
        array_type_name = type(o).__name__
        kernel_args.append(f"__out_{i}: wp.{array_type_name}(dtype={type_to_code(o.dtype)}, ndim={o.ndim})")
    code = code.format(
        func_name=func_name,
        kernel_args=", ".join(kernel_args),
        arg_names=", ".join(arg_names),
        tids=", ".join(tids),
        load_args="\n    ".join(load_args),
    )
    namespace = {}
    namespace.update({"wp": wp, "warp": wp, func_name: wp_func, "Any": Any})
    namespace.update(referenced_modules)
    exec(code, namespace)

    kernel = wp.Kernel(namespace["map_kernel"], key="map_kernel", source=code, module=module)
    if return_kernel:
        return kernel

    wp.launch(
        kernel,
        dim=out_shape,
        inputs=inputs,
        outputs=outputs,
        block_dim=block_dim,
        device=device,
    )

    if len(outputs) == 1:
        o = outputs[0]
    else:
        o = outputs

    return o


# code snippet for invoking cProfile
# cp = cProfile.Profile()
# cp.enable()
# for i in range(1000):
#     self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

# cp.disable()
# cp.print_stats(sort='tottime')
# exit(0)


# helper kernels for initializing NVDB volumes from a dense array
@wp.kernel
def copy_dense_volume_to_nano_vdb_v(volume: wp.uint64, values: wp.array(dtype=wp.vec3, ndim=3)):
    i, j, k = wp.tid()
    wp.volume_store_v(volume, i, j, k, values[i, j, k])


@wp.kernel
def copy_dense_volume_to_nano_vdb_f(volume: wp.uint64, values: wp.array(dtype=wp.float32, ndim=3)):
    i, j, k = wp.tid()
    wp.volume_store_f(volume, i, j, k, values[i, j, k])


@wp.kernel
def copy_dense_volume_to_nano_vdb_i(volume: wp.uint64, values: wp.array(dtype=wp.int32, ndim=3)):
    i, j, k = wp.tid()
    wp.volume_store_i(volume, i, j, k, values[i, j, k])


# represent an edge between v0, v1 with connected faces f0, f1, and opposite vertex o0, and o1
# winding is such that first tri can be reconstructed as {v0, v1, o0}, and second tri as { v1, v0, o1 }
class MeshEdge:
    def __init__(self, v0, v1, o0, o1, f0, f1):
        self.v0 = v0  # vertex 0
        self.v1 = v1  # vertex 1
        self.o0 = o0  # opposite vertex 1
        self.o1 = o1  # opposite vertex 2
        self.f0 = f0  # index of tri1
        self.f1 = f1  # index of tri2


class MeshAdjacency:
    def __init__(self, indices, num_tris):
        # map edges (v0, v1) to faces (f0, f1)
        self.edges = {}
        self.indices = indices

        for index, tri in enumerate(indices):
            self.add_edge(tri[0], tri[1], tri[2], index)
            self.add_edge(tri[1], tri[2], tri[0], index)
            self.add_edge(tri[2], tri[0], tri[1], index)

    def add_edge(self, i0, i1, o, f):  # index1, index2, index3, index of triangle
        key = (min(i0, i1), max(i0, i1))
        edge = None

        if key in self.edges:
            edge = self.edges[key]

            if edge.f1 != -1:
                print("Detected non-manifold edge")
                return
            else:
                # update other side of the edge
                edge.o1 = o
                edge.f1 = f
        else:
            # create new edge with opposite yet to be filled
            edge = MeshEdge(i0, i1, o, -1, f, -1)

        self.edges[key] = edge


def mem_report():  # pragma: no cover
    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation"""
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
        print(f"Type: {mem_type:<4} | Total Tensors: {total_numel:>8} | Used Memory: {total_mem:>8.2f} MB")

    import gc

    import torch

    gc.collect()

    LEN = 65
    objects = gc.get_objects()
    # print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, "GPU")
    _mem_report(host_tensors, "CPU")
    print("=" * LEN)


class ScopedDevice:
    """A context manager to temporarily change the current default device.

    For CUDA devices, this context manager makes the device's CUDA context
    current and restores the previous CUDA context on exit. This is handy when
    running Warp scripts as part of a bigger pipeline because it avoids any side
    effects of changing the CUDA context in the enclosed code.

    Attributes:
        device (Device): The device that will temporarily become the default
          device within the context.
        saved_device (Device): The previous default device. This is restored as
          the default device on exiting the context.
    """

    def __init__(self, device: Devicelike):
        """Initializes the context manager with a device.

        Args:
            device: The device that will temporarily become the default device
              within the context.
        """
        self.device = wp.get_device(device)

    def __enter__(self):
        # save the previous default device
        self.saved_device = self.device.runtime.default_device

        # make this the default device
        self.device.runtime.default_device = self.device

        # make it the current CUDA device so that device alias "cuda" will evaluate to this device
        self.device.context_guard.__enter__()

        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        # restore original CUDA context
        self.device.context_guard.__exit__(exc_type, exc_value, traceback)

        # restore original target device
        self.device.runtime.default_device = self.saved_device


class ScopedStream:
    """A context manager to temporarily change the current stream on a device.

    Attributes:
        stream (Stream or None): The stream that will temporarily become the device's
          default stream within the context.
        saved_stream (Stream): The device's previous current stream. This is
          restored as the device's current stream on exiting the context.
        sync_enter (bool): Whether to synchronize this context's stream with
          the device's previous current stream on entering the context.
        sync_exit (bool): Whether to synchronize the device's previous current
          with this context's stream on exiting the context.
        device (Device): The device associated with the stream.
    """

    def __init__(self, stream: wp.Stream | None, sync_enter: bool = True, sync_exit: bool = False):
        """Initializes the context manager with a stream and synchronization options.

        Args:
            stream: The stream that will temporarily become the device's
              default stream within the context.
            sync_enter (bool): Whether to synchronize this context's stream with
              the device's previous current stream on entering the context.
            sync_exit (bool): Whether to synchronize the device's previous current
              with this context's stream on exiting the context.
        """

        self.stream = stream
        self.sync_enter = sync_enter
        self.sync_exit = sync_exit
        if stream is not None:
            self.device = stream.device
            self.device_scope = ScopedDevice(self.device)

    def __enter__(self):
        if self.stream is not None:
            self.device_scope.__enter__()
            self.saved_stream = self.device.stream
            self.device.set_stream(self.stream, self.sync_enter)

        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self.device.set_stream(self.saved_stream, self.sync_exit)
            self.device_scope.__exit__(exc_type, exc_value, traceback)


TIMING_KERNEL = 1
TIMING_KERNEL_BUILTIN = 2
TIMING_MEMCPY = 4
TIMING_MEMSET = 8
TIMING_GRAPH = 16
TIMING_ALL = 0xFFFFFFFF


# timer utils
class ScopedTimer:
    indent = -1

    enabled = True

    def __init__(
        self,
        name: str,
        active: bool = True,
        print: bool = True,
        detailed: bool = False,
        dict: dict[str, list[float]] | None = None,
        use_nvtx: bool = False,
        color: int | str = "rapids",
        synchronize: bool = False,
        cuda_filter: int = 0,
        report_func: Callable[[list[TimingResult], str], None] | None = None,
        skip_tape: bool = False,
    ):
        """Context manager object for a timer

        Parameters:
            name: Name of timer
            active: Enables this timer
            print: At context manager exit, print elapsed time to ``sys.stdout``
            detailed: Collects additional profiling data using cProfile and calls ``print_stats()`` at context exit
            dict: A dictionary of lists to which the elapsed time will be appended using ``name`` as a key
            use_nvtx: If true, timing functionality is replaced by an NVTX range
            color: ARGB value (e.g. 0x00FFFF) or color name (e.g. 'cyan') associated with the NVTX range
            synchronize: Synchronize the CPU thread with any outstanding CUDA work to return accurate GPU timings
            cuda_filter: Filter flags for CUDA activity timing, e.g. ``warp.TIMING_KERNEL`` or ``warp.TIMING_ALL``
            report_func: A callback function to print the activity report.
              If ``None``,  :func:`wp.timing_print() <timing_print>` will be used.
            skip_tape: If true, the timer will not be recorded in the tape

        Attributes:
            extra_msg (str): Can be set to a string that will be added to the printout at context exit.
            elapsed (float): The duration of the ``with`` block used with this object
            timing_results (list[TimingResult]): The list of activity timing results, if collection was requested using ``cuda_filter``
        """
        self.name = name
        self.active = active and self.enabled
        self.print = print
        self.detailed = detailed
        self.dict = dict
        self.use_nvtx = use_nvtx
        self.color = color
        self.synchronize = synchronize
        self.skip_tape = skip_tape
        self.elapsed = 0.0
        self.cuda_filter = cuda_filter
        self.report_func = report_func or wp.timing_print
        self.extra_msg = ""  # Can be used to add to the message printed at manager exit

        if self.dict is not None:
            if name not in self.dict:
                self.dict[name] = []

    def __enter__(self):
        if not self.skip_tape and warp.context.runtime is not None and warp.context.runtime.tape is not None:
            warp.context.runtime.tape.record_scope_begin(self.name)
        if self.active:
            if self.synchronize:
                wp.synchronize()

            if self.cuda_filter:
                # begin CUDA activity collection, synchronizing if needed
                timing_begin(self.cuda_filter, synchronize=not self.synchronize)

            if self.detailed:
                self.cp = cProfile.Profile()
                self.cp.clear()
                self.cp.enable()

            if self.use_nvtx:
                import nvtx

                self.nvtx_range_id = nvtx.start_range(self.name, color=self.color)

            if self.print:
                ScopedTimer.indent += 1

                if warp.config.verbose:
                    indent = "    " * ScopedTimer.indent
                    print(f"{indent}{self.name} ...", flush=True)

            self.start = time.perf_counter_ns()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.skip_tape and warp.context.runtime is not None and warp.context.runtime.tape is not None:
            warp.context.runtime.tape.record_scope_end()
        if self.active:
            if self.synchronize:
                wp.synchronize()

            self.elapsed = (time.perf_counter_ns() - self.start) / 1000000.0

            if self.use_nvtx:
                import nvtx

                nvtx.end_range(self.nvtx_range_id)

            if self.detailed:
                self.cp.disable()
                self.cp.print_stats(sort="tottime")

            if self.cuda_filter:
                # end CUDA activity collection, synchronizing if needed
                self.timing_results = timing_end(synchronize=not self.synchronize)
            else:
                self.timing_results = []

            if self.dict is not None:
                self.dict[self.name].append(self.elapsed)

            if self.print:
                indent = "    " * ScopedTimer.indent

                if self.timing_results:
                    self.report_func(self.timing_results, indent=indent)
                    print()

                if self.extra_msg:
                    print(f"{indent}{self.name} took {self.elapsed:.2f} ms {self.extra_msg}")
                else:
                    print(f"{indent}{self.name} took {self.elapsed:.2f} ms")

                ScopedTimer.indent -= 1


# Allow temporarily enabling/disabling mempool allocators
class ScopedMempool:
    def __init__(self, device: Devicelike, enable: bool):
        self.device = wp.get_device(device)
        self.enable = enable

    def __enter__(self):
        self.saved_setting = wp.is_mempool_enabled(self.device)
        wp.set_mempool_enabled(self.device, self.enable)

    def __exit__(self, exc_type, exc_value, traceback):
        wp.set_mempool_enabled(self.device, self.saved_setting)


# Allow temporarily enabling/disabling mempool access
class ScopedMempoolAccess:
    def __init__(self, target_device: Devicelike, peer_device: Devicelike, enable: bool):
        self.target_device = target_device
        self.peer_device = peer_device
        self.enable = enable

    def __enter__(self):
        self.saved_setting = wp.is_mempool_access_enabled(self.target_device, self.peer_device)
        wp.set_mempool_access_enabled(self.target_device, self.peer_device, self.enable)

    def __exit__(self, exc_type, exc_value, traceback):
        wp.set_mempool_access_enabled(self.target_device, self.peer_device, self.saved_setting)


# Allow temporarily enabling/disabling peer access
class ScopedPeerAccess:
    def __init__(self, target_device: Devicelike, peer_device: Devicelike, enable: bool):
        self.target_device = target_device
        self.peer_device = peer_device
        self.enable = enable

    def __enter__(self):
        self.saved_setting = wp.is_peer_access_enabled(self.target_device, self.peer_device)
        wp.set_peer_access_enabled(self.target_device, self.peer_device, self.enable)

    def __exit__(self, exc_type, exc_value, traceback):
        wp.set_peer_access_enabled(self.target_device, self.peer_device, self.saved_setting)


class ScopedCapture:
    def __init__(self, device: Devicelike = None, stream=None, force_module_load=None, external=False):
        self.device = device
        self.stream = stream
        self.force_module_load = force_module_load
        self.external = external
        self.active = False
        self.graph = None

    def __enter__(self):
        try:
            wp.capture_begin(
                device=self.device, stream=self.stream, force_module_load=self.force_module_load, external=self.external
            )
            self.active = True
            return self
        except:
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            try:
                self.graph = wp.capture_end(device=self.device, stream=self.stream)
            finally:
                self.active = False


def check_p2p():
    """Check if the machine is configured properly for peer-to-peer transfers.

    Returns:
        A Boolean indicating whether the machine is configured properly for peer-to-peer transfers.
        On Linux, this function attempts to determine if IOMMU is enabled and will return `False` if IOMMU is detected.
        On other operating systems, it always return `True`.
    """

    # HACK: allow disabling P2P tests using an environment variable
    disable_p2p_tests = os.getenv("WARP_DISABLE_P2P_TESTS", default="0")
    if int(disable_p2p_tests):
        return False

    if sys.platform == "linux":
        # IOMMU enablement can affect peer-to-peer transfers.
        # On modern Linux, there should be IOMMU-related entries in the /sys file system.
        # This should be more reliable than checking kernel logs like dmesg.
        if os.path.isdir("/sys/class/iommu") and os.listdir("/sys/class/iommu"):
            return False
        if os.path.isdir("/sys/kernel/iommu_groups") and os.listdir("/sys/kernel/iommu_groups"):
            return False

    return True


class timing_result_t(ctypes.Structure):
    """CUDA timing struct for fetching values from C++"""

    _fields_ = (
        ("context", ctypes.c_void_p),
        ("name", ctypes.c_char_p),
        ("filter", ctypes.c_int),
        ("elapsed", ctypes.c_float),
    )


class TimingResult:
    """Timing result for a single activity."""

    def __init__(self, device, name, filter, elapsed):
        self.device: warp.context.Device = device
        """The device where the activity was recorded."""

        self.name: str = name
        """The activity name."""

        self.filter: int = filter
        """The type of activity (e.g., ``warp.TIMING_KERNEL``)."""

        self.elapsed: float = elapsed
        """The elapsed time in milliseconds."""


def timing_begin(cuda_filter: int = TIMING_ALL, synchronize: bool = True) -> None:
    """Begin detailed activity timing.

    Parameters:
        cuda_filter: Filter flags for CUDA activity timing, e.g. ``warp.TIMING_KERNEL`` or ``warp.TIMING_ALL``
        synchronize: Whether to synchronize all CUDA devices before timing starts
    """

    if synchronize:
        warp.synchronize()

    warp.context.runtime.core.wp_cuda_timing_begin(cuda_filter)


def timing_end(synchronize: bool = True) -> list[TimingResult]:
    """End detailed activity timing.

    Parameters:
        synchronize: Whether to synchronize all CUDA devices before timing ends

    Returns:
        A list of :class:`TimingResult` objects for all recorded activities.
    """

    if synchronize:
        warp.synchronize()

    # get result count
    count = warp.context.runtime.core.wp_cuda_timing_get_result_count()

    # get result array from C++
    result_buffer = (timing_result_t * count)()
    warp.context.runtime.core.wp_cuda_timing_end(ctypes.byref(result_buffer), count)

    # prepare Python result list
    results = []
    for r in result_buffer:
        device = warp.context.runtime.context_map.get(r.context)
        filter = r.filter
        elapsed = r.elapsed

        name = r.name.decode()
        if filter == TIMING_KERNEL:
            if name.endswith("forward"):
                # strip trailing "_cuda_kernel_forward"
                name = f"forward kernel {name[:-20]}"
            else:
                # strip trailing "_cuda_kernel_backward"
                name = f"backward kernel {name[:-21]}"
        elif filter == TIMING_KERNEL_BUILTIN:
            if name.startswith("wp::"):
                name = f"builtin kernel {name[4:]}"
            else:
                name = f"builtin kernel {name}"

        results.append(TimingResult(device, name, filter, elapsed))

    return results


def timing_print(results: list[TimingResult], indent: str = "") -> None:
    """Print timing results.

    Parameters:
        results: List of :class:`TimingResult` objects to print.
        indent: Optional indentation to prepend to all output lines.
    """

    if not results:
        print("No activity")
        return

    class Aggregate:
        def __init__(self, count=0, elapsed=0):
            self.count = count
            self.elapsed = elapsed

    device_totals = {}
    activity_totals = {}

    max_name_len = len("Activity")
    for r in results:
        name_len = len(r.name)
        max_name_len = max(max_name_len, name_len)

    activity_width = max_name_len + 1
    activity_dashes = "-" * activity_width

    print(f"{indent}CUDA timeline:")
    print(f"{indent}----------------+---------+{activity_dashes}")
    print(f"{indent}Time            | Device  | Activity")
    print(f"{indent}----------------+---------+{activity_dashes}")
    for r in results:
        device_agg = device_totals.get(r.device.alias)
        if device_agg is None:
            device_totals[r.device.alias] = Aggregate(count=1, elapsed=r.elapsed)
        else:
            device_agg.count += 1
            device_agg.elapsed += r.elapsed

        activity_agg = activity_totals.get(r.name)
        if activity_agg is None:
            activity_totals[r.name] = Aggregate(count=1, elapsed=r.elapsed)
        else:
            activity_agg.count += 1
            activity_agg.elapsed += r.elapsed

        print(f"{indent}{r.elapsed:12.6f} ms | {r.device.alias:7s} | {r.name}")

    print()
    print(f"{indent}CUDA activity summary:")
    print(f"{indent}----------------+---------+{activity_dashes}")
    print(f"{indent}Total time      | Count   | Activity")
    print(f"{indent}----------------+---------+{activity_dashes}")
    for name, agg in activity_totals.items():
        print(f"{indent}{agg.elapsed:12.6f} ms | {agg.count:7d} | {name}")

    print()
    print(f"{indent}CUDA device summary:")
    print(f"{indent}----------------+---------+{activity_dashes}")
    print(f"{indent}Total time      | Count   | Device")
    print(f"{indent}----------------+---------+{activity_dashes}")
    for device, agg in device_totals.items():
        print(f"{indent}{agg.elapsed:12.6f} ms | {agg.count:7d} | {device}")
