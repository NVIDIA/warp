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

"""Helpers to author OmniGraph attributes."""

import ctypes
import functools
import inspect
import math
import operator
from typing import (
    Any,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import omni.graph.core as og

import warp as wp

from .common import type_convert_og_to_warp

ATTR_BUNDLE_TYPE = og.Type(
    og.BaseDataType.RELATIONSHIP,
    1,
    0,
    og.AttributeRole.BUNDLE,
)


#   Names
# ------------------------------------------------------------------------------


_ATTR_PORT_TYPES = (
    og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT,
    og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
    og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE,
)

_ATTR_NAME_FMTS = {x: f"{og.get_port_type_namespace(x)}:{{}}" for x in _ATTR_PORT_TYPES}


def attr_join_name(
    port_type: og.AttributePortType,
    base_name: str,
) -> str:
    """Build an attribute name by prefixing it with its port type."""
    return _ATTR_NAME_FMTS[port_type].format(base_name)


def attr_get_base_name(
    attr: og.Attribute,
) -> str:
    """Retrieves an attribute base name."""
    name = attr.get_name()
    if (
        attr.get_type_name() == "bundle"
        and (attr.get_port_type() == og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT)
        and name.startswith("outputs_")
    ):
        # Output bundles are a bit special because they are in fact implemented
        # as USD primitives, and USD doesn't support the colon symbol `:` in
        # primitive names, thus output bundles are prefixed with `outputs_` in
        # OmniGraph instead of `outputs:` like everything else.
        return name[8:]

    return name.split(":")[-1]


def attr_get_name(
    attr: og.Attribute,
) -> str:
    """Retrieves an attribute name."""
    name = attr.get_name()
    if (
        attr.get_type_name() == "bundle"
        and (attr.get_port_type() == og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT)
        and name.startswith("outputs_")
    ):
        # Output bundles are a bit special because they are in fact implemented
        # as USD primitives, and USD doesn't support the colon symbol `:` in
        # primitive names, thus output bundles are prefixed with `outputs_` in
        # OmniGraph instead of `outputs:` like everything else.
        return attr_join_name(
            og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
            name[8:],
        )

    return name


#   Values
# ------------------------------------------------------------------------------


def attr_get(
    attr: og.AttributeData,
) -> Any:
    """Retrieves the value from an attribute living on the CPU."""
    return attr.get(on_gpu=False)


def attr_set(
    attr: og.AttributeData,
    value: Any,
) -> None:
    """Sets the given value onto an array attribute living on the CPU."""
    attr.set(value, on_gpu=False)


def attr_get_array_on_gpu(
    attr: og.AttributeData,
    dtype: type,
    read_only: bool = True,
) -> wp.array:
    """Retrieves the value of an array attribute living on the GPU."""
    attr.gpu_ptr_kind = og.PtrToPtrKind.CPU
    (ptr, _) = attr.get_array(
        on_gpu=True,
        get_for_write=not read_only,
        reserved_element_count=0 if read_only else attr.size(),
    )
    return from_omni_graph_ptr(ptr, (attr.size(),), dtype=dtype)


def attr_cast_array_to_warp(
    value: Union[np.array, og.DataWrapper],
    dtype: type,
    shape: Sequence[int],
    device: wp.context.Device,
) -> wp.array:
    """Casts an attribute array value to its corresponding warp type."""
    if device.is_cpu:
        return wp.array(
            value,
            dtype=dtype,
            shape=shape,
            device=device,
        )

    elif device.is_cuda:
        return from_omni_graph_ptr(
            value.memory,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    raise AssertionError(f"Unexpected device '{device.alias}'.")


#   Tracking
# ------------------------------------------------------------------------------


class AttrTracking:
    """Attributes state for tracking changes."""

    def __init__(self, names: Sequence[str]) -> None:
        self._names = names
        self._state = [None] * len(names)

    def have_attrs_changed(self, db: og.Database) -> bool:
        """Compare the current attribute values with the internal state."""
        for i, name in enumerate(self._names):
            cached_value = self._state[i]
            current_value = getattr(db.inputs, name)
            if isinstance(current_value, np.ndarray):
                if not np.array_equal(current_value, cached_value):
                    return True
            elif current_value != cached_value:
                return True

        return False

    def update_state(self, db: og.Database) -> None:
        """Updates the internal state with the current attribute values."""
        for i, name in enumerate(self._names):
            current_value = getattr(db.inputs, name)
            if isinstance(current_value, np.ndarray):
                self._state[i] = current_value.copy()
            else:
                self._state[i] = current_value


#   High-level Helper
# ------------------------------------------------------------------------------


def from_omni_graph_ptr(ptr, shape, dtype=None, device=None):
    return wp.array(
        dtype=dtype,
        ptr=0 if ptr == 0 else ctypes.cast(ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value,
        shape=shape,
        device=device,
        requires_grad=False,
    )


def from_omni_graph(
    value: Union[np.ndarray, og.DataWrapper, og.AttributeData, og.DynamicAttributeAccess],
    dtype: Optional[type] = None,
    shape: Optional[Sequence[int]] = None,
    device: Optional[wp.context.Device] = None,
) -> wp.array:
    """Casts an OmniGraph array data to its corresponding Warp type."""

    def from_data_wrapper(
        data: og.DataWrapper,
        dtype: Optional[type],
        shape: Optional[Sequence[int]],
        device: Optional[wp.context.Device],
    ) -> wp.array:
        if data.gpu_ptr_kind != og.PtrToPtrKind.CPU:
            raise RuntimeError("All pointers must live on the CPU, make sure to set 'cudaPointers' to 'cpu'.")
        elif not data.is_array:
            raise RuntimeError("The attribute data isn't an array.")

        if dtype is None:
            base_type = type_convert_og_to_warp(
                og.Type(
                    data.dtype.base_type,
                    tuple_count=data.dtype.tuple_count,
                    array_depth=0,
                    role=og.AttributeRole.MATRIX if data.dtype.is_matrix_type() else og.AttributeRole.NONE,
                ),
            )

            dim_count = len(data.shape)
            if dim_count == 1:
                dtype = base_type
            elif dim_count == 2:
                dtype = wp.types.vector(length=data.shape[1], dtype=base_type)
            elif dim_count == 3:
                dtype = wp.types.matrix(shape=(data.shape[1], data.shape[2]), dtype=base_type)
            else:
                raise RuntimeError("Arrays with more than 3 dimensions are not supported.")

        arr_size = data.shape[0] * data.dtype.size
        element_size = wp.types.type_size_in_bytes(dtype)

        if shape is None:
            # Infer a shape compatible with the dtype.
            for i in range(len(data.shape)):
                if functools.reduce(operator.mul, data.shape[: i + 1]) * element_size == arr_size:
                    shape = data.shape[: i + 1]
                    break

        if shape is None:
            if arr_size % element_size != 0:
                raise RuntimeError(
                    f"Cannot infer a size matching the Warp data type '{dtype.__name__}' with an array size of '{arr_size}' bytes."
                )
            shape = (arr_size // element_size,)

        src_device = wp.get_device(str(data.device))
        dst_device = device
        return from_omni_graph_ptr(
            data.memory,
            shape=shape,
            dtype=dtype,
            device=src_device,
        ).to(dst_device)

    def from_attr_data(
        data: og.AttributeData,
        dtype: Optional[type],
        shape: Optional[Sequence[int]],
        device: Optional[wp.context.Device],
    ) -> wp.array:
        if data.gpu_valid():
            on_gpu = True
        elif data.cpu_valid():
            on_gpu = False
        else:
            raise RuntimeError("The attribute data isn't valid.")

        if on_gpu:
            data_type = data.get_type()
            base_type = type_convert_og_to_warp(
                og.Type(
                    data_type.base_type,
                    tuple_count=data_type.tuple_count,
                    array_depth=0,
                    role=data_type.role,
                ),
            )

            if dtype is None:
                dtype = base_type

            arr_size = data.size() * wp.types.type_size_in_bytes(base_type)
            element_size = wp.types.type_size_in_bytes(dtype)

            if shape is None:
                # Infer a shape compatible with the dtype.
                if data_type.is_matrix_type():
                    dim = math.isqrt(data_type.tuple_count)
                    arr_shape = (data.size(), dim, dim)
                else:
                    arr_shape = (data.size(), data_type.tuple_count)

                for i in range(len(arr_shape)):
                    if functools.reduce(operator.mul, arr_shape[: i + 1]) * element_size == arr_size:
                        shape = arr_shape[: i + 1]
                        break

            if shape is None:
                if arr_size % element_size != 0:
                    raise RuntimeError(
                        f"Cannot infer a size matching the Warp data type '{dtype.__name__}' with an array size of '{arr_size}' bytes."
                    )
                shape = (arr_size // element_size,)

            data.gpu_ptr_kind = og.PtrToPtrKind.CPU
            (ptr, _) = data.get_array(
                on_gpu=True,
                get_for_write=not data.is_read_only(),
                reserved_element_count=0 if data.is_read_only() else data.size(),
            )

            src_device = wp.get_device("cuda")
            dst_device = device
            return from_omni_graph_ptr(
                ptr,
                shape=shape,
                dtype=dtype,
                device=src_device,
            ).to(dst_device)
        else:
            arr = data.get_array(
                on_gpu=False,
                get_for_write=not data.is_read_only(),
                reserved_element_count=0 if data.is_read_only() else data.size(),
            )
            return wp.from_numpy(arr, dtype=dtype, shape=shape, device=device)

    if isinstance(value, np.ndarray):
        return wp.from_numpy(value, dtype=dtype, shape=shape, device=device)
    elif isinstance(value, og.DataWrapper):
        return from_data_wrapper(value, dtype, shape, device)
    elif isinstance(value, og.AttributeData):
        return from_attr_data(value, dtype, shape, device)
    elif og.DynamicAttributeAccess in inspect.getmro(type(getattr(value, "_parent", None))):
        if device is None:
            device = wp.get_device()

        if device.is_cpu:
            return wp.from_numpy(value.cpu, dtype=dtype, shape=shape, device=device)
        elif device.is_cuda:
            return from_data_wrapper(value.gpu, dtype, shape, device)
        else:
            raise AssertionError(f"Unexpected device '{device.alias}'.")

    return None
