# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Helpers to author OmniGraph attributes."""

import functools
import operator
from typing import (
    Any,
    Optional,
    Union,
    Sequence,
)

import numpy as np
import omni.graph.core as og
import warp as wp


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

_ATTR_NAME_FMTS = {x: "{}:{{}}".format(og.get_port_type_namespace(x)) for x in _ATTR_PORT_TYPES}


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
    return wp.from_ptr(ptr, attr.size(), dtype=dtype)


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
            owner=False,
            device=device,
        )

    elif device.is_cuda:
        size = functools.reduce(operator.mul, shape)
        return wp.types.from_ptr(
            value.memory,
            size,
            dtype=dtype,
            shape=shape,
            device=device,
        )

    assert False, "Unexpected device '{}'.".format(device.alias)


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


def from_omni_graph(
    value: Union[np.array, og.DataWrapper],
    dtype: Optional[type] = None,
    shape: Optional[Sequence[int]] = None,
    device: Optional[wp.context.Device] = None,
) -> wp.array:
    """Converts an OmniGraph array value to its corresponding Warp type."""
    if dtype is None:
        dtype = float

    if shape is None:
        # The array value might define 2 dimensions when tuples such as
        # wp.vec3 are used as data type, so we preserve only the first
        # dimension to retrieve the actual shape since OmniGraph only
        # supports 1D arrays anyways.
        shape = value.shape[:1]

    if device is None:
        device = wp.get_device()

    return attr_cast_array_to_warp(value, dtype, shape, device)
