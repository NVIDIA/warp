# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import operator
from typing import (
    Any,
    NamedTuple,
    Optional,
    Union,
    Sequence,
)

import numpy as np
import omni.graph.core as og
import warp as wp

from omni.warp.scripts.common import (
    IntEnum,
    get_warp_type_from_data_type_name,
)

_WARP_TYPE_NAMESPACE = "wp"

#   Types
# ------------------------------------------------------------------------------

_BaseDType = og.BaseDataType
_AttrRole = og.AttributeRole

_DATA_TYPES_MAPPING = (
    ("bool", (_BaseDType.BOOL, 1, _AttrRole.NONE), "int8"),
    ("color3f", (_BaseDType.FLOAT, 3, _AttrRole.COLOR), "vec3"),
    ("color4f", (_BaseDType.FLOAT, 4, _AttrRole.COLOR), "vec4"),
    ("double", (_BaseDType.DOUBLE, 1, _AttrRole.NONE), "float64"),
    ("float", (_BaseDType.FLOAT, 1, _AttrRole.NONE), "float32"),
    ("float2", (_BaseDType.FLOAT, 2, _AttrRole.NONE), "vec2"),
    ("float3", (_BaseDType.FLOAT, 3, _AttrRole.NONE), "vec3"),
    ("float4", (_BaseDType.FLOAT, 4, _AttrRole.NONE), "vec4"),
    ("int", (_BaseDType.INT, 1, _AttrRole.NONE), "int32"),
    ("int64", (_BaseDType.INT64, 1, _AttrRole.NONE), "int64"),
    ("matrix2d", (_BaseDType.DOUBLE, 4, _AttrRole.MATRIX), "mat22"),
    ("matrix3d", (_BaseDType.DOUBLE, 9, _AttrRole.MATRIX), "mat33"),
    ("matrix4d", (_BaseDType.DOUBLE, 16, _AttrRole.MATRIX), "mat44"),
    ("normal3f", (_BaseDType.FLOAT, 3, _AttrRole.NORMAL), "vec3"),
    ("point3f", (_BaseDType.FLOAT, 3, _AttrRole.POSITION), "vec3"),
    ("quatf", (_BaseDType.FLOAT, 4, _AttrRole.QUATERNION), "quat"),
    ("texCoord2f", (_BaseDType.FLOAT, 2, _AttrRole.TEXCOORD), "vec2"),
    ("texCoord3f", (_BaseDType.FLOAT, 3, _AttrRole.TEXCOORD), "vec3"),
    ("timecode", (_BaseDType.DOUBLE, 1, _AttrRole.TIMECODE), "float32"),
    ("token", (_BaseDType.TOKEN, 1, _AttrRole.NONE), "uint64"),
    ("uchar", (_BaseDType.UCHAR, 1, _AttrRole.NONE), "uint8"),
    ("uint", (_BaseDType.UINT, 1, _AttrRole.NONE), "uint32"),
    ("uint64", (_BaseDType.UINT64, 1, _AttrRole.NONE), "uint64"),
    ("vector3f", (_BaseDType.FLOAT, 3, _AttrRole.VECTOR), "vec3"),
)

_SDF_DATA_TYPE_TO_OG = {k: v for (k, v, _) in _DATA_TYPES_MAPPING}
_SDF_DATA_TYPE_NAME_TO_WARP = {k: v for (k, _, v) in _DATA_TYPES_MAPPING}
_OG_DATA_TYPE_TO_WARP = {k: v for (_, k, v) in _DATA_TYPES_MAPPING}

SUPPORTED_OG_DATA_TYPES = tuple(
    og.Type(base_data_type, tuple_count=tuple_count, array_depth=0, role=role)
    for base_data_type, tuple_count, role in _OG_DATA_TYPE_TO_WARP.keys()
)
SUPPORTED_OG_ARRAY_TYPES = tuple(
    og.Type(base_data_type, tuple_count=tuple_count, array_depth=1, role=role)
    for base_data_type, tuple_count, role in _OG_DATA_TYPE_TO_WARP.keys()
)
SUPPORTED_OG_TYPES = SUPPORTED_OG_DATA_TYPES + SUPPORTED_OG_ARRAY_TYPES

SUPPORTED_SDF_DATA_TYPE_NAMES = tuple(_SDF_DATA_TYPE_NAME_TO_WARP.keys())

BUNDLE_ATTR_TYPE = og.Type(
    og.BaseDataType.RELATIONSHIP,
    1,
    0,
    og.AttributeRole.BUNDLE,
)


def convert_og_type_to_warp(
    og_type: og.Type,
    dim_count: int = 0,
    as_str: bool = False,
    str_namespace: Optional[str] = "wp",
) -> Union[Any, str]:
    """Converts an OmniGraph type into a compatible Warp type."""
    data_type_name = _OG_DATA_TYPE_TO_WARP.get(
        (og_type.base_type, og_type.tuple_count, og_type.role),
    )
    if data_type_name is None:
        raise RuntimeError("Unsupported attribute type '{}'.".format(og_type))

    return get_warp_type_from_data_type_name(
        data_type_name,
        dim_count=dim_count,
        as_str=as_str,
        str_namespace=str_namespace,
    )


def convert_sdf_type_name_to_warp(
    sdf_type_name: str,
    dim_count: int = 0,
    as_str: bool = False,
    str_namespace: Optional[str] = "wp",
) -> Union[Any, str]:
    """Converts a Sdf type name into a compatible Warp type."""
    data_type_name = _SDF_DATA_TYPE_NAME_TO_WARP.get(sdf_type_name)
    if data_type_name is None:
        raise RuntimeError("Unsupported attribute type '{}'.".format(sdf_type_name))

    return get_warp_type_from_data_type_name(
        data_type_name,
        dim_count=dim_count,
        as_str=as_str,
        str_namespace=str_namespace,
    )


def convert_sdf_type_name_to_og(sdf_type_name: str, is_array: bool = False) -> og.Type:
    """Converts a Sdf type name into its corresponding OmniGraph type."""
    data_type = _SDF_DATA_TYPE_TO_OG.get(sdf_type_name)
    if data_type is None:
        raise RuntimeError("Unsupported attribute type '{}'.".format(sdf_type_name))

    base_data_type, tuple_count, role = data_type
    return og.Type(
        base_data_type,
        tuple_count=tuple_count,
        array_depth=int(is_array),
        role=role,
    )


#   Names
# ------------------------------------------------------------------------------

_ATTR_PORT_TYPES = (
    og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT,
    og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
    og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE,
)

_ATTR_NAME_FMTS = {x: "{}:{{}}".format(og.get_port_type_namespace(x)) for x in _ATTR_PORT_TYPES}


def join_attr_name(port_type: og.AttributePortType, base_name: str) -> str:
    """Build an attribute name by prefixing it with its port type."""
    return _ATTR_NAME_FMTS[port_type].format(base_name)


def get_attr_base_name(attr: og.Attribute) -> str:
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


def get_attr_name(attr: og.Attribute) -> str:
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
        return join_attr_name(
            og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
            name[8:],
        )

    return name


#   Values
# ------------------------------------------------------------------------------


def cast_array_attr_value_to_warp(
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


#   Bundles
# ------------------------------------------------------------------------------


def insert_bundle_attr(
    bundle: og.BundleContents,
    name: str,
    type: og.Type,
) -> og.RuntimeAttribute:
    attr = bundle.attribute_by_name(name)
    if attr is None:
        attr = bundle.insert((type, name))

    return attr


#   User Attributes Information
# ------------------------------------------------------------------------------


class OutputArrayShapeSource(IntEnum):
    """Method to infer the shape of output attribute arrays."""

    AS_INPUT_OR_AS_KERNEL = (0, "as input if any, or as kernel")
    AS_KERNEL = (1, "as kernel")


class OutputBundleTypeSource(IntEnum):
    """Method to infer the type of output attribute bundles."""

    AS_INPUT = (0, "as input if any")
    AS_INPUT_OR_EXPLICIT = (1, "as input if any, or explicit")
    EXPLICIT = (2, "explicit")


class OutputAttributeInfo(NamedTuple):
    """Information relating to an output node attribute."""

    array_shape_source: Optional[OutputArrayShapeSource]
    bundle_type_source: Optional[OutputBundleTypeSource]
    bundle_type_explicit: Optional[str] = None


class AttributeInfo(NamedTuple):
    """Information relating to a node attribute.

    This struct contains all the metadata required by the node to initialize
    and evaluate. This includes compiling the kernel and initializing the Inputs
    and Outputs structs that are then passed to the kernel as parameters.

    We don't directly store the array shape, if any, since it is possible that
    it might vary between each evaluation of the node's compute. Instead,
    we store which method to use to infer the array's shape and let the node
    determine the actual shape during each compute step.

    Note
    ----

    The `warp_type` member represents the type of the kernel parameter
    corresdonding to that attribute. If the attribute is a bundle, then it is
    expected to be a `wp.struct` holding the values of the bundle, unless
    the bundle is of type :class:`Array`, in which case `warp_type` should be
    a standard `wp.array`.
    """

    port_type: og.AttributePortType
    base_name: str
    og_type: og.Type
    warp_type: type
    output: Optional[OutputAttributeInfo] = None

    @property
    def name(self) -> str:
        return join_attr_name(self.port_type, self.base_name)

    @property
    def og_data_type(self) -> og.Type:
        return og.Type(
            self.og_type.base_type,
            tuple_count=self.og_type.tuple_count,
            array_depth=0,
            role=self.og_type.role,
        )

    @property
    def is_array(self) -> bool:
        return self.og_type.array_depth > 0

    @property
    def is_bundle(self) -> bool:
        return self.og_type == BUNDLE_ATTR_TYPE

    @property
    def dim_count(self) -> int:
        if self.is_array:
            return self.warp_type.ndim

        return 0

    @property
    def warp_data_type(self) -> type:
        if self.is_array:
            return self.warp_type.dtype

        return self.warp_type

    @property
    def warp_type_name(self) -> str:
        if self.is_bundle:
            return self.warp_type.cls.__name__

        return get_warp_type_from_data_type_name(
            self.warp_data_type.__name__,
            dim_count=self.dim_count,
            as_str=True,
            str_namespace=_WARP_TYPE_NAMESPACE,
        )

    @property
    def warp_data_type_name(self) -> str:
        if self.is_bundle:
            return self.warp_type.cls.__name__

        return get_warp_type_from_data_type_name(
            self.warp_data_type.__name__,
            dim_count=0,
            as_str=True,
            str_namespace=_WARP_TYPE_NAMESPACE,
        )
