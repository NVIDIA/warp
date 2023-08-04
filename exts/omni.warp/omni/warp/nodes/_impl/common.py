# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""General helpers for this extension."""

from enum import Enum
from typing import (
    Any,
    Optional,
    Union,
)

import omni.graph.core as og
import warp as wp


#   General
# ------------------------------------------------------------------------------


class IntEnum(int, Enum):
    """Base class for integer enumerators with labels."""

    def __new__(cls, value, label):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.label = label
        return obj


#   Timer
# ------------------------------------------------------------------------------


class NodeTimer(object):
    """Context wrapping Warp's scoped timer for use with nodes."""

    def __init__(self, name: str, db: Any, active: bool = False) -> None:
        name = "{}:{}".format(db.node.get_prim_path(), name)
        self.timer = wp.ScopedTimer(name, active=active, synchronize=True)

    def __enter__(self) -> None:
        self.timer.__enter__()
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.timer.__exit__(type, value, traceback)


#   Types
# ------------------------------------------------------------------------------

_BaseDType = og.BaseDataType
_AttrRole = og.AttributeRole

# fmt: off

_DATA_TYPES_MAPPING = (
    ("bool"      , (_BaseDType.BOOL  ,  1, _AttrRole.NONE      ), "int8"   ),
    ("color3f"   , (_BaseDType.FLOAT ,  3, _AttrRole.COLOR     ), "vec3"   ),
    ("color4f"   , (_BaseDType.FLOAT ,  4, _AttrRole.COLOR     ), "vec4"   ),
    ("double"    , (_BaseDType.DOUBLE,  1, _AttrRole.NONE      ), "float64"),
    ("float"     , (_BaseDType.FLOAT ,  1, _AttrRole.NONE      ), "float32"),
    ("float2"    , (_BaseDType.FLOAT ,  2, _AttrRole.NONE      ), "vec2"   ),
    ("float3"    , (_BaseDType.FLOAT ,  3, _AttrRole.NONE      ), "vec3"   ),
    ("float4"    , (_BaseDType.FLOAT ,  4, _AttrRole.NONE      ), "vec4"   ),
    ("int"       , (_BaseDType.INT   ,  1, _AttrRole.NONE      ), "int32"  ),
    ("int64"     , (_BaseDType.INT64 ,  1, _AttrRole.NONE      ), "int64"  ),
    ("matrix2d"  , (_BaseDType.DOUBLE,  4, _AttrRole.MATRIX    ), "mat22"  ),
    ("matrix3d"  , (_BaseDType.DOUBLE,  9, _AttrRole.MATRIX    ), "mat33"  ),
    ("matrix4d"  , (_BaseDType.DOUBLE, 16, _AttrRole.MATRIX    ), "mat44"  ),
    ("normal3f"  , (_BaseDType.FLOAT ,  3, _AttrRole.NORMAL    ), "vec3"   ),
    ("point3f"   , (_BaseDType.FLOAT ,  3, _AttrRole.POSITION  ), "vec3"   ),
    ("quatf"     , (_BaseDType.FLOAT ,  4, _AttrRole.QUATERNION), "quat"   ),
    ("texCoord2f", (_BaseDType.FLOAT ,  2, _AttrRole.TEXCOORD  ), "vec2"   ),
    ("texCoord3f", (_BaseDType.FLOAT ,  3, _AttrRole.TEXCOORD  ), "vec3"   ),
    ("timecode"  , (_BaseDType.DOUBLE,  1, _AttrRole.TIMECODE  ), "float32"),
    ("token"     , (_BaseDType.TOKEN ,  1, _AttrRole.NONE      ), "uint64" ),
    ("uchar"     , (_BaseDType.UCHAR ,  1, _AttrRole.NONE      ), "uint8"  ),
    ("uint"      , (_BaseDType.UINT  ,  1, _AttrRole.NONE      ), "uint32" ),
    ("uint64"    , (_BaseDType.UINT64,  1, _AttrRole.NONE      ), "uint64" ),
    ("vector3f"  , (_BaseDType.FLOAT ,  3, _AttrRole.VECTOR    ), "vec3"   ),
)

_SDF_DATA_TYPE_TO_OG        = {k: v for (k, v, _) in _DATA_TYPES_MAPPING}
_SDF_DATA_TYPE_NAME_TO_WARP = {k: v for (k, _, v) in _DATA_TYPES_MAPPING}
_OG_DATA_TYPE_TO_WARP       = {k: v for (_, k, v) in _DATA_TYPES_MAPPING}

# fmt: on

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


def get_warp_type_from_data_type_name(
    data_type_name: str,
    dim_count: int = 0,
    as_str: bool = False,
    str_namespace: Optional[str] = "wp",
):
    if as_str:
        prefix = "" if str_namespace is None else "{}.".format(str_namespace)

        if dim_count == 0:
            return "{prefix}{dtype}".format(prefix=prefix, dtype=data_type_name)

        if dim_count == 1:
            return "{prefix}array(dtype={prefix}{dtype})".format(
                prefix=prefix,
                dtype=data_type_name,
            )

        return "{prefix}array(dtype={prefix}{dtype}, ndim={ndim})".format(
            prefix=prefix,
            dtype=data_type_name,
            ndim=dim_count,
        )

    dtype = getattr(wp.types, data_type_name)
    if dim_count == 0:
        return dtype

    if dim_count == 1:
        return wp.array(dtype=dtype)

    return wp.array(dtype=dtype, ndim=dim_count)


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


def convert_sdf_type_name_to_og(
    sdf_type_name: str,
    is_array: bool = False,
) -> og.Type:
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
