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


class NodeTimer:
    """Context wrapping Warp's scoped timer for use with nodes."""

    def __init__(self, name: str, db: Any, active: bool = False) -> None:
        name = f"{db.node.get_prim_path()}:{name}"
        self.timer = wp.ScopedTimer(name, active=active, synchronize=True)

    def __enter__(self) -> None:
        self.timer.__enter__()
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.timer.__exit__(type, value, traceback)


#   Device
# ------------------------------------------------------------------------------


def device_get_cuda_compute() -> wp.context.Device:
    """Retrieves the preferred CUDA device for computing purposes."""
    query_fn = getattr(og, "get_compute_cuda_device", None)
    cuda_device_idx = 0 if query_fn is None else query_fn()
    return wp.get_device(f"cuda:{cuda_device_idx}")


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
    ("matrix2d"  , (_BaseDType.DOUBLE,  4, _AttrRole.MATRIX    ), "mat22d" ),
    ("matrix3d"  , (_BaseDType.DOUBLE,  9, _AttrRole.MATRIX    ), "mat33d" ),
    ("matrix4d"  , (_BaseDType.DOUBLE, 16, _AttrRole.MATRIX    ), "mat44d" ),
    ("normal3f"  , (_BaseDType.FLOAT ,  3, _AttrRole.NORMAL    ), "vec3"   ),
    ("point3f"   , (_BaseDType.FLOAT ,  3, _AttrRole.POSITION  ), "vec3"   ),
    ("quatf"     , (_BaseDType.FLOAT ,  4, _AttrRole.QUATERNION), "quat"   ),
    ("texCoord2f", (_BaseDType.FLOAT ,  2, _AttrRole.TEXCOORD  ), "vec2"   ),
    ("texCoord3f", (_BaseDType.FLOAT ,  3, _AttrRole.TEXCOORD  ), "vec3"   ),
    ("timecode"  , (_BaseDType.DOUBLE,  1, _AttrRole.TIMECODE  ), "float64"),
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
        prefix = "" if str_namespace is None else f"{str_namespace}."

        if dim_count == 0:
            return f"{prefix}{data_type_name}"

        if dim_count == 1:
            return f"{prefix}array(dtype={prefix}{data_type_name})"

        return f"{prefix}array(dtype={prefix}{data_type_name}, ndim={dim_count})"

    dtype = getattr(wp.types, data_type_name)
    if dim_count == 0:
        return dtype

    if dim_count == 1:
        return wp.array(dtype=dtype)

    return wp.array(dtype=dtype, ndim=dim_count)


def type_convert_og_to_warp(
    og_type: og.Type,
    dim_count: Optional[int] = None,
    as_str: bool = False,
    str_namespace: Optional[str] = "wp",
) -> Union[Any, str]:
    """Converts an OmniGraph type into a compatible Warp type."""
    data_type_name = _OG_DATA_TYPE_TO_WARP.get(
        (og_type.base_type, og_type.tuple_count, og_type.role),
    )
    if data_type_name is None:
        raise RuntimeError(f"Unsupported attribute type '{og_type}'.")

    if dim_count is None:
        dim_count = og_type.array_depth

    return get_warp_type_from_data_type_name(
        data_type_name,
        dim_count=dim_count,
        as_str=as_str,
        str_namespace=str_namespace,
    )


def type_convert_sdf_name_to_warp(
    sdf_type_name: str,
    dim_count: Optional[int] = None,
    as_str: bool = False,
    str_namespace: Optional[str] = "wp",
) -> Union[Any, str]:
    """Converts a Sdf type name into a compatible Warp type."""
    if sdf_type_name.endswith("[]"):
        sdf_type_name = sdf_type_name[:-2]
        if dim_count is None:
            dim_count = 1
    elif dim_count is None:
        dim_count = 0

    data_type_name = _SDF_DATA_TYPE_NAME_TO_WARP.get(sdf_type_name)
    if data_type_name is None:
        raise RuntimeError(f"Unsupported attribute type '{sdf_type_name}'.")

    return get_warp_type_from_data_type_name(
        data_type_name,
        dim_count=dim_count,
        as_str=as_str,
        str_namespace=str_namespace,
    )


def type_convert_sdf_name_to_og(
    sdf_type_name: str,
    is_array: Optional[bool] = None,
) -> og.Type:
    """Converts a Sdf type name into its corresponding OmniGraph type."""
    if sdf_type_name.endswith("[]"):
        sdf_type_name = sdf_type_name[:-2]
        if is_array is None:
            is_array = True
    elif is_array is None:
        is_array = False

    data_type = _SDF_DATA_TYPE_TO_OG.get(sdf_type_name)
    if data_type is None:
        raise RuntimeError(f"Unsupported attribute type '{sdf_type_name}'.")

    base_data_type, tuple_count, role = data_type
    return og.Type(
        base_data_type,
        tuple_count=tuple_count,
        array_depth=int(is_array),
        role=role,
    )
