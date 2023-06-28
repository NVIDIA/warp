# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from enum import Enum
from typing import (
    Any,
    Mapping,
    Optional,
)

import carb
import warp as wp


class IntEnum(int, Enum):
    """Base class for integer enumerators with labels."""

    def __new__(cls, value, label):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.label = label
        return obj


def get_annotations(obj: Any) -> Mapping[str, Any]:
    """Alternative to `inspect.get_annotations()` for Python 3.9 and older."""
    # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
    if isinstance(obj, type):
        return obj.__dict__.get("__annotations__", {})

    return getattr(obj, "__annotations__", {})


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


def log_info(msg):
    carb.log_info("[omni.warp] {}".format(msg))


def log_warn(msg):
    carb.log_warn("[omni.warp] {}".format(msg))


def log_error(msg):
    carb.log_error("[omni.warp] {}".format(msg))
