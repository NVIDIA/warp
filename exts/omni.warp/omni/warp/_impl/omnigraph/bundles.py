# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generic helpers to author OmniGraph bundles."""

from typing import Optional

import numpy as np
import omni.graph.core as og
import warp as wp

from omni.warp._impl.omnigraph.attributes import (
    attr_get_cpu_array,
    attr_get_gpu_array,
    attr_set_cpu_array,
)
from omni.warp._impl.usd import prim_get_world_xform


#   High-level Bundle API (og.BundleContents)
# ------------------------------------------------------------------------------


def bundle_get_child_count(
    bundle: og.BundleContents,
) -> int:
    """Retrieves the number of children defined for a bundle."""
    return bundle.bundle.get_child_bundle_count()


def bundle_get_prim_type(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> str:
    """Retrieves the primitive type."""
    attr = bundle_get_attr(bundle, "sourcePrimType", child_idx)
    return attr_get_cpu_array(attr)


def bundle_get_world_xform(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> np.ndarray:
    """Retrieves the world transformation matrix."""
    attr = bundle_get_attr(bundle, "worldMatrix", child_idx)
    if attr is None:
        return np.identity(4)

    return attr_get_cpu_array(attr).reshape(4, 4)


def bundle_create_child(
    bundle: og.BundleContents,
    child_idx: int,
) -> og.IBundle2:
    """Creates a single child bundle if it doesn't already exist."""
    if child_idx < bundle.bundle.get_child_bundle_count():
        return bundle.bundle.get_child_bundle(child_idx)

    return bundle.bundle.create_child_bundle("prim{}".format(child_idx))


def bundle_get_attr(
    bundle: og.BundleContents,
    name: str,
    child_idx: int,
) -> Optional[og.AttributeData]:
    """Retrieves a bundle attribute from its name."""
    if bundle.bundle.get_child_bundle_count():
        attr = bundle.bundle.get_child_bundle(child_idx).get_attribute_by_name(name)
    else:
        attr = bundle.bundle.get_attribute_by_name(name)

    if not attr.is_valid():
        return None

    return attr


def bundle_define_prim_attrs(
    bundle: og.BundleContents,
    prim_type: str,
    xform_prim_path: Optional[str] = None,
    child_idx: int = 0,
) -> None:
    """Defines the primitive attributes."""
    child_bundle = bundle_create_child(bundle, child_idx)
    xform = prim_get_world_xform(xform_prim_path)

    prim_type_attr = bundle_create_attr(
        child_bundle,
        "sourcePrimType",
        og.Type(
            og.BaseDataType.TOKEN,
            tuple_count=1,
            array_depth=0,
            role=og.AttributeRole.NONE,
        ),
    )
    attr_set_cpu_array(prim_type_attr, prim_type)

    world_matrix_attr = bundle_create_attr(
        child_bundle,
        "worldMatrix",
        og.Type(
            og.BaseDataType.DOUBLE,
            tuple_count=16,
            array_depth=0,
            role=og.AttributeRole.MATRIX,
        ),
    )
    attr_set_cpu_array(world_matrix_attr, xform)


#   Low-level Bundle API (og.IBundle2)
# ------------------------------------------------------------------------------


def bundle_create_attr(
    bundle: og.IBundle2,
    name: str,
    og_type: og.Type,
    size: int = 0,
) -> og.AttributeData:
    """Creates a bundle attribute if it doesn't already exist."""
    attr = bundle.get_attribute_by_name(name)
    if attr.is_valid() and attr.get_type() == og_type and attr.size() == size:
        return attr

    return bundle.create_attribute(name, og_type, element_count=size)


def bundle_copy_attr_value(
    dst_bundle: og.IBundle2,
    src_bundle: og.IBundle2,
    name: str,
    dtype: og.Type,
) -> None:
    """Copies an attribute value from one bundle to another."""
    dst_attr = dst_bundle.get_attribute_by_name(name)
    src_attr = src_bundle.get_attribute_by_name(name)

    if not dst_attr.is_valid() or not src_attr.is_valid():
        return

    wp.copy(
        attr_get_gpu_array(dst_attr, dtype, read_only=False),
        attr_get_gpu_array(src_attr, dtype, read_only=True),
    )
