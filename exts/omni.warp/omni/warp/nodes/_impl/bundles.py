# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Helpers to author OmniGraph bundles."""

from typing import (
    Optional,
    Sequence,
)

import numpy as np
import omni.graph.core as og
import warp as wp

from omni.warp.nodes._impl.attributes import (
    attr_get,
    attr_get_array_on_gpu,
    attr_set,
)


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
    return attr_get(attr)


def bundle_set_prim_type(
    bundle: og.BundleContents,
    prim_type: str,
    child_idx: int = 0,
) -> None:
    """Sets the primitive type."""
    child_bundle = bundle_create_child(bundle, child_idx)
    attr = bundle_create_attr(
        child_bundle,
        "sourcePrimType",
        og.Type(
            og.BaseDataType.TOKEN,
            tuple_count=1,
            array_depth=0,
            role=og.AttributeRole.NONE,
        ),
    )
    attr_set(attr, prim_type)


def bundle_get_world_xform(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> np.ndarray:
    """Retrieves the world transformation matrix."""
    attr = bundle_get_attr(bundle, "worldMatrix", child_idx)
    if attr is None:
        return np.identity(4)

    return attr_get(attr).reshape(4, 4)


def bundle_set_world_xform(
    bundle: og.BundleContents,
    xform: np.ndarray,
    child_idx: int = 0,
) -> None:
    """Sets the bundle's world transformation matrix."""
    child_bundle = bundle_create_child(bundle, child_idx)
    attr = bundle_create_attr(
        child_bundle,
        "worldMatrix",
        og.Type(
            og.BaseDataType.DOUBLE,
            tuple_count=16,
            array_depth=0,
            role=og.AttributeRole.MATRIX,
        ),
    )
    attr_set(attr, xform)


def bundle_create_child(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> og.IBundle2:
    """Creates a single child bundle if it doesn't already exist."""
    if child_idx < bundle.bundle.get_child_bundle_count():
        return bundle.bundle.get_child_bundle(child_idx)

    return bundle.bundle.create_child_bundle("prim{}".format(child_idx))


def bundle_get_attr(
    bundle: og.BundleContents,
    name: str,
    child_idx: int = 0,
) -> Optional[og.AttributeData]:
    """Retrieves a bundle attribute from its name."""
    if bundle.bundle.get_child_bundle_count():
        attr = bundle.bundle.get_child_bundle(child_idx).get_attribute_by_name(name)
    else:
        attr = bundle.bundle.get_attribute_by_name(name)

    if not attr.is_valid():
        return None

    return attr


def bundle_has_changed(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> bool:
    """Checks whether the contents of the bundle has changed."""
    with bundle.changes() as bundle_changes:
        child_bundle = bundle.bundle.get_child_bundle(child_idx)
        return bundle_changes.get_change(child_bundle) != og.BundleChangeType.NONE


def bundle_have_attrs_changed(
    bundle: og.BundleContents,
    attr_names: Sequence[str],
    child_idx: int = 0,
) -> bool:
    """Checks whether the contents of a bundle's attributes have changed."""
    with bundle.changes() as bundle_changes:
        child_bundle = bundle.bundle.get_child_bundle(child_idx)
        for attr_name in attr_names:
            attr = child_bundle.get_attribute_by_name(attr_name)
            if bundle_changes.get_change(attr) != og.BundleChangeType.NONE:
                return True

    return False


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
        attr_get_array_on_gpu(dst_attr, dtype, read_only=False),
        attr_get_array_on_gpu(src_attr, dtype, read_only=True),
    )
