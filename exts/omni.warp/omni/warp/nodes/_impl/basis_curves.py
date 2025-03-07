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

"""Helpers to author basis curves geometries represented as OmniGraph bundles."""

from typing import Optional

import numpy as np
import omni.graph.core as og

import warp as wp

from .attributes import (
    attr_get_array_on_gpu,
    attr_set,
)
from .bundles import (
    bundle_copy_attr_value,
    bundle_create_attr,
    bundle_create_child,
    bundle_create_metadata_attr,
    bundle_get_attr,
    bundle_set_prim_type,
    bundle_set_world_xform,
)
from .points import (
    points_get_display_color,
    points_get_local_extent,
    points_get_points,
    points_get_widths,
    points_get_world_extent,
)

#   Public API
# ------------------------------------------------------------------------------


def basis_curves_create_bundle(
    dst_bundle: og.BundleContents,
    point_count: int,
    curve_count: int,
    type: Optional[str] = None,
    basis: Optional[str] = None,
    wrap: Optional[str] = None,
    xform: Optional[np.ndarray] = None,
    create_display_color: bool = False,
    create_widths: bool = False,
    child_idx: int = 0,
) -> None:
    """Creates and initializes point cloud attributes within a bundle."""
    child_bundle = bundle_create_child(dst_bundle, child_idx)
    bundle_create_attr(
        child_bundle,
        "points",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.POSITION,
        ),
        size=point_count,
    )
    bundle_create_attr(
        child_bundle,
        "curveVertexCounts",
        og.Type(
            og.BaseDataType.INT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
        size=curve_count,
    )

    if type is not None:
        attr = bundle_create_attr(
            child_bundle,
            "type",
            og.Type(
                og.BaseDataType.TOKEN,
                tuple_count=1,
                array_depth=0,
                role=og.AttributeRole.NONE,
            ),
        )
        attr_set(attr, type)

    if basis is not None:
        attr = bundle_create_attr(
            child_bundle,
            "basis",
            og.Type(
                og.BaseDataType.TOKEN,
                tuple_count=1,
                array_depth=0,
                role=og.AttributeRole.NONE,
            ),
        )
        attr_set(attr, basis)

    if wrap is not None:
        attr = bundle_create_attr(
            child_bundle,
            "warp",
            og.Type(
                og.BaseDataType.TOKEN,
                tuple_count=1,
                array_depth=0,
                role=og.AttributeRole.NONE,
            ),
        )
        attr_set(attr, wrap)

    bundle_set_prim_type(dst_bundle, "BasisCurves", child_idx=child_idx)

    if xform is not None:
        bundle_set_world_xform(dst_bundle, xform, child_idx=child_idx)

    if create_display_color:
        bundle_create_attr(
            child_bundle,
            "primvars:displayColor",
            og.Type(
                og.BaseDataType.FLOAT,
                tuple_count=3,
                array_depth=1,
                role=og.AttributeRole.COLOR,
            ),
            size=point_count,
        )
        interp_attr = bundle_create_metadata_attr(
            child_bundle,
            "primvars:displayColor",
            "interpolation",
            og.Type(
                og.BaseDataType.TOKEN,
                tuple_count=1,
                array_depth=0,
                role=og.AttributeRole.NONE,
            ),
        )
        attr_set(interp_attr, "vertex")

    if create_widths:
        bundle_create_attr(
            child_bundle,
            "widths",
            og.Type(
                og.BaseDataType.FLOAT,
                tuple_count=1,
                array_depth=1,
                role=og.AttributeRole.NONE,
            ),
            size=point_count,
        )


def basis_curves_copy_bundle(
    dst_bundle: og.BundleContents,
    src_bundle: og.BundleContents,
    deep_copy: bool = False,
    child_idx: int = 0,
) -> None:
    """Creates and initializes points attributes from an existing bundle."""
    dst_child_bundle = bundle_create_child(dst_bundle, child_idx)
    src_child_bundle = src_bundle.bundle.get_child_bundle(child_idx)
    dst_child_bundle.copy_bundle(src_child_bundle)

    if deep_copy:
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "points", wp.vec3)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "curveVertexCounts", int)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "primvars:displayColor", wp.vec3)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "widths", float)


def basis_curves_get_point_count(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> int:
    """Retrieves the number of points."""
    return bundle_get_attr(bundle, "points", child_idx).size()


def basis_curves_get_curve_count(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> int:
    """Retrieves the number of curves."""
    return bundle_get_attr(bundle, "curveVertexCounts", child_idx).size()


def basis_curves_get_points(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle points attribute as a Warp array."""
    return points_get_points(bundle, child_idx=child_idx)


def basis_curves_get_widths(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=float):
    """Retrieves the bundle widths attribute as a Warp array."""
    return points_get_widths(bundle, child_idx=child_idx)


def basis_curves_get_display_color(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle display color attribute as a Warp array."""
    return points_get_display_color(bundle, child_idx=child_idx)


def basis_curves_get_curve_vertex_counts(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=int):
    """Retrieves the bundle curve vertex counts attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "curveVertexCounts", child_idx)
    return attr_get_array_on_gpu(attr, int, read_only=bundle.read_only)


def basis_curves_get_local_extent(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> np.ndarray:
    """Retrieves the local extent of the geometry points."""
    return points_get_local_extent(bundle, child_idx=child_idx)


def basis_curves_get_world_extent(
    bundle: og.BundleContents,
    axis_aligned: bool = False,
    child_idx: int = 0,
) -> np.ndarray:
    """Retrieves the world extent of the geometry points."""
    return points_get_world_extent(
        bundle,
        axis_aligned=axis_aligned,
        child_idx=child_idx,
    )
