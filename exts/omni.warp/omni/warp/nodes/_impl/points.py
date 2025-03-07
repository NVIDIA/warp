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

"""Helpers to author point cloud geometries represented as OmniGraph bundles."""

from math import inf
from typing import Optional

import numpy as np
import omni.graph.core as og

import warp as wp

from .attributes import (
    attr_get,
    attr_get_array_on_gpu,
    attr_set,
)
from .bundles import (
    bundle_copy_attr_value,
    bundle_create_attr,
    bundle_create_child,
    bundle_create_metadata_attr,
    bundle_get_attr,
    bundle_get_world_xform,
    bundle_set_prim_type,
    bundle_set_world_xform,
)

#   Public API
# ------------------------------------------------------------------------------


def points_create_bundle(
    dst_bundle: og.BundleContents,
    point_count: int,
    xform: Optional[np.ndarray] = None,
    create_display_color: bool = False,
    create_masses: bool = False,
    create_velocities: bool = False,
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

    bundle_set_prim_type(dst_bundle, "Points", child_idx=child_idx)

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

    if create_masses:
        bundle_create_attr(
            child_bundle,
            "masses",
            og.Type(
                og.BaseDataType.FLOAT,
                tuple_count=1,
                array_depth=1,
                role=og.AttributeRole.NONE,
            ),
            size=point_count,
        )

    if create_velocities:
        bundle_create_attr(
            child_bundle,
            "velocities",
            og.Type(
                og.BaseDataType.FLOAT,
                tuple_count=3,
                array_depth=1,
                role=og.AttributeRole.VECTOR,
            ),
            size=point_count,
        )

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


def points_copy_bundle(
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
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "primvars:displayColor", wp.vec3)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "masses", float)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "velocities", wp.vec3)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "widths", float)


def points_get_point_count(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> int:
    """Retrieves the number of points."""
    return bundle_get_attr(bundle, "points", child_idx).size()


def points_get_points(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle points attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "points", child_idx)
    return attr_get_array_on_gpu(attr, wp.vec3, read_only=bundle.read_only)


def points_get_velocities(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle velocities attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "velocities", child_idx)
    return attr_get_array_on_gpu(attr, wp.vec3, read_only=bundle.read_only)


def points_get_widths(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=float):
    """Retrieves the bundle widths attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "widths", child_idx)
    return attr_get_array_on_gpu(attr, float, read_only=bundle.read_only)


def points_get_masses(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=float):
    """Retrieves the bundle masses attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "masses", child_idx)
    return attr_get_array_on_gpu(attr, float, read_only=bundle.read_only)


def points_get_display_color(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle display color attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "primvars:displayColor", child_idx)
    return attr_get_array_on_gpu(attr, wp.vec3, read_only=bundle.read_only)


def points_get_local_extent(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> np.ndarray:
    """Retrieves the local extent of the geometry points."""
    # Some standard workflows include a single 'extent' attribute when defining
    # geometry primitives on the stage.
    attr = bundle_get_attr(bundle, "extent", child_idx)
    if attr is not None:
        return attr_get(attr)

    # Alternatively, the ReadPrims node offers an option to compute the bounding
    # box which results in a triple of 'bboxMinCorner', 'bboxMaxCorner',
    # and 'bboxTransform' attributes.
    min_attr = bundle_get_attr(bundle, "bboxMinCorner", child_idx)
    max_attr = bundle_get_attr(bundle, "bboxMaxCorner", child_idx)
    if min_attr is not None and max_attr is not None:
        return np.stack(
            (
                attr_get(min_attr),
                attr_get(max_attr),
            ),
        )

    # The last resort is to compute the extent ourselves from
    # the point positions.
    points = points_get_points(bundle, child_idx=child_idx)
    min_extent = wp.array((+inf, +inf, +inf), dtype=wp.vec3)
    max_extent = wp.array((-inf, -inf, -inf), dtype=wp.vec3)
    wp.launch(
        _compute_extent_kernel,
        dim=len(points),
        inputs=[points],
        outputs=[min_extent, max_extent],
    )
    return np.concatenate((min_extent.numpy(), max_extent.numpy()))


def points_get_world_extent(
    bundle: og.BundleContents,
    axis_aligned: bool = False,
    child_idx: int = 0,
) -> np.ndarray:
    """Retrieves the world extent of the geometry points."""
    extent = points_get_local_extent(bundle, child_idx=child_idx)
    xform = bundle_get_world_xform(bundle, child_idx=child_idx)

    if axis_aligned:
        points = np.array(
            (
                (extent[0][0], extent[0][1], extent[0][2]),
                (extent[0][0], extent[0][1], extent[1][2]),
                (extent[0][0], extent[1][1], extent[0][2]),
                (extent[0][0], extent[1][1], extent[1][2]),
                (extent[1][0], extent[1][1], extent[1][2]),
                (extent[1][0], extent[0][1], extent[1][2]),
                (extent[1][0], extent[1][1], extent[0][2]),
                (extent[1][0], extent[0][1], extent[0][2]),
            ),
        )
    else:
        points = extent

    points = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
    points = np.dot(xform.T, points[:, :, None]).squeeze()[:-1, :].T
    return np.array(
        (
            np.amin(points, axis=0),
            np.amax(points, axis=0),
        )
    )


#   Private Helpers
# ------------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _compute_extent_kernel(
    points: wp.array(dtype=wp.vec3),
    out_min_extent: wp.array(dtype=wp.vec3),
    out_max_extent: wp.array(dtype=wp.vec3),
):
    """Computes the extent of a point cloud."""
    tid = wp.tid()
    wp.atomic_min(out_min_extent, 0, points[tid])
    wp.atomic_max(out_max_extent, 0, points[tid])
