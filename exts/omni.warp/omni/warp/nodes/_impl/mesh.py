# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Helpers to author mesh geometries represented as OmniGraph bundles."""

from typing import Optional

import numpy as np
import omni.graph.core as og
import warp as wp

from omni.warp.nodes._impl.attributes import (
    attr_get,
    attr_get_array_on_gpu,
)
from omni.warp.nodes._impl.bundles import (
    bundle_copy_attr_value,
    bundle_create_attr,
    bundle_create_child,
    bundle_get_attr,
    bundle_set_prim_type,
    bundle_set_world_xform,
)
from omni.warp.nodes._impl.points import (
    points_get_points,
    points_get_velocities,
    points_get_local_extent,
    points_get_world_extent,
)


def mesh_create_bundle(
    dst_bundle: og.BundleContents,
    point_count: int,
    vertex_count: int,
    face_count: int,
    xform: Optional[np.ndarray] = None,
    child_idx: int = 0,
) -> None:
    """Creates and initializes mesh attributes within a bundle."""
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
        "normals",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.NORMAL,
        ),
        size=vertex_count,
    )
    bundle_create_attr(
        child_bundle,
        "primvars:st",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=2,
            array_depth=1,
            role=og.AttributeRole.TEXCOORD,
        ),
        size=vertex_count,
    )
    bundle_create_attr(
        child_bundle,
        "faceVertexCounts",
        og.Type(
            og.BaseDataType.INT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
        size=face_count,
    )
    bundle_create_attr(
        child_bundle,
        "faceVertexIndices",
        og.Type(
            og.BaseDataType.INT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
        size=vertex_count,
    )

    bundle_set_prim_type(dst_bundle, "Mesh", child_idx=child_idx)

    if xform is not None:
        bundle_set_world_xform(dst_bundle, xform, child_idx=child_idx)


def mesh_copy_bundle(
    dst_bundle: og.BundleContents,
    src_bundle: og.BundleContents,
    deep_copy: bool = False,
    child_idx: int = 0,
) -> None:
    """Creates and initializes mesh attributes from an existing bundle."""
    dst_child_bundle = bundle_create_child(dst_bundle, child_idx)
    src_child_bundle = src_bundle.bundle.get_child_bundle(child_idx)
    dst_child_bundle.copy_bundle(src_child_bundle)

    if deep_copy:
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "points", wp.vec3)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "normals", wp.vec3)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "primvars:st", wp.vec2)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "faceVertexCounts", int)
        bundle_copy_attr_value(dst_child_bundle, src_child_bundle, "faceVertexIndices", int)


def mesh_get_point_count(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> int:
    """Retrieves the number of points."""
    return bundle_get_attr(bundle, "points", child_idx).size()


def mesh_get_vertex_count(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> int:
    """Retrieves the number of vertices."""
    attr = bundle_get_attr(bundle, "faceVertexCounts", child_idx)
    return int(np.sum(attr_get(attr)))


def mesh_get_face_count(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> int:
    """Retrieves the number of faces."""
    return bundle_get_attr(bundle, "faceVertexCounts", child_idx).size()


def mesh_get_points(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle points attribute as a Warp array."""
    return points_get_points(bundle, child_idx=child_idx)


def mesh_get_velocities(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle velocities attribute as a Warp array."""
    return points_get_velocities(bundle, child_idx=child_idx)


def mesh_get_normals(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle normals attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "normals", child_idx)
    return attr_get_array_on_gpu(attr, wp.vec3, read_only=bundle.read_only)


def mesh_get_uvs(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=wp.vec2):
    """Retrieves the bundle UVs attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "primvars:st", child_idx)
    return attr_get_array_on_gpu(attr, wp.vec2, read_only=bundle.read_only)


def mesh_get_face_vertex_counts(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=int):
    """Retrieves the bundle face vertex counts attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "faceVertexCounts", child_idx)
    return attr_get_array_on_gpu(attr, int, read_only=bundle.read_only)


def mesh_get_face_vertex_indices(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=int):
    """Retrieves the bundle face vertex indices attribute as a Warp array."""
    attr = bundle_get_attr(bundle, "faceVertexIndices", child_idx)
    return attr_get_array_on_gpu(attr, int, read_only=bundle.read_only)


def mesh_get_local_extent(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> np.ndarray:
    """Retrieves the local extent of the geometry mesh."""
    return points_get_local_extent(bundle, child_idx=child_idx)


def mesh_get_world_extent(
    bundle: og.BundleContents,
    axis_aligned: bool = False,
    child_idx: int = 0,
) -> np.ndarray:
    """Retrieves the world extent of the geometry mesh."""
    return points_get_world_extent(
        bundle,
        axis_aligned=axis_aligned,
        child_idx=child_idx,
    )


def mesh_get_triangulated_face_vertex_indices(
    bundle: og.BundleContents,
    child_idx: int = 0,
) -> wp.array(dtype=int):
    """Retrieves a triangulated version of the face vertex indices."""
    counts = mesh_get_face_vertex_counts(bundle, child_idx=child_idx).numpy()
    if np.all(counts == 3):
        return mesh_get_face_vertex_indices(bundle, child_idx=child_idx)

    indices = mesh_get_face_vertex_indices(bundle, child_idx=child_idx).numpy()
    tri_face_count = np.sum(np.subtract(counts, 2))
    out = np.empty(tri_face_count * 3, dtype=int)

    dst_offset = 0
    src_offset = 0
    for count in counts:
        for i in range(count - 2):
            out[dst_offset] = indices[src_offset]
            out[dst_offset + 1] = indices[src_offset + i + 1]
            out[dst_offset + 2] = indices[src_offset + i + 2]
            dst_offset += 3

        src_offset += count

    return wp.array(out, dtype=int, copy=True)
