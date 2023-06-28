# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Public Python API exposed by the omni.warp extension."""

__all__ = [
    "NodeTimer",
    "bundle_define_prim_attrs",
    "bundle_get_child_count",
    "bundle_get_prim_type",
    "bundle_get_world_xform",
    "from_omni_graph",
    "mesh_create_bundle",
    "mesh_copy_bundle",
    "mesh_get_face_count",
    "mesh_get_face_vertex_counts",
    "mesh_get_face_vertex_indices",
    "mesh_get_local_extent",
    "mesh_get_normals",
    "mesh_get_point_count",
    "mesh_get_points",
    "mesh_get_triangulated_face_vertex_indices",
    "mesh_get_uvs",
    "mesh_get_velocities",
    "mesh_get_vertex_count",
    "mesh_get_world_extent",
    "points_create_bundle",
    "points_copy_bundle",
    "points_get_local_extent",
    "points_get_masses",
    "points_get_point_count",
    "points_get_points",
    "points_get_velocities",
    "points_get_widths",
    "points_get_world_extent",
    "prim_get_world_xform",
]

from typing import (
    Any,
    Optional,
    Union,
    Sequence,
)

import numpy as np
import omni.graph.core as og
import omni.usd
from pxr import (
    Usd,
    UsdGeom,
)
import warp as wp

from .scripts.omnigraph.attributes import (
    attr_set_cpu_array,
    attr_cast_array_to_warp,
)
from .scripts.omnigraph.bundles import (
    bundle_create_attr,
    bundle_create_child,
    bundle_get_child_count,
    bundle_get_prim_type,
    bundle_get_world_xform,
)
from .scripts.omnigraph.mesh import (
    mesh_create_bundle,
    mesh_copy_bundle,
    mesh_get_face_count,
    mesh_get_face_vertex_counts,
    mesh_get_face_vertex_indices,
    mesh_get_local_extent,
    mesh_get_normals,
    mesh_get_point_count,
    mesh_get_points,
    mesh_get_triangulated_face_vertex_indices,
    mesh_get_uvs,
    mesh_get_velocities,
    mesh_get_vertex_count,
    mesh_get_world_extent,
)
from .scripts.omnigraph.points import (
    points_create_bundle,
    points_copy_bundle,
    points_get_local_extent,
    points_get_masses,
    points_get_point_count,
    points_get_points,
    points_get_velocities,
    points_get_widths,
    points_get_world_extent,
)

# Register the extension by importing its entry point class.
from .scripts.extension import OmniWarpExtension


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


#   USD Primitives
# ------------------------------------------------------------------------------


def prim_get_world_xform(prim_path: Optional[str]) -> None:
    """Retrieves the world transformation matrix from a USD primitive."""
    if prim_path is not None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
            prim = UsdGeom.Xformable(prim)
            return prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    return np.identity(4)


#   OmniGraph Attribute Values
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


#   OmniGraph Bundles
# ------------------------------------------------------------------------------


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
