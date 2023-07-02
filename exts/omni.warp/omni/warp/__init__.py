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

from ._impl.common import (
    NodeTimer,
)
from ._impl.omnigraph.attributes import (
    from_omni_graph,
)
from ._impl.omnigraph.bundles import (
    bundle_define_prim_attrs,
    bundle_get_child_count,
    bundle_get_prim_type,
    bundle_get_world_xform,
)
from ._impl.omnigraph.mesh import (
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
from ._impl.omnigraph.points import (
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
from ._impl.usd import prim_get_world_xform

# Register the extension by importing its entry point class.
from ._impl.extension import OmniWarpExtension
