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

"""Public Python API exposed by the omni.warp.nodes package."""

__all__ = [
    "AttrTracking",
    "NodeTimer",
    "basis_curves_copy_bundle",
    "basis_curves_create_bundle",
    "basis_curves_get_curve_count",
    "basis_curves_get_curve_vertex_counts",
    "basis_curves_get_display_color",
    "basis_curves_get_local_extent",
    "basis_curves_get_point_count",
    "basis_curves_get_points",
    "basis_curves_get_widths",
    "basis_curves_get_world_extent",
    "bundle_get_attr",
    "bundle_get_child_count",
    "bundle_get_prim_type",
    "bundle_get_world_xform",
    "bundle_has_changed",
    "bundle_have_attrs_changed",
    "bundle_set_prim_type",
    "bundle_set_world_xform",
    "device_get_cuda_compute",
    "from_omni_graph",
    "from_omni_graph_ptr",
    "mesh_copy_bundle",
    "mesh_create_bundle",
    "mesh_get_display_color",
    "mesh_get_face_count",
    "mesh_get_face_vertex_counts",
    "mesh_get_face_vertex_indices",
    "mesh_get_local_extent",
    "mesh_get_normals",
    "mesh_get_point_count",
    "mesh_get_points",
    "mesh_get_uvs",
    "mesh_get_velocities",
    "mesh_get_vertex_count",
    "mesh_get_world_extent",
    "mesh_triangulate",
    "points_copy_bundle",
    "points_create_bundle",
    "points_get_display_color",
    "points_get_local_extent",
    "points_get_masses",
    "points_get_point_count",
    "points_get_points",
    "points_get_velocities",
    "points_get_widths",
    "points_get_world_extent",
    "type_convert_og_to_warp",
    "type_convert_sdf_name_to_og",
    "type_convert_sdf_name_to_warp",
]

from ._impl.attributes import (
    AttrTracking,
    from_omni_graph,
    from_omni_graph_ptr,
)
from ._impl.basis_curves import (
    basis_curves_copy_bundle,
    basis_curves_create_bundle,
    basis_curves_get_curve_count,
    basis_curves_get_curve_vertex_counts,
    basis_curves_get_display_color,
    basis_curves_get_local_extent,
    basis_curves_get_point_count,
    basis_curves_get_points,
    basis_curves_get_widths,
    basis_curves_get_world_extent,
)
from ._impl.bundles import (
    bundle_get_attr,
    bundle_get_child_count,
    bundle_get_prim_type,
    bundle_get_world_xform,
    bundle_has_changed,
    bundle_have_attrs_changed,
)
from ._impl.common import (
    NodeTimer,
    device_get_cuda_compute,
    type_convert_og_to_warp,
    type_convert_sdf_name_to_og,
    type_convert_sdf_name_to_warp,
)
from ._impl.mesh import (
    mesh_copy_bundle,
    mesh_create_bundle,
    mesh_get_display_color,
    mesh_get_face_count,
    mesh_get_face_vertex_counts,
    mesh_get_face_vertex_indices,
    mesh_get_local_extent,
    mesh_get_normals,
    mesh_get_point_count,
    mesh_get_points,
    mesh_get_uvs,
    mesh_get_velocities,
    mesh_get_vertex_count,
    mesh_get_world_extent,
    mesh_triangulate,
)
from ._impl.points import (
    points_copy_bundle,
    points_create_bundle,
    points_get_display_color,
    points_get_local_extent,
    points_get_masses,
    points_get_point_count,
    points_get_points,
    points_get_velocities,
    points_get_widths,
    points_get_world_extent,
)
