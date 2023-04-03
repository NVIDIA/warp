# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Public Python API exposed by the omni.warp extension."""

__all__ = [
    "MeshAttributeFlags",
    "NodeTimer",
    "PointsAttributeFlags",
    "define_prim_attrs",
    "from_omni_graph",
    "get_prim_type",
    "get_world_xform",
    "get_world_xform_from_prim",
    "mesh_clear_dirty_attributes",
    "mesh_create_bundle",
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
    "mesh_set_dirty_attributes",
    "points_clear_dirty_attributes",
    "points_create_bundle",
    "points_get_local_extent",
    "points_get_masses",
    "points_get_point_count",
    "points_get_points",
    "points_get_velocities",
    "points_get_widths",
    "points_get_world_extent",
    "points_set_dirty_attributes",
]

from enum import IntFlag
from math import inf
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

from .scripts.attributes import (
    cast_array_attr_value_to_warp,
    insert_bundle_attr,
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


#   Attribute Values
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

    return cast_array_attr_value_to_warp(value, dtype, shape, device)


#   USD Stage
# ------------------------------------------------------------------------------


def get_world_xform_from_prim(prim_path: str) -> None:
    """Retrieves the world transformation matrix from a USD primitive."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
        prim = UsdGeom.Xformable(prim)
        return prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    return np.identity(4)


#   Bundle Attributes
# ------------------------------------------------------------------------------


def get_prim_type(bundle: og.BundleContents) -> str:
    """Retrieves the primitive type."""
    attr = bundle.attribute_by_name("sourcePrimType")
    return attr.cpu_value


def get_world_xform(bundle: og.BundleContents) -> np.ndarray:
    """Retrieves the world transformation matrix."""
    attr = bundle.attribute_by_name("worldMatrix")
    if attr is None:
        return np.identity(4)

    return attr.cpu_value.reshape(4, 4)


def define_prim_attrs(
    bundle: og.BundleContents,
    source_prim_path: str,
    source_prim_type: str,
) -> None:
    """Defines the primitive attributes."""
    source_prim_path_attr = insert_bundle_attr(
        bundle,
        "sourcePrimPath",
        og.Type(
            og.BaseDataType.TOKEN,
            tuple_count=1,
            array_depth=0,
            role=og.AttributeRole.NONE,
        ),
    )
    source_prim_path_attr.cpu_value = source_prim_path

    source_prim_type_attr = insert_bundle_attr(
        bundle,
        "sourcePrimType",
        og.Type(
            og.BaseDataType.TOKEN,
            tuple_count=1,
            array_depth=0,
            role=og.AttributeRole.NONE,
        ),
    )
    source_prim_type_attr.cpu_value = source_prim_type

    world_matrix_attr = insert_bundle_attr(
        bundle,
        "worldMatrix",
        og.Type(
            og.BaseDataType.DOUBLE,
            tuple_count=16,
            array_depth=0,
            role=og.AttributeRole.MATRIX,
        ),
    )
    world_matrix_attr.cpu_value = get_world_xform_from_prim(source_prim_path)


#   Point Cloud Geometries
# ------------------------------------------------------------------------------


class PointsAttributeFlags(IntFlag):
    """Flags representing point cloud attributes having been updated."""

    NONE = 0
    POINTS = 1 << 0
    VELOCITIES = 1 << 1
    WIDTHS = 1 << 2
    MASSES = 1 << 3
    ALL = POINTS | VELOCITIES | WIDTHS | MASSES


def points_create_bundle(bundle: og.BundleContents, point_count: int) -> None:
    """Creates and initializes point cloud attributes within a bundle."""
    # Create the attributes.
    points_attr = insert_bundle_attr(
        bundle,
        "points",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.POSITION,
        ),
    )
    velocities_attr = insert_bundle_attr(
        bundle,
        "velocities",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.VECTOR,
        ),
    )
    widths_attr = insert_bundle_attr(
        bundle,
        "widths",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
    )
    masses_attr = insert_bundle_attr(
        bundle,
        "masses",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
    )

    # Set the size of the array attributes.
    points_attr.size = point_count
    velocities_attr.size = point_count
    widths_attr.size = point_count
    masses_attr.size = point_count


def points_get_point_count(bundle: og.BundleContents) -> int:
    """Retrieves the number of points."""
    return bundle.attribute_by_name("points").size


def points_get_points(bundle: og.BundleContents) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle points attribute as a Warp array."""
    attr = bundle.attribute_by_name("points")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=wp.vec3)


def points_get_velocities(bundle: og.BundleContents) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle velocities attribute as a Warp array."""
    attr = bundle.attribute_by_name("velocities")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=wp.vec3)


def points_get_widths(bundle: og.BundleContents) -> wp.array(dtype=float):
    """Retrieves the bundle widths attribute as a Warp array."""
    attr = bundle.attribute_by_name("widths")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=float)


def points_get_masses(bundle: og.BundleContents) -> wp.array(dtype=float):
    """Retrieves the bundle masses attribute as a Warp array."""
    attr = bundle.attribute_by_name("masses")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=float)


def points_get_local_extent(bundle: og.BundleContents) -> np.ndarray:
    """Retrieves the local extent of the geometry points."""
    # Some standard workflows include a single 'extent' attribute when defining
    # geometry primitives on the stage.
    attr = bundle.attribute_by_name("extent")
    if attr is not None:
        return attr.cpu_value

    # Alternatively, the ReadPrims node offers an option to compute the bounding
    # box which results in a triple of 'bboxMinCorner', 'bboxMaxCorner',
    # and 'bboxTransform' attributes.
    min_attr = bundle.attribute_by_name("bboxMinCorner")
    max_attr = bundle.attribute_by_name("bboxMaxCorner")
    if min_attr is not None and max_attr is not None:
        return np.stack((min_attr.cpu_value, max_attr.cpu_value))

    # The last resort is to compute the extent ourselves from
    # the point positions.
    points = points_get_points(bundle)
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
) -> np.ndarray:
    """Retrieves the world extent of the geometry points."""
    extent = points_get_local_extent(bundle)
    xform = get_world_xform(bundle)

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


def points_clear_dirty_attributes(bundle: og.BundleContents) -> None:
    """Clears the list of attributes that have been updated."""
    attr = _create_dirty_attrs_attr(bundle)
    attr.cpu_value = ""


def points_set_dirty_attributes(
    bundle: og.BundleContents,
    flags: PointsAttributeFlags,
) -> None:
    """Sets attributes that have been updated."""
    attr = _create_dirty_attrs_attr(bundle)
    values = attr.cpu_value.split()
    additional_values = []

    if PointsAttributeFlags.POINTS in flags:
        additional_values.append("points")

    if PointsAttributeFlags.VELOCITIES in flags:
        additional_values.append("velocities")

    if PointsAttributeFlags.WIDTHS in flags:
        additional_values.append("widths")

    if PointsAttributeFlags.MASSES in flags:
        additional_values.append("masses")

    additional_values = tuple(x for x in additional_values if x not in values)
    values.extend(additional_values)
    attr.cpu_value = " ".join(values)


#   Mesh Geometries
# ------------------------------------------------------------------------------


class MeshAttributeFlags(IntFlag):
    """Flags representing mesh attributes having been updated."""

    NONE = 0
    POINTS = 1 << 0
    NORMALS = 1 << 1
    UVS = 1 << 2
    TOPOLOGY = 1 << 3
    ALL = POINTS | NORMALS | UVS | TOPOLOGY


def mesh_create_bundle(
    bundle: og.BundleContents,
    point_count: int,
    vertex_count: int,
    face_count: int,
) -> None:
    """Creates and initializes mesh attributes within a bundle."""
    # Create the attributes.
    points_attr = insert_bundle_attr(
        bundle,
        "points",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.POSITION,
        ),
    )
    normals_attr = insert_bundle_attr(
        bundle,
        "normals",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.NORMAL,
        ),
    )
    uvs_attr = insert_bundle_attr(
        bundle,
        "primvars:st",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=2,
            array_depth=1,
            role=og.AttributeRole.TEXCOORD,
        ),
    )
    face_vertex_counts_attr = insert_bundle_attr(
        bundle,
        "faceVertexCounts",
        og.Type(
            og.BaseDataType.INT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
    )
    face_vertex_indices_attr = insert_bundle_attr(
        bundle,
        "faceVertexIndices",
        og.Type(
            og.BaseDataType.INT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
    )

    # Set the size of the array attributes.
    points_attr.size = point_count
    normals_attr.size = vertex_count
    uvs_attr.size = vertex_count
    face_vertex_counts_attr.size = face_count
    face_vertex_indices_attr.size = vertex_count


def mesh_get_point_count(bundle: og.BundleContents) -> int:
    """Retrieves the number of points."""
    return bundle.attribute_by_name("points").size


def mesh_get_vertex_count(bundle: og.BundleContents) -> int:
    """Retrieves the number of vertices."""
    return int(np.sum(bundle.attribute_by_name("faceVertexCounts").cpu_value))


def mesh_get_face_count(bundle: og.BundleContents) -> int:
    """Retrieves the number of faces."""
    return bundle.attribute_by_name("faceVertexCounts").size


def mesh_get_points(bundle: og.BundleContents) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle points attribute as a Warp array."""
    attr = bundle.attribute_by_name("points")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=wp.vec3)


def mesh_get_velocities(bundle: og.BundleContents) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle velocities attribute as a Warp array."""
    attr = bundle.attribute_by_name("velocities")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=wp.vec3)


def mesh_get_normals(bundle: og.BundleContents) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle normals attribute as a Warp array."""
    attr = bundle.attribute_by_name("normals")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=wp.vec3)


def mesh_get_uvs(bundle: og.BundleContents) -> wp.array(dtype=wp.vec2):
    """Retrieves the bundle UVs attribute as a Warp array."""
    attr = bundle.attribute_by_name("primvars:st")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=wp.vec2)


def mesh_get_face_vertex_counts(bundle: og.BundleContents) -> wp.array(dtype=int):
    """Retrieves the bundle face vertex counts attribute as a Warp array."""
    attr = bundle.attribute_by_name("faceVertexCounts")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=int)


def mesh_get_face_vertex_indices(bundle: og.BundleContents) -> wp.array(dtype=int):
    """Retrieves the bundle face vertex indices attribute as a Warp array."""
    attr = bundle.attribute_by_name("faceVertexIndices")
    return wp.from_ptr(attr.gpu_value.memory, attr.size, dtype=int)


def mesh_get_local_extent(bundle: og.BundleContents) -> np.ndarray:
    """Retrieves the local extent of the geometry mesh."""
    return points_get_local_extent(bundle)


def mesh_get_world_extent(
    bundle: og.BundleContents,
    axis_aligned: bool = False,
) -> np.ndarray:
    """Retrieves the world extent of the geometry mesh."""
    return points_get_world_extent(bundle, axis_aligned=axis_aligned)


def mesh_get_triangulated_face_vertex_indices(
    bundle: og.BundleContents,
) -> wp.array(dtype=int):
    """Retrieves a triangulated version of the face vertex indices."""
    counts = mesh_get_face_vertex_counts(bundle).numpy()
    if np.all(counts == 3):
        return mesh_get_face_vertex_indices(bundle)

    indices = mesh_get_face_vertex_indices(bundle).numpy()
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


def mesh_clear_dirty_attributes(bundle: og.BundleContents) -> None:
    """Clears the list of attributes that have been updated."""
    attr = _create_dirty_attrs_attr(bundle)
    attr.cpu_value = ""


def mesh_set_dirty_attributes(
    bundle: og.BundleContents,
    flags: MeshAttributeFlags,
) -> None:
    """Sets attributes that have been updated."""
    attr = _create_dirty_attrs_attr(bundle)
    values = attr.cpu_value.split()
    additional_values = []

    if MeshAttributeFlags.POINTS in flags:
        additional_values.append("points")

    if MeshAttributeFlags.NORMALS in flags:
        additional_values.append("normals")

    if MeshAttributeFlags.UVS in flags:
        additional_values.append("primvars:st")

    if MeshAttributeFlags.TOPOLOGY in flags:
        additional_values.extend(("faceVertexCounts", "faceVertexIndices"))

    additional_values = tuple(x for x in additional_values if x not in values)
    values.extend(additional_values)
    attr.cpu_value = " ".join(values)


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


def _create_dirty_attrs_attr(
    bundle: og.BundleContents,
) -> og.RuntimeAttribute:
    """Creates a new dirty attributes attribute into a bundle."""
    return insert_bundle_attr(
        bundle,
        "dirtyAttrs",
        og.Type(
            og.BaseDataType.UCHAR,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.TEXT,
        ),
    )
