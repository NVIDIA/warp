# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Public Python API exposed by the omni.warp extension."""

__all__ = [
    "NodeTimer",
    "define_prim_attrs",
    "from_omni_graph",
    "get_child_bundle_count",
    "get_prim_type",
    "get_world_xform",
    "get_world_xform_from_prim",
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
    "points_create_bundle",
    "points_get_local_extent",
    "points_get_masses",
    "points_get_point_count",
    "points_get_points",
    "points_get_velocities",
    "points_get_widths",
    "points_get_world_extent",
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

from .scripts.attributes import cast_array_attr_value_to_warp

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


def get_world_xform_from_prim(prim_path: Optional[str]) -> None:
    """Retrieves the world transformation matrix from a USD primitive."""
    if prim_path is not None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
            prim = UsdGeom.Xformable(prim)
            return prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    return np.identity(4)


#   Bundle Attributes
# ------------------------------------------------------------------------------


def get_child_bundle_count(
    bundle: og.BundleContents,
) -> int:
    """Retrieves the number of children defined for a bundle."""
    return bundle.bundle.get_child_bundle_count()


def get_prim_type(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> str:
    """Retrieves the primitive type."""
    attr = _get_bundle_attr_by_name(bundle, "sourcePrimType", child_bundle_idx)
    return _get_cpu_array(attr)


def get_world_xform(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> np.ndarray:
    """Retrieves the world transformation matrix."""
    attr = _get_bundle_attr_by_name(bundle, "worldMatrix", child_bundle_idx)
    if attr is None:
        return np.identity(4)

    return _get_cpu_array(attr).reshape(4, 4)


def define_prim_attrs(
    bundle: og.BundleContents,
    prim_type: str,
    xform_prim_path: Optional[str] = None,
    child_bundle_idx: int = 0,
) -> None:
    """Defines the primitive attributes."""
    child_bundle = _create_child_bundle(bundle, child_bundle_idx)
    xform = get_world_xform_from_prim(xform_prim_path)

    prim_type_attr = child_bundle.create_attribute(
        "sourcePrimType",
        og.Type(
            og.BaseDataType.TOKEN,
            tuple_count=1,
            array_depth=0,
            role=og.AttributeRole.NONE,
        ),
    )
    _set_cpu_array(prim_type_attr, prim_type)

    world_matrix_attr = child_bundle.create_attribute(
        "worldMatrix",
        og.Type(
            og.BaseDataType.DOUBLE,
            tuple_count=16,
            array_depth=0,
            role=og.AttributeRole.MATRIX,
        ),
    )
    _set_cpu_array(world_matrix_attr, xform)


#   Point Cloud Geometries
# ------------------------------------------------------------------------------


def points_create_bundle(
    bundle: og.BundleContents,
    point_count: int,
    child_bundle_idx: int = 0,
) -> None:
    """Creates and initializes point cloud attributes within a bundle."""
    child_bundle = _create_child_bundle(bundle, child_bundle_idx)
    child_bundle.create_attribute(
        "points",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.POSITION,
        ),
        element_count=point_count,
    )
    child_bundle.create_attribute(
        "velocities",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.VECTOR,
        ),
        element_count=point_count,
    )
    child_bundle.create_attribute(
        "widths",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
        element_count=point_count,
    )
    child_bundle.create_attribute(
        "masses",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
        element_count=point_count,
    )


def points_get_point_count(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> int:
    """Retrieves the number of points."""
    return _get_bundle_attr_by_name(bundle, "points", child_bundle_idx).size()


def points_get_points(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle points attribute as a Warp array."""
    attr = _get_bundle_attr_by_name(bundle, "points", child_bundle_idx)
    return _get_gpu_array(attr, wp.vec3, read_only=bundle.read_only)


def points_get_velocities(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle velocities attribute as a Warp array."""
    attr = _get_bundle_attr_by_name(bundle, "velocities", child_bundle_idx)
    return _get_gpu_array(attr, wp.vec3, read_only=bundle.read_only)


def points_get_widths(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=float):
    """Retrieves the bundle widths attribute as a Warp array."""
    attr = _get_bundle_attr_by_name(bundle, "widths", child_bundle_idx)
    return _get_gpu_array(attr, float, read_only=bundle.read_only)


def points_get_masses(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=float):
    """Retrieves the bundle masses attribute as a Warp array."""
    attr = _get_bundle_attr_by_name(bundle, "masses", child_bundle_idx)
    return _get_gpu_array(attr, float, read_only=bundle.read_only)


def points_get_local_extent(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> np.ndarray:
    """Retrieves the local extent of the geometry points."""
    # Some standard workflows include a single 'extent' attribute when defining
    # geometry primitives on the stage.
    attr = _get_bundle_attr_by_name(bundle, "extent", child_bundle_idx)
    if attr is not None:
        return _get_cpu_array(attr)

    # Alternatively, the ReadPrims node offers an option to compute the bounding
    # box which results in a triple of 'bboxMinCorner', 'bboxMaxCorner',
    # and 'bboxTransform' attributes.
    min_attr = _get_bundle_attr_by_name(bundle, "bboxMinCorner", child_bundle_idx)
    max_attr = _get_bundle_attr_by_name(bundle, "bboxMaxCorner", child_bundle_idx)
    if min_attr is not None and max_attr is not None:
        return np.stack(
            (
                _get_cpu_array(min_attr),
                _get_cpu_array(max_attr),
            ),
        )

    # The last resort is to compute the extent ourselves from
    # the point positions.
    points = points_get_points(bundle, child_bundle_idx=child_bundle_idx)
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
    child_bundle_idx: int = 0,
) -> np.ndarray:
    """Retrieves the world extent of the geometry points."""
    extent = points_get_local_extent(bundle, child_bundle_idx=child_bundle_idx)
    xform = get_world_xform(bundle, child_bundle_idx=child_bundle_idx)

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


#   Mesh Geometries
# ------------------------------------------------------------------------------


def mesh_create_bundle(
    bundle: og.BundleContents,
    point_count: int,
    vertex_count: int,
    face_count: int,
    child_bundle_idx: int = 0,
) -> None:
    """Creates and initializes mesh attributes within a bundle."""
    child_bundle = _create_child_bundle(bundle, child_bundle_idx)
    child_bundle.create_attribute(
        "points",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.POSITION,
        ),
        element_count=point_count,
    )
    child_bundle.create_attribute(
        "normals",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.NORMAL,
        ),
        element_count=vertex_count,
    )
    child_bundle.create_attribute(
        "primvars:st",
        og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=2,
            array_depth=1,
            role=og.AttributeRole.TEXCOORD,
        ),
        element_count=vertex_count,
    )
    child_bundle.create_attribute(
        "faceVertexCounts",
        og.Type(
            og.BaseDataType.INT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
        element_count=face_count,
    )
    child_bundle.create_attribute(
        "faceVertexIndices",
        og.Type(
            og.BaseDataType.INT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
        element_count=vertex_count,
    )


def mesh_get_point_count(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> int:
    """Retrieves the number of points."""
    return _get_bundle_attr_by_name(bundle, "points", child_bundle_idx).size()


def mesh_get_vertex_count(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> int:
    """Retrieves the number of vertices."""
    attr = _get_bundle_attr_by_name(bundle, "faceVertexCounts", child_bundle_idx)
    return int(np.sum(_get_cpu_array(attr)))


def mesh_get_face_count(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> int:
    """Retrieves the number of faces."""
    return _get_bundle_attr_by_name(bundle, "faceVertexCounts", child_bundle_idx).size()


def mesh_get_points(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle points attribute as a Warp array."""
    return points_get_points(bundle, child_bundle_idx=child_bundle_idx)


def mesh_get_velocities(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle velocities attribute as a Warp array."""
    return points_get_velocities(bundle, child_bundle_idx=child_bundle_idx)


def mesh_get_normals(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=wp.vec3):
    """Retrieves the bundle normals attribute as a Warp array."""
    attr = _get_bundle_attr_by_name(bundle, "normals", child_bundle_idx)
    return _get_gpu_array(attr, wp.vec3, read_only=bundle.read_only)


def mesh_get_uvs(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=wp.vec2):
    """Retrieves the bundle UVs attribute as a Warp array."""
    attr = _get_bundle_attr_by_name(bundle, "primvars:st", child_bundle_idx)
    return _get_gpu_array(attr, wp.vec2, read_only=bundle.read_only)


def mesh_get_face_vertex_counts(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=int):
    """Retrieves the bundle face vertex counts attribute as a Warp array."""
    attr = _get_bundle_attr_by_name(bundle, "faceVertexCounts", child_bundle_idx)
    return _get_gpu_array(attr, int, read_only=bundle.read_only)


def mesh_get_face_vertex_indices(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=int):
    """Retrieves the bundle face vertex indices attribute as a Warp array."""
    attr = _get_bundle_attr_by_name(bundle, "faceVertexIndices", child_bundle_idx)
    return _get_gpu_array(attr, int, read_only=bundle.read_only)


def mesh_get_local_extent(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> np.ndarray:
    """Retrieves the local extent of the geometry mesh."""
    return points_get_local_extent(bundle, child_bundle_idx=child_bundle_idx)


def mesh_get_world_extent(
    bundle: og.BundleContents,
    axis_aligned: bool = False,
    child_bundle_idx: int = 0,
) -> np.ndarray:
    """Retrieves the world extent of the geometry mesh."""
    return points_get_world_extent(
        bundle,
        axis_aligned=axis_aligned,
        child_bundle_idx=child_bundle_idx,
    )


def mesh_get_triangulated_face_vertex_indices(
    bundle: og.BundleContents,
    child_bundle_idx: int = 0,
) -> wp.array(dtype=int):
    """Retrieves a triangulated version of the face vertex indices."""
    counts = mesh_get_face_vertex_counts(bundle, child_bundle_idx=child_bundle_idx).numpy()
    if np.all(counts == 3):
        return mesh_get_face_vertex_indices(bundle, child_bundle_idx=child_bundle_idx)

    indices = mesh_get_face_vertex_indices(bundle, child_bundle_idx=child_bundle_idx).numpy()
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


def _create_child_bundle(
    bundle: og.BundleContents,
    child_bundle_idx: int,
) -> og.IBundle2:
    """Creates a single child bundle if it doesn't already exist."""
    if child_bundle_idx < bundle.bundle.get_child_bundle_count():
        return bundle.bundle.get_child_bundle(child_bundle_idx)

    return bundle.bundle.create_child_bundle("prim{}".format(child_bundle_idx))


def _get_bundle_attr_by_name(
    bundle: og.BundleContents,
    name: str,
    child_bundle_idx: int,
) -> Optional[og.AttributeData]:
    """Retrieves a bundle attribute from its name."""
    if bundle.bundle.get_child_bundle_count():
        attr = bundle.bundle.get_child_bundle(child_bundle_idx).get_attribute_by_name(name)
    else:
        attr = bundle.bundle.get_attribute_by_name(name)

    if not attr.is_valid():
        return None

    return attr


def _get_gpu_array(
    attr: og.AttributeData,
    dtype: type,
    read_only: bool = True,
) -> wp.array:
    """Retrieves the value of an array attribute living on the GPU."""
    attr.gpu_ptr_kind = og.PtrToPtrKind.CPU
    (ptr, _) = attr.get_array(
        on_gpu=True,
        get_for_write=not read_only,
        reserved_element_count=0 if read_only else attr.size(),
    )
    return wp.from_ptr(ptr, attr.size(), dtype=dtype)


def _get_cpu_array(
    attr: og.AttributeData,
    read_only: bool = True,
) -> Union[np.ndarray, str]:
    """Retrieves the value of an array attribute living on the CPU."""
    return attr.get_array(
        on_gpu=False,
        get_for_write=not read_only,
        reserved_element_count=0 if read_only else attr.size(),
    )


def _set_cpu_array(attr: og.AttributeData, value: Sequence) -> None:
    """Sets the given value onto an array attribute living on the CPU."""
    attr.set(value, on_gpu=False)
