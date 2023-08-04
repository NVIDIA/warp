# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Node converting a volume into a geometry mesh."""

import traceback

import omni.graph.core as og
import warp as wp

import omni.warp.nodes
from omni.warp.nodes.ogn.OgnMeshFromVolumeDatabase import OgnMeshFromVolumeDatabase


PROFILING = False


#   Kernels
# ------------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def transform_points_kernel(
    points: wp.array(dtype=wp.vec3),
    center: wp.vec3,
    scale: wp.vec3,
    out_points: wp.array(dtype=wp.vec3),
):
    """Transform the points with the given offset and scale values."""
    tid = wp.tid()

    pos = points[tid]
    pos = pos - center
    pos = wp.cw_mul(pos, scale)
    out_points[tid] = pos


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self.mc = None

        self.is_valid = False

        self.attr_tracking = omni.warp.nodes.AttrTracking(
            (
                "dim1",
                "dim2",
                "dim3",
                "maxPoints",
                "maxTriangles",
            ),
        )

    def needs_initialization(self, db: OgnMeshFromVolumeDatabase) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if not self.is_valid:
            return True

        if self.attr_tracking.have_attrs_changed(db):
            return True

        return False

    def initialize(self, db: OgnMeshFromVolumeDatabase) -> bool:
        """Initializes the internal state."""
        # Initialize Warp's marching cubes helper.
        mc = wp.MarchingCubes(
            int(db.inputs.dim1),
            int(db.inputs.dim2),
            int(db.inputs.dim3),
            db.inputs.maxPoints,
            db.inputs.maxTriangles,
        )

        # Store the class members.
        self.mc = mc

        self.attr_tracking.update_state(db)

        return True


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnMeshFromVolumeDatabase) -> None:
    """Evaluates the node."""
    db.outputs.mesh.changes().activate()

    if not db.inputs.data.memory or db.inputs.data.shape[0] == 0:
        return

    state = db.internal_state

    # Initialize the internal state if it hasn't been already.
    if state.needs_initialization(db):
        if not state.initialize(db):
            return

    dims = (db.inputs.dim1, db.inputs.dim2, db.inputs.dim3)
    size = dims[0] * dims[1] * dims[2]

    if db.inputs.data.shape[0] != size:
        raise RuntimeError(
            "The length of the input array data doesn't match with "
            "the given size: `{} != {}`.".format(db.inputs.data.shape[0], size)
        )

    # Alias the incoming memory to a Warp array.
    data = wp.from_ptr(
        db.inputs.data.memory,
        size,
        dtype=float,
        shape=dims,
    )

    with omni.warp.nodes.NodeTimer("surface_mesh", db, active=PROFILING):
        # Let Warp's marching cubes helper generate the mesh surface at
        # the given ISO threshold value.
        state.mc.surface(data, db.inputs.threshold)

    # The generated surface is triangulated, so we have 3 vertices per face.
    face_count = int(len(state.mc.indices) / 3)
    vertex_count = len(state.mc.indices)
    point_count = len(state.mc.verts)

    if not point_count or not vertex_count or not face_count:
        return

    # Warp's marching cubes helper allocates its own arrays to store
    # the resulting mesh geometry but, eventually, we need to write that data
    # to OmniGraph, so we create a new geometry mesh within the output bundle.
    omni.warp.nodes.mesh_create_bundle(
        db.outputs.mesh,
        point_count,
        vertex_count,
        face_count,
        xform=db.inputs.transform,
    )

    out_points = omni.warp.nodes.mesh_get_points(db.outputs.mesh)
    out_face_vertex_counts = omni.warp.nodes.mesh_get_face_vertex_counts(
        db.outputs.mesh,
    )
    out_face_vertex_indices = omni.warp.nodes.mesh_get_face_vertex_indices(
        db.outputs.mesh,
    )

    # Copy the data to the output geometry mesh bundle.
    wp.copy(out_points, state.mc.verts)
    wp.copy(out_face_vertex_indices, state.mc.indices)

    # Set all faces to be triangles.
    out_face_vertex_counts.fill_(3)

    # Transform the mesh to fit the given center and size values.
    center = (
        dims[0] * 0.5,
        dims[1] * 0.5,
        dims[2] * 0.5,
    )
    scale = (
        db.inputs.size[0] / dims[0],
        db.inputs.size[1] / dims[1],
        db.inputs.size[2] / dims[2],
    )
    with omni.warp.nodes.NodeTimer("transform_points", db, active=PROFILING):
        wp.launch(
            transform_points_kernel,
            dim=point_count,
            inputs=[
                state.mc.verts,
                center,
                scale,
            ],
            outputs=[
                out_points,
            ],
        )


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnMeshFromVolume:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnMeshFromVolumeDatabase) -> None:
        device = wp.get_device("cuda:0")

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            db.internal_state.is_valid = False
            return

        db.internal_state.is_valid = True

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
