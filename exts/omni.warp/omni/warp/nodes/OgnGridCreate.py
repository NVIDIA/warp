# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Node creating a geometry mesh grid."""

import traceback

import numpy as np
import omni.graph.core as og
import warp as wp

import omni.warp
from omni.warp.ogn.OgnGridCreateDatabase import OgnGridCreateDatabase
from omni.warp.scripts.kernels.grid_create import grid_create_launch_kernel


PROFILING = False


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self.prim_path = None
        self.center = None
        self.size = None
        self.dims = None

        self.is_valid = False


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnGridCreateDatabase) -> None:
    """Evaluates the node."""
    if not db.outputs.mesh.valid:
        return

    state = db.internal_state

    # Set the USD primitive path and type.
    omni.warp.define_prim_attrs(
        db.outputs.mesh,
        db.inputs.sourcePrimPath,
        "Mesh",
    )

    # Query the geometry components that need to be updated.
    if (
        not state.is_valid
        or db.inputs.sourcePrimPath != state.prim_path
        or not np.array_equal(db.inputs.dims, state.dims)
    ):
        updates = omni.warp.MeshAttributeFlags.ALL
    elif not np.array_equal(db.inputs.center, state.center) or not np.array_equal(db.inputs.size, state.size):
        updates = omni.warp.MeshAttributeFlags.POINTS
    else:
        updates = omni.warp.MeshAttributeFlags.NONE

    if updates == omni.warp.MeshAttributeFlags.NONE:
        # If no update needs to be done, there ought to be an output bundle
        # already existing, so we read it and notify downstrean nodes that
        # nothing changed.
        omni.warp.mesh_clear_dirty_attributes(db.outputs.mesh)
        omni.warp.mesh_set_dirty_attributes(db.outputs.mesh, updates)
        return

    # Compute the mesh's topology counts.
    face_count = db.inputs.dims[0] * db.inputs.dims[1]
    vertex_count = face_count * 4
    point_count = (db.inputs.dims[0] + 1) * (db.inputs.dims[1] + 1)

    # Create a new geometry mesh within the output bundle.
    omni.warp.mesh_create_bundle(
        db.outputs.mesh,
        point_count,
        vertex_count,
        face_count,
    )

    with omni.warp.NodeTimer("grid_create", db, active=PROFILING):
        # Evaluate the kernel.
        grid_create_launch_kernel(
            omni.warp.mesh_get_points(db.outputs.mesh),
            omni.warp.mesh_get_face_vertex_counts(db.outputs.mesh),
            omni.warp.mesh_get_face_vertex_indices(db.outputs.mesh),
            omni.warp.mesh_get_normals(db.outputs.mesh),
            omni.warp.mesh_get_uvs(db.outputs.mesh),
            db.inputs.center.tolist(),
            db.inputs.size.tolist(),
            db.inputs.dims.tolist(),
            update_topology=omni.warp.MeshAttributeFlags.TOPOLOGY in updates,
        )

    # Notify downstream nodes of updates done to the geometry.
    omni.warp.mesh_clear_dirty_attributes(db.outputs.mesh)
    omni.warp.mesh_set_dirty_attributes(db.outputs.mesh, updates)

    # Cache the node attribute values relevant to this internal state.
    # They're the ones used to check whether the geometry needs
    # to be updated or not.
    state.prim_path = db.inputs.sourcePrimPath
    state.center = db.inputs.center.copy()
    state.size = db.inputs.size.copy()
    state.dims = db.inputs.dims.copy()


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnGridCreate:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnGridCreateDatabase) -> None:
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
