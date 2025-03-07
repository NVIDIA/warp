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

"""Node creating a geometry mesh grid."""

import traceback

import omni.graph.core as og
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnGridCreateDatabase import OgnGridCreateDatabase

import warp as wp

from .kernels.grid_create import grid_create_launch_kernel

PROFILING = False


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self.is_valid = False

        self.attr_tracking = omni.warp.nodes.AttrTracking(
            (
                "transform",
                "size",
                "dims",
            ),
        )


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnGridCreateDatabase) -> None:
    """Evaluates the node."""
    db.outputs.mesh.changes().activate()

    if not db.outputs.mesh.valid:
        return

    state = db.per_instance_state

    if state.is_valid and not state.attr_tracking.have_attrs_changed(db):
        return

    # Compute the mesh's topology counts.
    face_count = db.inputs.dims[0] * db.inputs.dims[1]
    vertex_count = face_count * 4
    point_count = (db.inputs.dims[0] + 1) * (db.inputs.dims[1] + 1)

    # Create a new geometry mesh within the output bundle.
    omni.warp.nodes.mesh_create_bundle(
        db.outputs.mesh,
        point_count,
        vertex_count,
        face_count,
        xform=db.inputs.transform,
        create_normals=True,
        create_uvs=True,
    )

    with omni.warp.nodes.NodeTimer("grid_create", db, active=PROFILING):
        # Evaluate the kernel.
        grid_create_launch_kernel(
            omni.warp.nodes.mesh_get_points(db.outputs.mesh),
            omni.warp.nodes.mesh_get_face_vertex_counts(db.outputs.mesh),
            omni.warp.nodes.mesh_get_face_vertex_indices(db.outputs.mesh),
            omni.warp.nodes.mesh_get_normals(db.outputs.mesh),
            omni.warp.nodes.mesh_get_uvs(db.outputs.mesh),
            db.inputs.size.tolist(),
            db.inputs.dims.tolist(),
        )

    state.attr_tracking.update_state(db)


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnGridCreate:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnGridCreateDatabase) -> None:
        device = omni.warp.nodes.device_get_cuda_compute()

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            db.per_instance_state.is_valid = False
            return

        db.per_instance_state.is_valid = True

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
