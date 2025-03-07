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

"""Sample node deforming a geometry mesh."""

import traceback

import omni.graph.core as og
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnSampleMeshDeformDatabase import OgnSampleMeshDeformDatabase

import warp as wp

PROFILING = False


#   Kernels
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def deform_mesh_kernel(
    points: wp.array(dtype=wp.vec3),
    time: float,
    out_points: wp.array(dtype=wp.vec3),
):
    """Kernel to deform a geometry mesh."""
    tid = wp.tid()

    pos = points[tid]
    displacement = wp.vec3(0.0, wp.sin(time + pos[0] * 0.1) * 10.0, 0.0)
    out_points[tid] = pos + displacement


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnSampleMeshDeformDatabase) -> None:
    """Evaluates the node."""
    if not db.inputs.mesh.valid or not db.outputs.mesh.valid:
        return

    # Copy the input geometry mesh bundle and read its contents.
    db.outputs.mesh = db.inputs.mesh

    # Retrieve the input and output point data.
    points = omni.warp.nodes.mesh_get_points(db.inputs.mesh)
    out_points = omni.warp.nodes.mesh_get_points(db.outputs.mesh)

    with omni.warp.nodes.NodeTimer("deform_mesh", db, active=PROFILING):
        # Evaluate the kernel once per point.
        wp.launch(
            kernel=deform_mesh_kernel,
            dim=len(points),
            inputs=[
                points,
                db.inputs.time,
            ],
            outputs=[
                out_points,
            ],
        )


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnSampleMeshDeform:
    """Node."""

    @staticmethod
    def compute(db: OgnSampleMeshDeformDatabase) -> None:
        device = omni.warp.nodes.device_get_cuda_compute()

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
