# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Sample node deforming a geometry mesh."""

import traceback

import omni.graph.core as og
import warp as wp

import omni.warp.nodes
from omni.warp.nodes.ogn.OgnSampleMeshDeformDatabase import OgnSampleMeshDeformDatabase


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
        device = wp.get_device("cuda:0")

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
