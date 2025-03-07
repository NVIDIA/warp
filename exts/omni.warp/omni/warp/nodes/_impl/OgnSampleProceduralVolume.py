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

"""Sample node generating a procedural volume."""

import traceback

import omni.graph.core as og
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnSampleProceduralVolumeDatabase import OgnSampleProceduralVolumeDatabase

import warp as wp

MIN_RES = 8

PROFILING = False

#   Kernels
# ------------------------------------------------------------------------------


@wp.func
def sdf_create_box(pos: wp.vec3, size: wp.vec3):
    """Creates a SDF box primitive."""
    # https://iquilezles.org/articles/distfunctions
    q = wp.vec3(
        wp.abs(pos[0]) - size[0],
        wp.abs(pos[1]) - size[1],
        wp.abs(pos[2]) - size[2],
    )
    qp = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    return wp.length(qp) + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)


@wp.func
def sdf_create_torus(pos: wp.vec3, major_radius: float, minor_radius: float):
    """Creates a SDF torus primitive."""
    # https://iquilezles.org/articles/distfunctions
    q = wp.vec2(wp.length(wp.vec2(pos[0], pos[2])) - major_radius, pos[1])
    return wp.length(q) - minor_radius


@wp.func
def sdf_translate(pos: wp.vec3, offset: wp.vec3):
    """Translates a SDF position vector with an offset."""
    return pos - offset


@wp.func
def sdf_rotate(pos: wp.vec3, angles: wp.vec3):
    """Rotates a SDF position vector using Euler angles."""
    rot = wp.quat_rpy(
        wp.radians(angles[0]),
        wp.radians(angles[1]),
        wp.radians(angles[2]),
    )
    return wp.quat_rotate_inv(rot, pos)


@wp.func
def sdf_smooth_min(a: float, b: float, radius: float):
    """Creates a SDF torus primitive."""
    # https://iquilezles.org/articles/smin
    h = wp.max(radius - wp.abs(a - b), 0.0) / radius
    return wp.min(a, b) - h * h * h * radius * (1.0 / 6.0)


@wp.kernel(enable_backward=False)
def generate_volume_kernel(
    torus_altitude: float,
    torus_major_radius: float,
    torus_minor_radius: float,
    smooth_min_radius: float,
    dim: int,
    time: float,
    out_data: wp.array3d(dtype=float),
):
    """Kernel to generate a SDF volume based on primitives."""
    i, j, k = wp.tid()

    # Retrieve the position of the current cell in a normalized [-1, 1] range
    # for each dimension.
    pos = wp.vec3(
        2.0 * ((float(i) + 0.5) / float(dim)) - 1.0,
        2.0 * ((float(j) + 0.5) / float(dim)) - 1.0,
        2.0 * ((float(k) + 0.5) / float(dim)) - 1.0,
    )

    box = sdf_create_box(
        sdf_translate(pos, wp.vec3(0.0, -0.7, 0.0)),
        wp.vec3(0.9, 0.3, 0.9),
    )
    torus = sdf_create_torus(
        sdf_rotate(
            sdf_translate(pos, wp.vec3(0.0, torus_altitude, 0.0)),
            wp.vec3(wp.sin(time) * 90.0, wp.cos(time) * 45.0, 0.0),
        ),
        torus_major_radius,
        torus_minor_radius,
    )
    out_data[i, j, k] = sdf_smooth_min(box, torus, smooth_min_radius)


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnSampleProceduralVolumeDatabase) -> None:
    """Evaluates the node."""
    # Enforce a minimum dimension or else nothing's left to draw.
    dim = max(db.inputs.dim, MIN_RES)

    db.outputs.data_size = dim * dim * dim
    data = omni.warp.nodes.from_omni_graph(
        db.outputs.data,
        dtype=float,
        shape=(dim, dim, dim),
    )

    with omni.warp.nodes.NodeTimer("generate_volume", db, active=PROFILING):
        wp.launch(
            kernel=generate_volume_kernel,
            dim=data.shape,
            inputs=[
                db.inputs.torusAltitude,
                db.inputs.torusMajorRadius,
                db.inputs.torusMinorRadius,
                db.inputs.smoothMinRadius,
                dim,
                db.inputs.time,
            ],
            outputs=[
                data,
            ],
        )


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnSampleProceduralVolume:
    """Node."""

    @staticmethod
    def compute(db: OgnSampleProceduralVolumeDatabase) -> None:
        device = omni.warp.nodes.device_get_cuda_compute()

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
