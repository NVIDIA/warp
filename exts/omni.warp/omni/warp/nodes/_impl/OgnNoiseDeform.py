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

"""Node deforming points using a noise."""

import hashlib
import traceback

import omni.graph.core as og
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnNoiseDeformDatabase import OgnNoiseDeformDatabase

import warp as wp

USE_GRAPH = True

PROFILING = False

FUNC_PERLIN = wp.constant(0)
FUNC_CURL = wp.constant(1)

FUNC_MAPPING = {
    "perlin": FUNC_PERLIN,
    "curl": FUNC_CURL,
}
UP_AXIS_MAPPING = {
    "+X": (0, 1.0),
    "+Y": (1, 1.0),
    "+Z": (2, 1.0),
    "-X": (0, -1.0),
    "-Y": (1, -1.0),
    "-Z": (2, -1.0),
}


#   Kernels
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def deform_noise_kernel(
    points: wp.array(dtype=wp.vec3),
    partial: bool,
    axis: int,
    axis_sign: float,
    falloff_begin: float,
    falloff_end: float,
    falloff: float,
    func: int,
    cell_size: float,
    offset: float,
    amplitude: wp.vec3,
    seed: wp.uint32,
    out_points: wp.array(dtype=wp.vec3),
):
    """Kernel to deform points using a noise."""
    tid = wp.tid()

    seed = wp.rand_init(int(seed))

    pos = points[tid]
    noise_pos = wp.vec3(pos / cell_size)

    if func == FUNC_PERLIN:
        displacement = wp.vec3(
            wp.noise(
                seed,
                wp.vec4(
                    noise_pos[0],
                    noise_pos[1],
                    noise_pos[2],
                    offset,
                ),
            ),
            wp.noise(
                seed,
                wp.vec4(
                    noise_pos[0],
                    noise_pos[1],
                    noise_pos[2],
                    offset + 1234.5,
                ),
            ),
            wp.noise(
                seed,
                wp.vec4(
                    noise_pos[0],
                    noise_pos[1],
                    noise_pos[2],
                    offset + 6789.0,
                ),
            ),
        )
    elif func == FUNC_CURL:
        displacement = wp.curlnoise(
            seed,
            wp.vec4(
                noise_pos[0],
                noise_pos[1],
                noise_pos[2],
                offset,
            ),
        )

    if partial:
        if falloff < 1e-3:
            if (pos[axis] - falloff_begin) * axis_sign > 0:
                influence = 1.0
            else:
                influence = 0.0
        else:
            if axis_sign < 0.0:
                dist = wp.clamp(pos[axis], falloff_end, falloff_begin)
            else:
                dist = wp.clamp(pos[axis], falloff_begin, falloff_end)

            influence = axis_sign * (dist - falloff_begin) / falloff
    else:
        influence = 1.0

    displacement[0] *= amplitude[0] * influence
    displacement[1] *= amplitude[1] * influence
    displacement[2] *= amplitude[2] * influence
    out_points[tid] = pos + displacement


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnNoiseDeformDatabase) -> None:
    """Evaluates the node."""
    # Copy the input primitives bundle.
    db.outputs.prims = db.inputs.prims

    partial = db.inputs.mode == "partial"
    func = FUNC_MAPPING[db.inputs.func]
    (axis, axis_sign) = UP_AXIS_MAPPING[db.inputs.upAxis]
    falloff_begin = db.inputs.base * axis_sign
    falloff_end = (db.inputs.base + db.inputs.falloff) * axis_sign
    time_offset = db.inputs.time * db.inputs.speed
    amplitude = db.inputs.axisAmplitude * db.inputs.amplitude

    prim_count = omni.warp.nodes.bundle_get_child_count(db.inputs.prims)
    for i in range(prim_count):
        # Retrieve the input and output point data.
        in_points = omni.warp.nodes.mesh_get_points(
            db.inputs.prims,
            child_idx=i,
        )
        out_points = omni.warp.nodes.mesh_get_points(
            db.outputs.prims,
            child_idx=i,
        )

        # Compute a unique seed for the given primitive by hashing its path.
        # We cannot directly use the child index since bundle child ordering
        # is currently not guaranteed and can change between sessions, which makes
        # the result non-deterministic.
        prim_path_attr = omni.warp.nodes.bundle_get_attr(db.inputs.prims, "sourcePrimPath", i)
        prim_path = prim_path_attr.get(on_gpu=False)
        prim_seed = int(hashlib.md5(prim_path.encode("utf-8")).hexdigest(), 16)

        # Evaluate the kernel once per point.
        wp.launch(
            deform_noise_kernel,
            dim=len(in_points),
            inputs=(
                in_points,
                partial,
                axis,
                axis_sign,
                falloff_begin,
                falloff_end,
                db.inputs.falloff,
                func,
                db.inputs.cellSize,
                time_offset,
                amplitude,
                db.inputs.seed + prim_seed * 1234,
            ),
            outputs=(out_points,),
        )


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnNoiseDeform:
    """Node."""

    @staticmethod
    def compute(db: OgnNoiseDeformDatabase) -> None:
        device = omni.warp.nodes.device_get_cuda_compute()

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
