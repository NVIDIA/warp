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

"""Node creating a grid geometry simulated with a wave equation solver."""

import traceback
from math import sqrt

import numpy as np
import omni.graph.core as og
import omni.timeline
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnWaveSolveDatabase import OgnWaveSolveDatabase

import warp as wp

from .kernels.grid_create import grid_create_launch_kernel

PROFILING = False


#   Kernels
# -----------------------------------------------------------------------------


@wp.func
def sample_height(
    height_map: wp.array(dtype=float),
    x: int,
    z: int,
    point_count_x: int,
    point_count_z: int,
):
    # Clamp to the grid's bounds.
    x = wp.clamp(x, 0, point_count_x - 1)
    z = wp.clamp(z, 0, point_count_z - 1)
    return height_map[z * point_count_x + x]


@wp.func
def laplacian(
    height_map: wp.array(dtype=float),
    x: int,
    z: int,
    point_count_x: int,
    point_count_z: int,
):
    # See https://en.wikipedia.org/wiki/Wave_equation.
    ddx = (
        sample_height(height_map, x + 1, z, point_count_x, point_count_z)
        - sample_height(height_map, x, z, point_count_x, point_count_z) * 2.0
        + sample_height(height_map, x - 1, z, point_count_x, point_count_z)
    )
    ddz = (
        sample_height(height_map, x, z + 1, point_count_x, point_count_z)
        - sample_height(height_map, x, z, point_count_x, point_count_z) * 2.0
        + sample_height(height_map, x, z - 1, point_count_x, point_count_z)
    )
    return ddx + ddz


@wp.kernel(enable_backward=False)
def displace_kernel(
    point_count_x: int,
    center_x: float,
    center_z: float,
    radius: float,
    amplitude: float,
    time: float,
    out_height_map_0: wp.array(dtype=float),
    out_height_map_1: wp.array(dtype=float),
):
    tid = wp.tid()

    x = tid % point_count_x
    z = tid // point_count_x

    dx = float(x) - center_x
    dz = float(z) - center_z

    dist_sq = float(dx * dx + dz * dz)

    if dist_sq < radius * radius:
        height = amplitude * wp.sin(time)
        out_height_map_0[tid] = height
        out_height_map_1[tid] = height


@wp.kernel(enable_backward=False)
def simulate_kernel(
    point_count_x: int,
    point_count_z: int,
    inv_cell_size: float,
    speed: float,
    damping: float,
    dt: float,
    height_map_1: wp.array(dtype=float),
    out_height_map_0: wp.array(dtype=float),
):
    tid = wp.tid()

    x = tid % point_count_x
    z = tid // point_count_x

    d = laplacian(height_map_1, x, z, point_count_x, point_count_z)
    d *= inv_cell_size * inv_cell_size

    # Integrate and write the result in the 'previous' height map buffer since
    # it will be then swapped to become the 'current' one.
    h0 = out_height_map_0[tid]
    h1 = height_map_1[tid]
    out_height_map_0[tid] = h1 * 2.0 - h0 + (d * speed - (h1 - h0) * damping) * dt * dt


@wp.kernel(enable_backward=False)
def update_mesh_kernel(
    height_map: wp.array(dtype=float),
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    height = height_map[tid]
    pos = out_points[tid]

    out_points[tid] = wp.vec3(pos[0], height, pos[2])


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self.height_map_0 = None
        self.height_map_1 = None

        self.time = 0.0

        self.is_valid = False

        self.attr_tracking = omni.warp.nodes.AttrTracking(
            (
                "size",
                "cellSize",
            ),
        )

    def needs_initialization(self, db: OgnWaveSolveDatabase) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if not self.is_valid:
            return True

        if self.attr_tracking.have_attrs_changed(db):
            return True

        if db.inputs.time < self.time:
            # Reset the simulation when we're rewinding.
            return True

        return False

    def initialize(
        self,
        db: OgnWaveSolveDatabase,
        dims: np.ndarray,
    ) -> bool:
        """Initializes the internal state."""
        point_count = omni.warp.nodes.mesh_get_point_count(db.outputs.mesh)

        # Initialize a double buffering for the height map.
        height_map_0 = wp.zeros(point_count, dtype=float)
        height_map_1 = wp.zeros(point_count, dtype=float)

        # Build the grid mesh.
        grid_create_launch_kernel(
            omni.warp.nodes.mesh_get_points(db.outputs.mesh),
            omni.warp.nodes.mesh_get_face_vertex_counts(db.outputs.mesh),
            omni.warp.nodes.mesh_get_face_vertex_indices(db.outputs.mesh),
            omni.warp.nodes.mesh_get_normals(db.outputs.mesh),
            omni.warp.nodes.mesh_get_uvs(db.outputs.mesh),
            db.inputs.size.tolist(),
            dims.tolist(),
            update_topology=True,
        )

        # Store the class members.
        self.height_map_0 = height_map_0
        self.height_map_1 = height_map_1

        self.attr_tracking.update_state(db)

        return True


#   Compute
# ------------------------------------------------------------------------------


def displace(
    db: OgnWaveSolveDatabase,
    dims: np.ndarray,
    cell_size: np.ndarray,
) -> None:
    """Displaces the height map with the collider."""
    state = db.per_instance_state

    # Retrieve some data from the grid mesh.
    xform = omni.warp.nodes.bundle_get_world_xform(db.outputs.mesh)

    # Retrieve some data from the collider mesh.
    collider_xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.collider)
    collider_extent = omni.warp.nodes.mesh_get_world_extent(
        db.inputs.collider,
        axis_aligned=True,
    )

    try:
        xform_inv = np.linalg.inv(xform)
    except np.linalg.LinAlgError:
        # On the first run, OG sometimes return an invalid matrix,
        # so we default it to the identity one.
        xform_inv = np.identity(4)

    # Retrieve the collider's position in the grid's object space.
    collider_pos = np.pad(collider_xform[3][:3], (0, 1), constant_values=1)
    collider_pos = np.dot(xform_inv.T, collider_pos)

    # Compute the collider's radius.
    collider_radius = np.amax(collider_extent[1] - collider_extent[0]) * 0.5

    # Determine the point around which the grid will be displaced.
    center_x = (dims[0] + 1) * 0.5 - float(collider_pos[0]) / cell_size[0]
    center_z = (dims[1] + 1) * 0.5 - float(collider_pos[2]) / cell_size[1]

    # Clamp the deformation center to the grid's bounds.
    center_x = max(0, min(dims[0], center_x))
    center_z = max(0, min(dims[1], center_z))

    # Apply the displacement if the collider is in contact with the grid.
    contact_radius_sq = (collider_radius**2) - (abs(collider_pos[1]) ** 2)
    if contact_radius_sq > 0:
        cell_size_uniform = (cell_size[0] + cell_size[1]) * 0.5
        center_radius = sqrt(contact_radius_sq) / cell_size_uniform
        wp.launch(
            kernel=displace_kernel,
            dim=omni.warp.nodes.mesh_get_point_count(db.outputs.mesh),
            inputs=[
                dims[0] + 1,
                center_x,
                center_z,
                center_radius,
                db.inputs.amplitude,
                db.inputs.time,
            ],
            outputs=[
                state.height_map_0,
                state.height_map_1,
            ],
        )


def simulate(
    db: OgnWaveSolveDatabase,
    dims: np.ndarray,
    cell_size: np.ndarray,
    sim_dt: bool,
) -> None:
    """Solves the wave simulation."""
    state = db.per_instance_state

    cell_size_uniform = (cell_size[0] + cell_size[1]) * 0.5
    wp.launch(
        kernel=simulate_kernel,
        dim=omni.warp.nodes.mesh_get_point_count(db.outputs.mesh),
        inputs=[
            dims[0] + 1,
            dims[1] + 1,
            1.0 / cell_size_uniform,
            db.inputs.speed,
            db.inputs.damping,
            sim_dt,
            state.height_map_1,
        ],
        outputs=[
            state.height_map_0,
        ],
    )

    # Swap the height map buffers
    state.height_map_0, state.height_map_1 = (
        state.height_map_1,
        state.height_map_0,
    )


def update_mesh(db: OgnWaveSolveDatabase) -> None:
    """Updates the output grid mesh."""
    state = db.per_instance_state

    wp.launch(
        kernel=update_mesh_kernel,
        dim=omni.warp.nodes.mesh_get_point_count(db.outputs.mesh),
        inputs=[
            state.height_map_1,
        ],
        outputs=[
            omni.warp.nodes.mesh_get_points(db.outputs.mesh),
        ],
    )


def compute(db: OgnWaveSolveDatabase) -> None:
    """Evaluates the node."""
    db.outputs.mesh.changes().activate()

    if not db.outputs.mesh.valid:
        return

    state = db.per_instance_state

    # Compute the number of divisions.
    dims = (db.inputs.size / db.inputs.cellSize).astype(int)

    # Compute the mesh's topology counts.
    face_count = dims[0] * dims[1]
    vertex_count = face_count * 4
    point_count = (dims[0] + 1) * (dims[1] + 1)

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

    if state.needs_initialization(db):
        # Initialize the internal state if it hasn't been already.
        if not state.initialize(db, dims):
            return
    else:
        # We skip the simulation if it has just been initialized.

        # Retrieve the simulation's delta time.
        timeline = omni.timeline.get_timeline_interface()
        sim_rate = timeline.get_ticks_per_second()
        sim_dt = 1.0 / sim_rate

        # Infer the size of each cell from the overall grid size and the number
        # of dimensions.
        cell_size = db.inputs.size / dims

        if db.inputs.collider.valid:
            with omni.warp.nodes.NodeTimer("displace", db, active=PROFILING):
                # Deform the grid with a displacement value if the collider
                # is in contact with it.
                displace(db, dims, cell_size)

        with omni.warp.nodes.NodeTimer("simulate", db, active=PROFILING):
            # Simulate the ripples using the wave equation.
            simulate(db, dims, cell_size, sim_dt)

        with omni.warp.nodes.NodeTimer("update_mesh", db, active=PROFILING):
            # Update the mesh points with the height map resulting from
            # the displacement and simulation steps.
            update_mesh(db)

    # Store the current time.
    state.time = db.inputs.time


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnWaveSolve:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnWaveSolveDatabase) -> None:
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
