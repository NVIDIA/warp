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

"""Node generating particles inside a mesh."""

import traceback
from typing import Tuple

import numpy as np
import omni.graph.core as og
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnParticlesFromMeshDatabase import OgnParticlesFromMeshDatabase

import warp as wp

PROFILING = False


#   Kernels
# ------------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def transform_points_kernel(
    points: wp.array(dtype=wp.vec3),
    xform: wp.mat44,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_points[tid] = wp.transform_point(xform, points[tid])


@wp.kernel(enable_backward=False)
def sample_mesh_kernel(
    mesh: wp.uint64,
    grid_lower_bound: wp.vec3,
    max_points: int,
    min_sdf: float,
    max_sdf: float,
    spacing: float,
    spacing_jitter: float,
    seed: int,
    out_point_count: wp.array(dtype=int),
    out_points: wp.array(dtype=wp.vec3),
):
    x, y, z = wp.tid()

    # Retrieve the cell's center position.
    cell_pos = (
        grid_lower_bound
        + wp.vec3(
            float(x) + 0.5,
            float(y) + 0.5,
            float(z) + 0.5,
        )
        * spacing
    )

    # Query the closest location on the mesh.
    max_dist = 1000.0
    query = wp.mesh_query_point(mesh, cell_pos, max_dist)
    if not query.result:
        return

    # Evaluates the position of the closest mesh location found.
    mesh_pos = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

    # Check that the cell's distance to the mesh location is within
    # the desired range.
    dist = wp.length(cell_pos - mesh_pos) * query.sign
    if dist < min_sdf or dist > max_sdf:
        return

    # Increment the counter of valid point locations found.
    point_index = wp.atomic_add(out_point_count, 0, 1)
    if point_index > max_points:
        return

    # Compute the spacing jitter value while making sure it's normalized
    # in a range [-1, 1].
    rng = wp.rand_init(seed, point_index)
    jitter = wp.vec3(
        wp.randf(rng) * 2.0 - 1.0,
        wp.randf(rng) * 2.0 - 1.0,
        wp.randf(rng) * 2.0 - 1.0,
    )

    # Store the point position.
    out_points[point_index] = cell_pos + jitter * spacing_jitter


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self.mesh = None

        self.is_valid = False

        self.attr_tracking = omni.warp.nodes.AttrTracking(
            (
                "transform",
                "seed",
                "minSdf",
                "maxSdf",
                "radius",
                "spacing",
                "spacingJitter",
                "mass",
                "velocityDir",
                "velocityAmount",
                "maxPoints",
            ),
        )

    def needs_initialization(self, db: OgnParticlesFromMeshDatabase) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if not self.is_valid:
            return True

        if omni.warp.nodes.bundle_has_changed(db.inputs.mesh):
            return True

        return False

    def initialize(self, db: OgnParticlesFromMeshDatabase) -> bool:
        """Initializes the internal state."""
        point_count = omni.warp.nodes.mesh_get_point_count(db.inputs.mesh)
        xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.mesh)

        # Transform the mesh's point positions into world space.
        world_point_positions = wp.empty(point_count, dtype=wp.vec3)
        wp.launch(
            kernel=transform_points_kernel,
            dim=point_count,
            inputs=[
                omni.warp.nodes.mesh_get_points(db.inputs.mesh),
                xform.T,
            ],
            outputs=[
                world_point_positions,
            ],
        )

        # Initialize Warp's mesh instance, which requires
        # a triangulated topology.
        face_vertex_indices = omni.warp.nodes.mesh_triangulate(db.inputs.mesh)
        mesh = wp.Mesh(
            points=world_point_positions,
            velocities=wp.zeros(point_count, dtype=wp.vec3),
            indices=face_vertex_indices,
        )

        # Store the class members.
        self.mesh = mesh

        return True


#   Compute
# ------------------------------------------------------------------------------


def spawn_particles(db: OgnParticlesFromMeshDatabase) -> Tuple[wp.array, int]:
    """Spawns the particles by filling the given point positions array."""
    # Initialize an empty array that will hold the particle positions.
    points = wp.empty(db.inputs.maxPoints, dtype=wp.vec3)

    # Retrieve the mesh's aligned bounding box.
    extent = omni.warp.nodes.mesh_get_world_extent(
        db.inputs.mesh,
        axis_aligned=True,
    )

    # Compute the emitter's bounding box size.
    extent_size = extent[1] - extent[0]

    # Infer the emitter's grid dimensions from its bounding box size and
    # the requested spacing.
    spacing = max(db.inputs.spacing, 1e-6)
    dims = (extent_size / spacing).astype(int) + 1
    dims = np.maximum(dims, 1)

    # Add one particle per grid cell located within the mesh geometry.
    point_count = wp.zeros(1, dtype=int)
    wp.launch(
        kernel=sample_mesh_kernel,
        dim=dims,
        inputs=[
            db.per_instance_state.mesh.id,
            extent[0],
            db.inputs.maxPoints,
            db.inputs.minSdf,
            db.inputs.maxSdf,
            spacing,
            db.inputs.spacingJitter,
            db.inputs.seed,
        ],
        outputs=[
            point_count,
            points,
        ],
    )

    # Retrieve the actual number of particles created.
    point_count = min(int(point_count.numpy()[0]), db.inputs.maxPoints)

    return (points, point_count)


def compute(db: OgnParticlesFromMeshDatabase) -> None:
    """Evaluates the node."""
    db.outputs.particles.changes().activate()

    if not db.inputs.mesh.valid or not db.outputs.particles.valid:
        return

    state = db.per_instance_state

    # Initialize the internal state if it hasn't been already.
    if state.needs_initialization(db):
        if not state.initialize(db):
            return
    elif not state.attr_tracking.have_attrs_changed(db):
        return

    with omni.warp.nodes.NodeTimer("spawn_particles", db, active=PROFILING):
        # Spawn new particles inside the mesh.
        (points, point_count) = spawn_particles(db)

    # Create a new geometry points within the output bundle.
    omni.warp.nodes.points_create_bundle(
        db.outputs.particles,
        point_count,
        xform=db.inputs.transform,
        create_masses=True,
        create_velocities=True,
        create_widths=True,
    )

    # Copy the point positions onto the output bundle.
    wp.copy(
        omni.warp.nodes.points_get_points(db.outputs.particles),
        points,
        count=point_count,
    )

    if point_count:
        velocities = omni.warp.nodes.points_get_velocities(db.outputs.particles)
        if db.inputs.velocityAmount < 1e-6:
            velocities.fill_(0.0)
        else:
            # Retrieve the mesh's world transformation.
            xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.mesh)

            # Retrieve the normalized velocity direction.
            vel = db.inputs.velocityDir
            vel /= np.linalg.norm(vel)

            # Transform the velocity local direction with the mesh's world
            # rotation matrix to get the velocity direction in world space.
            vel = np.dot(xform[:3, :3].T, vel)

            # Scale the result to get the velocity's magnitude.
            vel *= db.inputs.velocityAmount

            # Store the velocities in the output bundle.
            velocities.fill_(wp.vec3(vel))

        # Store the radius in the output bundle.
        widths = omni.warp.nodes.points_get_widths(db.outputs.particles)
        widths.fill_(db.inputs.radius * 2.0)

        # Store the mass in the output bundle.
        masses = omni.warp.nodes.points_get_masses(db.outputs.particles)
        masses.fill_(db.inputs.mass)

    state.attr_tracking.update_state(db)


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnParticlesFromMesh:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnParticlesFromMeshDatabase) -> None:
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
