# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Node generating particles inside a mesh."""

import traceback
from typing import Tuple

import numpy as np
import omni.graph.core as og
import warp as wp

import omni.warp
from omni.warp.ogn.OgnParticlesFromMeshDatabase import OgnParticlesFromMeshDatabase


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
    tid = wp.tid()
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
    sign = float(0.0)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    if not wp.mesh_query_point(
        mesh,
        cell_pos,
        max_dist,
        sign,
        face_index,
        face_u,
        face_v,
    ):
        return

    # Evaluates the position of the closest mesh location found.
    mesh_pos = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    # Check that the cell's distance to the mesh location is within
    # the desired range.
    dist = wp.length(cell_pos - mesh_pos) * sign
    if dist < min_sdf or dist > max_sdf:
        return

    # Increment the counter of valid point locations found.
    point_index = wp.atomic_add(out_point_count, 0, 1)
    if point_index > max_points:
        return

    # Compute the spacing jitter value while making sure it's normalized
    # in a range [-1, 1].
    rng = wp.rand_init(seed, tid)
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
        self._point_count = None
        self._xform = None
        self._extent = None

        self.xform_prim_path = None
        self.seed = None
        self.min_sdf = None
        self.max_sdf = None
        self.radius = None
        self.spacing = None
        self.spacing_jitter = None
        self.mass = None
        self.vel_dir = None
        self.vel_amount = None
        self.max_pts = None

        self.mesh = None

        self.is_valid = False

    def needs_initialization(self, db: OgnParticlesFromMeshDatabase) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if not self.is_valid:
            return True

        # To query whether the input geometry has changed, we could hash its
        # attributes but instead we do a cheap approximation.
        point_count = omni.warp.mesh_get_point_count(db.inputs.mesh)
        xform = omni.warp.get_world_xform(db.inputs.mesh)
        extent = omni.warp.mesh_get_local_extent(db.inputs.mesh)
        has_geometry_changed = (
            point_count != self._point_count
            or not np.array_equal(xform, self._xform)
            or not np.array_equal(extent, self._extent)
        )

        if has_geometry_changed:
            return True

        return False

    def initialize(self, db: OgnParticlesFromMeshDatabase) -> bool:
        """Initializes the internal state."""
        point_count = omni.warp.mesh_get_point_count(db.inputs.mesh)
        xform = omni.warp.get_world_xform(db.inputs.mesh)
        extent = omni.warp.mesh_get_local_extent(db.inputs.mesh)

        # Transform the mesh's point positions into world space.
        world_point_positions = wp.empty(point_count, dtype=wp.vec3)
        wp.launch(
            kernel=transform_points_kernel,
            dim=point_count,
            inputs=[
                omni.warp.mesh_get_points(db.inputs.mesh),
                xform.T,
            ],
            outputs=[
                world_point_positions,
            ],
        )

        # Initialize Warp's mesh instance, which requires
        # a triangulated topology.
        face_vertex_indices = omni.warp.mesh_get_triangulated_face_vertex_indices(db.inputs.mesh)
        mesh = wp.Mesh(
            points=world_point_positions,
            velocities=wp.zeros(point_count, dtype=wp.vec3),
            indices=face_vertex_indices,
        )

        # Store the class members.
        self.mesh = mesh

        # Cache the node attribute values relevant to this internal state.
        # They're the ones used to check whether it needs to be reinitialized
        # or not.
        self._point_count = point_count
        self._xform = xform.copy()
        self._extent = extent.copy()

        return True

    def have_setting_attrs_changed(self, db: OgnParticlesFromMeshDatabase) -> bool:
        """Checks if the values of the attributes that set-up the node have changed."""
        return (
            db.inputs.xformPrimPath != self.xform_prim_path
            or db.inputs.seed != self.seed
            or db.inputs.minSdf != self.min_sdf
            or db.inputs.maxSdf != self.max_sdf
            or db.inputs.radius != self.radius
            or db.inputs.spacing != self.spacing
            or db.inputs.spacingJitter != self.spacing_jitter
            or db.inputs.mass != self.mass
            or not np.array_equal(db.inputs.velocityDir, self.vel_dir)
            or db.inputs.velocityAmount != self.vel_amount
            or db.inputs.maxPoints != self.max_pts
        )

    def store_setting_attrs(self, db: OgnParticlesFromMeshDatabase) -> None:
        """Stores the values of the attributes that set-up the node."""
        self.xform_prim_path = db.inputs.xformPrimPath
        self.seed = db.inputs.seed
        self.min_sdf = db.inputs.minSdf
        self.max_sdf = db.inputs.maxSdf
        self.radius = db.inputs.radius
        self.spacing = db.inputs.spacing
        self.spacing_jitter = db.inputs.spacingJitter
        self.mass = db.inputs.mass
        self.vel_dir = db.inputs.velocityDir.copy()
        self.vel_amount = db.inputs.velocityAmount
        self.max_pts = db.inputs.maxPoints


#   Compute
# ------------------------------------------------------------------------------


def spawn_particles(db: OgnParticlesFromMeshDatabase) -> Tuple[wp.array, int]:
    """Spawns the particles by filling the given point positions array."""
    # Initialize an empty array that will hold the particle positions.
    points = wp.empty(db.inputs.maxPoints, dtype=wp.vec3)

    # Retrieve the mesh's aligned bounding box.
    extent = omni.warp.mesh_get_world_extent(
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
            db.internal_state.mesh.id,
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

    state = db.internal_state

    # Initialize the internal state if it hasn't been already.
    if state.needs_initialization(db):
        if not state.initialize(db):
            return
    elif not state.have_setting_attrs_changed(db):
        return

    with omni.warp.NodeTimer("spawn_particles", db, active=PROFILING):
        # Spawn new particles inside the mesh.
        (points, point_count) = spawn_particles(db)

    # Create a new geometry points within the output bundle.
    omni.warp.points_create_bundle(db.outputs.particles, point_count)

    # Copy the point positions onto the output bundle.
    wp.copy(
        omni.warp.points_get_points(db.outputs.particles),
        points,
        count=point_count,
    )

    if point_count:
        velocities = omni.warp.points_get_velocities(db.outputs.particles)
        if db.inputs.velocityAmount < 1e-6:
            velocities.fill_(0.0)
        else:
            # Retrieve the mesh's world transformation.
            xform = omni.warp.get_world_xform(db.inputs.mesh)

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
        widths = omni.warp.points_get_widths(db.outputs.particles)
        widths.fill_(db.inputs.radius * 2.0)

        # Store the mass in the output bundle.
        masses = omni.warp.points_get_masses(db.outputs.particles)
        masses.fill_(db.inputs.mass)

    # Set the USD primitive path and type.
    omni.warp.define_prim_attrs(
        db.outputs.particles,
        "Points",
        xform_prim_path=db.inputs.xformPrimPath,
    )

    state.store_setting_attrs(db)


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnParticlesFromMesh:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnParticlesFromMeshDatabase) -> None:
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
