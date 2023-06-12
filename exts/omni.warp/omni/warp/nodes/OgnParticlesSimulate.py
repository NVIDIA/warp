# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Node simulating particles."""

from math import inf
import traceback

import numpy as np
import omni.graph.core as og
import omni.timeline
import warp as wp
import warp.sim

import omni.warp
from omni.warp.ogn.OgnParticlesSimulateDatabase import OgnParticlesSimulateDatabase


USE_GRAPH = True

PROFILING = False


#   Kernels
# ------------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def query_max_value_kernel(
    values: wp.array(dtype=float),
    out_max: wp.array(dtype=float),
):
    wp.atomic_max(out_max, 0, values[wp.tid()])


@wp.kernel(enable_backward=False)
def transform_points_kernel(
    points: wp.array(dtype=wp.vec3),
    xform: wp.mat44,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_points[tid] = wp.transform_point(xform, points[tid])


@wp.kernel(enable_backward=False)
def transform_collider_kernel(
    collider_points_0: wp.array(dtype=wp.vec3),
    collider_points_1: wp.array(dtype=wp.vec3),
    xform_0: wp.mat44,
    xform_1: wp.mat44,
    sim_dt: float,
    out_points: wp.array(dtype=wp.vec3),
    out_velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    points_0 = wp.transform_point(xform_0, collider_points_0[tid])
    points_1 = wp.transform_point(xform_1, collider_points_1[tid])

    out_points[tid] = points_0
    out_velocities[tid] = (points_1 - points_0) / sim_dt


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self._substep_count = None
        self._gravity = None
        self._global_scale = None
        self._contact_elastic_stiffness = None
        self._contact_friction_stiffness = None
        self._contact_friction_coeff = None
        self._contact_damping_stiffness = None
        self._particles_query_range = None
        self._particles_contact_adhesion = None
        self._particles_contact_cohesion = None
        self._collider_contact_distance = None
        self._collider_contact_query_range = None
        self._ground_enabled = None
        self._ground_altitude = None
        self._particles_point_count = None
        self._particles_xform = None
        self._particles_extent = None

        self.sim_dt = None
        self.model = None
        self.max_radius = None
        self.integrator = None
        self.state_0 = None
        self.state_1 = None
        self.collider_point_count = None
        self.collider_xform = None
        self.collider_extent = None
        self.collider_mesh = None
        self.collider_points_0 = None
        self.collider_points_1 = None
        self.graph = None

        self.enabled = True
        self.time = 0.0

        self.is_valid = False

    def needs_initialization(self, db: OgnParticlesSimulateDatabase) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if not self.is_valid or not db.inputs.enabled:
            return True

        if (
            not self.enabled
            or db.inputs.substepCount != self._substep_count
            or not np.array_equal(db.inputs.gravity, self._gravity)
            or db.inputs.globalScale != self._global_scale
            or db.inputs.contactElasticStiffness != self._contact_elastic_stiffness
            or db.inputs.contactFrictionStiffness != self._contact_friction_stiffness
            or db.inputs.contactFrictionCoeff != self._contact_friction_coeff
            or db.inputs.contactDampingStiffness != self._contact_damping_stiffness
            or db.inputs.particlesQueryRange != self._particles_query_range
            or db.inputs.particlesContactAdhesion != self._particles_contact_adhesion
            or db.inputs.particlesContactCohesion != self._particles_contact_cohesion
            or db.inputs.colliderContactDistance != self._collider_contact_distance
            or db.inputs.colliderContactQueryRange != self._collider_contact_query_range
            or db.inputs.groundEnabled != self._ground_enabled
            or db.inputs.groundAltitude != self._ground_altitude
        ):
            return True

        # To query whether the input particles have changed, we could hash their
        # attributes but instead we do a cheap approximation.
        particles_point_count = omni.warp.points_get_point_count(db.inputs.particles)
        particles_xform = omni.warp.get_world_xform(db.inputs.particles)
        particles_extent = omni.warp.points_get_local_extent(db.inputs.particles)
        has_geometry_changed = (
            particles_point_count != self._particles_point_count
            or not np.array_equal(particles_xform, self._particles_xform)
            or not np.array_equal(particles_extent, self._particles_extent)
        )

        if has_geometry_changed:
            return True

        if db.inputs.time < self.time:
            # Reset the simulation when we're rewinding.
            return True

        return False

    def initialize(self, db: OgnParticlesSimulateDatabase) -> bool:
        """Initializes the internal state."""
        particles_point_count = omni.warp.points_get_point_count(db.inputs.particles)
        particles_xform = omni.warp.get_world_xform(db.inputs.particles)
        particles_extent = omni.warp.points_get_local_extent(db.inputs.particles)

        # Compute the simulation time step.
        timeline = omni.timeline.get_timeline_interface()
        sim_rate = timeline.get_ticks_per_second()
        sim_dt = 1.0 / sim_rate

        # Retrieve the radius of the largest particle.
        widths = omni.warp.points_get_widths(db.inputs.particles)
        max_width = wp.array((-inf,), dtype=float)
        wp.launch(
            query_max_value_kernel,
            dim=len(widths),
            inputs=[widths],
            outputs=[max_width],
        )
        max_radius = float(max_width.numpy()[0]) * 0.5

        # Initialize Warp's simulation model builder.
        builder = wp.sim.ModelBuilder()

        # Register the input particles to the system.
        positions = omni.warp.points_get_points(db.inputs.particles).numpy()
        velocities = omni.warp.points_get_velocities(db.inputs.particles).numpy()
        masses = omni.warp.points_get_masses(db.inputs.particles).numpy()
        for pos, vel, mass in zip(positions, velocities, masses):
            builder.add_particle(pos=pos, vel=vel, mass=mass)

        if db.inputs.collider.valid:
            # Retrieve some data from the collider mesh.
            collider_points = omni.warp.mesh_get_points(db.inputs.collider)
            collider_xform = omni.warp.get_world_xform(db.inputs.collider)
            collider_extent = omni.warp.mesh_get_local_extent(db.inputs.collider)

            # Transform the collider point positions into world space.
            collider_world_points = wp.empty(
                len(collider_points),
                dtype=wp.vec3,
            )
            wp.launch(
                kernel=transform_points_kernel,
                dim=len(collider_points),
                inputs=[
                    collider_points,
                    collider_xform.T,
                ],
                outputs=[
                    collider_world_points,
                ],
            )

            # Initialize Warp's mesh instance, which requires
            # triangulated meshes.
            collider_face_vertex_indices = omni.warp.mesh_get_triangulated_face_vertex_indices(
                db.inputs.collider,
            )
            collider_mesh = wp.sim.Mesh(
                collider_world_points.numpy(),
                collider_face_vertex_indices.numpy(),
                compute_inertia=False,
            )

            # Register the collider geometry mesh into Warp's simulation model
            # builder.
            builder.add_shape_mesh(
                body=-1,
                mesh=collider_mesh,
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                scale=(1.0, 1.0, 1.0),
            )

            # Store the collider's point positions as internal state.
            collider_points_0 = wp.empty_like(collider_points)
            collider_points_1 = wp.empty_like(collider_points)
            wp.copy(collider_points_0, collider_points)
            wp.copy(collider_points_1, collider_points)

            # Store the class members.
            self.collider_point_count = len(collider_points)
            self.collider_xform = collider_xform.copy()
            self.collider_extent = collider_extent.copy()
            self.collider_mesh = collider_mesh
            self.collider_points_0 = collider_points_0
            self.collider_points_1 = collider_points_1
        else:
            self.collider_mesh = None

        # Register the ground.
        builder.set_ground_plane(
            offset=-db.inputs.groundAltitude,
            ke=db.inputs.contactElasticStiffness * db.inputs.globalScale,
            kd=db.inputs.contactDampingStiffness * db.inputs.globalScale,
            kf=db.inputs.contactFrictionStiffness * db.inputs.globalScale,
            mu=db.inputs.contactFrictionCoeff,
        )

        # Build the simulation model.
        model = builder.finalize()

        # Allocate a single contact per particle.
        model.allocate_soft_contacts(model.particle_count)

        # Initialize the integrator.
        integrator = wp.sim.SemiImplicitIntegrator()

        # Set the model properties.
        model.ground = db.inputs.groundEnabled
        model.gravity = db.inputs.gravity
        model.particle_radius = max_radius
        model.particle_adhesion = db.inputs.particlesContactAdhesion
        model.particle_cohesion = db.inputs.particlesContactCohesion
        model.particle_ke = db.inputs.contactElasticStiffness * db.inputs.globalScale
        model.particle_kf = db.inputs.contactFrictionStiffness * db.inputs.globalScale
        model.particle_mu = db.inputs.contactFrictionCoeff
        model.particle_kd = db.inputs.contactDampingStiffness * db.inputs.globalScale
        model.soft_contact_ke = db.inputs.contactElasticStiffness * db.inputs.globalScale
        model.soft_contact_kf = db.inputs.contactFrictionStiffness * db.inputs.globalScale
        model.soft_contact_mu = db.inputs.contactFrictionCoeff
        model.soft_contact_kd = db.inputs.contactDampingStiffness * db.inputs.globalScale
        model.soft_contact_distance = db.inputs.colliderContactDistance
        model.soft_contact_margin = db.inputs.colliderContactDistance * db.inputs.colliderContactQueryRange

        # Store the class members.
        self.sim_dt = sim_dt
        self.model = model
        self.max_radius = max_radius
        self.integrator = integrator
        self.state_0 = model.state()
        self.state_1 = model.state()

        if USE_GRAPH:
            # Create the CUDA graph.
            wp.capture_begin()
            step(db)
            self.graph = wp.capture_end()
        else:
            self.graph = None

        # Cache the node attribute values relevant to this internal state.
        # They're the ones used to check whether it needs to be reinitialized
        # or not.
        self._substep_count = db.inputs.substepCount
        self._gravity = db.inputs.gravity.copy()
        self._global_scale = db.inputs.globalScale
        self._contact_elastic_stiffness = db.inputs.contactElasticStiffness
        self._contact_friction_stiffness = db.inputs.contactFrictionStiffness
        self._contact_friction_coeff = db.inputs.contactFrictionCoeff
        self._contact_damping_stiffness = db.inputs.contactDampingStiffness
        self._particles_query_range = db.inputs.particlesQueryRange
        self._particles_contact_adhesion = db.inputs.particlesContactAdhesion
        self._particles_contact_cohesion = db.inputs.particlesContactCohesion
        self._collider_contact_distance = db.inputs.colliderContactDistance
        self._collider_contact_query_range = db.inputs.colliderContactQueryRange
        self._ground_enabled = db.inputs.groundEnabled
        self._ground_altitude = db.inputs.groundAltitude
        self._particles_point_count = particles_point_count
        self._particles_xform = particles_xform.copy()
        self._particles_extent = particles_extent.copy()

        return True


#   Compute
# ------------------------------------------------------------------------------


def update_collider(
    db: OgnParticlesSimulateDatabase,
    points: wp.array,
    xform: np.ndarray,
) -> None:
    """Updates the collider state."""
    state = db.internal_state

    # Swap the previous and current collider point positions.
    (state.collider_points_0, state.collider_points_1) = (
        state.collider_points_1,
        state.collider_points_0,
    )

    # Store the current point positions.
    wp.copy(state.collider_points_1, points)

    # Store the previous and current world transformations.
    xform_0 = state.collider_xform
    xform_1 = xform

    # Update the internal point positions and velocities.
    wp.launch(
        kernel=transform_collider_kernel,
        dim=len(state.collider_mesh.vertices),
        inputs=[
            state.collider_points_1,
            state.collider_points_0,
            xform_0.T,
            xform_1.T,
            state.sim_dt,
        ],
        outputs=[
            state.collider_mesh.mesh.points,
            state.collider_mesh.mesh.velocities,
        ],
    )

    # Refit the BVH.
    state.collider_mesh.mesh.refit()


def step(db: OgnParticlesSimulateDatabase) -> None:
    """Steps through the simulation."""
    state = db.internal_state

    sim_dt = state.sim_dt / db.inputs.substepCount

    # Run the collision detection once per frame.
    wp.sim.collide(state.model, state.state_0)

    for _ in range(db.inputs.substepCount):
        state.state_0.clear_forces()
        state.integrator.simulate(
            state.model,
            state.state_0,
            state.state_1,
            sim_dt,
        )

        # Swap the previous and current states.
        (state.state_0, state.state_1) = (state.state_1, state.state_0)


def simulate(db: OgnParticlesSimulateDatabase) -> None:
    """Simulates the particles at the current time."""
    state = db.internal_state

    state.model.particle_grid.build(
        state.state_0.particle_q,
        state.max_radius * db.inputs.particlesQueryRange,
    )

    if USE_GRAPH:
        wp.capture_launch(state.graph)
    else:
        step(db)


def compute(db: OgnParticlesSimulateDatabase) -> None:
    """Evaluates the node."""
    if not db.inputs.particles.valid or not db.outputs.particles.valid:
        return

    state = db.internal_state

    if not db.inputs.enabled:
        # Pass through the data.
        db.outputs.particles = db.inputs.particles

        # Store whether the simulation was last enabled.
        state.enabled = False
        return

    # Initialize the internal state if it hasn't been already.
    if state.needs_initialization(db):
        # We want to use the input particles geometry as the initial state
        # of the simulation so we copy its bundle to the output one.
        db.outputs.particles = db.inputs.particles

        if not state.initialize(db):
            return

    if db.inputs.collider.valid and state.collider_mesh is not None:
        # To query whether the input collider has changed, we could hash its
        # attributes but instead we do a cheap approximation.
        collider_points = omni.warp.mesh_get_points(db.inputs.collider)
        collider_xform = omni.warp.get_world_xform(db.inputs.collider)
        collider_extent = omni.warp.mesh_get_local_extent(db.inputs.collider)
        has_geometry_changed = (
            len(collider_points) != state.collider_point_count
            or not np.array_equal(collider_xform, state.collider_xform)
            or not np.array_equal(collider_extent, state.collider_extent)
        )

        if has_geometry_changed:
            with omni.warp.NodeTimer("update_collider", db, active=PROFILING):
                # The collider might be animated so we need to update its state.
                update_collider(db, collider_points, collider_xform)

            # Update the state members.
            state.collider_point_count = len(collider_points)
            state.collider_xform = collider_xform.copy()
            state.collider_extent = collider_extent.copy()

    with omni.warp.NodeTimer("simulate", db, active=PROFILING):
        # Run the particles simulation at the current time.
        simulate(db)

    # Store the current point positions into the bundle.
    out_points = omni.warp.points_get_points(db.outputs.particles)
    wp.copy(out_points, state.state_0.particle_q)

    # Store whether the simulation was last enabled.
    state.enabled = True

    # Store the current time.
    state.time = db.inputs.time


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnParticlesSimulate:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnParticlesSimulateDatabase) -> None:
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
