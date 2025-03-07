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

"""Node simulating particles."""

import traceback
from math import inf

import numpy as np
import omni.graph.core as og
import omni.timeline
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnParticlesSimulateDatabase import OgnParticlesSimulateDatabase

import warp as wp

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
def compute_particles_inv_mass_kernel(
    masses: wp.array(dtype=float),
    out_inv_masses: wp.array(dtype=float),
):
    tid = wp.tid()
    out_inv_masses[tid] = 1.0 / masses[tid]


@wp.kernel(enable_backward=False)
def compute_particles_radius_kernel(
    widths: wp.array(dtype=float),
    out_radii: wp.array(dtype=float),
):
    tid = wp.tid()
    out_radii[tid] = widths[tid] * 0.5


@wp.kernel(enable_backward=False)
def transform_points_kernel(
    points: wp.array(dtype=wp.vec3),
    xform: wp.mat44,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_points[tid] = wp.transform_point(xform, points[tid])


@wp.kernel(enable_backward=False)
def update_collider_kernel(
    points_0: wp.array(dtype=wp.vec3),
    points_1: wp.array(dtype=wp.vec3),
    xform_0: wp.mat44,
    xform_1: wp.mat44,
    sim_dt: float,
    out_points: wp.array(dtype=wp.vec3),
    out_velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    point_0 = wp.transform_point(xform_0, points_0[tid])
    point_1 = wp.transform_point(xform_1, points_1[tid])

    out_points[tid] = point_0
    out_velocities[tid] = (point_1 - point_0) / sim_dt


@wp.kernel(enable_backward=False)
def update_particles_kernel(
    points_0: wp.array(dtype=wp.vec3),
    xform: wp.mat44,
    out_points: wp.array(dtype=wp.vec3),
    out_velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    point = wp.transform_point(xform, points_0[tid])
    diff = point - points_0[tid]

    out_points[tid] = point
    out_velocities[tid] = out_velocities[tid] + diff


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self.sim_dt = None
        self.sim_tick = None
        self.model = None
        self.integrator = None
        self.state_0 = None
        self.state_1 = None
        self.xform = None
        self.collider_xform = None
        self.collider_mesh = None
        self.collider_points_0 = None
        self.collider_points_1 = None
        self.graph = None

        self.sim_enabled = True
        self.time = 0.0

        self.is_valid = False

        self.attr_tracking = omni.warp.nodes.AttrTracking(
            (
                "substepCount",
                "gravity",
                "globalScale",
                "contactElasticStiffness",
                "contactFrictionStiffness",
                "contactFrictionCoeff",
                "contactDampingStiffness",
                "particlesQueryRange",
                "particlesContactAdhesion",
                "particlesContactCohesion",
                "colliderContactDistance",
                "colliderContactQueryRange",
                "groundEnabled",
                "groundAltitude",
            ),
        )

    def needs_initialization(self, db: OgnParticlesSimulateDatabase) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if not self.is_valid or not db.inputs.enabled or not self.sim_enabled:
            return True

        if self.attr_tracking.have_attrs_changed(db):
            return True

        if db.inputs.time < self.time:
            # Reset the simulation when we're rewinding.
            return True

        return False

    def initialize(
        self,
        db: OgnParticlesSimulateDatabase,
        device: wp.context.Device,
    ) -> bool:
        """Initializes the internal state."""
        # Lazy load warp.sim here to not slow down extension loading.
        import warp.sim

        # Compute the simulation time step.
        timeline = omni.timeline.get_timeline_interface()
        sim_rate = timeline.get_ticks_per_second()
        sim_dt = 1.0 / sim_rate

        # Initialize Warp's simulation model builder.
        builder = wp.sim.ModelBuilder()

        # Retrieve some data from the particles points.
        points = omni.warp.nodes.points_get_points(db.inputs.particles)
        xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.particles)

        # Transform the particles point positions into world space.
        world_points = wp.empty(len(points), dtype=wp.vec3)
        wp.launch(
            kernel=transform_points_kernel,
            dim=len(points),
            inputs=[
                points,
                xform.T,
            ],
            outputs=[
                world_points,
            ],
        )

        if db.inputs.collider.valid:
            # Retrieve some data from the collider mesh.
            collider_points = omni.warp.nodes.mesh_get_points(db.inputs.collider)
            collider_xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.collider)

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
            collider_face_vertex_indices = omni.warp.nodes.mesh_triangulate(
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
            self.collider_xform = collider_xform.copy()
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

        # Register the input particles into the system.
        model.particle_count = omni.warp.nodes.points_get_point_count(db.inputs.particles)

        model.particle_q = world_points
        model.particle_qd = omni.warp.nodes.points_get_velocities(db.inputs.particles)
        model.particle_mass = omni.warp.nodes.points_get_masses(db.inputs.particles)

        model.particle_inv_mass = wp.empty_like(model.particle_mass)
        wp.launch(
            compute_particles_inv_mass_kernel,
            dim=model.particle_count,
            inputs=[model.particle_mass],
            outputs=[model.particle_inv_mass],
        )

        widths = omni.warp.nodes.points_get_widths(db.inputs.particles)
        model.particle_radius = wp.empty_like(widths)
        wp.launch(
            compute_particles_radius_kernel,
            dim=model.particle_count,
            inputs=[widths],
            outputs=[model.particle_radius],
        )

        model.particle_flags = wp.empty(model.particle_count, dtype=wp.uint32)
        model.particle_flags.fill_(warp.sim.model.PARTICLE_FLAG_ACTIVE.value)

        max_width = wp.array((-inf,), dtype=float)
        wp.launch(
            query_max_value_kernel,
            dim=model.particle_count,
            inputs=[widths],
            outputs=[max_width],
        )
        model.particle_max_radius = float(max_width.numpy()[0]) * 0.5

        # Allocate a single contact per particle.
        model.allocate_soft_contacts(model.particle_count)

        # Initialize the integrator.
        integrator = wp.sim.SemiImplicitIntegrator()

        # Set the model properties.
        model.ground = db.inputs.groundEnabled
        model.gravity = db.inputs.gravity
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
        model.soft_contact_margin = db.inputs.colliderContactDistance * db.inputs.colliderContactQueryRange

        # Store the class members.
        self.sim_dt = sim_dt
        self.sim_tick = 0
        self.model = model
        self.integrator = integrator
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.xform = xform.copy()

        if USE_GRAPH:
            # Create the CUDA graph. We first manually load the necessary
            # modules to avoid the capture to load all the modules that are
            # registered and possibly not relevant.
            wp.load_module(device=device)
            wp.load_module(module=warp.sim, device=device, recursive=True)
            wp.capture_begin(force_module_load=False)
            try:
                step(db)
            finally:
                self.graph = wp.capture_end()

        self.attr_tracking.update_state(db)

        return True


#   Compute
# ------------------------------------------------------------------------------


def update_collider(
    db: OgnParticlesSimulateDatabase,
) -> None:
    """Updates the collider state."""
    state = db.per_instance_state

    points = omni.warp.nodes.mesh_get_points(db.inputs.collider)
    xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.collider)

    # Swap the previous and current collider point positions.
    (state.collider_points_0, state.collider_points_1) = (
        state.collider_points_1,
        state.collider_points_0,
    )

    # Store the current point positions.
    wp.copy(state.collider_points_1, points)

    # Retrieve the previous and current world transformations.
    xform_0 = state.collider_xform
    xform_1 = xform

    # Update the internal point positions and velocities.
    wp.launch(
        kernel=update_collider_kernel,
        dim=len(state.collider_mesh.vertices),
        inputs=[
            state.collider_points_0,
            state.collider_points_1,
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

    # Update the state members.
    state.collider_xform = xform.copy()


def update_particles(
    db: OgnParticlesSimulateDatabase,
) -> None:
    """Updates the particles state."""
    state = db.per_instance_state

    xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.particles)

    # Retrieve the previous and current world transformations.
    xform_0 = state.xform
    xform_1 = xform

    try:
        xform_0_inv = np.linalg.inv(xform_0)
    except np.linalg.LinAlgError:
        # On the first run, OG sometimes return an invalid matrix,
        # so we default it to the identity one.
        xform_0_inv = np.identity(4)

    # Update the internal point positions and velocities.
    wp.launch(
        kernel=update_particles_kernel,
        dim=len(state.state_0.particle_q),
        inputs=[
            state.state_0.particle_q,
            np.matmul(xform_0_inv, xform_1).T,
        ],
        outputs=[
            state.state_0.particle_q,
            state.state_0.particle_qd,
        ],
    )

    # Update the state members.
    state.xform = xform.copy()


def step(db: OgnParticlesSimulateDatabase) -> None:
    """Steps through the simulation."""
    state = db.per_instance_state

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
    state = db.per_instance_state

    state.model.particle_grid.build(
        state.state_0.particle_q,
        state.model.particle_max_radius * db.inputs.particlesQueryRange,
    )

    if USE_GRAPH:
        wp.capture_launch(state.graph)
    else:
        step(db)


def compute(db: OgnParticlesSimulateDatabase, device: wp.context.Device) -> None:
    """Evaluates the node."""
    if not db.inputs.particles.valid or not db.outputs.particles.valid:
        return

    state = db.per_instance_state

    if not db.inputs.enabled:
        # Pass through the data.
        db.outputs.particles = db.inputs.particles

        # Store whether the simulation was last enabled.
        state.sim_enabled = False
        return

    if state.needs_initialization(db):
        # Initialize the internal state if it hasn't been already.

        # We want to use the input particles geometry as the initial state
        # of the simulation so we copy its bundle to the output one.
        db.outputs.particles = db.inputs.particles

        if not state.initialize(db, device):
            return
    else:
        # We skip the simulation if it has just been initialized.

        if state.sim_tick == 0 and omni.warp.nodes.bundle_has_changed(db.inputs.particles):
            if not state.initialize(db, device):
                return

        if (
            db.inputs.collider.valid
            and state.collider_mesh is not None
            and omni.warp.nodes.bundle_has_changed(db.inputs.collider)
        ):
            # The collider might be animated so we need to update its state.
            update_collider(db)

        if omni.warp.nodes.bundle_have_attrs_changed(db.inputs.particles, ("worldMatrix",)):
            update_particles(db)

        with omni.warp.nodes.NodeTimer("simulate", db, active=PROFILING):
            # Run the particles simulation at the current time.
            simulate(db)

        with omni.warp.nodes.NodeTimer("transform_points_to_local_space", db, active=PROFILING):
            # Retrieve some data from the particles points.
            xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.particles)

            try:
                xform_inv = np.linalg.inv(xform)
            except np.linalg.LinAlgError:
                # On the first run, OG sometimes return an invalid matrix,
                # so we default it to the identity one.
                xform_inv = np.identity(4)

            # Transform the particles point positions back into local space
            # and store them into the bundle.
            out_points = omni.warp.nodes.points_get_points(db.outputs.particles)
            wp.launch(
                kernel=transform_points_kernel,
                dim=len(out_points),
                inputs=[
                    state.state_0.particle_q,
                    xform_inv.T,
                ],
                outputs=[
                    out_points,
                ],
            )

        # Increment the simulation tick.
        state.sim_tick += 1

    # Store whether the simulation was last enabled.
    state.sim_enabled = True

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
        device = omni.warp.nodes.device_get_cuda_compute()

        try:
            with wp.ScopedDevice(device):
                compute(db, device)
        except Exception:
            db.log_error(traceback.format_exc())
            db.per_instance_state.is_valid = False
            return

        db.per_instance_state.is_valid = True

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
