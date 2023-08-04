# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Node simulating cloth."""

import traceback

import numpy as np
import omni.graph.core as og
import omni.timeline
import warp as wp
import warp.sim

import omni.warp.nodes
from omni.warp.nodes.ogn.OgnClothSimulateDatabase import OgnClothSimulateDatabase


USE_GRAPH = True

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
def update_cloth_kernel(
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
                "clothDensity",
                "clothTriElasticStiffness",
                "clothTriAreaStiffness",
                "clothTriDampingStiffness",
                "clothEdgeBendingStiffness",
                "clothEdgeDampingStiffness",
                "colliderContactDistance",
                "colliderContactQueryRange",
                "groundEnabled",
                "groundAltitude",
            ),
        )

    def needs_initialization(self, db: OgnClothSimulateDatabase) -> bool:
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
        db: OgnClothSimulateDatabase,
        device: wp.context.Device,
    ) -> bool:
        """Initializes the internal state."""
        # Compute the simulation time step.
        timeline = omni.timeline.get_timeline_interface()
        sim_rate = timeline.get_ticks_per_second()
        sim_dt = 1.0 / sim_rate

        # Initialize Warp's simulation model builder.
        builder = wp.sim.ModelBuilder()

        # Retrieve some data from the cloth mesh.
        points = omni.warp.nodes.mesh_get_points(db.inputs.cloth)
        xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.cloth)

        # Transform the cloth point positions into world space.
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

        # Register the cloth geometry mesh into Warp's simulation model builder,
        # which requires triangulated meshes.
        face_vertex_indices = omni.warp.nodes.mesh_get_triangulated_face_vertex_indices(db.inputs.cloth)
        builder.add_cloth_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=world_points.numpy(),
            indices=face_vertex_indices.numpy(),
            density=db.inputs.clothDensity,
            tri_ke=db.inputs.clothTriElasticStiffness * db.inputs.globalScale,
            tri_ka=db.inputs.clothTriAreaStiffness * db.inputs.globalScale,
            tri_kd=db.inputs.clothTriDampingStiffness * db.inputs.globalScale,
            edge_ke=db.inputs.clothEdgeBendingStiffness * db.inputs.globalScale,
            edge_kd=db.inputs.clothEdgeDampingStiffness * db.inputs.globalScale,
        )

        # Set a uniform mass to avoid large discrepencies.
        avg_mass = np.mean(builder.particle_mass)
        builder.particle_mass = np.full(
            (len(builder.particle_mass),),
            avg_mass,
        )

        if db.inputs.collider.valid:
            # Retrieve some data from the collider mesh.
            collider_points = omni.warp.nodes.mesh_get_points(db.inputs.collider)
            collider_xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.collider)

            # Transform the collider point position into world space.
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
            collider_face_vertex_indices = omni.warp.nodes.mesh_get_triangulated_face_vertex_indices(
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

        # Allocate a single contact per particle.
        model.allocate_soft_contacts(model.particle_count)

        # Initialize the integrator.
        integrator = wp.sim.SemiImplicitIntegrator()

        # Set the model properties.
        model.ground = db.inputs.groundEnabled
        model.gravity = db.inputs.gravity
        model.soft_contact_ke = db.inputs.contactElasticStiffness * db.inputs.globalScale
        model.soft_contact_kf = db.inputs.contactFrictionStiffness * db.inputs.globalScale
        model.soft_contact_mu = db.inputs.contactFrictionCoeff
        model.soft_contact_kd = db.inputs.contactDampingStiffness * db.inputs.globalScale
        model.soft_contact_margin = db.inputs.colliderContactDistance * db.inputs.colliderContactQueryRange
        model.particle_radius = db.inputs.colliderContactDistance

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
            step(db)
            self.graph = wp.capture_end()
        else:
            self.graph = None

        self.attr_tracking.update_state(db)

        return True


#   Compute
# ------------------------------------------------------------------------------


def update_collider(
    db: OgnClothSimulateDatabase,
) -> None:
    """Updates the collider state."""
    state = db.internal_state

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

    # Update the state members.
    state.collider_xform = xform.copy()


def update_cloth(
    db: OgnClothSimulateDatabase,
) -> None:
    """Updates the cloth state."""
    state = db.internal_state

    xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.cloth)

    # Retrieve the previous and current world transformations.
    xform_0 = state.xform
    xform_1 = xform

    # Update the internal point positions and velocities.
    wp.launch(
        kernel=update_cloth_kernel,
        dim=len(state.state_0.particle_q),
        inputs=[
            state.state_0.particle_q,
            np.matmul(np.linalg.inv(xform_0), xform_1).T,
        ],
        outputs=[
            state.state_0.particle_q,
            state.state_0.particle_qd,
        ],
    )

    # Update the state members.
    state.xform = xform.copy()


def step(db: OgnClothSimulateDatabase) -> None:
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


def simulate(db: OgnClothSimulateDatabase) -> None:
    """Simulates the cloth at the current time."""
    state = db.internal_state

    if USE_GRAPH:
        wp.capture_launch(state.graph)
    else:
        step(db)


def compute(db: OgnClothSimulateDatabase, device: wp.context.Device) -> None:
    """Evaluates the node."""
    if not db.inputs.cloth.valid or not db.outputs.cloth.valid:
        return

    state = db.internal_state

    if not db.inputs.enabled:
        # Pass through the data.
        db.outputs.cloth = db.inputs.cloth

        # Store whether the simulation was last enabled.
        state.sim_enabled = False
        return

    if state.needs_initialization(db):
        # Initialize the internal state if it hasn't been already.

        # We want to use the input cloth geometry as the initial state
        # of the simulation so we copy its bundle to the output one.
        omni.warp.nodes.mesh_copy_bundle(
            db.outputs.cloth,
            db.inputs.cloth,
            deep_copy=True,
        )

        if not state.initialize(db, device):
            return
    else:
        # We skip the simulation if it has just been initialized.

        if state.sim_tick == 0 and omni.warp.nodes.bundle_has_changed(db.inputs.cloth):
            if not state.initialize(db, device):
                return

        if (
            db.inputs.collider.valid
            and state.collider_mesh is not None
            and omni.warp.nodes.bundle_has_changed(db.inputs.collider)
        ):
            # The collider might be animated so we need to update its state.
            update_collider(db)

        if omni.warp.nodes.bundle_have_attrs_changed(db.inputs.cloth, ("worldMatrix",)):
            update_cloth(db)

        with omni.warp.nodes.NodeTimer("simulate", db, active=PROFILING):
            # Run the cloth simulation at the current time.
            simulate(db)

        with omni.warp.nodes.NodeTimer("transform_points_to_local_space", db, active=PROFILING):
            # Retrieve some data from the cloth mesh.
            xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.cloth)

            # Transform the cloth point positions back into local space
            # and store them into the bundle.
            out_points = omni.warp.nodes.points_get_points(db.outputs.cloth)
            wp.launch(
                kernel=transform_points_kernel,
                dim=len(out_points),
                inputs=[
                    state.state_0.particle_q,
                    np.linalg.inv(xform).T,
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


class OgnClothSimulate:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnClothSimulateDatabase) -> None:
        device = wp.get_device("cuda:0")

        try:
            with wp.ScopedDevice(device):
                compute(db, device)
        except Exception:
            db.log_error(traceback.format_exc())
            db.internal_state.is_valid = False
            return

        db.internal_state.is_valid = True

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
