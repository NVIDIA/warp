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

"""Node simulating cloth."""

import traceback

import numpy as np
import omni.graph.core as og
import omni.timeline
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnClothSimulateDatabase import OgnClothSimulateDatabase

import warp as wp

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


@wp.kernel
def update_contacts_kernel(
    points: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    sim_dt: float,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_points[tid] = points[tid] + velocities[tid] * sim_dt


def basis_curve_points_from_springs_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    idx = indices[tid]
    out_points[tid] = points[idx]


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

        self.visualization_enabled = False
        self.sim_enabled = True
        self.time = 0.0

        self.is_valid = False

        self.attr_tracking = omni.warp.nodes.AttrTracking(
            (
                "substepCount",
                "gravity",
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
        # Lazy load warp.sim here to not slow down extension loading.
        import warp.sim

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
        face_vertex_indices = omni.warp.nodes.mesh_triangulate(db.inputs.cloth)
        builder.add_cloth_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=world_points.numpy(),
            indices=face_vertex_indices.numpy(),
            density=db.inputs.clothDensity,
            tri_ke=db.inputs.clothTriElasticStiffness,
            tri_ka=db.inputs.clothTriAreaStiffness,
            tri_kd=db.inputs.clothTriDampingStiffness,
            tri_drag=db.inputs.clothTriDrag,
            tri_lift=db.inputs.clothTriLift,
            edge_ke=db.inputs.clothEdgeBendingStiffness,
            edge_kd=db.inputs.clothEdgeDampingStiffness,
        )

        # Set a uniform mass to avoid large discrepancies.
        avg_mass = np.mean(builder.particle_mass)
        builder.particle_mass = np.full(
            (len(builder.particle_mass),),
            avg_mass,
        )

        # Register any spring constraint.
        for src_idx, dst_idx in db.inputs.springIndexPairs:
            builder.add_spring(
                src_idx,
                dst_idx,
                ke=db.inputs.springElasticStiffness,
                kd=db.inputs.springDampingStiffness,
                control=1.0,
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
                density=0.0,
                ke=0.0,
                kd=0.0,
                kf=0.0,
                mu=0.0,
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
            offset=-db.inputs.groundAltitude + db.inputs.colliderContactDistance,
            ke=db.inputs.contactElasticStiffness,
            kd=db.inputs.contactDampingStiffness,
            kf=db.inputs.contactFrictionStiffness,
            mu=db.inputs.contactFrictionCoeff,
        )

        # Create the coloring required by the VBD integrator.
        builder.color()

        # Build the simulation model.
        model = builder.finalize()

        # Allocate a single contact per particle.
        model.allocate_soft_contacts(model.particle_count)

        # Initialize the integrator.
        integrator = wp.sim.VBDIntegrator(model, iterations=1)

        # Set the model properties.
        model.ground = db.inputs.groundEnabled
        model.gravity = db.inputs.gravity
        model.soft_contact_ke = db.inputs.contactElasticStiffness
        model.soft_contact_kf = db.inputs.contactFrictionStiffness
        model.soft_contact_mu = db.inputs.contactFrictionCoeff
        model.soft_contact_kd = db.inputs.contactDampingStiffness
        model.soft_contact_margin = db.inputs.colliderContactDistance * db.inputs.colliderContactQueryRange
        model.particle_radius.fill_(db.inputs.colliderContactDistance)

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
            wp.set_module_options({"block_dim": 256}, warp.sim.integrator_vbd)
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
    db: OgnClothSimulateDatabase,
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


def update_cloth(
    db: OgnClothSimulateDatabase,
) -> None:
    """Updates the cloth state."""
    state = db.per_instance_state

    xform = omni.warp.nodes.bundle_get_world_xform(db.inputs.cloth)

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
        kernel=update_cloth_kernel,
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


def step(db: OgnClothSimulateDatabase) -> None:
    """Steps through the simulation."""
    state = db.per_instance_state

    sim_dt = state.sim_dt / db.inputs.substepCount

    # Run the collision detection once per frame.
    wp.sim.collide(state.model, state.state_0)

    for _ in range(db.inputs.substepCount):
        state.state_0.clear_forces()

        wp.launch(
            update_contacts_kernel,
            state.model.soft_contact_max,
            inputs=[
                state.model.soft_contact_body_pos,
                state.model.soft_contact_body_vel,
                sim_dt,
            ],
            outputs=[
                state.model.soft_contact_body_pos,
            ],
        )

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
    state = db.per_instance_state

    if USE_GRAPH:
        wp.capture_launch(state.graph)
    else:
        step(db)


def compute(db: OgnClothSimulateDatabase, device: wp.context.Device) -> None:
    """Evaluates the node."""
    if not db.inputs.cloth.valid or not db.outputs.cloth.valid:
        return

    state = db.per_instance_state

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
        db.outputs.cloth = db.inputs.cloth

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

            try:
                xform_inv = np.linalg.inv(xform)
            except np.linalg.LinAlgError:
                # On the first run, OG sometimes return an invalid matrix,
                # so we default it to the identity one.
                xform_inv = np.identity(4)

            # Transform the cloth point positions back into local space
            # and store them into the bundle.
            out_points = omni.warp.nodes.points_get_points(db.outputs.cloth)
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

    # Clear any previous visualization data.
    if state.visualization_enabled:
        db.outputs.visualization.bundle.clear_contents()
        state.visualization_enabled = False

    # Each type of visualization goes into its own primitive.
    visualization_prim = 0

    # Visualize the spring constraints as curves.
    if db.inputs.springVisualize and db.inputs.springIndexPairs.size > 0:
        spring_indices = omni.warp.nodes.from_omni_graph(
            db.inputs.springIndexPairs, dtype=int, shape=(db.inputs.springIndexPairs.size,)
        )

        # Create a new set of geometry curves within the output bundle.
        omni.warp.nodes.basis_curves_create_bundle(
            db.outputs.visualization,
            len(db.inputs.springIndexPairs) * 2,
            len(db.inputs.springIndexPairs),
            type="linear",
            xform=omni.warp.nodes.bundle_get_world_xform(db.outputs.cloth),
            create_display_color=True,
            create_widths=True,
            child_idx=visualization_prim,
        )

        # Set the curve point positions by looking up the model's particles
        # with the spring indices.
        out_points = omni.warp.nodes.basis_curves_get_points(
            db.outputs.visualization,
            child_idx=visualization_prim,
        )
        wp.launch(
            kernel=basis_curve_points_from_springs_kernel,
            dim=len(out_points),
            inputs=[
                state.state_0.particle_q,
                spring_indices,
            ],
            outputs=[
                out_points,
            ],
        )

        # Set the number of points per curve. Each curve represents a constraint
        # between 2 points, so we set them all to a length of 2.
        out_counts = omni.warp.nodes.basis_curves_get_curve_vertex_counts(
            db.outputs.visualization,
            child_idx=visualization_prim,
        )
        out_counts.fill_(2)

        # Set the curve widths.
        out_widths = omni.warp.nodes.basis_curves_get_widths(
            db.outputs.visualization,
            child_idx=visualization_prim,
        )
        out_widths.fill_(db.inputs.springVisualizeWidth)

        # Set the curve colours.
        out_colors = omni.warp.nodes.basis_curves_get_display_color(
            db.outputs.visualization,
            child_idx=visualization_prim,
        )
        out_colors.fill_(wp.vec3(db.inputs.springVisualizeColor))

        # Store whether any visualization was last enabled.
        state.visualization_enabled = True

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
