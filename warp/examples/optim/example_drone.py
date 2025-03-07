# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Example Drone
#
# A drone and its 4 propellers is simulated with the goal of reaching
# different targets via model-predictive control (MPC) that continuously
# optimizes the control trajectory.
#
###########################################################################

import os
from typing import Optional, Tuple

import numpy as np

import warp as wp
import warp.examples
import warp.optim
import warp.sim
import warp.sim.render
from warp.sim.collide import box_sdf, capsule_sdf, cone_sdf, cylinder_sdf, mesh_sdf, plane_sdf, sphere_sdf

DEFAULT_DRONE_PATH = os.path.join(warp.examples.get_asset_directory(), "crazyflie.usd")  # Path to input drone asset


@wp.struct
class Propeller:
    body: int
    pos: wp.vec3
    dir: wp.vec3
    thrust: float
    power: float
    diameter: float
    height: float
    max_rpm: float
    max_thrust: float
    max_torque: float
    turning_direction: float
    max_speed_square: float


@wp.kernel
def increment_seed(
    seed: wp.array(dtype=int),
):
    seed[0] += 1


@wp.kernel
def sample_gaussian(
    mean_trajectory: wp.array(dtype=float, ndim=3),
    noise_scale: float,
    num_control_points: int,
    control_dim: int,
    control_limits: wp.array(dtype=float, ndim=2),
    seed: wp.array(dtype=int),
    rollout_trajectories: wp.array(dtype=float, ndim=3),
):
    env_id, point_id, control_id = wp.tid()
    unique_id = (env_id * num_control_points + point_id) * control_dim + control_id
    r = wp.rand_init(seed[0], unique_id)
    mean = mean_trajectory[0, point_id, control_id]
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    sample = mean + noise_scale * wp.randn(r)
    for _i in range(10):
        if sample < lo or sample > hi:
            sample = mean + noise_scale * wp.randn(r)
        else:
            break
    rollout_trajectories[env_id, point_id, control_id] = wp.clamp(sample, lo, hi)


@wp.kernel
def replicate_states(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    bodies_per_env: int,
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    env_offset = tid * bodies_per_env
    for i in range(bodies_per_env):
        body_q_out[env_offset + i] = body_q_in[i]
        body_qd_out[env_offset + i] = body_qd_in[i]


@wp.kernel
def drone_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    targets: wp.array(dtype=wp.vec3),
    prop_control: wp.array(dtype=float),
    step: int,
    horizon_length: int,
    weighting: float,
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    tf = body_q[env_id]
    target = targets[0]

    pos_drone = wp.transform_get_translation(tf)
    pos_cost = wp.length_sq(pos_drone - target)
    altitude_cost = wp.max(pos_drone[1] - 0.75, 0.0) + wp.max(0.25 - pos_drone[1], 0.0)
    upvector = wp.vec3(0.0, 1.0, 0.0)
    drone_up = wp.transform_vector(tf, upvector)
    upright_cost = 1.0 - wp.dot(drone_up, upvector)

    vel_drone = body_qd[env_id]

    # Encourage zero velocity.
    vel_cost = wp.length_sq(vel_drone)

    control = wp.vec4(
        prop_control[env_id * 4 + 0],
        prop_control[env_id * 4 + 1],
        prop_control[env_id * 4 + 2],
        prop_control[env_id * 4 + 3],
    )
    control_cost = wp.dot(control, control)

    discount = 0.8 ** wp.float(horizon_length - step - 1) / wp.float(horizon_length) ** 2.0

    pos_weight = 1000.0
    altitude_weight = 100.0
    control_weight = 0.05
    vel_weight = 0.1
    upright_weight = 10.0
    total_weight = pos_weight + altitude_weight + control_weight + vel_weight + upright_weight

    wp.atomic_add(
        cost,
        env_id,
        (
            pos_cost * pos_weight
            + altitude_cost * altitude_weight
            + control_cost * control_weight
            + vel_cost * vel_weight
            + upright_cost * upright_weight
        )
        * (weighting / total_weight)
        * discount,
    )


@wp.kernel
def collision_cost(
    body_q: wp.array(dtype=wp.transform),
    obstacle_ids: wp.array(dtype=int, ndim=2),
    shape_X_bs: wp.array(dtype=wp.transform),
    geo: wp.sim.ModelShapeGeometry,
    margin: float,
    weighting: float,
    cost: wp.array(dtype=wp.float32),
):
    env_id, obs_id = wp.tid()
    shape_index = obstacle_ids[env_id, obs_id]

    px = wp.transform_get_translation(body_q[env_id])

    X_bs = shape_X_bs[shape_index]

    # transform particle position to shape local space
    x_local = wp.transform_point(wp.transform_inverse(X_bs), px)

    # geo description
    geo_type = geo.type[shape_index]
    geo_scale = geo.scale[shape_index]

    # evaluate shape sdf
    d = 1e6

    if geo_type == wp.sim.GEO_SPHERE:
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
    elif geo_type == wp.sim.GEO_BOX:
        d = box_sdf(geo_scale, x_local)
    elif geo_type == wp.sim.GEO_CAPSULE:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == wp.sim.GEO_CYLINDER:
        d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == wp.sim.GEO_CONE:
        d = cone_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == wp.sim.GEO_MESH:
        mesh = geo.source[shape_index]
        min_scale = wp.min(geo_scale)
        max_dist = margin / min_scale
        d = mesh_sdf(mesh, wp.cw_div(x_local, geo_scale), max_dist)
        d *= min_scale  # TODO fix this, mesh scaling needs to be handled properly
    elif geo_type == wp.sim.GEO_SDF:
        volume = geo.source[shape_index]
        xpred_local = wp.volume_world_to_index(volume, wp.cw_div(x_local, geo_scale))
        nn = wp.vec3(0.0, 0.0, 0.0)
        d = wp.volume_sample_grad_f(volume, xpred_local, wp.Volume.LINEAR, nn)
    elif geo_type == wp.sim.GEO_PLANE:
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local)

    d = wp.max(d, 0.0)
    if d < margin:
        c = margin - d
        wp.atomic_add(cost, env_id, weighting * c)


@wp.kernel
def enforce_control_limits(
    control_limits: wp.array(dtype=float, ndim=2),
    control_points: wp.array(dtype=float, ndim=3),
):
    env_id, t_id, control_id = wp.tid()
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    control_points[env_id, t_id, control_id] = wp.clamp(control_points[env_id, t_id, control_id], lo, hi)


@wp.kernel
def pick_best_trajectory(
    rollout_trajectories: wp.array(dtype=float, ndim=3),
    lowest_cost_id: int,
    best_traj: wp.array(dtype=float, ndim=3),
):
    t_id, control_id = wp.tid()
    best_traj[0, t_id, control_id] = rollout_trajectories[lowest_cost_id, t_id, control_id]


@wp.kernel
def interpolate_control_linear(
    control_points: wp.array(dtype=float, ndim=3),
    control_dofs: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    torque_dim: int,
    torques: wp.array(dtype=float),
):
    env_id, control_id = wp.tid()
    t_id = int(t)
    frac = t - wp.floor(t)
    control_left = control_points[env_id, t_id, control_id]
    control_right = control_points[env_id, t_id + 1, control_id]
    torque_id = env_id * torque_dim + control_dofs[control_id]
    action = control_left * (1.0 - frac) + control_right * frac
    torques[torque_id] = action * control_gains[control_id]


@wp.kernel
def compute_prop_wrenches(
    props: wp.array(dtype=Propeller),
    controls: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    prop = props[tid]
    control = controls[tid]
    tf = body_q[prop.body]
    dir = wp.transform_vector(tf, prop.dir)
    force = dir * prop.max_thrust * control
    torque = dir * prop.max_torque * control * prop.turning_direction
    moment_arm = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque += wp.cross(moment_arm, force)
    # Apply angular damping.
    torque *= 0.8
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(torque, force))


def define_propeller(
    drone: int,
    pos: wp.vec3,
    fps: float,
    thrust: float = 0.109919,
    power: float = 0.040164,
    diameter: float = 0.2286,
    height: float = 0.01,
    max_rpm: float = 6396.667,
    turning_direction: float = 1.0,
):
    # Air density at sea level.
    air_density = 1.225  # kg / m^3

    rps = max_rpm / fps
    max_speed = rps * wp.TAU  # radians / sec
    rps_square = rps**2

    prop = Propeller()
    prop.body = drone
    prop.pos = pos
    prop.dir = wp.vec3(0.0, 1.0, 0.0)
    prop.thrust = thrust
    prop.power = power
    prop.diameter = diameter
    prop.height = height
    prop.max_rpm = max_rpm
    prop.max_thrust = thrust * air_density * rps_square * diameter**4
    prop.max_torque = power * air_density * rps_square * diameter**5 / wp.TAU
    prop.turning_direction = turning_direction
    prop.max_speed_square = max_speed**2

    return prop


class Drone:
    def __init__(
        self,
        name: str,
        fps: float,
        trajectory_shape: Tuple[int, int],
        variation_count: int = 1,
        size: float = 1.0,
        requires_grad: bool = False,
        state_count: Optional[int] = None,
    ) -> None:
        self.variation_count = variation_count
        self.requires_grad = requires_grad

        # Current tick of the simulation, including substeps.
        self.sim_tick = 0

        # Initialize the helper to build a physics scene.
        builder = wp.sim.ModelBuilder()
        builder.rigid_contact_margin = 0.05

        # Initialize the rigid bodies, propellers, and colliders.
        props = []
        colliders = []
        crossbar_length = size
        crossbar_height = size * 0.05
        crossbar_width = size * 0.05
        carbon_fiber_density = 1750.0  # kg / m^3
        for i in range(variation_count):
            # Register the drone as a rigid body in the simulation model.
            body = builder.add_body(name=f"{name}_{i}")

            # Define the shapes making up the drone's rigid body.
            builder.add_shape_box(
                body,
                hx=crossbar_length,
                hy=crossbar_height,
                hz=crossbar_width,
                density=carbon_fiber_density,
                collision_group=i,
            )
            builder.add_shape_box(
                body,
                hx=crossbar_width,
                hy=crossbar_height,
                hz=crossbar_length,
                density=carbon_fiber_density,
                collision_group=i,
            )

            # Initialize the propellers.
            props.extend(
                (
                    define_propeller(
                        body,
                        wp.vec3(crossbar_length, 0.0, 0.0),
                        fps,
                        turning_direction=-1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(-crossbar_length, 0.0, 0.0),
                        fps,
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(0.0, 0.0, crossbar_length),
                        fps,
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(0.0, 0.0, -crossbar_length),
                        fps,
                        turning_direction=-1.0,
                    ),
                ),
            )

            # Initialize the colliders.
            colliders.append(
                (
                    builder.add_shape_capsule(
                        -1,
                        pos=(0.5, 2.0, 0.5),
                        radius=0.15,
                        half_height=2.0,
                        collision_group=i,
                    ),
                ),
            )
        self.props = wp.array(props, dtype=Propeller)
        self.colliders = wp.array(colliders, dtype=int)

        # Build the model and set-up its properties.
        self.model = builder.finalize(requires_grad=requires_grad)
        self.model.ground = False

        # Initialize the required simulation states.
        if requires_grad:
            self.states = tuple(self.model.state() for _ in range(state_count + 1))
            self.controls = tuple(self.model.control() for _ in range(state_count))
        else:
            # When only running a forward simulation, we don't need to store
            # the history of the states at each step, instead we use double
            # buffering to represent the previous and next states.
            self.states = [self.model.state(), self.model.state()]
            self.controls = (self.model.control(),)

        # create array for the propeller controls
        for control in self.controls:
            control.prop_controls = wp.zeros(len(self.props), dtype=float, requires_grad=requires_grad)

        # Define the trajectories as arrays of control points.
        # The point data has an additional item to support linear interpolation.
        self.trajectories = wp.zeros(
            (variation_count, trajectory_shape[0], trajectory_shape[1]),
            dtype=float,
            requires_grad=requires_grad,
        )

        # Store some miscellaneous info.
        self.body_count = len(builder.body_q)
        self.collider_count = self.colliders.shape[1]
        self.collision_radius = crossbar_length

    @property
    def state(self) -> wp.sim.State:
        return self.states[self.sim_tick if self.requires_grad else 0]

    @property
    def next_state(self) -> wp.sim.State:
        return self.states[self.sim_tick + 1 if self.requires_grad else 1]

    @property
    def control(self) -> wp.sim.Control:
        return self.controls[min(len(self.controls) - 1, self.sim_tick) if self.requires_grad else 0]


class Example:
    def __init__(
        self,
        stage_path="example_drone.usd",
        verbose=False,
        render_rollouts=False,
        drone_path=DEFAULT_DRONE_PATH,
        num_frames=360,
        num_rollouts=16,
        headless=True,
    ) -> None:
        # Number of frames per second.
        self.fps = 60

        # Duration of the simulation in number of frames.
        self.num_frames = num_frames

        # Number of simulation substeps to take per step.
        self.sim_substep_count = 1

        # Delta time between each simulation substep.
        self.frame_dt = 1.0 / self.fps

        # Delta time between each simulation substep.
        self.sim_dt = self.frame_dt / self.sim_substep_count

        # Frame number used for simulation and rendering.
        self.frame = 0

        # Targets positions that the drone will try to reach in turn.
        self.targets = (
            wp.vec3(0.0, 0.5, 1.0),
            wp.vec3(1.0, 0.5, 0.0),
        )

        # Define the index of the active target.
        # We start with -1 since it'll be incremented on the first frame.
        self.target_idx = -1
        # use a Warp array to store the current target so that we can assign
        # a new target to it while retaining the original CUDA graph.
        self.current_target = wp.array([self.targets[self.target_idx + 1]], dtype=wp.vec3)

        # Number of steps to run at each frame for the optimisation pass.
        self.optim_step_count = 20

        # Time steps between control points.
        self.control_point_step = 10

        # Number of control horizon points to interpolate between.
        self.control_point_count = 3

        self.control_point_data_count = self.control_point_count + 1
        self.control_dofs = wp.array((0, 1, 2, 3), dtype=int)
        self.control_dim = len(self.control_dofs)
        self.control_gains = wp.array((0.8,) * self.control_dim, dtype=float)
        self.control_limits = wp.array(((0.1, 1.0),) * self.control_dim, dtype=float)

        drone_size = 0.2

        # Declare the reference drone.
        self.drone = Drone(
            "drone",
            self.fps,
            (self.control_point_data_count, self.control_dim),
            size=drone_size,
        )

        # Declare the drone's rollouts.
        # These allow to run parallel simulations in order to find the best
        # trajectory at each control point.
        self.rollout_count = num_rollouts
        self.rollout_step_count = self.control_point_step * self.control_point_count
        self.rollouts = Drone(
            "rollout",
            self.fps,
            (self.control_point_data_count, self.control_dim),
            variation_count=self.rollout_count,
            size=drone_size,
            requires_grad=True,
            state_count=self.rollout_step_count * self.sim_substep_count,
        )

        self.seed = wp.zeros(1, dtype=int)
        self.rollout_costs = wp.zeros(self.rollout_count, dtype=float, requires_grad=True)

        # Use the Euler integrator for stepping through the simulation.
        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.optimizer = wp.optim.SGD(
            [self.rollouts.trajectories.flatten()],
            lr=1e-2,
            nesterov=False,
            momentum=0.0,
        )

        self.tape = None

        if stage_path:
            if not headless:
                self.renderer = wp.sim.render.SimRendererOpenGL(self.drone.model, stage_path, fps=self.fps)
            else:
                # Helper to render the physics scene as a USD file.
                self.renderer = wp.sim.render.SimRenderer(self.drone.model, stage_path, fps=self.fps)

            if isinstance(self.renderer, warp.sim.render.SimRendererUsd):
                from pxr import UsdGeom

                # Remove the default drone geometries.
                drone_root_prim = self.renderer.stage.GetPrimAtPath("/root/body_0_drone_0")
                for prim in drone_root_prim.GetChildren():
                    self.renderer.stage.RemovePrim(prim.GetPath())

                # Add a reference to the drone geometry.
                drone_prim = self.renderer.stage.OverridePrim(f"{drone_root_prim.GetPath()}/crazyflie")
                drone_prim.GetReferences().AddReference(drone_path)
                drone_xform = UsdGeom.Xform(drone_prim)
                drone_xform.AddTranslateOp().Set((0.0, -0.05, 0.0))
                drone_xform.AddRotateYOp().Set(45.0)
                drone_xform.AddScaleOp().Set((drone_size * 20.0,) * 3)

                # Get the propellers to spin
                for turning_direction in ("cw", "ccw"):
                    spin = 100.0 * 360.0 * self.num_frames / self.fps
                    spin = spin if turning_direction == "ccw" else -spin
                    for side in ("back", "front"):
                        prop_prim = self.renderer.stage.OverridePrim(
                            f"{drone_prim.GetPath()}/propeller_{turning_direction}_{side}"
                        )
                        prop_xform = UsdGeom.Xform(prop_prim)
                        rot = prop_xform.AddRotateYOp()
                        rot.Set(0.0, 0.0)
                        rot.Set(spin, self.num_frames)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        self.optim_graph = None

        self.render_rollouts = render_rollouts
        self.verbose = verbose

    def update_drone(self, drone: Drone) -> None:
        drone.state.clear_forces()

        wp.launch(
            interpolate_control_linear,
            dim=(
                drone.variation_count,
                self.control_dim,
            ),
            inputs=(
                drone.trajectories,
                self.control_dofs,
                self.control_gains,
                drone.sim_tick / (self.sim_substep_count * self.control_point_step),
                self.control_dim,
            ),
            outputs=(drone.control.prop_controls,),
        )

        wp.launch(
            compute_prop_wrenches,
            dim=len(drone.props),
            inputs=(
                drone.props,
                drone.control.prop_controls,
                drone.state.body_q,
                drone.model.body_com,
            ),
            outputs=(drone.state.body_f,),
        )

        self.integrator.simulate(
            drone.model,
            drone.state,
            drone.next_state,
            self.sim_dt,
            drone.control,
        )

        drone.sim_tick += 1

    def forward(self):
        # Evaluate the rollouts with their costs.
        self.rollouts.sim_tick = 0
        self.rollout_costs.zero_()
        wp.launch(
            replicate_states,
            dim=self.rollout_count,
            inputs=(
                self.drone.state.body_q,
                self.drone.state.body_qd,
                self.drone.body_count,
            ),
            outputs=(
                self.rollouts.state.body_q,
                self.rollouts.state.body_qd,
            ),
        )

        for i in range(self.rollout_step_count):
            for _ in range(self.sim_substep_count):
                self.update_drone(self.rollouts)

            wp.launch(
                drone_cost,
                dim=self.rollout_count,
                inputs=(
                    self.rollouts.state.body_q,
                    self.rollouts.state.body_qd,
                    self.current_target,
                    self.rollouts.control.prop_controls,
                    i,
                    self.rollout_step_count,
                    1e3,
                ),
                outputs=(self.rollout_costs,),
            )
            wp.launch(
                collision_cost,
                dim=(
                    self.rollout_count,
                    self.rollouts.collider_count,
                ),
                inputs=(
                    self.rollouts.state.body_q,
                    self.rollouts.colliders,
                    self.rollouts.model.shape_transform,
                    self.rollouts.model.shape_geo,
                    self.rollouts.collision_radius,
                    1e4,
                ),
                outputs=(self.rollout_costs,),
            )

    def step_optimizer(self):
        if self.optim_graph is None:
            self.tape = wp.Tape()
            with self.tape:
                self.forward()
            self.rollout_costs.grad.fill_(1.0)
            self.tape.backward()
        else:
            wp.capture_launch(self.optim_graph)

        self.optimizer.step([self.rollouts.trajectories.grad.flatten()])

        # Enforce limits on the control points.
        wp.launch(
            enforce_control_limits,
            dim=self.rollouts.trajectories.shape,
            inputs=(self.control_limits,),
            outputs=(self.rollouts.trajectories,),
        )
        self.tape.zero()

    def step(self):
        if self.frame % int((self.num_frames / len(self.targets))) == 0:
            if self.verbose:
                print(f"Choosing new flight target: {self.target_idx + 1}")

            self.target_idx += 1
            self.target_idx %= len(self.targets)

            # Assign the new target to the current target array.
            self.current_target.assign([self.targets[self.target_idx]])

        if self.use_cuda_graph and self.optim_graph is None:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.rollout_costs.grad.fill_(1.0)
                self.tape.backward()
            self.optim_graph = capture.graph

        # Sample control waypoints around the nominal trajectory.
        noise_scale = 0.15
        wp.launch(
            sample_gaussian,
            dim=(
                self.rollouts.trajectories.shape[0] - 1,
                self.rollouts.trajectories.shape[1],
                self.rollouts.trajectories.shape[2],
            ),
            inputs=(
                self.drone.trajectories,
                noise_scale,
                self.control_point_data_count,
                self.control_dim,
                self.control_limits,
                self.seed,
            ),
            outputs=(self.rollouts.trajectories,),
        )

        wp.launch(
            increment_seed,
            dim=1,
            inputs=(),
            outputs=(self.seed,),
        )

        for _ in range(self.optim_step_count):
            self.step_optimizer()

        # Pick the best trajectory.
        wp.synchronize()
        lowest_cost_id = np.argmin(self.rollout_costs.numpy())
        wp.launch(
            pick_best_trajectory,
            dim=(
                self.control_point_data_count,
                self.control_dim,
            ),
            inputs=(
                self.rollouts.trajectories,
                lowest_cost_id,
            ),
            outputs=(self.drone.trajectories,),
        )
        self.rollouts.trajectories[-1].assign(self.drone.trajectories[0])

        # Simulate the drone.
        self.drone.sim_tick = 0
        for _ in range(self.sim_substep_count):
            self.update_drone(self.drone)

            # Swap the drone's states.
            (self.drone.states[0], self.drone.states[1]) = (self.drone.states[1], self.drone.states[0])

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.frame / self.fps)
        self.renderer.render(self.drone.state)

        # Render a sphere as the current target.
        self.renderer.render_sphere(
            "target",
            self.targets[self.target_idx],
            wp.quat_identity(),
            0.05,
            color=(1.0, 0.0, 0.0),
        )

        # Render the rollout trajectories.
        if self.render_rollouts:
            costs = self.rollout_costs.numpy()

            positions = np.array([x.body_q.numpy()[:, :3] for x in self.rollouts.states])

            min_cost = np.min(costs)
            max_cost = np.max(costs)
            for i in range(self.rollout_count):
                # Flip colors, so red means best trajectory, blue worst.
                color = wp.render.bourke_color_map(-max_cost, -min_cost, -costs[i])
                self.renderer.render_line_strip(
                    name=f"rollout_{i}",
                    vertices=positions[:, i],
                    color=color,
                    radius=0.001,
                )

        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_drone.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=360, help="Total number of frames.")
    parser.add_argument("--num_rollouts", type=int, default=16, help="Number of drone rollouts.")
    parser.add_argument(
        "--drone_path",
        type=str,
        default=os.path.join(warp.examples.get_asset_directory(), "crazyflie.usd"),
        help="Path to the USD file to use as the reference for the drone prim in the output stage.",
    )
    parser.add_argument("--render_rollouts", action="store_true", help="Add rollout trajectories to the output stage.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            verbose=args.verbose,
            render_rollouts=args.render_rollouts,
            drone_path=args.drone_path,
            num_frames=args.num_frames,
            num_rollouts=args.num_rollouts,
            headless=args.headless,
        )
        for _i in range(args.num_frames):
            example.step()
            example.render()
            example.frame += 1

            loss = np.min(example.rollout_costs.numpy())
            print(f"[{example.frame:3d}/{example.num_frames}] loss={loss:.8f}")

        if example.renderer is not None:
            example.renderer.save()
