# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from enum import Enum
from typing import Tuple

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class RenderMode(Enum):
    NONE = "none"
    OPENGL = "opengl"
    USD = "usd"

    def __str__(self):
        return self.value


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"

    def __str__(self):
        return self.value


def compute_env_offsets(num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Y"):
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets


class Environment:
    sim_name: str = "Environment"

    frame_dt = 1.0 / 60.0

    episode_duration = 5.0  # seconds

    # whether to play the simulation indefinitely when using the OpenGL renderer
    continuous_opengl_render: bool = True

    sim_substeps_euler: int = 16
    sim_substeps_xpbd: int = 5

    euler_settings = dict()
    xpbd_settings = dict()

    render_mode: RenderMode = RenderMode.OPENGL
    opengl_render_settings = dict()
    usd_render_settings = dict(scaling=10.0)
    show_rigid_contact_points = False
    contact_points_radius = 1e-3
    show_joints = False
    # whether OpenGLRenderer should render each environment in a separate tile
    use_tiled_rendering = False

    # whether to apply model.joint_q, joint_qd to bodies before simulating
    eval_fk: bool = True

    profile: bool = False

    use_graph_capture: bool = wp.get_preferred_device().is_cuda

    num_envs: int = 100

    activate_ground_plane: bool = True

    integrator_type: IntegratorType = IntegratorType.XPBD

    up_axis: str = "Y"
    gravity: float = -9.81
    env_offset: Tuple[float, float, float] = (1.0, 0.0, 1.0)

    # stiffness and damping for joint attachment dynamics used by Euler
    joint_attach_ke: float = 32000.0
    joint_attach_kd: float = 50.0

    # distance threshold at which contacts are generated
    rigid_contact_margin: float = 0.05

    # whether each environment should have its own collision group
    # to avoid collisions between environments
    separate_collision_group_per_env: bool = True

    plot_body_coords: bool = False
    plot_joint_coords: bool = False

    requires_grad: bool = False

    # control-related definitions, to be updated by derived classes
    control_dim: int = 0

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--integrator",
            help="Type of integrator",
            type=IntegratorType,
            choices=list(IntegratorType),
            default=self.integrator_type.value,
        )
        self.parser.add_argument(
            "--visualizer",
            help="Type of renderer",
            type=RenderMode,
            choices=list(RenderMode),
            default=self.render_mode.value,
        )
        self.parser.add_argument(
            "--num_envs", help="Number of environments to simulate", type=int, default=self.num_envs
        )
        self.parser.add_argument("--profile", help="Enable profiling", type=bool, default=self.profile)

    def parse_args(self):
        args = self.parser.parse_args()
        self.integrator_type = args.integrator
        self.render_mode = args.visualizer
        self.num_envs = args.num_envs
        self.profile = args.profile

    def init(self):
        if self.integrator_type == IntegratorType.EULER:
            self.sim_substeps = self.sim_substeps_euler
        elif self.integrator_type == IntegratorType.XPBD:
            self.sim_substeps = self.sim_substeps_xpbd

        self.episode_frames = int(self.episode_duration / self.frame_dt)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = int(self.episode_duration / self.sim_dt)

        if self.use_tiled_rendering and self.render_mode == RenderMode.OPENGL:
            # no environment offset when using tiled rendering
            self.env_offset = (0.0, 0.0, 0.0)

        builder = wp.sim.ModelBuilder()
        builder.rigid_contact_margin = self.rigid_contact_margin
        try:
            articulation_builder = wp.sim.ModelBuilder()
            self.create_articulation(articulation_builder)
            env_offsets = compute_env_offsets(self.num_envs, self.env_offset, self.up_axis)
            for i in range(self.num_envs):
                xform = wp.transform(env_offsets[i], wp.quat_identity())
                builder.add_builder(
                    articulation_builder, xform, separate_collision_group=self.separate_collision_group_per_env
                )
            self.bodies_per_env = len(articulation_builder.body_q)
        except NotImplementedError:
            # custom simulation setup where something other than an articulation is used
            self.setup(builder)
            self.bodies_per_env = len(builder.body_q)

        self.model = builder.finalize()
        self.device = self.model.device
        if not self.device.is_cuda:
            self.use_graph_capture = False
        self.model.ground = self.activate_ground_plane

        self.model.joint_attach_ke = self.joint_attach_ke
        self.model.joint_attach_kd = self.joint_attach_kd

        # set up current and next state to be used by the integrator
        self.state_0 = None
        self.state_1 = None

        if self.integrator_type == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator(**self.euler_settings)
        elif self.integrator_type == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(**self.xpbd_settings)

        self.renderer = None
        if self.profile:
            self.render_mode = RenderMode.NONE
        if self.render_mode == RenderMode.OPENGL:
            self.renderer = wp.sim.render.SimRendererOpenGL(
                self.model,
                self.sim_name,
                up_axis=self.up_axis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                contact_points_radius=self.contact_points_radius,
                show_joints=self.show_joints,
                **self.opengl_render_settings,
            )
            if self.use_tiled_rendering and self.num_envs > 1:
                floor_id = self.model.shape_count - 1
                # all shapes except the floor
                instance_ids = np.arange(floor_id, dtype=np.int32).tolist()
                shapes_per_env = floor_id // self.num_envs
                additional_instances = []
                if self.activate_ground_plane:
                    additional_instances.append(floor_id)
                self.renderer.setup_tiled_rendering(
                    instances=[
                        instance_ids[i * shapes_per_env : (i + 1) * shapes_per_env] + additional_instances
                        for i in range(self.num_envs)
                    ]
                )
        elif self.render_mode == RenderMode.USD:
            filename = os.path.join(os.path.dirname(__file__), "..", "outputs", self.sim_name + ".usd")
            self.renderer = wp.sim.render.SimRendererUsd(
                self.model,
                filename,
                up_axis=self.up_axis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                **self.usd_render_settings,
            )

    def create_articulation(self, builder):
        raise NotImplementedError

    def setup(self, builder):
        pass

    def customize_model(self, model):
        pass

    def before_simulate(self):
        pass

    def after_simulate(self):
        pass

    def custom_update(self):
        pass

    @property
    def state(self):
        # shortcut to current state
        return self.state_0

    def update(self):
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.custom_update()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self, state=None):
        if self.renderer is not None:
            with wp.ScopedTimer("render", False):
                self.render_time += self.frame_dt
                self.renderer.begin_frame(self.render_time)
                # render state 1 (swapped with state 0 just before)
                self.renderer.render(state or self.state_1)
                self.renderer.end_frame()

    def run(self):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.render_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if self.eval_fk:
            wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.before_simulate()

        if self.renderer is not None:
            self.render(self.state_0)

            if self.render_mode == RenderMode.OPENGL:
                self.renderer.paused = True

        profiler = {}

        if self.use_graph_capture:
            # create update graph
            wp.capture_begin()
            self.update()
            graph = wp.capture_end()

        if self.plot_body_coords:
            q_history = []
            q_history.append(self.state_0.body_q.numpy().copy())
            qd_history = []
            qd_history.append(self.state_0.body_qd.numpy().copy())
            delta_history = []
            delta_history.append(self.state_0.body_deltas.numpy().copy())
            num_con_history = []
            num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())
        if self.plot_joint_coords:
            joint_q_history = []
            joint_q = wp.zeros_like(self.model.joint_q)
            joint_qd = wp.zeros_like(self.model.joint_qd)

        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):
            running = True
            while running:
                for f in range(self.episode_frames):
                    if self.use_graph_capture:
                        wp.capture_launch(graph)
                        self.sim_time += self.frame_dt
                    else:
                        self.update()
                        self.sim_time += self.frame_dt

                        if not self.profile:
                            if self.plot_body_coords:
                                q_history.append(self.state_0.body_q.numpy().copy())
                                qd_history.append(self.state_0.body_qd.numpy().copy())
                                delta_history.append(self.state_0.body_deltas.numpy().copy())
                                num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())

                            if self.plot_joint_coords:
                                wp.sim.eval_ik(self.model, self.state_0, joint_q, joint_qd)
                                joint_q_history.append(joint_q.numpy().copy())

                    self.render()
                    if self.render_mode == RenderMode.OPENGL and self.renderer.has_exit:
                        running = False
                        break

                if not self.continuous_opengl_render or self.render_mode != RenderMode.OPENGL:
                    break

            wp.synchronize()

        self.after_simulate()

        avg_time = np.array(profiler["simulate"]).mean() / self.episode_frames
        avg_steps_second = 1000.0 * float(self.num_envs) / avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        if self.renderer is not None:
            self.renderer.save()

        if self.plot_body_coords:
            import matplotlib.pyplot as plt

            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)
            num_con_history = np.array(num_con_history)

            # find bodies with non-zero mass
            body_indices = np.where(self.model.body_mass.numpy() > 0)[0]
            body_indices = body_indices[:5]  # limit number of bodies to plot

            fig, ax = plt.subplots(len(body_indices), 7, figsize=(10, 10), squeeze=False)
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            for i, j in enumerate(body_indices):
                ax[i, 0].set_title(f"Body {j} Position")
                ax[i, 0].grid()
                ax[i, 1].set_title(f"Body {j} Orientation")
                ax[i, 1].grid()
                ax[i, 2].set_title(f"Body {j} Linear Velocity")
                ax[i, 2].grid()
                ax[i, 3].set_title(f"Body {j} Angular Velocity")
                ax[i, 3].grid()
                ax[i, 4].set_title(f"Body {j} Linear Delta")
                ax[i, 4].grid()
                ax[i, 5].set_title(f"Body {j} Angular Delta")
                ax[i, 5].grid()
                ax[i, 6].set_title(f"Body {j} Num Contacts")
                ax[i, 6].grid()
                ax[i, 0].plot(q_history[:, j, :3])
                ax[i, 1].plot(q_history[:, j, 3:])
                ax[i, 2].plot(qd_history[:, j, 3:])
                ax[i, 3].plot(qd_history[:, j, :3])
                ax[i, 4].plot(delta_history[:, j, 3:])
                ax[i, 5].plot(delta_history[:, j, :3])
                ax[i, 6].plot(num_con_history[:, j])
                ax[i, 0].set_xlim(0, self.sim_steps)
                ax[i, 1].set_xlim(0, self.sim_steps)
                ax[i, 2].set_xlim(0, self.sim_steps)
                ax[i, 3].set_xlim(0, self.sim_steps)
                ax[i, 4].set_xlim(0, self.sim_steps)
                ax[i, 5].set_xlim(0, self.sim_steps)
                ax[i, 6].set_xlim(0, self.sim_steps)
                ax[i, 6].yaxis.get_major_locator().set_params(integer=True)
            plt.show()

        if self.plot_joint_coords:
            import matplotlib.pyplot as plt

            joint_q_history = np.array(joint_q_history)
            dof_q = joint_q_history.shape[1]
            ncols = int(np.ceil(np.sqrt(dof_q)))
            nrows = int(np.ceil(dof_q / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True,
            )

            joint_id = 0
            joint_type_names = {
                wp.sim.JOINT_BALL: "ball",
                wp.sim.JOINT_REVOLUTE: "hinge",
                wp.sim.JOINT_PRISMATIC: "slide",
                wp.sim.JOINT_UNIVERSAL: "universal",
                wp.sim.JOINT_COMPOUND: "compound",
                wp.sim.JOINT_FREE: "free",
                wp.sim.JOINT_FIXED: "fixed",
                wp.sim.JOINT_DISTANCE: "distance",
                wp.sim.JOINT_D6: "D6",
            }
            joint_lower = self.model.joint_limit_lower.numpy()
            joint_upper = self.model.joint_limit_upper.numpy()
            joint_type = self.model.joint_type.numpy()
            while joint_id < len(joint_type) - 1 and joint_type[joint_id] == wp.sim.JOINT_FIXED:
                # skip fixed joints
                joint_id += 1
            q_start = self.model.joint_q_start.numpy()
            qd_start = self.model.joint_qd_start.numpy()
            qd_i = qd_start[joint_id]
            for dim in range(ncols * nrows):
                ax = axes[dim // ncols, dim % ncols]
                if dim >= dof_q:
                    ax.axis("off")
                    continue
                ax.grid()
                ax.plot(joint_q_history[:, dim])
                if joint_type[joint_id] != wp.sim.JOINT_FREE:
                    lower = joint_lower[qd_i]
                    if abs(lower) < 2 * np.pi:
                        ax.axhline(lower, color="red")
                    upper = joint_upper[qd_i]
                    if abs(upper) < 2 * np.pi:
                        ax.axhline(upper, color="red")
                joint_name = joint_type_names[joint_type[joint_id]]
                ax.set_title(f"$\\mathbf{{q_{{{dim}}}}}$ ({self.model.joint_name[joint_id]} / {joint_name} {joint_id})")
                if joint_id < self.model.joint_count - 1 and q_start[joint_id + 1] == dim + 1:
                    joint_id += 1
                    qd_i = qd_start[joint_id]
                else:
                    qd_i += 1
            plt.tight_layout()
            plt.show()

        return 1000.0 * float(self.num_envs) / avg_time


def run_env(Demo):
    demo = Demo()
    demo.parse_args()
    if demo.profile:
        import matplotlib.pyplot as plt

        env_count = 2
        env_times = []
        env_size = []

        for i in range(15):
            demo.num_envs = env_count
            demo.init()
            steps_per_second = demo.run()

            env_size.append(env_count)
            env_times.append(steps_per_second)

            env_count *= 2

        # dump times
        for i in range(len(env_times)):
            print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

        # plot
        plt.figure(1)
        plt.plot(env_size, env_times)
        plt.xscale("log")
        plt.xlabel("Number of Envs")
        plt.yscale("log")
        plt.ylabel("Steps/Second")
        plt.show()
    else:
        demo.init()
        return demo.run()
