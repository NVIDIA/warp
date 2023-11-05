# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# USD Environment
#
# Shows how to load a USD file containing USD Physics schema definitions.
#
###########################################################################

import warp as wp
import warp.sim

from environment import Environment, run_env


class UsdEnvironment(Environment):
    sim_name = "env_usd"
    opengl_render_settings = dict(scaling=10.0, draw_grid=True)
    usd_render_settings = dict(scaling=100.0)

    episode_duration = 2.0

    sim_substeps_euler = 64
    sim_substeps_xpbd = 8

    xpbd_settings = dict(
        iterations=10,
        enable_restitution=True,
        joint_linear_relaxation=0.8,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    # USD files define their own ground plane if necessary
    activate_ground_plane = False
    num_envs = 1

    plot_body_coords = False

    def create_articulation(self, builder):
        usd_filename = wp.sim.resolve_usd_from_url(
            "http://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/2022.2.1/Isaac/Robots/Franka/franka_instanceable.usd",
            target_folder_name=".panda_usd_files")
        settings = wp.sim.parse_usd(
            usd_filename,
            builder,
            default_thickness=0.01,
            # ignore collision meshes from Franka robot
            ignore_paths=[".*collisions.*"],
            default_ke=1e6,
        )

        self.frame_dt = 1.0 / settings["fps"]
        if settings["duration"] > 0.0:
            self.episode_duration = settings["duration"]
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.episode_frames = int(self.episode_duration / self.frame_dt)
        self.sim_steps = int(self.episode_duration / self.sim_dt)
        self.sim_time = 0.0
        self.render_time = 0.0
        self.up_axis = settings["up_axis"]

    def before_simulate(self):
        # print some information about the loaded model
        if self.model.shape_count > 0:
            print("shape_transform", self.model.shape_transform.numpy())
            print("geo_scale", self.model.shape_geo.scale.numpy())
        if self.model.joint_count > 0:
            print("joint parent", self.model.joint_parent.numpy())
            print("joint child", self.model.joint_child.numpy())
            if len(self.model.joint_q) > 0:
                print("joint q", self.model.joint_q.numpy())
            if len(self.model.joint_axis) > 0:
                print("joint axis", self.model.joint_axis.numpy())
                print("joint target", self.model.joint_target.numpy())
                print("joint target ke", self.model.joint_target_ke.numpy())
                print("joint target kd", self.model.joint_target_kd.numpy())
                print("joint limit lower", self.model.joint_limit_lower.numpy())
                print("joint limit upper", self.model.joint_limit_upper.numpy())
            print("joint_X_p", self.model.joint_X_p.numpy())
            print("joint_X_c", self.model.joint_X_c.numpy())
        if self.model.body_count > 0:
            print("COM", self.model.body_com.numpy())
            print("Mass", self.model.body_mass.numpy())
            print("Inertia", self.model.body_inertia.numpy())
            print("body_q", self.state.body_q.numpy())


if __name__ == "__main__":
    run_env(UsdEnvironment)
