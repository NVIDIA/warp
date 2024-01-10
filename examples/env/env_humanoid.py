# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Humanoid environment
#
# Shows how to set up a simulation of a rigid-body Humanoid articulation based
# on the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import math
import warp as wp
import warp.sim

from environment import Environment, run_env


class HumanoidEnvironment(Environment):
    sim_name = "env_humanoid"
    env_offset = (2.0, 0.0, 2.0)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    xpbd_settings = dict(
        iterations=2,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.5,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    def create_articulation(self, builder):
        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "../assets/nv_humanoid.xml"),
            builder,
            stiffness=0.0,
            damping=0.1,
            armature=0.007,
            armature_scale=10.0,
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e2,
            contact_mu=0.5,
            contact_restitution=0.0,
            limit_ke=1.0e2,
            limit_kd=1.0e1,
            enable_self_collisions=True,
            up_axis="y",
        )

        builder.joint_q[:7] = [0.0, 1.7, 0.0, *wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)]


if __name__ == "__main__":
    run_env(HumanoidEnvironment)
