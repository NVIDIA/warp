# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Cartpole environment
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a URDF using the Environment class.
# Note this example does not include a trained policy.
#
###########################################################################

import os
import math
import warp as wp
import warp.sim

from environment import Environment, run_env


class CartpoleEnvironment(Environment):
    sim_name = "env_cartpole"
    env_offset = (2.0, 0.0, 2.0)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    activate_ground_plane = False

    show_joints = True

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "../assets/cartpole.urdf"),
            builder,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
        )

        # joint initial positions
        builder.joint_q[-3:] = [0.0, 0.3, 0.0]

        builder.joint_target[:3] = [0.0, 0.0, 0.0]


if __name__ == "__main__":
    run_env(CartpoleEnvironment)
