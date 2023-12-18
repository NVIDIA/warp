# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Kinematics
#
# Tests rigid body forward and backwards kinematics through the
# wp.sim.eval_ik() and wp.sim.eval_fk() methods.
#
###########################################################################

import math
import os

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    def __init__(self, stage, num_envs=1, device=None, verbose=False):
        self.verbose = verbose

        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        self.num_envs = num_envs

        for i in range(num_envs):
            wp.sim.parse_mjcf(
                os.path.join(os.path.dirname(__file__), "assets/nv_ant.xml"),
                builder,
                stiffness=0.0,
                damping=1.0,
                armature=0.1,
                contact_ke=1.0e4,
                contact_kd=1.0e2,
                contact_kf=1.0e2,
                contact_mu=0.75,
                limit_ke=1.0e3,
                limit_kd=1.0e1,
                up_axis="y",
            )

            coord_count = 15
            dof_count = 14

            coord_start = i * coord_count
            dof_start = i * dof_count

            # base
            builder.joint_q[coord_start : coord_start + 3] = [i * 2.0, 0.70, 0.0]
            builder.joint_q[coord_start + 3 : coord_start + 7] = wp.quat_from_axis_angle(
                wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5
            )

            # joints
            builder.joint_q[coord_start + 7 : coord_start + coord_count] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            builder.joint_qd[dof_start + 6 : dof_start + dof_count] = [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True
        self.model.joint_attach_ke *= 16.0
        self.model.joint_attach_kd *= 4.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.renderer = wp.sim.render.SimRenderer(path=stage, model=self.model, scaling=50.0)

        self.frame_dt = 1.0 / 60.0

    def update(self):
        self.state = self.model.state()

        # save a copy of joint values
        q_fk = self.model.joint_q.numpy()
        qd_fk = self.model.joint_qd.numpy()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        q_ik = wp.zeros_like(self.model.joint_q)
        qd_ik = wp.zeros_like(self.model.joint_qd)

        wp.sim.eval_ik(self.model, self.state, q_ik, qd_ik)

        q_err = q_fk - q_ik.numpy()
        qd_err = qd_fk - qd_ik.numpy()

        if self.verbose:
            print(f"q_err = {q_err}")
            print(f"qd_err = {qd_err}")

        assert np.abs(q_err).max() < 1.0e-6
        assert np.abs(qd_err).max() < 1.0e-6

        self.sim_time += self.frame_dt

    def render(self):
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state)
        self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_kinematics.usd")

    example = Example(stage_path, num_envs=1, verbose=True)
    example.update()
    example.render()

    example.renderer.save()
