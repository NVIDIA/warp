# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import unittest

import warp as wp
import warp.sim
from warp.tests.unittest_utils import *


def test_fk_ik(test, device):
    builder = wp.sim.ModelBuilder()

    num_envs = 1

    for i in range(num_envs):
        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "../examples/assets/nv_ant.xml"),
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
    model = builder.finalize(device=device)
    model.ground = True
    model.joint_attach_ke *= 16.0
    model.joint_attach_kd *= 4.0

    state = model.state()

    # save a copy of joint values
    q_fk = model.joint_q.numpy()
    qd_fk = model.joint_qd.numpy()

    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state)

    q_ik = wp.zeros_like(model.joint_q, device=device)
    qd_ik = wp.zeros_like(model.joint_qd, device=device)

    wp.sim.eval_ik(model, state, q_ik, qd_ik)

    assert_np_equal(q_fk, q_ik.numpy(), tol=1e-6)
    assert_np_equal(qd_fk, qd_ik.numpy(), tol=1e-6)


devices = get_test_devices()


class TestSimKinematics(unittest.TestCase):
    pass


add_function_test(TestSimKinematics, "test_fk_ik", test_fk_ik, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
