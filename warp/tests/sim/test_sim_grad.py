# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import platform
import unittest

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from warp.tests.unittest_utils import *


@wp.kernel
def evaluate_loss(
    joint_q: wp.array(dtype=float),
    weighting: float,
    target: float,
    # output
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    d = (target - joint_q[tid * 2 + 1]) ** 2.0
    wp.atomic_add(loss, 0, weighting * d)


@wp.kernel
def assign_action(action: wp.array(dtype=float), joint_act: wp.array(dtype=float)):
    tid = wp.tid()
    joint_act[2 * tid] = action[tid]


@wp.kernel
def assign_force(action: wp.array(dtype=float), body_f: wp.array(dtype=wp.spatial_vector)):
    tid = wp.tid()
    body_f[2 * tid] = wp.spatial_vector(0.0, 0.0, 0.0, action[tid], 0.0, 0.0)


def gradcheck(func, inputs, device, eps=1e-1, tol=1e-2, print_grad=False):
    """
    Checks that the gradient of the Warp kernel is correct by comparing it to the
    numerical gradient computed using finite differences.
    """

    def f(xs):
        # call the kernel without taping for finite differences
        wp_xs = [wp.array(xs[i], ndim=1, dtype=inputs[i].dtype, device=device) for i in range(len(inputs))]
        output = func(*wp_xs)
        return output.numpy()[0]

    # compute analytical gradient
    tape = wp.Tape()
    with tape:
        output = func(*inputs)

    tape.backward(loss=output)

    # compute numerical gradient
    np_xs = []
    for i in range(len(inputs)):
        np_xs.append(inputs[i].numpy().flatten().copy())

    for i in range(len(inputs)):
        fd_grad = np.zeros_like(np_xs[i])
        for j in range(len(np_xs[i])):
            np_xs[i][j] += eps
            y1 = f(np_xs)
            np_xs[i][j] -= 2 * eps
            y2 = f(np_xs)
            np_xs[i][j] += eps
            fd_grad[j] = (y1 - y2) / (2 * eps)

        # compare gradients
        ad_grad = tape.gradients[inputs[i]].numpy()
        if print_grad:
            print("grad ad:", ad_grad)
            print("grad fd:", fd_grad)
        assert_np_equal(ad_grad, fd_grad, tol=tol)
        # ensure the signs match
        assert np.allclose(ad_grad * fd_grad > 0, True)

    tape.zero()


def test_sphere_pushing_on_rails(
    test,
    device,
    joint_type,
    integrator_type,
    apply_force=False,
    static_contacts=True,
    print_grad=False,
):
    if platform.system() == "Darwin":
        test.skipTest("Crashes on Mac runners")

    # Two spheres on a rail (prismatic or D6 joint), one is pushed, the other is passive.
    # The absolute distance to a target is measured and gradients are compared for
    # a push that is too far and too close.
    num_envs = 2
    num_steps = 150
    sim_substeps = 10
    dt = 1 / 30

    target = 3.0

    if integrator_type == 0:
        contact_ke = 1e3
        contact_kd = 1e1
    else:
        contact_ke = 1e3
        contact_kd = 1e1

    complete_builder = wp.sim.ModelBuilder()

    complete_builder.default_shape_ke = contact_ke
    complete_builder.default_shape_kd = contact_kd

    for _ in range(num_envs):
        builder = wp.sim.ModelBuilder(gravity=0.0)

        builder.default_shape_ke = complete_builder.default_shape_ke
        builder.default_shape_kd = complete_builder.default_shape_kd

        b0 = builder.add_body(name="pusher")
        builder.add_shape_sphere(b0, radius=0.4, density=100.0)

        b1 = builder.add_body(name="passive")
        builder.add_shape_sphere(b1, radius=0.47, density=100.0)

        if joint_type == 0:
            builder.add_joint_prismatic(-1, b0)
            builder.add_joint_prismatic(-1, b1)
        else:
            builder.add_joint_d6(-1, b0, linear_axes=[wp.sim.JointAxis((1.0, 0.0, 0.0))])
            builder.add_joint_d6(-1, b1, linear_axes=[wp.sim.JointAxis((1.0, 0.0, 0.0))])

        builder.joint_q[-2:] = [0.0, 2.0]
        complete_builder.add_builder(builder)

    assert complete_builder.body_count == 2 * num_envs
    assert complete_builder.joint_count == 2 * num_envs
    assert set(complete_builder.shape_collision_group) == set(range(1, num_envs + 1))

    complete_builder.gravity = 0.0
    model = complete_builder.finalize(device=device, requires_grad=True)
    model.ground = False
    model.joint_attach_ke = 32000.0 * 16
    model.joint_attach_kd = 500.0 * 4

    model.shape_geo.scale.requires_grad = False
    model.shape_geo.thickness.requires_grad = False

    if static_contacts:
        wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, model)
        model.rigid_contact_margin = 10.0
        state = model.state()
        wp.sim.collide(model, state)

    if integrator_type == 0:
        integrator = wp.sim.FeatherstoneIntegrator(model, update_mass_matrix_every=num_steps * sim_substeps)
    elif integrator_type == 1:
        integrator = wp.sim.SemiImplicitIntegrator()
        sim_substeps *= 5
    else:
        integrator = wp.sim.XPBDIntegrator(iterations=2, rigid_contact_relaxation=1.0)

    # renderer = wp.sim.render.SimRendererOpenGL(model, "test_sim_grad.usd", scaling=1.0)
    renderer = None
    render_time = 0.0

    if renderer:
        renderer.render_sphere("target", pos=wp.vec3(target, 0, 0), rot=wp.quat_identity(), radius=0.1, color=(1, 0, 0))

    def rollout(action: wp.array) -> wp.array:
        nonlocal render_time
        states = [model.state() for _ in range(num_steps * sim_substeps + 1)]

        wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, states[0])

        control_active = model.control()
        control_nop = model.control()

        if not apply_force:
            wp.launch(
                assign_action,
                dim=num_envs,
                inputs=[action],
                outputs=[control_active.joint_act],
                device=model.device,
            )

        i = 0
        for step in range(num_steps):
            state = states[i]
            if not static_contacts:
                wp.sim.collide(model, state)
            if apply_force:
                control = control_nop
            else:
                control = control_active if step < 10 else control_nop
            if renderer:
                renderer.begin_frame(render_time)
                renderer.render(state)
                renderer.end_frame()
                render_time += dt
            for _ in range(sim_substeps):
                state = states[i]
                next_state = states[i + 1]
                if apply_force and step < 10:
                    wp.launch(
                        assign_force,
                        dim=num_envs,
                        inputs=[action],
                        outputs=[state.body_f],
                        device=model.device,
                    )
                integrator.simulate(model, state, next_state, dt / sim_substeps, control)
                i += 1

        if not isinstance(integrator, wp.sim.FeatherstoneIntegrator):
            # compute generalized coordinates
            wp.sim.eval_ik(model, states[-1], states[-1].joint_q, states[-1].joint_qd)

        loss = wp.zeros(1, requires_grad=True, device=device)
        weighting = 1.0
        wp.launch(
            evaluate_loss,
            dim=num_envs,
            inputs=[states[-1].joint_q, weighting, target],
            outputs=[loss],
            device=model.device,
        )

        # if renderer:
        #     renderer.save()

        return loss

    action_too_far = wp.array(
        [80.0 for _ in range(num_envs)],
        device=device,
        dtype=wp.float32,
        requires_grad=True,
    )
    tol = 2e-1
    if isinstance(integrator, wp.sim.XPBDIntegrator):
        # Euler, XPBD do not yield as accurate gradients, but at least the
        # signs should match
        tol = 0.1
    gradcheck(rollout, [action_too_far], device=device, eps=0.2, tol=tol, print_grad=print_grad)

    action_too_close = wp.array(
        [40.0 for _ in range(num_envs)],
        device=device,
        dtype=wp.float32,
        requires_grad=True,
    )
    gradcheck(rollout, [action_too_close], device=device, eps=0.2, tol=tol, print_grad=print_grad)


devices = get_test_devices()


class TestSimGradients(unittest.TestCase):
    pass


for jt_type, jt_name in enumerate(["prismatic", "d6"]):
    test_name = f"test_sphere_pushing_on_rails_{jt_name}"

    def test_fn(self, device, jt_type=jt_type, int_type=1):
        return test_sphere_pushing_on_rails(
            self, device, jt_type, int_type, apply_force=True, static_contacts=True, print_grad=False
        )

    add_function_test(TestSimGradients, test_name, test_fn, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
