# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
import warp.sim
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
    # wp.atomic_add(loss, 0, weighting * (target - joint_q[tid * 2 + 1]) ** 2.0)
    d = wp.abs(target - joint_q[tid * 2 + 1])
    wp.atomic_add(loss, 0, weighting * d)


@wp.kernel
def assign_action(action: wp.array(dtype=float), joint_act: wp.array(dtype=float)):
    tid = wp.tid()
    joint_act[2 * tid] = action[tid]


def gradcheck(func, inputs, device, eps=1e-1, tol=1e-2):
    """
    Checks that the gradient of the Warp kernel is correct by comparing it to the
    numerical gradient computed using finite differences.
    """

    def f(xs):
        # call the kernel without taping for finite differences
        wp_xs = [wp.array(xs[i], ndim=1, dtype=inputs[i].dtype, device=device) for i in range(len(inputs))]
        output = func(*wp_xs)
        return output.numpy()[0]

    # compute numerical gradient
    numerical_grad = []
    np_xs = []
    for i in range(len(inputs)):
        np_xs.append(inputs[i].numpy().flatten().copy())
        numerical_grad.append(np.zeros_like(np_xs[-1]))
        inputs[i].requires_grad = True

    for i in range(len(np_xs)):
        for j in range(len(np_xs[i])):
            np_xs[i][j] += eps
            y1 = f(np_xs)
            np_xs[i][j] -= 2 * eps
            y2 = f(np_xs)
            np_xs[i][j] += eps
            numerical_grad[i][j] = (y1 - y2) / (2 * eps)

    # compute analytical gradient
    tape = wp.Tape()
    with tape:
        output = func(*inputs)

    tape.backward(loss=output)

    # compare gradients
    for i in range(len(inputs)):
        grad = tape.gradients[inputs[i]]
        assert_np_equal(grad.numpy(), numerical_grad[i], tol=tol)
        # ensure the signs match
        assert np.allclose(grad.numpy() * numerical_grad[i] > 0, True)

    tape.zero()


def test_box_pushing_on_rails(test, device, joint_type, integrator_type):
    # Two boxes on a rail (prismatic or D6 joint), one is pushed, the other is passive.
    # The absolute distance to a target is measured and gradients are compared for
    # a push that is too far and too close.
    num_envs = 2
    num_steps = 200
    sim_substeps = 2
    dt = 1 / 30

    target = 5.0

    if integrator_type == 0:
        contact_ke = 1e5
        contact_kd = 1e3
    else:
        contact_ke = 1e5
        contact_kd = 1e1

    complete_builder = wp.sim.ModelBuilder()

    complete_builder.default_shape_ke = contact_ke
    complete_builder.default_shape_kd = contact_kd

    for _ in range(num_envs):
        builder = wp.sim.ModelBuilder()

        builder.default_shape_ke = complete_builder.default_shape_ke
        builder.default_shape_kd = complete_builder.default_shape_kd

        b0 = builder.add_body(name="pusher")
        builder.add_shape_box(b0, density=1000.0)

        b1 = builder.add_body(name="passive")
        builder.add_shape_box(b1, hx=0.4, hy=0.4, hz=0.4, density=1000.0)

        if joint_type == 0:
            builder.add_joint_prismatic(-1, b0)
            builder.add_joint_prismatic(-1, b1)
        else:
            builder.add_joint_d6(-1, b0, linear_axes=[wp.sim.JointAxis((1.0, 0.0, 0.0))])
            builder.add_joint_d6(-1, b1, linear_axes=[wp.sim.JointAxis((1.0, 0.0, 0.0))])

        builder.joint_q[-2:] = [0.0, 1.0]
        complete_builder.add_builder(builder)

    assert complete_builder.body_count == 2 * num_envs
    assert complete_builder.joint_count == 2 * num_envs
    assert set(complete_builder.shape_collision_group) == set(range(1, num_envs + 1))

    complete_builder.gravity = 0.0
    model = complete_builder.finalize(device=device, requires_grad=True)
    model.ground = False
    model.joint_attach_ke = 32000.0 * 16
    model.joint_attach_kd = 500.0 * 4

    if integrator_type == 0:
        integrator = wp.sim.FeatherstoneIntegrator(model, update_mass_matrix_every=num_steps * sim_substeps)
    elif integrator_type == 1:
        integrator = wp.sim.SemiImplicitIntegrator()
        sim_substeps *= 5
    else:
        integrator = wp.sim.XPBDIntegrator(iterations=2, rigid_contact_relaxation=1.0)

    # renderer = wp.sim.render.SimRenderer(model, "test_sim_grad.usd", scaling=1.0)
    renderer = None
    render_time = 0.0

    def rollout(action: wp.array) -> wp.array:
        nonlocal render_time
        states = [model.state() for _ in range(num_steps * sim_substeps + 1)]

        if not isinstance(integrator, wp.sim.FeatherstoneIntegrator):
            # apply initial generalized coordinates
            wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, states[0])

        control_active = model.control()
        control_nop = model.control()

        wp.launch(
            assign_action,
            dim=num_envs,
            inputs=[action],
            outputs=[control_active.joint_act],
            device=model.device,
        )

        i = 0
        for step in range(num_steps):
            wp.sim.collide(model, states[i])
            control = control_active if step < 10 else control_nop
            if renderer:
                renderer.begin_frame(render_time)
                renderer.render(states[i])
                renderer.end_frame()
                render_time += dt
            for _ in range(sim_substeps):
                integrator.simulate(model, states[i], states[i + 1], dt / sim_substeps, control)
                i += 1

        if not isinstance(integrator, wp.sim.FeatherstoneIntegrator):
            # compute generalized coordinates
            wp.sim.eval_ik(model, states[-1], states[-1].joint_q, states[-1].joint_qd)

        loss = wp.zeros(1, requires_grad=True, device=device)
        wp.launch(
            evaluate_loss,
            dim=num_envs,
            inputs=[states[-1].joint_q, 1.0, target],
            outputs=[loss],
            device=model.device,
        )

        if renderer:
            renderer.save()

        return loss

    action_too_far = wp.array(
        [5000.0 for _ in range(num_envs)],
        device=device,
        dtype=wp.float32,
        requires_grad=True,
    )
    tol = 1e-2
    if isinstance(integrator, wp.sim.XPBDIntegrator):
        # Euler, XPBD do not yield as accurate gradients, but at least the
        # signs should match
        tol = 0.1
    gradcheck(rollout, [action_too_far], device=device, eps=0.2, tol=tol)

    action_too_close = wp.array(
        [1500.0 for _ in range(num_envs)],
        device=device,
        dtype=wp.float32,
        requires_grad=True,
    )
    gradcheck(rollout, [action_too_close], device=device, eps=0.2, tol=tol)


devices = get_test_devices()


class TestSimGradients(unittest.TestCase):
    pass


for int_type, int_name in enumerate(["featherstone", "semiimplicit"]):
    for jt_type, jt_name in enumerate(["prismatic", "d6"]):
        test_name = f"test_box_pushing_on_rails_{int_name}_{jt_name}"

        def test_fn(self, device, jt_type=jt_type, int_type=int_type):
            return test_box_pushing_on_rails(self, device, jt_type, int_type)

        add_function_test(TestSimGradients, test_name, test_fn, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
