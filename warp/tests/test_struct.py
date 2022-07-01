# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()

@wp.struct
class Model:
    dt: float
    gravity: wp.vec3
    m: wp.array(dtype=float)

@wp.struct
class State:
    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)

@wp.kernel
def kernel_step(state_in: State, state_out: State, model: Model):
    i = wp.tid()

    state_out.v[i] = state_in.v[i] + model.gravity / model.m[i] * model.dt
    state_out.x[i] = state_in.x[i] + state_out.v[i] * model.dt

def test_step(test, device):
    dim = 5

    dt = 0.01
    gravity = np.array([0, 0, -9.81])

    m = np.ones(dim)

    m_model = wp.array(m, dtype=float, device=device)

    model = Model()
    model.m = m_model
    model.dt = dt
    model.gravity = wp.vec3(0, 0, -9.81)

    np.random.seed(0)
    x = np.random.normal(size=(dim, 3))
    v = np.random.normal(size=(dim, 3))

    x_expected = x + (v + gravity / m[:, None] * dt) * dt

    x_in = wp.array(x, dtype=wp.vec3, device=device)
    v_in = wp.array(v, dtype=wp.vec3, device=device)

    state_in = State()
    state_in.x = x_in
    state_in.v = v_in

    state_out = State()
    state_out.x = wp.empty_like(x_in)
    state_out.v = wp.empty_like(v_in)

    with CheckOutput(test):
        wp.launch(kernel_step, dim=dim, inputs=[state_in, state_out, model], device=device)

    assert_np_equal(state_out.x.numpy(), x_expected, tol=1e-6)


@wp.kernel
def kernel_loss(x: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(loss, 0, x[i][0] * x[i][0] + x[i][1] * x[i][1] + x[i][2] * x[i][2])


def test_step_grad(test, device):
    dim = 5

    dt = 0.01
    gravity = np.array([0, 0, -9.81])

    np.random.seed(0)
    m = np.random.rand(dim) + 0.1

    m_model = wp.array(m, dtype=float, device=device, requires_grad=True)

    model = Model()
    model.m = m_model
    model.dt = dt
    model.gravity = wp.vec3(0, 0, -9.81)

    x = np.random.normal(size=(dim, 3))
    v = np.random.normal(size=(dim, 3))

    x_in = wp.array(x, dtype=wp.vec3, device=device, requires_grad=True)
    v_in = wp.array(v, dtype=wp.vec3, device=device, requires_grad=True)

    state_in = State()
    state_in.x = x_in
    state_in.v = v_in

    state_out = State()
    state_out.x = wp.empty_like(x_in, requires_grad=True)
    state_out.v = wp.empty_like(v_in, requires_grad=True)

    loss = wp.empty(1, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()

    with tape:
        wp.launch(kernel_step, dim=dim, inputs=[state_in, state_out, model], device=device)
        wp.launch(kernel_loss, dim=dim, inputs=[state_out.x, loss], device=device)

    tape.backward(loss)

    dl_dx = 2 * state_out.x.numpy()
    dl_dv = dl_dx * dt

    dv_dm = - gravity * dt / m[:, None] ** 2
    dl_dm = (dl_dv * dv_dm).sum(-1)

    assert_np_equal(state_out.x.grad.numpy(), dl_dx, tol=1e-6)
    assert_np_equal(state_in.x.grad.numpy(), dl_dx, tol=1e-6)
    assert_np_equal(state_out.v.grad.numpy(), dl_dv, tol=1e-6)
    assert_np_equal(state_in.v.grad.numpy(), dl_dv, tol=1e-6)
    assert_np_equal(model.m.grad.numpy(), dl_dm, tol=1e-6)


def register(parent):
    devices = wp.get_devices()
    class TestStruct(parent):
        pass

    add_function_test(TestStruct, "test_step", test_step, devices=devices)
    add_function_test(TestStruct, "test_step_grad", test_step_grad, devices=devices)

    return TestStruct

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
