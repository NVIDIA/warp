# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import math

# include parent path
import os
import sys
from warp.utils import ScopedTimer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Bounce:

    # seconds
    sim_duration = 0.6

    # control frequency
    frame_dt = 1.0/60.0
    frame_steps = int(sim_duration/frame_dt)

    # sim frequency
    sim_substeps = 8
    sim_steps = frame_steps * sim_substeps
    sim_dt = frame_dt / sim_substeps
    sim_time = 0.0

    render_time = 0.0

    train_iters = 500
    train_rate = 0.01

    def __init__(self, render=True, profile=False, adapter='cpu'):

        builder = wp.sim.ModelBuilder()

        builder.add_particle(pos=(0, 1.0, 0.0), vel=(5.0, -5.0, 0.0), mass=1.0)
        builder.add_shape_box(body=-1, pos=(2.0, 1.0, 0.0), hx=0.25, hy=1.0, hz=1.0)

        self.device = adapter
        self.profile = profile

        self.model = builder.finalize(adapter)
        self.model.ground = True

        self.model.soft_contact_ke = 1.e+4
        self.model.soft_contact_kf = 0.0
        self.model.soft_contact_kd = 1.e+1
        self.model.soft_contact_mu = 0.25
        self.model.soft_contact_margin = 10.0
                
        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = (-2.0, 1.0, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, device=adapter, requires_grad=True)  

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps+1):
            self.states.append(self.model.state(requires_grad=True))

        # one-shot contact creation (valid if we're doing simple collision against a non-changing normal)
        wp.sim.collide(self.model, self.states[0])

        if (self.render):
            self.stage = wp.sim.render.SimRenderer(self.model, "tests/outputs/example_sim_grad_bounce.usd")


    @wp.kernel
    def loss_kernel(pos: wp.array(dtype=wp.vec3),
                    target: wp.vec3, 
                    loss: wp.array(dtype=float)):

        # distance to target
        loss[0] = wp.length(pos[0]-target)

    @wp.kernel
    def step_kernel(x: wp.array(dtype=wp.vec3),
                    grad: wp.array(dtype=wp.vec3),
                    alpha: float):

        # gradient descent step
        x[0] = x[0] - grad[0]*alpha


    def compute_loss(self):

        # run control loop
        for i in range(self.sim_steps):

            self.states[i].clear_forces()

            self.integrator.simulate(self.model, 
                                     self.states[i], 
                                     self.states[i+1], 
                                     self.sim_dt)
        
        # compute loss on final state
        wp.launch(self.loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss], device=self.device)

        return self.loss

    def render(self):

        for i in range(0, self.sim_steps, self.sim_substeps):

            self.stage.begin_frame(self.render_time)
            self.stage.render(self.states[i])
            self.stage.render_box(pos=self.target, rot=wp.quat_identity(), extents=(0.1, 0.1, 0.1), name="target")
            self.stage.end_frame()

            self.render_time += self.frame_dt

        self.stage.save()

    def check_grad(self):

        param = self.states[0].particle_qd

        # initial value
        x_c = param.numpy().flatten()

        # compute numeric gradient
        x_grad_numeric = np.zeros_like(x_c)

        for i in range(len(x_c)):
                
            eps = 1.e-3

            step = np.zeros_like(x_c)
            step[i] = eps

            x_1 = x_c + step
            x_0 = x_c - step

            param.assign(x_1)
            l_1 = self.compute_loss().numpy()[0]

            param.assign(x_0)
            l_0 = self.compute_loss().numpy()[0]

            dldx = (l_1-l_0)/(eps*2.0)

            x_grad_numeric[i] = dldx

        # reset initial state
        param.assign(x_c)

        # compute analytic gradient
        tape = wp.Tape()
        with tape:
            l = self.compute_loss()

        tape.backward(l)

        x_grad_analytic = tape.gradients[param]

        print(f"numeric grad: {x_grad_numeric}")
        print(f"analytic grad: {x_grad_analytic}")



    def train(self, mode='gd'):

        tape = wp.Tape()

        for i in range(self.train_iters):
   
            with ScopedTimer("Forward", active=self.profile):
                with tape:
                    self.compute_loss()

            with ScopedTimer("Backward", active=self.profile):
                tape.backward(self.loss)

            with ScopedTimer("Render", active=self.profile):
                self.render()

            with ScopedTimer("Step", active=self.profile):
                x = self.states[0].particle_qd
                x_grad = tape.gradients[self.states[0].particle_qd]

                print(f"Iter: {i} Loss: {self.loss}")
                print(f"   x: {x} g: {x_grad}")

                wp.launch(self.step_kernel, dim=1, inputs=[x, x_grad, self.train_rate], device=self.device)

            tape.reset()


    def train_graph(self, mode='gd'):

        # capture forward/backward passes
        tape = wp.Tape(capture=True)
        with tape:
            self.compute_loss()

        tape.backward(self.loss)

        # replay and optimize
        for i in range(self.train_iters):
   
            with ScopedTimer("Replay", active=self.profile):
                tape.replay()

            with ScopedTimer("Render", active=self.profile):
                self.render()

            with ScopedTimer("Step", active=self.profile):
                x = self.states[0].particle_qd
                x_grad = tape.gradients[self.states[0].particle_qd]

                print(f"Iter: {i} Loss: {self.loss}")
                
                wp.launch(self.step_kernel, dim=1, inputs=[x, x_grad, self.train_rate], device=self.device)




bounce = Bounce(adapter="cuda", profile=False, render=True)
bounce.check_grad()
bounce.train_graph('gd')
 