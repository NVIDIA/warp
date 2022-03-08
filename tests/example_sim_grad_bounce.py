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

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

# TODO: array +=, -= operators?
# TODO: passing arrays as parameters to simple arguments (e.g.: wp.vec3) has error in backwards pass

@wp.kernel
def loss(pos: wp.array(dtype=wp.vec3),
         target: wp.vec3, 
         loss: wp.array(dtype=float)):

    loss[0] = wp.length(pos[0]-target)

@wp.kernel
def step(x: wp.array(dtype=wp.vec3),
         grad: wp.array(dtype=wp.vec3),
         alpha: float):

    x[0] = x[0] - grad[0]*alpha



class Ballistic:

    sim_duration = 1.5       # seconds

    # control frequency
    frame_dt = 1.0/60.0
    frame_steps = int(sim_duration/frame_dt)

    # sim frequency
    sim_substeps = 16
    sim_steps = frame_steps * sim_substeps
    sim_dt = frame_dt / sim_substeps
    sim_time = 0.0

    render_time = 0.0

    train_iters = 100
    train_rate = 0.05

    def __init__(self, render=True, adapter='cpu'):

        builder = wp.sim.ModelBuilder()

        builder.add_particle(pos=(0, 1.0, 0.0), vel=(0.1, 0.0, 0.0), mass=1.0)

        self.device = adapter
        self.render = render

        self.model = builder.finalize(adapter)
        self.model.ground = True

        self.model.soft_contact_ke = 1.e+3
        self.model.soft_contact_kf = 1.e+1
        self.model.soft_contact_kd = 1.0
        self.model.soft_contact_mu = 0.25

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = (2.0, 1.0, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, device=adapter, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps+1):
            self.states.append(self.model.state(requires_grad=True))

        self.graph = None

        if (self.render):
            self.stage = wp.sim.render.SimRenderer(self.model, "tests/outputs/example_sim_grad_bounce.usda")

    def compute_loss(self):

        # run control loop
        for i in range(self.sim_steps):

            self.states[i].clear_forces()
            self.states[i+1].clear_forces()

            self.integrator.simulate(self.model, 
                                     self.states[i], 
                                     self.states[i+1], 
                                     self.sim_dt)

            if (i%self.sim_substeps == 0) and self.render:

                self.stage.begin_frame(self.render_time)
                self.stage.render(self.states[i])
                self.stage.render_sphere(pos=self.target, rot=wp.quat_identity(), radius=0.1, name="target")
                self.stage.end_frame()

                self.render_time += self.frame_dt

        if (self.render):
            self.stage.save()

        # compute loss on final state
        wp.launch(kernel=loss, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss], device=self.device)

    def train(self, mode='gd'):

        tape = wp.Tape()

        for i in range(self.train_iters):
   
            with ScopedTimer("Forward"):
                with tape:
                    self.compute_loss()

            with ScopedTimer("Backward"):#, detailed=True):
                tape.backward(self.loss)

            with ScopedTimer("Step"):
                x = self.states[0].particle_qd
                x_grad = tape.gradients[self.states[0].particle_qd]

                print(f"Iter: {i} Loss: {self.loss}")
                print(f"   x: {x} g: {x_grad}")

                wp.launch(kernel=step, dim=1, inputs=[x, x_grad, self.train_rate], device=self.device)

            tape.reset()


    def train_graph(self, mode='gd'):

        tape = wp.Tape()

        for i in range(self.train_iters):

            if self.graph == None:

                wp.capture_begin()

                with tape:
                    self.compute_loss()

                tape.backward(self.loss)

                self.graph = wp.capture_end()

            wp.capture_launch(self.graph)

            with ScopedTimer("Step"):
                x = self.states[0].particle_qd
                x_grad = tape.gradients[self.states[0].particle_qd]

                print(f"Iter: {i} Loss: {self.loss}")
                print(f"   x: {x} g: {x_grad}")

                wp.launch(kernel=step, dim=1, inputs=[x, x_grad, self.train_rate], device=self.device)


ballistic = Ballistic(adapter="cuda", render=False)
ballistic.train('gd')
 