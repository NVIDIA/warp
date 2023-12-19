# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Grad Bounce
#
# Shows how to use Warp to optimize the initial velocity of a particle
# such that it bounces off the wall and floor in order to hit a target.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import os

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    # distance to target
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, stage=None, enable_rendering=True, profile=False, adapter=None, verbose=False):
        self.device = wp.get_device()
        self.verbose = verbose

        # seconds
        sim_duration = 0.6

        # control frequency
        self.frame_dt = 1.0 / 60.0
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 8
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        self.train_iters = 250
        self.train_rate = 0.02

        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2

        builder = wp.sim.ModelBuilder()
        builder.add_particle(pos=wp.vec3(-0.5, 1.0, 0.0), vel=wp.vec3(5.0, -5.0, 0.0), mass=1.0)
        builder.add_shape_box(body=-1, pos=wp.vec3(2.0, 1.0, 0.0), hx=0.25, hy=1.0, hz=1.0, ke=ke, kf=kf, kd=kd, mu=mu)

        self.device = wp.get_device(adapter)
        self.profile = profile

        self.model = builder.finalize(self.device)
        self.model.ground = True

        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_margin = 10.0
        self.model.soft_contact_restitution = 1.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = (-2.0, 1.5, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps + 1):
            self.states.append(self.model.state(requires_grad=True))

        # one-shot contact creation (valid if we're doing simple collision against a constant normal plane)
        wp.sim.collide(self.model, self.states[0])

        self.enable_rendering = enable_rendering
        self.renderer = None
        if self.enable_rendering:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=1.0)

        # capture forward/backward passes
        wp.capture_begin(self.device)
        try:
            self.tape = wp.Tape()
            with self.tape:
                self.compute_loss()
            self.tape.backward(self.loss)
        finally:
            self.graph = wp.capture_end(self.device)

    def compute_loss(self):
        # run control loop
        for i in range(self.sim_steps):
            self.states[i].clear_forces()

            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

        # compute loss on final state
        wp.launch(loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss], device=self.device)

        return self.loss

    def update(self):
        with wp.ScopedTimer("Step", active=self.profile):
            # forward + backward
            wp.capture_launch(self.graph)

            # gradient descent step
            x = self.states[0].particle_qd
            wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate], device=self.device)

            x_grad = self.tape.gradients[self.states[0].particle_qd]

            if self.verbose:
                print(f"Iter: {self.iter} Loss: {self.loss}")
                print(f"    x: {x} g: {x_grad}")

            # clear grads for next iteration
            self.tape.zero()

            self.iter = self.iter + 1

    def render(self):
        if self.enable_rendering:
            with wp.ScopedTimer("Render", active=self.profile):
                # draw trajectory
                traj_verts = [self.states[0].particle_q.numpy()[0].tolist()]

                for i in range(0, self.sim_steps, self.sim_substeps):
                    traj_verts.append(self.states[i].particle_q.numpy()[0].tolist())

                    self.renderer.begin_frame(self.render_time)
                    self.renderer.render(self.states[i])
                    self.renderer.render_box(
                        pos=self.target, rot=wp.quat_identity(), extents=(0.1, 0.1, 0.1), name="target"
                    )
                    self.renderer.render_line_strip(
                        vertices=traj_verts,
                        color=wp.render.bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
                        radius=0.02,
                        name=f"traj_{self.iter-1}",
                    )
                    self.renderer.end_frame()

                    self.render_time += self.frame_dt

    def check_grad(self):
        param = self.states[0].particle_qd

        # initial value
        x_c = param.numpy().flatten()

        # compute numeric gradient
        x_grad_numeric = np.zeros_like(x_c)

        for i in range(len(x_c)):
            eps = 1.0e-3

            step = np.zeros_like(x_c)
            step[i] = eps

            x_1 = x_c + step
            x_0 = x_c - step

            param.assign(x_1)
            l_1 = self.compute_loss().numpy()[0]

            param.assign(x_0)
            l_0 = self.compute_loss().numpy()[0]

            dldx = (l_1 - l_0) / (eps * 2.0)

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

        tape.zero()

    def run(self):
        # replay and optimize
        for i in range(self.train_iters):
            self.update()
            # render every 16 iters
            if i % 16 == 0:
                self.render()

        if self.enable_rendering:
            self.renderer.save()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_grad_bounce.usd")
    example = Example(stage_path, profile=False, enable_rendering=True, verbose=True)
    example.check_grad()
    example.run()
