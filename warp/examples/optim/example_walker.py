# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Walker
#
# Trains a tetrahedral mesh quadruped to run. Feeds 8 time-varying input
# phases as inputs into a single layer fully connected network with a tanh
# activation function. Interprets the output of the network as tet
# activations, which are fed into the wp.sim soft mesh model. This is
# simulated forward in time and then evaluated based on the center of mass
# momentum of the mesh.
#
###########################################################################

import os
import math

import warp as wp
import warp.sim
import warp.optim
import warp.sim.render

import numpy as np
from pxr import Usd, UsdGeom

wp.init()


@wp.kernel
def loss_kernel(com: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    tid = wp.tid()
    vx = com[tid][0]
    vy = com[tid][1]
    vz = com[tid][2]
    delta = wp.sqrt(vz * vz) + wp.sqrt(vy * vy) - vx

    wp.atomic_add(loss, 0, delta)


@wp.kernel
def com_kernel(velocities: wp.array(dtype=wp.vec3), n: int, com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v = velocities[tid]
    a = v / wp.float32(n)
    wp.atomic_add(com, 0, a)


@wp.kernel
def compute_phases(phases: wp.array(dtype=float), sim_time: float):
    tid = wp.tid()
    phases[tid] = wp.sin(phase_freq * sim_time + wp.float32(tid) * phase_step)


@wp.kernel
def activation_function(tet_activations: wp.array(dtype=float), activation_inputs: wp.array(dtype=float)):
    tid = wp.tid()
    activation = wp.tanh(activation_inputs[tid])
    tet_activations[tid] = activation_strength * activation


phase_count = 8
phase_step = wp.constant((2.0 * math.pi) / phase_count)
phase_freq = wp.constant(5.0)
activation_strength = wp.constant(0.3)


class Example:
    def __init__(self, stage=None, profile=False, verbose=False):
        self.profile = profile
        self.verbose = verbose

        # sim settings
        self.episode_duration = 5.0  # seconds

        self.frame_dt = 1.0 / 60.0
        self.frame_count = int(self.episode_duration / self.frame_dt)

        self.sim_substeps = 80
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.iter = 0
        self.train_rate = 0.025

        self.phase_count = phase_count

        self.render_time = 0.0

        # bear
        asset_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "../assets/bear.usd"))

        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/bear"))
        points = geom.GetPointsAttr().Get()

        xform = geom.ComputeLocalToWorldTransform(0.0)
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        self.points = points
        self.tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        # sim model
        builder = wp.sim.ModelBuilder()
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=2.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=self.points,
            indices=self.tet_indices,
            density=1.0,
            k_mu=2000.0,
            k_lambda=2000.0,
            k_damp=2.0,
            tri_ke=0.0,
            tri_ka=1e-8,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
        )

        # finalize model
        self.model = builder.finalize(requires_grad=True)
        self.control = self.model.control()

        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_kf = 10.0
        self.model.soft_contact_mu = 0.7

        radii = wp.zeros(self.model.particle_count, dtype=float)
        radii.fill_(0.05)
        self.model.particle_radius = radii
        self.model.ground = True

        # allocate sim states
        self.states = []
        for i in range(self.frame_count * self.sim_substeps + 1):
            self.states.append(self.model.state(requires_grad=True))

        # initialize the integrator.
        self.integrator = wp.sim.SemiImplicitIntegrator()

        # model input
        self.phases = []
        for i in range(self.frame_count):
            self.phases.append(wp.zeros(self.phase_count, dtype=float, requires_grad=True))

        # single layer linear network
        rng = np.random.default_rng(42)
        k = 1.0 / self.phase_count
        weights = rng.uniform(-np.sqrt(k), np.sqrt(k), (self.model.tet_count, self.phase_count))
        self.weights = wp.array(weights, dtype=float, requires_grad=True)
        self.bias = wp.zeros(self.model.tet_count, dtype=float, requires_grad=True)

        # tanh activation layer
        self.activation_inputs = []
        self.tet_activations = []
        for i in range(self.frame_count):
            self.activation_inputs.append(wp.zeros(self.model.tet_count, dtype=float, requires_grad=True))
            self.tet_activations.append(wp.zeros(self.model.tet_count, dtype=float, requires_grad=True))

        # optimization
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.coms = []
        for i in range(self.frame_count):
            self.coms.append(wp.zeros(1, dtype=wp.vec3, requires_grad=True))
        self.optimizer = warp.optim.Adam([self.weights.flatten()], lr=self.train_rate)

        # rendering
        self.renderer = None
        if stage:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage)

        # capture forward/backward passes
        self.use_graph = wp.get_device().is_cuda
        if self.use_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    for i in range(self.frame_count):
                        self.forward(i)
                self.tape.backward(self.loss)
            self.graph = capture.graph

    def forward(self, frame):
        with wp.ScopedTimer("network", active=self.profile):
            # build sinusoidal input phases
            wp.launch(kernel=compute_phases, dim=self.phase_count, inputs=[self.phases[frame], self.sim_time])
            # fully connected, linear transformation layer
            wp.matmul(
                self.weights,
                self.phases[frame].reshape((self.phase_count, 1)),
                self.bias.reshape((self.model.tet_count, 1)),
                self.activation_inputs[frame].reshape((self.model.tet_count, 1)),
            )
            # tanh activation function
            wp.launch(
                kernel=activation_function,
                dim=self.model.tet_count,
                inputs=[self.tet_activations[frame], self.activation_inputs[frame]],
            )
            self.control.tet_activations = self.tet_activations[frame]

        with wp.ScopedTimer("simulate", active=self.profile):
            # run simulation loop
            for i in range(self.sim_substeps):
                self.states[frame * self.sim_substeps + i].clear_forces()
                self.integrator.simulate(
                    self.model,
                    self.states[frame * self.sim_substeps + i],
                    self.states[frame * self.sim_substeps + i + 1],
                    self.sim_dt,
                    self.control,
                )
                self.sim_time += self.sim_dt

        with wp.ScopedTimer("loss", active=self.profile):
            # compute center of mass velocity
            wp.launch(
                com_kernel,
                dim=self.model.particle_count,
                inputs=[
                    self.states[(frame + 1) * self.sim_substeps].particle_qd,
                    self.model.particle_count,
                    self.coms[frame],
                ],
                outputs=[],
            )
            # compute loss
            wp.launch(loss_kernel, dim=1, inputs=[self.coms[frame], self.loss], outputs=[])

    def step(self):
        with wp.ScopedTimer("step", active=self.profile):
            if self.use_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    for i in range(self.frame_count):
                        self.forward(i)
                self.tape.backward(self.loss)

            # optimization
            x = self.weights.grad.flatten()
            self.optimizer.step([x])

        loss = self.loss.numpy()
        if self.verbose:
            print(f"Iteration {self.iter}: {loss}")

        # reset sim
        self.sim_time = 0.0
        self.states[0] = self.model.state(requires_grad=True)

        # clear grads and zero arrays for next iteration
        self.tape.zero()
        self.loss.zero_()
        for i in range(self.frame_count):
            self.coms[i].zero_()

        self.iter += 1

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=self.profile):
            for i in range(self.frame_count + 1):
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i * self.sim_substeps])
                self.renderer.end_frame()

                self.render_time += self.frame_dt


if __name__ == "__main__":
    stage_path = os.path.join(wp.examples.get_output_directory(), "example_walker.usd")

    example = Example(stage_path, profile=False, verbose=True)

    for i in range(30):
        example.step()
        example.render()

    if example.renderer:
        example.renderer.save()
