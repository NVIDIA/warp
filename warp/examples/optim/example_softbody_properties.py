# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Example FEM Bounce
#
# Shows how to use Warp to optimize for the material parameters of a soft body,
# such that it bounces off the wall and floor in order to hit a target.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the material parameters, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import numpy as np

import warp as wp
import warp.optim
import warp.sim
import warp.sim.render


@wp.kernel
def assign_param(params: wp.array(dtype=wp.float32), tet_materials: wp.array2d(dtype=wp.float32)):
    tid = wp.tid()
    params_idx = 2 * wp.tid() % params.shape[0]
    tet_materials[tid, 0] = params[params_idx]
    tet_materials[tid, 1] = params[params_idx + 1]


@wp.kernel
def com_kernel(particle_q: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    point = particle_q[tid]
    a = point / wp.float32(particle_q.shape[0])

    # Atomically add the point coordinates to the accumulator
    wp.atomic_add(com, 0, a)


@wp.kernel
def loss_kernel(
    target: wp.vec3,
    com: wp.array(dtype=wp.vec3),
    pos_error: wp.array(dtype=float),
    loss: wp.array(dtype=float),
):
    diff = com[0] - target
    pos_error[0] = wp.dot(diff, diff)
    norm = pos_error[0]
    loss[0] = norm


@wp.kernel
def enforce_constraint_kernel(lower_bound: wp.float32, upper_bound: wp.float32, x: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if x[tid] < lower_bound:
        x[tid] = lower_bound
    elif x[tid] > upper_bound:
        x[tid] = upper_bound


class Example:
    def __init__(
        self,
        stage_path="example_softbody_properties.usd",
        material_behavior="anisotropic",
        verbose=False,
    ):
        self.verbose = verbose
        self.material_behavior = material_behavior

        # seconds
        sim_duration = 1.0

        # control frequency
        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 16
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        self.train_rate = 1e7

        self.losses = []

        self.hard_lower_bound = wp.float32(500.0)
        self.hard_upper_bound = wp.float32(4e6)

        # Create FEM model.
        self.cell_dim = 2
        self.cell_size = 0.1
        center = self.cell_size * self.cell_dim * 0.5
        self.grid_origin = wp.vec3(-0.5, 1.0, -center)
        self.create_model()

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = wp.vec3(-1.0, 1.5, 0.0)
        # Initialize material parameters
        if self.material_behavior == "anisotropic":
            # Different Lame parameters for each tet
            self.material_params = wp.array(
                self.model.tet_materials.numpy()[:, :2].flatten(),
                dtype=wp.float32,
                requires_grad=True,
            )
        else:
            # Same Lame parameters for all tets
            self.material_params = wp.array(
                self.model.tet_materials.numpy()[0, :2].flatten(),
                dtype=wp.float32,
                requires_grad=True,
            )

        self.optimizer = wp.optim.SGD(
            [self.material_params],
            lr=self.train_rate,
            nesterov=False,
        )

        self.com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True)
        self.pos_error = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(self.model.state())

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = None

        # capture forward/backward passes
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph = capture.graph

    def create_model(self):
        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = 0.0005

        total_mass = 0.2
        num_particles = (self.cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (self.cell_size**3)
        if self.verbose:
            print(f"Particle density: {particle_density}")

        young_mod = 1.5 * 1e4
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

        builder.add_soft_grid(
            pos=self.grid_origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(5.0, -5.0, 0.0),
            dim_x=self.cell_dim,
            dim_y=self.cell_dim,
            dim_z=self.cell_dim,
            cell_x=self.cell_size,
            cell_y=self.cell_size,
            cell_z=self.cell_size,
            density=particle_density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            tri_ke=1e-4,
            tri_ka=1e-4,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )

        ke = 1.0e3
        kf = 0.0
        kd = 1.0e0
        mu = 0.2
        builder.add_shape_box(
            body=-1,
            pos=wp.vec3(2.0, 1.0, 0.0),
            hx=0.25,
            hy=1.0,
            hz=1.0,
            ke=ke,
            kf=kf,
            kd=kd,
            mu=mu,
        )

        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True

        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_margin = 0.001
        self.model.soft_contact_restitution = 1.0

    def forward(self):
        wp.launch(
            kernel=assign_param,
            dim=self.model.tet_count,
            inputs=(self.material_params,),
            outputs=(self.model.tet_materials,),
        )
        # run control loop
        for i in range(self.sim_steps):
            wp.sim.collide(self.model, self.states[i])
            self.states[i].clear_forces()

            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

        # Update loss
        # Compute the center of mass for the last time step.
        wp.launch(
            kernel=com_kernel,
            dim=self.model.particle_count,
            inputs=(self.states[-1].particle_q,),
            outputs=(self.com,),
        )

        # calculate loss
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=(
                self.target,
                self.com,
            ),
            outputs=(self.pos_error, self.loss),
        )

        return self.loss

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(loss=self.loss)

            if self.verbose:
                self.log_step()

            self.optimizer.step([self.material_params.grad])

            wp.launch(
                kernel=enforce_constraint_kernel,
                dim=self.material_params.shape[0],
                inputs=(
                    self.hard_lower_bound,
                    self.hard_upper_bound,
                ),
                outputs=(self.material_params,),
            )

            self.losses.append(self.loss.numpy()[0])

            # clear grads for next iteration
            self.tape.zero()
            self.loss.zero_()
            self.com.zero_()
            self.pos_error.zero_()

            self.iter = self.iter + 1

    def log_step(self):
        x = self.material_params.numpy().reshape(-1, 2)
        x_grad = self.material_params.grad.numpy().reshape(-1, 2)

        print(f"Iter: {self.iter} Loss: {self.loss.numpy()[0]}")

        print(f"Pos error: {np.sqrt(self.pos_error.numpy()[0])}")

        print(
            f"Max Mu: {np.max(x[:, 0])}, Min Mu: {np.min(x[:, 0])}, "
            f"Max Lambda: {np.max(x[:, 1])}, Min Lambda: {np.min(x[:, 1])}"
        )

        print(
            f"Max Mu Grad: {np.max(x_grad[:, 0])}, Min Mu Grad: {np.min(x_grad[:, 0])}, "
            f"Max Lambda Grad: {np.max(x_grad[:, 1])}, Min Lambda Grad: {np.min(x_grad[:, 1])}"
        )

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            # draw trajectory
            traj_verts = [np.mean(self.states[0].particle_q.numpy(), axis=0).tolist()]
            for i in range(0, self.sim_steps, self.sim_substeps):
                traj_verts.append(np.mean(self.states[i].particle_q.numpy(), axis=0).tolist())

                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i])
                self.renderer.render_box(
                    pos=self.target,
                    rot=wp.quat_identity(),
                    extents=(0.1, 0.1, 0.1),
                    name="target",
                    color=(0.0, 0.0, 0.0),
                )
                self.renderer.render_line_strip(
                    vertices=traj_verts,
                    color=wp.render.bourke_color_map(0.0, self.losses[0], self.losses[-1]),
                    radius=0.02,
                    name=f"traj_{self.iter - 1}",
                )
                self.renderer.end_frame()

                from pxr import Gf, UsdGeom

                particles_prim = self.renderer.stage.GetPrimAtPath("/root/particles")
                particles = UsdGeom.Points.Get(self.renderer.stage, particles_prim.GetPath())
                particles.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)], time=self.renderer.time)

                self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_softbody_properties.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=300,
        help="Total number of training iterations.",
    )
    parser.add_argument(
        "--material_behavior",
        default="anisotropic",
        choices=["anisotropic", "isotropic"],
        help="Set material behavior to be Anisotropic or Isotropic.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out additional status messages during execution.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            material_behavior=args.material_behavior,
            verbose=args.verbose,
        )

        # replay and optimize
        for i in range(args.train_iters):
            example.step()
            if i == 0 or i % 50 == 0 or i == args.train_iters - 1:
                example.render()

        if example.renderer:
            example.renderer.save()
