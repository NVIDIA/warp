# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cloth Self Contact
#
# This simulation demonstrates twisting an FEM cloth model using the VBD
# integrator, showcasing its ability to handle complex self-contacts while
# ensuring it remains intersection-free.
#
###########################################################################

import math
import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
from warp.sim.model import PARTICLE_FLAG_ACTIVE


@wp.kernel
def initialize_rotation(
    particle_indices_to_rot: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_index = particle_indices_to_rot[wp.tid()]

    p = pos[particle_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center

    root = wp.dot(op, rot_axis) * rot_axis

    root_to_p = p - root

    roots[tid] = root
    roots_to_ps[tid] = root_to_p


@wp.kernel
def apply_rotation(
    time: float,
    angular_velocity: float,
    particle_indices_to_rot: wp.array(dtype=wp.int32),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_index = particle_indices_to_rot[wp.tid()]

    rot_axis = rot_axes[tid]

    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    theta = time * angular_velocity

    R = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )

    root = roots[tid]
    root_to_p = roots_to_ps[tid]
    root_to_p_rot = R * root_to_p
    p_rot = root + root_to_p_rot

    pos_0[particle_index] = p_rot
    pos_1[particle_index] = p_rot


class Example:
    def __init__(self, stage_path="example_cloth_self_contact.usd", num_frames=1500):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.num_substeps = 10
        self.iterations = 10
        self.dt = self.frame_dt / self.num_substeps

        self.num_frames = num_frames
        self.sim_time = 0.0
        self.profiler = {}

        self.rot_angular_velocity = math.pi / 6
        self.rot_end_time = 21

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        self.input_scale_factor = 1.0
        self.renderer_scale_factor = 0.01

        vertices = [wp.vec3(v) * self.input_scale_factor for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        builder = wp.sim.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=3.0e-5,
        )
        builder.color()
        self.model = builder.finalize()
        self.model.ground = False
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.2

        # set up contact query and contact detection distances
        self.model.soft_contact_radius = 0.2
        self.model.soft_contact_margin = 0.35

        cloth_size = 50
        left_side = [cloth_size - 1 + i * cloth_size for i in range(cloth_size)]
        right_side = [i * cloth_size for i in range(cloth_size)]
        rot_point_indices = left_side + right_side

        if len(rot_point_indices):
            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in rot_point_indices:
                flags[fixed_vertex_id] = wp.uint32(int(flags[fixed_vertex_id]) & ~int(PARTICLE_FLAG_ACTIVE))

            self.model.particle_flags = wp.array(flags)

        self.integrator = wp.sim.VBDIntegrator(
            self.model,
            self.iterations,
            handle_self_contact=True,
        )
        self.state0 = self.model.state()
        self.state1 = self.model.state()

        rot_axes = [[1, 0, 0]] * len(right_side) + [[-1, 0, 0]] * len(left_side)

        self.rot_point_indices = wp.array(rot_point_indices, dtype=int)
        self.rot_centers = wp.zeros(len(rot_point_indices), dtype=wp.vec3)
        self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)

        self.roots = wp.zeros_like(self.rot_centers)
        self.roots_to_ps = wp.zeros_like(self.rot_centers)

        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[
                self.rot_point_indices,
                self.state0.particle_q,
                self.rot_centers,
                self.rot_axes,
                self.roots,
                self.roots_to_ps,
            ],
        )

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=40.0)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step", print=False, dict=self.profiler):
            for _ in range(self.num_substeps):
                if self.sim_time < self.rot_end_time:
                    wp.launch(
                        kernel=apply_rotation,
                        dim=self.rot_point_indices.shape[0],
                        inputs=[
                            self.sim_time,
                            self.rot_angular_velocity,
                            self.rot_point_indices,
                            self.rot_centers,
                            self.rot_axes,
                            self.roots,
                            self.roots_to_ps,
                            self.state0.particle_q,
                            self.state1.particle_q,
                        ],
                    )

                self.integrator.simulate(self.model, self.state0, self.state1, self.dt)

                (self.state0, self.state1) = (self.state1, self.state0)

                self.sim_time += self.dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", print=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cloth_self_contact.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1500, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_frames=args.num_frames)

        for i in range(example.num_frames):
            example.step()
            example.render()
            print(f"[{i:4d}/{example.num_frames}]")

        frame_times = example.profiler["step"]
        print("\nAverage frame sim time: {:.2f} ms".format(sum(frame_times) / len(frame_times)))

        if example.renderer:
            example.renderer.save()
