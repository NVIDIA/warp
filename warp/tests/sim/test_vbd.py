# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import io
import unittest

import warp as wp
import warp.optim
import warp.sim
import warp.sim.graph_coloring
import warp.sim.integrator_vbd
from warp.sim.model import PARTICLE_FLAG_ACTIVE
from warp.tests.unittest_utils import *

# fmt: off
CLOTH_POINTS = [
    (-50.0000000, 0.0000000, -50.0000000),
    (-38.8888893, 11.1111107, -50.0000000),
    (-27.7777786, 22.2222214, -50.0000000),
    (-16.6666679, 33.3333321, -50.0000000),
    (-5.5555558, 44.4444427, -50.0000000),
    (5.5555558, 55.5555573, -50.0000000),
    (16.6666679, 66.6666641, -50.0000000),
    (27.7777786, 77.7777786, -50.0000000),
    (38.8888893, 88.8888855, -50.0000000),
    (50.0000000, 100.0000000, -50.0000000),
    (-50.0000000, 0.0000000, -38.8888893),
    (-38.8888893, 11.1111107, -38.8888893),
    (-27.7777786, 22.2222214, -38.8888893),
    (-16.6666679, 33.3333321, -38.8888893),
    (-5.5555558, 44.4444427, -38.8888893),
    (5.5555558, 55.5555573, -38.8888893),
    (16.6666679, 66.6666641, -38.8888893),
    (27.7777786, 77.7777786, -38.8888893),
    (38.8888893, 88.8888855, -38.8888893),
    (50.0000000, 100.0000000, -38.8888893),
    (-50.0000000, 0.0000000, -27.7777786),
    (-38.8888893, 11.1111107, -27.7777786),
    (-27.7777786, 22.2222214, -27.7777786),
    (-16.6666679, 33.3333321, -27.7777786),
    (-5.5555558, 44.4444427, -27.7777786),
    (5.5555558, 55.5555573, -27.7777786),
    (16.6666679, 66.6666641, -27.7777786),
    (27.7777786, 77.7777786, -27.7777786),
    (38.8888893, 88.8888855, -27.7777786),
    (50.0000000, 100.0000000, -27.7777786),
    (-50.0000000, 0.0000000, -16.6666679),
    (-38.8888893, 11.1111107, -16.6666679),
    (-27.7777786, 22.2222214, -16.6666679),
    (-16.6666679, 33.3333321, -16.6666679),
    (-5.5555558, 44.4444427, -16.6666679),
    (5.5555558, 55.5555573, -16.6666679),
    (16.6666679, 66.6666641, -16.6666679),
    (27.7777786, 77.7777786, -16.6666679),
    (38.8888893, 88.8888855, -16.6666679),
    (50.0000000, 100.0000000, -16.6666679),
    (-50.0000000, 0.0000000, -5.5555558),
    (-38.8888893, 11.1111107, -5.5555558),
    (-27.7777786, 22.2222214, -5.5555558),
    (-16.6666679, 33.3333321, -5.5555558),
    (-5.5555558, 44.4444427, -5.5555558),
    (5.5555558, 55.5555573, -5.5555558),
    (16.6666679, 66.6666641, -5.5555558),
    (27.7777786, 77.7777786, -5.5555558),
    (38.8888893, 88.8888855, -5.5555558),
    (50.0000000, 100.0000000, -5.5555558),
    (-50.0000000, 0.0000000, 5.5555558),
    (-38.8888893, 11.1111107, 5.5555558),
    (-27.7777786, 22.2222214, 5.5555558),
    (-16.6666679, 33.3333321, 5.5555558),
    (-5.5555558, 44.4444427, 5.5555558),
    (5.5555558, 55.5555573, 5.5555558),
    (16.6666679, 66.6666641, 5.5555558),
    (27.7777786, 77.7777786, 5.5555558),
    (38.8888893, 88.8888855, 5.5555558),
    (50.0000000, 100.0000000, 5.5555558),
    (-50.0000000, 0.0000000, 16.6666679),
    (-38.8888893, 11.1111107, 16.6666679),
    (-27.7777786, 22.2222214, 16.6666679),
    (-16.6666679, 33.3333321, 16.6666679),
    (-5.5555558, 44.4444427, 16.6666679),
    (5.5555558, 55.5555573, 16.6666679),
    (16.6666679, 66.6666641, 16.6666679),
    (27.7777786, 77.7777786, 16.6666679),
    (38.8888893, 88.8888855, 16.6666679),
    (50.0000000, 100.0000000, 16.6666679),
    (-50.0000000, 0.0000000, 27.7777786),
    (-38.8888893, 11.1111107, 27.7777786),
    (-27.7777786, 22.2222214, 27.7777786),
    (-16.6666679, 33.3333321, 27.7777786),
    (-5.5555558, 44.4444427, 27.7777786),
    (5.5555558, 55.5555573, 27.7777786),
    (16.6666679, 66.6666641, 27.7777786),
    (27.7777786, 77.7777786, 27.7777786),
    (38.8888893, 88.8888855, 27.7777786),
    (50.0000000, 100.0000000, 27.7777786),
    (-50.0000000, 0.0000000, 38.8888893),
    (-38.8888893, 11.1111107, 38.8888893),
    (-27.7777786, 22.2222214, 38.8888893),
    (-16.6666679, 33.3333321, 38.8888893),
    (-5.5555558, 44.4444427, 38.8888893),
    (5.5555558, 55.5555573, 38.8888893),
    (16.6666679, 66.6666641, 38.8888893),
    (27.7777786, 77.7777786, 38.8888893),
    (38.8888893, 88.8888855, 38.8888893),
    (50.0000000, 100.0000000, 38.8888893),
    (-50.0000000, 0.0000000, 50.0000000),
    (-38.8888893, 11.1111107, 50.0000000),
    (-27.7777786, 22.2222214, 50.0000000),
    (-16.6666679, 33.3333321, 50.0000000),
    (-5.5555558, 44.4444427, 50.0000000),
    (5.5555558, 55.5555573, 50.0000000),
    (16.6666679, 66.6666641, 50.0000000),
    (27.7777786, 77.7777786, 50.0000000),
    (38.8888893, 88.8888855, 50.0000000),
    (50.0000000, 100.0000000, 50.0000000),
]

CLOTH_FACES = [
    1, 12, 2,
    1, 11, 12,
    2, 12, 3,
    12, 13, 3,
    3, 14, 4,
    3, 13, 14,
    4, 14, 5,
    14, 15, 5,
    5, 16, 6,
    5, 15, 16,
    6, 16, 7,
    16, 17, 7,
    7, 18, 8,
    7, 17, 18,
    8, 18, 9,
    18, 19, 9,
    9, 20, 10,
    9, 19, 20,
    11, 21, 12,
    21, 22, 12,
    12, 23, 13,
    12, 22, 23,
    13, 23, 14,
    23, 24, 14,
    14, 25, 15,
    14, 24, 25,
    15, 25, 16,
    25, 26, 16,
    16, 27, 17,
    16, 26, 27,
    17, 27, 18,
    27, 28, 18,
    18, 29, 19,
    18, 28, 29,
    19, 29, 20,
    29, 30, 20,
    21, 32, 22,
    21, 31, 32,
    22, 32, 23,
    32, 33, 23,
    23, 34, 24,
    23, 33, 34,
    24, 34, 25,
    34, 35, 25,
    25, 36, 26,
    25, 35, 36,
    26, 36, 27,
    36, 37, 27,
    27, 38, 28,
    27, 37, 38,
    28, 38, 29,
    38, 39, 29,
    29, 40, 30,
    29, 39, 40,
    31, 41, 32,
    41, 42, 32,
    32, 43, 33,
    32, 42, 43,
    33, 43, 34,
    43, 44, 34,
    34, 45, 35,
    34, 44, 45,
    35, 45, 36,
    45, 46, 36,
    36, 47, 37,
    36, 46, 47,
    37, 47, 38,
    47, 48, 38,
    38, 49, 39,
    38, 48, 49,
    39, 49, 40,
    49, 50, 40,
    41, 52, 42,
    41, 51, 52,
    42, 52, 43,
    52, 53, 43,
    43, 54, 44,
    43, 53, 54,
    44, 54, 45,
    54, 55, 45,
    45, 56, 46,
    45, 55, 56,
    46, 56, 47,
    56, 57, 47,
    47, 58, 48,
    47, 57, 58,
    48, 58, 49,
    58, 59, 49,
    49, 60, 50,
    49, 59, 60,
    51, 61, 52,
    61, 62, 52,
    52, 63, 53,
    52, 62, 63,
    53, 63, 54,
    63, 64, 54,
    54, 65, 55,
    54, 64, 65,
    55, 65, 56,
    65, 66, 56,
    56, 67, 57,
    56, 66, 67,
    57, 67, 58,
    67, 68, 58,
    58, 69, 59,
    58, 68, 69,
    59, 69, 60,
    69, 70, 60,
    61, 72, 62,
    61, 71, 72,
    62, 72, 63,
    72, 73, 63,
    63, 74, 64,
    63, 73, 74,
    64, 74, 65,
    74, 75, 65,
    65, 76, 66,
    65, 75, 76,
    66, 76, 67,
    76, 77, 67,
    67, 78, 68,
    67, 77, 78,
    68, 78, 69,
    78, 79, 69,
    69, 80, 70,
    69, 79, 80,
    71, 81, 72,
    81, 82, 72,
    72, 83, 73,
    72, 82, 83,
    73, 83, 74,
    83, 84, 74,
    74, 85, 75,
    74, 84, 85,
    75, 85, 76,
    85, 86, 76,
    76, 87, 77,
    76, 86, 87,
    77, 87, 78,
    87, 88, 78,
    78, 89, 79,
    78, 88, 89,
    79, 89, 80,
    89, 90, 80,
    81, 92, 82,
    81, 91, 92,
    82, 92, 83,
    92, 93, 83,
    83, 94, 84,
    83, 93, 94,
    84, 94, 85,
    94, 95, 85,
    85, 96, 86,
    85, 95, 96,
    86, 96, 87,
    96, 97, 87,
    87, 98, 88,
    87, 97, 98,
    88, 98, 89,
    98, 99, 89,
    89, 100, 90,
    89, 99, 100
]

# fmt: on
class VBDClothSim:
    def __init__(self, device, use_cuda_graph=False):
        self.frame_dt = 1 / 60
        self.num_test_frames = 100
        self.num_substeps = 10
        self.iterations = 10
        self.dt = self.frame_dt / self.num_substeps
        self.device = device
        self.use_cuda_graph = self.device.is_cuda and use_cuda_graph
        self.builder = wp.sim.ModelBuilder()

    def set_up_sagging_experiment(self):
        stiffness = 1e5
        kd = 1.0e-7

        self.input_scale_factor = 1.0
        self.renderer_scale_factor = 0.01
        vertices = [wp.vec3(v) * self.input_scale_factor for v in CLOTH_POINTS]
        faces_flatten = [fv - 1 for fv in CLOTH_FACES]

        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 200.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vertices,
            indices=faces_flatten,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stiffness,
            tri_ka=stiffness,
            tri_kd=kd,
        )
        self.fixed_particles = [0, 9]

    def set_up_bending_experiment(self):
        stretching_stiffness = 1e4
        stretching_damping = 1e-6
        bending_damping = 1e-3
        # fmt: off
        vs = [[-6.0, 0.0, -6.0], [-3.6, 0.0, -6.0], [-1.2, 0.0, -6.0], [1.2, 0.0, -6.0], [3.6, 0.0, -6.0], [6.0, 0.0, -6.0],
         [-6.0, 0.0, -3.6], [-3.6, 0.0, -3.6], [-1.2, 0.0, -3.6], [1.2, 0.0, -3.6], [3.6, 0.0, -3.6], [6.0, 0.0, -3.6],
         [-6.0, 0.0, -1.2], [-3.6, 0.0, -1.2], [-1.2, 0.0, -1.2], [1.2, 0.0, -1.2], [3.6, 0.0, -1.2], [6.0, 0.0, -1.2],
         [-6.0, 0.0, 1.2], [-3.6, 0.0, 1.2], [-1.2, 0.0, 1.2], [1.2, 0.0, 1.2], [3.6, 0.0, 1.2], [6.0, 0.0, 1.2],
         [-6.0, 0.0, 3.6], [-3.6, 0.0, 3.6], [-1.2, 0.0, 3.6], [1.2, 0.0, 3.6], [3.6, 0.0, 3.6], [6.0, 0.0, 3.6],
         [-6.0, 0.0, 6.0], [-3.6, 0.0, 6.0], [-1.2, 0.0, 6.0], [1.2, 0.0, 6.0], [3.6, 0.0, 6.0], [6.0, 0.0, 6.0]]

        fs = [0, 7, 1, 0, 6, 7, 1, 7, 2, 7, 8, 2, 2, 9, 3, 2, 8, 9, 3, 9, 4, 9, 10, 4, 4, 11, 5, 4, 10, 11, 6, 12, 7, 12, 13,
         7, 7, 14, 8, 7, 13, 14, 8, 14, 9, 14, 15, 9, 9, 16, 10, 9, 15, 16, 10, 16, 11, 16, 17, 11, 12, 19, 13, 12, 18,
         19, 13, 19, 14, 19, 20, 14, 14, 21, 15, 14, 20, 21, 15, 21, 16, 21, 22, 16, 16, 23, 17, 16, 22, 23, 18, 24, 19,
         24, 25, 19, 19, 26, 20, 19, 25, 26, 20, 26, 21, 26, 27, 21, 21, 28, 22, 21, 27, 28, 22, 28, 23, 28, 29, 23, 24,
         31, 25, 24, 30, 31, 25, 31, 26, 31, 32, 26, 26, 33, 27, 26, 32, 33, 27, 33, 28, 33, 34, 28, 28, 35, 29, 28, 34,
         35]
        # fmt: on

        vs = [wp.vec3(v) for v in vs]

        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 10.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=stretching_damping,
            edge_ke=10,
            edge_kd=bending_damping,
        )

        self.builder.add_cloth_mesh(
            pos=wp.vec3(15.0, 10.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=stretching_damping,
            edge_ke=100,
            edge_kd=bending_damping,
        )

        self.builder.add_cloth_mesh(
            pos=wp.vec3(30.0, 10.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=stretching_damping,
            edge_ke=1000,
            edge_kd=bending_damping,
        )

        self.fixed_particles = [0, 29, 36, 65, 72, 101]

    def set_up_non_zero_rest_angle_bending_experiment(self):
        # fmt: off
        vs = [
            [  0.     ,  10.     , -10.     ],
            [  0.     ,  10.     ,  10.     ],
            [  7.07107,   7.07107, -10.     ],
            [  7.07107,   7.07107,  10.     ],
            [ 10.     ,   0.     , -10.     ],
            [ 10.     ,  -0.     ,  10.     ],
            [  7.07107,  -7.07107, -10.     ],
            [  7.07107,  -7.07107,  10.     ],
            [  0.     , -10.     , -10.     ],
            [  0.     , -10.     ,  10.     ],
            [ -7.07107,  -7.07107, -10.     ],
            [ -7.07107,  -7.07107,  10.     ],
            [-10.     ,   0.     , -10.     ],
            [-10.     ,  -0.     ,  10.     ],
            [ -7.07107,   7.07107, -10.     ],
            [ -7.07107,   7.07107,  10.     ],
        ]
        fs = [
          1,  2,  0,
          3,  4,  2,
          5,  6,  4,
          7,  8,  6,
          9, 10,  8,
         11, 12, 10,
          3,  5,  4,
         13, 14, 12,
         15,  0, 14,
          1,  3,  2,
          5,  7,  6,
          7,  9,  8,
          9, 11, 10,
         11, 13, 12,
        ]
        # fmt: on

        stretching_stiffness = 1e4
        stretching_damping = 1e-6
        edge_ke = 1000
        bending_damping = 1e-2
        vs = [wp.vec3(v) for v in vs]

        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 10.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=stretching_damping,
            edge_ke=edge_ke,
            edge_kd=bending_damping,
        )
        self.fixed_particles = [0, 1]

    def finalize(self):
        self.builder.color()

        self.model = self.builder.finalize(device=self.device)
        self.model.ground = True
        self.model.gravity = wp.vec3(0, -1000.0, 0)
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2

        self.set_points_fixed(self.model, self.fixed_particles)

        self.integrator = wp.sim.VBDIntegrator(self.model, self.iterations)
        self.state0 = self.model.state()
        self.state1 = self.model.state()

        self.init_pos = np.array(self.state0.particle_q.numpy(), copy=True)

        self.graph = None
        if self.use_cuda_graph:
            wp.load_module(device=self.device)
            wp.set_module_options({"block_dim": 256}, warp.sim.integrator_vbd)
            wp.load_module(warp.sim.integrator_vbd, device=self.device)
            with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _step in range(self.num_substeps * self.num_test_frames):
            self.integrator.simulate(self.model, self.state0, self.state1, self.dt, None)
            (self.state0, self.state1) = (self.state1, self.state0)

    def run(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def set_points_fixed(self, model, fixed_particles):
        if len(fixed_particles):
            flags = model.particle_flags.numpy()
            for fixed_v_id in fixed_particles:
                flags[fixed_v_id] = wp.uint32(int(flags[fixed_v_id]) & ~int(PARTICLE_FLAG_ACTIVE))

            model.particle_flags = wp.array(flags, device=model.device)


def test_vbd_cloth(test, device):
    with contextlib.redirect_stdout(io.StringIO()) as f:
        example = VBDClothSim(device)
        example.set_up_bending_experiment()
        example.finalize()
        example.model.ground = False

    test.assertRegex(
        f.getvalue(),
        r"Warp UserWarning: The graph is not optimizable anymore, terminated with a max/min ratio: 2.0 without reaching the target ratio: 1.1",
    )

    example.run()

    # examine that the simulation does not explode
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((final_pos < 1e5).all())
    # examine that the simulation have moved
    test.assertTrue((example.init_pos != final_pos).any())


def test_vbd_cloth_cuda_graph(test, device):
    with contextlib.redirect_stdout(io.StringIO()) as f:
        example = VBDClothSim(device, use_cuda_graph=True)
        example.set_up_sagging_experiment()
        example.finalize()

    test.assertRegex(
        f.getvalue(),
        r"Warp UserWarning: The graph is not optimizable anymore, terminated with a max/min ratio: 2.0 without reaching the target ratio: 1.1",
    )

    example.run()

    # examine that the simulation does not explode
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((final_pos < 1e5).all())
    # examine that the simulation have moved
    test.assertTrue((example.init_pos != final_pos).any())


def test_vbd_bending(test, device):
    example = VBDClothSim(device, use_cuda_graph=True)
    example.set_up_bending_experiment()
    example.finalize()

    example.run()

    # examine that the simulation does not explode
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((final_pos < 1e5).all())
    # examine that the simulation have moved
    test.assertTrue((example.init_pos != final_pos).any())


def test_vbd_bending_non_zero_rest_angle_bending(test, device):
    example = VBDClothSim(device, use_cuda_graph=True)
    example.set_up_non_zero_rest_angle_bending_experiment()
    example.finalize()
    example.run()

    # examine that the simulation does not explode
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((final_pos < 1e5).all())
    # examine that the simulation have moved
    test.assertTrue((example.init_pos != final_pos).any())


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestVbd(unittest.TestCase):
    pass


add_function_test(TestVbd, "test_vbd_cloth", test_vbd_cloth, devices=devices)
add_function_test(TestVbd, "test_vbd_cloth_cuda_graph", test_vbd_cloth_cuda_graph, devices=cuda_devices)
add_function_test(TestVbd, "test_vbd_bending", test_vbd_bending, devices=devices, check_output=False)
add_function_test(
    TestVbd,
    "test_vbd_bending_non_zero_rest_angle_bending",
    test_vbd_bending_non_zero_rest_angle_bending,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
