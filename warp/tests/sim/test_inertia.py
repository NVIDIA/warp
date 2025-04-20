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

import unittest

import numpy as np

import warp as wp
from warp.render.render_opengl import OpenGLRenderer
from warp.sim import ModelBuilder
from warp.sim.inertia import (
    compute_box_inertia,
    compute_mesh_inertia,
    compute_sphere_inertia,
)
from warp.tests.unittest_utils import assert_np_equal


class TestInertia(unittest.TestCase):
    def test_cube_mesh_inertia(self):
        # Unit cube
        vertices = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
        indices = [
            [1, 2, 3],
            [7, 6, 5],
            [4, 5, 1],
            [5, 6, 2],
            [2, 6, 7],
            [0, 3, 7],
            [0, 1, 3],
            [4, 7, 5],
            [0, 4, 1],
            [1, 5, 2],
            [3, 2, 7],
            [4, 0, 7],
        ]

        mass_0, com_0, I_0, volume_0 = compute_mesh_inertia(
            density=1000, vertices=vertices, indices=indices, is_solid=True
        )

        self.assertAlmostEqual(mass_0, 1000.0, delta=1e-6)
        self.assertAlmostEqual(volume_0, 1.0, delta=1e-6)
        assert_np_equal(np.array(com_0), np.array([0.5, 0.5, 0.5]), tol=1e-6)

        # Check against analytical inertia
        mass_box, com_box, I_box = compute_box_inertia(1000.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(mass_box, mass_0, delta=1e-6)
        assert_np_equal(np.array(com_box), np.zeros(3), tol=1e-6)
        assert_np_equal(np.array(I_0), np.array(I_box), tol=1e-4)

        # Compute hollow box inertia
        mass_0_hollow, com_0_hollow, I_0_hollow, volume_0_hollow = compute_mesh_inertia(
            density=1000,
            vertices=vertices,
            indices=indices,
            is_solid=False,
            thickness=0.1,
        )
        assert_np_equal(np.array(com_0_hollow), np.array([0.5, 0.5, 0.5]), tol=1e-6)

        # Add vertex between [0.0, 0.0, 0.0] and [1.0, 0.0, 0.0]
        vertices.append([0.5, 0.0, 0.0])
        indices[5] = [0, 8, 7]
        indices.append([8, 3, 7])
        indices[6] = [0, 1, 8]
        indices.append([8, 1, 3])

        mass_1, com_1, I_1, volume_1 = compute_mesh_inertia(
            density=1000, vertices=vertices, indices=indices, is_solid=True
        )

        # Inertia values should be the same as before
        self.assertAlmostEqual(mass_1, mass_0, delta=1e-6)
        self.assertAlmostEqual(volume_1, volume_0, delta=1e-6)
        assert_np_equal(np.array(com_1), np.array([0.5, 0.5, 0.5]), tol=1e-6)
        assert_np_equal(np.array(I_1), np.array(I_0), tol=1e-4)

        # Compute hollow box inertia
        mass_1_hollow, com_1_hollow, I_1_hollow, volume_1_hollow = compute_mesh_inertia(
            density=1000,
            vertices=vertices,
            indices=indices,
            is_solid=False,
            thickness=0.1,
        )

        # Inertia values should be the same as before
        self.assertAlmostEqual(mass_1_hollow, mass_0_hollow, delta=2e-3)
        self.assertAlmostEqual(volume_1_hollow, volume_0_hollow, delta=1e-6)
        assert_np_equal(np.array(com_1_hollow), np.array([0.5, 0.5, 0.5]), tol=1e-6)
        assert_np_equal(np.array(I_1_hollow), np.array(I_0_hollow), tol=1e-4)

    def test_sphere_mesh_inertia(self):
        vertices, indices = OpenGLRenderer._create_sphere_mesh(radius=2.5, num_latitudes=500, num_longitudes=500)

        offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vertices = vertices[:, :3] + offset

        mass_mesh, com_mesh, I_mesh, vol_mesh = compute_mesh_inertia(
            density=1000,
            vertices=vertices,
            indices=indices,
            is_solid=True,
        )

        # Check against analytical inertia
        mass_sphere, _, I_sphere = compute_sphere_inertia(1000.0, 2.5)
        self.assertAlmostEqual(mass_mesh, mass_sphere, delta=1e2)
        assert_np_equal(np.array(com_mesh), np.array(offset), tol=2e-3)
        assert_np_equal(np.array(I_mesh), np.array(I_sphere), tol=4e2)
        # Check volume
        self.assertAlmostEqual(vol_mesh, 4.0 / 3.0 * np.pi * 2.5**3, delta=3e-2)

    def test_body_inertia(self):
        vertices, indices = OpenGLRenderer._create_sphere_mesh(radius=2.5, num_latitudes=500, num_longitudes=500)

        offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vertices = vertices[:, :3] + offset

        builder = ModelBuilder()
        b = builder.add_body()
        tf = wp.transform(wp.vec3(4.0, 5.0, 6.0), wp.quat_rpy(0.5, -0.8, 1.3))
        builder.add_shape_mesh(
            b,
            pos=tf.p,
            rot=tf.q,
            mesh=wp.sim.Mesh(vertices=vertices, indices=indices),
            density=1000.0,
        )
        transformed_com = wp.transform_point(tf, wp.vec3(*offset))
        assert_np_equal(np.array(builder.body_com[0]), np.array(transformed_com), tol=2e-3)
        mass_sphere, _, I_sphere = compute_sphere_inertia(1000.0, 2.5)
        assert_np_equal(np.array(builder.body_inertia[0]), np.array(I_sphere), tol=4e2)
        self.assertAlmostEqual(builder.body_mass[0], mass_sphere, delta=1e2)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
