# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# include parent path
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from pxr import Usd, UsdGeom

import warp as wp


class Cloth:
    def __init__(
        self, lower, dx, dy, radius, stretch_stiffness, bend_stiffness, shear_stiffness, mass, fix_corners=True
    ):
        self.triangles = []

        self.positions = []
        self.velocities = []
        self.inv_masses = []

        self.spring_indices = []
        self.spring_lengths = []
        self.spring_stiffness = []
        self.spring_damping = []

        def grid(x, y, stride):
            return y * stride + x

        def create_spring(i, j, stiffness, damp=10.0):
            length = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[j]))

            self.spring_indices.append(i)
            self.spring_indices.append(j)
            self.spring_lengths.append(length)
            self.spring_stiffness.append(stiffness)
            self.spring_damping.append(damp)

        for y in range(dy):
            for x in range(dx):
                p = np.array(lower) + radius * np.array((float(x), float(0.0), float(y)))

                self.positions.append(p)
                self.velocities.append(np.zeros(3))

                if x > 0 and y > 0:
                    self.triangles.append(grid(x - 1, y - 1, dx))
                    self.triangles.append(grid(x, y - 1, dx))
                    self.triangles.append(grid(x, y, dx))

                    self.triangles.append(grid(x - 1, y - 1, dx))
                    self.triangles.append(grid(x, y, dx))
                    self.triangles.append(grid(x - 1, y, dx))

                if fix_corners and y == 0 and (x == 0 or x == dx - 1):
                    w = 0.0
                else:
                    w = 1.0 / mass

                self.inv_masses.append(w)

        # horizontal springs
        for y in range(dy):
            for x in range(dx):
                index0 = y * dx + x

                if x > 0:
                    index1 = y * dx + x - 1
                    create_spring(index0, index1, stretch_stiffness)

                if x > 1 and bend_stiffness > 0.0:
                    index2 = y * dx + x - 2
                    create_spring(index0, index2, bend_stiffness)

                if y > 0 and x < dx - 1 and shear_stiffness > 0.0:
                    indexDiag = (y - 1) * dx + x + 1
                    create_spring(index0, indexDiag, shear_stiffness)

                if y > 0 and x > 0 and shear_stiffness > 0.0:
                    indexDiag = (y - 1) * dx + x - 1
                    create_spring(index0, indexDiag, shear_stiffness)

        # vertical
        for x in range(dx):
            for y in range(dy):
                index0 = y * dx + x

                if y > 0:
                    index1 = (y - 1) * dx + x
                    create_spring(index0, index1, stretch_stiffness)

                if y > 1 and bend_stiffness > 0.0:
                    index2 = (y - 2) * dx + x
                    create_spring(index0, index2, bend_stiffness)

        # harden to np arrays
        self.positions = np.array(self.positions, dtype=np.float32)
        self.velocities = np.array(self.velocities, dtype=np.float32)
        self.inv_masses = np.array(self.inv_masses, dtype=np.float32)
        self.spring_lengths = np.array(self.spring_lengths, dtype=np.float32)
        self.spring_indices = np.array(self.spring_indices, dtype=np.int32)
        self.spring_stiffness = np.array(self.spring_stiffness, dtype=np.float32)
        self.spring_damping = np.array(self.spring_damping, dtype=np.float32)

        self.num_particles = len(self.positions)
        self.num_springs = len(self.spring_lengths)
        self.num_tris = int(len(self.triangles) / 3)


def run_benchmark(mode, dim, timers, render=False):
    # params
    sim_width = dim
    sim_height = dim

    sim_fps = 60.0
    sim_substeps = 16
    sim_duration = 1.0
    sim_frames = int(sim_duration * sim_fps)
    sim_dt = 1.0 / sim_fps
    sim_time = 0.0

    # wave constants
    k_stretch = 1000.0
    k_shear = 1000.0
    k_bend = 1000.0
    # k_damp = 0.0

    cloth = Cloth(
        lower=(0.0, 0.0, 0.0),
        dx=sim_width,
        dy=sim_height,
        radius=0.1,
        stretch_stiffness=k_stretch,
        bend_stiffness=k_bend,
        shear_stiffness=k_shear,
        mass=0.1,
        fix_corners=True,
    )

    if render:
        # set up grid for visualization
        stage = Usd.Stage.CreateNew("benchmark.usd")
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(sim_duration * sim_fps)
        stage.SetTimeCodesPerSecond(sim_fps)

        grid = UsdGeom.Mesh.Define(stage, "/root")
        grid.GetPointsAttr().Set(cloth.positions, 0.0)
        grid.GetFaceVertexIndicesAttr().Set(cloth.triangles, 0.0)
        grid.GetFaceVertexCountsAttr().Set([3] * cloth.num_tris, 0.0)

    with wp.ScopedTimer("Initialization", dict=timers):
        if mode == "warp_cpu":
            import benchmark_cloth_warp

            integrator = benchmark_cloth_warp.WpIntegrator(cloth, "cpu")

        elif mode == "warp_gpu":
            import benchmark_cloth_warp

            integrator = benchmark_cloth_warp.WpIntegrator(cloth, "cuda")

        elif mode == "taichi_cpu":
            import benchmark_cloth_taichi

            integrator = benchmark_cloth_taichi.TiIntegrator(cloth, "cpu")

        elif mode == "taichi_gpu":
            import benchmark_cloth_taichi

            integrator = benchmark_cloth_taichi.TiIntegrator(cloth, "cuda")

        elif mode == "numpy":
            import benchmark_cloth_numpy

            integrator = benchmark_cloth_numpy.NpIntegrator(cloth)

        elif mode == "cupy":
            import benchmark_cloth_cupy

            integrator = benchmark_cloth_cupy.CpIntegrator(cloth)

        elif mode == "numba":
            import benchmark_cloth_numba

            integrator = benchmark_cloth_numba.NbIntegrator(cloth)

        elif mode == "torch_cpu":
            import benchmark_cloth_pytorch

            integrator = benchmark_cloth_pytorch.TrIntegrator(cloth, "cpu")

        elif mode == "torch_gpu":
            import benchmark_cloth_pytorch

            integrator = benchmark_cloth_pytorch.TrIntegrator(cloth, "cuda")

        elif mode == "jax_cpu":
            os.environ["JAX_PLATFORM_NAME"] = "cpu"

            import benchmark_cloth_jax

            integrator = benchmark_cloth_jax.JxIntegrator(cloth)

        elif mode == "jax_gpu":
            os.environ["JAX_PLATFORM_NAME"] = "gpu"

            import benchmark_cloth_jax

            integrator = benchmark_cloth_jax.JxIntegrator(cloth)

        elif mode == "paddle_cpu":
            import benchmark_cloth_paddle

            integrator = benchmark_cloth_paddle.TrIntegrator(cloth, "cpu")

        elif mode == "paddle_gpu":
            import benchmark_cloth_paddle

            integrator = benchmark_cloth_paddle.TrIntegrator(cloth, "gpu")

        else:
            raise RuntimeError("Unknown simulation backend")

            # run one warm-up iteration to accurately measure initialization time (some engines do lazy init)
            positions = integrator.simulate(sim_dt, sim_substeps)

    label = f"Dim ({dim}^2)"

    # run simulation
    for _i in range(sim_frames):
        # simulate
        with wp.ScopedTimer(label, dict=timers):
            positions = integrator.simulate(sim_dt, sim_substeps)

        if render:
            grid.GetPointsAttr().Set(positions, sim_time * sim_fps)

        sim_time += sim_dt

    if render:
        stage.Save()


# record profiling information
timers = {}

if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    mode = "warp_gpu"

run_benchmark(mode, 32, timers, render=False)
run_benchmark(mode, 64, timers, render=False)
run_benchmark(mode, 128, timers, render=False)

# write results

for k, v in timers.items():
    print(f"{k:16} min: {np.min(v):8.2f} max: {np.max(v):8.2f} avg: {np.mean(v):8.2f}")

report = open(os.path.join("benchmark.csv"), "a")
writer = csv.writer(report, delimiter=",")

if report.tell() == 0:
    writer.writerow(["Name", "Init", "Dim (32^2)", "Dim (64^2)", "Dim (128^2)"])

writer.writerow(
    [
        mode,
        np.max(timers["Initialization"]),
        np.mean(timers["Dim (32^2)"]),
        np.mean(timers["Dim (64^2)"]),
        np.mean(timers["Dim (128^2)"]),
    ]
)

report.close()
