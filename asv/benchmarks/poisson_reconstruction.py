# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp
import warp.geometry as geometry

from .benchmarks_utils import setup_once


class PoissonReconstruction:
    params = ([6, 8], [0.0, 2.0], ["float32", "float64"])
    param_names = ["depth", "point_weight", "precision"]
    number = 1
    repeat = 5
    timeout = 300

    @setup_once
    def setup(self, depth, point_weight, precision):
        if not wp.is_cuda_available():
            raise NotImplementedError("Poisson reconstruction requires CUDA")
        self.device = wp.get_device("cuda:0")
        dtype = wp.vec3d if precision == "float64" else wp.vec3
        rng = np.random.default_rng(2026)
        normals = rng.normal(size=(100_000, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        self.positions = wp.array(normals * np.array([1.0, 0.8, 0.6]), dtype=dtype, device=self.device)
        self.normals = wp.array(normals / np.array([1.0, 0.8, 0.6]), dtype=dtype, device=self.device)
        self.queries = wp.array(rng.uniform(-0.5, 0.5, (65_536, 3)), dtype=dtype, device=self.device)
        self.result = geometry.poisson_reconstruct(self.positions, self.normals, depth=depth, point_weight=point_weight)
        self.result.evaluate(self.queries)
        wp.synchronize_device(self.device)

    def time_reconstruct(self, depth, point_weight, precision):
        geometry.poisson_reconstruct(self.positions, self.normals, depth=depth, point_weight=point_weight)
        wp.synchronize_device(self.device)

    def time_evaluate(self, depth, point_weight, precision):
        self.result.evaluate(self.queries)
        wp.synchronize_device(self.device)


class PoissonMeshExtraction:
    params = ([6, 8], ["sparse", "dense"])
    param_names = ["depth", "method"]
    number = 1
    repeat = 5
    timeout = 300

    @setup_once
    def setup(self, depth, method):
        self.reconstruction = PoissonReconstruction()
        self.reconstruction.setup(depth, 2.0, "float32")
        geometry.poisson_extract_mesh(self.reconstruction.result, method=method)
        wp.synchronize_device(self.reconstruction.device)

    def time_extract(self, depth, method):
        geometry.poisson_extract_mesh(self.reconstruction.result, method=method)
        wp.synchronize_device(self.reconstruction.device)

    def time_points_to_mesh(self, depth, method):
        result = geometry.poisson_reconstruct(self.reconstruction.positions, self.reconstruction.normals, depth=depth)
        geometry.poisson_extract_mesh(result, method=method)
        wp.synchronize_device(self.reconstruction.device)
