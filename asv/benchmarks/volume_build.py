# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for building NanoVDB volumes."""

import numpy as np

import warp as wp

from .benchmarks_utils import setup_once

NUM_POINTS = 1 << 20
POINTS_PER_VOXEL = 15
VOXEL_SIZE = 0.5


class VolumeAllocateByVoxelsCPU:
    """Measure building an index volume on CPU from shuffled world-space points, several per voxel."""

    repeat = 15
    number = 1

    @setup_once
    def setup(self):
        wp.init()
        self.device = wp.get_device("cpu")

        rng = np.random.default_rng(42)
        half_extent = 0.5 * VOXEL_SIZE * (NUM_POINTS / POINTS_PER_VOXEL) ** (1.0 / 3.0)
        points_np = rng.uniform(-half_extent, half_extent, size=(NUM_POINTS, 3)).astype(np.float32)
        self.points = wp.array(points_np, dtype=wp.vec3, device=self.device)

        for _ in range(2):
            self.time_allocate()

    def time_allocate(self):
        wp.Volume.allocate_by_voxels(self.points, voxel_size=VOXEL_SIZE, device=self.device)
