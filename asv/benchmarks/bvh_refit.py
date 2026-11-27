# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for refitting and rebuilding a CUDA BVH in place."""

import numpy as np
from asv_runner.benchmarks.mark import skip_for_params

import warp as wp

from .benchmarks_utils import CudaEventTimer, capture_launches, setup_once

# Large enough that refit time grows with the item count on a Blackwell MIG
# slice, so the per-item work dominates the fixed launch cost.
NUM_ITEMS = 1 << 18
# Calls per timed graph replay. Each sample also covers the host's time to
# start the replay; these counts keep that share below about 1%.
REFITS_PER_SAMPLE = 16
REBUILDS_PER_SAMPLE = 4


def _random_boxes(device):
    rng = np.random.default_rng(42)
    lowers = rng.random((NUM_ITEMS, 3), dtype=np.float32) * 10.0
    uppers = lowers + rng.random((NUM_ITEMS, 3), dtype=np.float32) * 0.05
    return wp.array(lowers, dtype=wp.vec3, device=device), wp.array(uppers, dtype=wp.vec3, device=device)


class BvhRefit:
    """Refit a BVH over random boxes, as a simulation step does after moving its items."""

    params = (["lbvh", "sah"], [1, 4])
    param_names = ["constructor", "leaf_size"]
    timer = CudaEventTimer()
    # Calibrated on a Blackwell MIG slice, where each sample varies by about
    # 1.5% at most: with the default two rounds, 8 samples per commit keep the
    # 99% confidence interval within 2.5% of the median. Averaging calls or
    # taking more samples narrowed it without changing any comparison, and
    # process startup dominates the run time either way.
    number = 1
    repeat = 4

    @setup_once
    def setup(self, constructor, leaf_size):
        wp.init()
        device = wp.get_device("cuda:0")
        self.lowers, self.uppers = _random_boxes(device)
        self.bvh = wp.Bvh(self.lowers, self.uppers, constructor=constructor, leaf_size=leaf_size)
        self.graph = capture_launches(self.bvh.refit, REFITS_PER_SAMPLE, device)
        wp.capture_launch(self.graph)
        wp.synchronize_device(device)

    # A SAH tree with one item per leaf refits like the LBVH row with the same
    # leaf size; skipping it saves a benchmark process per round.
    @skip_for_params([("sah", 1)])
    def time_refit(self, constructor, leaf_size):
        wp.capture_launch(self.graph)


class BvhRebuild:
    """Rebuild a BVH in place over random boxes.

    A CUDA rebuild always produces an LBVH, so only an LBVH with the default
    leaf size is timed: after its first rebuild, a tree built with another
    constructor runs the same code, and the leaf size changes only how leaves
    are marked.
    """

    timer = CudaEventTimer()
    # Calibrated with BvhRefit; see the comment there.
    number = 1
    repeat = 4

    @setup_once
    def setup(self):
        wp.init()
        device = wp.get_device("cuda:0")
        if not wp.is_mempool_enabled(device):
            raise NotImplementedError("Capturing a rebuild requires the CUDA memory pool")
        self.lowers, self.uppers = _random_boxes(device)
        self.bvh = wp.Bvh(self.lowers, self.uppers, constructor="lbvh")
        self.graph = capture_launches(lambda: self.bvh.rebuild("lbvh"), REBUILDS_PER_SAMPLE, device)
        wp.capture_launch(self.graph)
        wp.synchronize_device(device)

    def time_rebuild(self):
        wp.capture_launch(self.graph)
