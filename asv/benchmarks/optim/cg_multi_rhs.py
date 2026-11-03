# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from warp.optim import linear

from ..benchmarks_utils import setup_once
from ._rhs_problem import problem


class CGMultiRHS:
    params = [4, 8, 24, 33]
    param_names = ["nrhs"]
    number = 1
    repeat = 10
    rounds = 2

    @setup_once
    def setup(self, nrhs):
        self.device = wp.get_device("cuda:0")
        with wp.ScopedDevice(self.device):
            # The captured graph addresses these arrays, so they must stay alive.
            self.a, self.b = problem(nrhs)
            self.out = wp.empty_like(self.b)
            self.action = linear.CGSolve(self.a, nrhs, working_dtype=wp.float32, tol=1e-5, maxiter=100)
            self.action.solve(self.b, self.out)
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(force_module_load=False) as capture:
                self.action.solve(self.b, self.out)
            self.graph = capture.graph
            wp.capture_launch(self.graph)
            wp.synchronize_device(self.device)

    def time_prepared(self, nrhs):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
