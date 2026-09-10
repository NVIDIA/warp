# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp
import warp.sparse as wps


class BsrSetTransposeCompact:
    """Benchmark captured compact transposes across row lengths and storage occupancy."""

    params = ["stencil8", "stencil32", "dense", "overallocated", "wide_hot_column", "shared_columns8"]
    param_names = ["pattern"]
    rounds = 1
    repeat = 3
    number = 10

    def setup(self, pattern):
        wp.init()
        self.device = wp.get_device("cuda:0")
        nrow = ncol = 65536
        active_rows = nrow
        degree = 8
        if pattern == "stencil32":
            nrow = ncol = active_rows = 16384
            degree = 32
        elif pattern == "dense":
            nrow = ncol = active_rows = degree = 1024
        elif pattern == "overallocated":
            nrow, ncol, active_rows = 1310720, 524288, 3000
        elif pattern == "wide_hot_column":
            degree = 1

        rows = np.repeat(np.arange(active_rows, dtype=np.int32), degree)
        columns = (rows + np.tile(np.arange(degree, dtype=np.int32), active_rows)) % ncol
        if pattern == "wide_hot_column":
            columns.fill(0)
        elif pattern == "shared_columns8":
            columns = np.tile(np.arange(degree, dtype=np.int32), active_rows)
        columns = np.sort(columns.reshape(active_rows, degree), axis=1).reshape(-1)
        nnz = rows.size
        capacity = 10485760 if pattern == "overallocated" else nnz
        offsets = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=nrow)))).astype(np.int32)

        with wp.ScopedDevice(self.device):
            self.src = wps.bsr_zeros(nrow, ncol, float)
            self.src.notify_nnz_changed(nnz=capacity)
            self.src.columns.fill_(-1)
            self.src.values.fill_(1.0)
            wp.copy(self.src.offsets, wp.array(offsets, dtype=int))
            wp.copy(self.src.columns, wp.array(columns, dtype=int), count=nnz)
            self.dest = wps.bsr_zeros(ncol, nrow, float)
            wps.bsr_set_transpose(self.dest, self.src)
            with wp.ScopedCapture() as capture:
                wps.bsr_set_transpose(self.dest, self.src)
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def time_cuda(self, pattern):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
