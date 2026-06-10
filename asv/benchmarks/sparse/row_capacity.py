# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import numpy as np

import warp as wp
import warp.sparse as wps


def _require_row_capacity_support():
    probe = wps.bsr_zeros(0, 0, float)
    if not hasattr(probe, "row_counts"):
        raise NotImplementedError("Sparse row capacity support is not available")
    if not hasattr(wps, "bsr_compress"):
        raise NotImplementedError("bsr_compress is not available")
    if "topology" not in inspect.signature(wps.bsr_axpy).parameters:
        raise NotImplementedError("Sparse topology policies are not available")


def _make_row_capacity_matrix(
    nrow: int, ncol: int, active_per_row: int, capacity_per_row: int, device, col_shift: int = 0
):
    offsets = np.arange(nrow + 1, dtype=np.int32) * capacity_per_row
    row_counts = np.full(nrow, active_per_row, dtype=np.int32)

    columns = np.full(nrow * capacity_per_row, -1, dtype=np.int32)
    values = np.zeros(nrow * capacity_per_row, dtype=np.float32)

    row_beg = np.arange(nrow, dtype=np.int32)[:, None] % max(ncol - active_per_row - col_shift, 1)
    active_cols = row_beg + col_shift + np.arange(active_per_row, dtype=np.int32)[None, :]
    row_values = 1.0 / (1.0 + np.arange(active_per_row, dtype=np.float32))

    columns.reshape(nrow, capacity_per_row)[:, :active_per_row] = active_cols
    values.reshape(nrow, capacity_per_row)[:, :active_per_row] = row_values

    mat = wps.bsr_zeros(nrow, ncol, float, device=device)
    mat.nnz = columns.size
    mat.offsets = wp.array(offsets, dtype=int, device=device)
    mat.row_counts = wp.array(row_counts, dtype=int, device=device)
    mat.columns = wp.array(columns, dtype=int, device=device)
    mat.values = wp.array(values, dtype=float, device=device)
    return mat


def _make_duplicate_pattern(capacity_per_row: int):
    base_cols = np.array([0, 2, 1, 2, 3, 1, 4, 3], dtype=np.int32)
    base_vals = np.array([1.0, 2.0, 3.0, -0.5, 4.0, 1.0, 5.0, 0.25], dtype=np.float32)

    repeat_count = (capacity_per_row + base_cols.size - 1) // base_cols.size
    local_cols = np.concatenate([base_cols + 5 * repeat for repeat in range(repeat_count)])[:capacity_per_row]
    local_vals = np.tile(base_vals, repeat_count)[:capacity_per_row]
    return local_cols, local_vals


def _make_duplicate_triplets(nrow: int, ncol: int, capacity_per_row: int):
    local_cols, local_vals = _make_duplicate_pattern(capacity_per_row)

    row_beg = np.arange(nrow, dtype=np.int32)[:, None] % max(ncol - local_cols.max() - 1, 1)
    rows = np.repeat(np.arange(nrow, dtype=np.int32), capacity_per_row)
    columns = (row_beg + local_cols[None, :]).reshape(-1)
    values = np.tile(local_vals, nrow)
    return rows, columns, values


def _make_duplicate_candidate_matrix(nrow: int, ncol: int, capacity_per_row: int, device):
    offsets = np.arange(nrow + 1, dtype=np.int32) * capacity_per_row
    row_counts = np.full(nrow, capacity_per_row, dtype=np.int32)
    _, columns, values = _make_duplicate_triplets(nrow, ncol, capacity_per_row)

    mat = wps.bsr_zeros(nrow, ncol, float, device=device)
    mat.nnz = columns.size
    mat.offsets = wp.array(offsets, dtype=int, device=device)
    mat.row_counts = wp.array(row_counts, dtype=int, device=device)
    mat.columns = wp.array(columns, dtype=int, device=device)
    mat.values = wp.array(values, dtype=float, device=device)
    return mat


class BsrMvGappedRows:
    """Test matrix-vector multiplication on a matrix with row-local slack capacity."""

    rounds = 1
    repeat = 2
    number = 20

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._mat = _make_row_capacity_matrix(
                32768, 32768, active_per_row=4, capacity_per_row=8, device=self.device
            )
            self._x = wp.ones(shape=self._mat.shape[1], dtype=wp.float32)
            self._y = wp.zeros(shape=self._mat.shape[0], dtype=wp.float32)
            self._mat.nnz_sync()
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_mv(self._mat, self._x, self._y, alpha=1.0, beta=0.0)

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrCompressGappedRows:
    """Test compact export from row-local candidate storage with duplicate columns."""

    rounds = 1
    repeat = 2
    number = 5
    pool_size = 64

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._src_template = _make_duplicate_candidate_matrix(32768, 32768, capacity_per_row=8, device=self.device)
            self._src_index = 0
            self._src_pool = [wps.bsr_copy(self._src_template, topology="padded") for _ in range(self.pool_size)]
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        if self._src_index == len(self._src_pool):
            self._src_pool.append(wps.bsr_copy(self._src_template, topology="padded"))

        src = self._src_pool[self._src_index]
        self._src_index += 1
        wps.bsr_compress(src)

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrSetFromTripletsPaddedRows:
    """Test padded triplet construction from row-local duplicate candidates."""

    rounds = 1
    repeat = 2
    number = 5

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        nrow = 32768
        ncol = 32768
        capacity_per_row = 8
        rows, columns, values = _make_duplicate_triplets(nrow, ncol, capacity_per_row)

        with wp.ScopedDevice(self.device):
            self._rows = wp.array(rows, dtype=int, device=self.device)
            self._columns = wp.array(columns, dtype=int, device=self.device)
            self._values = wp.array(values, dtype=float, device=self.device)
            self._dest = _make_row_capacity_matrix(nrow, ncol, active_per_row=0, capacity_per_row=8, device=self.device)
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_set_from_triplets(self._dest, self._rows, self._columns, self._values, topology="padded")

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrAxpyPaddedRows:
    """Test padded topology insertion into existing row capacity."""

    rounds = 1
    repeat = 2
    number = 5

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._x = _make_row_capacity_matrix(32768, 32768, active_per_row=4, capacity_per_row=4, device=self.device)
            self._y = _make_row_capacity_matrix(32768, 32768, active_per_row=0, capacity_per_row=8, device=self.device)
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_axpy(self._x, self._y, alpha=1.0, beta=0.0, topology="padded")

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrSetFromTripletsPaddedWideRows:
    """Test padded triplet construction with wider row-local duplicate candidates."""

    rounds = 1
    repeat = 2
    number = 3

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        nrow = 32768
        ncol = 32768
        capacity_per_row = 32
        rows, columns, values = _make_duplicate_triplets(nrow, ncol, capacity_per_row)

        with wp.ScopedDevice(self.device):
            self._rows = wp.array(rows, dtype=int, device=self.device)
            self._columns = wp.array(columns, dtype=int, device=self.device)
            self._values = wp.array(values, dtype=float, device=self.device)
            self._dest = _make_row_capacity_matrix(
                nrow, ncol, active_per_row=0, capacity_per_row=capacity_per_row, device=self.device
            )
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_set_from_triplets(self._dest, self._rows, self._columns, self._values, topology="padded")

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrAxpyPaddedMergeWideRows:
    """Test padded row-local merge into existing row capacity."""

    rounds = 1
    repeat = 2
    number = 5

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._x = _make_row_capacity_matrix(
                32768, 32768, active_per_row=16, capacity_per_row=16, device=self.device, col_shift=8
            )
            self._y = _make_row_capacity_matrix(
                32768, 32768, active_per_row=16, capacity_per_row=64, device=self.device
            )
            self._work = wps.bsr_axpy_work_arrays()
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_axpy(self._x, self._y, alpha=1.0, beta=1.0, topology="padded", work_arrays=self._work)

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrSetTransposePaddedWideRows:
    """Test padded transpose into wider row-local capacity."""

    rounds = 1
    repeat = 2
    number = 3

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._src = _make_row_capacity_matrix(
                4096, 4096, active_per_row=16, capacity_per_row=16, device=self.device
            )
            self._dest = _make_row_capacity_matrix(
                4096, 4096, active_per_row=0, capacity_per_row=32, device=self.device
            )
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_set_transpose(self._dest, self._src, topology="padded")

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrMMPaddedWideRows:
    """Test padded matrix multiplication into wider row-local capacity."""

    rounds = 1
    repeat = 2
    number = 3

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._x = _make_row_capacity_matrix(32768, 32768, active_per_row=8, capacity_per_row=8, device=self.device)
            self._y = _make_row_capacity_matrix(32768, 32768, active_per_row=8, capacity_per_row=8, device=self.device)
            self._z = _make_row_capacity_matrix(32768, 32768, active_per_row=0, capacity_per_row=32, device=self.device)
            self._work = wps.bsr_mm_work_arrays()
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_mm(self._x, self._y, self._z, topology="padded", work_arrays=self._work)

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)
