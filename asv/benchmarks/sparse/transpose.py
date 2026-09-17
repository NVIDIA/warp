# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark CUDA Graph BSR transposes; see transpose.md for reproduction."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import warp as wp
import warp.sparse as wps


class BsrSetTransposeCompact:
    """Benchmark captured compact transposes across row lengths and storage occupancy."""

    params = ["stencil8", "stencil32", "dense", "overallocated", "wide_hot_column", "shared_columns8"]
    param_names = ["pattern"]
    # Let ASV calibrate repetitions. Each call times 100 operations to amortize
    # launch/synchronization overhead (about 1-300 ms per batch on an RTX 4090).
    transposes_per_batch = 100

    def setup(self, pattern):
        wp.init()
        self.device = wp.get_device("cuda:0")
        # Hold active blocks near 512K while varying the stencil width.
        nrow = ncol = 65536
        active_rows = nrow
        degree = 8
        if pattern == "stencil32":
            nrow = ncol = active_rows = 16384
            degree = 32
        elif pattern == "dense":
            nrow = ncol = active_rows = degree = 1024
        elif pattern == "overallocated":
            # Teapot-sized reserved S2/Q1 spaces: 20 and 8 nodes per 64K cells.
            # 3000 rows * 8 blocks approximates its measured ~22K active blocks.
            nrow, ncol, active_rows = 20 * 65536, 8 * 65536, 3000
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
        capacity = nrow * degree if pattern == "overallocated" else nnz
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
                for _ in range(self.transposes_per_batch):
                    wps.bsr_set_transpose(self.dest, self.src)
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def time_100_transposes(self, pattern):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)


def main():
    """Run the same captured batches as ASV and export per-transpose timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--patterns", nargs="+", choices=BsrSetTransposeCompact.params, default=BsrSetTransposeCompact.params
    )
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least two")
    wp.init()
    result = {
        "warp_version": wp.__version__,
        "warp_path": str(Path(wp.__file__).resolve()),
        "device": wp.get_device("cuda:0").name,
        "platform": platform.platform(),
        "transposes_per_batch": BsrSetTransposeCompact.transposes_per_batch,
        "patterns": {},
    }
    for pattern in args.patterns:
        benchmark = BsrSetTransposeCompact()
        benchmark.setup(pattern)
        # Warm the captured graph before sampling; setup/compilation are excluded.
        benchmark.time_100_transposes(pattern)
        samples = []
        for _ in range(args.samples):
            start = time.perf_counter()
            benchmark.time_100_transposes(pattern)
            samples.append(1000.0 * (time.perf_counter() - start) / benchmark.transposes_per_batch)
        result["patterns"][pattern] = {"median_ms": float(np.median(samples)), "samples_ms": samples}
        print(pattern, result["patterns"][pattern], flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
