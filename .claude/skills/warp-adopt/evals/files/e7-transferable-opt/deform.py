# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cageform — evaluate a cage deformation for every sample against every cell.

For each sample point and each cage cell, evaluate the cell's three linear
blend weights and keep the cell with the smallest total weight magnitude.

Usage:
    python deform.py --stages
"""

import argparse
import time

import numpy as np


def make_case(seed=0, n_cells=3000, n_points=40000):
    rng = np.random.default_rng(seed)
    centres = rng.random((n_cells, 1, 2)).astype(np.float32)
    cells = centres + rng.normal(0.0, 0.008, size=(n_cells, 3, 2)).astype(np.float32)
    points = rng.random((n_points, 2)).astype(np.float32)
    return cells, points


def blend_weights(cells, points):
    """(C,3,2) cells x (P,2) points -> (C,P) total weight magnitude.

    Each weight is recomputed from the cell's raw corner positions.
    """
    v0, v1, v2 = cells[:, 0], cells[:, 1], cells[:, 2]

    def cross(p, a, b):
        return (p[None, :, 0] - a[:, None, 0]) * (b[:, None, 1] - a[:, None, 1]) - (
            p[None, :, 1] - a[:, None, 1]
        ) * (b[:, None, 0] - a[:, None, 0])

    area = (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]) - (v2[:, 1] - v0[:, 1]) * (v1[:, 0] - v0[:, 0])
    w0 = cross(points, v1, v2) / area[:, None]
    w1 = cross(points, v2, v0) / area[:, None]
    w2 = cross(points, v0, v1) / area[:, None]
    return np.abs(w0) + np.abs(w1) + np.abs(w2)


def select(cells, points, block=2048):
    best_i = np.zeros(len(points), dtype=np.int64)
    best_d = np.full(len(points), np.inf, dtype=np.float32)
    for s in range(0, len(points), block):
        chunk = points[s : s + block]
        d = blend_weights(cells, chunk)
        best_i[s : s + block] = d.argmin(axis=0)
        best_d[s : s + block] = d.min(axis=0)
    return best_i, best_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()
    cells, points = make_case()

    t0 = time.perf_counter()
    idx, dist = select(cells, points)
    dt = time.perf_counter() - t0

    print(f"selected cells for {len(idx)} points, mean weight {dist.mean():.6f}")
    if args.stages:
        print(f"select: {dt:.3f} s over {len(cells)} cells x {len(points)} points")


if __name__ == "__main__":
    main()
