# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""rasterise — accumulate a triangle mesh into a raster.

Two workload sources:

    --data mesh        the shipped mesh: a triangulated grid, as the tool is used
    --data synthetic   the built-in generator (default): random vertices

Both produce the same triangle count, and the raster is the same size, so they
look interchangeable in a benchmark table.

Usage:
    python rasterise.py --stages
    python rasterise.py --data mesh --stages
"""

import argparse
import time

import numpy as np

W, H = 400, 300


def synthetic_triangles(n, seed=0):
    """Random vertices in the unit square, three at a time."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 3, 2))


def mesh_triangles(n, seed=0):
    """A triangulated grid covering the unit square: neighbouring vertices."""
    side = int(np.ceil(np.sqrt(n / 2.0)))
    a = np.linspace(0.0, 1.0, side + 1)
    gx, gy = np.meshgrid(a, a)
    v = np.stack([gx.ravel(), gy.ravel()], axis=1)
    idx = np.arange((side + 1) ** 2).reshape(side + 1, side + 1)
    v00, v01 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    v10, v11 = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([v00, v01, v10], axis=1),
                           np.stack([v01, v11, v10], axis=1)])[:n]
    return v[tris]


def mean_area(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.abs(np.cross(b - a, c - a)).mean() / 2.0)


def shade(tris):
    """Per-triangle shading weight: cheap, one pass over the vertices."""
    return tris.mean(axis=1).sum(axis=1)


def accumulate(tris, weights, width=W, height=H):
    """Scan-convert every triangle over its bounding box."""
    out = np.zeros((height, width), dtype=np.float64)
    xs = np.arange(width) + 0.5
    ys = np.arange(height) + 0.5
    for t in range(tris.shape[0]):
        (x0, y0), (x1, y1), (x2, y2) = tris[t] * (width, height)
        lo_x, hi_x = int(max(0, min(x0, x1, x2))), int(min(width, max(x0, x1, x2) + 1))
        lo_y, hi_y = int(max(0, min(y0, y1, y2))), int(min(height, max(y0, y1, y2) + 1))
        if hi_x <= lo_x or hi_y <= lo_y:
            continue
        px = xs[lo_x:hi_x][None, :]
        py = ys[lo_y:hi_y][:, None]
        d = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if d == 0:
            continue
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / d
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / d
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        out[lo_y:hi_y, lo_x:hi_x] += inside * weights[t]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triangles", type=int, default=2_000)
    ap.add_argument("--data", choices=("synthetic", "mesh"), default="synthetic")
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()

    gen = synthetic_triangles if args.data == "synthetic" else mesh_triangles
    tris = gen(args.triangles)
    area = mean_area(tris)

    t0 = time.perf_counter()
    weights = shade(tris)
    t_shade = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = accumulate(tris, weights)
    t_accumulate = time.perf_counter() - t0
    total = t_shade + t_accumulate

    print(f"workload={args.data}  triangles={args.triangles:,}  "
          f"mean triangle area={area:.6f} of the unit square  "
          f"(= {area*W*H:,.0f} pixels each, {area*W*H*args.triangles/(W*H):,.1f}x canvas coverage)")
    if args.stages:
        print(f"{'stage':<12}{'seconds':>10}{'share':>9}")
        for name, dt in (("shade", t_shade), ("accumulate", t_accumulate)):
            print(f"{name:<12}{dt:>10.4f}{100*dt/total:>8.1f}%")
    print(f"total {total:.4f} s, {int((out > 0).sum()):,} covered pixels")


if __name__ == "__main__":
    main()
