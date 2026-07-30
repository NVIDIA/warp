# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""meshkit — turn a sampled field into a textured mesh.

Usage:
    python run_pipeline.py --stages
"""

import argparse
import time

import numpy as np


def neighbour_search(points, cutoff=0.25):
    """For every point, count neighbours within a cutoff (all-pairs)."""
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    return (d2 < cutoff * cutoff).sum(axis=1)


def shade_vertices(vertices, weights):
    """Dense linear algebra: project vertices through a learned basis."""
    out = vertices
    for _ in range(30):
        out = np.tanh(out @ weights)
    return out


def pack_atlas(faces):
    """Single-threaded chart packing, ported from the reference C implementation."""
    placed = []
    cursor_x = cursor_y = row_h = 0.0
    for row in faces:
        w, h = float(row[0]), float(row[1])
        if cursor_x + w > 1.0:
            cursor_x = 0.0
            cursor_y += row_h
            row_h = 0.0
        placed.append((cursor_x, cursor_y))
        cursor_x += w
        row_h = max(row_h, h)
    return placed


def write_obj(vertices, path="mesh.obj"):
    with open(path, "w") as fh:
        for v in vertices:
            fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    points = rng.random((3600, 3)).astype(np.float32)
    vertices = rng.random((4000, 256)).astype(np.float32)
    weights = rng.random((256, 256)).astype(np.float32)
    faces = (rng.random((260000, 2)) * 0.02).astype(np.float32)

    t = {}
    t0 = time.perf_counter()
    counts = neighbour_search(points)
    t["neighbour_search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    shaded = shade_vertices(vertices, weights)
    t["shade_vertices"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    placed = pack_atlas(faces)
    t["pack_atlas"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = write_obj(rng.random((60000, 3)))
    t["write_obj"] = time.perf_counter() - t0

    total = sum(t.values())
    print(f"neighbours={int(counts.sum())} shaded={shaded.shape} charts={len(placed)} -> {out}")
    if args.stages:
        print(f"{'stage':<20}{'seconds':>10}{'share':>9}")
        for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
            print(f"{k:<20}{v:>10.3f}{100 * v / total:>8.1f}%")
        print(f"{'TOTAL':<20}{total:>10.3f}")


if __name__ == "__main__":
    main()
