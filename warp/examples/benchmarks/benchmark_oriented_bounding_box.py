# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ``warp.geometry.oriented_bounding_box``: refinement quality and subsample speed.

Two effects are measured against the axis-aligned bounding box (AABB) baseline:

* **Refinement quality** -- how tight the box gets (``vol / aabb``) versus ``num_samples``,
  with and without the coarse-to-fine refinement rounds. Refinement lets a small sample
  count reach the quality of a much larger one.
* **Subsample speed** -- wall-clock of the fit on a large cloud when candidate orientations
  are scored over a strided subsample (``max_search_points``) versus every point. The
  returned box is re-measured over the full cloud either way, so it stays exact.

    uv run warp/examples/benchmarks/benchmark_oriented_bounding_box.py
"""

import numpy as np

import warp as wp
import warp.geometry as geo


def _rotated_rod(rng, num_points, extents=(2.0, 0.4, 0.15)):
    """A strongly elongated, off-axis point cloud: orientation matters a lot here."""
    extents = np.asarray(extents, dtype=np.float64)
    axis = np.array([0.3, -0.7, 0.5])
    axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(0.9) * k + (1.0 - np.cos(0.9)) * (k @ k)
    pts = rng.uniform(-0.5, 0.5, (num_points, 3)) * extents
    return np.ascontiguousarray(pts @ rotation.T, dtype=np.float32)


def _aabb_volume(p_np):
    return float(np.prod(p_np.max(axis=0) - p_np.min(axis=0)))


def _measure(points, **kwargs):
    _, _, measure = geo.oriented_bounding_box(points, **kwargs)
    return float(measure.numpy()[0])


def _time_ms(points, reps=5, **kwargs):
    geo.oriented_bounding_box(points, **kwargs)  # warm up
    wp.synchronize_device()
    with wp.ScopedTimer("obb", print=False) as timer:
        for _ in range(reps):
            geo.oriented_bounding_box(points, **kwargs)
        wp.synchronize_device()
    return timer.elapsed / reps


def main():
    rng = np.random.default_rng(0)

    # Isolate the spiral (PCA off) so refinement's contribution is visible; with PCA on, the
    # principal axes already nail an elongated rod regardless of sample count.
    print("Refinement quality (vol / aabb, lower is better) on a rotated rod, 50k points, PCA off:")
    p_np = _rotated_rod(rng, 50_000)
    points = wp.array(p_np, dtype=wp.vec3)
    aabb = _aabb_volume(p_np)
    print(f"  {'num_samples':>12}{'refine=0':>12}{'refine=4':>12}{'refine=8':>12}")
    for num_samples in (16, 64, 256, 1024, 4096):
        row = [
            _measure(
                points,
                num_samples=num_samples,
                num_refine_starts=1,
                num_refine_iterations=r,
                num_spin_samples=0,
                include_pca=False,
            )
            / aabb
            for r in (0, 4, 8)
        ]
        print(f"  {num_samples:>12}{row[0]:>12.4f}{row[1]:>12.4f}{row[2]:>12.4f}")

    print("\nSubsample speed on a large cloud (num_samples=4096, num_refine_iterations=4, default starts and spin):")
    print(f"  {'points':>12}{'full (ms)':>14}{'subsample (ms)':>18}{'speedup':>10}{'vol ratio':>12}")
    for n in (250_000, 1_000_000, 4_000_000):
        p_np = _rotated_rod(rng, n)
        points = wp.array(p_np, dtype=wp.vec3)
        full_ms = _time_ms(points, max_search_points=None)
        sub_ms = _time_ms(points, max_search_points=100_000)
        full_vol = _measure(points, max_search_points=None)
        sub_vol = _measure(points, max_search_points=100_000)
        print(f"  {n:>12}{full_ms:>14.2f}{sub_ms:>18.2f}{full_ms / sub_ms:>9.1f}x{sub_vol / full_vol:>12.4f}")


if __name__ == "__main__":
    main()
