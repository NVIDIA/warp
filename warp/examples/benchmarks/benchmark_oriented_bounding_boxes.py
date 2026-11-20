# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ``warp.geometry.oriented_bounding_boxes`` against a loop of ``oriented_bounding_box`` calls.

Batching exists to amortize launch overhead when many small point sets are fitted. This
benchmark sweeps the number of point sets and their size, and reports the wall-clock of
fitting all of them one at a time versus in a single batched call, plus the ratio of the
achieved measures (which is exactly 1 when PCA is off, since batching must not change results).

    uv run warp/examples/benchmarks/benchmark_oriented_bounding_boxes.py
"""

import numpy as np

import warp as wp
import warp.geometry as geo


def _rotated_rods(rng, num_shapes, num_points):
    """Elongated, randomly rotated point clouds of equal size, stacked as ``(num_shapes, num_points, 3)``."""
    rods = rng.uniform(-0.5, 0.5, (num_shapes, num_points, 3)) * np.array([2.0, 0.4, 0.15])
    q, _ = np.linalg.qr(rng.standard_normal((num_shapes, 3, 3)))
    return np.einsum("sij,spj->spi", q, rods).astype(np.float32)


def _time_ms(fn, reps):
    fn()  # warm up (and compile)
    wp.synchronize_device()
    with wp.ScopedTimer("obb", print=False) as timer:
        for _ in range(reps):
            fn()
        wp.synchronize_device()
    return timer.elapsed / reps


def _bench_row(rng, num_shapes, num_points):
    kwargs = {"include_pca": False}
    shapes_np = _rotated_rods(rng, num_shapes, num_points)
    singles = [wp.array(s, dtype=wp.vec3) for s in shapes_np]
    points = wp.array(shapes_np.reshape(-1, 3), dtype=wp.vec3)
    offsets = wp.array(np.arange(num_shapes + 1, dtype=np.int32) * num_points, dtype=wp.int32)

    def loop():
        return [geo.oriented_bounding_box(s, **kwargs) for s in singles]

    def batched():
        return geo.oriented_bounding_boxes(points, offsets, validate=False, max_points_per_shape=num_points, **kwargs)

    reps = 3 if num_shapes * num_points >= 1_000_000 else 5
    loop_ms = _time_ms(loop, reps)
    batched_ms = _time_ms(batched, reps)

    loop_measure = np.array([float(r[2].numpy()[0]) for r in loop()])
    batched_measure = batched()[2].numpy()
    ratio = float(np.max(np.abs(batched_measure / loop_measure - 1.0))) + 1.0
    print(
        f"  {num_shapes:>7}{num_points:>14}{loop_ms:>12.2f}{batched_ms:>14.2f}"
        f"{loop_ms / batched_ms:>9.1f}x{ratio:>15.4f}"
    )


def main():
    rng = np.random.default_rng(0)

    print("Fitting many point sets: loop of oriented_bounding_box vs one oriented_bounding_boxes call (PCA off).")
    print(
        f"  {'shapes':>7}{'points/shape':>14}{'loop (ms)':>12}{'batched (ms)':>14}{'speedup':>10}{'measure ratio':>15}"
    )
    for num_points in (32, 512, 8192, 100_000):
        for num_shapes in (1, 16, 256, 1024):
            if num_shapes * num_points <= 20_000_000:
                _bench_row(rng, num_shapes, num_points)


if __name__ == "__main__":
    main()
