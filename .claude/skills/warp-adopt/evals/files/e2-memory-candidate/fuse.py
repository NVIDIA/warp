# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""voxfuse — fuse a batch of range images into a voxel volume.

Usage:
    python fuse.py                      # default CLI run
    python fuse.py --resolution 128     # larger volume
    python fuse.py --stages             # per-stage wall time and peak RSS
"""

import argparse
import resource
import time

import numpy as np


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def render_views(n_views=64, height=360, width=480, seed=0):
    """Produce the range images that will be fused (the expensive part)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_views, height, width), dtype=np.float32)
    for i in range(n_views):
        img = rng.random((height, width), dtype=np.float32)
        for _ in range(32):
            img = (img + np.roll(img, 1, 0) + np.roll(img, -1, 0) + np.roll(img, 1, 1) + np.roll(img, -1, 1)) / 5.0
        out[i] = 2.0 + img
    return out


def integrate_volume(depth_images, resolution=256, batch=10):
    """Project every voxel into every camera of a batch and fuse.

    resolution: voxels per side of the cubic volume.
    batch:      how many views are integrated per call.
    """
    n = resolution ** 3
    grid = np.stack(
        np.meshgrid(*(np.linspace(-1.0, 1.0, resolution, dtype=np.float32),) * 3, indexing="ij"),
        axis=0,
    ).reshape(3, n)
    homo = np.concatenate([grid, np.ones((1, n), dtype=np.float32)], axis=0)

    values = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)

    for start in range(0, len(depth_images), batch):
        views = depth_images[start : start + batch]
        b = len(views)
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (b, 1, 1))
        extrinsics[:, 2, 3] = -3.0
        # every voxel, in every camera of the batch
        cam = extrinsics @ homo[None, :, :]
        depth = np.sqrt((cam[:, :3, :] ** 2).sum(axis=1))
        sampled = np.resize(views.reshape(b, -1), depth.shape).astype(np.float32)
        sdf = np.clip(sampled - depth, -0.1, 0.1)
        valid = (depth > 0).astype(np.float32)
        values += (sdf * valid).sum(axis=0)
        weights += valid.sum(axis=0)

    return values / np.maximum(weights, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=32)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()

    t = {}
    t0 = time.perf_counter()
    depth = render_views()
    t["render_views"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vol = integrate_volume(depth, resolution=args.resolution, batch=args.batch)
    t["integrate_volume"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    surface = int((np.abs(vol) < 0.05).sum())
    t["extract_surface"] = time.perf_counter() - t0

    total = sum(t.values())
    print(f"resolution={args.resolution} batch={args.batch} surface_voxels={surface}")
    if args.stages:
        print(f"{'stage':<18}{'seconds':>10}{'share':>9}")
        for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
            print(f"{k:<18}{v:>10.3f}{100 * v / total:>8.1f}%")
        print(f"{'TOTAL':<18}{total:>10.3f}")
        print(f"peak RSS: {peak_rss_mib():.0f} MiB")


if __name__ == "__main__":
    main()
