# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""scanalign — align two point scans and emit a correspondence manifest.

Usage:
    python align_scans.py            # run the pipeline on the bundled scans
    python align_scans.py --stages   # also print per-stage wall time
"""

import argparse
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_A = os.path.join(HERE, "scan_a.txt")
SCAN_B = os.path.join(HERE, "scan_b.txt")
KEYPOINTS = 700


def ensure_inputs(n_a=300000, n_b=200000):
    """Materialise the bundled scans on first run (deterministic, ~20 MB)."""
    for path, n, seed in ((SCAN_A, n_a, 1), (SCAN_B, n_b, 2)):
        if os.path.exists(path):
            continue
        rng = np.random.default_rng(seed)
        pts = rng.normal(0.0, 3.0, size=(n, 3))
        inten = rng.random(n) * 2.0
        with open(path, "w") as fh:
            fh.write("# x y z intensity\n")
            for (x, y, z), i in zip(pts, inten):
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {i:.4f}\n")


def parse_scan(path):
    """Read a whitespace-delimited scan file into a list of points."""
    points = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            intensity = float(parts[3])
            if intensity <= 0.0:
                continue
            norm = math.sqrt(x * x + y * y + z * z)
            points.append((x, y, z, intensity, norm))
    return points


def select_keypoints(points, k=KEYPOINTS):
    """Keep the k brightest returns; only these take part in matching."""
    return sorted(points, key=lambda p: -p[3])[:k]


def match_points(a, b):
    """For every keypoint in A, find the nearest keypoint in B (brute force)."""
    pa = np.array([[p[0], p[1], p[2]] for p in a], dtype=np.float32)
    pb = np.array([[p[0], p[1], p[2]] for p in b], dtype=np.float32)
    d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(axis=1)
    return idx, np.sqrt(d2[np.arange(len(pa)), idx])


def write_manifest(idx, dist, path=None):
    path = path or os.path.join(HERE, "manifest.txt")
    with open(path, "w") as fh:
        for i, (j, d) in enumerate(zip(idx, dist)):
            fh.write(f"{i} {j} {d:.6f}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()

    ensure_inputs()

    t = {}
    t0 = time.perf_counter()
    a = parse_scan(SCAN_A)
    b = parse_scan(SCAN_B)
    t["parse_scans"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ka, kb = select_keypoints(a), select_keypoints(b)
    t["select_keypoints"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx, dist = match_points(ka, kb)
    t["match_points"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = write_manifest(idx, dist)
    t["write_manifest"] = time.perf_counter() - t0

    total = sum(t.values())
    print(f"matched {len(idx)} keypoints -> {out}")
    if args.stages:
        print(f"{'stage':<18}{'seconds':>10}{'share':>9}")
        for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
            print(f"{k:<18}{v:>10.3f}{100 * v / total:>8.1f}%")
        print(f"{'TOTAL':<18}{total:>10.3f}")


if __name__ == "__main__":
    main()
