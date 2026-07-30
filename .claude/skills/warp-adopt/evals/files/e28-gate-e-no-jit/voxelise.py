# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""surveykit - on-device sweep processing for the SV-40 appliance.

See README.md for the hardware and deployment constraints.

`neighbourhood_features` is the hot stage: for every point it scans the sweep
for points inside the feature radius and accumulates a small descriptor. It is
irregular, branchy, spatially local and embarrassingly parallel - which is
exactly why it keeps coming up.

Commands:

    python voxelise.py --profile     # per-stage split for one sweep
    python voxelise.py --sweep-size  # how the hot stage scales

Only numpy is required.
"""
import argparse
import time

import numpy as np

N_POINTS = 6_000
RADIUS = 0.06


def sweep(n=N_POINTS, seed=17):
    """One survey sweep: points on a rough ground plane with structure."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    z = (0.08 * np.sin(9 * xy[:, 0]) * np.cos(7 * xy[:, 1])
         + rng.normal(scale=0.01, size=n))
    return np.column_stack([xy, z])


def deskew(pts):
    """Motion compensation. Dense, regular, already fast."""
    t = np.linspace(0.0, 1.0, len(pts))[:, None]
    return pts + np.array([0.004, -0.002, 0.0]) * t


def neighbourhood_features(pts, r=RADIUS):
    """The hot stage: per-point neighbourhood descriptor within `r`.

    Chunked so the pair matrix stays bounded, but still an all-pairs scan.
    """
    n = len(pts)
    out = np.empty((n, 4))
    chunk = 512
    for s in range(0, n, chunk):
        q = pts[s:s + chunk]
        d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        m = d2 < r * r
        cnt = m.sum(1)
        w = m / np.maximum(cnt, 1)[:, None]
        centroid = w @ pts
        spread = np.sqrt(np.maximum(
            (w * d2).sum(1) - ((w * np.sqrt(d2)).sum(1)) ** 2, 0.0))
        out[s:s + chunk] = np.column_stack([cnt, centroid[:, 2], spread,
                                            np.linalg.norm(
                                                centroid - q, axis=1)])
    return out


def classify(feat):
    """Runs on the GPU today, through libsurvey_cuda.so."""
    w = np.array([0.3, -1.2, 2.0, 0.5])
    return (feat @ w > 0.4).astype(np.int8)


def emit(labels):
    return int(labels.sum())


def run(n=N_POINTS):
    pts = sweep(n)
    timings = {}
    t0 = time.perf_counter()
    pts = deskew(pts)
    timings['deskew'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feat = neighbourhood_features(pts)
    timings['neighbourhood_features'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lab = classify(feat)
    timings['classify'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    emit(lab)
    timings['emit'] = time.perf_counter() - t0
    return timings


def cmd_profile():
    timings = run()
    total = sum(timings.values())
    print(f'one sweep, {N_POINTS} points, feature radius {RADIUS}\n')
    print(f"{'stage':<26} {'ms':>10} {'share':>8}")
    for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f'{k:<26} {v * 1e3:>10.1f} {100 * v / total:>7.1f}%')
    print(f"\n{'total':<26} {total * 1e3:>10.1f} ms")


def cmd_sweep_size():
    print(f"{'points':>8} {'features ms':>13} {'vs n^2':>8}")
    base = None
    for n in (1_500, 3_000, 6_000, 12_000):
        pts = deskew(sweep(n))
        t0 = time.perf_counter()
        neighbourhood_features(pts)
        t = (time.perf_counter() - t0) * 1e3
        if base is None:
            base, base_n = t, n
        print(f'{n:>8} {t:>13.1f} {t / base / (n / base_n) ** 2:>8.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--sweep-size', action='store_true')
    a = ap.parse_args()
    if a.sweep_size:
        cmd_sweep_size()
    else:
        cmd_profile()
