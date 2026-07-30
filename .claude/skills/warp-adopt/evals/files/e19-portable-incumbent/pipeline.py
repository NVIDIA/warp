# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fieldkit batch pipeline.

Stages: load -> resample_field -> normalise -> emit.

`resample_field` is the irregular one: per-sample gather with a variable
neighbourhood, early exit, and a scatter-accumulate. It is also the one the
shared `.cu` source in `kernels/` implements for all three supported vendors
(see README.md).

Run `python pipeline.py --profile`.
"""

import argparse
import time

import numpy as np


def load(n=12000, seed=0):
    """Decode the acquisition buffer and sort into acquisition order."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2 ** 31, size=(n, 4), dtype=np.int64)
    for _ in range(14):
        raw = (raw * 1103515245 + 12345) % (2 ** 31)
    pts = (raw[:, :3] / (2 ** 31)) * 8.0
    val = (raw[:, 3] / (2 ** 31)) * 2.0 - 1.0
    order = np.argsort(pts[:, 0], kind="stable")
    return pts[order], val[order]


def smooth(f, n_out=12000, passes=1400):
    """Reconstruct the field at sample resolution and post-filter it."""
    g = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(f)), f)
    for _ in range(passes):
        g = 0.25 * np.roll(g, 1) + 0.5 * g + 0.25 * np.roll(g, -1)
    return g


def resample_field(pts, val, centres, radius=0.30, chunk=96):
    """Irregular: variable neighbourhood per centre, early exit, accumulate."""
    r2 = radius * radius
    out = np.zeros(len(centres))
    cnt = np.zeros(len(centres))
    for s in range(0, len(centres), chunk):
        c = centres[s:s + chunk]
        d = ((c[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        m = d <= r2
        w = np.where(m, 1.0 / (d + 1e-6), 0.0)
        out[s:s + chunk] = (w * val[None, :]).sum(1)
        cnt[s:s + chunk] = m.sum(1)
    return out / np.maximum(cnt, 1.0)


def normalise(f):
    return (f - f.mean()) / (f.std() + 1e-9)


def emit(f, reps=120):
    """Rank and write the output field."""
    acc = 0.0
    for _ in range(reps):
        acc += float(np.abs(np.sort(f)).cumsum()[-1])
    return acc / reps


def run(n=12000, m=700, seed=0, profile=False):
    t = {}
    t0 = time.perf_counter()
    pts, val = load(n, seed)
    rng = np.random.default_rng(seed + 1)
    centres = rng.uniform(0, 8, size=(m, 3))
    t["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    f = resample_field(pts, val, centres)
    t["resample_field"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    f = smooth(f)
    t["smooth"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    f = normalise(f)
    t["normalise"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = emit(f)
    t["emit"] = time.perf_counter() - t0

    if profile:
        total = sum(t.values())
        print(f"batch: {n} samples, {m} centres\n")
        for k, v in t.items():
            print(f"  {k:16s} {v*1e3:9.1f} ms   {100*v/total:5.1f}%")
        print(f"  {'TOTAL':16s} {total*1e3:9.1f} ms")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--centres", type=int, default=700)
    a = ap.parse_args()
    r = run(a.samples, a.centres, profile=a.profile)
    if not a.profile:
        print(f"emit = {r:.6f}")
