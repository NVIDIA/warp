# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""relaxkit — iterative field relaxation with a histogram summary.

The project already ships a device backend (``backend.py``-style helpers below,
used by :func:`histogram` and :func:`normalise`).  The relaxation loop predates
it and still runs one point at a time on the host.

Usage:
    python relaxkit.py                 # relax the bundled field
    python relaxkit.py --stages        # per-stage wall time and device bytes
    python relaxkit.py --points 400000 --iters 40
"""

import argparse
import time

import numpy as np

DEVICE_BYTES = 0
DEVICE_LAUNCHES = 0


# --------------------------------------------------------------- device backend
# The project's existing accelerator backend.  Real deployments dispatch these
# to compiled device kernels; the accounting is what matters for this fixture.
class DeviceArray:
    def __init__(self, data):
        global DEVICE_BYTES
        self._a = np.asarray(data)
        DEVICE_BYTES += self._a.nbytes

    @property
    def shape(self):
        return self._a.shape

    def to_host(self):
        return np.array(self._a)


def to_device(a):
    return a if isinstance(a, DeviceArray) else DeviceArray(a)


def as_host(a):
    return a.to_host() if isinstance(a, DeviceArray) else np.asarray(a)


def device_histogram(values, bins):
    """Device kernel: bin values.  Already accelerated."""
    global DEVICE_LAUNCHES
    DEVICE_LAUNCHES += 1
    v = as_host(values)
    idx = np.clip((v * bins).astype(np.int64), 0, bins - 1)
    return to_device(np.bincount(idx, minlength=bins))


def device_normalise(values):
    """Device kernel: scale to [0, 1].  Already accelerated."""
    global DEVICE_LAUNCHES
    DEVICE_LAUNCHES += 1
    v = as_host(values)
    lo, hi = v.min(), v.max()
    return to_device((v - lo) / max(1e-12, hi - lo))


# ------------------------------------------------------------------- hot path
def relax(points, field, iters=20, step=0.01):
    """Advect every point along the field gradient, ``iters`` times.

    One point at a time, on the host: this loop predates the device backend and
    was never moved onto it.
    """
    pts = as_host(points).copy()
    fld = as_host(field)
    n_cells = fld.shape[0]
    for _ in range(iters):
        for i in range(pts.shape[0]):
            cell = int(pts[i] * n_cells)
            if cell < 0:
                cell = 0
            elif cell >= n_cells:
                cell = n_cells - 1
            pts[i] = min(1.0, max(0.0, pts[i] + step * fld[cell]))
    return pts


def run(n_points=100_000, n_cells=4096, iters=20, bins=256, seed=0, stages=False):
    rng = np.random.default_rng(seed)
    points = to_device(rng.random(n_points))
    field = to_device(rng.random(n_cells) - 0.5)

    timings = []

    def timed(name, fn):
        before_b, before_l = DEVICE_BYTES, DEVICE_LAUNCHES
        t0 = time.perf_counter()
        out = fn()
        timings.append((name, time.perf_counter() - t0,
                        DEVICE_BYTES - before_b, DEVICE_LAUNCHES - before_l))
        return out

    normalised = timed("normalise", lambda: device_normalise(field))
    moved = timed("relax", lambda: relax(points, normalised, iters=iters))
    hist = timed("histogram", lambda: device_histogram(moved, bins))

    if stages:
        print(f"{'stage':<12}{'seconds':>10}{'device bytes':>15}{'device launches':>18}")
        for name, dt, nbytes, launches in timings:
            print(f"{name:<12}{dt:>10.3f}{nbytes:>15,}{launches:>18}")
    return moved, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=100_000)
    ap.add_argument("--cells", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()
    moved, hist = run(args.points, args.cells, args.iters, stages=args.stages)
    print(f"relaxed {args.points:,} points over {args.iters} iterations; "
          f"histogram total {int(as_host(hist).sum()):,}")


if __name__ == "__main__":
    main()
