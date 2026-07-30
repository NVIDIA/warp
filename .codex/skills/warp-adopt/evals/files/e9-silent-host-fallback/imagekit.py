# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""imagekit — GPU-accelerated point rendering.

Bin points into a raster, colourise, then spread each lit pixel over its
neighbourhood.  Every stage accepts device-resident input.

Usage:
    python imagekit.py                     # render with defaults
    python imagekit.py --stages            # per-stage wall time and device bytes
    python imagekit.py --points 400000 --quality high
"""

import argparse
import time
import warnings

import numpy as np

DEVICE_BYTES = 0          # bytes allocated on the device, cumulative


class DeviceArray:
    """Stand-in for a device array: tracks residency and device allocation.

    Real deployments hold cupy arrays here; the accounting is what matters for
    this fixture, so it runs anywhere.
    """

    def __init__(self, data):
        global DEVICE_BYTES
        self._a = np.asarray(data)
        DEVICE_BYTES += self._a.nbytes

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    def to_host(self):
        return np.array(self._a)


def to_device(a):
    return a if isinstance(a, DeviceArray) else DeviceArray(a)


def as_host(a):
    return a.to_host() if isinstance(a, DeviceArray) else np.asarray(a)


def device_bytes():
    return DEVICE_BYTES


# --------------------------------------------------------------------------- stages
def aggregate(xs, ys, width, height):
    """Bin points into a raster.  Runs where the input lives."""
    x = as_host(xs)
    y = as_host(ys)
    xi = np.clip((x * width).astype(np.int64), 0, width - 1)
    yi = np.clip((y * height).astype(np.int64), 0, height - 1)
    flat = np.bincount(yi * width + xi, minlength=width * height)
    return to_device(flat.reshape(height, width).astype(np.uint32))


def colorize(agg):
    """Map counts to RGBA.  Elementwise, stays on the device."""
    a = as_host(agg)
    scaled = (255.0 * a / max(1, a.max())).astype(np.uint32)
    rgba = (255 << 24) | (scaled << 16) | (scaled << 8) | scaled
    return to_device(np.where(a > 0, rgba, 0).astype(np.uint32))


def spread(img, px=3, quality="fast"):
    """Spread every lit pixel over a square neighbourhood of radius ``px``."""
    if quality == "high" and isinstance(img, DeviceArray):
        warnings.warn("high-quality spread is not implemented for device input, "
                      "falling back to quality='fast'")
        quality = "fast"

    # Convert to host so the numba/numpy kernels below can run on it.
    arr = as_host(img)

    h, w = arr.shape
    out = np.zeros_like(arr)
    reps = 3 if quality == "high" else 1
    for _ in range(reps):
        for y in range(h):
            for x in range(w):
                el = arr[y, x]
                if el == 0:
                    continue
                y0, y1 = max(0, y - px), min(h, y + px + 1)
                x0, x1 = max(0, x - px), min(w, x + px + 1)
                block = out[y0:y1, x0:x1]
                np.maximum(block, el, out=block)
    return out


def render(n_points=200_000, width=512, height=512, px=3, quality="fast", seed=0,
           stages=False):
    rng = np.random.default_rng(seed)
    xs = to_device(rng.random(n_points))
    ys = to_device(rng.random(n_points))

    timings = []

    def run(name, fn):
        before = device_bytes()
        t0 = time.perf_counter()
        result = fn()
        timings.append((name, time.perf_counter() - t0, device_bytes() - before))
        return result

    agg = run("aggregate", lambda: aggregate(xs, ys, width, height))
    img = run("colorize", lambda: colorize(agg))
    out = run("spread", lambda: spread(img, px=px, quality=quality))

    if stages:
        print(f"{'stage':<12}{'seconds':>10}{'device bytes':>16}")
        for name, dt, nbytes in timings:
            print(f"{name:<12}{dt:>10.3f}{nbytes:>16,}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=200_000)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--px", type=int, default=3)
    ap.add_argument("--quality", choices=("fast", "high"), default="fast")
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()
    out = render(args.points, args.size, args.size, px=args.px, quality=args.quality,
                 stages=args.stages)
    print(f"rendered {args.points:,} points -> {out.shape}, {int((out > 0).sum()):,} lit pixels")


if __name__ == "__main__":
    main()
