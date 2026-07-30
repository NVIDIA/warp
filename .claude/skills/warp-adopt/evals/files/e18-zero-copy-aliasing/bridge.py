# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""bridge - a zero-copy interop bridge and the cache built on top of it.

`Bridge.wrap()` is zero-copy by design: it hands the backend a view onto the
caller's array so no copy is made on the hot path. `AccelCache.build()` wraps
the caller's vertices; `AccelCache.refit()` pushes updated vertices into the
structure without rebuilding it.

The two compose into a trap. Every single-shot check passes: build, query, and
compare against the reference all agree exactly. The damage only appears when
`refit()` is called in a loop, which is the case the cache exists for.

This fixture is about validating the HARNESS, not the kernel. Run
`python bridge.py --report`.
"""

import argparse

import numpy as np


class Bridge:
    """Zero-copy handle onto a caller-owned array."""

    def __init__(self, arr):
        self._view = arr          # NOT a copy

    def push(self, values):
        self._view[:] = values    # writes through into the caller's array

    def read(self):
        return self._view


class AccelCache:
    """Caller-owned acceleration structure with build / refit / query."""

    def __init__(self, verts):
        self.bridge = Bridge(verts)
        self._centroid = verts.mean(0)

    def refit(self, verts):
        self.bridge.push(verts)
        self._centroid = self.bridge.read().mean(0)
        return self

    def query(self, points):
        v = self.bridge.read()
        d = ((points[:, None, :] - v[None, :, :]) ** 2).sum(-1)
        return d.min(1)


def reference_query(points, verts):
    d = ((points[:, None, :] - verts[None, :, :]) ** 2).sum(-1)
    return d.min(1)


def make_case(n_verts=120, n_points=400, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, (n_verts, 3)), rng.normal(0, 1, (n_points, 3)))


def static_checks(verts, points):
    """Everything a single-shot audit would run."""
    out = {}
    c = AccelCache(verts.copy())
    out["build+query matches reference"] = bool(
        np.allclose(c.query(points), reference_query(points, verts)))
    v2 = verts + 0.05
    c2 = AccelCache(verts.copy())
    c2.refit(v2)
    out["refit+query matches reference"] = bool(
        np.allclose(c2.query(points), reference_query(points, v2)))
    c3 = AccelCache(verts.copy())
    a = c3.query(points)
    b = c3.query(points)
    out["repeated query deterministic"] = bool(np.array_equal(a, b))
    return out


def aliasing_check(verts, points):
    """The check the static suite is missing."""
    owner = verts.copy()
    before = owner.copy()
    c = AccelCache(owner)
    c.refit(owner + 0.05)
    return bool(np.array_equal(owner, before)), float(np.abs(owner - before).max())


def loop(verts, points, steps=40, aliased=True):
    """A deforming-geometry loop: displacement grows by a fixed step each
    iteration and the cache is refit. Returns the mean-distance history."""
    base = verts.copy()
    delta = np.zeros_like(base)
    cache = AccelCache(base if aliased else base.copy())
    hist = []
    for i in range(steps):
        delta = delta + 0.01
        moved = base + delta
        cache.refit(moved)
        hist.append(float(cache.query(points).mean()))
    return hist, base


def report():
    verts, points = make_case()
    print("static single-shot checks (what a first audit runs):")
    for k, v in static_checks(verts, points).items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    ok, drift = aliasing_check(verts, points)
    print(f"\nthe check that is missing:")
    print(f"  [{'PASS' if ok else 'FAIL'}] refit does not mutate the caller's "
          f"array   (max drift {drift:.4f})")

    hist_a, base_a = loop(verts, points, aliased=True)
    hist_b, base_b = loop(verts, points, aliased=False)
    print(f"\n40-step refit loop:")
    print(f"  aliased  (as written) : first {hist_a[0]:.4f}  last {hist_a[-1]:.4f}")
    print(f"  corrected (own copy)  : first {hist_b[0]:.4f}  last {hist_b[-1]:.4f}")
    rel = abs(hist_a[-1] - hist_b[-1]) / max(abs(hist_b[-1]), 1e-30)
    print(f"  relative error introduced by the aliasing bug: {rel*100:.1f}%")
    print(f"  caller's base array drifted by {np.abs(base_a - verts).max():.4f} "
          f"(corrected run: {np.abs(base_b - verts).max():.4f})")
    print("\nThe aliased loop produces a smooth, plausible curve. Nothing about "
          "it looks\nwrong; it is simply a different function. A result of this "
          "shape must be\nattributed to the harness before it is attributed to "
          "the backend.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        v, p = make_case()
        c = AccelCache(v)
        print(f"mean d2 = {c.query(p).mean():.6f}")
