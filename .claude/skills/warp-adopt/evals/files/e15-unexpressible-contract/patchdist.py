# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""patchdist - squared distance from points to a triangulated patch surface.

Public contract
---------------
`patch_distance(points, tris, min_patch_area=MIN_PATCH_AREA)` returns, for each
point, the squared distance to the closest triangle and that triangle's index.

`min_patch_area` is an **absolute** area threshold, documented as:

    Triangles whose area is below this value are treated as their three
    boundary segments rather than as a filled face.

That rule is intentional and is covered by `test_patchdist.py`. It is part of
the contract, not an accident.

`nearest_patch_fast()` is what a general-purpose accelerated closest-point
primitive gives you: the ordinary point-triangle distance, with no notion of
`min_patch_area`. It is not a drop-in replacement.

Run `python patchdist.py --report` for the summary this fixture exists to show.
"""

import argparse
import time

import numpy as np

MIN_PATCH_AREA = 5e-3


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------
def make_surface(n):
    """n x n grid over the unit square, split into 2*n^2 triangles in z=0.

    Every triangle has area 1/(2*n^2), so `n` alone decides whether the asset
    sits above or below MIN_PATCH_AREA.
    """
    xs = np.linspace(0.0, 1.0, n + 1)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[:-1, 1:].ravel()
    d = idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)], 0)
    return verts[faces]  # (T, 3, 3)


SHIPPED_N = 32   # the asset the product actually ships: 2048 triangles
TOY_N = 8        # the size a developer reaches for when writing a quick check


def sample_points(count, seed=0, spread=0.10, zsigma=0.02):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-spread, 1.0 + spread, size=(count, 2))
    z = rng.normal(0.0, zsigma, size=(count, 1))
    return np.concatenate([xy, z], axis=1)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _areas(tris):
    return 0.5 * np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=-1)


def _seg_d2(p, a, b):
    """Squared distance from p (..,3) to segment ab, broadcasting."""
    ab = b - a
    t = np.einsum("...i,...i->...", p - a, ab)
    denom = np.einsum("...i,...i->...", ab, ab)
    t = np.where(denom < 1e-12, 0.0, t / np.maximum(denom, 1e-12))
    t = np.clip(t, 0.0, 1.0)[..., None]
    d = p - (a + t * ab)
    return np.einsum("...i,...i->...", d, d)


def _inside_and_perp(points, tris):
    """(inside, perp_d2) for every (point, triangle) pair. Planar z=0 tris."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    p = points[:, None, :]
    e0 = (v1 - v0)[None]
    e1 = (v2 - v0)[None]
    e2 = p - v0[None]
    d00 = np.einsum("...i,...i->...", e0, e0)
    d01 = np.einsum("...i,...i->...", e0, e1)
    d11 = np.einsum("...i,...i->...", e1, e1)
    d20 = np.einsum("...i,...i->...", e2, e0)
    d21 = np.einsum("...i,...i->...", e2, e1)
    denom = d00 * d11 - d01 * d01 + 1e-12
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2
    inside = (w0 >= 0) & (w0 <= 1) & (w1 >= 0) & (w1 <= 1) & (w2 >= 0) & (w2 <= 1)
    perp = p - (v0[None] + w1[..., None] * e0 + w2[..., None] * e1)
    return inside, np.einsum("...i,...i->...", perp, perp)


def _edge_d2(points, tris):
    p = points[:, None, :]
    v0, v1, v2 = tris[None, :, 0], tris[None, :, 1], tris[None, :, 2]
    return np.minimum(np.minimum(_seg_d2(p, v0, v1), _seg_d2(p, v0, v2)),
                      _seg_d2(p, v1, v2))


def patch_distance(points, tris, min_patch_area=MIN_PATCH_AREA):
    """The contract. Small triangles are their three boundary segments."""
    inside, perp = _inside_and_perp(points, tris)
    big = (_areas(tris) >= min_patch_area)[None, :]
    d2 = np.where(inside & big, perp, _edge_d2(points, tris))
    idx = np.argmin(d2, axis=1)
    return d2[np.arange(len(points)), idx], idx


def nearest_patch_fast(points, tris):
    """What a generic accelerated closest-point primitive returns.

    Ordinary point-triangle distance. No `min_patch_area` notion at all.
    """
    inside, perp = _inside_and_perp(points, tris)
    d2 = np.where(inside, perp, _edge_d2(points, tris))
    idx = np.argmin(d2, axis=1)
    return d2[np.arange(len(points)), idx], idx


def admissible_candidates(points, tris, min_patch_area=MIN_PATCH_AREA):
    """How many triangles an accelerated structure must still examine.

    d_contract >= d_true for every pair, so a structure that prunes on the
    true distance stays admissible - but it can only prune to the set
    {t : d_true(p,t) < best_contract(p)}. This returns |that set| per point,
    which is the work a contract-faithful accelerated path cannot avoid.
    """
    best, _ = patch_distance(points, tris, min_patch_area)
    inside, perp = _inside_and_perp(points, tris)
    d_true = np.where(inside, perp, _edge_d2(points, tris))
    return (d_true < best[:, None]).sum(axis=1)


def _timeit(fn, reps=2):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def report():
    pts = sample_points(2500, seed=1)
    print(f"min_patch_area = {MIN_PATCH_AREA}   (an ABSOLUTE area threshold)\n")
    for tag, n in (("SHIPPED asset", SHIPPED_N), ("toy asset", TOY_N)):
        tris = make_surface(n)
        ar = _areas(tris)
        below = float((ar < MIN_PATCH_AREA).mean() * 100)
        dc, ic = patch_distance(pts, tris)
        df, if_ = nearest_patch_fast(pts, tris)
        diff = np.abs(dc - df)
        pct = float((diff > 1e-9).mean() * 100)
        rel = float((diff / np.maximum(dc, 1e-12)).max())
        cand = admissible_candidates(pts, tris)
        print(f"{tag}: n={n}, {len(tris)} triangles, per-face area {ar[0]:.3e}")
        print(f"  faces below min_patch_area       : {below:.1f}%")
        print(f"  points where fast != contract    : {pct:.1f}%")
        print(f"    magnitude of that disagreement : max abs {diff.max():.3e}, "
              f"max rel {rel:.3f}")
        print(f"  closest-face index differs       : "
              f"{float((ic != if_).mean()*100):.1f}%")
        print(f"  candidates a contract-faithful accelerated path must still "
              f"examine: mean {cand.mean():.1f}, max {cand.max()} of {len(tris)}")
        tf = _timeit(lambda: nearest_patch_fast(pts, tris))
        tc = _timeit(lambda: patch_distance(pts, tris))
        print(f"  measured cost of fidelity        : {tc/tf:.2f}x "
              f"({tc*1e3:.0f} ms vs {tf*1e3:.0f} ms)\n")
    print("Note the shape of the disagreement on the shipped asset: FREQUENT "
          "but small in\nabsolute terms. A tolerance chosen on absolute error "
          "alone will hide it; the\nrelative error reaches 1.0 for points "
          "lying on the surface, where the contract\nreturns the distance to "
          "the nearest edge and the fast primitive returns ~0.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n", type=int, default=SHIPPED_N)
    ap.add_argument("--points", type=int, default=4000)
    a = ap.parse_args()
    if a.report:
        report()
    else:
        tris = make_surface(a.n)
        pts = sample_points(a.points)
        d, i = patch_distance(pts, tris)
        print(f"n={a.n} tris={len(tris)} points={a.points} "
              f"mean_d2={d.mean():.6e}")
