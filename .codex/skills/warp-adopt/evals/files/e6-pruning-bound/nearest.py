# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""shapenear — find, for each query, the shape with the smallest scaled distance.

The metric is scale-normalised: a shape's distance is the Euclidean distance
from the query to the shape centre divided by the shape's own radius, so a
large shape "reaches" further than a small one.

    score(q, s) = |q - centre_s| / radius_s

`brute()` is the reference implementation.  `pruned()` is our optimised
version: it first computes a cheap conservative bound from each shape's
bounding box, then evaluates the real metric only for the most promising
candidates.

Usage:
    python nearest.py
"""

import argparse
import time

import numpy as np

CANDIDATES = 8


def make_case(seed=0, n_shapes=4000, n_queries=6000):
    rng = np.random.default_rng(seed)
    centres = rng.random((n_shapes, 2)).astype(np.float32)
    radii = (0.002 + rng.random(n_shapes) * 0.03).astype(np.float32)
    queries = rng.random((n_queries, 2)).astype(np.float32)
    return centres, radii, queries


def brute(centres, radii, queries, block=1024):
    out = np.zeros(len(queries), dtype=np.int64)
    for s in range(0, len(queries), block):
        q = queries[s : s + block]
        d = np.sqrt(((q[:, None, :] - centres[None, :, :]) ** 2).sum(-1))
        out[s : s + block] = (d / radii[None, :]).argmin(axis=1)
    return out


def pruned(centres, radii, queries, block=1024, candidates=CANDIDATES):
    """Bound-first selection: rank shapes by a conservative box bound, then
    evaluate the true metric on the best `candidates` only."""
    lo = centres - radii[:, None]
    hi = centres + radii[:, None]

    out = np.zeros(len(queries), dtype=np.int64)
    for s in range(0, len(queries), block):
        q = queries[s : s + block]
        dx = np.maximum(np.maximum(lo[None, :, 0] - q[:, None, 0], q[:, None, 0] - hi[None, :, 0]), 0.0)
        dy = np.maximum(np.maximum(lo[None, :, 1] - q[:, None, 1], q[:, None, 1] - hi[None, :, 1]), 0.0)
        bound = np.sqrt(dx * dx + dy * dy) / radii[None, :]

        keep = np.argpartition(bound, candidates, axis=1)[:, :candidates]
        cc = centres[keep]
        rr = radii[keep]
        d = np.sqrt(((q[:, None, :] - cc) ** 2).sum(-1)) / rr
        out[s : s + block] = keep[np.arange(len(q)), d.argmin(axis=1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=6000)
    args = ap.parse_args()
    centres, radii, queries = make_case(n_queries=args.queries)

    t0 = time.perf_counter()
    a = brute(centres, radii, queries)
    t_brute = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = pruned(centres, radii, queries)
    t_pruned = time.perf_counter() - t0

    print(f"brute  {t_brute:8.3f} s")
    print(f"pruned {t_pruned:8.3f} s   ({t_brute / max(t_pruned, 1e-9):.1f}x)")
    print(f"assignments returned: {len(a)} / {len(b)}")


if __name__ == "__main__":
    main()
