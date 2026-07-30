# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cellassign — assign every sample to its best-matching template.

For each sample point, pick the template minimising

    d(sample, template) = |w0| + |w1| + |w2|

where the w are the sample's coordinates in that template's local frame,
obtained by dividing through the template's signed area.  Ties resolve to the
lowest template index.

Usage:
    python assign.py            # assign the bundled samples
    python assign.py --dump     # also write assignments.npy
"""

import argparse

import numpy as np

TEMPLATES_PER_CHUNK = 10


def local_coords(templates, points):
    """(T,3,2) templates x (N,2) points -> (T,N) coordinates, three per point."""
    v0, v1, v2 = templates[:, 0], templates[:, 1], templates[:, 2]

    def para(p, a, b):
        return (p[None, :, 0] - a[:, None, 0]) * (b[:, None, 1] - a[:, None, 1]) - (
            p[None, :, 1] - a[:, None, 1]
        ) * (b[:, None, 0] - a[:, None, 0])

    area = (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]) - (v2[:, 1] - v0[:, 1]) * (v1[:, 0] - v0[:, 0])
    w0 = para(points, v1, v2) / area[:, None]
    w1 = para(points, v2, v0) / area[:, None]
    w2 = para(points, v0, v1) / area[:, None]
    return w0, w1, w2


def assign(templates, points, templates_per_chunk=TEMPLATES_PER_CHUNK):
    """Chunked selection over the template list."""
    n = points.shape[0]
    num_templates = templates.shape[0]
    best_d = np.full(n, np.finfo(np.float32).max, dtype=np.float32)
    best_i = np.zeros(n, dtype=np.int64)

    for c in range(num_templates // templates_per_chunk):
        s = c * templates_per_chunk
        e = min((c + 1) * templates_per_chunk, num_templates)
        w0, w1, w2 = local_coords(templates[s:e], points)
        d = np.abs(w0) + np.abs(w1) + np.abs(w2)
        d_values = d.min(axis=0)
        d_indices = d.argmin(axis=0)
        condition = d_values < best_d
        best_d = np.where(condition, d_values, best_d)
        best_i = np.where(condition, d_indices + s, best_i)
    return best_i, best_d


def load_case(seed=0, num_templates=404, num_points=20000):
    rng = np.random.default_rng(seed)
    centres = rng.random((num_templates, 1, 2)).astype(np.float32)
    offsets = rng.normal(0.0, 0.05, size=(num_templates, 3, 2)).astype(np.float32)
    templates = centres + offsets
    # a few templates in this library are exactly flat
    for flat in (7, 118, 254):
        templates[flat] = templates[flat][0]
    points = rng.random((num_points, 2)).astype(np.float32)
    return templates, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    templates, points = load_case()
    idx, dist = assign(templates, points)
    print(f"assigned {len(idx)} points over {len(templates)} templates")
    print(f"distinct templates used: {len(np.unique(idx))}")
    print(f"mean distance: {np.nanmean(dist):.6f}")
    if args.dump:
        np.save("assignments.npy", idx)
        print("wrote assignments.npy")


if __name__ == "__main__":
    main()
