# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""opskit - low-level operations for graph and point-cloud models.

This is a library. It has no application of its own: every entry point here is
called by downstream model code. `bench_ops.py` is the per-operation benchmark
that ships with it; `examples/train_step.py` is a shipped model that uses it.

Operations:

    op_knn      k-nearest neighbours, any feature dimension
    op_radius   fixed-capacity radius neighbour list
    op_spline   per-edge spline-weighted feature transform
    op_scatter  segmented max-reduction over an edge list
    op_dense    dense linear layer

Only numpy is required.
"""
import numpy as np


def op_knn(x, k, ptr=None):
    """k nearest neighbours of every row of `x`, among the rows of `x`.

    Works in whatever dimension `x` has: 3 for positions, hundreds for
    learned features. Cost is O(n^2 * (D + k)) either way.
    """
    ptr = np.array([0, len(x)]) if ptr is None else ptr
    out = np.empty((len(x), k), dtype=np.int64)
    for b in range(len(ptr) - 1):
        lo, hi = int(ptr[b]), int(ptr[b + 1])
        p = x[lo:hi]
        x2 = (p * p).sum(1)
        d2 = x2[:, None] + x2[None, :] - 2.0 * (p @ p.T)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(d2, part, axis=1)
        out[lo:hi] = np.take_along_axis(
            part, np.argsort(vals, axis=1, kind='stable'), axis=1) + lo
    return out


def op_radius(pos, r, max_neighbors=32):
    """Every point within `r`, capped at `max_neighbors`, in index order."""
    n = len(pos)
    rows, cols = [], []
    chunk = 1024
    for s in range(0, n, chunk):
        q = pos[s:s + chunk]
        d2 = ((q * q).sum(1)[:, None] + (pos * pos).sum(1)[None, :]
              - 2.0 * (q @ pos.T))
        m = d2 < r * r
        rr, cc = np.nonzero(m)
        first = np.searchsorted(rr, np.arange(len(q)))
        rank = np.arange(len(rr)) - first[rr]
        keep = rank < max_neighbors
        rows.append(rr[keep] + s)
        cols.append(cc[keep])
    return np.stack([np.concatenate(rows), np.concatenate(cols)])


def op_spline(edge_feat, basis, weight):
    """Per-edge spline-weighted transform: [E, C] x [E, B] x [B, C, C]."""
    acc = np.zeros((len(edge_feat), weight.shape[2]))
    for b in range(basis.shape[1]):
        acc += (edge_feat @ weight[b]) * basis[:, b:b + 1]
    return acc


def op_scatter(src, index, n_out):
    """Segmented max-reduction of `src` rows into `n_out` slots."""
    out = np.full((n_out, src.shape[1]), -np.inf)
    np.maximum.at(out, index, src)
    return np.where(np.isfinite(out), out, 0.0)


def op_dense(x, w):
    return np.maximum(x @ w, 0.0)
