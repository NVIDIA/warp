# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""kdlite — radius queries against a static point set.

The public entry point is :func:`query`. Every call validates its inputs,
converts them to the internal layout, and allocates the output before the
search itself runs.  :func:`query_prepared` skips that preamble for callers who
already hold prepared arrays — it exists for the batch path and is not what the
public API uses.

Usage:
    python kdlite.py --sweep          # per-call cost vs search cost, across sizes
    python kdlite.py --queries 20000
"""

import argparse
import time

import numpy as np

CELL = 0.05


class Index:
    def __init__(self, points, seed=0):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        keys = np.floor(self.points / CELL).astype(np.int64)
        self.n_cells = int(np.ceil(1.0 / CELL))
        flat = np.clip(keys[:, 1], 0, self.n_cells - 1) * self.n_cells + \
            np.clip(keys[:, 0], 0, self.n_cells - 1)
        order = np.argsort(flat, kind="stable")
        self.sorted = self.points[order]
        self.start = np.searchsorted(flat[order], np.arange(self.n_cells ** 2 + 1))

    def _cell_slice(self, cx, cy):
        c = cy * self.n_cells + cx
        return self.sorted[self.start[c]:self.start[c + 1]]

    # -- the search itself: cell-local, only the 3x3 neighbourhood ---------
    def _search(self, q, radius, out):
        r2 = radius * radius
        n = self.n_cells
        for i in range(q.shape[0]):
            cx = min(n - 1, max(0, int(q[i, 0] / CELL)))
            cy = min(n - 1, max(0, int(q[i, 1] / CELL)))
            total = 0
            for gy in range(max(0, cy - 1), min(n, cy + 2)):
                block = self.sorted[self.start[gy * n + max(0, cx - 1)]:
                                    self.start[gy * n + min(n, cx + 2)]]
                if block.size:
                    dx = block[:, 0] - q[i, 0]
                    dy = block[:, 1] - q[i, 1]
                    total += int(np.count_nonzero(dx * dx + dy * dy <= r2))
            out[i] = total
        return out

    # -- the public call --------------------------------------------------
    def query(self, q, radius=0.02):
        """Validate, convert, allocate, then search."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        q = np.asarray(q, dtype=np.float64)          # copies when the caller passes lists
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("queries must be (n, 2)")
        if not np.isfinite(q).all():                 # full pass over the query set
            raise ValueError("queries must be finite")
        if not q.flags["C_CONTIGUOUS"]:
            q = np.ascontiguousarray(q)
        _ = self.points.min(axis=0), self.points.max(axis=0)   # bounds recomputed over the
        #                                                      whole index on every call
        out = np.zeros(q.shape[0], dtype=np.int64)
        return self._search(q, radius, out)

    def query_prepared(self, q, radius, out):
        """No validation, no conversion, no allocation — caller owns all three."""
        return self._search(q, radius, out)


def best_of(fn, reps=3):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=2_000_000)
    ap.add_argument("--queries", type=int, default=2_000)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    index = Index(rng.random((args.points, 2)))

    sizes = [1, 8, 64, 512, args.queries] if args.sweep else [args.queries]
    print(f"{'queries':>9}{'public call':>14}{'prepared':>12}{'wrapper':>11}"
          f"{'wrapper share':>15}")
    for n in sizes:
        q = rng.random((n, 2))
        out = np.zeros(n, dtype=np.int64)
        t_public = best_of(lambda: index.query(q, args.radius))
        t_prepared = best_of(lambda: index.query_prepared(q, args.radius, out))
        wrapper = t_public - t_prepared
        print(f"{n:>9}{t_public*1e3:>13.3f}m{t_prepared*1e3:>11.3f}m"
              f"{wrapper*1e3:>10.3f}m{100*wrapper/t_public:>14.1f}%")


if __name__ == "__main__":
    main()
