# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""bidist - the public shape-matching score.

`shape_score(A, B)` is the number every caller reads. It is the sum of TWO
independent brute-force sweeps:

    forward(A -> B)   for each a in A, the closest b in B
    reverse(B -> A)   for each b in B, the closest a in A

Both are O(|A| * |B|). An acceleration structure built over B answers `forward`
and does nothing at all for `reverse`, which needs a structure over A.

`forward_indexed()` is a uniform-grid accelerated forward sweep, to show what
accelerating one half actually buys at the public entry point.

Run `python bidist.py --report`.
"""

import argparse
import time

import numpy as np


def make_clouds(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.random(n) * 2 * np.pi
    a = np.stack([np.cos(t), np.sin(t)], 1) + rng.normal(0, 0.02, (n, 2))
    t2 = rng.random(n) * 2 * np.pi
    b = 1.05 * np.stack([np.cos(t2), np.sin(t2)], 1) + rng.normal(0, 0.02, (n, 2))
    return a, b


def _brute_min(P, Q, chunk=512):
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        d = ((P[s:s + chunk, None, :] - Q[None, :, :]) ** 2).sum(-1)
        out[s:s + chunk] = d.min(1)
    return out


def forward(A, B):
    return _brute_min(A, B)


def reverse(B, A):
    return _brute_min(B, A)


def shape_score(A, B):
    """The public entry point."""
    return float(forward(A, B).mean() + reverse(B, A).mean())


def forward_indexed(A, B, cell=0.05):
    """Uniform-grid accelerated forward sweep. EXACT: returns the same values
    as forward().

    Rings are expanded until the next ring cannot contain anything closer -
    any point in ring r+1 is at least r*cell away - so no candidate is lost.
    """
    from collections import defaultdict

    lo = np.minimum(A.min(0), B.min(0)) - cell
    gb = np.floor((B - lo) / cell).astype(np.int64)
    buckets = defaultdict(list)
    for i, (x, y) in enumerate(gb):
        buckets[(int(x), int(y))].append(i)
    buckets = {k: np.asarray(v) for k, v in buckets.items()}

    ga = np.floor((A - lo) / cell).astype(np.int64)
    out = np.empty(len(A))
    for i in range(len(A)):
        cx, cy = int(ga[i, 0]), int(ga[i, 1])
        best = np.inf
        r = 0
        while True:
            cand = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    b = buckets.get((cx + dx, cy + dy))
                    if b is not None:
                        cand.append(b)
            if cand:
                idx = np.concatenate(cand)
                d = ((A[i] - B[idx]) ** 2).sum(-1).min()
                if d < best:
                    best = d
            if np.isfinite(best) and r * cell >= np.sqrt(best):
                break
            r += 1
            if r > 400:
                break
        out[i] = best
    return out


def _t(fn, reps=1):
    fn()
    s = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - s) / reps


def report():
    A, B = make_clouds()
    print(f"|A| = {len(A)}, |B| = {len(B)}\n")
    tf = _t(lambda: forward(A, B))
    tr = _t(lambda: reverse(B, A))
    tp = _t(lambda: shape_score(A, B))
    print(f"forward(A->B)          : {tf*1e3:8.1f} ms   "
          f"{100*tf/(tf+tr):5.1f}% of the two sweeps")
    print(f"reverse(B->A)          : {tr*1e3:8.1f} ms   "
          f"{100*tr/(tf+tr):5.1f}%")
    print(f"shape_score (public)   : {tp*1e3:8.1f} ms\n")

    ti = _t(lambda: forward_indexed(A, B))
    exact = np.allclose(forward(A, B), forward_indexed(A, B))
    print(f"forward_indexed        : {ti*1e3:8.1f} ms   "
          f"({tf/ti:.1f}x faster than forward, values identical: {exact})")
    hybrid = ti + tr
    print(f"\nIf ONLY forward is accelerated:")
    print(f"  seam speedup quoted in isolation : {tf/ti:6.1f}x")
    print(f"  public shape_score speedup       : {tp/hybrid:6.2f}x   "
          f"<- what the caller actually gets")
    print(f"  ceiling from the 50/50 split     : {(tf+tr)/tr:6.2f}x   "
          f"(reverse is untouched)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        A, B = make_clouds()
        print(f"shape_score = {shape_score(A, B):.6f}")
