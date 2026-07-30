# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nnkit - the project's maintained neighbour-count backend.

`count_within()` is the accelerated backend the project ships. It is not a
prototype: it is the fastest implementation in the codebase, it is documented,
it has a test suite (`test_nnkit.py`), it has been maintained across four
releases, and it produces exactly the documented contract including the
`<=` boundary rule and the deterministic ordering of `neighbours_of()`.

It is also O(N*M): it forms every pair. That is a property of the ALGORITHM,
not of the backend it is written in.

`grid_count_within()` is a plain, unoptimised implementation of a better
algorithm, written in the same language, running on the same hardware, using no
new dependency. It exists here only to size the algorithmic gap.

Run `python nnkit.py --report`.
"""

import argparse
import time

import numpy as np

__version__ = "4.2.0"
CONTRACT_RADIUS_INCLUSIVE = True     # points at exactly `radius` are counted


def make_points(n, seed=0, clusters=12):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 10, size=(clusters, 2))
    which = rng.integers(0, clusters, size=n)
    return centres[which] + rng.normal(0, 0.35, size=(n, 2))


def count_within(P, Q, radius, chunk=256):
    """The maintained backend. For each p in P, how many q in Q lie within
    `radius` (inclusive). Chunked to bound peak memory."""
    r2 = radius * radius
    out = np.empty(len(P), dtype=np.int64)
    for s in range(0, len(P), chunk):
        d = ((P[s:s + chunk, None, :] - Q[None, :, :]) ** 2).sum(-1)
        out[s:s + chunk] = (d <= r2).sum(1)
    return out


def grid_count_within(P, Q, radius):
    """Same contract, better algorithm: bucket Q into cells of side `radius`
    and visit only the 3x3 cell neighbourhood. No new dependency, same
    language, same hardware."""
    from collections import defaultdict

    r2 = radius * radius
    lo = np.minimum(P.min(0), Q.min(0)) - radius
    gq = np.floor((Q - lo) / radius).astype(np.int64)
    buckets = defaultdict(list)
    for i, (x, y) in enumerate(gq):
        buckets[(int(x), int(y))].append(i)
    buckets = {k: np.asarray(v) for k, v in buckets.items()}

    gp = np.floor((P - lo) / radius).astype(np.int64)
    out = np.zeros(len(P), dtype=np.int64)
    for i in range(len(P)):
        cx, cy = int(gp[i, 0]), int(gp[i, 1])
        total = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                b = buckets.get((cx + dx, cy + dy))
                if b is None:
                    continue
                d = ((P[i] - Q[b]) ** 2).sum(-1)
                total += int((d <= r2).sum())
        out[i] = total
    return out


def _t(fn, reps=1):
    fn()
    s = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - s) / reps


def report():
    radius = 0.5
    print(f"nnkit {__version__}   radius={radius}   "
          f"(maintained backend, 4 releases, tested)\n")
    print(f"  {'N=M':>8} {'backend ms':>12} {'grid ms':>10} {'ratio':>8}  agree")
    for n in (2000, 8000, 20000):
        P = make_points(n, seed=1)
        Q = make_points(n, seed=2)
        tb = _t(lambda: count_within(P, Q, radius))
        tg = _t(lambda: grid_count_within(P, Q, radius))
        agree = np.array_equal(count_within(P, Q, radius),
                               grid_count_within(P, Q, radius))
        print(f"  {n:8d} {tb*1e3:12.1f} {tg*1e3:10.1f} {tb/tg:7.1f}x  {agree}")

    print("\n  scaling check: the backend's cost should track N*M if it forms "
          "every pair.")
    ns, ts = [], []
    for n in (2000, 4000, 8000):
        P, Q = make_points(n, seed=1), make_points(n, seed=2)
        ns.append(n)
        ts.append(_t(lambda: count_within(P, Q, radius)))
    for i in range(1, len(ns)):
        print(f"    N {ns[i-1]} -> {ns[i]} ({(ns[i]/ns[i-1])**2:.0f}x the "
              f"pairs): {ts[i]/ts[i-1]:.1f}x the time")

    print("\n  The backend is correct, maintained and the fastest thing in the "
          "project.\n  It is still the wrong algorithm, and the replacement "
          "above needs no new\n  dependency, no new language and no new "
          "hardware.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        P, Q = make_points(4000, seed=1), make_points(4000, seed=2)
        print(f"mean neighbours = {count_within(P, Q, 0.5).mean():.3f}")
