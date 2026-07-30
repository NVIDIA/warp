# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""knnkit - k-nearest-neighbour edge lists, two shipped backends.

`knn_reference` is the portable path. It initialises its running best distance
to infinity.

`knn_fast` is the accelerated path that ships for the supported devices. It
keeps a fixed-size best-distance array initialised to a sentinel constant and
only ever inserts a candidate that beats the current worst slot; slots that
were never filled stay at -1 and are dropped from the returned edge list.

Both are used through the same public function, `knn`, which picks a backend.

Commands:

    python knnkit.py scale-sweep   # edges returned vs coordinate scale
    python knnkit.py degrees       # per-query neighbour counts at one scale
    python knnkit.py selftest      # the checks that ship with the library

Only numpy is required.
"""
import argparse

import numpy as np

N_POINTS = 64
K = 4

# Initial value of the running best-distance array in the accelerated path.
SENTINEL = 1e10


def cloud(n=N_POINTS, scale=1.0, seed=11):
    """A compact cloud, then uniformly rescaled. Shape is scale-invariant."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * scale


def knn_reference(x, k=K):
    """Portable path: best distance starts at infinity."""
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1, kind='stable')[:, :k]
    row = np.repeat(np.arange(len(x)), k)
    return np.stack([row, idx.reshape(-1)])


def knn_fast(x, k=K):
    """Accelerated path: best distance starts at SENTINEL.

    A candidate is inserted only when it beats the current worst slot, so a
    candidate whose squared distance is >= SENTINEL can never be inserted at
    all.  Unfilled slots keep their -1 marker and are then masked out of the
    returned edge list.
    """
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    reachable = d2 < SENTINEL          # everything else can never be inserted
    d2_masked = np.where(reachable, d2, np.inf)
    idx = np.argsort(d2_masked, axis=1, kind='stable')[:, :k]
    filled = np.take_along_axis(reachable, idx, axis=1)
    idx = np.where(filled, idx, -1)
    row = np.repeat(np.arange(len(x)), k)
    col = idx.reshape(-1)
    keep = col >= 0
    return np.stack([row[keep], col[keep]])


def knn(x, k=K, backend='fast'):
    return (knn_fast if backend == 'fast' else knn_reference)(x, k)


# --------------------------------------------------------------------------
def cmd_scale_sweep():
    print(f'n={N_POINTS}, k={K}; every scale is the same cloud, '
          f'uniformly rescaled, so the correct answer is scale-invariant\n')
    print(f"{'scale':>10} {'max sq dist':>14} {'kth sq dist (max)':>19} "
          f"{'reference':>10} {'fast':>7} {'expected':>9}")
    for s in (1e0, 1e2, 1e4, 3e4, 1e5, 3e5, 1e6, 1e9):
        x = cloud(scale=s)
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        kth = np.sort(d2, axis=1)[:, K - 1].max()
        ref = knn_reference(x).shape[1]
        fast = knn_fast(x).shape[1]
        flag = '' if fast == N_POINTS * K else '   <-- short'
        print(f'{s:>10.0e} {d2.max():>14.4g} {kth:>19.4g} '
              f'{ref:>10} {fast:>7} {N_POINTS * K:>9}{flag}')
    print(f'\nsentinel = {SENTINEL:.4g}; a candidate is only reachable while '
          f'its squared distance stays under it')


def cmd_degrees():
    scale = 1e5
    x = cloud(scale=scale)
    fast = knn_fast(x)
    deg = np.bincount(fast[0], minlength=len(x))
    print(f'scale={scale:.0e}, k={K}')
    print(f'  queries with a full {K} neighbours : {(deg == K).sum()}')
    print(f'  queries with fewer               : {(deg < K).sum()}')
    print(f'  queries with none                : {(deg == 0).sum()}')
    print(f'  min / max degree returned        : {deg.min()} / {deg.max()}')
    print('  reference returns exactly k for every query:',
          bool((np.bincount(knn_reference(x)[0], minlength=len(x))
                == K).all()))


def cmd_selftest():
    """The checks that ship with the library."""
    checks = []
    x = cloud(scale=1.0)
    a, b = knn_fast(x), knn_reference(x)
    checks.append(('fast matches reference on the standard cloud',
                   np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1))))
    checks.append(('edge list is query-major',
                   bool((np.diff(a[0]) >= 0).all())))
    checks.append(('every reference index is in range',
                   bool(((a[1] >= 0) & (a[1] < len(x))).all())))
    small = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
                      [4, 0, 0]])
    checks.append(('nearest neighbour of point 0 is point 1',
                   knn_fast(small, k=2)[1][1] == 1))
    checks.append(('k larger than n is clamped',
                   knn_fast(small, k=5).shape[1] == 25))
    for name, ok in checks:
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
    print(f'\n{sum(1 for _, ok in checks if ok)}/{len(checks)} passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command',
                    choices=['scale-sweep', 'degrees', 'selftest'])
    args = ap.parse_args()
    {'scale-sweep': cmd_scale_sweep, 'degrees': cmd_degrees,
     'selftest': cmd_selftest}[args.command]()
