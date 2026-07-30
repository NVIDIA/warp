# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""opskit's own per-operation benchmark.

Each operation is timed once, at one representative scale, and the results are
ranked. This is what `make bench` runs and what the release notes quote.

    python bench_ops.py

Only numpy is required.
"""
import time

import numpy as np

import opskit

REPEAT = 2


def _time(fn):
    best = float('inf')
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)

    pos = rng.random((2048, 3))
    edges = rng.integers(0, 2048, (300_000,))
    edge_feat = rng.normal(size=(300_000, 32))
    basis = rng.random((300_000, 8))
    weight = rng.normal(size=(8, 32, 32))
    dense_x = rng.normal(size=(2048, 128))
    dense_w = rng.normal(size=(128, 128))

    cases = [
        ('op_spline', lambda: opskit.op_spline(edge_feat, basis, weight)),
        ('op_knn', lambda: opskit.op_knn(pos, 16)),
        ('op_scatter', lambda: opskit.op_scatter(edge_feat, edges, 2048)),
        ('op_radius', lambda: opskit.op_radius(pos, 0.05)),
        ('op_dense', lambda: opskit.op_dense(dense_x, dense_w)),
    ]

    results = [(name, _time(fn)) for name, fn in cases]
    total = sum(t for _, t in results)

    print('opskit operation benchmark\n')
    print(f"{'operation':<14} {'ms':>10} {'share':>8}")
    for name, t in sorted(results, key=lambda kv: -kv[1]):
        print(f'{name:<14} {t * 1e3:>10.1f} {100 * t / total:>7.1f}%')
    print(f"\n{'total':<14} {total * 1e3:>10.1f} ms")
    print('\nscales: op_knn 2048 pts D=3 k=16; op_radius 2048 pts r=0.05; '
          'op_spline/op_scatter 300k edges C=32; op_dense 2048x128x128')


if __name__ == '__main__':
    main()
