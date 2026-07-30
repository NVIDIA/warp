# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A shipped opskit model: dynamic-graph point cloud classifier.

This is the reference consumer in `examples/`, and it is what the users who
file performance issues are actually running.

The graph is rebuilt at every layer: once from the point positions, then twice
more from the learned features, which is what makes it "dynamic".

    python train_step.py --profile     # per-call split for one step
    python train_step.py --shapes      # what each op is called with

Only numpy is required.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import opskit  # noqa: E402

N_POINTS = 2048
K = 6
C = 128
RADIUS = 0.05

CALLS = []


def _timed(name, detail, fn):
    t0 = time.perf_counter()
    out = fn()
    CALLS.append((name, detail, time.perf_counter() - t0))
    return out


def edge_conv(x, k, w, tag):
    """One dynamic-graph layer: rebuild the graph, then gather-transform-max."""
    idx = _timed('op_knn', f'{tag}: n={len(x)} D={x.shape[1]} k={k}',
                 lambda: opskit.op_knn(x, k))
    row = np.repeat(np.arange(len(x)), k)
    col = idx.reshape(-1)
    msg = _timed('op_dense', f'{tag}: {len(row)}x{2 * x.shape[1]}',
                 lambda: opskit.op_dense(
                     np.concatenate([x[row], x[col] - x[row]], axis=1), w))
    return _timed('op_scatter', f'{tag}: {len(row)} edges -> {len(x)}',
                  lambda: opskit.op_scatter(msg, row, len(x)))


def step(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.random((N_POINTS, 3))
    w1 = rng.normal(size=(6, C)) * 0.1
    w2 = rng.normal(size=(2 * C, C)) * 0.1
    w3 = rng.normal(size=(2 * C, C)) * 0.1
    head = rng.normal(size=(C, 40)) * 0.1

    # a local-neighbourhood feature, computed once from the positions
    _timed('op_radius', f'n={N_POINTS} r={RADIUS}',
           lambda: opskit.op_radius(pos, RADIUS))

    x = edge_conv(pos, K, w1, 'layer1 (positions)')
    x = edge_conv(x, K, w2, 'layer2 (features)')
    x = edge_conv(x, K, w3, 'layer3 (features)')
    return _timed('op_dense', f'head: {N_POINTS}x{C}',
                  lambda: opskit.op_dense(x, head))


def cmd_profile():
    CALLS.clear()
    step()
    CALLS.clear()          # discard the warm-up
    step()
    total = sum(t for _, _, t in CALLS)

    print(f'one training step: {N_POINTS} points, k={K}, {C} channels\n')
    print(f"{'op':<12} {'where':<34} {'ms':>9} {'share':>8}")
    for name, detail, t in CALLS:
        print(f'{name:<12} {detail:<34} {t * 1e3:>9.1f} '
              f'{100 * t / total:>7.1f}%')

    per_op = {}
    for name, _, t in CALLS:
        per_op[name] = per_op.get(name, 0.0) + t
    print(f'\n{"per operation":<12} {"":<34} {"ms":>9} {"share":>8}')
    for name, t in sorted(per_op.items(), key=lambda kv: -kv[1]):
        print(f'{name:<12} {"":<34} {t * 1e3:>9.1f} {100 * t / total:>7.1f}%')
    print(f'\n{"total":<12} {"":<34} {total * 1e3:>9.1f} ms')

    knn = [(d, t) for n, d, t in CALLS if n == 'op_knn']
    knn_total = sum(t for _, t in knn)
    d3 = sum(t for d, t in knn if 'D=3' in d)
    print(f'\nop_knn breakdown: {knn_total * 1e3:.1f} ms total; '
          f'the D=3 call is {d3 * 1e3:.1f} ms '
          f'({100 * d3 / knn_total:.1f}% of op_knn, '
          f'{100 * d3 / total:.1f}% of the step)')


def cmd_shapes():
    CALLS.clear()
    step()
    print(f"{'op':<12} {'call':<40}")
    for name, detail, _ in CALLS:
        print(f'{name:<12} {detail:<40}')
    print('\nop_spline is not called by this model.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--shapes', action='store_true')
    a = ap.parse_args()
    cmd_shapes() if a.shapes else cmd_profile()
