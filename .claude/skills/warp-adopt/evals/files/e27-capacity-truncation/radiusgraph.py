# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""radiusgraph - fixed-capacity radius neighbour lists.

`radius_scan` and `radius_blocks` are the two backends that ship today. Both
answer the same question - every point within `r` - and both stop once they
have `max_neighbors` of them. They differ only in the order they enumerate
candidates, so when a query has more than `max_neighbors` neighbours they keep
*different* ones.

`radius_cells` is a candidate that enumerates in cell order, so it keeps a
third subset.

The public API returns only the edge list. It does not return the true
neighbour count, and it has no overflow flag, so a caller cannot tell a query
that was exactly full from one that was truncated.

Commands:

    python radiusgraph.py capacity   # how often the cap actually binds
    python radiusgraph.py agree      # pairwise edge-set overlap + validity
    python radiusgraph.py overflow   # exactly-full vs truncated
    python radiusgraph.py sweep      # capacity pressure vs r

Only numpy is required.
"""
import argparse

import numpy as np

N_POINTS = 3_000
R_DEFAULT = 0.10          # the radius the shipped config uses
MAX_NEIGHBORS = 32        # the shipped cap


def cloud(n=N_POINTS, seed=5):
    """A clumpy cloud: most points sit in dense clusters, some in the field.

    Real scenes look like this, and it is what makes a fixed cap bind.
    """
    rng = np.random.default_rng(seed)
    n_clust = int(n * 0.8)
    centres = rng.random((24, 3))
    pick = rng.integers(0, len(centres), n_clust)
    clustered = centres[pick] + rng.normal(scale=0.035, size=(n_clust, 3))
    field = rng.random((n - n_clust, 3))
    return np.clip(np.concatenate([clustered, field]), 0.0, 1.0)


def _within(pos, r):
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    return d2 < r * r


def _collect(mask, order, max_nn):
    """Keep the first `max_nn` neighbours in the given enumeration order."""
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    rows, cols = [], []
    for i in range(mask.shape[0]):
        idx = np.flatnonzero(mask[i])
        idx = idx[np.argsort(rank[idx], kind='stable')][:max_nn]
        rows.append(np.full(len(idx), i, dtype=np.int64))
        cols.append(idx)
    return np.stack([np.concatenate(rows), np.concatenate(cols)])


def radius_scan(pos, r=R_DEFAULT, max_nn=MAX_NEIGHBORS):
    """Shipped backend A: ascending point index."""
    return _collect(_within(pos, r), np.arange(len(pos)), max_nn)


def radius_blocks(pos, r=R_DEFAULT, max_nn=MAX_NEIGHBORS):
    """Shipped backend B: tiled enumeration, so a different subset survives."""
    n = len(pos)
    tile = 64
    order = np.concatenate([np.arange(s, n, tile) for s in range(tile)])
    return _collect(_within(pos, r), order, max_nn)


def radius_cells(pos, r=R_DEFAULT, max_nn=MAX_NEIGHBORS):
    """Candidate: cell order (points sorted by their grid cell)."""
    ijk = np.floor(pos / r).astype(np.int64)
    ijk -= ijk.min(0)
    dims = ijk.max(0) + 1
    cid = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    return _collect(_within(pos, r), np.argsort(cid, kind='stable'), max_nn)


BACKENDS = {'scan (shipped A)': radius_scan,
            'blocks (shipped B)': radius_blocks,
            'cells (candidate)': radius_cells}


# --------------------------------------------------------------------------
def _sets(ei, n):
    out = [set() for _ in range(n)]
    for q, c in zip(ei[0].tolist(), ei[1].tolist()):
        out[q].add(c)
    return out


def cmd_capacity():
    pos = cloud()
    true_deg = _within(pos, R_DEFAULT).sum(1)
    at_cap = true_deg >= MAX_NEIGHBORS
    print(f'n={len(pos)}, r={R_DEFAULT}, max_neighbors={MAX_NEIGHBORS}\n')
    print(f'  true degree: min {true_deg.min()}, median '
          f'{int(np.median(true_deg))}, mean {true_deg.mean():.1f}, '
          f'max {true_deg.max()}')
    print(f'  queries at or over capacity : {at_cap.sum()} / {len(pos)} '
          f'({100 * at_cap.mean():.1f}%)')
    print(f'  neighbours discarded by the cap : '
          f'{int(np.clip(true_deg - MAX_NEIGHBORS, 0, None).sum())}')
    for name, fn in BACKENDS.items():
        ei = fn(pos)
        print(f'  {name:<20} returns {ei.shape[1]} edges')


def cmd_agree():
    pos = cloud()
    n = len(pos)
    true_deg = _within(pos, R_DEFAULT).sum(1)
    at_cap = true_deg >= MAX_NEIGHBORS
    eis = {k: fn(pos) for k, fn in BACKENDS.items()}

    print('every returned pair really is within r:')
    for name, ei in eis.items():
        d2 = ((pos[ei[1]] - pos[ei[0]]) ** 2).sum(-1)
        print(f'  {name:<20} {int((d2 < R_DEFAULT ** 2).sum())} / '
              f'{ei.shape[1]} valid')

    sets = {k: _sets(v, n) for k, v in eis.items()}
    names = list(BACKENDS)
    print('\npairwise agreement on the retained subset:')
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sets[names[i]], sets[names[j]]
            same_all = sum(1 for q in range(n) if a[q] == b[q])
            same_cap = sum(1 for q in range(n) if at_cap[q] and a[q] == b[q])
            overlap = sum(len(a[q] & b[q]) for q in range(n))
            total = sum(len(a[q]) for q in range(n))
            print(f'  {names[i]:<20} vs {names[j]:<20} '
                  f'identical for {same_all}/{n} queries '
                  f'({same_cap}/{int(at_cap.sum())} of the truncated ones); '
                  f'edge overlap {100 * overlap / total:.1f}%')


def cmd_overflow():
    pos = cloud()
    true_deg = _within(pos, R_DEFAULT).sum(1)
    exactly = int((true_deg == MAX_NEIGHBORS).sum())
    over = int((true_deg > MAX_NEIGHBORS).sum())
    ei = radius_scan(pos)
    deg = np.bincount(ei[0], minlength=len(pos))
    print(f'queries whose true degree is exactly {MAX_NEIGHBORS} : {exactly}')
    print(f'queries whose true degree is greater        : {over}')
    print(f'queries the API reports {MAX_NEIGHBORS} edges for   : '
          f'{int((deg == MAX_NEIGHBORS).sum())}')
    print('\nthe returned edge list is identical in both cases, and no count '
          'or overflow flag is returned, so the caller cannot tell them apart')


def cmd_sweep():
    pos = cloud()
    print(f"{'r':>8} {'mean true deg':>14} {'at capacity':>13} "
          f"{'scan vs cells identical':>25}")
    for r in (0.03, 0.05, 0.07, 0.10, 0.14):
        w = _within(pos, r)
        deg = w.sum(1)
        a = _sets(radius_scan(pos, r), len(pos))
        c = _sets(radius_cells(pos, r), len(pos))
        same = sum(1 for q in range(len(pos)) if a[q] == c[q])
        print(f'{r:>8.2f} {deg.mean():>14.1f} '
              f'{100 * (deg >= MAX_NEIGHBORS).mean():>12.1f}% '
              f'{100 * same / len(pos):>24.1f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command',
                    choices=['capacity', 'agree', 'overflow', 'sweep'])
    args = ap.parse_args()
    {'capacity': cmd_capacity, 'agree': cmd_agree, 'overflow': cmd_overflow,
     'sweep': cmd_sweep}[args.command]()
