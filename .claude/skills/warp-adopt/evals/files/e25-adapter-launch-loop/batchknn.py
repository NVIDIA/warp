# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""batchknn - k-nearest neighbours over a batch of independent point sets.

`knn_scan` is the shipped path: one all-pairs scan per example.

`knn_cells_perexample` is a candidate cell index, wrapped the way an adapter
usually gets written first: it issues **one call per example** from the host,
and each call pays a fixed per-call setup (descriptor construction plus a
contiguity/validation pass) that does not depend on how many points the call
covers.

`knn_cells_fused` is the *same index and the same algorithm*, issued as a
single call over the whole batch.

All three return identical results. The only difference between the last two is
where the loop lives, so any gap between them belongs to the adapter - not to
the algorithm, and not to whatever backend the kernel is eventually written in.

Commands:

    python batchknn.py sweep     # total points fixed, number of examples varies
    python batchknn.py bench     # the shipped batch shape, all three paths
    python batchknn.py verify    # all three agree exactly

Only numpy is required.
"""
import argparse
import time

import numpy as np

TOTAL_POINTS = 32_768
SHIPPED_EXAMPLES = 16
K = 8

# Work the query API does once per call: it rebuilds a descriptor table and
# revalidates the index.  Flat in the size of the call.
_SETUP_ROWS = 900_000
_SETUP_PASSES = 32


def _per_call_setup():
    """Per-call wrapper work. O(1) in the size of the call."""
    tbl = np.arange(_SETUP_ROWS, dtype=np.float64)
    acc = 0.0
    for _ in range(_SETUP_PASSES):
        acc += float(np.sqrt(tbl).sum())
    return acc


def make_batch(total=TOTAL_POINTS, examples=SHIPPED_EXAMPLES, seed=3):
    """`examples` equal-sized clouds, each uniform in the unit cube."""
    rng = np.random.default_rng(seed)
    per = total // examples
    pos = rng.random((examples * per, 3))
    ptr = np.arange(0, examples * per + 1, per, dtype=np.int64)
    return pos, ptr


def _topk(d2, k):
    """Indices of the k smallest, in ascending order."""
    part = np.argpartition(d2, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(d2, part, axis=1)
    return np.take_along_axis(part, np.argsort(vals, axis=1, kind='stable'),
                              axis=1)


# --------------------------------------------------------------------------
# incumbent
# --------------------------------------------------------------------------
def _scan_one(p, k):
    """All-pairs distances, vectorised through the BLAS.

    This is the strong form of the incumbent, not a strawman loop: the
    candidate has to beat a tuned O(n^2) scan, not a naive one.
    """
    x2 = (p * p).sum(1)
    d2 = x2[:, None] + x2[None, :] - 2.0 * (p @ p.T)
    return _topk(d2, k)


def knn_scan(pos, ptr, k=K):
    """All-pairs scan per example. The shipped path."""
    out = np.empty((ptr[-1], k), dtype=np.int64)
    for b in range(len(ptr) - 1):
        lo, hi = ptr[b], ptr[b + 1]
        out[lo:hi] = _scan_one(pos[lo:hi], k) + lo
    return out


# --------------------------------------------------------------------------
# candidate: one uniform cell index, two adapters around it
# --------------------------------------------------------------------------
_OFFS = np.array([(i, j, l) for i in (-1, 0, 1)
                  for j in (-1, 0, 1) for l in (-1, 0, 1)], dtype=np.int64)


def _cells_one(p, k):
    """Exact k-NN via a uniform cell index.

    The 27-cell block around a query covers every point within `cell` of it,
    so a query whose k-th distance is within `cell` is answered exactly.  The
    (few) queries that are not are finished with a full scan, which keeps the
    result exact and makes this a cost question rather than a correctness one.
    """
    span = p.max(0) - p.min(0)
    cell = float((span.prod() * k * 0.75 / max(len(p), 1)) ** (1.0 / 3.0))

    ijk = np.floor(p / cell).astype(np.int64)
    ijk -= ijk.min(0)
    dims = ijk.max(0) + 1
    cid = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

    order = np.argsort(cid, kind='stable')
    cid_sorted = cid[order]
    rank = np.arange(len(cid)) - np.searchsorted(cid_sorted, cid_sorted)
    cap = int(rank.max()) + 1                      # no cell ever overflows
    table = np.full((int(dims.prod()), cap), -1, dtype=np.int64)
    table[cid_sorted, rank] = order

    nbr = ijk[:, None, :] + _OFFS[None, :, :]
    ok = ((nbr >= 0) & (nbr < dims)).all(-1)
    nbr = np.clip(nbr, 0, dims - 1)
    ncid = (nbr[..., 0] * dims[1] + nbr[..., 1]) * dims[2] + nbr[..., 2]
    cand = np.where(ok[..., None], table[ncid], -1).reshape(len(p), -1)

    pts = p[np.where(cand >= 0, cand, 0)]
    d2 = ((pts - p[:, None, :]) ** 2).sum(-1)
    d2 = np.where(cand >= 0, d2, np.inf)

    if d2.shape[1] < k:
        return _scan_one(p, k), len(p)
    sel = _topk(d2, k)
    out = np.take_along_axis(cand, sel, axis=1)
    kth = np.take_along_axis(d2, sel[:, -1:], axis=1)[:, 0]

    redo = np.nonzero(~(kth <= cell * cell))[0]    # also catches inf
    if len(redo):
        q = p[redo]
        d2r = ((q * q).sum(1)[:, None] + (p * p).sum(1)[None, :]
               - 2.0 * (q @ p.T))
        out[redo] = _topk(d2r, k)
    return out, len(redo)


def knn_cells_perexample(pos, ptr, k=K):
    """Candidate, one host call per example. Pays _per_call_setup() B times."""
    out = np.empty((ptr[-1], k), dtype=np.int64)
    for b in range(len(ptr) - 1):
        _per_call_setup()
        lo, hi = ptr[b], ptr[b + 1]
        out[lo:hi] = _cells_one(pos[lo:hi], k)[0] + lo
    return out


def knn_cells_fused(pos, ptr, k=K):
    """Same index, same algorithm, one host call for the whole batch."""
    _per_call_setup()
    out = np.empty((ptr[-1], k), dtype=np.int64)
    for b in range(len(ptr) - 1):
        lo, hi = ptr[b], ptr[b + 1]
        out[lo:hi] = _cells_one(pos[lo:hi], k)[0] + lo
    return out


# --------------------------------------------------------------------------
def _time(fn, *a, repeat=2):
    best = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def cmd_sweep():
    print(f'total points held at {TOTAL_POINTS}; only the number of examples '
          f'(and therefore host calls) changes\n')
    print(f"{'examples':>9} {'pts/ex':>8} {'per-example ms':>15} "
          f"{'fused ms':>10} {'gap ms':>9} {'gap/call ms':>12}")
    for b in (1, 2, 4, 8, 16, 32, 64):
        pos, ptr = make_batch(TOTAL_POINTS, b)
        t_loop = _time(knn_cells_perexample, pos, ptr)
        t_fused = _time(knn_cells_fused, pos, ptr)
        gap = t_loop - t_fused
        print(f'{b:>9} {TOTAL_POINTS // b:>8} {t_loop:>15.1f} '
              f'{t_fused:>10.1f} {gap:>9.1f} {gap / max(b - 1, 1):>12.2f}')
    print('\nthe fused column is the algorithm; the gap is host cost that '
          'scales with the number of calls, not with the problem')


def cmd_bench():
    pos, ptr = make_batch()
    print(f'shipped batch shape: {SHIPPED_EXAMPLES} examples x '
          f'{TOTAL_POINTS // SHIPPED_EXAMPLES} points, k={K}\n')
    base = None
    for name, fn in (('knn_scan (shipped)', knn_scan),
                     ('knn_cells_perexample', knn_cells_perexample),
                     ('knn_cells_fused', knn_cells_fused)):
        t = _time(fn, pos, ptr)
        if base is None:
            base = t
        print(f'  {name:<24} {t:8.1f} ms   {base / t:5.2f}x vs shipped')
    print(f'\n  one per-call setup costs '
          f'{_time(_per_call_setup):.2f} ms; the per-example adapter pays it '
          f'{SHIPPED_EXAMPLES} times')


def cmd_verify():
    pos, ptr = make_batch(6144, 8)
    a = knn_scan(pos, ptr)
    b = knn_cells_perexample(pos, ptr)
    c = knn_cells_fused(pos, ptr)
    print('per-example adapter == shipped :', np.array_equal(a, b))
    print('fused adapter       == shipped :', np.array_equal(a, c))
    print('per-example         == fused   :', np.array_equal(b, c))
    tot = sum(_cells_one(pos[ptr[i]:ptr[i + 1]], K)[1]
              for i in range(len(ptr) - 1))
    print(f'queries finished by fallback scan: {tot} / {ptr[-1]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['sweep', 'bench', 'verify'])
    args = ap.parse_args()
    {'sweep': cmd_sweep, 'bench': cmd_bench, 'verify': cmd_verify}[
        args.command]()
