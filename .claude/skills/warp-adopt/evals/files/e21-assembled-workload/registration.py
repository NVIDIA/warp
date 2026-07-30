# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""shapefit - iterative rigid-ish registration of a point cloud onto a target.

The inner loop is a nearest-neighbour lookup from the moving cloud into the
static target. Two implementations ship:

  * `nn_bruteforce`  - all-pairs scan, the current default
  * `nn_grid`        - uniform cell list, exact, with a radius that expands
                       until the answer is provably the nearest, then an exact
                       fallback scan

The grid path costs whatever the search radius forces it to touch, so its cost
depends on how far the moving cloud currently is from the target - a quantity
that shrinks by more than an order of magnitude while `fit` converges.

Entry points:

    python registration.py fit --iters 40 --capture snaps.npz
    python registration.py bench --workload assembled
    python registration.py bench --workload captured --snapshots snaps.npz
    python registration.py stats  --snapshots snaps.npz

Only numpy is required. Inputs are generated deterministically on first use.
"""
import argparse
import time

import numpy as np

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
N_TARGET = 20000
N_MOVING = 12000


def _sphere(n, radius, centre, seed):
    """Points on a sphere - a 2-D surface embedded in 3-D."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * radius + np.asarray(centre)).astype(np.float64)


def _torus(n, R, r, centre, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 2 * np.pi, n)
    v = rng.uniform(0.0, 2 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return (np.stack([x, y, z], axis=1) + np.asarray(centre)).astype(np.float64)


def target_cloud():
    """The static reference the app always registers against."""
    return _sphere(N_TARGET, 1.0, (0.0, 0.0, 0.0), seed=11)


def initial_moving():
    """Where `fit` starts: same shape family, displaced and mis-scaled."""
    return _sphere(N_MOVING, 1.35, (0.55, -0.30, 0.20), seed=23)


def unrelated_cloud():
    """A different shipped asset. Real data - but never registered against the
    sphere by this application."""
    return _torus(N_MOVING, 1.6, 0.45, (0.0, 0.0, 0.0), seed=37)


# --------------------------------------------------------------------------
# nearest neighbour
# --------------------------------------------------------------------------
def nn_bruteforce(q, ref, chunk=2048):
    """Exact all-pairs NN. Returns (sq_dist, index)."""
    out_d = np.empty(len(q))
    out_i = np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), chunk):
        blk = q[s:s + chunk]
        d = ((blk[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        j = d.argmin(1)
        out_i[s:s + chunk] = j
        out_d[s:s + chunk] = d[np.arange(len(blk)), j]
    return out_d, out_i


class CellGrid:
    """Uniform cell list over `ref`, with an exact expanding-radius query.

    `cell` is the one tuning knob. The query enumerates every cell overlapping
    the box of half-extent r about the point; a candidate at squared distance
    d2 <= r*r is provably the global nearest, otherwise r doubles. After
    `max_rounds` the query falls through to an exact scan, so the result is
    always correct - only the cost changes.
    """

    def __init__(self, ref, cell):
        self.ref = ref
        self.cell = float(cell)
        self.lo = ref.min(0)
        keys = np.floor((ref - self.lo) / self.cell).astype(np.int64)
        self.dims = keys.max(0) + 1
        flat = (keys[:, 0] * self.dims[1] + keys[:, 1]) * self.dims[2] + keys[:, 2]
        order = np.argsort(flat, kind="stable")
        self.order = order
        self.sorted_flat = flat[order]
        self.candidates_scanned = 0
        self.fallback_queries = 0

    def _cells_in_box(self, p, r):
        lo = np.floor((p - r - self.lo) / self.cell).astype(np.int64)
        hi = np.floor((p + r - self.lo) / self.cell).astype(np.int64)
        lo = np.clip(lo, 0, self.dims - 1)
        hi = np.clip(hi, 0, self.dims - 1)
        ax = np.arange(lo[0], hi[0] + 1)
        ay = np.arange(lo[1], hi[1] + 1)
        az = np.arange(lo[2], hi[2] + 1)
        g = np.stack(np.meshgrid(ax, ay, az, indexing="ij"), -1).reshape(-1, 3)
        return (g[:, 0] * self.dims[1] + g[:, 1]) * self.dims[2] + g[:, 2]

    def query_one(self, p, max_rounds=6):
        r = self.cell
        for _ in range(max_rounds):
            flats = self._cells_in_box(p, r)
            lo = np.searchsorted(self.sorted_flat, flats, side="left")
            hi = np.searchsorted(self.sorted_flat, flats, side="right")
            idx = np.concatenate([self.order[a:b] for a, b in zip(lo, hi)]) \
                if len(flats) else np.empty(0, dtype=np.int64)
            self.candidates_scanned += len(idx)
            if len(idx):
                d = ((self.ref[idx] - p) ** 2).sum(-1)
                k = d.argmin()
                if d[k] <= r * r:
                    return d[k], idx[k]
            r *= 2.0
        self.fallback_queries += 1
        d = ((self.ref - p) ** 2).sum(-1)
        self.candidates_scanned += len(self.ref)
        k = d.argmin()
        return d[k], k


def nn_grid(q, ref, cell):
    g = CellGrid(ref, cell)
    out_d = np.empty(len(q))
    out_i = np.empty(len(q), dtype=np.int64)
    for i, p in enumerate(q):
        out_d[i], out_i[i] = g.query_one(p)
    return out_d, out_i, g


def median_spacing(ref, k=512, seed=5):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(ref), size=min(k, len(ref)), replace=False)
    d = ((ref[sel][:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    d[np.arange(len(sel)), sel] = np.inf
    return float(np.sqrt(d.min(1)).mean())


# --------------------------------------------------------------------------
# the application
# --------------------------------------------------------------------------
def fit(iters, capture=None, snap_at=(0, 3, 8, 16, 39), verbose=True):
    """Register the moving cloud onto the target with damped similarity ICP.

    The update is a global scale + translation estimated from the current
    correspondences, so the fit converges to a genuine non-zero residual set by
    the sampling density - not to an exact point-for-point overlap.
    """
    tgt = target_cloud()
    mov = initial_moving().copy()
    snaps = {}
    for it in range(iters):
        d, j = nn_bruteforce(mov, tgt)
        rms = float(np.sqrt(d.mean()))
        if it in snap_at:
            snaps[f"it{it}"] = mov.copy()
            if verbose:
                print(f"  it {it:3d}  rms {rms:.6f}")
        m = tgt[j]
        cm, ct = mov.mean(0), m.mean(0)
        num = float(((mov - cm) * (m - ct)).sum())
        den = float(((mov - cm) ** 2).sum())
        s = num / den if den > 0 else 1.0
        damp = 0.45
        s = 1.0 + damp * (s - 1.0)
        shift = damp * (ct - cm)
        mov = (mov - cm) * s + cm + shift
    if capture:
        np.savez_compressed(capture, target=tgt, **snaps)
        if verbose:
            print(f"captured {len(snaps)} snapshots -> {capture}")
    return snaps


def _bench_pair(name, q, ref, cell):
    t0 = time.perf_counter()
    bd, bi = nn_bruteforce(q, ref)
    t_brute = time.perf_counter() - t0
    t0 = time.perf_counter()
    gd, gi, g = nn_grid(q, ref, cell)
    t_grid = time.perf_counter() - t0
    exact = bool(np.allclose(bd, gd) and (bi == gi).all())
    dnn = float(np.sqrt(bd).mean())
    print(f"  {name:<28} brute {t_brute:7.3f}s   grid {t_grid:7.3f}s   "
          f"speedup {t_brute / t_grid:5.2f}x   mean d_NN {dnn:.5f}   "
          f"cand/query {g.candidates_scanned / len(q):8.1f}   "
          f"fallbacks {g.fallback_queries:5d}   exact {exact}")
    return {"brute_s": t_brute, "grid_s": t_grid, "speedup": t_brute / t_grid,
            "d_nn": dnn, "cand_per_query": g.candidates_scanned / len(q),
            "fallbacks": g.fallback_queries, "exact": exact}


def bench(workload, snapshots, cell_mult, n_query):
    tgt = target_cloud()
    sp = median_spacing(tgt)
    cell = cell_mult * sp
    print(f"target: {len(tgt)} points, median spacing {sp:.6f}, "
          f"cell {cell:.6f} ({cell_mult}x spacing)")
    if workload == "assembled":
        print("workload: ASSEMBLED - moving cloud replaced by an unrelated "
              "shipped asset")
        q = unrelated_cloud()[:n_query]
        _bench_pair("unrelated asset -> target", q, tgt, cell)
    else:
        if not snapshots:
            raise SystemExit("--snapshots is required for --workload captured; "
                             "run `fit --capture snaps.npz` first")
        z = np.load(snapshots)
        print("workload: CAPTURED - moving cloud taken from a real fit() run")
        for k in sorted([k for k in z.files if k.startswith("it")],
                        key=lambda s: int(s[2:])):
            _bench_pair(f"{k} -> target", z[k][:n_query], tgt, cell)


def stats(snapshots):
    tgt = target_cloud()
    sp = median_spacing(tgt)
    print(f"target median spacing: {sp:.6f}")
    unrel = unrelated_cloud()
    d, _ = nn_bruteforce(unrel[:2000], tgt)
    print(f"  unrelated asset : mean d_NN {np.sqrt(d).mean():.6f}  "
          f"= {np.sqrt(d).mean() / sp:6.1f}x spacing")
    if snapshots:
        z = np.load(snapshots)
        for k in sorted([k for k in z.files if k.startswith("it")],
                        key=lambda s: int(s[2:])):
            d, _ = nn_bruteforce(z[k][:2000], tgt)
            m = float(np.sqrt(d).mean())
            print(f"  {k:<16}: mean d_NN {m:.6f}  = {m / sp:6.1f}x spacing")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--iters", type=int, default=40)
    f.add_argument("--capture", default=None)
    b = sub.add_parser("bench")
    b.add_argument("--workload", choices=["assembled", "captured"], default="assembled")
    b.add_argument("--snapshots", default=None)
    b.add_argument("--cell-mult", type=float, default=4.0)
    b.add_argument("--n-query", type=int, default=3000)
    s = sub.add_parser("stats")
    s.add_argument("--snapshots", default=None)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.iters, a.capture)
    elif a.cmd == "bench":
        bench(a.workload, a.snapshots, a.cell_mult, a.n_query)
    else:
        stats(a.snapshots)


if __name__ == "__main__":
    main()
