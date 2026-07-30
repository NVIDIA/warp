# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""surfkit - k-nearest-neighbour lookup for points on a scanned surface.

`knn_bruteforce` is the shipped path: an all-pairs scan per query.

`knn_cells` is a candidate cell list. Its search starts at the cell size and
doubles until the k-th candidate is provably inside the searched radius, so
**the cell size sets the initial search radius** and therefore how much work
every query does. Too coarse and each query drags in a large neighbourhood;
too fine and it pays repeated expansion rounds.

The one tuning knob ships with `auto_cell()`, which estimates spacing from the
*bounding-box volume*. The scan data is a surface - a 2-D manifold in 3-D - so
a volumetric estimate is wrong for it and the default lands well off the
optimum.

Commands:

    python neighbours.py spacing     # heuristic vs measured spacing
    python neighbours.py bench       # candidate at its shipped default only
    python neighbours.py sweep       # candidate across cell sizes

`candidates scanned per query` is the reproducible work metric; wall time is
reported alongside it.

Only numpy is required.
"""
import argparse
import time

import numpy as np

N_POINTS = 120_000
N_QUERY = 1_200
K = 16


N_STRAY = 6


def surface_points(n=N_POINTS, seed=7, stray=N_STRAY):
    """Points on a bumpy closed surface: intrinsic dimension 2, embedded in 3.

    Plus a handful of stray returns far off the object, which every real scan
    has. They are irrelevant to the k-NN answers but they enlarge the bounding
    box, which is all a volume-derived heuristic looks at.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    bump = 1.0 + 0.12 * np.sin(3.0 * v[:, 0]) * np.cos(3.0 * v[:, 1])
    surf = (v * bump[:, None]).astype(np.float64)
    if stray:
        s = rng.normal(size=(stray, 3))
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        surf = np.concatenate([surf, (s * rng.uniform(7.0, 9.0, (stray, 1)))])
    return surf


def volume_points(n=N_POINTS, seed=7):
    """A procedural stand-in: the same point count filling the same box.

    This is what an assessment reaches for when it has no real asset. Its
    intrinsic dimension is 3, not 2, so a cell size tuned on it is tuned for a
    different problem.
    """
    rng = np.random.default_rng(seed)
    ref = surface_points(n, seed)
    lo, hi = ref.min(0), ref.max(0)
    return lo + rng.random((n, 3)) * (hi - lo)


def measured_spacing(pts, k=400, seed=1):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pts), size=min(k, len(pts)), replace=False)
    d = ((pts[sel][:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    d[np.arange(len(sel)), sel] = np.inf
    return float(np.median(np.sqrt(d.min(1))))


def auto_cell(pts, per_cell=2.0):
    """Shipped default. Spacing from bounding-box VOLUME density.

    Correct for points filling a volume; far too coarse for points on a surface.
    """
    ext = pts.max(0) - pts.min(0)
    vol = float(np.prod(ext))
    return float((vol * per_cell / max(len(pts), 1)) ** (1.0 / 3.0))


def knn_bruteforce(query, pts, k=K, chunk=128):
    out = np.empty((len(query), k), dtype=np.int64)
    for s in range(0, len(query), chunk):
        blk = query[s:s + chunk]
        d = ((blk[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        dd = np.take_along_axis(d, idx, 1)
        order = np.argsort(dd, axis=1)
        out[s:s + chunk] = np.take_along_axis(idx, order, 1)
    return out


class CellList:
    def __init__(self, pts, cell):
        self.pts = pts
        self.cell = float(cell)
        self.lo = pts.min(0)
        keys = np.floor((pts - self.lo) / self.cell).astype(np.int64)
        self.dims = keys.max(0) + 1
        flat = (keys[:, 0] * self.dims[1] + keys[:, 1]) * self.dims[2] + keys[:, 2]
        self.order = np.argsort(flat, kind="stable")
        self.sorted_flat = flat[self.order]
        self.scanned = 0
        self.rounds = 0
        self.fallbacks = 0

    def _gather(self, q, r):
        lo = np.floor((q - r - self.lo) / self.cell).astype(np.int64)
        hi = np.floor((q + r - self.lo) / self.cell).astype(np.int64)
        lo = np.clip(lo, 0, self.dims - 1)
        hi = np.clip(hi, 0, self.dims - 1)
        ax = np.arange(lo[0], hi[0] + 1)
        ay = np.arange(lo[1], hi[1] + 1)
        az = np.arange(lo[2], hi[2] + 1)
        g = np.stack(np.meshgrid(ax, ay, az, indexing="ij"), -1).reshape(-1, 3)
        flats = (g[:, 0] * self.dims[1] + g[:, 1]) * self.dims[2] + g[:, 2]
        a = np.searchsorted(self.sorted_flat, flats, side="left")
        b = np.searchsorted(self.sorted_flat, flats, side="right")
        keep = b > a
        if not keep.any():
            return np.empty(0, dtype=np.int64)
        a, b = a[keep], b[keep]
        counts = b - a
        total = int(counts.sum())
        starts = np.repeat(a, counts)
        offs = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        return self.order[starts + offs]

    def knn(self, q, k=K, max_rounds=5):
        r = self.cell
        for _ in range(max_rounds):
            idx = self._gather(q, r)
            self.rounds += 1
            self.scanned += len(idx)
            if len(idx) >= k:
                d = ((self.pts[idx] - q) ** 2).sum(-1)
                sel = np.argpartition(d, k - 1)[:k]
                # admissible only if the k-th neighbour is inside the radius
                if d[sel].max() <= r * r:
                    return idx[sel][np.argsort(d[sel])]
            r *= 2.0
        self.fallbacks += 1
        d = ((self.pts - q) ** 2).sum(-1)
        self.scanned += len(self.pts)
        sel = np.argpartition(d, k - 1)[:k]
        return sel[np.argsort(d[sel])]


def knn_cells(query, pts, cell, k=K):
    cl = CellList(pts, cell)
    out = np.stack([cl.knn(q, k) for q in query])
    return out, cl


def _time(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0


def _setup():
    pts = surface_points()
    rng = np.random.default_rng(21)
    query = pts[rng.choice(len(pts), N_QUERY, replace=False)]
    return pts, query, measured_spacing(pts)


def run(cell_values, verbose=True):
    pts, query, sp = _setup()
    ref, t_brute = _time(lambda: knn_bruteforce(query, pts))
    if verbose:
        print(f"points {len(pts)}  queries {len(query)}  k={K}  "
              f"measured spacing {sp:.6f}")
        print(f"bruteforce (shipped) : {t_brute:7.3f} s")
        print(f"{'cell':>12}{'x spacing':>11}{'time s':>10}{'speedup':>9}"
              f"{'scanned/query':>15}{'rounds/q':>10}{'exact':>7}")
    rows = []
    for cell in cell_values:
        (got, cl), t = _time(lambda c=cell: knn_cells(query, pts, c))
        ok = bool((got == ref).all())
        rows.append({"cell": cell, "mult": cell / sp, "time": t,
                     "speedup": t_brute / t, "scanned": cl.scanned / len(query),
                     "rounds": cl.rounds / len(query), "fallbacks": cl.fallbacks,
                     "exact": ok})
        if verbose:
            r = rows[-1]
            print(f"{cell:>12.6f}{r['mult']:>11.2f}{t:>10.3f}{r['speedup']:>8.2f}x"
                  f"{r['scanned']:>15.1f}{r['rounds']:>10.2f}{str(ok):>7}")
    return sp, t_brute, rows


def _best_multiplier(pts, mults, seed=21):
    """Sweep cell = m * spacing on `pts`; return (spacing, best m, table)."""
    rng = np.random.default_rng(seed)
    query = pts[rng.choice(len(pts), N_QUERY, replace=False)]
    sp = measured_spacing(pts)
    ref, t_brute = _time(lambda: knn_bruteforce(query, pts))
    rows = []
    for m in mults:
        (got, cl), t = _time(lambda c=m * sp: knn_cells(query, pts, c))
        rows.append({"mult": m, "time": t, "speedup": t_brute / t,
                     "scanned": cl.scanned / len(query),
                     "exact": bool((got == ref).all())})
    best = min(rows, key=lambda r: r["time"])
    return sp, best["mult"], rows, t_brute


def cmd_transfer():
    """Tune on the procedural stand-in, then use that setting on real data."""
    mults = (0.5, 1, 2, 3, 4, 6, 8, 12)
    vol = volume_points()
    surf = surface_points()

    sp_v, m_vol, rows_v, tb_v = _best_multiplier(vol, mults)
    sp_s, m_surf, rows_s, tb_s = _best_multiplier(surf, mults)

    print("tuned on the procedural stand-in (uniform in the same box):")
    print(f"  measured spacing {sp_v:.6f}, best cell = {m_vol:g}x spacing "
          f"-> {max(r['speedup'] for r in rows_v):.1f}x vs bruteforce")
    print("\ntuned on the real scan (points on a surface):")
    print(f"  measured spacing {sp_s:.6f}, best cell = {m_surf:g}x spacing "
          f"-> {max(r['speedup'] for r in rows_s):.1f}x vs bruteforce")

    on_surf = {r["mult"]: r for r in rows_s}
    carried = on_surf[m_vol]
    native = on_surf[m_surf]
    print("\ncarrying the stand-in's setting onto the real scan:")
    print(f"  cell = {m_vol:g}x spacing -> {carried['speedup']:.1f}x, "
          f"{carried['scanned']:.0f} scanned/query")
    print(f"  cell = {m_surf:g}x spacing -> {native['speedup']:.1f}x, "
          f"{native['scanned']:.0f} scanned/query")
    print(f"  the transferred setting costs "
          f"{native['speedup'] / carried['speedup']:.2f}x on the real data")
    print("\nboth are exact; this is a cost question. The optimum moves "
          "because the two workloads have different intrinsic dimension, so a "
          "knob tuned on the stand-in is not the knob for the shipped data.")


def cmd_spacing():
    pts = surface_points()
    sp = measured_spacing(pts)
    ac = auto_cell(pts)
    print(f"bounding box extent      : {pts.max(0) - pts.min(0)}")
    print(f"auto_cell() default      : {ac:.6f}")
    print(f"measured median spacing  : {sp:.6f}")
    print(f"ratio (default / actual) : {ac / sp:.2f}x")
    print()
    print("auto_cell() assumes points fill the bounding volume. These lie on a")
    print("surface, so the default is too coarse and every query starts with a")
    print("search radius far larger than it needs.")


def cmd_bench():
    pts = surface_points()
    run([auto_cell(pts)])
    print()
    print("This is the candidate at its shipped default. Before drawing any")
    print("conclusion from it, run `sweep`.")


def cmd_sweep():
    pts = surface_points()
    sp = measured_spacing(pts)
    ac = auto_cell(pts)
    cells = sorted({round(m * sp, 8) for m in (0.5, 1, 2, 3, 4, 6, 8, 12)} | {round(ac, 8)})
    sp_, tb, rows = run(cells)
    best = min(rows, key=lambda x: x["time"])
    dflt = min(rows, key=lambda x: abs(x["cell"] - ac))
    print()
    print(f"default cell {dflt['cell']:.6f} ({dflt['mult']:.2f}x spacing) -> "
          f"{dflt['speedup']:.2f}x vs bruteforce, {dflt['scanned']:.0f} scanned/query")
    print(f"best    cell {best['cell']:.6f} ({best['mult']:.2f}x spacing) -> "
          f"{best['speedup']:.2f}x vs bruteforce, {best['scanned']:.0f} scanned/query")
    print(f"tuning the one documented knob is worth "
          f"{best['speedup'] / dflt['speedup']:.1f}x on time and "
          f"{dflt['scanned'] / best['scanned']:.1f}x on work")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["bench", "sweep", "spacing",
                                    "transfer"])
    a = ap.parse_args()
    {"spacing": cmd_spacing, "bench": cmd_bench, "sweep": cmd_sweep,
     "transfer": cmd_transfer}[a.cmd]()


if __name__ == "__main__":
    main()
