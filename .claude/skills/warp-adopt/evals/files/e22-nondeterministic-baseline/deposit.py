# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fieldkit - deposit per-particle contributions onto a coarse cell grid.

`deposit_incumbent` is the shipped kernel. It accumulates with an
unspecified-order atomic add, so the summation order for a given cell differs
between calls; the code reproduces that by permuting before accumulating.

`deposit_sorted` is a candidate formulation that groups by cell first and sums
each segment contiguously. It computes the same quantity.

Commands:

    python deposit.py compare        # candidate vs incumbent, bitwise
    python deposit.py selfcheck      # incumbent vs ITSELF, repeated calls
    python deposit.py oracle         # both vs a float64 reference
    python deposit.py bench

Only numpy is required.
"""
import argparse
import os
import time

import numpy as np

N_PARTICLES = 400_000
N_CELLS = 2_000


def make_inputs(seed=17):
    """Deterministic inputs. Values span several magnitudes, which is what
    makes summation order visible in float32."""
    rng = np.random.default_rng(seed)
    cell = rng.integers(0, N_CELLS, size=N_PARTICLES).astype(np.int64)
    mag = 10.0 ** rng.uniform(-3.0, 3.0, size=N_PARTICLES)
    val = (rng.normal(size=N_PARTICLES) * mag).astype(np.float32)
    return val, cell


def deposit_incumbent(val, cell, n_cells=N_CELLS):
    """Shipped path. Atomic accumulation: the per-cell summation order is not
    specified and is not stable between calls."""
    # a fresh, non-reproducible permutation per call, standing in for the
    # hardware's atomic arrival order
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    order = rng.permutation(len(val))
    out = np.zeros(n_cells, dtype=np.float32)
    np.add.at(out, cell[order], val[order])
    return out


def deposit_sorted(val, cell, n_cells=N_CELLS):
    """Candidate. Group by cell, then sum each segment contiguously.
    Deterministic, and computes the same mathematical quantity."""
    order = np.argsort(cell, kind="stable")
    c = cell[order]
    v = val[order].astype(np.float32)
    out = np.zeros(n_cells, dtype=np.float32)
    # segment boundaries
    starts = np.searchsorted(c, np.arange(n_cells), side="left")
    ends = np.searchsorted(c, np.arange(n_cells), side="right")
    for i in range(n_cells):
        if ends[i] > starts[i]:
            out[i] = np.float32(v[starts[i]:ends[i]].sum(dtype=np.float32))
    return out


def deposit_reference(val, cell, n_cells=N_CELLS):
    """float64 oracle. Independent of both implementations above."""
    out = np.zeros(n_cells, dtype=np.float64)
    np.add.at(out, cell, val.astype(np.float64))
    return out


def _rel(a, b):
    scale = max(float(np.abs(b).max()), 1e-30)
    return float(np.abs(a - b).max() / scale)


def cmd_compare():
    val, cell = make_inputs()
    inc = deposit_incumbent(val, cell)
    cand = deposit_sorted(val, cell)
    print("candidate vs incumbent")
    print(f"  bitwise equal            : {np.array_equal(inc, cand)}")
    print(f"  max abs diff             : {float(np.abs(inc - cand).max()):.6e}")
    print(f"  max rel diff             : {_rel(cand, inc):.6e}")
    print()
    print("A bitwise-equality contract fails here. Before concluding the")
    print("candidate is defective, run `selfcheck`.")


def cmd_selfcheck(repeats=10):
    val, cell = make_inputs()
    runs = [deposit_incumbent(val, cell) for _ in range(repeats)]
    base = runs[0]
    spread = max(float(np.abs(base - r).max()) for r in runs[1:])
    rel = max(_rel(r, base) for r in runs[1:])
    ident = all(np.array_equal(base, r) for r in runs[1:])
    print(f"incumbent vs itself, {repeats} calls on identical inputs")
    print(f"  all runs bitwise equal   : {ident}")
    print(f"  max abs self-spread      : {spread:.6e}")
    print(f"  max rel self-spread      : {rel:.6e}")
    print()
    print("This is the floor for any equality tolerance. A contract tighter")
    print("than this is one the incumbent itself cannot satisfy.")


def cmd_oracle():
    val, cell = make_inputs()
    ref = deposit_reference(val, cell)
    inc = deposit_incumbent(val, cell).astype(np.float64)
    cand = deposit_sorted(val, cell).astype(np.float64)
    print("scored against the float64 reference (both sides)")
    print(f"  incumbent max rel error  : {_rel(inc, ref):.6e}")
    print(f"  candidate max rel error  : {_rel(cand, ref):.6e}")
    print()
    print("Report both, not just the candidate's disagreement with the")
    print("incumbent.")


def cmd_bench(repeats=5):
    val, cell = make_inputs()
    for name, fn in (("incumbent", deposit_incumbent), ("sorted", deposit_sorted)):
        fn(val, cell)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(val, cell)
            ts.append(time.perf_counter() - t0)
        print(f"  {name:<12}{np.median(ts) * 1e3:9.2f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["compare", "selfcheck", "oracle", "bench"])
    a = ap.parse_args()
    {"compare": cmd_compare, "selfcheck": cmd_selfcheck,
     "oracle": cmd_oracle, "bench": cmd_bench}[a.cmd]()


if __name__ == "__main__":
    main()
