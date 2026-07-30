# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""seedlabel - assign each sample the index of its nearest seed.

The returned *index* is part of the public contract: downstream code uses it to
look up per-seed attributes, so a different-but-equally-near index is not free
to vary the way the distance is.

Two float32 implementations:

  * `nearest_incumbent` - single ascending scan, strict `<`, so the lowest
                          index wins a tie
  * `nearest_tiled`     - the same scan blocked into tiles and reduced across
                          tiles, which is what any accelerated formulation does

and a float64 oracle, `nearest_oracle`.

Commands:

    python label.py compare      # candidate vs incumbent
    python label.py oracle       # BOTH scored against the float64 oracle
    python label.py reachable    # how often the index is well-defined at all
    python label.py layout --layout {lattice,scatter}

`--layout lattice` is the shipped geometry: seeds on a regular lattice, where
samples routinely sit equidistant between neighbours. `--layout scatter` is
random seeds, where exact ties are rare. The same comparison gives wildly
different mismatch rates on the two.

Only numpy is required.
"""
import argparse

import numpy as np

N_SAMPLES = 60_000


def make_seeds(layout, seed=3):
    rng = np.random.default_rng(seed)
    if layout == "lattice":
        # seeds on a regular 12x12x12 lattice: perpendicular bisectors between
        # neighbours are dense, so equidistant samples are common
        g = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        s = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        return s.astype(np.float32)
    s = rng.uniform(-1.0, 1.0, size=(12 ** 3, 3))
    return s.astype(np.float32)


def make_samples(layout, seeds, seed=9):
    rng = np.random.default_rng(seed)
    if layout == "lattice":
        # half the samples sit exactly on midpoints between neighbouring seeds
        n_mid = N_SAMPLES // 2
        i = rng.integers(0, len(seeds), n_mid)
        j = rng.integers(0, len(seeds), n_mid)
        mid = ((seeds[i].astype(np.float64) + seeds[j].astype(np.float64)) / 2.0)
        rest = rng.uniform(-1.0, 1.0, size=(N_SAMPLES - n_mid, 3))
        return np.concatenate([mid, rest]).astype(np.float32)
    return rng.uniform(-1.0, 1.0, size=(N_SAMPLES, 3)).astype(np.float32)


def nearest_incumbent(q, seeds):
    """float32, single ascending scan, lowest index wins a tie."""
    best_d = np.full(len(q), np.inf, dtype=np.float32)
    best_i = np.zeros(len(q), dtype=np.int64)
    for j in range(len(seeds)):
        d = ((q - seeds[j]) ** 2).sum(-1, dtype=np.float32)
        upd = d < best_d          # strict: first (lowest) index keeps the tie
        best_d = np.where(upd, d, best_d)
        best_i = np.where(upd, j, best_i)
    return best_d, best_i


def nearest_tiled(q, seeds, tile=64):
    """float32, blocked scan reduced across tiles. Same arithmetic, different
    visit order - which is all it takes to resolve a tie differently."""
    best_d = np.full(len(q), np.inf, dtype=np.float32)
    best_i = np.zeros(len(q), dtype=np.int64)
    n = len(seeds)
    for s in range(0, n, tile):
        e = min(s + tile, n)
        d = ((q[:, None, :] - seeds[None, s:e, :]) ** 2).sum(-1, dtype=np.float32)
        k = d.argmin(1)
        dk = d[np.arange(len(q)), k]
        # `<=`, not `<`: an accelerated traversal has no defined visit order, so
        # whichever tied candidate is reduced last is the one that survives.
        upd = dk <= best_d
        best_d = np.where(upd, dk, best_d)
        best_i = np.where(upd, k + s, best_i)
    return best_d, best_i


def nearest_oracle(q, seeds, chunk=4096):
    """float64 reference, independent of both."""
    qq = q.astype(np.float64)
    ss = seeds.astype(np.float64)
    out_d = np.empty(len(q))
    out_i = np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), chunk):
        blk = qq[s:s + chunk]
        d = ((blk[:, None, :] - ss[None, :, :]) ** 2).sum(-1)
        j = d.argmin(1)
        out_i[s:s + chunk] = j
        out_d[s:s + chunk] = d[np.arange(len(blk)), j]
    return out_d, out_i


def _load(layout):
    seeds = make_seeds(layout)
    q = make_samples(layout, seeds)
    return q, seeds


def cmd_compare(layout):
    q, seeds = _load(layout)
    di, ii = nearest_incumbent(q, seeds)
    dt, it = nearest_tiled(q, seeds)
    n = len(q)
    print(f"layout={layout}  samples={n}  seeds={len(seeds)}")
    print(f"  index mismatch candidate vs incumbent : {int((ii != it).sum())} / {n} "
          f"({100.0 * (ii != it).mean():.2f}%)")
    print(f"  max abs distance difference           : {float(np.abs(di - dt).max()):.3e}")
    print()
    print("The distances agree; only the indices differ. Do not stop here -")
    print("run `oracle` and `reachable`.")


def cmd_oracle(layout):
    q, seeds = _load(layout)
    _, ii = nearest_incumbent(q, seeds)
    _, it = nearest_tiled(q, seeds)
    do, io = nearest_oracle(q, seeds)
    n = len(q)
    inc_bad = ii != io
    cand_bad = it != io
    print(f"layout={layout}  scored against the float64 oracle (both sides)")
    print(f"  incumbent index != oracle : {int(inc_bad.sum())} / {n} "
          f"({100.0 * inc_bad.mean():.2f}%)")
    print(f"  candidate index != oracle : {int(cand_bad.sum())} / {n} "
          f"({100.0 * cand_bad.mean():.2f}%)")
    print(f"  both wrong on the same sample : {int((inc_bad & cand_bad).sum())}")
    # is the disagreement a genuine tie, or a real error?
    qq = q.astype(np.float64)
    ss = seeds.astype(np.float64)
    sel = np.nonzero(cand_bad)[0]
    if len(sel):
        d_cand = ((qq[sel] - ss[it[sel]]) ** 2).sum(-1)
        d_or = ((qq[sel] - ss[io[sel]]) ** 2).sum(-1)
        exact_tie = int(np.sum(d_cand == d_or))
        print(f"  of the candidate's mismatches, exactly equidistant in float64 : "
              f"{exact_tie} / {len(sel)}")
        print(f"  worst excess distance chosen by candidate : "
              f"{float((d_cand - d_or).max()):.3e}")


def cmd_reachable(layout):
    """How often is the contested output even well defined?"""
    q, seeds = _load(layout)
    qq = q.astype(np.float64)
    ss = seeds.astype(np.float64)
    n = len(q)
    unamb = 0
    chunk = 4096
    for s in range(0, n, chunk):
        blk = qq[s:s + chunk]
        d = ((blk[:, None, :] - ss[None, :, :]) ** 2).sum(-1)
        part = np.partition(d, 1, axis=1)
        unamb += int(np.sum(part[:, 0] < part[:, 1]))
    print(f"layout={layout}")
    print(f"  samples whose nearest seed is strictly unique : {unamb} / {n} "
          f"({100.0 * unamb / n:.2f}%)")
    print(f"  samples exactly equidistant between >=2 seeds : {n - unamb} / {n} "
          f"({100.0 * (n - unamb) / n:.2f}%)")
    print()
    print("A mismatch count means nothing without this number.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["compare", "oracle", "reachable", "layout"])
    ap.add_argument("--layout", choices=["lattice", "scatter"], default="lattice")
    a = ap.parse_args()
    if a.cmd == "layout":
        for lay in ("lattice", "scatter"):
            cmd_compare(lay)
            cmd_reachable(lay)
            print()
        return
    {"compare": cmd_compare, "oracle": cmd_oracle,
     "reachable": cmd_reachable}[a.cmd](a.layout)


if __name__ == "__main__":
    main()
