# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tilecut — one-shot CLI: cut one tile set from one input and exit.

This is invoked once per input file by an outer batch script; the process does
not persist between inputs.

Usage:
    python thumbnail.py --input sample.raw
    python thumbnail.py --input sample.raw --repeat 20   # dev loop only
"""

import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "sample.raw")


def ensure_input(path=DEFAULT_INPUT, side=1024):
    if not os.path.exists(path):
        rng = np.random.default_rng(0)
        rng.random((side, side), dtype=np.float32).tofile(path)
    return path


def load(path, side=1024):
    return np.fromfile(path, dtype=np.float32).reshape(side, side)


def cut_tiles(image, seeds):
    """Assign every pixel to its nearest seed, then reduce per seed."""
    h, w = image.shape
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1)
    flat = image.ravel()
    sums = np.zeros(len(seeds), dtype=np.float64)
    counts = np.zeros(len(seeds), dtype=np.int64)
    for s in range(0, len(coords), 65536):
        blk = coords[s : s + 65536]
        lab = ((blk[:, None, :] - seeds[None, :, :]) ** 2).sum(-1).argmin(axis=1)
        sums += np.bincount(lab, weights=flat[s : s + 65536], minlength=len(seeds))
        counts += np.bincount(lab, minlength=len(seeds))
    return sums / np.maximum(counts, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    ensure_input(args.input)
    rng = np.random.default_rng(3)
    seeds = rng.random((args.seeds, 2)).astype(np.float32)

    image = load(args.input)
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        tiles = cut_tiles(image, seeds)
    dt = (time.perf_counter() - t0) / args.repeat
    print(f"{len(tiles)} tiles, mean {tiles.mean():.6f}, {dt:.3f} s/call")


if __name__ == "__main__":
    main()
