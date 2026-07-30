# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gridkit — fit a field to samples, then export it.

Two commands are installed by this package:

    gridkit-fit      (python gridkit.py fit)
    gridkit-export   (python gridkit.py export)

Both accept --profile to print the built-in stage timers.
"""

import argparse
import time
from collections import OrderedDict

import numpy as np

TIMERS = OrderedDict()


def timed(name):
    def deco(fn):
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            TIMERS[name] = TIMERS.get(name, 0.0) + (time.perf_counter() - t0)
            return r

        return wrapper

    return deco


# --------------------------------------------------------------------- fit


@timed("fit.sample_batch")
def sample_batch(field, rng, n=4096):
    idx = rng.integers(0, field.shape[0], size=n)
    return field[idx]


@timed("fit.forward_backward")
def forward_backward(batch, weights):
    out = batch
    for _ in range(8):
        out = np.tanh(out @ weights)
    grad = out.T @ batch
    return float(np.abs(grad).mean())


def cmd_fit(args):
    rng = np.random.default_rng(0)
    field = rng.random((200000, 32)).astype(np.float32)
    weights = rng.random((32, 32)).astype(np.float32)
    loss = 0.0
    for _ in range(args.iters):
        batch = sample_batch(field, rng)
        loss = forward_backward(batch, weights)
    print(f"fit done: {args.iters} iterations, final |grad|={loss:.6f}")


# ------------------------------------------------------------------ export


@timed("export.build_grid")
def build_grid(resolution):
    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    return np.stack(np.meshgrid(lin, lin, indexing="ij"), axis=-1).reshape(-1, 2)


@timed("export.nearest_seed")
def nearest_seed(grid, seeds):
    d2 = ((grid[:, None, :] - seeds[None, :, :]) ** 2).sum(-1)
    return d2.argmin(axis=1)


@timed("export.write")
def write(labels, path="field.txt"):
    with open(path, "w") as fh:
        fh.write(f"{len(labels)}\n")
    return path


def cmd_export(args):
    rng = np.random.default_rng(1)
    grid = build_grid(args.resolution)
    seeds = rng.random((args.seeds, 2)).astype(np.float32)
    labels = nearest_seed(grid, seeds)
    out = write(labels)
    print(f"export done: {len(labels)} cells -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--iters", type=int, default=2000)
    f.add_argument("--profile", action="store_true")
    f.set_defaults(func=cmd_fit)
    e = sub.add_parser("export")
    e.add_argument("--resolution", type=int, default=256)
    e.add_argument("--seeds", type=int, default=400)
    e.add_argument("--profile", action="store_true")
    e.set_defaults(func=cmd_export)
    args = ap.parse_args()

    t0 = time.perf_counter()
    args.func(args)
    total = time.perf_counter() - t0

    if args.profile:
        print(f"\n{'timer':<24}{'seconds':>10}{'share':>9}")
        for k, v in sorted(TIMERS.items(), key=lambda kv: -kv[1]):
            print(f"{k:<24}{v:>10.3f}{100 * v / total:>8.1f}%")
        print(f"{'WALL':<24}{total:>10.3f}")


if __name__ == "__main__":
    main()
