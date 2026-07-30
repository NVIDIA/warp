# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tally — summarise a sensor stream, then map the survivors through a calibration table.

Two stages, both per-element and both branchy:

    classify(signal)          one pass over the whole stream, skip NaN, threshold
    calibrate(values, table)  per-element table lookup on the survivors

``--probe`` runs a read-only pass over the same bytes as each stage, doing no
per-element work. That is the floor any implementation of the stage has to pay
just to look at its input.

Usage:
    python tally.py --stages
    python tally.py --probe
    python tally.py --rows 200000000 --survivors 4000000
"""

import argparse
import time

import numpy as np

TABLE_SIZE = 4096


def make_stream(n, survivors, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.random(n)
    signal[::97] = np.nan
    values = rng.random(survivors)
    table = np.linspace(0.5, 2.0, TABLE_SIZE)
    return signal, values, table


def classify(signal, threshold=0.5):
    """Per element: skip NaN, compare against a threshold, count."""
    return int(np.count_nonzero(signal > threshold))


def calibrate(values, table):
    """Per element: derive an index and read the calibration table at it."""
    idx = (values * (len(table) - 1)).astype(np.int32)
    np.clip(idx, 0, len(table) - 1, out=idx)
    return table[idx]


def read_probe(a):
    """Touch the same bytes, do no per-element work."""
    return float(a.sum())


def best_of(fn, reps=5):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000_000)
    ap.add_argument("--survivors", type=int, default=2_000_000)
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    signal, values, table = make_stream(args.rows, args.survivors)
    t_classify = best_of(lambda: classify(signal))
    t_calibrate = best_of(lambda: calibrate(values, table))
    total = t_classify + t_calibrate

    print(f"{'stage':<12}{'seconds':>10}{'share':>9}{'input GB':>11}{'GB/s':>9}")
    for name, dt, nbytes in (("classify", t_classify, signal.nbytes),
                             ("calibrate", t_calibrate, values.nbytes)):
        print(f"{name:<12}{dt:>10.4f}{100*dt/total:>8.1f}%{nbytes/1e9:>11.3f}"
              f"{nbytes/dt/1e9:>9.1f}")

    if args.probe or args.stages:
        p_classify = best_of(lambda: read_probe(signal))
        p_calibrate = best_of(lambda: read_probe(values))
        print(f"\n{'stage':<12}{'stage s':>10}{'probe s':>10}{'probe GB/s':>12}"
              f"{'headroom':>11}")
        for name, dt, pt, nbytes in (("classify", t_classify, p_classify, signal.nbytes),
                                     ("calibrate", t_calibrate, p_calibrate, values.nbytes)):
            print(f"{name:<12}{dt:>10.4f}{pt:>10.4f}{nbytes/pt/1e9:>12.1f}"
                  f"{100*(dt-pt)/dt:>10.1f}%")
        print("\nheadroom = the share of the stage that is not the cost of reading its "
              "input;\nno implementation of the same stage can recover more than that.")


if __name__ == "__main__":
    main()
