# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""profkit - a four-stage pipeline with a bundled accelerator profiler.

`--profile` uses the tracer that ships with the library. The tracer records an
event when a stage begins dispatching and another when its kernel retires, and
the summary adds up everything it recorded.

`--wall` reports plain synchronized wall time for the same run, stage by
stage, with no tracer involved.

Both are printed by `--both`.

Commands:

    python profkit.py --profile
    python profkit.py --wall
    python profkit.py --both

Only numpy is required.
"""
import argparse
import time

import numpy as np

N = 1_200_000


def _load(rng):
    return rng.random(N)


def _classify(x):
    return (x * 7.3).astype(np.int32) % 11


def _aggregate(x, lab):
    return np.bincount(lab, weights=x, minlength=11)


def _emit(agg):
    return float(np.sqrt(np.abs(agg)).sum())


STAGES = ('load', 'classify', 'aggregate', 'emit')


class Tracer:
    """The profiler that ships with profkit.

    Each stage is recorded twice: once as the dispatching op and once as the
    kernel that retires for it. `summary()` totals every recorded event.
    """

    def __init__(self):
        self.events = []

    def record(self, stage, seconds):
        self.events.append((f'op:{stage}', seconds))
        self.events.append((f'kernel:{stage}', seconds))

    def summary(self):
        per = {}
        for name, sec in self.events:
            per[name] = per.get(name, 0.0) + sec
        total = sum(per.values())
        return per, total


def run(tracer=None):
    rng = np.random.default_rng(0)
    timings = {}

    t0 = time.perf_counter()
    x = _load(rng)
    timings['load'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lab = _classify(x)
    timings['classify'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    agg = _aggregate(x, lab)
    timings['aggregate'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _emit(agg)
    timings['emit'] = time.perf_counter() - t0

    if tracer is not None:
        for s in STAGES:
            tracer.record(s, timings[s])
    return timings


def cmd_profile():
    tracer = Tracer()
    run(tracer)
    per, total = tracer.summary()
    print('accelerator profile\n')
    print(f"{'event':<20} {'ms':>10} {'share':>8}")
    for name, sec in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f'{name:<20} {sec * 1e3:>10.2f} {100 * sec / total:>7.1f}%')
    print(f"\n{'total accelerator':<20} {total * 1e3:>10.2f} ms")
    return total


def cmd_wall():
    timings = run()
    total = sum(timings.values())
    print('synchronized wall time\n')
    print(f"{'stage':<20} {'ms':>10} {'share':>8}")
    for s in STAGES:
        print(f'{s:<20} {timings[s] * 1e3:>10.2f} '
              f'{100 * timings[s] / total:>7.1f}%')
    print(f"\n{'total wall':<20} {total * 1e3:>10.2f} ms")
    return total


def cmd_both():
    t0 = time.perf_counter()
    tracer = Tracer()
    timings = run(tracer)
    outer = time.perf_counter() - t0
    _, traced = tracer.summary()
    wall = sum(timings.values())
    print(f'total reported by the tracer : {traced * 1e3:8.2f} ms')
    print(f'total stage wall time        : {wall * 1e3:8.2f} ms')
    print(f'whole-run wall time          : {outer * 1e3:8.2f} ms')
    print(f'\ntracer total / stage wall total = {traced / wall:.2f}')
    print(f'tracer total / whole-run wall  = {traced / outer:.2f}')
    print('\nper-stage shares, both ways:')
    per, ttot = tracer.summary()
    print(f"{'stage':<12} {'tracer share':>13} {'wall share':>12}")
    for s in STAGES:
        tr = (per[f'op:{s}'] + per[f'kernel:{s}']) / ttot
        print(f'{s:<12} {100 * tr:>12.1f}% '
              f'{100 * timings[s] / wall:>11.1f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--wall', action='store_true')
    ap.add_argument('--both', action='store_true')
    a = ap.parse_args()
    if a.profile:
        cmd_profile()
    elif a.wall:
        cmd_wall()
    else:
        cmd_both()
