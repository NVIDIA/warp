#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver skeleton — copy one per bottleneck and fill the two marked places.

Everything measurement-shaped is delegated to measure.py, so what belongs here
is only what is genuinely per-evaluation: how the workload is built, and what
the cases are. Keep the workload identical across cases; only the implementation
under test may differ.

Run it, then transcribe the emitted records into the report. A table cell with
no record in that file reads "not measured".

Usage:
    flock -x "$GPU_LOCK" -c 'python driver-<bottleneck>.py'
Arguments:
    Edit CASES, SIZES, INPUT_DISTRIBUTION, build_workload() and run().
Output:
    One run-unique JSON Lines file under results/.
Exit codes:
    0 on completion; subprocess failures are recorded in the emitted JSON.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure

# Run-unique, so a re-run cannot silently overwrite results the report cites.
RESULTS = Path("results/bench-<bottleneck>-<runid>.jsonl")

CASES = ["baseline", "warp"]
SIZES = (10**3, 10**6)

# Required report field, and for relational operations routinely the largest
# lever on the result — cost depends on the joint distribution of the inputs.
INPUT_DISTRIBUTION = "<how the inputs relate, not just their sizes>"


def build_workload(size):
    """Return whatever the cases consume. Identical for every case."""
    raise NotImplementedError  # <-- FILL IN


def run(case, workload):
    """Return a zero-argument callable performing one end-to-end public call."""
    raise NotImplementedError  # <-- FILL IN


def scaled(workload):
    """Return make_fn(k) -> callable doing k times the work, for the null test.
    Return None if the work cannot be scaled, and say so in the report."""
    return None


def measure_case(case, size):
    measure.device_baseline()
    workload = build_workload(size)
    fn = run(case, workload)
    warp_candidate = case == "warp"
    device = "cuda:0" if warp_candidate else None

    make_fn = scaled(workload)
    return {
        "case": case,
        "size": size,
        "input_distribution": INPUT_DISTRIBUTION,
        # protocol default: one discarded warm-up, then one timed run
        "timing": measure.time_it(fn, device=device, warp_candidate=warp_candidate),
        "null_test": measure.null_test(make_fn, device=device, warp_candidate=warp_candidate)
        if make_fn
        else {"pass": None, "note": "work not scalable"},
        "host": measure.host_memory(),
        # NVML per-process: framework-agnostic. Pass backend= only to attach
        # that allocator's live/reserved split as a supplement.
        "device": measure.device_memory(),
    }


if __name__ == "__main__":
    import os

    if case := measure.isolated_case():
        # child: one JSON record on stdout
        measure.emit(None, measure_case(case, int(os.environ["SIZE"])))
    else:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        # Sweep decades, not one representative size: relative performance is a
        # function of scale, not a property of the implementations.
        for size in SIZES:
            for c in CASES:
                measure.emit(RESULTS, measure.run_isolated(c, extra_env={"SIZE": str(size)}))
        print(f"wrote {RESULTS}")
