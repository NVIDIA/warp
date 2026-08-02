# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch entry point: print the inspection report for a generated field.

This is the command that runs in CI and pays the startup cost.
"""

from __future__ import annotations

from field_solver.inspect_field import inspect_field

import warp as wp


def main() -> None:
    """Print the inspection report."""
    if wp.get_cuda_device_count() == 0:
        raise RuntimeError("This program requires a CUDA device, but Warp sees none.")

    report = inspect_field()
    for name in sorted(report):
        print(f"{name}: {report[name]:.6f}")


if __name__ == "__main__":
    main()
