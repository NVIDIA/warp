# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Condition one recorded batch, or rebuild kernels for operators."""

from __future__ import annotations

import sys

import numpy as np

from .conditioning import condition_batch
from .maintenance import rebuild_kernels


def main() -> None:
    """Run one batch, or honor the operator --rebuild flag."""
    if "--rebuild" in sys.argv[1:]:
        rebuild_kernels()
        print("kernels rebuilt")
        return
    samples = (np.arange(64, dtype=np.float32) - np.float32(31.5)) / np.float32(8.0)
    conditioned = condition_batch(samples, threshold=0.35, gain=1.25)
    print(f"{conditioned.size} samples, sum {conditioned.sum():.4f}")


if __name__ == "__main__":
    main()
