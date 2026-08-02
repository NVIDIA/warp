# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one tiled projection session, or relink device code for operators."""

from __future__ import annotations

import sys

import numpy as np

import warp as wp

from .maintenance import rebuild_device_code
from .session import run_session


def main() -> None:
    """Run one session, or honor the operator --relink flag."""
    # --relink only clears caches on disk, so it stays available on a machine
    # with no device. Everything past this point runs on the GPU.
    if "--relink" in sys.argv[1:]:
        rebuild_device_code()
        print("device code relinked")
        return

    if wp.get_cuda_device_count() == 0:
        raise RuntimeError(
            "This program requires a CUDA device, but Warp sees none. "
            "Its projection stages run on tile matmul, which has no CPU path."
        )

    features = (np.arange(64, dtype=np.float32).reshape(8, 8) / np.float32(9.0)).copy()
    basis = (np.arange(64, dtype=np.float32).reshape(8, 8)[::-1] / np.float32(11.0)).copy()
    coefficients, rebuilt = run_session(features, basis)
    print(f"coefficients {coefficients.shape} sum {coefficients.sum():.4f}")
    print(f"rebuilt {rebuilt.shape} sum {rebuilt.sum():.4f}")


if __name__ == "__main__":
    main()
