# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a deterministic image pyramid example."""

from __future__ import annotations

import numpy as np

import warp as wp

from .pyramid import ImagePyramid


def main() -> None:
    """Print the shapes of three filtered image levels."""
    if wp.get_cuda_device_count() == 0:
        raise RuntimeError("This program requires a CUDA device, but Warp sees none.")

    image = np.arange(63, dtype=np.float32).reshape(7, 9)
    print(tuple(level.shape for level in ImagePyramid(device="cuda:0").build(image)))


if __name__ == "__main__":
    main()
