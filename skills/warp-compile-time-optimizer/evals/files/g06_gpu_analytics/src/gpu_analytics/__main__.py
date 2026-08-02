# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one deterministic GPU analytics sample."""

from __future__ import annotations

import numpy as np

import warp as wp

from .analytics import AnalyticsBank


def main() -> None:
    """Print the transformed array shapes for two numeric series."""
    if wp.get_cuda_device_count() == 0:
        raise RuntimeError("This program requires a CUDA device, but Warp sees none.")

    values32 = np.arange(16, dtype=np.float32) / np.float32(7.0)
    values64 = np.arange(16, dtype=np.float64) / np.float64(11.0)
    result = AnalyticsBank(device="cuda:0").run((values32, values64))
    print(tuple(item.shape for item in result.transformed))


if __name__ == "__main__":
    main()
