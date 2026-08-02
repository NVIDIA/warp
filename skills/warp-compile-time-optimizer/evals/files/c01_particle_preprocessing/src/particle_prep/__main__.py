# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a deterministic particle preprocessing example."""

from __future__ import annotations

import numpy as np

from .pipeline import ParticlePipeline
from .plugins import Plugin


def main() -> None:
    """Print the result of a fixed particle preprocessing workload."""
    pipeline = ParticlePipeline(
        center=0.25,
        scale=1.5,
        lower=-0.8,
        upper=0.9,
        plugins=(Plugin("bias", 0.125),),
        device="cpu",
    )
    values = np.array([-1.0, 0.0, 0.25, 1.0], dtype=np.float32)
    print(pipeline.run(values))


if __name__ == "__main__":
    main()
