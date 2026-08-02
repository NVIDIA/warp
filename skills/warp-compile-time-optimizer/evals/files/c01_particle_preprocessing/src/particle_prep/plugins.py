# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional particle value transformations."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class Plugin:
    """Describe an optional particle transform.

    Args:
        kind: Either ``"bias"`` or ``"square"``.
        amount: The value added by the ``"bias"`` transform.
    """

    kind: str
    amount: float = 0.0

    def make_kernel(self) -> object:
        """Build the transform for this plugin."""
        if self.kind == "bias":

            @wp.kernel(module="unique")
            def bias_kernel(values: wp.array[float], amount: float) -> None:
                index = wp.tid()
                values[index] = values[index] + amount

            return bias_kernel
        if self.kind == "square":

            @wp.kernel(module="unique")
            def square_kernel(values: wp.array[float], amount: float) -> None:
                index = wp.tid()
                values[index] = values[index] * values[index]

            return square_kernel
        raise ValueError(f"unsupported particle plugin: {self.kind}")
