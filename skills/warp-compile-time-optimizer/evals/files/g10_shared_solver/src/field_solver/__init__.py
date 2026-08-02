# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared scalar-field operators with two consumers.

``field_solver.kernels`` is imported by both :mod:`field_solver.inspect_field`
and :mod:`field_solver.calibrate`, so its module options apply to both.
"""

from __future__ import annotations

__all__ = ["calibrate", "inspect_field", "kernels"]
