# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-dimensional finite-difference stepping with selectable boundaries."""

from .solver import PDEStepper

__all__ = ["PDEStepper"]
