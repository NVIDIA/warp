# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic GPU analytics for paired floating-point series."""

from .analytics import AnalyticsBank, AnalyticsResult
from .diagnostics import diagnose

__all__ = ["AnalyticsBank", "AnalyticsResult", "diagnose"]
