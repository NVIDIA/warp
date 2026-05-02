# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test and trial field types for finite element formulations.

This module provides field types used in FEM weak formulations for defining bilinear
and linear forms, local operations, and restricting fields to subdomains.
"""

# isort: skip_file

from warp._src.fem.field import FieldRestriction as FieldRestriction
from warp._src.fem.field import LocalTestField as LocalTestField
from warp._src.fem.field import TestField as TestField
from warp._src.fem.field import TrialField as TrialField
