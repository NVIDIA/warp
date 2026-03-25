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


# TODO: Remove after cleaning up the public API.

from warp._src.fem import field as _field


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_field, "warp.fem", name)
