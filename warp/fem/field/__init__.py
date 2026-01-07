# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
