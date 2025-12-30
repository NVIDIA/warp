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

"""Function space types for finite element discretizations.

This module provides specialized function space classes for vector-valued fields with
different transformation behaviors (collocated, contravariant, and covariant). The
:mod:`.shape` submodule contains shape function definitions.
"""

# isort: skip_file

from warp._src.fem.space import CollocatedFunctionSpace as CollocatedFunctionSpace
from warp._src.fem.space import ContravariantFunctionSpace as ContravariantFunctionSpace
from warp._src.fem.space import CovariantFunctionSpace as CovariantFunctionSpace

from . import shape as shape


# TODO: Remove after cleaning up the public API.

from warp._src.fem import space as _space


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_space, "warp.fem", name)
