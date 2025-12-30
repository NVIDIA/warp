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

"""Type introspection and construction utilities.

This module provides functions for runtime type checking, creating custom vector,
matrix, quaternion, and transformation types, and querying type properties.
"""

# isort: skip_file

from warp._src.types import is_array as is_array
from warp._src.types import is_composite as is_composite
from warp._src.types import is_float as is_float
from warp._src.types import is_int as is_int
from warp._src.types import is_matrix as is_matrix
from warp._src.types import is_quaternion as is_quaternion
from warp._src.types import is_scalar as is_scalar
from warp._src.types import is_struct as is_struct
from warp._src.types import is_tile as is_tile
from warp._src.types import is_transformation as is_transformation
from warp._src.types import is_value as is_value
from warp._src.types import is_vector as is_vector
from warp._src.types import matrix as matrix
from warp._src.types import quaternion as quaternion
from warp._src.types import transformation as transformation
from warp._src.types import type_ctype as type_ctype
from warp._src.types import type_is_array as type_is_array
from warp._src.types import type_is_composite as type_is_composite
from warp._src.types import type_is_float as type_is_float
from warp._src.types import type_is_int as type_is_int
from warp._src.types import type_is_matrix as type_is_matrix
from warp._src.types import type_is_quaternion as type_is_quaternion
from warp._src.types import type_is_scalar as type_is_scalar
from warp._src.types import type_is_struct as type_is_struct
from warp._src.types import type_is_tile as type_is_tile
from warp._src.types import type_is_transformation as type_is_transformation
from warp._src.types import type_is_value as type_is_value
from warp._src.types import type_is_vector as type_is_vector
from warp._src.types import type_repr as type_repr
from warp._src.types import type_size as type_size
from warp._src.types import type_size_in_bytes as type_size_in_bytes
from warp._src.types import types_equal as types_equal
from warp._src.types import vector as vector


# TODO: Remove after cleaning up the public API.

from warp._src import types as _types


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_types, "warp", name)
