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

# isort: skip_file

from warp._src.types import ARRAY_MAX_DIMS as ARRAY_MAX_DIMS
from warp._src.types import matrix as matrix
from warp._src.types import type_is_matrix as type_is_matrix
from warp._src.types import type_size_in_bytes as type_size_in_bytes
from warp._src.types import vector as vector
from warp._src.types import warp_type_to_np_dtype as warp_type_to_np_dtype

# Needed for `get_warp_type_from_data_type_name()` in `omni.warp.nodes`.
from warp._src.types import int8 as int8
from warp._src.types import int32 as int32
from warp._src.types import int64 as int64
from warp._src.types import uint8 as uint8
from warp._src.types import uint64 as uint64
from warp._src.types import uint32 as uint32
from warp._src.types import float32 as float32
from warp._src.types import float64 as float64
from warp._src.types import mat22d as mat22d
from warp._src.types import mat33d as mat33d
from warp._src.types import mat44d as mat44d
from warp._src.types import quat as quat
from warp._src.types import vec2 as vec2
from warp._src.types import vec3 as vec3
from warp._src.types import vec4 as vec4


# TODO: Remove after cleaning up the public API.

from warp._src import types as _types


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_types, "wp", name)
