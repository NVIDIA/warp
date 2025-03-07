# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import warp as wp
from warp.fem.types import Coords


@wp.func
def project_on_seg_at_origin(q: Any, seg: Any, len_sq: float):
    s = wp.clamp(wp.dot(q, seg) / len_sq, 0.0, 1.0)
    return wp.length_sq(q - s * seg), s


@wp.func
def project_on_tri_at_origin(q: Any, e1: Any, e2: Any):
    e1e1 = wp.dot(e1, e1)
    e1e2 = wp.dot(e1, e2)
    e2e2 = wp.dot(e2, e2)

    det = e1e1 * e2e2 - e1e2 * e1e2

    if det > e1e1 * e2e2 * 1.0e-6:
        e1p = wp.dot(e1, q)
        e2p = wp.dot(e2, q)

        s = (e2e2 * e1p - e1e2 * e2p) / det
        t = (e1e1 * e2p - e1e2 * e1p) / det

        if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
            # point inside triangle (distance can be non-zero in 3D case)
            return wp.length_sq(q - s * e1 - t * e2), Coords(1.0 - s - t, s, t)

    d1, s1 = project_on_seg_at_origin(q, e1, e1e1)
    d2, s2 = project_on_seg_at_origin(q, e2, e2e2)
    d12, s12 = project_on_seg_at_origin(q - e1, e2 - e1, wp.length_sq(e2 - e1))

    if d1 <= d2:
        if d1 <= d12:
            return d1, Coords(1.0 - s1, s1, 0.0)
    elif d2 <= d12:
        return d2, Coords(1.0 - s2, 0.0, s2)

    return d12, Coords(0.0, 1.0 - s12, s12)


@wp.func
def project_on_tet_at_origin(q: wp.vec3, e1: wp.vec3, e2: wp.vec3, e3: wp.vec3):
    mat = wp.inverse(wp.matrix_from_cols(e1, e2, e3))
    coords = mat * q

    if wp.min(coords) >= 0.0 and coords[0] + coords[1] + coords[2] <= 1.0:
        return 0.0, coords

    # Not inside tet, compare closest point on each tri

    d12, s12 = project_on_tri_at_origin(q, e1, e2)
    d23, s23 = project_on_tri_at_origin(q, e2, e3)
    d31, s31 = project_on_tri_at_origin(q, e3, e1)
    d123, s123 = project_on_tri_at_origin(q - e1, e2 - e1, e3 - e1)

    dmin = wp.min(wp.vec4(d12, d23, d31, d123))

    if dmin == d12:
        return dmin, Coords(s12[1], s12[2], 0.0)
    elif dmin == d23:
        return dmin, Coords(0.0, s23[1], s23[2])
    elif dmin == d31:
        return dmin, Coords(s31[2], 0.0, s31[1])
    else:
        return dmin, s123
