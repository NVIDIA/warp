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

from warp.fem.space import SpaceRestriction

from .field import DiscreteField


class FieldRestriction:
    """Restriction of a discrete field to a given GeometryDomain"""

    def __init__(self, space_restriction: SpaceRestriction, field: DiscreteField):
        if field.space.dimension - 1 == space_restriction.space_topology.dimension:
            field = field.trace()

        if field.space.dimension != space_restriction.space_topology.dimension:
            raise ValueError("Incompatible space and field dimensions")

        if field.space.topology != space_restriction.space_topology:
            raise ValueError("Incompatible field and space restriction topologies")

        self.space_restriction = space_restriction
        self.domain = self.space_restriction.domain
        self.field = field
        self.space = self.field.space
