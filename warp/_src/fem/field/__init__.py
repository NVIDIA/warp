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

from typing import Optional, Union

from warp.fem.domain import Cells, GeometryDomain
from warp.fem.space import FunctionSpace, SpacePartition, SpaceRestriction, make_space_partition, make_space_restriction

from .field import DiscreteField, FieldLike, GeometryField, ImplicitField, NonconformingField, SpaceField, UniformField
from .nodal_field import NodalField
from .restriction import FieldRestriction
from .virtual import LocalTestField, LocalTrialField, TestField, TrialField


def make_restriction(
    field: DiscreteField,
    space_restriction: Optional[SpaceRestriction] = None,
    domain: Optional[GeometryDomain] = None,
    device=None,
) -> FieldRestriction:
    """
    Restricts a discrete field to a subset of elements.

    Args:
        field: the discrete field to restrict
        space_restriction: the function space restriction defining the subset of elements to consider
        domain: if ``space_restriction`` is not provided, the :py:class:`Domain` defining the subset of elements to consider
        device: Warp device on which to perform and store computations

    Returns:
        the field restriction
    """

    if space_restriction is None:
        space_restriction = make_space_restriction(space_partition=field.space_partition, domain=domain, device=device)

    return FieldRestriction(field=field, space_restriction=space_restriction)


def make_test(
    space: FunctionSpace,
    space_restriction: Optional[SpaceRestriction] = None,
    space_partition: Optional[SpacePartition] = None,
    domain: Optional[GeometryDomain] = None,
    device=None,
) -> TestField:
    """
    Constructs a test field over a function space or its restriction

    Args:
        space: the function space
        space_restriction: restriction of the space topology to a domain
        space_partition: if `space_restriction` is ``None``, the optional subset of node indices to consider
        domain: if `space_restriction` is ``None``, optional subset of elements to consider
        device: Warp device on which to perform and store computations

    Returns:
        the test field
    """

    if space_restriction is None:
        space_restriction = make_space_restriction(
            space_topology=space.topology, space_partition=space_partition, domain=domain, device=device
        )

    return TestField(space_restriction=space_restriction, space=space)


def make_trial(
    space: FunctionSpace,
    space_restriction: Optional[SpaceRestriction] = None,
    space_partition: Optional[SpacePartition] = None,
    domain: Optional[GeometryDomain] = None,
) -> TrialField:
    """
    Constructs a trial field over a function space or partition

    Args:
        space: the function space or function space restriction
        space_restriction: restriction of the space topology to a domain
        space_partition: if `space_restriction` is ``None``, the optional subset of node indices to consider
        domain: if `space_restriction` is ``None``, optional subset of elements to consider
        device: Warp device on which to perform and store computations

    Returns:
        the trial field
    """

    if space_restriction is not None:
        domain = space_restriction.domain
        space_partition = space_restriction.space_partition

    if space_partition is None:
        if domain is None:
            domain = Cells(geometry=space.geometry)
        space_partition = make_space_partition(
            space_topology=space.topology, geometry_partition=domain.geometry_partition
        )
    elif domain is None:
        domain = Cells(geometry=space_partition.geo_partition)

    return TrialField(space, space_partition, domain)


def make_discrete_field(
    space: FunctionSpace,
    space_partition: Optional[SpacePartition] = None,
) -> DiscreteField:
    """Constructs  a zero-initialized discrete field over a function space or partition

    See also: :meth:`warp.fem.FunctionSpace.make_field`
    """
    return space.make_field(space_partition=space_partition)
