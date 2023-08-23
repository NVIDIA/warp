from typing import Union, Optional

from warp.fem.domain import GeometryDomain, Cells
from warp.fem.space import FunctionSpace, SpaceRestriction, SpacePartition, make_space_partition, make_space_restriction

from .discrete_field import DiscreteField
from .restriction import FieldRestriction
from .test import TestField
from .trial import TrialField

from .nodal_field import NodalField

FieldLike = Union[DiscreteField, FieldRestriction, TestField, TrialField]


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
        space_restriction = make_space_restriction(
            space=field.space, space_partition=field.space_partition, domain=domain, device=device
        )

    return FieldRestriction(field=field, space_restriction=space_restriction)


def make_test(
    space: Union[FunctionSpace, SpaceRestriction] = None,
    space_partition: SpacePartition = None,
    domain: GeometryDomain = None,
    device=None,
) -> TestField:
    """
    Constructs a test field over a function space or its restriction

    Args:
        space: the function space or function space restriction
        space_partition: if ``space`` is a whole function space, the optional subset of node indices to consider
        domain: if ``space`` is a whole function space, the optional subset of elements to consider
        device: Warp device on which to perform and store computations

    Returns:
        the test field
    """

    if not isinstance(space, SpaceRestriction):
        if space is None:
            space = space_partition.space

        if domain is None:
            domain = Cells(geometry=space.geometry)

        if space_partition is None:
            space_partition = make_space_partition(space, domain.geometry_partition)

        space = make_space_restriction(space=space, space_partition=space_partition, domain=domain, device=device)

    return TestField(space)


def make_trial(
    space: Union[FunctionSpace, SpaceRestriction] = None,
    space_partition: SpacePartition = None,
    domain: GeometryDomain = None,
) -> TrialField:
    """
    Constructs a trial field over a function space or partition

    Args:
        space: the function space or function space restriction
        space_partition: if ``space`` is a whole function space, the optional subset of node indices to consider
        domain: if ``space`` is a whole function space, the optional subset of elements to consider
        device: Warp device on which to perform and store computations

    Returns:
        the trial field
    """

    if isinstance(space, SpaceRestriction):
        domain = space.domain
        space = space.space
        space_partition = space.space_partition

    if space is None:
        space = space_partition.space

    if domain is None:
        domain = Cells(geometry=space.geometry)

    if space_partition is None:
        space_partition = make_space_partition(space, domain.geometry_partition)

    return TrialField(space, space_partition, domain)
