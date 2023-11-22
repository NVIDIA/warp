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
