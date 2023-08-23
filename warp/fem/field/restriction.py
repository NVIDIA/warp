from warp.fem.space import SpaceRestriction
from .discrete_field import DiscreteField


class FieldRestriction:
    """Restriction of a space to a given GeometryDomain"""

    def __init__(self, space_restriction: SpaceRestriction, field: DiscreteField):

        if field.space.DIMENSION - 1 == space_restriction.space.DIMENSION:
            field = field.trace()

        if field.space.DIMENSION != space_restriction.space.DIMENSION:
            raise ValueError("Incompatible space and field dimensions")

        self.space_restriction = space_restriction
        self.space = self.space_restriction.space
        self.domain = self.space_restriction.domain
        self.field = field
