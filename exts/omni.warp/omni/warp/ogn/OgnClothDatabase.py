"""Support for simplified access to data on nodes of type omni.warp.OgnCloth

Cloth simulation node
"""

import omni.graph.core as og
from contextlib import suppress
from inspect import getfullargspec
class OgnClothDatabase(og.Database):
    """Helper class providing simplified access to data on nodes of type omni.warp.OgnCloth

    Class Members:
        node: Node being evaluated

    Attribute Value Properties:
        Inputs:
            inputs.cloth_indices
            inputs.cloth_positions
            inputs.cloth_transform
            inputs.collider_indices
            inputs.collider_offset
            inputs.collider_positions
            inputs.collider_transform
            inputs.density
            inputs.gravity
            inputs.ground
            inputs.ground_plane
            inputs.k_contact_damp
            inputs.k_contact_elastic
            inputs.k_contact_friction
            inputs.k_contact_mu
            inputs.k_edge_bend
            inputs.k_edge_damp
            inputs.k_tri_area
            inputs.k_tri_damp
            inputs.k_tri_elastic
            inputs.num_substeps
        Outputs:
            outputs.positions
    """
    # This is an internal object that provides per-class storage of a per-node data dictionary
    PER_NODE_DATA = {}
    # This is an internal object that describes unchanging attributes in a generic way
    # The values in this list are in no particular order, as a per-attribute tuple
    #     Name, Type, ExtendedTypeIndex, UiName, Description, Metadata, Is_Required, DefaultValue
    # You should not need to access any of this data directly, use the defined database interfaces
    INTERFACE = og.Database._get_interface([
        ('inputs:cloth_indices', 'int[]', 0, None, 'Particle indices', {}, True, []),
        ('inputs:cloth_positions', 'point3f[]', 0, None, 'Particle positions', {}, True, []),
        ('inputs:cloth_transform', 'frame4d', 0, None, 'Local to World transform for cloth vertices', {}, True, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        ('inputs:collider_indices', 'int[]', 0, None, 'Collision indices', {}, True, []),
        ('inputs:collider_offset', 'float', 0, None, '', {}, True, 0.01),
        ('inputs:collider_positions', 'point3f[]', 0, None, 'Collision positions', {}, True, []),
        ('inputs:collider_transform', 'frame4d', 0, None, 'Local to world transform for collider vertices', {}, True, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        ('inputs:density', 'float', 0, None, '', {}, True, 100.0),
        ('inputs:gravity', 'vector3f', 0, None, '', {}, True, [0.0, -9.8, 0.0]),
        ('inputs:ground', 'bool', 0, None, '', {}, True, False),
        ('inputs:ground_plane', 'vector3f', 0, None, '', {}, True, [0.0, 1.0, 0.0]),
        ('inputs:k_contact_damp', 'float', 0, None, '', {}, True, 100.0),
        ('inputs:k_contact_elastic', 'float', 0, None, '', {}, True, 5000.0),
        ('inputs:k_contact_friction', 'float', 0, None, '', {}, True, 2000.0),
        ('inputs:k_contact_mu', 'float', 0, None, '', {}, True, 0.75),
        ('inputs:k_edge_bend', 'float', 0, None, '', {}, True, 1.0),
        ('inputs:k_edge_damp', 'float', 0, None, '', {}, True, 0.0),
        ('inputs:k_tri_area', 'float', 0, None, '', {}, True, 1000.0),
        ('inputs:k_tri_damp', 'float', 0, None, '', {}, True, 10.0),
        ('inputs:k_tri_elastic', 'float', 0, None, '', {}, True, 1000.0),
        ('inputs:num_substeps', 'int', 0, None, '', {}, True, 32),
        ('outputs:positions', 'point3f[]', 0, None, 'Particle positions', {}, True, None),
    ])
    @classmethod
    def _populate_role_data(cls):
        """Populate a role structure with the non-default roles on this node type"""
        role_data = super()._populate_role_data()
        role_data.inputs.cloth_positions = og.Database.ROLE_POINT
        role_data.inputs.cloth_transform = og.Database.ROLE_TRANSFORM
        role_data.inputs.collider_positions = og.Database.ROLE_POINT
        role_data.inputs.collider_transform = og.Database.ROLE_TRANSFORM
        role_data.inputs.gravity = og.Database.ROLE_VECTOR
        role_data.inputs.ground_plane = og.Database.ROLE_VECTOR
        role_data.outputs.positions = og.Database.ROLE_POINT
        return role_data
    class ValuesForInputs:
        """Helper class that creates natural hierarchical access to input attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes):
            """Initialize simplified access for the attribute data"""
            self.context_helper = context_helper
            self.node = node
            self.attributes = attributes
            self.setting_locked = False

        @property
        def cloth_indices(self):
            return self.context_helper.get_attr_value(self.attributes.cloth_indices)

        @cloth_indices.setter
        def cloth_indices(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.cloth_indices)
            self.context_helper.set_attr_value(value, self.attributes.cloth_indices)
            self.cloth_indices_size = self.context_helper.get_elem_count(self.attributes.cloth_indices)

        @property
        def cloth_positions(self):
            return self.context_helper.get_attr_value(self.attributes.cloth_positions)

        @cloth_positions.setter
        def cloth_positions(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.cloth_positions)
            self.context_helper.set_attr_value(value, self.attributes.cloth_positions)
            self.cloth_positions_size = self.context_helper.get_elem_count(self.attributes.cloth_positions)

        @property
        def cloth_transform(self):
            return self.context_helper.get_attr_value(self.attributes.cloth_transform)

        @cloth_transform.setter
        def cloth_transform(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.cloth_transform)
            self.context_helper.set_attr_value(value, self.attributes.cloth_transform)

        @property
        def collider_indices(self):
            return self.context_helper.get_attr_value(self.attributes.collider_indices)

        @collider_indices.setter
        def collider_indices(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.collider_indices)
            self.context_helper.set_attr_value(value, self.attributes.collider_indices)
            self.collider_indices_size = self.context_helper.get_elem_count(self.attributes.collider_indices)

        @property
        def collider_offset(self):
            return self.context_helper.get_attr_value(self.attributes.collider_offset)

        @collider_offset.setter
        def collider_offset(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.collider_offset)
            self.context_helper.set_attr_value(value, self.attributes.collider_offset)

        @property
        def collider_positions(self):
            return self.context_helper.get_attr_value(self.attributes.collider_positions)

        @collider_positions.setter
        def collider_positions(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.collider_positions)
            self.context_helper.set_attr_value(value, self.attributes.collider_positions)
            self.collider_positions_size = self.context_helper.get_elem_count(self.attributes.collider_positions)

        @property
        def collider_transform(self):
            return self.context_helper.get_attr_value(self.attributes.collider_transform)

        @collider_transform.setter
        def collider_transform(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.collider_transform)
            self.context_helper.set_attr_value(value, self.attributes.collider_transform)

        @property
        def density(self):
            return self.context_helper.get_attr_value(self.attributes.density)

        @density.setter
        def density(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.density)
            self.context_helper.set_attr_value(value, self.attributes.density)

        @property
        def gravity(self):
            return self.context_helper.get_attr_value(self.attributes.gravity)

        @gravity.setter
        def gravity(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.gravity)
            self.context_helper.set_attr_value(value, self.attributes.gravity)

        @property
        def ground(self):
            return self.context_helper.get_attr_value(self.attributes.ground)

        @ground.setter
        def ground(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.ground)
            self.context_helper.set_attr_value(value, self.attributes.ground)

        @property
        def ground_plane(self):
            return self.context_helper.get_attr_value(self.attributes.ground_plane)

        @ground_plane.setter
        def ground_plane(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.ground_plane)
            self.context_helper.set_attr_value(value, self.attributes.ground_plane)

        @property
        def k_contact_damp(self):
            return self.context_helper.get_attr_value(self.attributes.k_contact_damp)

        @k_contact_damp.setter
        def k_contact_damp(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_contact_damp)
            self.context_helper.set_attr_value(value, self.attributes.k_contact_damp)

        @property
        def k_contact_elastic(self):
            return self.context_helper.get_attr_value(self.attributes.k_contact_elastic)

        @k_contact_elastic.setter
        def k_contact_elastic(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_contact_elastic)
            self.context_helper.set_attr_value(value, self.attributes.k_contact_elastic)

        @property
        def k_contact_friction(self):
            return self.context_helper.get_attr_value(self.attributes.k_contact_friction)

        @k_contact_friction.setter
        def k_contact_friction(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_contact_friction)
            self.context_helper.set_attr_value(value, self.attributes.k_contact_friction)

        @property
        def k_contact_mu(self):
            return self.context_helper.get_attr_value(self.attributes.k_contact_mu)

        @k_contact_mu.setter
        def k_contact_mu(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_contact_mu)
            self.context_helper.set_attr_value(value, self.attributes.k_contact_mu)

        @property
        def k_edge_bend(self):
            return self.context_helper.get_attr_value(self.attributes.k_edge_bend)

        @k_edge_bend.setter
        def k_edge_bend(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_edge_bend)
            self.context_helper.set_attr_value(value, self.attributes.k_edge_bend)

        @property
        def k_edge_damp(self):
            return self.context_helper.get_attr_value(self.attributes.k_edge_damp)

        @k_edge_damp.setter
        def k_edge_damp(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_edge_damp)
            self.context_helper.set_attr_value(value, self.attributes.k_edge_damp)

        @property
        def k_tri_area(self):
            return self.context_helper.get_attr_value(self.attributes.k_tri_area)

        @k_tri_area.setter
        def k_tri_area(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_tri_area)
            self.context_helper.set_attr_value(value, self.attributes.k_tri_area)

        @property
        def k_tri_damp(self):
            return self.context_helper.get_attr_value(self.attributes.k_tri_damp)

        @k_tri_damp.setter
        def k_tri_damp(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_tri_damp)
            self.context_helper.set_attr_value(value, self.attributes.k_tri_damp)

        @property
        def k_tri_elastic(self):
            return self.context_helper.get_attr_value(self.attributes.k_tri_elastic)

        @k_tri_elastic.setter
        def k_tri_elastic(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.k_tri_elastic)
            self.context_helper.set_attr_value(value, self.attributes.k_tri_elastic)

        @property
        def num_substeps(self):
            return self.context_helper.get_attr_value(self.attributes.num_substeps)

        @num_substeps.setter
        def num_substeps(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.num_substeps)
            self.context_helper.set_attr_value(value, self.attributes.num_substeps)
    class ValuesForOutputs:
        """Helper class that creates natural hierarchical access to output attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes):
            """Initialize simplified access for the attribute data"""
            self.context_helper = context_helper
            self.node = node
            self.attributes = attributes
            self.positions_size = None

        @property
        def positions(self):
            return self.context_helper.get_attr_value(self.attributes.positions,getForWrite=True,writeElemCount=self.positions_size)

        @positions.setter
        def positions(self, value):
            self.context_helper.set_attr_value(value, self.attributes.positions)
            self.positions_size = self.context_helper.get_elem_count(self.attributes.positions)
    def __init__(self, context_helper, node):
        try:
            super().__init__(node, context_helper)
            self.inputs = OgnClothDatabase.ValuesForInputs(self.context_helper, node, self.attributes.inputs)
            self.outputs = OgnClothDatabase.ValuesForOutputs(self.context_helper, node, self.attributes.outputs)
        except AttributeError:
            self.log_hot_reload_problem(node)
    class abi:
        @staticmethod
        def get_node_type():
            get_node_type_function = getattr(OgnClothDatabase.NODE_TYPE_CLASS, 'get_node_type', None)
            if callable(get_node_type_function):
                return get_node_type_function()
            return 'omni.warp.OgnCloth'
        @staticmethod
        def compute(context_helper, node):
            db = OgnClothDatabase(context_helper, node)
            try:
                compute_function = getattr(OgnClothDatabase.NODE_TYPE_CLASS, 'compute', None)
                if callable(compute_function) and compute_function.__code__.co_argcount > 1:
                    return compute_function(context_helper, node)
                with suppress(AttributeError):
                    db.inputs.setting_locked = True
                try:
                    x = db.inputs
                    x = db.outputs
                except AttributeError:
                    db.log_hot_reload_problem(node)
                    return False
                return OgnClothDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:
                db.log_error(f'Assertion raised in compute - {error}')
            return False
        @staticmethod
        def initialize(context_helper, node):
            OgnClothDatabase._initialize_per_node_data(node)

            # Set any default values the attributes have specified
            db = OgnClothDatabase(context_helper, node)
            db.inputs.cloth_indices = []
            db.inputs.cloth_positions = []
            db.inputs.cloth_transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            db.inputs.collider_indices = []
            db.inputs.collider_offset = 0.01
            db.inputs.collider_positions = []
            db.inputs.collider_transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            db.inputs.density = 100.0
            db.inputs.gravity = [0.0, -9.8, 0.0]
            db.inputs.ground = False
            db.inputs.ground_plane = [0.0, 1.0, 0.0]
            db.inputs.k_contact_damp = 100.0
            db.inputs.k_contact_elastic = 5000.0
            db.inputs.k_contact_friction = 2000.0
            db.inputs.k_contact_mu = 0.75
            db.inputs.k_edge_bend = 1.0
            db.inputs.k_edge_damp = 0.0
            db.inputs.k_tri_area = 1000.0
            db.inputs.k_tri_damp = 10.0
            db.inputs.k_tri_elastic = 1000.0
            db.inputs.num_substeps = 32
            initialize_function = getattr(OgnClothDatabase.NODE_TYPE_CLASS, 'initialize', None)
            if callable(initialize_function):
                initialize_function(context_helper, node)
        @staticmethod
        def release(node):
            release_function = getattr(OgnClothDatabase.NODE_TYPE_CLASS, 'release', None)
            if callable(release_function):
                release_function(node)
            OgnClothDatabase._release_per_node_data(node)
        @staticmethod
        def update_node_version(context, node, old_version, new_version):
            update_node_version_function = getattr(OgnClothDatabase.NODE_TYPE_CLASS, 'update_node_version', None)
            if callable(update_node_version_function):
                return update_node_version_function(context, node, old_version, new_version)
            return False
        @staticmethod
        def initialize_type(node_type):
            initialize_type_function = getattr(OgnClothDatabase.NODE_TYPE_CLASS, 'initialize_type', None)
            needs_initializing = True
            if callable(initialize_type_function):
                needs_initializing = initialize_type_function(node_type)
            if needs_initializing:
                node_type.set_metadata('__extension', "omni.warp")
                node_type.set_metadata("__description", "Cloth simulation node")
                OgnClothDatabase.INTERFACE.add_to_node_type(node_type)
                node_type.set_has_state(True)
        @staticmethod
        def on_connection_type_resolve(node):
            on_connection_type_resolve_function = getattr(OgnClothDatabase.NODE_TYPE_CLASS, 'on_connection_type_resolve', None)
            if callable(on_connection_type_resolve_function):
                on_connection_type_resolve_function(node)
    NODE_TYPE_CLASS = None
    @staticmethod
    def register(node_type_class):
        OgnClothDatabase.NODE_TYPE_CLASS = node_type_class
        og.register_node_type(OgnClothDatabase.abi, 1)
    @staticmethod
    def deregister():
        og.deregister_node_type("omni.warp.OgnCloth")
