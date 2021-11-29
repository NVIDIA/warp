"""Support for simplified access to data on nodes of type omni.warp.OgnParticleSolver

Particle simulation node
"""

import omni.graph.core as og
import sys
import traceback
class OgnParticleSolverDatabase(og.Database):
    """Helper class providing simplified access to data on nodes of type omni.warp.OgnParticleSolver

    Class Members:
        node: Node being evaluated

    Attribute Value Properties:
        Inputs:
            inputs.collider
            inputs.collider_offset
            inputs.gravity
            inputs.ground
            inputs.ground_plane
            inputs.k_contact_adhesion
            inputs.k_contact_cohesion
            inputs.k_contact_damp
            inputs.k_contact_elastic
            inputs.k_contact_friction
            inputs.k_contact_mu
            inputs.mass
            inputs.num_substeps
            inputs.positions
            inputs.radius
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
        ('inputs:collider', 'bundle', 0, None, 'Collision Prim', {}, True, None),
        ('inputs:collider_offset', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '0.01'}, True, 0.01),
        ('inputs:gravity', 'vector3f', 0, None, '', {og.MetadataKeys.DEFAULT: '[0.0, -9.8, 0.0]'}, True, [0.0, -9.8, 0.0]),
        ('inputs:ground', 'bool', 0, None, '', {og.MetadataKeys.DEFAULT: 'false'}, True, False),
        ('inputs:ground_plane', 'vector3f', 0, None, '', {og.MetadataKeys.DEFAULT: '[0.0, 1.0, 0.0]'}, True, [0.0, 1.0, 0.0]),
        ('inputs:k_contact_adhesion', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '0.75'}, True, 0.75),
        ('inputs:k_contact_cohesion', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '0.75'}, True, 0.75),
        ('inputs:k_contact_damp', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '100.0'}, True, 100.0),
        ('inputs:k_contact_elastic', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '5000.0'}, True, 5000.0),
        ('inputs:k_contact_friction', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '2000.0'}, True, 2000.0),
        ('inputs:k_contact_mu', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '0.75'}, True, 0.75),
        ('inputs:mass', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '1.0'}, True, 1.0),
        ('inputs:num_substeps', 'int', 0, None, '', {og.MetadataKeys.DEFAULT: '32'}, True, 32),
        ('inputs:positions', 'point3f[]', 0, None, 'Particle positions', {og.MetadataKeys.DEFAULT: '[]'}, True, []),
        ('inputs:radius', 'float', 0, None, '', {og.MetadataKeys.DEFAULT: '10.0'}, True, 10.0),
        ('outputs:positions', 'point3f[]', 0, None, 'Particle positions', {}, True, None),
    ])
    @classmethod
    def _populate_role_data(cls):
        """Populate a role structure with the non-default roles on this node type"""
        role_data = super()._populate_role_data()
        role_data.inputs.gravity = og.Database.ROLE_VECTOR
        role_data.inputs.ground_plane = og.Database.ROLE_VECTOR
        role_data.inputs.positions = og.Database.ROLE_POINT
        role_data.outputs.positions = og.Database.ROLE_POINT
        return role_data
    class ValuesForInputs(og.DynamicAttributeAccess):
        """Helper class that creates natural hierarchical access to input attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            super().__init__(context_helper, node, attributes, dynamic_attributes)
            self.__bundles = og.BundleContainer(context_helper.context, node, attributes, [], read_only=True)

        @property
        def collider(self) -> og.BundleContents:
            """Get the bundle wrapper class for the attribute inputs.collider"""
            return self.__bundles.collider

        @property
        def collider_offset(self):
            return self._context_helper.get(self._attributes.collider_offset)

        @collider_offset.setter
        def collider_offset(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.collider_offset)
            self._context_helper.set_attr_value(value, self._attributes.collider_offset)

        @property
        def gravity(self):
            return self._context_helper.get(self._attributes.gravity)

        @gravity.setter
        def gravity(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.gravity)
            self._context_helper.set_attr_value(value, self._attributes.gravity)

        @property
        def ground(self):
            return self._context_helper.get(self._attributes.ground)

        @ground.setter
        def ground(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.ground)
            self._context_helper.set_attr_value(value, self._attributes.ground)

        @property
        def ground_plane(self):
            return self._context_helper.get(self._attributes.ground_plane)

        @ground_plane.setter
        def ground_plane(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.ground_plane)
            self._context_helper.set_attr_value(value, self._attributes.ground_plane)

        @property
        def k_contact_adhesion(self):
            return self._context_helper.get(self._attributes.k_contact_adhesion)

        @k_contact_adhesion.setter
        def k_contact_adhesion(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.k_contact_adhesion)
            self._context_helper.set_attr_value(value, self._attributes.k_contact_adhesion)

        @property
        def k_contact_cohesion(self):
            return self._context_helper.get(self._attributes.k_contact_cohesion)

        @k_contact_cohesion.setter
        def k_contact_cohesion(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.k_contact_cohesion)
            self._context_helper.set_attr_value(value, self._attributes.k_contact_cohesion)

        @property
        def k_contact_damp(self):
            return self._context_helper.get(self._attributes.k_contact_damp)

        @k_contact_damp.setter
        def k_contact_damp(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.k_contact_damp)
            self._context_helper.set_attr_value(value, self._attributes.k_contact_damp)

        @property
        def k_contact_elastic(self):
            return self._context_helper.get(self._attributes.k_contact_elastic)

        @k_contact_elastic.setter
        def k_contact_elastic(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.k_contact_elastic)
            self._context_helper.set_attr_value(value, self._attributes.k_contact_elastic)

        @property
        def k_contact_friction(self):
            return self._context_helper.get(self._attributes.k_contact_friction)

        @k_contact_friction.setter
        def k_contact_friction(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.k_contact_friction)
            self._context_helper.set_attr_value(value, self._attributes.k_contact_friction)

        @property
        def k_contact_mu(self):
            return self._context_helper.get(self._attributes.k_contact_mu)

        @k_contact_mu.setter
        def k_contact_mu(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.k_contact_mu)
            self._context_helper.set_attr_value(value, self._attributes.k_contact_mu)

        @property
        def mass(self):
            return self._context_helper.get(self._attributes.mass)

        @mass.setter
        def mass(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.mass)
            self._context_helper.set_attr_value(value, self._attributes.mass)

        @property
        def num_substeps(self):
            return self._context_helper.get(self._attributes.num_substeps)

        @num_substeps.setter
        def num_substeps(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.num_substeps)
            self._context_helper.set_attr_value(value, self._attributes.num_substeps)

        @property
        def positions(self):
            return self._context_helper.get_array(self._attributes.positions)

        @positions.setter
        def positions(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.positions)
            self._context_helper.set_attr_value(value, self._attributes.positions)
            self.positions_size = self._context_helper.get_elem_count(self._attributes.positions)

        @property
        def radius(self):
            return self._context_helper.get(self._attributes.radius)

        @radius.setter
        def radius(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.radius)
            self._context_helper.set_attr_value(value, self._attributes.radius)
    class ValuesForOutputs(og.DynamicAttributeAccess):
        """Helper class that creates natural hierarchical access to output attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            super().__init__(context_helper, node, attributes, dynamic_attributes)
            self.positions_size = None

        @property
        def positions(self):
            return self._context_helper.get_array(self._attributes.positions, get_for_write=True, reserved_element_count=self.positions_size)

        @positions.setter
        def positions(self, value):
            self._context_helper.set_attr_value(value, self._attributes.positions)
            self.positions_size = self._context_helper.get_elem_count(self._attributes.positions)
    class ValuesForState(og.DynamicAttributeAccess):
        """Helper class that creates natural hierarchical access to state attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            super().__init__(context_helper, node, attributes, dynamic_attributes)
    def __init__(self, context_helper, node):
        super().__init__(node, context_helper)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT)
        self.inputs = OgnParticleSolverDatabase.ValuesForInputs(context_helper, node, self.attributes.inputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT)
        self.outputs = OgnParticleSolverDatabase.ValuesForOutputs(context_helper, node, self.attributes.outputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE)
        self.state = OgnParticleSolverDatabase.ValuesForState(context_helper, node, self.attributes.state, dynamic_attributes)

    @property
    def context(self) -> og.GraphContext:
        return self.context_helper.context
    class abi:
        """Class defining the ABI interface for the node type"""
        @staticmethod
        def get_node_type():
            get_node_type_function = getattr(OgnParticleSolverDatabase.NODE_TYPE_CLASS, 'get_node_type', None)
            if callable(get_node_type_function):
                return get_node_type_function()
            return 'omni.warp.OgnParticleSolver'
        @staticmethod
        def compute(context_helper, node):
            db = OgnParticleSolverDatabase(context_helper, node)
            try:
                db.inputs._setting_locked = True
                compute_function = getattr(OgnParticleSolverDatabase.NODE_TYPE_CLASS, 'compute', None)
                if callable(compute_function) and compute_function.__code__.co_argcount > 1:
                    return compute_function(context_helper, node)
                return OgnParticleSolverDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:
                stack_trace = "".join(traceback.format_tb(sys.exc_info()[2].tb_next))
                db.log_error(f'Assertion raised in compute - {error}\n{stack_trace}', add_context=False)
            finally:
                db.inputs._setting_locked = False
            return False
        @staticmethod
        def initialize(context_helper, node):
            OgnParticleSolverDatabase._initialize_per_node_data(node)

            # Set any default values the attributes have specified
            db = OgnParticleSolverDatabase(context_helper, node)
            db.inputs.collider_offset = 0.01
            db.inputs.gravity = [0.0, -9.8, 0.0]
            db.inputs.ground = False
            db.inputs.ground_plane = [0.0, 1.0, 0.0]
            db.inputs.k_contact_adhesion = 0.75
            db.inputs.k_contact_cohesion = 0.75
            db.inputs.k_contact_damp = 100.0
            db.inputs.k_contact_elastic = 5000.0
            db.inputs.k_contact_friction = 2000.0
            db.inputs.k_contact_mu = 0.75
            db.inputs.mass = 1.0
            db.inputs.num_substeps = 32
            db.inputs.positions = []
            db.inputs.radius = 10.0
            initialize_function = getattr(OgnParticleSolverDatabase.NODE_TYPE_CLASS, 'initialize', None)
            if callable(initialize_function):
                initialize_function(context_helper, node)
        @staticmethod
        def release(node):
            release_function = getattr(OgnParticleSolverDatabase.NODE_TYPE_CLASS, 'release', None)
            if callable(release_function):
                release_function(node)
            OgnParticleSolverDatabase._release_per_node_data(node)
        @staticmethod
        def update_node_version(context, node, old_version, new_version):
            update_node_version_function = getattr(OgnParticleSolverDatabase.NODE_TYPE_CLASS, 'update_node_version', None)
            if callable(update_node_version_function):
                return update_node_version_function(context, node, old_version, new_version)
            return False
        @staticmethod
        def initialize_type(node_type):
            initialize_type_function = getattr(OgnParticleSolverDatabase.NODE_TYPE_CLASS, 'initialize_type', None)
            needs_initializing = True
            if callable(initialize_type_function):
                needs_initializing = initialize_type_function(node_type)
            if needs_initializing:
                node_type.set_metadata(og.MetadataKeys.EXTENSION, "omni.warp")
                node_type.set_metadata(og.MetadataKeys.DESCRIPTION, "Particle simulation node")
                node_type.set_metadata(og.MetadataKeys.LANGUAGE, "Python")
                OgnParticleSolverDatabase.INTERFACE.add_to_node_type(node_type)
                node_type.set_has_state(True)
        @staticmethod
        def on_connection_type_resolve(node):
            on_connection_type_resolve_function = getattr(OgnParticleSolverDatabase.NODE_TYPE_CLASS, 'on_connection_type_resolve', None)
            if callable(on_connection_type_resolve_function):
                on_connection_type_resolve_function(node)
    NODE_TYPE_CLASS = None
    @staticmethod
    def register(node_type_class):
        OgnParticleSolverDatabase.NODE_TYPE_CLASS = node_type_class
        og.register_node_type(OgnParticleSolverDatabase.abi, 1)
    @staticmethod
    def deregister():
        og.deregister_node_type("omni.warp.OgnParticleSolver")
