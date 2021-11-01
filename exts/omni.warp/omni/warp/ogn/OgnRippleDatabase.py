"""Support for simplified access to data on nodes of type omni.warp.OgnRipple

2D Wave equation solver
"""

import omni.graph.core as og
from contextlib import suppress
from inspect import getfullargspec
class OgnRippleDatabase(og.Database):
    """Helper class providing simplified access to data on nodes of type omni.warp.OgnRipple

    Class Members:
        node: Node being evaluated

    Attribute Value Properties:
        Inputs:
            inputs.buoyancy
            inputs.buoyancy_damp
            inputs.buoyancy_enabled
            inputs.collider_0
            inputs.collider_1
            inputs.collider_2
            inputs.collider_3
            inputs.damp
            inputs.delay
            inputs.density_0
            inputs.density_1
            inputs.density_2
            inputs.density_3
            inputs.displace
            inputs.gravity
            inputs.grid
            inputs.resolution
            inputs.speed
        Outputs:
            outputs.face_counts
            outputs.face_indices
            outputs.vertices
    """
    # This is an internal object that provides per-class storage of a per-node data dictionary
    PER_NODE_DATA = {}

    # This is an internal object that describes unchanging attributes in a generic way
    # The values in this list are in no particular order, as a per-attribute tuple
    #     Name, Type, ExtendedTypeIndex, UiName, Description, Metadata, Is_Required, DefaultValue
    # You should not need to access any of this data directly, use the defined database interfaces
    INTERFACE = og.Database._get_interface([
        ('inputs:buoyancy', 'float', 0, None, '', {}, True, 15.0),
        ('inputs:buoyancy_damp', 'float', 0, None, '', {}, True, 0.25),
        ('inputs:buoyancy_enabled', 'bool', 0, None, '', {}, True, False),
        ('inputs:collider_0', 'bundle', 0, None, '', {}, True, None),
        ('inputs:collider_1', 'bundle', 0, None, '', {}, True, None),
        ('inputs:collider_2', 'bundle', 0, None, '', {}, True, None),
        ('inputs:collider_3', 'bundle', 0, None, '', {}, True, None),
        ('inputs:damp', 'float', 0, None, '', {}, True, 0),
        ('inputs:delay', 'float', 0, None, '', {}, True, 0.0),
        ('inputs:density_0', 'float', 0, None, '', {}, True, 1.0),
        ('inputs:density_1', 'float', 0, None, '', {}, True, 1.0),
        ('inputs:density_2', 'float', 0, None, '', {}, True, 1.0),
        ('inputs:density_3', 'float', 0, None, '', {}, True, 1.0),
        ('inputs:displace', 'float', 0, None, '', {}, True, 1.0),
        ('inputs:gravity', 'float', 0, None, '', {}, True, -9.8),
        ('inputs:grid', 'bundle', 0, None, '', {}, True, None),
        ('inputs:resolution', 'float', 0, None, '', {}, True, 50.0),
        ('inputs:speed', 'float', 0, None, '', {}, True, 100.0),
        ('outputs:face_counts', 'int[]', 0, None, '', {}, True, None),
        ('outputs:face_indices', 'int[]', 0, None, '', {}, True, None),
        ('outputs:vertices', 'point3f[]', 0, None, '', {}, True, None),
    ])

    @classmethod
    def _populate_role_data(cls):
        """Populate a role structure with the non-default roles on this node type"""
        role_data = super()._populate_role_data()
        role_data.outputs.vertices = og.Database.ROLE_POINT
        return role_data

    class ValuesForInputs:
        """Helper class that creates natural hierarchical access to input attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes):
            """Initialize simplified access for the attribute data"""
            self.context_helper = context_helper
            self.node = node
            self.attributes = attributes
            self.bundles = og.BundleContainer(context_helper.context, node, attributes, read_only=True)
            self.setting_locked = False
        @property
        def buoyancy(self):
            return self.context_helper.get_attr_value(self.attributes.buoyancy)
        @buoyancy.setter
        def buoyancy(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.buoyancy)
            self.context_helper.set_attr_value(value, self.attributes.buoyancy)
        @property
        def buoyancy_damp(self):
            return self.context_helper.get_attr_value(self.attributes.buoyancy_damp)
        @buoyancy_damp.setter
        def buoyancy_damp(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.buoyancy_damp)
            self.context_helper.set_attr_value(value, self.attributes.buoyancy_damp)
        @property
        def buoyancy_enabled(self):
            return self.context_helper.get_attr_value(self.attributes.buoyancy_enabled)
        @buoyancy_enabled.setter
        def buoyancy_enabled(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.buoyancy_enabled)
            self.context_helper.set_attr_value(value, self.attributes.buoyancy_enabled)
        @property
        def collider_0(self) -> og.BundleContents:
            """Get the bundle wrapper class for the attribute inputs.collider_0"""
            return self.bundles.collider_0
        @property
        def collider_1(self) -> og.BundleContents:
            """Get the bundle wrapper class for the attribute inputs.collider_1"""
            return self.bundles.collider_1
        @property
        def collider_2(self) -> og.BundleContents:
            """Get the bundle wrapper class for the attribute inputs.collider_2"""
            return self.bundles.collider_2
        @property
        def collider_3(self) -> og.BundleContents:
            """Get the bundle wrapper class for the attribute inputs.collider_3"""
            return self.bundles.collider_3
        @property
        def damp(self):
            return self.context_helper.get_attr_value(self.attributes.damp)
        @damp.setter
        def damp(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.damp)
            self.context_helper.set_attr_value(value, self.attributes.damp)
        @property
        def delay(self):
            return self.context_helper.get_attr_value(self.attributes.delay)
        @delay.setter
        def delay(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.delay)
            self.context_helper.set_attr_value(value, self.attributes.delay)
        @property
        def density_0(self):
            return self.context_helper.get_attr_value(self.attributes.density_0)
        @density_0.setter
        def density_0(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.density_0)
            self.context_helper.set_attr_value(value, self.attributes.density_0)
        @property
        def density_1(self):
            return self.context_helper.get_attr_value(self.attributes.density_1)
        @density_1.setter
        def density_1(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.density_1)
            self.context_helper.set_attr_value(value, self.attributes.density_1)
        @property
        def density_2(self):
            return self.context_helper.get_attr_value(self.attributes.density_2)
        @density_2.setter
        def density_2(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.density_2)
            self.context_helper.set_attr_value(value, self.attributes.density_2)
        @property
        def density_3(self):
            return self.context_helper.get_attr_value(self.attributes.density_3)
        @density_3.setter
        def density_3(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.density_3)
            self.context_helper.set_attr_value(value, self.attributes.density_3)
        @property
        def displace(self):
            return self.context_helper.get_attr_value(self.attributes.displace)
        @displace.setter
        def displace(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.displace)
            self.context_helper.set_attr_value(value, self.attributes.displace)
        @property
        def gravity(self):
            return self.context_helper.get_attr_value(self.attributes.gravity)
        @gravity.setter
        def gravity(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.gravity)
            self.context_helper.set_attr_value(value, self.attributes.gravity)
        @property
        def grid(self) -> og.BundleContents:
            """Get the bundle wrapper class for the attribute inputs.grid"""
            return self.bundles.grid
        @property
        def resolution(self):
            return self.context_helper.get_attr_value(self.attributes.resolution)
        @resolution.setter
        def resolution(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.resolution)
            self.context_helper.set_attr_value(value, self.attributes.resolution)
        @property
        def speed(self):
            return self.context_helper.get_attr_value(self.attributes.speed)
        @speed.setter
        def speed(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.speed)
            self.context_helper.set_attr_value(value, self.attributes.speed)

    class ValuesForOutputs:
        """Helper class that creates natural hierarchical access to output attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes):
            """Initialize simplified access for the attribute data"""
            self.context_helper = context_helper
            self.node = node
            self.attributes = attributes
            self.face_counts_size = None
            self.face_indices_size = None
            self.vertices_size = None
        @property
        def face_counts(self):
            return self.context_helper.get_attr_value(self.attributes.face_counts,getForWrite=True,writeElemCount=self.face_counts_size)
        @face_counts.setter
        def face_counts(self, value):
            self.context_helper.set_attr_value(value, self.attributes.face_counts)
            self.face_counts_size = self.context_helper.get_elem_count(self.attributes.face_counts)
        @property
        def face_indices(self):
            return self.context_helper.get_attr_value(self.attributes.face_indices,getForWrite=True,writeElemCount=self.face_indices_size)
        @face_indices.setter
        def face_indices(self, value):
            self.context_helper.set_attr_value(value, self.attributes.face_indices)
            self.face_indices_size = self.context_helper.get_elem_count(self.attributes.face_indices)
        @property
        def vertices(self):
            return self.context_helper.get_attr_value(self.attributes.vertices,getForWrite=True,writeElemCount=self.vertices_size)
        @vertices.setter
        def vertices(self, value):
            self.context_helper.set_attr_value(value, self.attributes.vertices)
            self.vertices_size = self.context_helper.get_elem_count(self.attributes.vertices)
    def __init__(self, context_helper, node):
        super().__init__(node)
        self.context_helper = context_helper
        self.inputs = OgnRippleDatabase.ValuesForInputs(self.context_helper, node, self.attributes.inputs)
        self.outputs = OgnRippleDatabase.ValuesForOutputs(self.context_helper, node, self.attributes.outputs)
    class abi:
        @staticmethod
        def get_node_type():
            get_node_type_function = getattr(OgnRippleDatabase.NODE_TYPE_CLASS, 'get_node_type', None)
            if callable(get_node_type_function):
                return get_node_type_function()
            return 'omni.warp.OgnRipple'
        @staticmethod
        def compute(context_helper, node):
            db = OgnRippleDatabase(context_helper, node)
            try:
                compute_function = getattr(OgnRippleDatabase.NODE_TYPE_CLASS, 'compute', None)
                if callable(compute_function) and len(getfullargspec(compute_function).args) > 1:
                    return compute_function(context_helper, node)
                with suppress(AttributeError):
                    db.inputs.setting_locked = True
                return OgnRippleDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:
                db.log_error(f'Assertion raised in compute - {error}')
            return False
        @staticmethod
        def initialize(context_helper, node):
            OgnRippleDatabase._initialize_per_node_data(node)
            db = OgnRippleDatabase(context_helper, node)

            # Set any default values the attributes have specified
            db.inputs.buoyancy = 15.0
            db.inputs.buoyancy_damp = 0.25
            db.inputs.buoyancy_enabled = False
            db.inputs.damp = 0
            db.inputs.delay = 0.0
            db.inputs.density_0 = 1.0
            db.inputs.density_1 = 1.0
            db.inputs.density_2 = 1.0
            db.inputs.density_3 = 1.0
            db.inputs.displace = 1.0
            db.inputs.gravity = -9.8
            db.inputs.resolution = 50.0
            db.inputs.speed = 100.0
            initialize_function = getattr(OgnRippleDatabase.NODE_TYPE_CLASS, 'initialize', None)
            if callable(initialize_function):
                initialize_function(context_helper, node)
        @staticmethod
        def release(node):
            release_function = getattr(OgnRippleDatabase.NODE_TYPE_CLASS, 'release', None)
            if callable(release_function):
                release_function(node)
            OgnRippleDatabase._release_per_node_data(node)
        @staticmethod
        def update_node_version(context, node, old_version, new_version):
            update_node_version_function = getattr(OgnRippleDatabase.NODE_TYPE_CLASS, 'update_node_version', None)
            if callable(update_node_version_function):
                return update_node_version_function(context, node, old_version, new_version)
            return False
        @staticmethod
        def initialize_type(node_type):
            initialize_type_function = getattr(OgnRippleDatabase.NODE_TYPE_CLASS, 'initialize_type', None)
            needs_initializing = True
            if callable(initialize_type_function):
                needs_initializing = initialize_type_function(node_type)
            if needs_initializing:
                node_type.set_metadata('__extension', "omni.warp")
                node_type.set_metadata("__description", "2D Wave equation solver")
                OgnRippleDatabase.INTERFACE.add_to_node_type(node_type)
                node_type.set_has_state(True)
        @staticmethod
        def on_connection_type_resolve(node):
            on_connection_type_resolve_function = getattr(OgnRippleDatabase.NODE_TYPE_CLASS, 'on_connection_type_resolve', None)
            if callable(on_connection_type_resolve_function):
                on_connection_type_resolve_function(node)
    NODE_TYPE_CLASS = None
    @staticmethod
    def register(node_type_class):
        OgnRippleDatabase.NODE_TYPE_CLASS = node_type_class
        og.register_node(OgnRippleDatabase.abi, 1)
