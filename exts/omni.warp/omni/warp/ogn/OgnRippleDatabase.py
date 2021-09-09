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
            inputs.collider
            inputs.damp
            inputs.displace
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
        ('inputs:collider', 'bundle', 0, None, '', {}, True, None),
        ('inputs:damp', 'float', 0, None, '', {}, True, 0),
        ('inputs:displace', 'float', 0, None, '', {}, True, 1.0),
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
        def collider(self) -> og.BundleContents:
            """Get the bundle wrapper class for the attribute inputs.collider"""
            return self.bundles.collider

        @property
        def damp(self):
            return self.context_helper.get_attr_value(self.attributes.damp)

        @damp.setter
        def damp(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.damp)
            self.context_helper.set_attr_value(value, self.attributes.damp)

        @property
        def displace(self):
            return self.context_helper.get_attr_value(self.attributes.displace)

        @displace.setter
        def displace(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.displace)
            self.context_helper.set_attr_value(value, self.attributes.displace)
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
        try:
            super().__init__(node, context_helper)
            self.inputs = OgnRippleDatabase.ValuesForInputs(self.context_helper, node, self.attributes.inputs)
            self.outputs = OgnRippleDatabase.ValuesForOutputs(self.context_helper, node, self.attributes.outputs)
        except AttributeError:
            self.log_hot_reload_problem(node)
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
                return OgnRippleDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:
                db.log_error(f'Assertion raised in compute - {error}')
            return False
        @staticmethod
        def initialize(context_helper, node):
            OgnRippleDatabase._initialize_per_node_data(node)

            # Set any default values the attributes have specified
            db = OgnRippleDatabase(context_helper, node)
            db.inputs.damp = 0
            db.inputs.displace = 1.0
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
        og.register_node_type(OgnRippleDatabase.abi, 1)
    @staticmethod
    def deregister():
        og.deregister_node_type("omni.warp.OgnRipple")
