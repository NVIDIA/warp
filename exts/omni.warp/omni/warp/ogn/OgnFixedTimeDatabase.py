"""Support for simplified access to data on nodes of type omni.warp.FixedTime


"""

import omni.graph.core as og
from contextlib import suppress
from inspect import getfullargspec
class OgnFixedTimeDatabase(og.Database):
    """Helper class providing simplified access to data on nodes of type omni.warp.FixedTime

    Class Members:
        node: Node being evaluated

    Attribute Value Properties:
        Inputs:
            inputs.end
            inputs.fps
            inputs.start
        Outputs:
            outputs.time
    """
    # This is an internal object that provides per-class storage of a per-node data dictionary
    PER_NODE_DATA = {}
    # This is an internal object that describes unchanging attributes in a generic way
    # The values in this list are in no particular order, as a per-attribute tuple
    #     Name, Type, ExtendedTypeIndex, UiName, Description, Metadata, Is_Required, DefaultValue
    # You should not need to access any of this data directly, use the defined database interfaces
    INTERFACE = og.Database._get_interface([
        ('inputs:end', 'double', 0, None, '', {}, True, 1000.0),
        ('inputs:fps', 'double', 0, None, '', {}, True, 60.0),
        ('inputs:start', 'double', 0, None, '', {}, True, 0.0),
        ('outputs:time', 'double', 0, None, '', {}, True, 0.0),
    ])
    class ValuesForInputs:
        """Helper class that creates natural hierarchical access to input attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes):
            """Initialize simplified access for the attribute data"""
            self.context_helper = context_helper
            self.node = node
            self.attributes = attributes
            self.setting_locked = False

        @property
        def end(self):
            return self.context_helper.get_attr_value(self.attributes.end)

        @end.setter
        def end(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.end)
            self.context_helper.set_attr_value(value, self.attributes.end)

        @property
        def fps(self):
            return self.context_helper.get_attr_value(self.attributes.fps)

        @fps.setter
        def fps(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.fps)
            self.context_helper.set_attr_value(value, self.attributes.fps)

        @property
        def start(self):
            return self.context_helper.get_attr_value(self.attributes.start)

        @start.setter
        def start(self, value):
            if self.setting_locked:
                raise og.ReadOnlyError(self.attributes.start)
            self.context_helper.set_attr_value(value, self.attributes.start)
    class ValuesForOutputs:
        """Helper class that creates natural hierarchical access to output attributes"""
        def __init__(self, context_helper: og.ContextHelper, node: og.Node, attributes):
            """Initialize simplified access for the attribute data"""
            self.context_helper = context_helper
            self.node = node
            self.attributes = attributes

        @property
        def time(self):
            return self.context_helper.get_attr_value(self.attributes.time)

        @time.setter
        def time(self, value):
            self.context_helper.set_attr_value(value, self.attributes.time)
    def __init__(self, context_helper, node):
        try:
            super().__init__(node, context_helper)
            self.inputs = OgnFixedTimeDatabase.ValuesForInputs(self.context_helper, node, self.attributes.inputs)
            self.outputs = OgnFixedTimeDatabase.ValuesForOutputs(self.context_helper, node, self.attributes.outputs)
        except AttributeError:
            self.log_hot_reload_problem(node)
    class abi:
        @staticmethod
        def get_node_type():
            get_node_type_function = getattr(OgnFixedTimeDatabase.NODE_TYPE_CLASS, 'get_node_type', None)
            if callable(get_node_type_function):
                return get_node_type_function()
            return 'omni.warp.FixedTime'
        @staticmethod
        def compute(context_helper, node):
            db = OgnFixedTimeDatabase(context_helper, node)
            try:
                compute_function = getattr(OgnFixedTimeDatabase.NODE_TYPE_CLASS, 'compute', None)
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
                return OgnFixedTimeDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:
                db.log_error(f'Assertion raised in compute - {error}')
            return False
        @staticmethod
        def initialize(context_helper, node):
            OgnFixedTimeDatabase._initialize_per_node_data(node)

            # Set any default values the attributes have specified
            db = OgnFixedTimeDatabase(context_helper, node)
            db.inputs.end = 1000.0
            db.inputs.fps = 60.0
            db.inputs.start = 0.0
            db.outputs.time = 0.0
            initialize_function = getattr(OgnFixedTimeDatabase.NODE_TYPE_CLASS, 'initialize', None)
            if callable(initialize_function):
                initialize_function(context_helper, node)
        @staticmethod
        def release(node):
            release_function = getattr(OgnFixedTimeDatabase.NODE_TYPE_CLASS, 'release', None)
            if callable(release_function):
                release_function(node)
            OgnFixedTimeDatabase._release_per_node_data(node)
        @staticmethod
        def update_node_version(context, node, old_version, new_version):
            update_node_version_function = getattr(OgnFixedTimeDatabase.NODE_TYPE_CLASS, 'update_node_version', None)
            if callable(update_node_version_function):
                return update_node_version_function(context, node, old_version, new_version)
            return False
        @staticmethod
        def initialize_type(node_type):
            initialize_type_function = getattr(OgnFixedTimeDatabase.NODE_TYPE_CLASS, 'initialize_type', None)
            needs_initializing = True
            if callable(initialize_type_function):
                needs_initializing = initialize_type_function(node_type)
            if needs_initializing:
                node_type.set_metadata('__extension', "omni.warp")
                node_type.set_metadata("__description", "")
                OgnFixedTimeDatabase.INTERFACE.add_to_node_type(node_type)
        @staticmethod
        def on_connection_type_resolve(node):
            on_connection_type_resolve_function = getattr(OgnFixedTimeDatabase.NODE_TYPE_CLASS, 'on_connection_type_resolve', None)
            if callable(on_connection_type_resolve_function):
                on_connection_type_resolve_function(node)
    NODE_TYPE_CLASS = None
    @staticmethod
    def register(node_type_class):
        OgnFixedTimeDatabase.NODE_TYPE_CLASS = node_type_class
        og.register_node_type(OgnFixedTimeDatabase.abi, 1)
    @staticmethod
    def deregister():
        og.deregister_node_type("omni.warp.FixedTime")
