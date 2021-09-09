
"""
Dynamically import every file in a directory tree that looks like a Python Ogn Node.
This includes linked directories, which is the mechanism by which nodes can be hot-reloaded from the source tree.
"""
from omni.graph.tools.node_generator.register_ogn_nodes import register_ogn_nodes

register_ogn_nodes(__file__, "omni.warp")
