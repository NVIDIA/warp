# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the `from_omni_graph()` API."""

import omni.graph.core as og
import omni.graph.core.tests as ogts

import omni.warp
import warp as wp

from omni.warp.nodes.tests._common import (
    array_are_equal,
    attr_set_array,
    bundle_create_attr,
    bundle_get_attr,
    register_node,
)


#   Test Node Definitions
# -----------------------------------------------------------------------------


MAKE_DATA_NODE_DEF = """{
    "WarpTestsMakeData": {
        "version": 1,
        "description": "Make data.",
        "language": "Python",
        "uiName": "Make Data",
        "cudaPointers": "cpu",
        "outputs": {
            "floatArrayAttr": {
                "type": "float[]",
                "uiName": "Float Array",
                "description": "Float array."
            },
            "vec3ArrayAttr": {
                "type": "float[3][]",
                "uiName": "Vector3 Array",
                "description": "Vector3 array."
            },
            "mat4ArrayAttr": {
                "type": "matrixd[4][]",
                "uiName": "Matrix4 Array",
                "description": "Matrix4 array."
            },
            "bundleAttr": {
                "type": "bundle",
                "uiName": "Bundle",
                "description": "Bundle."
            }
        }
    }
}
"""


class MakeDataNode:
    @staticmethod
    def compute(db: og.Database) -> bool:
        db.outputs.floatArrayAttr = (1.0, 2.0, 3.0)

        db.outputs.vec3ArrayAttr = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

        db.outputs.mat4ArrayAttr = (
            (
                (1.0, 2.0, 3.0, 4.0),
                (2.0, 3.0, 4.0, 5.0),
                (3.0, 4.0, 5.0, 6.0),
                (4.0, 5.0, 6.0, 7.0),
            ),
            (
                (2.0, 3.0, 4.0, 5.0),
                (3.0, 4.0, 5.0, 6.0),
                (4.0, 5.0, 6.0, 7.0),
                (5.0, 6.0, 7.0, 8.0),
            ),
        )

        for variant in ("cpuBundle", "gpuBundle"):
            attr = bundle_create_attr(
                db.outputs.bundleAttr,
                "{}FloatArray".format(variant),
                og.Type(
                    og.BaseDataType.FLOAT,
                    tuple_count=1,
                    array_depth=1,
                    role=og.AttributeRole.NONE,
                ),
                size=3,
            )
            attr_set_array(
                attr,
                wp.array(db.outputs.floatArrayAttr, dtype=wp.float32),
                on_gpu=variant == "gpuBundle",
            )

            attr = bundle_create_attr(
                db.outputs.bundleAttr,
                "{}Vec3Array".format(variant),
                og.Type(
                    og.BaseDataType.FLOAT,
                    tuple_count=3,
                    array_depth=1,
                    role=og.AttributeRole.NONE,
                ),
                size=2,
            )
            attr_set_array(
                attr,
                wp.array(db.outputs.vec3ArrayAttr, dtype=wp.vec3),
                on_gpu=variant == "gpuBundle",
            )

            attr = bundle_create_attr(
                db.outputs.bundleAttr,
                "{}Mat4Array".format(variant),
                og.Type(
                    og.BaseDataType.DOUBLE,
                    tuple_count=16,
                    array_depth=1,
                    role=og.AttributeRole.MATRIX,
                ),
                size=2,
            )
            attr_set_array(
                attr,
                wp.array(db.outputs.mat4ArrayAttr, dtype=wp.mat44d),
                on_gpu=variant == "gpuBundle",
            )

        return True


FROM_OMNI_GRAPH_NODE_DEF = """{
    "WarpTestsFromOmniGraph": {
        "version": 1,
        "description": "From omni graph.",
        "language": "Python",
        "uiName": "From Omni Graph",
        "cudaPointers": "cpu",
        "inputs": {
            "anyFloatArrayAttr": {
                "type": "float[]",
                "uiName": "Float Array (Any)",
                "description": "Float array (any).",
                "memoryType": "any"
            },
            "cpuFloatArrayAttr": {
                "type": "float[]",
                "uiName": "Float Array (CPU)",
                "description": "Float array (cpu).",
                "memoryType": "cpu"
            },
            "gpuFloatArrayAttr": {
                "type": "float[]",
                "uiName": "Float Array (GPU)",
                "description": "Float array (gpu).",
                "memoryType": "cuda"
            },
            "anyVec3ArrayAttr": {
                "type": "float[3][]",
                "uiName": "Vector3 Array (Any)",
                "description": "Vector3 array (any).",
                "memoryType": "any"
            },
            "cpuVec3ArrayAttr": {
                "type": "float[3][]",
                "uiName": "Vector3 Array (CPU)",
                "description": "Vector3 array (cpu).",
                "memoryType": "cpu"
            },
            "gpuVec3ArrayAttr": {
                "type": "float[3][]",
                "uiName": "Vector3 Array (GPU)",
                "description": "Vector3 array (gpu).",
                "memoryType": "cuda"
            },
            "anyMat4ArrayAttr": {
                "type": "matrixd[4][]",
                "uiName": "Matrix4 Array (Any)",
                "description": "Matrix4 array (any).",
                "memoryType": "any"
            },
            "cpuMat4ArrayAttr": {
                "type": "matrixd[4][]",
                "uiName": "Matrix4 Array (CPU)",
                "description": "Matrix4 array (cpu).",
                "memoryType": "cpu"
            },
            "gpuMat4ArrayAttr": {
                "type": "matrixd[4][]",
                "uiName": "Matrix4 Array (GPU)",
                "description": "Matrix4 array (gpu).",
                "memoryType": "cuda"
            },
            "bundleAttr": {
                "type": "bundle",
                "uiName": "Bundle",
                "description": "Bundle."
            },
            "device": {
                "type": "string",
                "uiName": "Device",
                "description": "Device."
            }
        },
        "outputs": {
            "success": {
                "type": "bool",
                "uiName": "Success",
                "description": "Success."
            }
        }
    }
}
"""


def compute(db: og.Database) -> None:
    """Evaluates the node."""

    variants = ("any", "cpu", "gpu", "cpuBundle", "gpuBundle")

    # Float Array
    # --------------------------------------------------------------------------

    attr = "floatArray"
    for variant in variants:
        if variant in ("cpuBundle", "gpuBundle"):
            data = bundle_get_attr(db.inputs.bundleAttr, "{}FloatArray".format(variant))
        else:
            data = getattr(db.inputs, "{}FloatArrayAttr".format(variant))

        result = omni.warp.from_omni_graph(data)
        expected = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, shape=(3,))
        if not array_are_equal(result, expected):
            raise RuntimeError("Test for {} ({}) not passing.".format(attr, variant))

        # Cast the array to vec3.
        result = omni.warp.from_omni_graph(data, dtype=wp.vec3)
        expected = wp.array(((1.0, 2.0, 3.0),), dtype=wp.vec3, shape=(1,))
        if not array_are_equal(result, expected):
            raise RuntimeError("Test for {} ({}) with casting to wp.vec3 not passing.".format(attr, variant))

    # Vector3 Array
    # --------------------------------------------------------------------------

    attr = "vec3Array"
    for variant in variants:
        if variant in ("cpuBundle", "gpuBundle"):
            data = bundle_get_attr(db.inputs.bundleAttr, "{}Vec3Array".format(variant))
        else:
            data = getattr(db.inputs, "{}Vec3ArrayAttr".format(variant))

        result = omni.warp.from_omni_graph(data)
        expected = wp.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=wp.vec3, shape=(2,))
        if not array_are_equal(result, expected):
            raise RuntimeError("Test for {} ({}) not passing.".format(attr, variant))

        # Cast the array to floats while preserving the same shape.
        result = omni.warp.from_omni_graph(data, dtype=wp.float32)
        expected = wp.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=wp.float32, shape=(2, 3))
        if not array_are_equal(result, expected):
            raise RuntimeError("Test for {} ({}) with casting to float not passing.".format(attr, variant))

        # Cast the array to floats while flattening it to a single dimension.
        result = omni.warp.from_omni_graph(data, dtype=wp.float32, shape=(6,))
        expected = wp.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype=wp.float32, shape=(6,))
        if not array_are_equal(result, expected):
            raise RuntimeError("Test for {} ({}) with flattening not passing.".format(attr, variant))

    # Matrix4 Array
    # --------------------------------------------------------------------------

    attr = "mat4Array"
    for variant in variants:
        if variant in ("cpuBundle", "gpuBundle"):
            data = bundle_get_attr(db.inputs.bundleAttr, "{}Mat4Array".format(variant))
        else:
            data = getattr(db.inputs, "{}Mat4ArrayAttr".format(variant))

        # Due to OmniGraph only supporting 1-D arrays with elements that might
        # be represented as tuples, we can at best reconstruct 2-D arrays,
        # however arrays made of elements such as matrices require 3 dimensions
        # and hence are not something that we can infer from the data we're
        # being given, so we need to explicitly pass the dtype here.
        result = omni.warp.from_omni_graph(data, dtype=wp.mat44d)
        expected = wp.array(
            (
                (
                    (1.0, 2.0, 3.0, 4.0),
                    (2.0, 3.0, 4.0, 5.0),
                    (3.0, 4.0, 5.0, 6.0),
                    (4.0, 5.0, 6.0, 7.0),
                ),
                (
                    (2.0, 3.0, 4.0, 5.0),
                    (3.0, 4.0, 5.0, 6.0),
                    (4.0, 5.0, 6.0, 7.0),
                    (5.0, 6.0, 7.0, 8.0),
                ),
            ),
            dtype=wp.mat44d,
            shape=(2,),
        )
        if not array_are_equal(result, expected):
            raise RuntimeError("Test for {} ({}) not passing.".format(attr, variant))

        # Cast the array to vec4d.
        result = omni.warp.from_omni_graph(data, dtype=wp.vec4d)
        expected = wp.array(
            (
                (
                    (1.0, 2.0, 3.0, 4.0),
                    (2.0, 3.0, 4.0, 5.0),
                    (3.0, 4.0, 5.0, 6.0),
                    (4.0, 5.0, 6.0, 7.0),
                ),
                (
                    (2.0, 3.0, 4.0, 5.0),
                    (3.0, 4.0, 5.0, 6.0),
                    (4.0, 5.0, 6.0, 7.0),
                    (5.0, 6.0, 7.0, 8.0),
                ),
            ),
            dtype=wp.vec4d,
            shape=(2, 4),
        )
        if not array_are_equal(result, expected):
            raise RuntimeError("Test for {} ({}) with casting to vec4 not passing.".format(attr, variant))

        # Cast the array to floats while flattening it to a single dimension.
        result = omni.warp.from_omni_graph(data, dtype=wp.float64, shape=(32,))
        expected = wp.array(
            (
                1.0, 2.0, 3.0, 4.0,
                2.0, 3.0, 4.0, 5.0,
                3.0, 4.0, 5.0, 6.0,
                4.0, 5.0, 6.0, 7.0,
                2.0, 3.0, 4.0, 5.0,
                3.0, 4.0, 5.0, 6.0,
                4.0, 5.0, 6.0, 7.0,
                5.0, 6.0, 7.0, 8.0,
            ),
            dtype=wp.float64,
            shape=(32,),
        )
        if not array_are_equal(result, expected):
            raise RuntimeError("Test for {} ({}) with flattening not passing.".format(attr, variant))


class FromOmniGraphNode:
    @staticmethod
    def compute(db: og.Database) -> bool:
        device = wp.get_device(db.inputs.device)

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.outputs.success = False
            raise

        db.outputs.success = True
        return True


#   Test Case
# -----------------------------------------------------------------------------


class TestApiFromOmniGraph(ogts.OmniGraphTestCase):
    async def setUp(self) -> None:
        await super().setUp()

        register_node(MakeDataNode, MAKE_DATA_NODE_DEF, "omni.warp.nodes")
        register_node(FromOmniGraphNode, FROM_OMNI_GRAPH_NODE_DEF, "omni.warp.nodes")

        (graph, _, _, _) = og.Controller.edit(
            "/Graph",
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("MakeData", "omni.warp.nodes.WarpTestsMakeData"),
                    ("FromOmniGraph", "omni.warp.nodes.WarpTestsFromOmniGraph"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("MakeData.outputs:floatArrayAttr", "FromOmniGraph.inputs:anyFloatArrayAttr"),
                    ("MakeData.outputs:floatArrayAttr", "FromOmniGraph.inputs:cpuFloatArrayAttr"),
                    ("MakeData.outputs:floatArrayAttr", "FromOmniGraph.inputs:gpuFloatArrayAttr"),
                    ("MakeData.outputs:vec3ArrayAttr", "FromOmniGraph.inputs:anyVec3ArrayAttr"),
                    ("MakeData.outputs:vec3ArrayAttr", "FromOmniGraph.inputs:cpuVec3ArrayAttr"),
                    ("MakeData.outputs:vec3ArrayAttr", "FromOmniGraph.inputs:gpuVec3ArrayAttr"),
                    ("MakeData.outputs:mat4ArrayAttr", "FromOmniGraph.inputs:anyMat4ArrayAttr"),
                    ("MakeData.outputs:mat4ArrayAttr", "FromOmniGraph.inputs:cpuMat4ArrayAttr"),
                    ("MakeData.outputs:mat4ArrayAttr", "FromOmniGraph.inputs:gpuMat4ArrayAttr"),
                    ("MakeData.outputs_bundleAttr", "FromOmniGraph.inputs:bundleAttr"),
                ],
            },
        )

        self.graph = graph

    async def test_main(self):
        node = og.Controller.node(("FromOmniGraph", self.graph))
        device_attr = og.Controller.attribute("inputs:device", node)
        success_attr = og.Controller.attribute("outputs:success", node)

        device_attr.set("cpu")
        await og.Controller.evaluate(self.graph)
        self.assertTrue(success_attr.get())

        device_attr.set("cuda:0")
        await og.Controller.evaluate(self.graph)
        self.assertTrue(success_attr.get())
