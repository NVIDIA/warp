# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the `from_omni_graph()` API."""

import omni.graph.core as og
import omni.graph.core.tests as ogts
import omni.warp

import warp as wp

from ._common import (
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
            device = omni.warp.nodes.device_get_cuda_compute() if variant == "gpuBundle" else wp.get_device("cpu")
            with wp.ScopedDevice(device):
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
                    on_gpu=device.is_cuda,
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
                    on_gpu=device.is_cuda,
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
                    on_gpu=device.is_cuda,
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

    for variant in variants:
        if variant in ("cpuBundle", "gpuBundle"):
            data = bundle_get_attr(db.inputs.bundleAttr, "{}FloatArray".format(variant))
        else:
            data = getattr(db.inputs, "{}FloatArrayAttr".format(variant))

        result = omni.warp.from_omni_graph(data)
        expected = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, shape=(3,))
        array_are_equal(result, expected)

        # Cast the array to vec3.
        result = omni.warp.from_omni_graph(data, dtype=wp.vec3)
        expected = wp.array(((1.0, 2.0, 3.0),), dtype=wp.vec3, shape=(1,))
        array_are_equal(result, expected)

    # Vector3 Array
    # --------------------------------------------------------------------------

    for variant in variants:
        if variant in ("cpuBundle", "gpuBundle"):
            data = bundle_get_attr(db.inputs.bundleAttr, "{}Vec3Array".format(variant))
        else:
            data = getattr(db.inputs, "{}Vec3ArrayAttr".format(variant))

        result = omni.warp.from_omni_graph(data)
        expected = wp.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=wp.vec3, shape=(2,))
        array_are_equal(result, expected)

        # Cast the array to floats while preserving the same shape.
        result = omni.warp.from_omni_graph(data, dtype=wp.float32)
        expected = wp.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=wp.float32, shape=(2, 3))
        array_are_equal(result, expected)

        # Cast the array to floats while flattening it to a single dimension.
        result = omni.warp.from_omni_graph(data, dtype=wp.float32, shape=(6,))
        expected = wp.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype=wp.float32, shape=(6,))
        array_are_equal(result, expected)

    # Matrix4 Array
    # --------------------------------------------------------------------------

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
        array_are_equal(result, expected)

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
        array_are_equal(result, expected)

        # Cast the array to floats while flattening it to a single dimension.
        result = omni.warp.from_omni_graph(data, dtype=wp.float64, shape=(32,))
        expected = wp.array(
            (
                1.0,
                2.0,
                3.0,
                4.0,
                2.0,
                3.0,
                4.0,
                5.0,
                3.0,
                4.0,
                5.0,
                6.0,
                4.0,
                5.0,
                6.0,
                7.0,
                2.0,
                3.0,
                4.0,
                5.0,
                3.0,
                4.0,
                5.0,
                6.0,
                4.0,
                5.0,
                6.0,
                7.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ),
            dtype=wp.float64,
            shape=(32,),
        )
        array_are_equal(result, expected)


class FromOmniGraphNode:
    @staticmethod
    def compute(db: og.Database) -> bool:
        device = (
            omni.warp.nodes.device_get_cuda_compute() if db.inputs.device == "cuda" else wp.get_device(db.inputs.device)
        )

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

        device_attr.set("cuda")
        await og.Controller.evaluate(self.graph)
        self.assertTrue(success_attr.get())
