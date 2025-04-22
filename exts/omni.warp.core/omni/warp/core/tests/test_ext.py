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

"""Tests for the Warp core library in Kit.

Only a trimmed down list of tests is run since the full suite is too slow.

More information about testing in Kit:
    https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/testing_exts_python.html
"""

import importlib

import omni.kit.test

import warp as wp

TEST_DESCS = (
    ("cuda.test_mempool", "TestMempool"),
    ("cuda.test_peer", "TestPeer"),
    ("cuda.test_pinned", "TestPinned"),
    ("cuda.test_streams", "TestStreams"),
    ("interop.test_dlpack", "TestDLPack"),
    ("geometry.test_bvh", "TestBvh"),
    ("geometry.test_hash_grid", "TestHashGrid"),
    ("geometry.test_marching_cubes", "TestMarchingCubes"),
    ("geometry.test_mesh", "TestMesh"),
    ("geometry.test_mesh_query_aabb", "TestMeshQueryAABBMethods"),
    ("geometry.test_mesh_query_point", "TestMeshQueryPoint"),
    ("geometry.test_mesh_query_ray", "TestMeshQueryRay"),
    ("geometry.test_volume", "TestVolume"),
    ("geometry.test_volume_write", "TestVolumeWrite"),
    ("test_array_reduce", "TestArrayReduce"),
    ("test_bool", "TestBool"),
    ("test_builtins_resolution", "TestBuiltinsResolution"),
    ("test_codegen", "TestCodeGen"),
    ("test_compile_consts", "TestConstants"),
    ("test_conditional", "TestConditional"),
    ("test_copy", "TestCopy"),
    ("test_ctypes", "TestCTypes"),
    ("test_devices", "TestDevices"),
    ("test_fp16", "TestFp16"),
    ("test_func", "TestFunc"),
    ("test_generics", "TestGenerics"),
    ("test_grad_customs", "TestGradCustoms"),
    ("test_grad_debug", "TestGradDebug"),
    ("test_indexedarray", "TestIndexedArray"),
    ("test_launch", "TestLaunch"),
    ("test_lvalue", "TestLValue"),
    ("test_mat_lite", "TestMatLite"),
    ("test_math", "TestMath"),
    ("test_mlp", "TestMLP"),
    ("test_module_hashing", "TestModuleHashing"),
    ("test_modules_lite", "TestModuleLite"),
    ("test_noise", "TestNoise"),
    ("test_operators", "TestOperators"),
    ("test_rand", "TestRand"),
    ("test_rounding", "TestRounding"),
    ("test_runlength_encode", "TestRunlengthEncode"),
    ("test_scalar_ops", "TestScalarOps"),
    ("test_snippet", "TestSnippets"),
    ("test_static", "TestStatic"),
    ("test_tape", "TestTape"),
    ("test_transient_module", "TestTransientModule"),
    ("test_types", "TestTypes"),
    ("test_utils", "TestUtils"),
    ("test_vec_lite", "TestVecLite"),
    ("tile.test_tile_reduce", "TestTileReduce"),
)


test_clss = []
for module_name, cls_name in TEST_DESCS:
    module = importlib.import_module(f"warp.tests.{module_name}")
    cls = getattr(module, cls_name)

    # Change the base class from unittest.TestCase
    cls.__bases__ = (omni.kit.test.AsyncTestCase,)

    test_clss.append(cls)


# Each test class needs to be defined at the module level to be found by
# the test runners.
locals().update({str(i): x for i, x in enumerate(test_clss)})

# Clear caches to ensure determinism.
wp.clear_kernel_cache()
wp.clear_lto_cache()
