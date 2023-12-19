# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Warp Test Suites

This file is intended to define functions that return TestSuite objects, which
can be used in parallel or serial unit tests (with optional code coverage)
"""

import os
import unittest

START_DIRECTORY = os.path.realpath(os.path.dirname(__file__))
TOP_LEVEL_DIRECTORY = os.path.realpath(os.path.join(START_DIRECTORY, "..", ".."))


def _create_suite_from_test_classes(test_classes):
    suite = unittest.TestSuite()

    for test in test_classes:
        sub_suite = unittest.TestSuite()
        sub_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(test))
        suite.addTest(sub_suite)

    return suite


def auto_discover_suite(loader=unittest.defaultTestLoader, pattern="test*.py"):
    """Uses unittest auto-discovery to build a test suite (test_*.py pattern)"""

    return loader.discover(start_dir=START_DIRECTORY, pattern=pattern, top_level_dir=TOP_LEVEL_DIRECTORY)


def explicit_suite():
    """Example of a manually constructed test suite.

    Intended to be modified to create additional test suites
    """
    from warp.tests.test_adam import TestAdam
    from warp.tests.test_arithmetic import TestArithmetic
    from warp.tests.test_array import TestArray
    from warp.tests.test_array_reduce import TestArrayReduce
    from warp.tests.test_atomic import TestAtomic
    from warp.tests.test_bool import TestBool
    from warp.tests.test_builtins_resolution import TestBuiltinsResolution
    from warp.tests.test_bvh import TestBvh
    from warp.tests.test_closest_point_edge_edge import TestClosestPointEdgeEdgeMethods
    from warp.tests.test_codegen import TestCodeGen
    from warp.tests.test_compile_consts import TestConstants
    from warp.tests.test_conditional import TestConditional
    from warp.tests.test_copy import TestCopy
    from warp.tests.test_ctypes import TestCTypes
    from warp.tests.test_dense import TestDense
    from warp.tests.test_devices import TestDevices
    from warp.tests.test_dlpack import TestDLPack
    from warp.tests.test_examples import TestExamples, TestFemExamples, TestSimExamples
    from warp.tests.test_fabricarray import TestFabricArray
    from warp.tests.test_fast_math import TestFastMath
    from warp.tests.test_fem import TestFem
    from warp.tests.test_fp16 import TestFp16
    from warp.tests.test_func import TestFunc
    from warp.tests.test_generics import TestGenerics
    from warp.tests.test_grad import TestGrad
    from warp.tests.test_grad_customs import TestGradCustoms
    from warp.tests.test_hash_grid import TestHashGrid
    from warp.tests.test_import import TestImport
    from warp.tests.test_indexedarray import TestIndexedArray
    from warp.tests.test_intersect import TestIntersect
    from warp.tests.test_large import TestLarge
    from warp.tests.test_launch import TestLaunch
    from warp.tests.test_lerp import TestLerp
    from warp.tests.test_lvalue import TestLValue
    from warp.tests.test_marching_cubes import TestMarchingCubes
    from warp.tests.test_mat import TestMat
    from warp.tests.test_mat_lite import TestMatLite
    from warp.tests.test_mat_scalar_ops import TestMatScalarOps
    from warp.tests.test_math import TestMath
    from warp.tests.test_matmul import TestMatmul
    from warp.tests.test_matmul_lite import TestMatmulLite
    from warp.tests.test_mesh import TestMesh
    from warp.tests.test_mesh_query_aabb import TestMeshQueryAABBMethods
    from warp.tests.test_mesh_query_point import TestMeshQueryPoint
    from warp.tests.test_mesh_query_ray import TestMeshQueryRay
    from warp.tests.test_mlp import TestMLP
    from warp.tests.test_model import TestModel
    from warp.tests.test_modules_lite import TestModuleLite
    from warp.tests.test_multigpu import TestMultiGPU
    from warp.tests.test_noise import TestNoise
    from warp.tests.test_operators import TestOperators
    from warp.tests.test_options import TestOptions
    from warp.tests.test_pinned import TestPinned
    from warp.tests.test_print import TestPrint
    from warp.tests.test_quat import TestQuat
    from warp.tests.test_rand import TestRand
    from warp.tests.test_reload import TestReload
    from warp.tests.test_rounding import TestRounding
    from warp.tests.test_runlength_encode import TestRunlengthEncode
    from warp.tests.test_smoothstep import TestSmoothstep
    from warp.tests.test_snippet import TestSnippets
    from warp.tests.test_sparse import TestSparse
    from warp.tests.test_spatial import TestSpatial
    from warp.tests.test_streams import TestStreams
    from warp.tests.test_struct import TestStruct
    from warp.tests.test_tape import TestTape
    from warp.tests.test_torch import TestTorch
    from warp.tests.test_transient_module import TestTransientModule
    from warp.tests.test_types import TestTypes
    from warp.tests.test_utils import TestUtils
    from warp.tests.test_vec import TestVec
    from warp.tests.test_vec_lite import TestVecLite
    from warp.tests.test_vec_scalar_ops import TestVecScalarOps
    from warp.tests.test_volume import TestVolume
    from warp.tests.test_volume_write import TestVolumeWrite

    test_classes = [
        TestAdam,
        TestArithmetic,
        TestArray,
        TestArrayReduce,
        TestAtomic,
        TestBool,
        TestBuiltinsResolution,
        TestBvh,
        TestClosestPointEdgeEdgeMethods,
        TestCodeGen,
        TestConstants,
        TestConditional,
        TestCopy,
        TestCTypes,
        TestDense,
        TestDevices,
        TestDLPack,
        TestExamples,
        TestFemExamples,
        TestSimExamples,
        TestFabricArray,
        TestFastMath,
        TestFem,
        TestFp16,
        TestFunc,
        TestGenerics,
        TestGrad,
        TestGradCustoms,
        TestHashGrid,
        TestImport,
        TestIndexedArray,
        TestIntersect,
        TestLarge,
        TestLaunch,
        TestLerp,
        TestLValue,
        TestMarchingCubes,
        TestMat,
        TestMatLite,
        TestMatScalarOps,
        TestMath,
        TestMatmul,
        TestMatmulLite,
        TestMesh,
        TestMeshQueryAABBMethods,
        TestMeshQueryPoint,
        TestMeshQueryRay,
        TestMLP,
        TestModel,
        TestModuleLite,
        TestMultiGPU,
        TestNoise,
        TestOperators,
        TestOptions,
        TestPinned,
        TestPrint,
        TestQuat,
        TestRand,
        TestReload,
        TestRounding,
        TestRunlengthEncode,
        TestSmoothstep,
        TestSparse,
        TestSnippets,
        TestSpatial,
        TestStreams,
        TestStruct,
        TestTape,
        TestTorch,
        TestTransientModule,
        TestTypes,
        TestUtils,
        TestVec,
        TestVecLite,
        TestVecScalarOps,
        TestVolume,
        TestVolumeWrite,
    ]

    return _create_suite_from_test_classes(test_classes)


def kit_suite():
    """Tries to mimic the test suite used for testing omni.warp.core in Kit

    Requires manual updates with test_ext.py for now.
    """
    from warp.tests.test_array import TestArray
    from warp.tests.test_array_reduce import TestArrayReduce
    from warp.tests.test_bvh import TestBvh
    from warp.tests.test_codegen import TestCodeGen
    from warp.tests.test_compile_consts import TestConstants
    from warp.tests.test_conditional import TestConditional
    from warp.tests.test_ctypes import TestCTypes
    from warp.tests.test_devices import TestDevices
    from warp.tests.test_dlpack import TestDLPack
    from warp.tests.test_fabricarray import TestFabricArray
    from warp.tests.test_func import TestFunc
    from warp.tests.test_generics import TestGenerics
    from warp.tests.test_grad_customs import TestGradCustoms
    from warp.tests.test_hash_grid import TestHashGrid
    from warp.tests.test_indexedarray import TestIndexedArray
    from warp.tests.test_launch import TestLaunch
    from warp.tests.test_marching_cubes import TestMarchingCubes
    from warp.tests.test_mat_lite import TestMatLite
    from warp.tests.test_math import TestMath
    from warp.tests.test_matmul_lite import TestMatmulLite
    from warp.tests.test_mesh import TestMesh
    from warp.tests.test_mesh_query_aabb import TestMeshQueryAABBMethods
    from warp.tests.test_mesh_query_point import TestMeshQuery
    from warp.tests.test_mesh_query_ray import TestMeshQueryRay
    from warp.tests.test_modules_lite import TestModuleLite
    from warp.tests.test_noise import TestNoise
    from warp.tests.test_operators import TestOperators
    from warp.tests.test_quat import TestQuat
    from warp.tests.test_rand import TestRand
    from warp.tests.test_rounding import TestRounding
    from warp.tests.test_runlength_encode import TestRunlengthEncode
    from warp.tests.test_sparse import TestSparse
    from warp.tests.test_streams import TestStreams
    from warp.tests.test_tape import TestTape
    from warp.tests.test_transient_module import TestTransientModule
    from warp.tests.test_types import TestTypes
    from warp.tests.test_utils import TestUtils
    from warp.tests.test_vec_lite import TestVecLite
    from warp.tests.test_volume import TestVolume
    from warp.tests.test_volume_write import TestVolumeWrite

    test_classes = [
        TestArray,
        TestArrayReduce,
        TestBvh,
        TestCodeGen,
        TestConstants,
        TestConditional,
        TestCTypes,
        TestDevices,
        TestDLPack,
        TestFabricArray,
        TestFunc,
        TestGenerics,
        TestGradCustoms,
        TestHashGrid,
        TestIndexedArray,
        TestLaunch,
        TestMarchingCubes,
        TestMatLite,
        TestMath,
        TestMatmulLite,
        TestMesh,
        TestMeshQueryAABBMethods,
        TestMeshQuery,
        TestMeshQueryRay,
        TestModuleLite,
        TestNoise,
        TestOperators,
        TestQuat,
        TestRand,
        TestRounding,
        TestRunlengthEncode,
        TestSparse,
        TestStreams,
        TestTape,
        TestTransientModule,
        TestTypes,
        TestUtils,
        TestVecLite,
        TestVolume,
        TestVolumeWrite,
    ]

    return _create_suite_from_test_classes(test_classes)
