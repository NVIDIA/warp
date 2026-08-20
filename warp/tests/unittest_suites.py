# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp Test Suites

This file is intended to define functions that return TestSuite objects, which
can be used in parallel or serial unit tests (with optional code coverage)
"""

# This module is a lazy suite manifest: importing it should not import and
# register every Warp test module. Suite factories import only the selected
# test modules after callers have configured modes and, for serial runs,
# cleared Warp caches. Keep this exception local to this file.
# ruff: noqa: PLC0415

import os
import unittest

START_DIRECTORY = os.path.realpath(os.path.dirname(__file__))
TOP_LEVEL_DIRECTORY = os.path.realpath(os.path.join(START_DIRECTORY, "..", ".."))


def _create_suite_from_test_classes(test_loader, test_classes):
    suite = unittest.TestSuite()

    for test in test_classes:
        sub_suite = unittest.TestSuite()
        # Note that the test_loader might have testNamePatterns set
        sub_suite.addTest(test_loader.loadTestsFromTestCase(test))
        suite.addTest(sub_suite)

    return suite


def auto_discover_suite(loader=unittest.defaultTestLoader, pattern="test*.py"):
    """Uses unittest auto-discovery to build a test suite (test_*.py pattern)"""

    return loader.discover(start_dir=START_DIRECTORY, pattern=pattern, top_level_dir=TOP_LEVEL_DIRECTORY)


def _iter_class_suites(test_suite):
    """Iterate class-level test suites - test suites that contains test cases

    From unittest_parallel.py
    """
    has_cases = any(isinstance(suite, unittest.TestCase) for suite in test_suite)
    if has_cases:
        yield test_suite
    else:
        for suite in test_suite:
            yield from _iter_class_suites(suite)


def compare_unittest_suites(
    test_loader: unittest.TestLoader, test_suite_name: str, reference_suite: unittest.TestSuite
) -> unittest.TestSuite:
    """Print the test classes in ``reference_suite`` that the named suite omits.

    Returns:
        The test suite built for ``test_suite_name``.
    """

    test_suite_fn = SUITE_FACTORIES[test_suite_name]

    test_suite = test_suite_fn(test_loader)

    test_suite_classes_str = {
        type(test_suite._tests[0]).__name__
        for test_suite in list(_iter_class_suites(test_suite))
        if test_suite.countTestCases() > 0
    }

    reference_suite_classes_str = {
        type(test_suite._tests[0]).__name__
        for test_suite in list(_iter_class_suites(reference_suite))
        if test_suite.countTestCases() > 0
    }

    set_difference = reference_suite_classes_str - test_suite_classes_str

    print(f"Selected test suite '{test_suite_name}'")
    if len(set_difference) > 0:
        print(f"Test suite '{test_suite_name}' omits the following test classes:")
        for test_entry in set_difference:
            print(f"    {test_entry}")

    return test_suite


def default_suite(test_loader: unittest.TestLoader = unittest.defaultTestLoader):
    """Example of a manually constructed test suite.

    Intended to be modified to create additional test suites
    """
    # Keep suite member imports lazy. The serial runner clears Warp caches
    # before building this suite, and importing all test modules here registers
    # their kernels only after that point.
    from warp.tests.cuda.test_array_fill_capture import TestArrayFillCapture
    from warp.tests.cuda.test_async import TestAsync
    from warp.tests.cuda.test_capture_mode import TestCaptureMode
    from warp.tests.cuda.test_clang_cuda import TestClangCUDA
    from warp.tests.cuda.test_cluster_dim import TestClusterDim
    from warp.tests.cuda.test_cuda_arch_suffix import TestCudaArchSuffix
    from warp.tests.cuda.test_cuda_max_registers import TestCudaMaxRegisters
    from warp.tests.cuda.test_cuda_smem_spilling import TestCudaSmemSpilling
    from warp.tests.cuda.test_kernel_attributes import TestKernelAttributes
    from warp.tests.cuda.test_mempool import TestMempool
    from warp.tests.cuda.test_multigpu import TestMultiGPU
    from warp.tests.cuda.test_occupancy import TestOccupancy
    from warp.tests.cuda.test_peer import TestPeer
    from warp.tests.cuda.test_pinned import TestPinned
    from warp.tests.cuda.test_streams import TestStreams
    from warp.tests.cuda.test_texture import TestTexture
    from warp.tests.cuda.test_unified_memory import TestUnifiedMemory
    from warp.tests.deterministic.test_deterministic_backward import TestDeterministicBackward
    from warp.tests.deterministic.test_deterministic_counter import TestDeterministicCounter
    from warp.tests.deterministic.test_deterministic_graph_capture import TestDeterministicGraph
    from warp.tests.deterministic.test_deterministic_options import TestDeterministicOptions
    from warp.tests.deterministic.test_deterministic_scatter import TestDeterministicScatter
    from warp.tests.fem.test_fem_examples import TestFemDiffusionExamples, TestFemExamples
    from warp.tests.fem.test_fem_field import TestFemField
    from warp.tests.fem.test_fem_fp64 import TestFemFp64
    from warp.tests.fem.test_fem_geometry import TestFemGeometry
    from warp.tests.fem.test_fem_integrate import TestFemIntegrate
    from warp.tests.fem.test_fem_linalg import TestFemLinalg
    from warp.tests.fem.test_fem_multi_env import TestFemMultiEnv
    from warp.tests.fem.test_fem_quadrature import TestFemQuadrature
    from warp.tests.fem.test_fem_shape import TestFemShape
    from warp.tests.geometry.test_bvh import TestBvh
    from warp.tests.geometry.test_hash_grid import TestHashGrid
    from warp.tests.geometry.test_marching_cubes import TestMarchingCubes
    from warp.tests.geometry.test_mesh import TestMesh
    from warp.tests.geometry.test_mesh_query_aabb import TestMeshQueryAABBMethods
    from warp.tests.geometry.test_mesh_query_point import TestMeshQueryPoint
    from warp.tests.geometry.test_mesh_query_ray import TestMeshQueryRay
    from warp.tests.geometry.test_volume import TestVolume
    from warp.tests.geometry.test_volume_write import TestVolumeWrite
    from warp.tests.interop.test_dlpack import TestDLPack
    from warp.tests.interop.test_jax import TestJax
    from warp.tests.interop.test_torch import TestTorch
    from warp.tests.matrix.test_mat import TestMat
    from warp.tests.matrix.test_mat_assign_copy import TestMatAssignCopy
    from warp.tests.matrix.test_mat_basics import TestMatBasics
    from warp.tests.matrix.test_mat_constructors import TestMatConstructors
    from warp.tests.matrix.test_mat_elementwise_ops import TestMatElementwiseOps
    from warp.tests.matrix.test_mat_linalg import TestMatLinalg
    from warp.tests.matrix.test_mat_lite import TestMatLite
    from warp.tests.test_adam import TestAdam
    from warp.tests.test_allocation_tracker import TestAllocTracker
    from warp.tests.test_allocator import (
        TestAllocatorProtocol,
        TestCustomAllocator,
        TestRmmAllocator,
        TestTorchAllocator,
    )
    from warp.tests.test_apic import TestApic
    from warp.tests.test_apic_mesh import TestApicMesh
    from warp.tests.test_apic_utility_algorithms import TestApicSegmentedSort, TestApicUtilityAlgorithms
    from warp.tests.test_arithmetic import TestArithmetic
    from warp.tests.test_array import TestArray
    from warp.tests.test_array_reduce import TestArrayReduce
    from warp.tests.test_atomic import TestAtomic
    from warp.tests.test_atomic_bitwise import TestAtomicBitwise
    from warp.tests.test_atomic_cas import TestAtomicCAS
    from warp.tests.test_bf16 import TestBf16, TestBf16MlDtypes
    from warp.tests.test_block_dim_dispatch import TestBlockDimDispatch
    from warp.tests.test_bool import TestBool
    from warp.tests.test_builtins_resolution import TestBuiltinsResolution
    from warp.tests.test_closest_point_edge_edge import TestClosestPointEdgeEdgeMethods
    from warp.tests.test_codegen import TestCodeGen
    from warp.tests.test_codegen_instancing import TestCodeGenInstancing
    from warp.tests.test_compilation import TestCompilation
    from warp.tests.test_compile_consts import TestConstants
    from warp.tests.test_composite_component_adjoint import TestCompositeComponentAdjoint
    from warp.tests.test_conditional import TestConditional
    from warp.tests.test_constant_precision import TestConstantPrecision
    from warp.tests.test_context import TestContext
    from warp.tests.test_copy import TestCopy
    from warp.tests.test_core_library_binary import TestCoreLibraryBinary
    from warp.tests.test_cpu_precompiled_headers import TestCpuPrecompiledHeaders
    from warp.tests.test_ctypes import TestCTypes
    from warp.tests.test_cuda_profiler import TestCudaProfiler
    from warp.tests.test_dense import TestDense
    from warp.tests.test_devices import TestDevices
    from warp.tests.test_diagnostics import TestDiagnostics
    from warp.tests.test_enum import TestEnum
    from warp.tests.test_examples import (
        TestCoreExamples,
        TestOptimExamples,
    )
    from warp.tests.test_external_build import TestExternalBuild
    from warp.tests.test_fabricarray import TestFabricArray
    from warp.tests.test_factory_style_array_annotations import TestFactoryStyleArrayAnnotations
    from warp.tests.test_fast_math import TestFastMath
    from warp.tests.test_fixedarray import TestFixedArray
    from warp.tests.test_fp16 import TestFp16
    from warp.tests.test_func import TestFunc
    from warp.tests.test_func_parameter_targets import TestFuncParameterTargets
    from warp.tests.test_future_annotations import TestFutureAnnotations
    from warp.tests.test_generics import TestGenerics
    from warp.tests.test_grad import TestGrad
    from warp.tests.test_grad_customs import TestGradCustoms
    from warp.tests.test_grad_debug import TestGradDebug
    from warp.tests.test_graph import TestGraph
    from warp.tests.test_import import TestImport
    from warp.tests.test_indexedarray import TestIndexedArray
    from warp.tests.test_intersect import TestIntersect
    from warp.tests.test_iter import TestIter
    from warp.tests.test_kernel_cache import TestKernelCache
    from warp.tests.test_large import TestLarge
    from warp.tests.test_launch import TestLaunch
    from warp.tests.test_lerp import TestLerp
    from warp.tests.test_linear_solvers import TestLinearSolvers
    from warp.tests.test_logger import TestLogger
    from warp.tests.test_lvalue import TestLValue
    from warp.tests.test_math import TestMath
    from warp.tests.test_module_contamination import TestModuleContamination
    from warp.tests.test_module_hashing import TestModuleHashing
    from warp.tests.test_module_parallel_load import (
        TestModuleParallelLoad,
        TestParallelLoadSharedHelper,
    )
    from warp.tests.test_modules_lite import TestModuleLite
    from warp.tests.test_noise import TestNoise
    from warp.tests.test_operators import TestOperators
    from warp.tests.test_options import TestOptions
    from warp.tests.test_overwrite import TestOverwrite
    from warp.tests.test_print import TestPrint
    from warp.tests.test_quat import TestQuat
    from warp.tests.test_rand import TestRand
    from warp.tests.test_ref import TestRef
    from warp.tests.test_reload import TestReload
    from warp.tests.test_rounding import TestRounding
    from warp.tests.test_runlength_encode import TestRunlengthEncode
    from warp.tests.test_sanitize import TestSanitize
    from warp.tests.test_scalar_ops import TestScalarOps
    from warp.tests.test_sgd import TestSGD
    from warp.tests.test_smoothstep import TestSmoothstep
    from warp.tests.test_snippet import TestSnippets
    from warp.tests.test_sparse import TestSparse
    from warp.tests.test_spatial import TestSpatial
    from warp.tests.test_special_values import TestSpecialValues
    from warp.tests.test_static import TestStatic
    from warp.tests.test_struct import TestStruct
    from warp.tests.test_subscript_types import TestSubscriptTypes
    from warp.tests.test_tape import TestTape
    from warp.tests.test_template_launch_bounds import TestTemplateLaunchBounds
    from warp.tests.test_transient_module import TestTransientModule
    from warp.tests.test_triangle_closest_point import TestTriangleClosestPoint
    from warp.tests.test_types import TestTypes
    from warp.tests.test_unique_module import TestUniqueModule
    from warp.tests.test_utils import TestUtils
    from warp.tests.test_vec import TestVec
    from warp.tests.test_vec_constructors import TestVecConstructors
    from warp.tests.test_vec_lite import TestVecLite
    from warp.tests.test_vec_scalar_ops import TestVecScalarOps
    from warp.tests.test_verify_fp import TestVerifyFP
    from warp.tests.test_version import TestVersion
    from warp.tests.tile.test_tile import TestTile
    from warp.tests.tile.test_tile_atomic_bitwise import TestTileAtomicBitwise
    from warp.tests.tile.test_tile_cholesky import TestTileCholesky
    from warp.tests.tile.test_tile_cholesky_no_mathdx import TestTileCholeskyNoMathDx
    from warp.tests.tile.test_tile_composite_row import TestTileCompositeRow
    from warp.tests.tile.test_tile_empty import TestTileEmpty
    from warp.tests.tile.test_tile_fft import TestTileFFT
    from warp.tests.tile.test_tile_fft_no_mathdx import TestTileFFTNoMathDx
    from warp.tests.tile.test_tile_func_arg import TestTileFuncArg
    from warp.tests.tile.test_tile_fused_ops import TestTileFusedOps
    from warp.tests.tile.test_tile_large_offsets import TestTileLargeOffsets
    from warp.tests.tile.test_tile_load import TestTileLoad
    from warp.tests.tile.test_tile_load_assign import TestTileLoadAssign
    from warp.tests.tile.test_tile_load_extract import TestTileLoadExtract
    from warp.tests.tile.test_tile_load_indexed import TestTileLoadIndexed
    from warp.tests.tile.test_tile_load_vectorized import TestTileLoadVectorized
    from warp.tests.tile.test_tile_low_precision_stack import TestTileLowPrecisionStack
    from warp.tests.tile.test_tile_mathdx import TestTileMathDx
    from warp.tests.tile.test_tile_matmul import TestTileMatmul
    from warp.tests.tile.test_tile_matmul_no_mathdx import TestTileMatmulNoMathDx
    from warp.tests.tile.test_tile_matmul_strides import TestTileMatmulStrides
    from warp.tests.tile.test_tile_oob import TestTileOOB
    from warp.tests.tile.test_tile_reduce import TestTileReduce
    from warp.tests.tile.test_tile_shared_memory import (
        TestTileSharedMemory,
        TestTileSharedMemoryMessages,
    )
    from warp.tests.tile.test_tile_solve import TestTileSolve
    from warp.tests.tile.test_tile_solve_no_mathdx import TestTileSolveNoMathDx
    from warp.tests.tile.test_tile_sort import TestTileSort
    from warp.tests.tile.test_tile_stack import TestTileStack
    from warp.tests.tile.test_tile_struct import TestTileStruct
    from warp.tests.tile.test_tile_view import TestTileView

    test_classes = [
        TestAdam,
        TestAllocTracker,
        TestAllocatorProtocol,
        TestApic,
        TestApicMesh,
        TestApicSegmentedSort,
        TestApicUtilityAlgorithms,
        TestArithmetic,
        TestArray,
        TestArrayFillCapture,
        TestArrayReduce,
        TestAsync,
        TestAtomic,
        TestAtomicBitwise,
        TestAtomicCAS,
        TestBf16,
        TestBf16MlDtypes,
        TestBlockDimDispatch,
        TestBool,
        TestBuiltinsResolution,
        TestBvh,
        TestCaptureMode,
        TestClangCUDA,
        TestClosestPointEdgeEdgeMethods,
        TestClusterDim,
        TestCodeGen,
        TestCodeGenInstancing,
        TestCompilation,
        TestCompositeComponentAdjoint,
        TestConstants,
        TestConditional,
        TestConstantPrecision,
        TestContext,
        TestCopy,
        TestCoreLibraryBinary,
        TestCpuPrecompiledHeaders,
        TestCTypes,
        TestCudaArchSuffix,
        TestCudaProfiler,
        TestCustomAllocator,
        TestDense,
        TestDeterministicBackward,
        TestDeterministicCounter,
        TestDeterministicGraph,
        TestDeterministicOptions,
        TestDeterministicScatter,
        TestDevices,
        TestDiagnostics,
        TestDLPack,
        TestEnum,
        TestExternalBuild,
        TestCoreExamples,
        TestOptimExamples,
        TestFactoryStyleArrayAnnotations,
        TestFabricArray,
        TestFastMath,
        TestFemDiffusionExamples,
        TestFemExamples,
        TestFemField,
        TestFemFp64,
        TestFemGeometry,
        TestFemIntegrate,
        TestFemLinalg,
        TestFemMultiEnv,
        TestFemQuadrature,
        TestFemShape,
        TestFixedArray,
        TestFp16,
        TestFunc,
        TestFuncParameterTargets,
        TestFutureAnnotations,
        TestGenerics,
        TestGrad,
        TestGradCustoms,
        TestGradDebug,
        TestGraph,
        TestHashGrid,
        TestImport,
        TestIndexedArray,
        TestIntersect,
        TestIter,
        TestJax,
        TestKernelAttributes,
        TestKernelCache,
        TestLarge,
        TestLaunch,
        TestLerp,
        TestLinearSolvers,
        TestLogger,
        TestLValue,
        TestMarchingCubes,
        TestMat,
        TestMatConstructors,
        TestMatLite,
        TestMatAssignCopy,
        TestMatBasics,
        TestMatElementwiseOps,
        TestMatLinalg,
        TestMath,
        TestCudaMaxRegisters,
        TestMempool,
        TestMesh,
        TestMeshQueryAABBMethods,
        TestMeshQueryPoint,
        TestMeshQueryRay,
        TestModuleContamination,
        TestModuleHashing,
        TestModuleLite,
        TestModuleParallelLoad,
        TestMultiGPU,
        TestNoise,
        TestOccupancy,
        TestOperators,
        TestOptions,
        TestOverwrite,
        TestParallelLoadSharedHelper,
        TestPeer,
        TestPinned,
        TestPrint,
        TestQuat,
        TestRand,
        TestRef,
        TestReload,
        TestRmmAllocator,
        TestRounding,
        TestRunlengthEncode,
        TestSanitize,
        TestScalarOps,
        TestSGD,
        TestCudaSmemSpilling,
        TestSmoothstep,
        TestSnippets,
        TestSparse,
        TestSpatial,
        TestSpecialValues,
        TestStatic,
        TestStreams,
        TestStruct,
        TestSubscriptTypes,
        TestTape,
        TestTemplateLaunchBounds,
        TestTexture,
        TestTile,
        TestTileAtomicBitwise,
        TestTileCholesky,
        TestTileCholeskyNoMathDx,
        TestTileCompositeRow,
        TestTileEmpty,
        TestTileFFT,
        TestTileFFTNoMathDx,
        TestTileFuncArg,
        TestTileFusedOps,
        TestTileLargeOffsets,
        TestTileLoad,
        TestTileLoadAssign,
        TestTileLoadExtract,
        TestTileLoadIndexed,
        TestTileLoadVectorized,
        TestTileLowPrecisionStack,
        TestTileMathDx,
        TestTileMatmul,
        TestTileMatmulNoMathDx,
        TestTileMatmulStrides,
        TestTileOOB,
        TestTileReduce,
        TestTileSharedMemory,
        TestTileSharedMemoryMessages,
        TestTileSolve,
        TestTileSolveNoMathDx,
        TestTileSort,
        TestTileStack,
        TestTileStruct,
        TestTileView,
        TestTorch,
        TestTorchAllocator,
        TestTransientModule,
        TestTriangleClosestPoint,
        TestTypes,
        TestUniqueModule,
        TestUnifiedMemory,
        TestUtils,
        TestVec,
        TestVecConstructors,
        TestVecLite,
        TestVecScalarOps,
        TestVerifyFP,
        TestVersion,
        TestVolume,
        TestVolumeWrite,
    ]

    return _create_suite_from_test_classes(test_loader, test_classes)


def debug_suite(test_loader: unittest.TestLoader = unittest.defaultTestLoader):
    """Focused test suite for validating warp.config.mode = "debug".

    Debug mode compiles kernels without optimizations and with full debug info,
    which is significantly slower. This suite targets tests most likely to break
    specifically in debug mode: codegen, gradients, generics, tile operations,
    and complex kernel compilation patterns.

    Usage:
        python -m warp.tests --suite debug --warp-debug

    Known debug-mode failures:
        - fem.test_fem_geometry: test_deformed_geometry_cuda_0 exceeds the
          available memory on a 24 GiB MIG device

    tile.test_tile_struct is no longer listed above:
    tile.test_tile_low_precision_stack now covers its CUDA error 719 case
    without pulling that memory-heavy class into the parallel debug suite.

    The GitLab GPU job runs up to eight worker processes that share one GPU.
    It uses ``--isolate-test-processes`` so each process runs one class-level
    suite and exits, releasing its CUDA context before the worker slot is
    reused. The runner model is not pinned here and most GitLab GPUs have
    48 GB or more, so the budget is set by the smallest device the suite should
    still fit on: 24 GiB, which covers a quarter-slice MIG partition of an RTX
    PRO 6000 as well as a developer machine with an RTX 3090. A class is
    excluded when its isolated peak device memory exceeds the per-worker share
    of roughly 3 GiB. Measure a candidate by running its class alone in debug
    mode before adding it here.

    Excluded on that basis:
        - test_examples, fem.test_fem_examples, cuda.test_async, and
          fem.test_fem_fp64 spawn subprocesses that each add a CUDA context
        - fem.test_fem_integrate, fem.test_fem_multi_env, test_sparse, and
          test_copy, whose test_copy_large_stride allocates 6 GB on its own
    """
    # Keep imports lazy so importing this helper does not register unrelated
    # test kernels before a focused debug suite is selected.
    from warp.tests.aot.test_module_aot import TestModuleAOT
    from warp.tests.cuda.test_array_fill_capture import TestArrayFillCapture
    from warp.tests.cuda.test_capture_mode import TestCaptureMode
    from warp.tests.cuda.test_clang_cuda import TestClangCUDA
    from warp.tests.cuda.test_cluster_dim import TestClusterDim
    from warp.tests.cuda.test_conditional_captures import TestConditionalCaptures
    from warp.tests.cuda.test_cuda_arch_suffix import TestCudaArchSuffix
    from warp.tests.cuda.test_ipc import TestIpc
    from warp.tests.cuda.test_mempool import TestMempool
    from warp.tests.cuda.test_occupancy import TestOccupancy
    from warp.tests.cuda.test_peer import TestPeer
    from warp.tests.cuda.test_pinned import TestPinned
    from warp.tests.cuda.test_streams import TestStreams
    from warp.tests.cuda.test_texture import TestTexture
    from warp.tests.cuda.test_unified_memory import TestUnifiedMemory
    from warp.tests.deterministic.test_deterministic_backward import TestDeterministicBackward
    from warp.tests.deterministic.test_deterministic_counter import TestDeterministicCounter
    from warp.tests.deterministic.test_deterministic_graph_capture import TestDeterministicGraph
    from warp.tests.deterministic.test_deterministic_options import TestDeterministicOptions
    from warp.tests.deterministic.test_deterministic_scatter import TestDeterministicScatter
    from warp.tests.fem.test_fem_linalg import TestFemLinalg
    from warp.tests.geometry.test_bvh import TestBvh
    from warp.tests.geometry.test_grouped_bvh import TestGroupedBvh
    from warp.tests.geometry.test_hash_grid import TestHashGrid
    from warp.tests.geometry.test_marching_cubes import TestMarchingCubes
    from warp.tests.geometry.test_mesh import TestMesh
    from warp.tests.geometry.test_mesh_query_point import TestMeshQueryPoint
    from warp.tests.geometry.test_mesh_query_ray import TestMeshQueryRay
    from warp.tests.geometry.test_volume import TestVolume
    from warp.tests.geometry.test_volume_write import TestVolumeWrite
    from warp.tests.interop.test_dlpack import TestDLPack
    from warp.tests.matrix.test_mat import TestMat
    from warp.tests.matrix.test_mat_assign_copy import TestMatAssignCopy
    from warp.tests.matrix.test_mat_basics import TestMatBasics
    from warp.tests.matrix.test_mat_constructors import TestMatConstructors
    from warp.tests.matrix.test_mat_elementwise_ops import TestMatElementwiseOps
    from warp.tests.matrix.test_mat_linalg import TestMatLinalg
    from warp.tests.matrix.test_mat_lite import TestMatLite
    from warp.tests.test_adam import TestAdam
    from warp.tests.test_allocation_tracker import TestAllocTracker
    from warp.tests.test_allocator import (
        TestAllocatorProtocol,
        TestCustomAllocator,
        TestRmmAllocator,
        TestTorchAllocator,
    )
    from warp.tests.test_apic import TestApic
    from warp.tests.test_apic_mesh import TestApicMesh
    from warp.tests.test_apic_utility_algorithms import TestApicSegmentedSort, TestApicUtilityAlgorithms
    from warp.tests.test_arithmetic import TestArithmetic
    from warp.tests.test_array import TestArray
    from warp.tests.test_array_reduce import TestArrayReduce
    from warp.tests.test_assert import TestAssertDebug, TestAssertModeSwitch, TestAssertRelease
    from warp.tests.test_atomic import TestAtomic
    from warp.tests.test_atomic_bitwise import TestAtomicBitwise
    from warp.tests.test_atomic_cas import TestAtomicCAS
    from warp.tests.test_bf16 import TestBf16, TestBf16MlDtypes
    from warp.tests.test_block_dim_dispatch import TestBlockDimDispatch
    from warp.tests.test_bool import TestBool
    from warp.tests.test_builtins_resolution import TestBuiltinsResolution
    from warp.tests.test_closest_point_edge_edge import TestClosestPointEdgeEdgeMethods
    from warp.tests.test_codegen import TestCodeGen
    from warp.tests.test_codegen_instancing import TestCodeGenInstancing
    from warp.tests.test_coloring import TestColoring
    from warp.tests.test_compilation import TestCompilation
    from warp.tests.test_compile_consts import TestConstants
    from warp.tests.test_composite_component_adjoint import TestCompositeComponentAdjoint
    from warp.tests.test_conditional import TestConditional
    from warp.tests.test_constant_precision import TestConstantPrecision
    from warp.tests.test_context import TestContext
    from warp.tests.test_cpu_precompiled_headers import TestCpuPrecompiledHeaders
    from warp.tests.test_ctypes import TestCTypes
    from warp.tests.test_cuda_profiler import TestCudaProfiler
    from warp.tests.test_dense import TestDense
    from warp.tests.test_devices import TestDevices
    from warp.tests.test_diagnostics import TestDiagnostics
    from warp.tests.test_enum import TestEnum
    from warp.tests.test_fabricarray import TestFabricArray
    from warp.tests.test_factory_style_array_annotations import TestFactoryStyleArrayAnnotations
    from warp.tests.test_fast_math import TestFastMath
    from warp.tests.test_fixedarray import TestFixedArray
    from warp.tests.test_fp16 import TestFp16
    from warp.tests.test_func import TestFunc
    from warp.tests.test_func_parameter_targets import TestFuncParameterTargets
    from warp.tests.test_future_annotations import TestFutureAnnotations
    from warp.tests.test_generics import TestGenerics
    from warp.tests.test_grad import TestGrad
    from warp.tests.test_grad_customs import TestGradCustoms
    from warp.tests.test_grad_debug import TestGradDebug
    from warp.tests.test_graph import TestGraph
    from warp.tests.test_implicit_init import (
        TestImplicitInitArrayFromData,
        TestImplicitInitArrayFromPtr,
        TestImplicitInitBuiltinCall,
        TestImplicitInitGetCudaDeviceCount,
        TestImplicitInitGetCudaDevices,
        TestImplicitInitGetDevice,
        TestImplicitInitGetDevices,
        TestImplicitInitGetPreferredDevice,
        TestImplicitInitIsCpuAvailable,
        TestImplicitInitIsCudaAvailable,
        TestImplicitInitIsDeviceAvailable,
        TestImplicitInitIsMempoolAccessEnabled,
        TestImplicitInitIsMempoolAccessSupported,
        TestImplicitInitIsMempoolEnabled,
        TestImplicitInitIsMempoolSupported,
        TestImplicitInitIsPeerAccessEnabled,
        TestImplicitInitIsPeerAccessSupported,
        TestImplicitInitLaunch,
        TestImplicitInitSetDevice,
        TestImplicitInitStructMemberInit,
        TestImplicitInitTape,
    )
    from warp.tests.test_import import TestImport
    from warp.tests.test_indexedarray import TestIndexedArray
    from warp.tests.test_intersect import TestIntersect
    from warp.tests.test_iter import TestIter
    from warp.tests.test_kernel_cache import TestKernelCache
    from warp.tests.test_launch import TestLaunch
    from warp.tests.test_lerp import TestLerp
    from warp.tests.test_logger import TestLogger
    from warp.tests.test_lvalue import TestLValue
    from warp.tests.test_map import TestMap, TestMapDebug
    from warp.tests.test_math import TestMath
    from warp.tests.test_module_contamination import TestModuleContamination
    from warp.tests.test_module_hashing import (
        TestModuleHasherKernelOptions,
        TestModuleHashing,
        TestOptionResolution,
    )
    from warp.tests.test_module_parallel_load import TestModuleParallelLoad, TestParallelLoadSharedHelper
    from warp.tests.test_modules_lite import TestModuleLite
    from warp.tests.test_noise import TestNoise
    from warp.tests.test_operators import TestOperators
    from warp.tests.test_options import TestOptions
    from warp.tests.test_overwrite import TestOverwrite
    from warp.tests.test_print import TestPrint
    from warp.tests.test_quat import TestQuat
    from warp.tests.test_quat_assign_copy import TestQuatAssignCopy
    from warp.tests.test_rand import TestRand
    from warp.tests.test_ref import TestRef
    from warp.tests.test_reload import TestReload
    from warp.tests.test_rounding import TestRounding
    from warp.tests.test_runlength_encode import TestRunlengthEncode
    from warp.tests.test_scalar_ops import TestScalarOps
    from warp.tests.test_sgd import TestSGD
    from warp.tests.test_smoothstep import TestSmoothstep
    from warp.tests.test_snippet import TestSnippets
    from warp.tests.test_spatial import TestSpatial
    from warp.tests.test_spatial_assign_copy import TestSpatialAssignCopy
    from warp.tests.test_special_values import TestSpecialValues
    from warp.tests.test_static import TestStatic
    from warp.tests.test_struct import TestStruct
    from warp.tests.test_subscript_types import TestSubscriptTypes
    from warp.tests.test_tape import TestTape
    from warp.tests.test_template_launch_bounds import TestTemplateLaunchBounds
    from warp.tests.test_transient_module import TestTransientModule
    from warp.tests.test_triangle_closest_point import TestTriangleClosestPoint
    from warp.tests.test_tuple import TestTuple
    from warp.tests.test_types import TestTypes
    from warp.tests.test_unique_module import TestUniqueModule
    from warp.tests.test_unpack import TestUnpack
    from warp.tests.test_utils import TestUtils
    from warp.tests.test_vec import TestVec
    from warp.tests.test_vec_assign_copy import TestVecAssignCopy
    from warp.tests.test_vec_constructors import TestVecConstructors
    from warp.tests.test_vec_lite import TestVecLite
    from warp.tests.test_vec_scalar_ops import TestVecScalarOps
    from warp.tests.test_verify_fp import TestVerifyFP
    from warp.tests.test_version import TestVerifyLibraryVersion, TestVersion
    from warp.tests.tile.test_tile import TestTile
    from warp.tests.tile.test_tile_atomic_bitwise import TestTileAtomicBitwise
    from warp.tests.tile.test_tile_cholesky import TestTileCholesky
    from warp.tests.tile.test_tile_cholesky_no_mathdx import TestTileCholeskyNoMathDx
    from warp.tests.tile.test_tile_composite_row import TestTileCompositeRow
    from warp.tests.tile.test_tile_empty import TestTileEmpty
    from warp.tests.tile.test_tile_fft import TestTileFFT
    from warp.tests.tile.test_tile_fft_no_mathdx import TestTileFFTNoMathDx
    from warp.tests.tile.test_tile_func_arg import TestTileFuncArg
    from warp.tests.tile.test_tile_fused_ops import TestTileFusedOps
    from warp.tests.tile.test_tile_large_offsets import TestTileLargeOffsets
    from warp.tests.tile.test_tile_load import TestTileLoad
    from warp.tests.tile.test_tile_load_assign import TestTileLoadAssign
    from warp.tests.tile.test_tile_load_extract import TestTileLoadExtract
    from warp.tests.tile.test_tile_load_indexed import TestTileLoadIndexed
    from warp.tests.tile.test_tile_load_vectorized import TestTileLoadVectorized
    from warp.tests.tile.test_tile_low_precision_stack import TestTileLowPrecisionStack
    from warp.tests.tile.test_tile_mathdx import TestTileMathDx
    from warp.tests.tile.test_tile_matmul import TestTileMatmul
    from warp.tests.tile.test_tile_matmul_no_mathdx import TestTileMatmulNoMathDx
    from warp.tests.tile.test_tile_matmul_strides import TestTileMatmulStrides
    from warp.tests.tile.test_tile_oob import TestTileOOB
    from warp.tests.tile.test_tile_reduce import TestTileReduce
    from warp.tests.tile.test_tile_shared_memory import TestTileSharedMemory, TestTileSharedMemoryMessages
    from warp.tests.tile.test_tile_solve import TestTileSolve
    from warp.tests.tile.test_tile_solve_no_mathdx import TestTileSolveNoMathDx
    from warp.tests.tile.test_tile_sort import TestTileSort
    from warp.tests.tile.test_tile_stack import TestTileStack
    from warp.tests.tile.test_tile_view import TestTileView

    test_classes = [
        # Codegen & compilation
        TestBuiltinsResolution,
        TestCodeGen,
        TestCodeGenInstancing,
        TestCompositeComponentAdjoint,
        TestConditional,
        TestConstants,
        TestConstantPrecision,
        TestEnum,
        TestFastMath,
        TestFunc,
        TestFuncParameterTargets,
        TestGenerics,
        TestMath,
        TestModuleHashing,
        TestOperators,
        TestOptions,
        TestRef,
        TestScalarOps,
        TestStatic,
        # Gradients
        TestGrad,
        TestGradCustoms,
        TestGradDebug,
        TestTape,
        # Determinism
        TestDeterministicBackward,
        TestDeterministicCounter,
        TestDeterministicGraph,
        TestDeterministicOptions,
        TestDeterministicScatter,
        # Types & operators
        TestArithmetic,
        TestAtomic,
        TestFp16,
        TestLValue,
        TestStruct,
        TestTypes,
        # Tile (debug-safe)
        TestTile,
        TestTileAtomicBitwise,
        TestTileCompositeRow,
        TestTileFFT,
        TestTileFFTNoMathDx,
        TestTileLoad,
        TestTileLoadAssign,
        TestTileLoadExtract,
        TestTileLoadIndexed,
        TestTileLoadVectorized,
        TestTileLowPrecisionStack,
        TestTileMatmulNoMathDx,
        TestTileMatmulStrides,
        TestTileOOB,
        TestTileReduce,
        TestTileSharedMemory,
        TestTileSharedMemoryMessages,
        TestTileSort,
        TestTileView,
        # Geometry
        TestBvh,
        TestMesh,
        # Matrix
        TestMatLinalg,
        # Subsystem representatives
        TestApicSegmentedSort,
        TestApicUtilityAlgorithms,
        TestLaunch,
        TestSpatial,
        # Additional classes verified to pass under debug mode within the job's time budget.
        TestModuleAOT,
        TestArrayFillCapture,
        TestCaptureMode,
        TestClangCUDA,
        TestClusterDim,
        TestConditionalCaptures,
        TestCudaArchSuffix,
        TestIpc,
        TestMempool,
        TestOccupancy,
        TestPeer,
        TestPinned,
        TestStreams,
        TestTexture,
        TestUnifiedMemory,
        TestFemLinalg,
        TestHashGrid,
        TestMarchingCubes,
        TestVolume,
        TestVolumeWrite,
        TestDLPack,
        TestMatAssignCopy,
        TestMatBasics,
        TestMatConstructors,
        TestMatElementwiseOps,
        TestMatLite,
        TestAdam,
        TestAllocTracker,
        TestAllocatorProtocol,
        TestCustomAllocator,
        TestRmmAllocator,
        TestTorchAllocator,
        TestApic,
        TestApicMesh,
        TestArray,
        TestArrayReduce,
        TestAssertDebug,
        TestAssertModeSwitch,
        TestAssertRelease,
        TestAtomicBitwise,
        TestAtomicCAS,
        TestBf16,
        TestBf16MlDtypes,
        TestBlockDimDispatch,
        TestBool,
        TestClosestPointEdgeEdgeMethods,
        TestColoring,
        TestCompilation,
        TestContext,
        TestCpuPrecompiledHeaders,
        TestCTypes,
        TestCudaProfiler,
        TestDense,
        TestDevices,
        TestDiagnostics,
        TestFactoryStyleArrayAnnotations,
        TestFixedArray,
        TestFutureAnnotations,
        TestImplicitInitArrayFromData,
        TestImplicitInitArrayFromPtr,
        TestImplicitInitBuiltinCall,
        TestImplicitInitGetCudaDeviceCount,
        TestImplicitInitGetCudaDevices,
        TestImplicitInitGetDevice,
        TestImplicitInitGetDevices,
        TestImplicitInitGetPreferredDevice,
        TestImplicitInitIsCpuAvailable,
        TestImplicitInitIsCudaAvailable,
        TestImplicitInitIsDeviceAvailable,
        TestImplicitInitIsMempoolAccessEnabled,
        TestImplicitInitIsMempoolAccessSupported,
        TestImplicitInitIsMempoolEnabled,
        TestImplicitInitIsMempoolSupported,
        TestImplicitInitIsPeerAccessEnabled,
        TestImplicitInitIsPeerAccessSupported,
        TestImplicitInitLaunch,
        TestImplicitInitSetDevice,
        TestImplicitInitStructMemberInit,
        TestImplicitInitTape,
        TestImport,
        TestIndexedArray,
        TestIntersect,
        TestIter,
        TestKernelCache,
        TestLerp,
        TestLogger,
        TestMap,
        TestMapDebug,
        TestModuleContamination,
        TestModuleParallelLoad,
        TestParallelLoadSharedHelper,
        TestModuleLite,
        TestNoise,
        TestOverwrite,
        TestPrint,
        TestQuat,
        TestQuatAssignCopy,
        TestRand,
        TestReload,
        TestRounding,
        TestRunlengthEncode,
        TestSGD,
        TestSmoothstep,
        TestSnippets,
        TestSpatialAssignCopy,
        TestSpecialValues,
        TestSubscriptTypes,
        TestTemplateLaunchBounds,
        TestTransientModule,
        TestTriangleClosestPoint,
        TestTuple,
        TestUniqueModule,
        TestUnpack,
        TestUtils,
        TestVec,
        TestVecAssignCopy,
        TestVecConstructors,
        TestVecLite,
        TestVerifyFP,
        TestVerifyLibraryVersion,
        TestVersion,
        TestTileCholeskyNoMathDx,
        TestTileEmpty,
        TestTileFuncArg,
        TestTileFusedOps,
        TestTileLargeOffsets,
        TestTileSolve,
        TestTileSolveNoMathDx,
        TestTileStack,
        TestGroupedBvh,
        TestMeshQueryPoint,
        TestMeshQueryRay,
        TestMat,
        TestFabricArray,
        TestGraph,
        TestModuleHasherKernelOptions,
        TestOptionResolution,
        TestVecScalarOps,
        TestTileCholesky,
        TestTileMathDx,
        TestTileMatmul,
    ]

    return _create_suite_from_test_classes(test_loader, test_classes)


# Registry of named test-suite factories keyed by the ``--suite`` value that
# selects them. This is the single source of truth for the runner's suite
# choices: ``unittest_parallel`` derives its accepted ``--suite`` values from
# these keys, so a choice cannot be offered without a factory behind it. The
# "autodetect" special mode is intentionally absent; it discovers tests
# dynamically from a file pattern rather than resolving a fixed factory here.
SUITE_FACTORIES = {
    "default": default_suite,
    "debug": debug_suite,
}
