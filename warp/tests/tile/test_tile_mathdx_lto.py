# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import numpy as np

import warp as wp
from warp._src.context import ModuleBuilder
from warp.tests.unittest_utils import *

wp.init()  # For wp._src.context.runtime.core.wp_is_mathdx_enabled()

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)
TILE_DIM = 32


# Most kernels below are built through ModuleBuilder only to inspect how many LTOs the tile dispatch generates,
# so they add no NVRTC compilation. They share float32 register-tile parameters and TILE_DIM threads so their LTOs
# reuse the same on-disk cache entries. Because count_module_ltos() counts the entire owning module, unique modules
# isolate the enabled/disabled comparisons. The wp.grad() regression additionally launches its kernel to verify that
# retained backward LTOs execute correctly.


@wp.kernel(module="unique")
def tile_matmul_backward_enabled_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(0, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, 0), storage="register")
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel(module="unique", enable_backward=False)
def tile_matmul_kernel_option_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(0, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, 0), storage="register")
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel(module="unique", module_options={"enable_backward": False})
def tile_matmul_module_option_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(0, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, 0), storage="register")
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.func
def tile_matmul_helper(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(0, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, 0), storage="register")
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel(module="unique")
def tile_matmul_via_func_enabled_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    tile_matmul_helper(A, B, C)


@wp.kernel(module="unique", enable_backward=False)
def tile_matmul_via_func_disabled_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    tile_matmul_helper(A, B, C)


@wp.func
def tile_matmul_grad_helper(x: float):
    a = wp.tile_ones(shape=(TILE_M, TILE_M), dtype=wp.float32, storage="register") * x
    c = wp.tile_zeros(shape=(TILE_M, TILE_M), dtype=wp.float32)
    wp.tile_matmul(a, a, c)
    return wp.tile_sum(c)[0]


@wp.func
def tile_matmul_grad_target(x: float):
    return tile_matmul_grad_helper(x)


@wp.kernel(module="tile_mathdx_grad", enable_backward=False)
def tile_matmul_grad_forward_kernel(x: wp.array[float], out: wp.array[float]):
    out[0] = tile_matmul_grad_target(x[0])


@wp.kernel(module="tile_mathdx_grad", enable_backward=False)
def tile_matmul_via_grad_kernel(
    x: wp.array[float],
    grad_x: wp.array[float],
):
    grad_x[0] = wp.grad(tile_matmul_grad_target)(x[0])


# Both kernels share a single module on purpose, so they also share one memoized tile_matmul_helper body.
# module="unique" would give each its own module and defeat the test.
@wp.kernel(module="tile_mathdx_mixed_backward")
def tile_matmul_mixed_enabled_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    tile_matmul_helper(A, B, C)


@wp.kernel(module="tile_mathdx_mixed_backward", enable_backward=False)
def tile_matmul_mixed_disabled_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    tile_matmul_helper(A, B, C)


def count_module_ltos(kernel, device):
    """Return how many LTOs ``ModuleBuilder`` generates for the module owning ``kernel``.

    ``output_arch`` is not part of ``resolve_options()``; it is injected later during codegen. When it is None the
    tile dispatch returns early without building any LTO, so it must be supplied here or the count is always zero.
    ``block_dim`` matches the value the neighboring tests launch with, so the LTO cache entries are shared.
    """
    module = kernel.module
    options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {"output_arch": module._get_compile_arch(device)}
    return len(ModuleBuilder(module, options).ltoirs)


def test_enable_backward_kernel_option_reaches_lto_dispatch(test, device):
    """Suppress backward LTOs for a kernel-level ``enable_backward=False``, matching ``module_options``."""
    enabled = count_module_ltos(tile_matmul_backward_enabled_kernel, device)
    kernel_option = count_module_ltos(tile_matmul_kernel_option_kernel, device)
    module_option = count_module_ltos(tile_matmul_module_option_kernel, device)

    test.assertEqual(kernel_option, module_option)
    test.assertLess(kernel_option, enabled)


def test_enable_backward_kernel_option_reaches_func_bodies(test, device):
    """Drop backward LTOs from a ``@wp.func`` body when no kernel in the module needs the backward pass."""
    enabled = count_module_ltos(tile_matmul_via_func_enabled_kernel, device)
    disabled = count_module_ltos(tile_matmul_via_func_disabled_kernel, device)

    test.assertLess(disabled, enabled)


def test_enable_backward_union_preserves_mixed_module_backward(test, device):
    """Keep backward LTOs for a shared ``@wp.func`` when only some kernels in the module disable backward."""
    enabled = count_module_ltos(tile_matmul_via_func_enabled_kernel, device)
    disabled = count_module_ltos(tile_matmul_via_func_disabled_kernel, device)
    mixed = count_module_ltos(tile_matmul_mixed_enabled_kernel, device)

    # tile_matmul_mixed_disabled_kernel shares the module, so referencing it here documents that both
    # kernels are built together even though the count is taken from the module, not the kernel.
    test.assertIs(tile_matmul_mixed_disabled_kernel.module, tile_matmul_mixed_enabled_kernel.module)

    test.assertEqual(mixed, enabled)
    test.assertGreater(mixed, disabled)


def test_wp_grad_preserves_backward_ltos(test, device):
    """Keep and execute backward LTOs forced by ``wp.grad()`` in a forward-only kernel."""
    enabled = count_module_ltos(tile_matmul_via_func_enabled_kernel, device)
    forced = count_module_ltos(tile_matmul_via_grad_kernel, device)
    test.assertIs(tile_matmul_grad_forward_kernel.module, tile_matmul_via_grad_kernel.module)
    test.assertEqual(forced, enabled)

    x = 0.25
    x_wp = wp.array([x], dtype=wp.float32, device=device)
    grad_x_wp = wp.zeros_like(x_wp)

    wp.launch_tiled(
        tile_matmul_via_grad_kernel,
        dim=1,
        inputs=[x_wp],
        outputs=[grad_x_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    expected = 2.0 * TILE_M * TILE_M * TILE_M * x
    np.testing.assert_allclose(grad_x_wp.numpy(), [expected], rtol=1.0e-5, atol=1.0e-5)


class TestTileMathDxLTO(unittest.TestCase):
    def test_prepare_forced_adjoint_functions_skips_failed_functions(self):
        """Do not rebuild forced adjoints whose earlier build failed."""

        class TestFunction:
            pass

        func = TestFunction()
        func.adj = SimpleNamespace(
            builder_options={"enable_backward": False},
            called_user_functions=set(),
            force_adjoint_codegen=True,
            skip_build=True,
        )

        def fail_build(*args, **kwargs):
            self.fail("A skipped function was rebuilt")

        func.adj.build = fail_build

        builder = ModuleBuilder.__new__(ModuleBuilder)
        builder.functions = {func: None}
        builder._prepare_forced_adjoint_functions()

    def test_prepare_forced_adjoint_functions_rebuilds_each_function_once(self):
        """Stop selecting a forced adjoint after its first rebuild attempt."""

        class TestFunction:
            pass

        build_count = 0

        def count_build(*args, **kwargs):
            nonlocal build_count
            build_count += 1
            if build_count > 1:
                self.fail("A forced adjoint was rebuilt more than once")

        func = TestFunction()
        func.adj = SimpleNamespace(
            build=count_build,
            builder_options={"enable_backward": False},
            called_user_functions=set(),
            force_adjoint_codegen=True,
            skip_build=False,
        )

        builder = ModuleBuilder.__new__(ModuleBuilder)
        builder.functions = {func: None}
        builder._prepare_forced_adjoint_functions()

        self.assertEqual(build_count, 1)


mathdx_devices = get_cuda_test_devices() if wp._src.context.runtime.core.wp_is_mathdx_enabled() else []

add_function_test(
    TestTileMathDxLTO,
    "test_enable_backward_kernel_option_reaches_lto_dispatch",
    test_enable_backward_kernel_option_reaches_lto_dispatch,
    devices=mathdx_devices,
)
add_function_test(
    TestTileMathDxLTO,
    "test_enable_backward_kernel_option_reaches_func_bodies",
    test_enable_backward_kernel_option_reaches_func_bodies,
    devices=mathdx_devices,
)
add_function_test(
    TestTileMathDxLTO,
    "test_enable_backward_union_preserves_mixed_module_backward",
    test_enable_backward_union_preserves_mixed_module_backward,
    devices=mathdx_devices,
)
add_function_test(
    TestTileMathDxLTO,
    "test_wp_grad_preserves_backward_ltos",
    test_wp_grad_preserves_backward_ltos,
    devices=mathdx_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
