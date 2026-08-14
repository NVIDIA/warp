# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MathDx LTO selection in tile kernels that use automatic differentiation.

Add cases here when they inspect or run the forward and backward LTOs produced
for ``enable_backward``, ``wp.grad()``, and custom-gradient call graphs.
"""

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


@wp.kernel(module="unique", enable_backward=False)
def tile_matmul_grad_disabled_kernel(x: wp.array[float], out: wp.array[float]):
    a = wp.tile_ones(shape=(TILE_M, TILE_M), dtype=wp.float32, storage="register") * x[0]
    c = wp.tile_zeros(shape=(TILE_M, TILE_M), dtype=wp.float32)
    wp.tile_matmul(a, a, c)
    out[0] = wp.tile_sum(c)[0]


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


@wp.func
def tile_matmul_bf16_forward_first_helper(
    A: wp.array2d[wp.bfloat16], B: wp.array2d[wp.bfloat16], C: wp.array2d[wp.float32]
):
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(0, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, 0), storage="register")
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel(module="tile_mathdx_disjoint_forward_first", enable_backward=False)
def tile_matmul_bf16_forward_first_kernel(
    A: wp.array2d[wp.bfloat16], B: wp.array2d[wp.bfloat16], C: wp.array2d[wp.float32]
):
    tile_matmul_bf16_forward_first_helper(A, B, C)


@wp.kernel(module="tile_mathdx_disjoint_forward_first")
def tile_matmul_unrelated_backward_second_kernel():
    pass


@wp.kernel(module="tile_mathdx_disjoint_backward_first")
def tile_matmul_unrelated_backward_first_kernel():
    pass


@wp.func
def tile_matmul_bf16_backward_first_helper(
    A: wp.array2d[wp.bfloat16], B: wp.array2d[wp.bfloat16], C: wp.array2d[wp.float32]
):
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(0, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, 0), storage="register")
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel(module="tile_mathdx_disjoint_backward_first", enable_backward=False)
def tile_matmul_bf16_backward_first_kernel(
    A: wp.array2d[wp.bfloat16], B: wp.array2d[wp.bfloat16], C: wp.array2d[wp.float32]
):
    tile_matmul_bf16_backward_first_helper(A, B, C)


CUSTOM_GRAD_PROVIDER_MODULE = wp.Module("tile_mathdx_custom_grad_provider")
CUSTOM_GRAD_BACKWARD_MODULE = wp.Module("tile_mathdx_custom_grad_backward")
CUSTOM_GRAD_WP_GRAD_MODULE = wp.Module("tile_mathdx_custom_grad_wp_grad")
CUSTOM_GRAD_CALLABLE_MODULE = wp.Module("tile_mathdx_custom_grad_callable")
for module in (
    CUSTOM_GRAD_PROVIDER_MODULE,
    CUSTOM_GRAD_BACKWARD_MODULE,
    CUSTOM_GRAD_WP_GRAD_MODULE,
    CUSTOM_GRAD_CALLABLE_MODULE,
):
    module.options["enable_backward"] = False


@wp.func(module=CUSTOM_GRAD_PROVIDER_MODULE)
def tile_matmul_bf16_custom_grad_leaf(x: float):
    a = wp.tile_ones(shape=(TILE_M, TILE_M), dtype=wp.bfloat16, storage="register") * wp.bfloat16(x)
    b = wp.tile_ones(shape=(TILE_M, TILE_M), dtype=wp.bfloat16, storage="register")
    c = wp.tile_zeros(shape=(TILE_M, TILE_M), dtype=wp.float32)
    wp.tile_matmul(a, b, c)
    return wp.tile_sum(c)[0]


@wp.func(module=CUSTOM_GRAD_PROVIDER_MODULE)
def tile_matmul_bf16_custom_grad_target(x: float):
    return tile_matmul_bf16_custom_grad_leaf(x)


@wp.func_grad(tile_matmul_bf16_custom_grad_target)
def adj_tile_matmul_bf16_custom_grad_target(x: float, adj_ret: float):
    wp.adjoint[x] += 512.0 * adj_ret


@wp.kernel(module=CUSTOM_GRAD_BACKWARD_MODULE, enable_backward=True)
def tile_matmul_bf16_custom_grad_backward_kernel(x: wp.array[float], out: wp.array[float]):
    out[0] = tile_matmul_bf16_custom_grad_target(x[0])


@wp.kernel(module=CUSTOM_GRAD_WP_GRAD_MODULE, enable_backward=False)
def tile_matmul_bf16_custom_grad_wp_grad_kernel(x: wp.array[float], out: wp.array[float]):
    out[0] = wp.grad(tile_matmul_bf16_custom_grad_target)(x[0])


@wp.func(module=CUSTOM_GRAD_WP_GRAD_MODULE)
def tile_matmul_bf16_custom_grad_wp_grad_func(x: float):
    return wp.grad(tile_matmul_bf16_custom_grad_target)(x)


@wp.func(module=CUSTOM_GRAD_PROVIDER_MODULE)
def tile_matmul_custom_grad_callable_objective(x: float):
    return x * x


@wp.func(module=CUSTOM_GRAD_PROVIDER_MODULE)
def tile_matmul_bf16_custom_grad_callable_target(x: float) -> float:
    return tile_matmul_bf16_custom_grad_target(x) + tile_matmul_custom_grad_callable_objective(x)


@wp.func_grad(tile_matmul_bf16_custom_grad_callable_target)
def adj_tile_matmul_bf16_custom_grad_callable_target(x: float, adj_ret: float):
    wp.adjoint[x] += (512.0 + wp.grad(tile_matmul_custom_grad_callable_objective)(x)) * adj_ret


@wp.kernel(module=CUSTOM_GRAD_CALLABLE_MODULE, enable_backward=True)
def tile_matmul_bf16_custom_grad_callable_kernel():
    values = wp.tile_ones(shape=(TILE_M,), dtype=float, storage="register")
    wp.tile_map(tile_matmul_bf16_custom_grad_callable_target, values)


@wp.func
def stale_grad_force_leaf(x: float):
    return x * x


@wp.func
def stale_grad_force_target(x: float):
    return stale_grad_force_leaf(x)


@wp.kernel(module="tile_mathdx_grad_force_source", enable_backward=False)
def stale_grad_force_source_kernel(x: wp.array[float], out: wp.array[float]):
    out[0] = wp.grad(stale_grad_force_target)(x[0])


@wp.kernel(module="tile_mathdx_grad_force_sink", enable_backward=False)
def stale_grad_force_sink_kernel(x: wp.array[float], out: wp.array[float]):
    out[0] = stale_grad_force_target(x[0])


WP_GRAD_HELPER_MODULE = wp.Module("tile_mathdx_wp_grad_helper")
WP_GRAD_HELPER_MODULE.options["enable_backward"] = False


@wp.func(module=WP_GRAD_HELPER_MODULE)
def wp_grad_helper_objective(x: float):
    return x * x


@wp.func(module=WP_GRAD_HELPER_MODULE)
def wp_grad_helper_forward_leaf(x: float):
    return x + 1.0


@wp.func(module=WP_GRAD_HELPER_MODULE)
def tile_matmul_bf16_wp_grad_helper(x: float):
    a = wp.tile_ones(shape=(TILE_M, TILE_M), dtype=wp.bfloat16, storage="register") * wp.bfloat16(x)
    b = wp.tile_ones(shape=(TILE_M, TILE_M), dtype=wp.bfloat16, storage="register")
    c = wp.tile_zeros(shape=(TILE_M, TILE_M), dtype=wp.float32)
    wp.tile_matmul(a, b, c)
    return wp.tile_sum(c)[0] + wp_grad_helper_forward_leaf(x) + wp.grad(wp_grad_helper_objective)(x)


@wp.kernel(module=WP_GRAD_HELPER_MODULE, enable_backward=True)
def tile_matmul_bf16_wp_grad_helper_kernel(x: wp.array[float], out: wp.array[float]):
    out[0] = tile_matmul_bf16_wp_grad_helper(x[0])


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

    module = tile_matmul_mixed_enabled_kernel.module
    test.assertIs(tile_matmul_mixed_disabled_kernel.module, module)
    options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {"output_arch": module._get_compile_arch(device)}
    orders = (
        (tile_matmul_mixed_enabled_kernel, tile_matmul_mixed_disabled_kernel),
        (tile_matmul_mixed_disabled_kernel, tile_matmul_mixed_enabled_kernel),
    )

    for kernels in orders:
        with test.subTest(first_kernel=kernels[0].key):
            ordered_hasher = SimpleNamespace(get_unique_kernels=lambda kernels=kernels: kernels)
            mixed = len(ModuleBuilder(module, options, hasher=ordered_hasher).ltoirs)

            test.assertEqual(mixed, enabled)
            test.assertGreater(mixed, disabled)


def test_forward_only_bf16_helper_ignores_unrelated_backward_kernel(test, device):
    """Build a forward-only BF16 helper without backward LTOs when its module also has an unrelated
    backward-enabled kernel, independently of kernel build order."""

    cases = (
        (
            tile_matmul_bf16_forward_first_kernel,
            tile_matmul_unrelated_backward_second_kernel,
            tile_matmul_bf16_forward_first_helper,
        ),
        (
            tile_matmul_unrelated_backward_first_kernel,
            tile_matmul_bf16_backward_first_kernel,
            tile_matmul_bf16_backward_first_helper,
        ),
    )

    for first_kernel, second_kernel, helper in cases:
        with test.subTest(first_kernel=first_kernel.key):
            module = first_kernel.module
            test.assertIs(module, second_kernel.module)
            options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {
                "output_arch": module._get_compile_arch(device)
            }
            ordered_hasher = SimpleNamespace(get_unique_kernels=lambda kernels=(first_kernel, second_kernel): kernels)

            builder = ModuleBuilder(module, options, hasher=ordered_hasher)

            test.assertIn(helper, builder.functions)
            test.assertFalse(helper.adj.builder_options["enable_backward"])


def test_custom_grad_primal_stays_forward_only_in_backward_kernel(test, device):
    """Keep a custom-gradient primal and its callees forward-only when its kernel enables backward."""
    module = tile_matmul_bf16_custom_grad_backward_kernel.module
    options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {"output_arch": module._get_compile_arch(device)}
    builder = ModuleBuilder(module, options)

    test.assertIn(tile_matmul_bf16_custom_grad_target, builder.functions)
    test.assertIn(tile_matmul_bf16_custom_grad_leaf, builder.functions)
    test.assertFalse(tile_matmul_bf16_custom_grad_target.adj.builder_options["enable_backward"])
    test.assertFalse(tile_matmul_bf16_custom_grad_leaf.adj.builder_options["enable_backward"])
    test.assertFalse(tile_matmul_bf16_custom_grad_target.adj.used_by_backward_kernel)
    test.assertFalse(tile_matmul_bf16_custom_grad_leaf.adj.used_by_backward_kernel)

    x = wp.array([1.0], dtype=float, requires_grad=True, device=device)
    out = wp.zeros_like(x)
    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_matmul_bf16_custom_grad_backward_kernel,
            dim=1,
            inputs=[x],
            outputs=[out],
            block_dim=TILE_DIM,
            device=device,
        )

    tape.backward(grads={out: wp.ones_like(out)})
    np.testing.assert_allclose(out.numpy(), [512.0])
    # Every thread in the tiled block calls the scalar function and accumulates its custom gradient into x[0].
    np.testing.assert_allclose(x.grad.numpy(), [512.0 * TILE_DIM])


def test_wp_grad_custom_grad_target_stays_forward_only(test, device):
    """Use a custom gradient without generating backward code for its primal call graph."""
    module = tile_matmul_bf16_custom_grad_wp_grad_kernel.module
    options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {"output_arch": module._get_compile_arch(device)}
    builder = ModuleBuilder(module, options)

    test.assertIn(tile_matmul_bf16_custom_grad_target, builder.functions)
    test.assertIn(tile_matmul_bf16_custom_grad_leaf, builder.functions)
    test.assertIn(tile_matmul_bf16_custom_grad_target.custom_grad_func, builder.functions)
    test.assertFalse(tile_matmul_bf16_custom_grad_target.adj.force_adjoint_codegen)
    test.assertFalse(tile_matmul_bf16_custom_grad_leaf.adj.force_adjoint_codegen)
    test.assertFalse(tile_matmul_bf16_custom_grad_target.adj.builder_options["enable_backward"])
    test.assertFalse(tile_matmul_bf16_custom_grad_leaf.adj.builder_options["enable_backward"])

    x = wp.array([1.0], dtype=float, device=device)
    out = wp.zeros_like(x)
    wp.launch_tiled(
        tile_matmul_bf16_custom_grad_wp_grad_kernel,
        dim=1,
        inputs=[x],
        outputs=[out],
        block_dim=TILE_DIM,
        device=device,
    )
    np.testing.assert_allclose(out.numpy(), [512.0])


def test_custom_grad_callable_stays_forward_only_in_backward_kernel(test, device):
    """Keep a custom-gradient operator forward-only when passed to a differentiable builtin."""
    module = tile_matmul_bf16_custom_grad_callable_kernel.module
    options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {"output_arch": module._get_compile_arch(device)}
    builder = ModuleBuilder(module, options)

    test.assertIn(tile_matmul_bf16_custom_grad_callable_target, builder.functions)
    test.assertIn(tile_matmul_bf16_custom_grad_target, builder.functions)
    test.assertIn(tile_matmul_bf16_custom_grad_leaf, builder.functions)
    test.assertIn(tile_matmul_custom_grad_callable_objective, builder.functions)
    test.assertFalse(tile_matmul_bf16_custom_grad_callable_target.adj.builder_options["enable_backward"])
    test.assertFalse(tile_matmul_bf16_custom_grad_target.adj.builder_options["enable_backward"])
    test.assertFalse(tile_matmul_bf16_custom_grad_leaf.adj.builder_options["enable_backward"])
    test.assertTrue(tile_matmul_custom_grad_callable_objective.adj.builder_options["enable_backward"])
    test.assertTrue(tile_matmul_custom_grad_callable_objective.adj.force_adjoint_codegen)
    test.assertFalse(tile_matmul_bf16_custom_grad_callable_target.adj.used_by_backward_kernel)
    test.assertFalse(tile_matmul_bf16_custom_grad_target.adj.used_by_backward_kernel)
    test.assertFalse(tile_matmul_bf16_custom_grad_leaf.adj.used_by_backward_kernel)


def test_wp_grad_preserves_backward_ltos(test, device):
    """Keep and execute backward LTOs forced by ``wp.grad()`` in a forward-only kernel."""
    disabled = count_module_ltos(tile_matmul_grad_disabled_kernel, device)
    forced = count_module_ltos(tile_matmul_via_grad_kernel, device)
    test.assertIs(tile_matmul_grad_forward_kernel.module, tile_matmul_via_grad_kernel.module)
    test.assertGreater(forced, disabled)

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
    def test_wp_grad_helper_stays_forward_only(self):
        """Discover a ``wp.grad()`` helper before applying backward-only tile constraints."""
        module = tile_matmul_bf16_wp_grad_helper_kernel.module
        options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {"output_arch": None}

        builder = ModuleBuilder(module, options)

        self.assertIn(tile_matmul_bf16_wp_grad_helper, builder.functions)
        self.assertIn(wp_grad_helper_forward_leaf, builder.functions)
        self.assertIn(wp_grad_helper_objective, builder.functions)
        self.assertTrue(tile_matmul_bf16_wp_grad_helper.adj.uses_grad_call)
        self.assertTrue(tile_matmul_bf16_wp_grad_helper.adj.used_by_backward_kernel)
        self.assertFalse(tile_matmul_bf16_wp_grad_helper.adj.builder_options["enable_backward"])
        self.assertFalse(tile_matmul_bf16_wp_grad_helper.adj.force_adjoint_codegen)
        self.assertFalse(wp_grad_helper_forward_leaf.adj.used_by_backward_kernel)
        self.assertFalse(wp_grad_helper_forward_leaf.adj.builder_options["enable_backward"])
        self.assertFalse(wp_grad_helper_forward_leaf.adj.force_adjoint_codegen)
        self.assertFalse(wp_grad_helper_objective.adj.used_by_backward_kernel)
        self.assertTrue(wp_grad_helper_objective.adj.builder_options["enable_backward"])
        self.assertTrue(wp_grad_helper_objective.adj.force_adjoint_codegen)

    def test_wp_grad_custom_grad_builderless_target_stays_forward_only(self):
        """Keep ``wp.grad()`` analysis without a builder from generating backward code for a custom-gradient primal."""
        tile_matmul_bf16_custom_grad_target.adj.force_adjoint_codegen = False
        tile_matmul_bf16_custom_grad_leaf.adj.force_adjoint_codegen = False
        module = tile_matmul_bf16_custom_grad_wp_grad_func.module
        options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {
            "enable_backward": True,
            "output_arch": None,
        }

        tile_matmul_bf16_custom_grad_wp_grad_func.adj.build(None, options)

        self.assertFalse(tile_matmul_bf16_custom_grad_target.adj.force_adjoint_codegen)
        self.assertFalse(tile_matmul_bf16_custom_grad_leaf.adj.force_adjoint_codegen)
        self.assertFalse(tile_matmul_bf16_custom_grad_target.adj.builder_options["enable_backward"])
        self.assertFalse(tile_matmul_bf16_custom_grad_leaf.adj.builder_options["enable_backward"])

    def test_prepare_adjoint_functions_resets_stale_grad_force(self):
        """Derive ``wp.grad()`` force flags from the current builder instead of prior builds."""

        def build_single_kernel(kernel):
            module = kernel.module
            options = module.resolve_options(wp.config, block_dim=TILE_DIM) | {"output_arch": None}
            hasher = SimpleNamespace(get_unique_kernels=lambda: (kernel,))
            return ModuleBuilder(module, options, hasher=hasher)

        orders = (
            (stale_grad_force_sink_kernel, stale_grad_force_source_kernel),
            (stale_grad_force_source_kernel, stale_grad_force_sink_kernel),
        )
        for first_kernel, second_kernel in orders:
            with self.subTest(first_kernel=first_kernel.key):
                for kernel in (first_kernel, second_kernel):
                    builder = build_single_kernel(kernel)
                    expected = kernel is stale_grad_force_source_kernel

                    self.assertIn(stale_grad_force_target, builder.functions)
                    self.assertIn(stale_grad_force_leaf, builder.functions)
                    self.assertEqual(stale_grad_force_target.adj.force_adjoint_codegen, expected)
                    self.assertEqual(stale_grad_force_leaf.adj.force_adjoint_codegen, expected)
                    self.assertEqual(stale_grad_force_target.adj.builder_options["enable_backward"], expected)
                    self.assertEqual(stale_grad_force_leaf.adj.builder_options["enable_backward"], expected)

    def test_prepare_adjoint_functions_skips_failed_functions(self):
        """Do not rebuild forced adjoints whose earlier build failed."""

        class TestFunction:
            pass

        func = TestFunction()
        func.adj = SimpleNamespace(
            builder_options={"enable_backward": False},
            called_grad_functions=set(),
            called_user_functions=set(),
            force_adjoint_codegen=True,
            is_user_function=True,
            skip_build=True,
            uses_grad_call=False,
        )
        func.uses_generated_adjoint = True

        def fail_build(*args, **kwargs):
            self.fail("A skipped function was rebuilt")

        func.adj.build = fail_build

        kernel = TestFunction()
        kernel.adj = TestFunction()
        kernel.adj.called_grad_functions = {func}
        kernel.adj.called_user_functions = set()
        kernel.adj.skip_build = False
        kernel.adj.unvalidated_ref_calls = []
        kernel.adj.used_by_backward_kernel = False
        kernel.options = {"enable_backward": False}

        builder = ModuleBuilder.__new__(ModuleBuilder)
        builder.functions = {func: None}
        builder.kernels = (kernel,)
        builder.default_kernel_options = {"enable_backward": False}
        builder._prepare_adjoint_functions()

    def test_prepare_adjoint_functions_rebuilds_each_function_once(self):
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
        func.adj = TestFunction()
        func.adj.build = count_build
        func.adj.builder_options = {"enable_backward": False}
        func.adj.called_grad_functions = set()
        func.adj.called_user_functions = set()
        func.adj.force_adjoint_codegen = True
        func.adj.is_user_function = True
        func.adj.skip_build = False
        func.adj.uses_grad_call = False
        func.custom_grad_func = None
        func.uses_generated_adjoint = True

        kernel = TestFunction()
        kernel.adj = TestFunction()
        kernel.adj.called_grad_functions = {func}
        kernel.adj.called_user_functions = set()
        kernel.adj.skip_build = False
        kernel.adj.unvalidated_ref_calls = []
        kernel.adj.used_by_backward_kernel = False
        kernel.options = {"enable_backward": False}

        builder = ModuleBuilder.__new__(ModuleBuilder)
        builder.deferred_function_index = 0
        builder.deferred_functions = []
        builder.functions = {func: None}
        builder.kernels = (kernel,)
        builder.default_kernel_options = {"enable_backward": False}
        builder.options = {"enable_backward": False}
        builder._prepare_adjoint_functions()

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
    "test_forward_only_bf16_helper_ignores_unrelated_backward_kernel",
    test_forward_only_bf16_helper_ignores_unrelated_backward_kernel,
    devices=mathdx_devices,
)
add_function_test(
    TestTileMathDxLTO,
    "test_custom_grad_primal_stays_forward_only_in_backward_kernel",
    test_custom_grad_primal_stays_forward_only_in_backward_kernel,
    devices=mathdx_devices,
)
add_function_test(
    TestTileMathDxLTO,
    "test_wp_grad_custom_grad_target_stays_forward_only",
    test_wp_grad_custom_grad_target_stays_forward_only,
    devices=mathdx_devices,
)
add_function_test(
    TestTileMathDxLTO,
    "test_custom_grad_callable_stays_forward_only_in_backward_kernel",
    test_custom_grad_callable_stays_forward_only_in_backward_kernel,
    devices=mathdx_devices,
)
add_function_test(
    TestTileMathDxLTO,
    "test_wp_grad_preserves_backward_ltos",
    test_wp_grad_preserves_backward_ltos,
    devices=mathdx_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
