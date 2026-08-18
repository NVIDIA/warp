# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: add more tests for kernels and generics

import ast
import itertools
import os
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from collections.abc import Mapping
from importlib import util

import warp as wp
from warp._src.codegen import _analyze_reference_bindings
from warp._src.context import ModuleBuilder, ModuleHasher
from warp.tests.unittest_utils import *

FUNC_OVERLOAD_1 = """# -*- coding: utf-8 -*-
import warp as wp

@wp.func
def fn():
    wp.print(17)

@wp.func
def fn(value: int):
    wp.print(value)

@wp.kernel
def k():
    print(fn())
    print(fn(99))
"""

# should be same hash as FUNC_OVERLOAD_1
FUNC_OVERLOAD_2 = """# -*- coding: utf-8 -*-
import warp as wp

@wp.func
def fn():
    wp.print(17)

@wp.func
def fn(value: int):
    wp.print(value)

@wp.kernel
def k():
    print(fn())
    print(fn(99))
"""

# should be different hash than FUNC_OVERLOAD_1 (first overload is different)
FUNC_OVERLOAD_3 = """# -*- coding: utf-8 -*-
import warp as wp

@wp.func
def fn():
    wp.print(42)

@wp.func
def fn(value: int):
    wp.print(value)

@wp.kernel
def k():
    print(fn())
    print(fn(99))
"""

# should be different hash than FUNC_OVERLOAD_1 (second overload is different)
FUNC_OVERLOAD_4 = """# -*- coding: utf-8 -*-
import warp as wp

@wp.func
def fn():
    wp.print(17)

@wp.func
def fn(value: int):
    wp.print(value + 1)

@wp.kernel
def k():
    print(fn())
    print(fn(99))
"""

FUNC_GENERIC_1 = """# -*- coding: utf-8 -*-
import warp as wp

from typing import Any

@wp.func
def generic_fn(x: Any):
    return x * x

@wp.func
def generic_fn(x: Any, y: Any):
    return x * y

@wp.kernel
def k():
    print(generic_fn(17))
    print(generic_fn(17, 42))
"""

# should be same hash as FUNC_GENERIC_1
FUNC_GENERIC_2 = """# -*- coding: utf-8 -*-
import warp as wp

from typing import Any

@wp.func
def generic_fn(x: Any):
    return x * x

@wp.func
def generic_fn(x: Any, y: Any):
    return x * y

@wp.kernel
def k():
    print(generic_fn(17))
    print(generic_fn(17, 42))
"""

# should be different hash than FUNC_GENERIC_1 (first overload is different)
FUNC_GENERIC_3 = """# -*- coding: utf-8 -*-
import warp as wp

from typing import Any

@wp.func
def generic_fn(x: Any):
    return x + x

@wp.func
def generic_fn(x: Any, y: Any):
    return x * y

@wp.kernel
def k():
    print(generic_fn(17))
    print(generic_fn(17, 42))
"""

# should be different hash than FUNC_GENERIC_1 (second overload is different)
FUNC_GENERIC_4 = """# -*- coding: utf-8 -*-
import warp as wp

from typing import Any

@wp.func
def generic_fn(x: Any):
    return x * x

@wp.func
def generic_fn(x: Any, y: Any):
    return x + y

@wp.kernel
def k():
    print(generic_fn(17))
    print(generic_fn(17, 42))
"""


def load_code_as_module(code, name):
    file, file_path = tempfile.mkstemp(suffix=".py")

    try:
        with os.fdopen(file, "w") as f:
            f.write(code)

        spec = util.spec_from_file_location(name, file_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.remove(file_path)

    return wp.get_module(module.__name__)


def test_function_overload_hashing(test, device):
    """Verify that identical overloads share a hash and changed overloads do not."""
    m1 = load_code_as_module(FUNC_OVERLOAD_1, "func_overload_1")
    m2 = load_code_as_module(FUNC_OVERLOAD_2, "func_overload_2")
    m3 = load_code_as_module(FUNC_OVERLOAD_3, "func_overload_3")
    m4 = load_code_as_module(FUNC_OVERLOAD_4, "func_overload_4")

    hash1 = m1.hash_module()
    hash2 = m2.hash_module()
    hash3 = m3.hash_module()
    hash4 = m4.hash_module()

    test.assertEqual(hash2, hash1)
    test.assertNotEqual(hash3, hash1)
    test.assertNotEqual(hash4, hash1)


def test_function_generic_overload_hashing(test, device):
    """Verify that generic overload bodies contribute to the module hash."""
    m1 = load_code_as_module(FUNC_GENERIC_1, "func_generic_1")
    m2 = load_code_as_module(FUNC_GENERIC_2, "func_generic_2")
    m3 = load_code_as_module(FUNC_GENERIC_3, "func_generic_3")
    m4 = load_code_as_module(FUNC_GENERIC_4, "func_generic_4")

    hash1 = m1.hash_module()
    hash2 = m2.hash_module()
    hash3 = m3.hash_module()
    hash4 = m4.hash_module()

    test.assertEqual(hash2, hash1)
    test.assertNotEqual(hash3, hash1)
    test.assertNotEqual(hash4, hash1)


SIMPLE_MODULE = """# -*- coding: utf-8 -*-
import warp as wp

@wp.kernel
def k():
    pass
"""


@wp.kernel
def codegen_order_zulu(output: wp.array[wp.float32]):
    output[wp.tid()] = 1.0


@wp.kernel
def codegen_order_alpha(output: wp.array[wp.float32]):
    output[wp.tid()] = 2.0


@wp.kernel
def codegen_order_mike(output: wp.array[wp.float32]):
    output[wp.tid()] = 3.0


def make_same_key_kernel(value):
    @wp.kernel
    def same_key(output: wp.array[wp.float32]):
        output[wp.tid()] = value

    return same_key


same_key_first = make_same_key_kernel(1.0)
same_key_second = make_same_key_kernel(2.0)


@wp.func
def call_order_first(value: float):
    return value + 1.0


@wp.func
def call_order_second(value: float):
    return value + 2.0


@wp.kernel
def call_order_caller(output: wp.array[wp.float32]):
    output[wp.tid()] = call_order_first(1.0) + call_order_second(2.0) + call_order_first(3.0)


@wp.kernel
def artifact_order_kernel(output: wp.array[wp.float32]):
    output[wp.tid()] = 1.0


SHADOWED_ATTRIBUTE_MODULE = wp.Module("module_hashing_shadowed_attribute")
REFERENCE_TARGET_MODULE = wp.Module("module_hashing_reference_target")
REFERENCE_DECOY_MODULE = wp.Module("module_hashing_reference_decoy")
SHADOWING_CONSUMER_MODULE = wp.Module("module_hashing_shadowing_consumer")
CONTROL_FLOW_SCAN_MODULE = wp.Module("module_hashing_control_flow_scan")
CONTROL_FLOW_BUILD_MODULE = wp.Module("module_hashing_control_flow_build")


@wp.struct(None, module=SHADOWED_ATTRIBUTE_MODULE)
class ShadowedAttributeResult:
    value: float


@wp.func(module=SHADOWED_ATTRIBUTE_MODULE)
def make_shadowed_attribute_result(value: float) -> ShadowedAttributeResult:
    result = ShadowedAttributeResult()
    result.value = value
    return result


@wp.kernel(module=SHADOWED_ATTRIBUTE_MODULE)
def shadowed_attribute_first(out: wp.array[float]):
    e = make_shadowed_attribute_result(1.0)
    out[0] = e.value


@wp.kernel(module=SHADOWED_ATTRIBUTE_MODULE)
def shadowed_attribute_second(out: wp.array[float]):
    e = make_shadowed_attribute_result(2.0)
    out[0] = e.value


@wp.func(module=REFERENCE_DECOY_MODULE)
def shadowed_function_alias(value: float):
    return value + 100.0


@wp.func(module=REFERENCE_TARGET_MODULE)
def shadowed_function_target(value: float):
    return value * 2.0


@wp.kernel(module=SHADOWING_CONSUMER_MODULE)
def shadowed_function_alias_kernel(out: wp.array[float]):
    shadowed_function_alias = shadowed_function_target
    out[0] = shadowed_function_alias(3.0)


@wp.func(module=REFERENCE_TARGET_MODULE)
def shadowed_tuple_alias_target_a(value: float):
    return value + 10.0


@wp.func(module=REFERENCE_TARGET_MODULE)
def shadowed_tuple_alias_target_b(value: float):
    return value + 20.0


@wp.kernel(module=SHADOWING_CONSUMER_MODULE)
def shadowed_tuple_alias_kernel(out: wp.array[float]):
    first, second = shadowed_tuple_alias_target_a, shadowed_tuple_alias_target_b
    out[0] = first(1.0) + second(2.0)


@wp.func(module=REFERENCE_TARGET_MODULE)
def call_before_shadow(value: float):
    return value + 1.0


@wp.kernel(module=SHADOWING_CONSUMER_MODULE)
def call_before_shadow_kernel(out: wp.array[float]):
    out[0] = call_before_shadow(2.0)  # noqa: F823
    call_before_shadow = 0.0


@wp.func(module=REFERENCE_TARGET_MODULE)
def rhs_before_lhs(value: float):
    return value * 2.0


@wp.kernel(module=SHADOWING_CONSUMER_MODULE)
def rhs_before_lhs_kernel(out: wp.array[float]):
    rhs_before_lhs = rhs_before_lhs(3.0)  # noqa: F823
    out[0] = rhs_before_lhs


@wp.func(module=REFERENCE_TARGET_MODULE)
def branch_arm_reference(value: float):
    return value + 4.0


@wp.kernel(module=SHADOWING_CONSUMER_MODULE)
def branch_arm_reference_kernel(condition: bool, out: wp.array[float]):
    if condition:
        branch_arm_reference = 1.0
        out[0] = branch_arm_reference
    else:
        out[0] = branch_arm_reference(2.0)


CONSTANT_BRANCH_GLOBAL_FLAG = False


class ConstantBranchConfig:
    ENABLED = False


@wp.func(module=REFERENCE_TARGET_MODULE)
def constant_branch_target(value: float):
    return value + 5.0


@wp.func(module=REFERENCE_DECOY_MODULE)
def constant_branch_dead_reference(value: float):
    return value + 100.0


@wp.kernel(module=CONTROL_FLOW_SCAN_MODULE)
def constant_branch_literal_kernel(out: wp.array[float]):
    if False:
        constant_branch_target = 0.0
        out[0] = constant_branch_dead_reference(1.0)
    out[0] = constant_branch_target(2.0)


@wp.kernel(module=CONTROL_FLOW_SCAN_MODULE)
def constant_branch_local_kernel(out: wp.array[float]):
    condition = False
    if condition:
        constant_branch_target = 0.0
        out[0] = constant_branch_dead_reference(1.0)
    out[0] = constant_branch_target(3.0)


@wp.kernel(module=CONTROL_FLOW_SCAN_MODULE)
def constant_branch_global_kernel(out: wp.array[float]):
    if CONSTANT_BRANCH_GLOBAL_FLAG:
        constant_branch_target = 0.0
        out[0] = constant_branch_dead_reference(1.0)
    out[0] = constant_branch_target(4.0)


@wp.kernel(module=CONTROL_FLOW_SCAN_MODULE)
def constant_branch_true_kernel(out: wp.array[float]):
    condition = True
    if condition:
        out[0] = constant_branch_target(5.0)  # noqa: F823
    else:
        constant_branch_target = 0.0
        out[0] = constant_branch_dead_reference(1.0)


@wp.kernel(module=CONTROL_FLOW_SCAN_MODULE)
def constant_branch_static_kernel(out: wp.array[float]):
    if wp.static(False):
        constant_branch_target = 0.0
        out[0] = constant_branch_dead_reference(1.0)
    out[0] = constant_branch_target(3.0)


@wp.kernel(module=CONTROL_FLOW_SCAN_MODULE)
def constant_branch_static_local_kernel(out: wp.array[float]):
    condition = False
    if wp.static(condition):
        constant_branch_target = 0.0
        out[0] = constant_branch_dead_reference(1.0)
    out[0] = constant_branch_target(6.0)


@wp.kernel(module=CONTROL_FLOW_SCAN_MODULE)
def constant_branch_rebind_kernel(out: wp.array[float]):
    if CONSTANT_BRANCH_GLOBAL_FLAG:
        out[0] = constant_branch_dead_reference(1.0)
    else:
        out[0] = constant_branch_target(7.0)


@wp.kernel(module=CONTROL_FLOW_BUILD_MODULE)
def constant_branch_augassign_kernel(value: int, out: wp.array[float]):
    condition = 0
    condition += value
    if condition:
        out[0] = constant_branch_target(8.0)


@wp.kernel(module=CONTROL_FLOW_BUILD_MODULE)
def constant_branch_for_kernel(count: int, out: wp.array[float]):
    condition = 0
    for condition in range(count):  # noqa: B007
        pass
    if condition:
        out[0] = constant_branch_target(10.0)


@wp.kernel(module=CONTROL_FLOW_BUILD_MODULE)
def constant_branch_zero_range_kernel(out: wp.array[float]):
    for _ in range(0):
        constant_branch_target = 0.0
        out[0] = constant_branch_dead_reference(1.0)
    out[0] = constant_branch_target(11.0)


@wp.kernel(module=CONTROL_FLOW_BUILD_MODULE)
def constant_branch_attribute_kernel(out: wp.array[float]):
    if ConstantBranchConfig.ENABLED:
        constant_branch_target = 0.0
    out[0] = constant_branch_target(12.0)


@wp.kernel(module=CONTROL_FLOW_BUILD_MODULE)
def constant_branch_unrolled_range_kernel(out: wp.array[float]):
    for i in range(1):
        if i:
            constant_branch_target = 0.0
    out[0] = constant_branch_target(13.0)


def test_module_load(test, device):
    """Ensure that loading a module does not change its hash."""
    m = load_code_as_module(SIMPLE_MODULE, "simple_module")

    hash1 = m.hash_module()
    m.load(device)
    hash2 = m.hash_module()

    test.assertEqual(hash1, hash2)


class TestOptionResolution(unittest.TestCase):
    """Tests for centralized option resolution."""

    def test_none_vs_explicit_optimization_level(self):
        """Verify that the optimization-level sentinel differs from explicit levels.

        ``None`` selects the target-specific default, O2 for CPU and O3 for
        CUDA, so the hash must distinguish it from either explicit value.
        """
        m1 = load_code_as_module(SIMPLE_MODULE, "opt_level_none")
        m1.options["optimization_level"] = None

        m2 = load_code_as_module(SIMPLE_MODULE, "opt_level_explicit_2")
        m2.options["optimization_level"] = 2

        m3 = load_code_as_module(SIMPLE_MODULE, "opt_level_explicit_3")
        m3.options["optimization_level"] = 3

        old = wp.config.optimization_level
        try:
            wp.config.optimization_level = None
            hash_none = m1.hash_module()
            hash_o2 = m2.hash_module()
            hash_o3 = m3.hash_module()
            self.assertNotEqual(hash_none, hash_o2)
            self.assertNotEqual(hash_none, hash_o3)
            self.assertNotEqual(hash_o2, hash_o3)
        finally:
            wp.config.optimization_level = old

    def test_none_vs_explicit_mode(self):
        """Verify that resolved and explicit release modes produce the same hash."""
        m1 = load_code_as_module(SIMPLE_MODULE, "mode_none")
        m1.options["mode"] = None

        m2 = load_code_as_module(SIMPLE_MODULE, "mode_explicit")
        m2.options["mode"] = "release"

        old = wp.config.mode
        try:
            wp.config.mode = "release"
            self.assertEqual(m1.hash_module(), m2.hash_module())
        finally:
            wp.config.mode = old

    def test_config_change_propagates_to_hash(self):
        """Verify that changing the global mode changes a module's resolved hash."""
        m = load_code_as_module(SIMPLE_MODULE, "mode_propagation")
        m.options["mode"] = None

        old = wp.config.mode
        try:
            wp.config.mode = "release"
            hash_release = m.hash_module()

            wp.config.mode = "debug"
            hash_debug = m.hash_module()

            self.assertNotEqual(hash_release, hash_debug)
        finally:
            wp.config.mode = old

    def test_verify_fp_affects_hash(self):
        """Verify that ``verify_fp`` contributes to the module hash."""
        m = load_code_as_module(SIMPLE_MODULE, "verify_fp_test")

        old = wp.config.verify_fp
        try:
            wp.config.verify_fp = False
            hash_false = m.hash_module()

            wp.config.verify_fp = True
            hash_true = m.hash_module()

            self.assertNotEqual(hash_false, hash_true)
        finally:
            wp.config.verify_fp = old


class TestModuleHasherKernelOptions(unittest.TestCase):
    """Regression tests: kernel.options must participate in ModuleHasher."""

    def test_kernel_options_hashed(self):
        """Verify kernels that differ only in ``launch_bounds`` produce different module hashes.

        Before the fix, ``kernel.options`` was not fed into ``ContentHash``, so both hashes collided.
        """

        def make(bounds):
            @wp.kernel(launch_bounds=bounds, module="unique")
            def k(a: wp.array[int]):
                i = wp.tid()
                a[i] = i

            return k

        h_a = make(64).module.hash_module()
        h_b = make(128).module.hash_module()
        # Without the fix, these would collide because kernel name and body match.
        self.assertNotEqual(h_a, h_b)

    def test_cluster_dim_hashed(self):
        """Verify distinct ``cluster_dim`` values hash differently while identical values hash the same.

        ``cluster_dim`` is another kernel option fed through ``kernel.options``, so distinct values must not
        collide on a shared compiled module.
        """

        def make(cluster_dim):
            @wp.kernel(cluster_dim=cluster_dim, module="unique")
            def k(a: wp.array[int]):
                a[wp.tid()] = 0

            return k

        self.assertNotEqual(make(2).module.hash_module(), make(4).module.hash_module())
        self.assertEqual(make(2).module.hash_module(), make(2).module.hash_module())


class TestModuleHashing(unittest.TestCase):
    def test_unique_module_import_hash_before_explicit_init(self):
        """Verify unique-module hashing before explicit ``wp.init()``."""
        code = (
            "import warp as wp\n"
            "import warp._src.optim.linear\n"
            "wp.get_module('warp._src.optim.linear').hash_module()\n"
            "print('OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_init_invalidates_pre_runtime_module_options(self):
        """Verify that ``wp.init()`` invalidates pre-runtime hashes and options."""
        code = (
            "import warp as wp\n"
            "m = wp.get_module('__main__')\n"
            "block_dim = m.options['block_dim']\n"
            "m.get_module_hash()\n"
            "print('before', block_dim in m.hashers, block_dim in m.resolved_options)\n"
            "wp.init()\n"
            "print('after', block_dim in m.hashers, block_dim in m.resolved_options)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("before True True", result.stdout)
        self.assertIn("after False False", result.stdout)

    def test_shadowed_attribute_reference_is_local(self):
        """Verify a local attribute is not resolved through a same-named Warp global."""
        constants, _, _ = shadowed_attribute_first.adj.get_references()

        self.assertNotIn("e.value", constants)

    def test_shadowed_local_bindings_preserve_function_references(self):
        """Verify local bindings preserve only the functions visible at each use."""
        cases = (
            (shadowed_function_alias_kernel, {shadowed_function_target}),
            (shadowed_tuple_alias_kernel, {shadowed_tuple_alias_target_a, shadowed_tuple_alias_target_b}),
            (call_before_shadow_kernel, {call_before_shadow}),
            (rhs_before_lhs_kernel, {rhs_before_lhs}),
            (branch_arm_reference_kernel, {branch_arm_reference}),
        )

        for kernel, expected in cases:
            with self.subTest(kernel=kernel.key):
                _, _, functions = kernel.adj.get_references()
                self.assertEqual(set(functions), expected)

    def test_shadowing_dependency_profile(self):
        """Verify shadowing consumers retain real targets but not decoys."""
        references = SHADOWING_CONSUMER_MODULE.references

        self.assertIn(REFERENCE_TARGET_MODULE, references)
        self.assertNotIn(REFERENCE_DECOY_MODULE, references)

    def test_control_flow_references_are_conservative(self):
        """Verify control-flow scans retain all potentially generated functions."""
        cases = (
            (
                (
                    constant_branch_literal_kernel,
                    constant_branch_local_kernel,
                    constant_branch_global_kernel,
                    constant_branch_true_kernel,
                    constant_branch_static_kernel,
                    constant_branch_static_local_kernel,
                    constant_branch_rebind_kernel,
                    constant_branch_zero_range_kernel,
                ),
                {constant_branch_target, constant_branch_dead_reference},
            ),
            (
                (
                    constant_branch_augassign_kernel,
                    constant_branch_for_kernel,
                    constant_branch_attribute_kernel,
                    constant_branch_unrolled_range_kernel,
                ),
                {constant_branch_target},
            ),
        )

        for kernels, expected in cases:
            for kernel in kernels:
                with self.subTest(kernel=kernel.key):
                    _, _, functions = kernel.adj.get_references()
                    self.assertEqual(set(functions), expected)

    def test_control_flow_dependency_profiles(self):
        """Verify control-flow consumer modules retain both dependency classes."""
        for module in (CONTROL_FLOW_SCAN_MODULE, CONTROL_FLOW_BUILD_MODULE):
            with self.subTest(module=module.name):
                self.assertIn(REFERENCE_TARGET_MODULE, module.references)
                self.assertIn(REFERENCE_DECOY_MODULE, module.references)

    def test_constant_branch_regressions_build(self):
        """Verify constant-range and loop-carried conditions compile as scanned."""
        out = wp.zeros(1, dtype=float, device="cpu")

        wp.launch(constant_branch_zero_range_kernel, dim=1, inputs=[out], device="cpu")
        self.assertEqual(out.numpy()[0], 16.0)

        wp.launch(constant_branch_augassign_kernel, dim=1, inputs=[1, out], device="cpu")
        self.assertEqual(out.numpy()[0], 13.0)

        wp.launch(constant_branch_for_kernel, dim=1, inputs=[2, out], device="cpu")
        self.assertEqual(out.numpy()[0], 15.0)

        wp.launch(constant_branch_attribute_kernel, dim=1, inputs=[out], device="cpu")
        self.assertEqual(out.numpy()[0], 17.0)

        wp.launch(constant_branch_unrolled_range_kernel, dim=1, inputs=[out], device="cpu")
        self.assertEqual(out.numpy()[0], 18.0)

    def test_global_condition_rebinding_preserves_references(self):
        """Verify global-condition callees remain references across rebindings."""
        global CONSTANT_BRANCH_GLOBAL_FLAG

        old_value = CONSTANT_BRANCH_GLOBAL_FLAG
        try:
            for value in (False, True):
                with self.subTest(value=value):
                    CONSTANT_BRANCH_GLOBAL_FLAG = value
                    _, _, functions = constant_branch_rebind_kernel.adj.get_references()
                    self.assertEqual(set(functions), {constant_branch_target, constant_branch_dead_reference})
        finally:
            CONSTANT_BRANCH_GLOBAL_FLAG = old_value

    def test_reference_analysis_storage_is_bounded(self):
        """Verify reference analysis does not retain full per-node name sets."""
        source = "def function():\n" + "".join(f"    local_{i} = global_{i}\n" for i in range(2000))
        tree = ast.parse(source)
        already_tracing = tracemalloc.is_tracing()

        if not already_tracing:
            tracemalloc.start()
        try:
            retained_before, _ = tracemalloc.get_traced_memory()
            analysis = _analyze_reference_bindings(tree)
            retained_after, _ = tracemalloc.get_traced_memory()
        finally:
            if not already_tracing:
                tracemalloc.stop()

        self.assertEqual(len(analysis), 6000)
        self.assertLess(retained_after - retained_before, 16 * 1024 * 1024)

    def test_shadowed_reference_hashes_stable_after_build(self):
        """Verify module building does not change hashes for shadowed local roots."""
        cases = (
            (
                SHADOWED_ATTRIBUTE_MODULE,
                (shadowed_attribute_first, shadowed_attribute_second),
                (),
            ),
            (
                SHADOWING_CONSUMER_MODULE,
                (
                    shadowed_function_alias_kernel,
                    shadowed_tuple_alias_kernel,
                    call_before_shadow_kernel,
                    rhs_before_lhs_kernel,
                ),
                (call_before_shadow, rhs_before_lhs),
            ),
        )

        for module, kernels, recursively_hashed_functions in cases:
            with self.subTest(module=module.name):
                options = module.resolve_options(wp.config, block_dim=1)
                before = ModuleHasher(kernels, options)
                module_hash_before = before.get_hash()
                kernel_hashes_before = {kernel: kernel.hash for kernel in kernels}

                for function in recursively_hashed_functions:
                    self.assertIn(function, before.function_hashes)

                ModuleBuilder(module, options, hasher=before)

                after = ModuleHasher(kernels, options)
                for function in recursively_hashed_functions:
                    self.assertIn(function, after.function_hashes)
                self.assertEqual(after.get_hash(), module_hash_before)
                self.assertEqual(
                    {kernel: kernel.hash for kernel in kernels},
                    kernel_hashes_before,
                )

    @unittest.skipUnless(
        wp.is_cpu_available() and wp.is_cuda_available(),
        "Requires both CPU and CUDA devices",
    )
    def test_shadowed_reference_hashes_stable_across_devices(self):
        """Verify a CUDA build cannot invalidate CPU entry-point hashes."""

        def launch(kernel, device, expected):
            out = wp.empty(1, dtype=float, device=device)
            wp.launch(kernel, dim=1, outputs=[out], device=device)
            self.assertEqual(out.numpy()[0], expected)

        launch(shadowed_attribute_first, "cpu", 1.0)
        launch(shadowed_attribute_first, "cuda:0", 1.0)
        launch(shadowed_attribute_second, "cpu", 2.0)

    def test_codegen_is_independent_of_kernel_order(self):
        """Verify that kernel order does not affect hashes, source, or metadata."""
        kernels = (codegen_order_zulu, codegen_order_alpha, codegen_order_mike)
        module = codegen_order_zulu.module
        options = module.resolve_options(wp.config, 256)
        module_hashes = set()
        sources = {"cpu": set(), "cuda": set()}
        metadata = []

        for device, device_sources in sources.items():
            for kernel_order in itertools.permutations(kernels):
                hasher = ModuleHasher(kernel_order, options)
                builder = ModuleBuilder(module, options, hasher=hasher)
                module_hashes.add(hasher.get_hash())
                device_sources.add(builder.codegen(device))
                metadata.append(builder.build_meta())

        self.assertEqual(len(module_hashes), 1)
        self.assertEqual(len(sources["cpu"]), 1)
        self.assertEqual(len(sources["cuda"]), 1)
        for actual_meta in metadata[1:]:
            self.assertEqual(actual_meta, metadata[0])

    def test_codegen_orders_same_key_kernels_by_digest(self):
        """Verify that full digests order same-key kernels deterministically.

        The factory creates kernels with the same key but different captured
        values, which gives them distinct content hashes.
        """
        kernels = (same_key_first, same_key_second)
        module = same_key_first.module
        options = module.resolve_options(wp.config, 256)

        self.assertIs(same_key_second.module, module)
        self.assertEqual(same_key_first.key, same_key_second.key)

        sources = {"cpu": set(), "cuda": set()}
        for device, device_sources in sources.items():
            for kernel_order in itertools.permutations(kernels):
                hasher = ModuleHasher(kernel_order, options)
                builder = ModuleBuilder(module, options, hasher=hasher)
                self.assertNotEqual(same_key_first.hash, same_key_second.hash)
                device_sources.add(builder.codegen(device))

        self.assertEqual(len(sources["cpu"]), 1)
        self.assertEqual(len(sources["cuda"]), 1)

    def test_called_user_functions_preserve_discovery_order(self):
        """Verify that user-function calls retain their discovery order."""
        module = call_order_caller.module
        options = module.resolve_options(wp.config, 256)
        hasher = ModuleHasher((call_order_caller,), options)
        ModuleBuilder(module, options, hasher=hasher)

        calls = call_order_caller.adj.called_user_functions
        self.assertIsInstance(calls, Mapping)
        self.assertEqual(tuple(calls), (call_order_first, call_order_second))

    def test_module_builder_orders_independent_artifacts(self):
        """Verify deterministic ordering of declarations, link inputs, and metadata."""
        module = artifact_order_kernel.module
        options = module.resolve_options(wp.config, 256)
        hasher = ModuleHasher((artifact_order_kernel,), options)
        builder = ModuleBuilder(module, options, hasher=hasher)

        builder.ltoirs_decl["zulu"] = "void issue_1738_zulu();"
        builder.ltoirs_decl["alpha"] = "void issue_1738_alpha();"
        source = builder.codegen("cuda")
        self.assertLess(source.index("issue_1738_alpha"), source.index("issue_1738_zulu"))

        builder.ltoirs["zulu"] = b"zulu-lto"
        builder.ltoirs["alpha"] = b"alpha-lto"
        builder.fatbins["zulu"] = b"zulu-fatbin"
        builder.fatbins["alpha"] = b"alpha-fatbin"
        self.assertEqual(
            builder.get_link_inputs(),
            ([b"alpha-lto", b"zulu-lto"], [b"alpha-fatbin", b"zulu-fatbin"]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            meta_path = os.path.join(temp_dir, "module.meta")
            module._write_meta(meta_path, {"zulu": 0, "alpha": 1})
            with open(meta_path) as meta_file:
                serialized_meta = meta_file.read()

        self.assertEqual(serialized_meta, '{"alpha": 1, "zulu": 0}')


devices = get_test_devices()

add_function_test(TestModuleHashing, "test_function_overload_hashing", test_function_overload_hashing)
add_function_test(TestModuleHashing, "test_function_generic_overload_hashing", test_function_generic_overload_hashing)
add_function_test(TestModuleHashing, "test_module_load", test_module_load, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
