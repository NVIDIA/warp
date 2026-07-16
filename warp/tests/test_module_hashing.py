# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: add more tests for kernels and generics

import os
import subprocess
import sys
import tempfile
import unittest
from importlib import util

import warp as wp
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


def test_module_load(test, device):
    """Ensure that loading a module does not change its hash"""
    m = load_code_as_module(SIMPLE_MODULE, "simple_module")

    hash1 = m.hash_module()
    m.load(device)
    hash2 = m.hash_module()

    test.assertEqual(hash1, hash2)


class TestOptionResolution(unittest.TestCase):
    """Tests for centralized option resolution."""

    def test_none_vs_explicit_optimization_level(self):
        """optimization_level=None must hash differently from any explicit level.

        None is a sentinel meaning "use target-specific default" (O2 for CPU,
        O3 for CUDA), so the hash must distinguish it from explicit values.
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
        """mode=None and mode='release' must hash identically when config.mode is 'release'."""
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
        """Changing config.mode must change the hash when module mode is None."""
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
        """verify_fp=True vs False must produce different hashes."""
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
        """Unique-module import-time hashing must work before explicit ``wp.init()``."""
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
        """``wp.init()`` must clear hashes/options resolved before runtime existed."""
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


devices = get_test_devices()

add_function_test(TestModuleHashing, "test_function_overload_hashing", test_function_overload_hashing)
add_function_test(TestModuleHashing, "test_function_generic_overload_hashing", test_function_generic_overload_hashing)
add_function_test(TestModuleHashing, "test_module_load", test_module_load, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
