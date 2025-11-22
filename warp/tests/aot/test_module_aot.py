# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import importlib
import importlib.util
import io
import os
import shutil
import unittest
from pathlib import Path

import numpy as np

import warp as wp
import warp.tests.aot.aux_test_generic_multiple_overloads
import warp.tests.aot.aux_test_generic_no_overloads
import warp.tests.aot.aux_test_generic_one_overload
import warp.tests.aot.aux_test_hash_reload
import warp.tests.aot.aux_test_mixed_generic_kernels
import warp.tests.aot.aux_test_mixed_regular_and_generic
from warp.tests.unittest_utils import *

ADD_KERNEL_START = """import warp as wp


@wp.kernel
def add_kernel(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.int32)):
    pass
"""

ADD_KERNEL_FINAL = """import warp as wp


@wp.kernel
def add_kernel(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.int32)):
    i = wp.tid()
    res[i] = a[i] + b[i]
"""


def reload_module(module):
    # Clearing the .pyc file associated with a module is a necessary workaround
    # for `importlib.reload` to work as expected when run from within Kit.
    cache_file = importlib.util.cache_from_source(module.__file__)
    if os.path.exists(cache_file):
        os.remove(cache_file)
    importlib.reload(module)


TEST_CACHE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "test_module_aot_cache")))


def test_disable_hashing(test, device):
    """Test that module hashing can be disabled.

    A module is run, modified, and run again. The second run should not trigger
    a recompilation since the hash will not be used to detect changes.
    """

    try:
        shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)
        TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        wp.set_module_options(
            {"block_dim": 1 if device.is_cpu else 256},
            warp.tests.aot.aux_test_hash_reload,
        )

        a = wp.ones(10, dtype=wp.int32, device=device)
        b = wp.ones(10, dtype=wp.int32, device=device)
        res = wp.zeros((10,), dtype=wp.int32, device=device)

        # Write out the module and import it
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_hash_reload.py")), "w") as f:
            f.writelines(ADD_KERNEL_START)
        reload_module(warp.tests.aot.aux_test_hash_reload)

        # First launch, cold compile, expect res to be unchanged since kernel is empty
        wp.compile_aot_module(warp.tests.aot.aux_test_hash_reload, device, module_dir=TEST_CACHE_DIR, strip_hash=True)
        wp.load_aot_module(warp.tests.aot.aux_test_hash_reload, device, module_dir=TEST_CACHE_DIR, strip_hash=True)

        wp.launch(
            warp.tests.aot.aux_test_hash_reload.add_kernel,
            dim=a.shape,
            inputs=[a, b],
            outputs=[res],
            device=device,
        )

        assert_np_equal(res.numpy(), np.zeros((10,), dtype=np.int32))

        res.zero_()

        # Write out the modified module and import it
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_hash_reload.py")), "w") as f:
            f.writelines(ADD_KERNEL_FINAL)
        reload_module(warp.tests.aot.aux_test_hash_reload)

        # This time, the hash checks will be skipped so the previously compiled module will be loaded
        wp.load_aot_module(warp.tests.aot.aux_test_hash_reload, device, module_dir=TEST_CACHE_DIR, strip_hash=True)

        # Kernel is executed with the ADD_KERNEL_START code, not the ADD_KERNEL_FINAL code
        wp.launch(
            warp.tests.aot.aux_test_hash_reload.add_kernel,
            dim=a.shape,
            inputs=[a, b],
            outputs=[res],
            device=device,
        )

        assert_np_equal(res.numpy(), np.zeros((10,), dtype=np.int32))
    finally:
        # Clear the cache directory
        shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)
        # Revert the module default options and auxiliary file to the original states
        wp.set_module_options({"cuda_output": None, "strip_hash": False}, warp.tests.aot.aux_test_hash_reload)

        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_hash_reload.py")), "w") as f:
            f.writelines(ADD_KERNEL_FINAL)


def test_enable_hashing(test, device):
    """Ensure that the logic of test_disable_hashing is sound.

    This test sets "strip_hash" to False, so normal module hashing rules
    should be in effect.
    """

    try:
        shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)
        TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        wp.set_module_options(
            {"block_dim": 1 if device.is_cpu else 256},
            warp.tests.aot.aux_test_hash_reload,
        )

        a = wp.ones(10, dtype=wp.int32, device=device)
        b = wp.ones(10, dtype=wp.int32, device=device)
        res = wp.zeros((10,), dtype=wp.int32, device=device)

        # Write out the module and import it
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_hash_reload.py")), "w") as f:
            f.writelines(ADD_KERNEL_START)
        reload_module(warp.tests.aot.aux_test_hash_reload)

        # First launch, cold compile, expect no-op result
        wp.compile_aot_module(warp.tests.aot.aux_test_hash_reload, device, module_dir=TEST_CACHE_DIR, strip_hash=False)
        wp.load_aot_module(warp.tests.aot.aux_test_hash_reload, device, module_dir=TEST_CACHE_DIR, strip_hash=False)
        wp.launch(
            warp.tests.aot.aux_test_hash_reload.add_kernel,
            dim=a.shape,
            inputs=[a, b],
            outputs=[res],
            device=device,
        )

        assert_np_equal(res.numpy(), np.zeros((10,), dtype=np.int32))

        # Write out the modified module (results in a different hash) and import it
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_hash_reload.py")), "w") as f:
            f.writelines(ADD_KERNEL_FINAL)
        reload_module(warp.tests.aot.aux_test_hash_reload)

        # Trying to load the module should fail since a compiled module with the expected hash does not exist
        with test.assertRaises(FileNotFoundError):
            wp.load_aot_module(
                "warp.tests.aot.aux_test_hash_reload", device, module_dir=TEST_CACHE_DIR, strip_hash=False
            )
    finally:
        # Clear the cache directory
        shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)
        # Revert the module default options and auxiliary file to the original states
        wp.set_module_options({"cuda_output": None, "strip_hash": False}, warp.tests.aot.aux_test_hash_reload)

        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_hash_reload.py")), "w") as f:
            f.writelines(ADD_KERNEL_FINAL)


def test_module_load_resolution(test, device):
    """Test various ways to resolving a module when loading and compiling."""

    wp.set_module_options(
        {"block_dim": 1 if device.is_cpu else 256},
        warp.tests.aot.aux_test_hash_reload,
    )

    a = wp.ones(10, dtype=wp.int32, device=device)
    b = wp.ones(10, dtype=wp.int32, device=device)
    res = wp.zeros((10,), dtype=wp.int32, device=device)

    reload_module(warp.tests.aot.aux_test_hash_reload)
    wp.compile_aot_module(warp.tests.aot.aux_test_hash_reload, device)
    wp.load_aot_module(warp.tests.aot.aux_test_hash_reload, device)

    wp.launch(
        warp.tests.aot.aux_test_hash_reload.add_kernel,
        dim=a.shape,
        inputs=[a, b],
        outputs=[res],
        device=device,
    )
    assert_np_equal(res.numpy(), np.full((10,), 2, dtype=np.int32))

    reload_module(warp.tests.aot.aux_test_hash_reload)
    res.zero_()
    wp.compile_aot_module("warp.tests.aot.aux_test_hash_reload", device)
    wp.load_aot_module("warp.tests.aot.aux_test_hash_reload", device)

    wp.launch(
        warp.tests.aot.aux_test_hash_reload.add_kernel,
        dim=a.shape,
        inputs=[a, b],
        outputs=[res],
        device=device,
    )
    assert_np_equal(res.numpy(), np.full((10,), 2, dtype=np.int32))


class TestModuleAOT(unittest.TestCase):
    def test_module_compile_specified_arch_ptx(self):
        """Test that a module can be compiled for a specific architecture or architectures (PTX)."""

        if wp.get_cuda_device_count() == 0:
            self.skipTest("No CUDA devices found")

        if len(wp._src.context.runtime.nvrtc_supported_archs) < 2:
            self.skipTest("NVRTC must support at least two architectures to run this test")

        try:
            shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)
            TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            archs = list(wp._src.context.runtime.nvrtc_supported_archs)[:2]

            wp.compile_aot_module(
                warp.tests.aot.aux_test_hash_reload, arch=archs, module_dir=TEST_CACHE_DIR, use_ptx=True
            )

            # Make sure the expected files exist
            module_identifier = wp.get_module("warp.tests.aot.aux_test_hash_reload").get_module_identifier()
            for arch in archs:
                expected_filename = f"{module_identifier}.sm{arch}.ptx"
                expected_path = TEST_CACHE_DIR / expected_filename
                self.assertTrue(expected_path.exists(), f"Expected compiled PTX file not found: {expected_path}")

        finally:
            # Clear the cache directory
            shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)

    def test_module_compile_specified_arch_cubin(self):
        """Test that a module can be compiled for a specific architecture or architectures (CUBIN)."""

        if wp.get_cuda_device_count() == 0:
            self.skipTest("No CUDA devices found")

        if len(wp._src.context.runtime.nvrtc_supported_archs) < 2:
            self.skipTest("NVRTC must support at least two architectures to run this test")

        try:
            shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)
            TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            archs = list(wp._src.context.runtime.nvrtc_supported_archs)[:2]

            wp.compile_aot_module(
                warp.tests.aot.aux_test_hash_reload, arch=archs, module_dir=TEST_CACHE_DIR, use_ptx=False
            )

            # Make sure the expected files exist
            module_identifier = wp.get_module("warp.tests.aot.aux_test_hash_reload").get_module_identifier()
            for arch in archs:
                expected_filename = f"{module_identifier}.sm{arch}.cubin"
                expected_path = TEST_CACHE_DIR / expected_filename
                self.assertTrue(expected_path.exists(), f"Expected compiled CUBIN file not found: {expected_path}")

        finally:
            # Clear the cache directory
            shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)


def test_generic_kernel_no_overloads(test, device):
    """Test that AOT compilation fails with a clear error when a generic kernel has no overloads."""
    # Attempting to compile should raise a RuntimeError
    with test.assertRaises(RuntimeError) as cm:
        wp.compile_aot_module(warp.tests.aot.aux_test_generic_no_overloads, device)

    # Verify the error message is helpful
    error_msg = str(cm.exception)
    test.assertIn("No compilable kernels found", error_msg)
    test.assertIn("generic kernel with overloads", error_msg)


def test_generic_kernel_one_overload_with_strip_hash(test, device):
    """Test that AOT compilation succeeds with strip_hash=True when generic kernels have exactly one overload."""
    # Set module options to avoid block_dim being sticky across devices
    wp.set_module_options(
        {"block_dim": 1 if device.is_cpu else 256},
        warp.tests.aot.aux_test_generic_one_overload,
    )

    # This should succeed even with strip_hash=True
    wp.compile_aot_module(warp.tests.aot.aux_test_generic_one_overload, device, strip_hash=True)
    wp.load_aot_module(warp.tests.aot.aux_test_generic_one_overload, device, strip_hash=True)

    # Verify the kernel can be launched
    a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    wp.launch(
        warp.tests.aot.aux_test_generic_one_overload.scale_f32,
        dim=a.shape,
        inputs=[a, 2.0],
        device=device,
    )

    assert_np_equal(a.numpy(), np.array([2.0, 4.0, 6.0], dtype=np.float32))


def test_generic_kernel_multiple_overloads_with_strip_hash(test, device):
    """Test that AOT compilation fails with strip_hash=True when generic kernels have multiple overloads."""
    # Attempting to compile with strip_hash=True should raise a RuntimeError
    with test.assertRaises(RuntimeError) as cm:
        wp.compile_aot_module(warp.tests.aot.aux_test_generic_multiple_overloads, device, strip_hash=True)

    # Verify the error message is helpful
    error_msg = str(cm.exception)
    test.assertIn("strip_hash=True", error_msg)
    test.assertIn("multiple overloads", error_msg)
    test.assertIn("scale", error_msg)
    test.assertIn("hash suffixes", error_msg)


def test_generic_kernel_multiple_overloads_without_strip_hash(test, device):
    """Test that AOT compilation succeeds without strip_hash when generic kernels have multiple overloads."""
    # Set module options to avoid block_dim being sticky across devices
    wp.set_module_options(
        {"block_dim": 1 if device.is_cpu else 256},
        warp.tests.aot.aux_test_generic_multiple_overloads,
    )

    # This should succeed with strip_hash=False (the default)
    wp.compile_aot_module(warp.tests.aot.aux_test_generic_multiple_overloads, device, strip_hash=False)
    wp.load_aot_module(warp.tests.aot.aux_test_generic_multiple_overloads, device, strip_hash=False)

    # Verify both overloads can be launched
    a_f32 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    wp.launch(
        warp.tests.aot.aux_test_generic_multiple_overloads.scale_f32,
        dim=a_f32.shape,
        inputs=[a_f32, 2.0],
        device=device,
    )
    assert_np_equal(a_f32.numpy(), np.array([2.0, 4.0, 6.0], dtype=np.float32))

    a_f64 = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device=device)
    wp.launch(
        warp.tests.aot.aux_test_generic_multiple_overloads.scale_f64,
        dim=a_f64.shape,
        inputs=[a_f64, 3.0],
        device=device,
    )
    assert_np_equal(a_f64.numpy(), np.array([3.0, 6.0, 9.0], dtype=np.float64))


def test_generic_kernel_no_overloads_with_regular_kernel(test, device):
    """Test that a module with a regular kernel + generic kernel without overloads compiles with a warning."""
    # Set module options to avoid block_dim being sticky across devices
    wp.set_module_options(
        {"block_dim": 1 if device.is_cpu else 256},
        warp.tests.aot.aux_test_mixed_regular_and_generic,
    )

    # This should succeed with a warning about the generic kernel without overloads
    with contextlib.redirect_stdout(io.StringIO()) as f:
        wp.compile_aot_module(warp.tests.aot.aux_test_mixed_regular_and_generic, device)

    # Verify a warning was issued
    output = f.getvalue()
    test.assertIn("without overloads", output)
    test.assertIn("generic_scale", output)
    test.assertIn("Warp UserWarning", output)

    # Load and verify the regular kernel works
    wp.load_aot_module(warp.tests.aot.aux_test_mixed_regular_and_generic, device)

    a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    b = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=device)
    c = wp.zeros(3, dtype=wp.float32, device=device)

    wp.launch(
        warp.tests.aot.aux_test_mixed_regular_and_generic.regular_add,
        dim=a.shape,
        inputs=[a, b, c],
        device=device,
    )
    assert_np_equal(c.numpy(), np.array([5.0, 7.0, 9.0], dtype=np.float32))


def test_mixed_generic_kernels_some_without_overloads(test, device):
    """Test that a module with mixed generic kernels (some with, some without overloads) compiles with a warning."""
    # Set module options to avoid block_dim being sticky across devices
    wp.set_module_options(
        {"block_dim": 1 if device.is_cpu else 256},
        warp.tests.aot.aux_test_mixed_generic_kernels,
    )

    # This should succeed with a warning about the generic kernel without overloads
    with contextlib.redirect_stdout(io.StringIO()) as f:
        wp.compile_aot_module(warp.tests.aot.aux_test_mixed_generic_kernels, device, strip_hash=False)

    # Verify a warning was issued
    output = f.getvalue()
    test.assertIn("without overloads", output)
    test.assertIn("multiply_without_overloads", output)
    test.assertIn("Warp UserWarning", output)

    # Load and verify the kernel with overloads works
    wp.load_aot_module(warp.tests.aot.aux_test_mixed_generic_kernels, device, strip_hash=False)

    a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    wp.launch(
        warp.tests.aot.aux_test_mixed_generic_kernels.scale_f32,
        dim=a.shape,
        inputs=[a, 2.0],
        device=device,
    )
    assert_np_equal(a.numpy(), np.array([2.0, 4.0, 6.0], dtype=np.float32))


devices = get_test_devices()
add_function_test(TestModuleAOT, "test_disable_hashing", test_disable_hashing, devices=devices)
add_function_test(TestModuleAOT, "test_enable_hashing", test_enable_hashing, devices=devices)
add_function_test(TestModuleAOT, "test_module_load_resolution", test_module_load_resolution, devices=devices)
add_function_test(TestModuleAOT, "test_generic_kernel_no_overloads", test_generic_kernel_no_overloads, devices=devices)
add_function_test(
    TestModuleAOT,
    "test_generic_kernel_one_overload_with_strip_hash",
    test_generic_kernel_one_overload_with_strip_hash,
    devices=devices,
)
add_function_test(
    TestModuleAOT,
    "test_generic_kernel_multiple_overloads_with_strip_hash",
    test_generic_kernel_multiple_overloads_with_strip_hash,
    devices=devices,
)
add_function_test(
    TestModuleAOT,
    "test_generic_kernel_multiple_overloads_without_strip_hash",
    test_generic_kernel_multiple_overloads_without_strip_hash,
    devices=devices,
)
add_function_test(
    TestModuleAOT,
    "test_generic_kernel_no_overloads_with_regular_kernel",
    test_generic_kernel_no_overloads_with_regular_kernel,
    devices=devices,
)
add_function_test(
    TestModuleAOT,
    "test_mixed_generic_kernels_some_without_overloads",
    test_mixed_generic_kernels_some_without_overloads,
    devices=devices,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
