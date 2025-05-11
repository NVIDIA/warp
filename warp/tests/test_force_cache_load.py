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

import shutil
import unittest
from pathlib import Path

import warp as wp
import warp.tests.aux_test_force_cache_load as add_kernel_module
from warp.tests.unittest_utils import *

ADD_KERNEL_START = """import warp as wp


@wp.kernel
def add_kernel(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.int32)):
    pass


def run(a, b, res, device):
    wp.launch(add_kernel, dim=a.shape, inputs=[a, b], outputs=[res], device=device)
"""

ADD_KERNEL_FINAL = """import warp as wp


@wp.kernel
def add_kernel(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.int32)):
    i = wp.tid()
    res[i] = a[i] + b[i]


def run(a, b, res, device):
    wp.launch(add_kernel, dim=a.shape, inputs=[a, b], outputs=[res], device=device)
"""


def reload_module(module):
    # Clearing the .pyc file associated with a module is a necessary workaround
    # for `importlib.reload` to work as expected when run from within Kit.
    cache_file = importlib.util.cache_from_source(module.__file__)
    os.remove(cache_file)
    importlib.reload(module)


test_cache_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "test_cache_dir")))


def test_force_cache_load(test, device):
    """Test that module hashing can be disabled.

    A module is run, modified, and run again. The second run should not trigger
    a recompilation since the hash will not be used to detect changes.
    """

    try:
        shutil.rmtree(test_cache_dir, ignore_errors=True)
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        wp.set_module_options(
            {"cache_dir": test_cache_dir, "force_cache_load": True, "cuda_output": "ptx"}, add_kernel_module
        )

        a = wp.ones(10, dtype=wp.int32, device=device)
        b = wp.ones(10, dtype=wp.int32, device=device)
        res = wp.zeros((10,), dtype=wp.int32, device=device)

        # Write out the module and import it
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_force_cache_load.py")), "w") as f:
            f.writelines(ADD_KERNEL_START)
        reload_module(add_kernel_module)

        # First launch, cold compile, expect res to be unchanged since kernel is empty
        add_kernel_module.run(a, b, res, device)

        assert_np_equal(res.numpy(), np.zeros((10,), dtype=np.int32))

        res.zero_()

        # Write out the modified module and import it
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_force_cache_load.py")), "w") as f:
            f.writelines(ADD_KERNEL_FINAL)
        reload_module(add_kernel_module)

        # Launch again, should be loaded from cache despite the kernel now being res = a + b
        add_kernel_module.run(a, b, res, device)

        assert_np_equal(res.numpy(), np.zeros((10,), dtype=np.int32))
    finally:
        # Clear the cache directory
        shutil.rmtree(test_cache_dir, ignore_errors=True)
        # Revert the module default options and auxiliary file to the original states
        wp.set_module_options(
            {"force_cache_load": wp.config.force_cache_load, "cache_dir": None, "cuda_output": None}, add_kernel_module
        )

        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_force_cache_load.py")), "w") as f:
            f.writelines(ADD_KERNEL_FINAL)


def test_enable_hashing(test, device):
    """Ensure that logic of test_force_cache_load is sound.

    This test sets "force_cache_load" to False, so normal module hashing rules
    should be in effect.
    """

    try:
        shutil.rmtree(test_cache_dir, ignore_errors=True)
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        wp.set_module_options(
            {"cache_dir": test_cache_dir, "force_cache_load": False, "cuda_output": "ptx"}, add_kernel_module
        )

        a = wp.ones(10, dtype=wp.int32, device=device)
        b = wp.ones(10, dtype=wp.int32, device=device)
        res = wp.zeros((10,), dtype=wp.int32, device=device)

        # Write out the module and import it
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_force_cache_load.py")), "w") as f:
            f.writelines(ADD_KERNEL_START)
        reload_module(add_kernel_module)

        # First launch, cold compile, expect no-op result
        add_kernel_module.run(a, b, res, device)

        assert_np_equal(res.numpy(), np.zeros((10,), dtype=np.int32))

        # Modify the kernel, except no recompilation since "disable_hashing" is enabled
        res.zero_()

        # Write out the modified module and import it
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_force_cache_load.py")), "w") as f:
            f.writelines(ADD_KERNEL_FINAL)
        reload_module(add_kernel_module)

        # Launch again, should be recompiled since the module has has changed
        add_kernel_module.run(a, b, res, device)  # res = a + b
        assert_np_equal(res.numpy(), np.full((10,), 2, dtype=np.int32))
    finally:
        # Clear the cache directory
        shutil.rmtree(test_cache_dir, ignore_errors=True)
        # Revert the module default options and auxiliary file to the original states
        wp.set_module_options(
            {"force_cache_load": wp.config.force_cache_load, "cache_dir": None, "cuda_output": None}, add_kernel_module
        )

        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_force_cache_load.py")), "w") as f:
            f.writelines(ADD_KERNEL_FINAL)


class TestForceCacheLoad(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestForceCacheLoad, "test_force_cache_load", test_force_cache_load, devices=devices)
add_function_test(TestForceCacheLoad, "test_enable_hashing", test_enable_hashing, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
