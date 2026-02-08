# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for parallel module compilation via the max_workers option in
wp.force_load() and wp.load_module().
"""

import os
import tempfile
import unittest
import uuid
from importlib import util

import warp as wp


def _generate_module_code(index):
    """Generate source code for a module with a simple kernel.

    Each module has a unique name based on a UUID to guarantee fresh compilation
    (no cache hits).
    """
    uid = uuid.uuid4().hex[:12]
    module_name = f"_test_parallel_load_{index}_{uid}"

    code = f"""\
# -*- coding: utf-8 -*-
import warp as wp

@wp.kernel
def test_kernel_{index}(output: wp.array(dtype=float)):
    tid = wp.tid()
    x = float(tid) + 1.0
    x = wp.sin(x) + wp.cos(x)
    x = wp.sqrt(wp.abs(x) + 1.0)
    output[tid] = x
"""

    return code, module_name


def _load_code_as_module(code, name):
    """Write code to a temp file, import it, and return the warp module.

    Follows the pattern from test_module_hashing.py.
    """
    file, file_path = tempfile.mkstemp(suffix=".py")

    try:
        with os.fdopen(file, "w") as f:
            f.write(code)

        spec = util.spec_from_file_location(name, file_path)
        assert spec is not None and spec.loader is not None
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.remove(file_path)

    return wp.get_module(module.__name__)


def _generate_modules(count):
    """Generate and import multiple fresh modules. Returns list of warp modules."""
    modules = []
    for i in range(count):
        code, name = _generate_module_code(i)
        modules.append(_load_code_as_module(code, name))
    return modules


def _assert_modules_loaded_on_cpu(test, modules):
    for m in modules:
        has_cpu_exec = any(ctx is None for (ctx, _block_dim) in m.execs.keys())
        test.assertTrue(has_cpu_exec, f"Module {m.name} was not loaded on CPU")


class TestModuleParallelLoad(unittest.TestCase):
    def test_force_load_serial(self):
        """Verify that serial compilation (max_workers=0) loads modules correctly."""
        modules = _generate_modules(4)
        wp.force_load(device="cpu", modules=modules, max_workers=0)
        _assert_modules_loaded_on_cpu(self, modules)

    def test_force_load_parallel(self):
        """Verify that parallel compilation (max_workers=2) loads modules correctly."""
        modules = _generate_modules(4)
        wp.force_load(device="cpu", modules=modules, max_workers=2)
        _assert_modules_loaded_on_cpu(self, modules)

    def test_force_load_config_default(self):
        """Verify that wp.config.load_module_max_workers is respected when max_workers is not passed."""
        modules = _generate_modules(2)

        saved = wp.config.load_module_max_workers
        try:
            wp.config.load_module_max_workers = 2
            wp.force_load(device="cpu", modules=modules)
        finally:
            wp.config.load_module_max_workers = saved

        _assert_modules_loaded_on_cpu(self, modules)

    def test_force_load_config_none(self):
        """Verify that config=None auto-detects max_workers from os.cpu_count()."""
        modules = _generate_modules(2)

        saved = wp.config.load_module_max_workers
        try:
            wp.config.load_module_max_workers = None
            wp.force_load(device="cpu", modules=modules)
        finally:
            wp.config.load_module_max_workers = saved

        _assert_modules_loaded_on_cpu(self, modules)

    def test_force_load_single_module(self):
        """Verify that a single module with max_workers>1 falls back to serial without error."""
        modules = _generate_modules(1)
        wp.force_load(device="cpu", modules=modules, max_workers=2)
        _assert_modules_loaded_on_cpu(self, modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
