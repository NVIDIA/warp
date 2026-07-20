# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for parallel module compilation via the max_workers option in
wp.force_load() and wp.load_module().
"""

import os
import tempfile
import unittest
import uuid
from importlib import util

import warp as wp

# Chain of shared ``@wp.func`` helpers used by
# ``TestParallelLoadSharedHelper`` below. Multiple helpers calling each
# other transitively maximise the race surface: every kernel build
# walks the entire chain, so concurrent ``adj.build`` calls have many
# adjoint-state mutations to interleave with one another.


@wp.func
def _race_leaf(x: float) -> float:
    a = wp.sin(x) + wp.cos(x)
    a = a * a + 0.5
    a = wp.sqrt(wp.abs(a) + 1.0)
    b = wp.tan(x * 0.5) + wp.exp(-x * x * 0.01)
    return a + b


@wp.func
def _race_mid(x: float, k: int) -> float:
    s = float(0.0)
    for _ in range(4):
        s = s + _race_leaf(x + s)
    if k > 0:
        s = s * float(k)
    else:
        s = -s
    return s


@wp.func
def _race_helper(x: float, idx: int) -> float:
    y = _race_mid(x, idx)
    z = float(idx) + 1.0
    if idx % 2 == 0:
        y = y + _race_leaf(z)
    else:
        y = y - _race_leaf(z)
    return y


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
def test_kernel_{index}(output: wp.array[float]):
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


def _assert_modules_loaded_on_cuda(test, modules, device):
    for m in modules:
        ctx = device.context
        loaded = any(c == ctx for (c, _block_dim) in m.execs.keys())
        test.assertTrue(loaded, f"Module {m.name} was not loaded on {device}")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA codegen path race needs a CUDA device")
class TestParallelLoadSharedHelper(unittest.TestCase):
    """Regression test for the cross-module codegen race on CUDA.

    Each ``ModuleBuilder`` walks the reachable ``@wp.func`` graph and
    calls ``adj.build`` on each helper, which mutates per-Adjoint state
    (``adj.blocks``, ``adj.deferred_static_expressions``, ...). When
    two modules that reference the *same* helper build concurrently
    (e.g., via ``wp.force_load(max_workers > 1)``), without
    ``_codegen_lock`` around the codegen window the threads interleave
    their writes to the shared adjoint and the emitted .cu file has
    corrupt sections -- one helper's body emitted inside another, or
    references to ``var_*`` / ``_idx`` / ``dim`` that were never
    declared. nvrtc then rejects the file with dozens of syntax errors
    and ``Module._compile`` raises.

    Reproducing the race reliably requires:

    * a non-trivial *call graph* of shared ``@wp.func`` helpers
      (``_race_helper`` -> ``_race_mid`` -> ``_race_leaf``) so each
      module's ``ModuleBuilder`` does meaningful shared adjoint work;
    * enough modules submitted to ``force_load`` so the
      ``ThreadPoolExecutor`` actually runs several builds
      concurrently;
    * a retry loop (``ATTEMPTS``) so we accept a probabilistic
      reproduction -- the race is timing-dependent.

    With the codegen lock in place every attempt succeeds. Without
    the lock at least one attempt raises a build error.
    """

    NUM_MODULES = 8
    ATTEMPTS = 4

    @staticmethod
    def _make_kernel(idx: int):
        """Create a kernel in its own ``Module`` via ``module="unique"``.

        With ``module="unique"``, every factory output lands in its own ``Module`` object. If N kernels
        share a helper, N separate modules each try to inline that helper's adjoint at compile time.
        """

        @wp.kernel(module="unique")
        def k(out: wp.array[float]):
            tid = wp.tid()
            x = float(tid) + float(idx)
            out[tid] = _race_helper(x, idx)

        return k

    def _build_kernels(self, attempt: int):
        """Build a fresh set of unique modules for the given attempt.

        Vary the spawn count per attempt so each invocation builds new modules. After the first
        attempt the earlier kernels' modules are already loaded, so subsequent ``force_load`` calls
        would short-circuit on the hash check and never exercise the codegen path again.
        """
        offset = attempt * 1000
        kernels = [self._make_kernel(offset + i) for i in range(self.NUM_MODULES)]
        modules: list = []
        seen: set = set()
        for k in kernels:
            k.module.mark_modified()
            if id(k.module) in seen:
                continue
            seen.add(id(k.module))
            modules.append(k.module)
        return modules

    def test_force_load_parallel_with_shared_func(self):
        """N modules sharing a chain of ``@wp.func`` helpers must load
        successfully under ``max_workers > 1``. Without the codegen
        lock at least one of the ``ATTEMPTS`` parallel CUDA builds
        raises because the shared helpers' adjoints were clobbered
        mid-build."""
        device = wp.get_preferred_device()
        for attempt in range(self.ATTEMPTS):
            modules = self._build_kernels(attempt)
            try:
                wp.force_load(device=device, modules=modules, max_workers=self.NUM_MODULES)
            except Exception as e:
                self.fail(
                    f"attempt {attempt}: parallel build raised {type(e).__name__}: {e}. "
                    "Check the ``_codegen_lock`` window in warp._src.context.Module._compile."
                )
            _assert_modules_loaded_on_cuda(self, modules, device)

    def test_force_load_parallel_with_shared_func_high_concurrency(self):
        """Same race but with more modules than worker threads, so the
        ``ThreadPoolExecutor`` queues tasks and reuses workers between
        builds."""
        device = wp.get_preferred_device()
        max_workers = max(2, self.NUM_MODULES // 2)
        for attempt in range(self.ATTEMPTS):
            modules = self._build_kernels(attempt + 100) + self._build_kernels(attempt + 200)
            self.assertGreater(len(modules), max_workers)
            try:
                wp.force_load(device=device, modules=modules, max_workers=max_workers)
            except Exception as e:
                self.fail(
                    f"attempt {attempt}: parallel build raised {type(e).__name__}: {e} (high-concurrency variant)."
                )
            _assert_modules_loaded_on_cuda(self, modules, device)


if __name__ == "__main__":
    unittest.main(verbosity=2)
