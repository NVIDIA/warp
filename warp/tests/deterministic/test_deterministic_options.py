# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import warp as wp
import warp.tests.deterministic.test_deterministic_scatter as scatter_module
from warp._src import deterministic as wp_deterministic
from warp.tests.deterministic.common import DeterministicTestBase, all_devices, cpu_device, cuda_devices
from warp.tests.deterministic.test_deterministic_scatter import loop_scatter_add_kernel
from warp.tests.unittest_utils import add_function_test

_THIS_MODULE = sys.modules[__name__]


def _set_test_module_options(options):
    wp.set_module_options(options, module=_THIS_MODULE)


def _get_test_module_options():
    return wp.get_module_options(module=_THIS_MODULE)


@wp.struct
class _DetTileStackEntry:
    value: wp.int32
    weight: wp.float32


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
def struct_tile_stack_kernel(out: wp.array[wp.int32]):
    _i, j = wp.tid()
    stack = wp.tile_stack(capacity=16, dtype=_DetTileStackEntry)
    entry = _DetTileStackEntry()
    entry.value = j
    entry.weight = wp.float32(j) * wp.float32(0.25)
    wp.tile_stack_push(stack, entry, j < 8)
    popped, slot = wp.tile_stack_pop(stack)
    if slot >= 0:
        out[slot] = popped.value


def test_metadata_refresh_struct_tile_stack(test, device):
    """Verify deterministic cache metadata refresh handles struct tile-stack dtypes."""
    del device
    options = struct_tile_stack_kernel.module.resolve_options(wp.config) | {"output_arch": None}
    struct_tile_stack_kernel.adj.build(None, options)
    test.assertIsNotNone(struct_tile_stack_kernel.adj.det_meta)


def test_cpu_cache_hit_refreshes_deterministic_replay_metadata(test, device):
    """Verify cached CPU modules rebuild deterministic launch metadata."""
    del device

    command = [
        sys.executable,
        "-m",
        "warp.tests.deterministic.test_deterministic_backward",
        "TestDeterministicBackward.test_deterministic_backward_counter_store_rejected_cpu",
    ]

    with tempfile.TemporaryDirectory(prefix="wp_det_cpu_cache_") as cache_dir:
        env = os.environ.copy()
        env["WARP_CACHE_PATH"] = cache_dir
        env["PYTHONFAULTHANDLER"] = "1"

        first = subprocess.run(command, check=False, capture_output=True, text=True, env=env, timeout=120)
        test.assertEqual(
            first.returncode,
            0,
            msg=f"Initial CPU deterministic run failed:\nstdout:\n{first.stdout}\nstderr:\n{first.stderr}",
        )

        second = subprocess.run(command, check=False, capture_output=True, text=True, env=env, timeout=120)
        test.assertEqual(
            second.returncode,
            0,
            msg=f"Cached CPU deterministic run failed:\nstdout:\n{second.stdout}\nstderr:\n{second.stderr}",
        )


def test_cuda_mode_switch_uses_variant_metadata(test, device):
    """Verify launches use the loaded variant's metadata across deterministic mode switches."""
    command = [
        sys.executable,
        "-m",
        "warp.tests.deterministic.aux_test_deterministic_cache_mode_switch",
        device.alias,
    ]

    with tempfile.TemporaryDirectory(prefix="wp_det_cuda_cache_") as cache_dir:
        env = os.environ.copy()
        env["WARP_CACHE_PATH"] = cache_dir
        env["PYTHONFAULTHANDLER"] = "1"

        for cache_state in ("initial", "cached"):
            result = subprocess.run(command, check=False, capture_output=True, text=True, env=env, timeout=120)
            test.assertEqual(
                result.returncode,
                0,
                msg=(
                    f"{cache_state.capitalize()} CUDA mode-switch run failed:\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                ),
            )


def test_config_deterministic_max_records_default(test, device):
    """Verify new modules inherit ``wp.config.deterministic_max_records``."""
    del device
    old_max_records = wp.config.deterministic_max_records
    try:
        wp.config.deterministic_max_records = 7

        @wp.kernel(module="unique")
        def _config_max_records_kernel(output: wp.array[wp.float32]):
            tid = wp.tid()
            wp.atomic_add(output, tid % 2, 1.0)

        options = _config_max_records_kernel.module.resolve_options(wp.config)
        test.assertEqual(options["deterministic_max_records"], 7)
    finally:
        wp.config.deterministic_max_records = old_max_records


def test_module_option_override(test, device):
    """Verify per-module deterministic option works."""

    # Create a kernel with a per-module deterministic override.
    @wp.kernel(module_options={"deterministic": wp.DeterministicMode.GPU_TO_GPU}, module="unique")
    def per_kernel_det(
        data: wp.array[wp.float32],
        output: wp.array[wp.float32],
    ):
        tid = wp.tid()
        wp.atomic_add(output, tid % 4, data[tid])

    n = 256
    rng = np.random.default_rng(22)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    # Ensure the shared module is disabled but the unique module option still works.
    old_det = _get_test_module_options()["deterministic"]
    try:
        _set_test_module_options({"deterministic": wp.DeterministicMode.NOT_GUARANTEED})
        output = wp.zeros(4, dtype=wp.float32, device=device)
        wp.launch(per_kernel_det, dim=n, inputs=[data], outputs=[output], device=device)
        result = output.numpy()
        # Basic sanity: sum should be approximately correct.
        for bin_idx in range(4):
            mask = np.arange(n) % 4 == bin_idx
            expected_sum = data_np[mask].sum()
            np.testing.assert_allclose(result[bin_idx], expected_sum, rtol=1e-4)
    finally:
        _set_test_module_options({"deterministic": old_det})


def test_deterministic_mode_validation(test, device):
    """Verify deterministic mode accepts explicit enum values only."""
    del device

    test.assertEqual(wp_deterministic.DETERMINISTIC_NOT_GUARANTEED, wp.DeterministicMode.NOT_GUARANTEED)
    test.assertEqual(wp_deterministic.DETERMINISTIC_RUN_TO_RUN, wp.DeterministicMode.RUN_TO_RUN)
    test.assertEqual(wp_deterministic.DETERMINISTIC_GPU_TO_GPU, wp.DeterministicMode.GPU_TO_GPU)

    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def _enum_mode_kernel(output: wp.array[wp.float32]):
        output[0] = 1.0

    options = _enum_mode_kernel.module.resolve_options(wp.config)
    test.assertEqual(options["deterministic"], wp.DeterministicMode.RUN_TO_RUN)

    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = wp.DeterministicMode.GPU_TO_GPU
        test.assertEqual(wp.config.deterministic, wp.DeterministicMode.GPU_TO_GPU)

        with test.assertRaisesRegex(ValueError, "DeterministicMode"):
            wp.config.deterministic = "run_to_run"

        with test.assertRaisesRegex(ValueError, "DeterministicMode"):
            wp.config.deterministic = True
    finally:
        wp.config.deterministic = old_det

    def make_bad_kernel(value):
        @wp.kernel(module="unique", module_options={"deterministic": value})
        def _bad_mode_kernel(output: wp.array[wp.float32]):
            output[0] = 1.0

        _bad_mode_kernel.module.resolve_options(wp.config)

    for value in (True, False, "run_to_run", "deterministic"):
        with test.subTest(value=value), test.assertRaisesRegex(ValueError, "DeterministicMode"):
            make_bad_kernel(value)


def test_deterministic_enum_parity(test, device):
    """Keep Python deterministic constants aligned with the native enums."""
    del device

    native_source = (Path(wp.__file__).resolve().parent / "native" / "deterministic.cu").read_text()

    def parse_enum(enum_name):
        match = re.search(rf"enum {enum_name} \{{(.*?)\n\}};", native_source, re.DOTALL)
        if match is None:
            raise AssertionError(f"Failed to find enum {enum_name} in deterministic.cu")

        entries = {}
        for enum_line in match.group(1).splitlines():
            entry = enum_line.strip().rstrip(",")
            if not entry or "=" not in entry:
                continue
            name, value = (part.strip() for part in entry.split("=", 1))
            if name.isidentifier() and value.isdecimal():
                entries[name] = int(value)
        return entries

    native_reduce_ops = parse_enum("ReduceOp")
    native_deterministic_levels = parse_enum("DeterminismLevel")
    native_scalar_types = parse_enum("ScalarType")

    test.assertEqual(
        native_reduce_ops,
        {
            "REDUCE_OP_ADD": wp_deterministic.REDUCE_OP_ADD,
            "REDUCE_OP_MIN": wp_deterministic.REDUCE_OP_MIN,
            "REDUCE_OP_MAX": wp_deterministic.REDUCE_OP_MAX,
        },
    )
    test.assertEqual(
        native_deterministic_levels,
        {
            "DETERMINISTIC_NOT_GUARANTEED": int(wp_deterministic.DETERMINISTIC_NOT_GUARANTEED),
            "DETERMINISTIC_RUN_TO_RUN": int(wp_deterministic.DETERMINISTIC_RUN_TO_RUN),
            "DETERMINISTIC_GPU_TO_GPU": int(wp_deterministic.DETERMINISTIC_GPU_TO_GPU),
        },
    )
    test.assertEqual(
        native_scalar_types,
        {
            "SCALAR_HALF": wp_deterministic._SCALAR_TYPE_IDS[wp.float16],
            "SCALAR_FLOAT": wp_deterministic._SCALAR_TYPE_IDS[wp.float32],
            "SCALAR_DOUBLE": wp_deterministic._SCALAR_TYPE_IDS[wp.float64],
            "SCALAR_INT": wp_deterministic._SCALAR_TYPE_IDS[wp.int32],
            "SCALAR_UINT": wp_deterministic._SCALAR_TYPE_IDS[wp.uint32],
            "SCALAR_INT64": wp_deterministic._SCALAR_TYPE_IDS[wp.int64],
            "SCALAR_UINT64": wp_deterministic._SCALAR_TYPE_IDS[wp.uint64],
            "SCALAR_BFLOAT16": wp_deterministic._SCALAR_TYPE_IDS[wp.bfloat16],
        },
    )


def test_run_sort_reduce_uses_emitted_record_count(test, device):
    """Verify the postpass uses emitted records instead of full scatter capacity."""
    del device

    class FakeCore:
        def __init__(self):
            self.workspace_counts = []
            self.device_counts = []

        def wp_deterministic_sort_reduce_workspace_size(
            self, count, _op, _scalar_type, _components, _determinism_level
        ):
            self.workspace_counts.append(count)
            return 1

        def wp_deterministic_sort_reduce_device(
            self,
            _keys,
            _values,
            count,
            _dest_array,
            _dest_size,
            _op,
            _scalar_type,
            _components,
            _determinism_level,
            _workspace,
            _workspace_size,
        ):
            self.device_counts.append(count)

    class FakeRuntime:
        def __init__(self):
            self.core = FakeCore()

    runtime = FakeRuntime()
    device = wp.get_device("cpu")
    keys = wp.empty(1024, dtype=wp.int64, device=device)
    values = wp.empty(1024, dtype=wp.float32, device=device)
    counter = wp.array(np.array([3], dtype=np.int32), dtype=wp.int32, device=device)
    dest = wp.zeros(1, dtype=wp.float32, device=device)
    target = wp_deterministic.ScatterTarget(
        array_var_label="dest",
        helper_name="det_dest",
        family=wp_deterministic.DETERMINISTIC_FAMILY_ADD,
        value_dtype=wp.float32,
        value_ctype="float",
        scalar_dtype=wp.float32,
        reduce_op=wp_deterministic.REDUCE_OP_ADD,
    )

    record_count = wp_deterministic.get_scatter_record_count((keys, values, counter, 1024))
    workspaces = wp_deterministic.run_sort_reduce(
        runtime,
        [target],
        [(keys, values, counter, 1024)],
        [dest],
        device,
        wp_deterministic.DETERMINISTIC_RUN_TO_RUN,
        record_counts=[record_count],
    )

    test.assertEqual(record_count, 3)
    test.assertEqual(runtime.core.workspace_counts, [3])
    test.assertEqual(runtime.core.device_counts, [3])
    test.assertEqual(len(workspaces), 1)


def test_scatter_record_count_capture_uses_capacity(test, device):
    """Verify graph-capture launches avoid host-side scatter count readbacks."""
    del device
    record_count = wp_deterministic.get_scatter_record_count((None, None, None, 1024), stream_is_capturing=True)
    test.assertEqual(record_count, 1024)


def test_scatter_capacity_overflow_rejected(test, device):
    """Verify oversized deterministic scatter buffers fail before allocation."""
    del device

    options = loop_scatter_add_kernel.module.resolve_options(wp.config) | {"output_arch": None}
    loop_scatter_add_kernel.adj.build(None, options)

    meta = loop_scatter_add_kernel.adj.det_meta
    test.assertIsNotNone(meta)
    test.assertTrue(meta.has_scatter)

    with test.assertRaisesRegex(RuntimeError, "int32 limit"):
        wp_deterministic.allocate_scatter_buffers(
            meta.scatter_targets,
            meta,
            wp_deterministic.DETERMINISTIC_SCATTER_MAX_CAPACITY + 1,
            wp.get_device("cpu"),
            max_records=1,
        )


class TestDeterministicOptions(DeterministicTestBase):
    """Test deterministic options and native helper behavior."""

    deterministic_modules = (_THIS_MODULE, scatter_module)


def _add(name, devices=all_devices):
    add_function_test(TestDeterministicOptions, name, globals()[name], devices=devices)


_add("test_metadata_refresh_struct_tile_stack", devices=[cpu_device])
_add("test_cpu_cache_hit_refreshes_deterministic_replay_metadata", devices=[cpu_device])
_add("test_cuda_mode_switch_uses_variant_metadata", devices=cuda_devices)
_add("test_config_deterministic_max_records_default", devices=[cpu_device])
_add("test_module_option_override")
_add("test_deterministic_mode_validation", devices=[cpu_device])
_add("test_deterministic_enum_parity")
_add("test_run_sort_reduce_uses_emitted_record_count", devices=[cpu_device])
_add("test_scatter_record_count_capture_uses_capacity", devices=[cpu_device])
_add("test_scatter_capacity_overflow_rejected", devices=[cpu_device])


if __name__ == "__main__":
    unittest.main(verbosity=2)
