# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public CUDA kernel property query.

The tests cover the complete property dictionary, argument validation, module
variant selection, lazy compilation, errors, side effects, and module lifetime.
"""

import unittest
import weakref
from typing import Any
from unittest import mock

import warp as wp
import warp._src.context as warp_context
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

ATTRIBUTE_BLOCK_DIM = 64


@wp.kernel(enable_backward=False)
def attribute_kernel(out: wp.array[float]):
    tid = wp.tid()
    out[tid] = float(tid)


@wp.kernel(enable_backward=False, module="unique")
def lazy_attribute_kernel(out: wp.array[float]):
    tid = wp.tid()
    out[tid] = float(tid) + 1.0


@wp.kernel(cluster_dim=2, enable_backward=False, module="unique")
def inspection_attribute_kernel(out: wp.array[float]):
    tid = wp.tid()
    out[tid] = float(tid) + 2.0


@wp.kernel(enable_backward=False)
def generic_attribute_kernel(out: wp.array[Any]):
    tid = wp.tid()
    out[tid] = out[tid]


generic_float_attribute_kernel = wp.overload(generic_attribute_kernel, [wp.array[float]])


def constructor_kernel_func(out: wp.array[float]):
    tid = wp.tid()
    out[tid] = float(tid)


def test_cuda_kernel_public_properties(test, device):
    """Verify that public queries return complete, stable property dictionaries."""
    properties = wp.get_cuda_kernel_properties(attribute_kernel, device=device, block_dim=ATTRIBUTE_BLOCK_DIM)
    test.assertEqual(set(properties), {"register_count", "local_memory_size"})
    test.assertIsInstance(properties["register_count"], int)
    test.assertIsInstance(properties["local_memory_size"], int)
    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)
    test.assertEqual(
        properties,
        wp.get_cuda_kernel_properties(
            attribute_kernel,
            device=device,
            block_dim=ATTRIBUTE_BLOCK_DIM,
        ),
    )


def test_cuda_kernel_properties_use_one_native_batch(test, device):
    """Verify that resource queries fetch all properties in one native call."""
    expected_forward = object()
    module_exec = mock.Mock()
    module_exec._get_forward_cuda_kernel.return_value = expected_forward

    def query(context, forward, values, count):
        test.assertEqual(context, device.context)
        test.assertIs(forward, expected_forward)
        test.assertEqual(count, 2)
        values[0] = 17
        values[1] = 23
        return True

    with (
        mock.patch.object(attribute_kernel.module, "load", return_value=module_exec),
        mock.patch.object(
            warp_context.runtime.core,
            "wp_cuda_get_kernel_properties",
            side_effect=query,
        ) as native_query,
    ):
        properties = wp.get_cuda_kernel_properties(attribute_kernel, device=device, block_dim=ATTRIBUTE_BLOCK_DIM)

    test.assertEqual(properties, {"register_count": 17, "local_memory_size": 23})
    native_query.assert_called_once()
    module_exec._get_forward_cuda_kernel.assert_called_once_with(attribute_kernel)


def test_cuda_kernel_properties_loads_requested_variant(test, device):
    """Verify that an explicit block dimension selects the matching module variant."""
    attribute_kernel.module.unload()

    properties = wp.get_cuda_kernel_properties(
        attribute_kernel,
        device=device,
        block_dim=ATTRIBUTE_BLOCK_DIM,
    )

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)
    module_exec = attribute_kernel.module.execs[(device.context, ATTRIBUTE_BLOCK_DIM)]
    test.assertEqual(module_exec.block_dim, ATTRIBUTE_BLOCK_DIM)


def test_cuda_kernel_properties_accepts_index_block_dim(test, device):
    """Verify that resource queries resolve indexable block dimensions."""

    class IndexableBlockDim:
        def __index__(self):
            return ATTRIBUTE_BLOCK_DIM

    attribute_kernel.module.unload()
    block_dim = IndexableBlockDim()

    properties = wp.get_cuda_kernel_properties(
        attribute_kernel,
        device=device,
        block_dim=block_dim,
    )

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)
    module_exec = attribute_kernel.module.execs[(device.context, ATTRIBUTE_BLOCK_DIM)]
    test.assertEqual(module_exec.block_dim, ATTRIBUTE_BLOCK_DIM)


def test_cuda_kernel_properties_uses_module_default(test, device):
    """Verify that an omitted block dimension selects the module default variant."""
    attribute_kernel.module.unload()
    default_block_dim = attribute_kernel.module.options["block_dim"]

    properties = wp.get_cuda_kernel_properties(attribute_kernel, device=device)

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)
    module_exec = attribute_kernel.module.execs[(device.context, default_block_dim)]
    test.assertEqual(module_exec.block_dim, default_block_dim)


def test_cuda_kernel_properties_compiles_lazily(test, device):
    """Verify that a resource query compiles a kernel before its first launch."""
    lazy_attribute_kernel.module.unload()
    exec_key = (device.context, ATTRIBUTE_BLOCK_DIM)
    test.assertNotIn(exec_key, lazy_attribute_kernel.module.execs)

    properties = wp.get_cuda_kernel_properties(
        lazy_attribute_kernel,
        device=device,
        block_dim=ATTRIBUTE_BLOCK_DIM,
    )

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)
    test.assertIn(exec_key, lazy_attribute_kernel.module.execs)


def test_cuda_kernel_properties_avoids_shared_memory_configuration(test, device):
    """Verify that a fresh resource query avoids shared-memory configuration."""
    attribute_kernel.module.unload()

    with mock.patch.object(
        warp_context.runtime.core,
        "wp_cuda_configure_kernel_shared_memory",
        wraps=warp_context.runtime.core.wp_cuda_configure_kernel_shared_memory,
    ) as configure_shared_memory:
        properties = wp.get_cuda_kernel_properties(
            attribute_kernel,
            device=device,
            block_dim=ATTRIBUTE_BLOCK_DIM,
        )

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)
    configure_shared_memory.assert_not_called()


def test_cuda_kernel_properties_avoids_cluster_configuration(test, device):
    """Verify that a fresh resource query avoids cluster configuration."""
    if device.arch < 90:
        test.skipTest("CUDA thread block clusters require sm_90 or newer.")

    compile_arch = max((arch for arch in wp.get_cuda_supported_archs() if 90 <= arch <= device.arch), default=None)
    if compile_arch is None:
        test.skipTest("NVRTC has no cluster-capable target for this device.")

    original_arch = wp.config.ptx_target_arch
    test.addCleanup(setattr, wp.config, "ptx_target_arch", original_arch)
    wp.config.ptx_target_arch = compile_arch
    inspection_attribute_kernel.module.unload()

    with mock.patch.object(
        warp_context.runtime.core,
        "wp_cuda_set_kernel_cluster_attrs",
        wraps=warp_context.runtime.core.wp_cuda_set_kernel_cluster_attrs,
    ) as set_cluster_attrs:
        properties = wp.get_cuda_kernel_properties(
            inspection_attribute_kernel,
            device=device,
            block_dim=ATTRIBUTE_BLOCK_DIM,
        )

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)
    set_cluster_attrs.assert_not_called()


def test_cuda_kernel_properties_accepts_constructor_kernel(test, device):
    """Verify that resource queries accept an explicitly constructed Warp kernel."""
    kernel = wp.Kernel(func=constructor_kernel_func)

    properties = wp.get_cuda_kernel_properties(
        kernel,
        device=device,
        block_dim=ATTRIBUTE_BLOCK_DIM,
    )

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)


def test_cuda_kernel_properties_accept_generic_overload(test, device):
    """Verify that resource queries accept a concrete generic-kernel overload."""
    properties = wp.get_cuda_kernel_properties(
        generic_float_attribute_kernel,
        device=device,
        block_dim=ATTRIBUTE_BLOCK_DIM,
    )

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)


def test_cuda_kernel_properties_reject_invalid_block_dim(test, device):
    """Verify that resource queries reject non-positive and non-integral block dimensions."""
    invalid_block_dims = (True, 0, -1, 1.5, "64")

    with mock.patch.object(attribute_kernel.module, "load", wraps=attribute_kernel.module.load) as load:
        for block_dim in invalid_block_dims:
            with test.subTest(block_dim=block_dim):
                with test.assertRaisesRegex(ValueError, "block_dim must be a positive integer"):
                    wp.get_cuda_kernel_properties(attribute_kernel, device=device, block_dim=block_dim)

    load.assert_not_called()


def test_cuda_kernel_properties_accepts_zero_local_memory(test, device):
    """Verify that zero local-memory bytes is treated as a valid driver result."""

    def query_zero(_context, _forward, values, count):
        test.assertEqual(count, 2)
        values[0] = 7
        values[1] = 0
        return True

    with mock.patch.object(warp_context.runtime.core, "wp_cuda_get_kernel_properties", side_effect=query_zero):
        result = wp.get_cuda_kernel_properties(
            attribute_kernel,
            device=device,
            block_dim=ATTRIBUTE_BLOCK_DIM,
        )

    test.assertEqual(result, {"register_count": 7, "local_memory_size": 0})


def test_cuda_kernel_properties_report_load_failure(test, device):
    """Verify that module-load failures identify the kernel, device, and block dimension."""
    with (
        mock.patch.object(attribute_kernel.module, "load", side_effect=RuntimeError("mock load failure")),
        test.assertRaisesRegex(
            RuntimeError,
            rf"Failed to load CUDA kernel '{attribute_kernel.key}'.*{device}.*block_dim={ATTRIBUTE_BLOCK_DIM}",
        ),
    ):
        wp.get_cuda_kernel_properties(
            attribute_kernel,
            device=device,
            block_dim=ATTRIBUTE_BLOCK_DIM,
        )


def test_cuda_kernel_properties_report_missing_hook(test, device):
    """Verify that a missing forward hook produces a contextual runtime error."""
    module_exec = mock.Mock()
    module_exec._get_forward_cuda_kernel.return_value = None

    with (
        mock.patch.object(attribute_kernel.module, "load", return_value=module_exec),
        test.assertRaisesRegex(
            RuntimeError,
            rf"Failed to load forward CUDA kernel '{attribute_kernel.key}'.*{device}.*block_dim={ATTRIBUTE_BLOCK_DIM}",
        ),
    ):
        wp.get_cuda_kernel_properties(
            attribute_kernel,
            device=device,
            block_dim=ATTRIBUTE_BLOCK_DIM,
        )


def test_cuda_kernel_properties_use_forward_entry_point(test, device):
    """Verify that resource queries pass the forward CUDA entry point to the driver."""
    forward_handle = object()
    backward_handle = object()
    module_exec = mock.Mock()
    module_exec._get_forward_cuda_kernel.return_value = forward_handle

    def query(context, forward, values, count):
        test.assertEqual(context, device.context)
        test.assertIs(forward, forward_handle)
        test.assertEqual(count, 2)
        values[0] = 7
        values[1] = 0
        return True

    with (
        mock.patch.object(attribute_kernel.module, "load", return_value=module_exec),
        mock.patch.object(
            warp_context.runtime.core, "wp_cuda_get_kernel_properties", side_effect=query
        ) as native_query,
    ):
        result = wp.get_cuda_kernel_properties(attribute_kernel, device=device, block_dim=ATTRIBUTE_BLOCK_DIM)

    test.assertEqual(result, {"register_count": 7, "local_memory_size": 0})
    native_query.assert_called_once()
    test.assertIsNot(native_query.call_args.args[1], backward_handle)
    module_exec._get_forward_cuda_kernel.assert_called_once_with(attribute_kernel)


def test_cuda_kernel_properties_retain_module_exec(test, device):
    """Verify that resource queries retain the loaded module during the native call."""
    attribute_kernel.module.unload()
    module_exec = attribute_kernel.module.load(device, ATTRIBUTE_BLOCK_DIM)
    module_exec_ref = weakref.ref(module_exec)
    del module_exec

    def unload_during_query(_context, _forward, values, count):
        attribute_kernel.module.unload()
        test.assertIsNotNone(module_exec_ref())
        test.assertEqual(count, 2)
        values[0] = 7
        values[1] = 0
        return True

    with mock.patch.object(
        warp_context.runtime.core,
        "wp_cuda_get_kernel_properties",
        side_effect=unload_during_query,
    ):
        properties = wp.get_cuda_kernel_properties(attribute_kernel, device=device, block_dim=ATTRIBUTE_BLOCK_DIM)

    test.assertGreater(properties["register_count"], 0)
    test.assertGreaterEqual(properties["local_memory_size"], 0)
    test.assertIsNone(module_exec_ref())


def test_cuda_kernel_properties_report_driver_failure(test, device):
    """Verify that failed driver queries preserve the CUDA error text."""

    def fail_partial(_context, _forward, values, _count):
        values[0] = 7
        return False

    with (
        mock.patch.object(
            warp_context.runtime.core,
            "wp_cuda_get_kernel_properties",
            side_effect=fail_partial,
        ),
        mock.patch.object(warp_context.runtime, "get_error_string", return_value="mock CUDA error"),
        test.assertRaisesRegex(RuntimeError, "Failed to query CUDA kernel properties.*mock CUDA error"),
    ):
        wp.get_cuda_kernel_properties(attribute_kernel, device=device, block_dim=ATTRIBUTE_BLOCK_DIM)


devices = get_selected_cuda_test_devices()


class TestKernelAttributes(unittest.TestCase):
    def test_cuda_kernel_properties_reject_generic_parent(self):
        """Verify that resource queries require a concrete generic-kernel overload."""
        with self.assertRaisesRegex(RuntimeError, "requires a concrete overload.*wp.overload"):
            wp.get_cuda_kernel_properties(generic_attribute_kernel)

    def test_cuda_kernel_properties_reject_non_kernel(self):
        """Verify that resource queries reject undecorated callables and strings."""

        def undecorated_kernel():
            return None

        for invalid_kernel in (undecorated_kernel, "not a kernel"):
            with self.subTest(invalid_kernel=invalid_kernel):
                with self.assertRaisesRegex(TypeError, "expected a wp.Kernel"):
                    wp.get_cuda_kernel_properties(invalid_kernel)

    def test_cuda_kernel_properties_reject_cpu(self):
        """Verify that CUDA resource queries reject CPU devices."""
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
            wp.get_cuda_kernel_properties(attribute_kernel, device="cpu")


add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_public_properties",
    test_cuda_kernel_public_properties,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_use_one_native_batch",
    test_cuda_kernel_properties_use_one_native_batch,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_loads_requested_variant",
    test_cuda_kernel_properties_loads_requested_variant,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_accepts_index_block_dim",
    test_cuda_kernel_properties_accepts_index_block_dim,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_uses_module_default",
    test_cuda_kernel_properties_uses_module_default,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_compiles_lazily",
    test_cuda_kernel_properties_compiles_lazily,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_avoids_shared_memory_configuration",
    test_cuda_kernel_properties_avoids_shared_memory_configuration,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_avoids_cluster_configuration",
    test_cuda_kernel_properties_avoids_cluster_configuration,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_accepts_constructor_kernel",
    test_cuda_kernel_properties_accepts_constructor_kernel,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_accept_generic_overload",
    test_cuda_kernel_properties_accept_generic_overload,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_reject_invalid_block_dim",
    test_cuda_kernel_properties_reject_invalid_block_dim,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_accepts_zero_local_memory",
    test_cuda_kernel_properties_accepts_zero_local_memory,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_report_load_failure",
    test_cuda_kernel_properties_report_load_failure,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_report_missing_hook",
    test_cuda_kernel_properties_report_missing_hook,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_use_forward_entry_point",
    test_cuda_kernel_properties_use_forward_entry_point,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_retain_module_exec",
    test_cuda_kernel_properties_retain_module_exec,
    devices=devices,
)
add_function_test(
    TestKernelAttributes,
    "test_cuda_kernel_properties_report_driver_failure",
    test_cuda_kernel_properties_report_driver_failure,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
