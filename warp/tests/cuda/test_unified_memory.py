# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-device array access and launch verification modes.

These tests cover Warp's conservative memory-access capability reporting,
default launch behavior for mixed-device array arguments, and opt-in launch
verification through ``wp.config.launch_verification_mode`` enum modes. They
also check that checked verification uses allocation-specific CUDA access rules
where possible: ordinary CPU memory, pinned CPU memory, default CUDA
allocations, CUDA memory pool allocations, and array views backed by a parent
allocation.
"""

import contextlib
import unittest
from unittest.mock import patch

import numpy as np

import warp as wp
import warp._src.context as warp_context
from warp.tests.unittest_utils import *


@contextlib.contextmanager
def launch_verification_mode(mode: wp.LaunchVerificationMode):
    """Temporarily set launch array-access verification mode and restore the previous value."""

    old_value = wp.config.launch_verification_mode
    wp.config.launch_verification_mode = mode
    try:
        yield
    finally:
        wp.config.launch_verification_mode = old_value


@contextlib.contextmanager
def emulate_non_coherent_uva_cuda_device(device):
    """Temporarily emulate a CUDA device that can access pinned but not pageable CPU memory."""

    old_is_uva = device.is_uva
    old_cpu_memory_access = device.is_cpu_memory_access_from_gpu_supported
    device.is_uva = True
    device.is_cpu_memory_access_from_gpu_supported = False
    try:
        yield
    finally:
        device.is_uva = old_is_uva
        device.is_cpu_memory_access_from_gpu_supported = old_cpu_memory_access


class DelegatingAllocator:
    """Allocator that delegates to the built-in default CUDA allocator."""

    def __init__(self, device):
        self._inner = device.default_allocator

    def allocate(self, size_in_bytes):
        return self._inner.allocate(size_in_bytes)

    def deallocate(self, ptr, size_in_bytes):
        self._inner.deallocate(ptr, size_in_bytes)


@wp.kernel
def read_cpu_write_gpu(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i] * 2.0


@wp.kernel
def read_indexed_cpu_write_gpu(src: wp.indexedarray[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i] * 2.0


@wp.kernel
def write_output_array(dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = float(i) + 10.0


@wp.kernel
def read_gpu_write_cpu(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i] + 3.0


def test_unified_memory_device_capabilities(test, device):
    """Memory-access capability flags are exposed as booleans on every device."""

    for attr in (
        "is_cpu_memory_access_from_gpu_supported",
        "is_gpu_memory_access_from_cpu_supported",
        "is_cpu_gpu_atomic_supported",
    ):
        test.assertIsInstance(getattr(device, attr), bool)

    if device.is_cpu:
        test.assertFalse(device.is_cpu_memory_access_from_gpu_supported)
        test.assertFalse(device.is_gpu_memory_access_from_cpu_supported)
        test.assertFalse(device.is_cpu_gpu_atomic_supported)


def test_unified_memory_launch_verification_mode_config(test, device):
    """Launch verification mode is an enum-backed public config setting."""

    test.assertIs(wp.LaunchVerificationMode, wp.config.LaunchVerificationMode)
    test.assertEqual(int(wp.LaunchVerificationMode.RELAXED), 0)
    test.assertEqual(int(wp.LaunchVerificationMode.CHECKED), 1)
    test.assertEqual(int(wp.LaunchVerificationMode.STRICT), 2)
    test.assertIs(wp.config.launch_verification_mode, wp.LaunchVerificationMode.RELAXED)
    old_config_name = "verify_launch_" + "array_access"
    test.assertFalse(hasattr(wp.config, old_config_name))

    old_value = wp.config.launch_verification_mode
    try:
        for mode in wp.LaunchVerificationMode:
            wp.config.launch_verification_mode = mode
            test.assertIs(wp.config.launch_verification_mode, mode)

        for value in (False, True, 0, 1, 2, 999, "checked"):
            with test.assertRaisesRegex(ValueError, "launch_verification_mode"):
                wp.config.launch_verification_mode = value
    finally:
        wp.config.launch_verification_mode = old_value


def test_unified_memory_can_access(test, device):
    """Device and array access queries report conservative reachability."""

    cpu = wp.get_device("cpu")
    cpu_array = wp.empty(4, dtype=wp.float32, device=cpu)

    test.assertTrue(device.can_access(device))
    test.assertTrue(cpu.can_access(cpu))
    test.assertTrue(wp.can_access(cpu, cpu_array))

    with test.assertRaisesRegex(TypeError, "Warp arrays"):
        wp.can_access(cpu, cpu)

    annotation_only_resources = (
        wp.array(dtype=wp.float32),
        wp.array[wp.float32],
        wp.indexedarray(dtype=wp.float32),
        wp.indexedarray[wp.float32],
    )
    for resource in annotation_only_resources:
        with test.subTest(resource=resource):
            with test.assertRaisesRegex(TypeError, "concrete Warp array"):
                wp.can_access(cpu, resource)

    if device.is_cuda:
        cuda_array = wp.empty(4, dtype=wp.float32, device=device)

        test.assertEqual(device.can_access(cpu), device.is_cpu_memory_access_from_gpu_supported)
        test.assertFalse(cpu.can_access(device))
        test.assertEqual(wp.can_access(device, cpu_array), device.is_cpu_memory_access_from_gpu_supported)
        test.assertFalse(wp.can_access(cpu, cuda_array))

        if device.is_uva:
            pinned_cpu_array = wp.empty(4, dtype=wp.float32, device=cpu, pinned=True)
            test.assertTrue(wp.can_access(device, pinned_cpu_array))

        for other in wp.get_cuda_devices():
            if other == device:
                test.assertTrue(device.can_access(other))
            else:
                if other.is_mempool_enabled:
                    test.assertEqual(device.can_access(other), wp.is_mempool_access_enabled(other, device))
                else:
                    test.assertEqual(device.can_access(other), wp.is_peer_access_enabled(other, device))


def test_unified_memory_checked_rejects_indexedarray_with_inaccessible_indices(test, device):
    """Indexed array launch verification must validate both data and index arrays."""

    data = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device="cpu", pinned=True)
    indices = wp.array(np.array([0, 1, 2, 3], dtype=np.int32), dtype=wp.int32, device="cpu")
    src = wp.indexedarray1d(data, [indices])
    dst = wp.empty(src.size, dtype=wp.float32, device=device)

    test.assertTrue(data.pinned)
    test.assertFalse(indices.pinned)

    with emulate_non_coherent_uva_cuda_device(device):
        test.assertFalse(wp.can_access(device, src))
        with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
            with test.assertRaisesRegex(RuntimeError, "array allocation is not accessible or cannot be verified"):
                wp.launch(read_indexed_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device)


def test_unified_memory_record_cmd_skips_default_access_check(test, device):
    """Command recording should not restore the old unconditional same-device check.

    This covers the non-executing ``record_cmd=True`` path, which used to run the
    same array packing validation as an immediate launch. Relaxed mode should
    still allow the command object to be created with mixed CPU/CUDA arguments.
    """

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device="cpu")
    dst = wp.empty(4, dtype=wp.float32, device=device)

    with launch_verification_mode(wp.LaunchVerificationMode.RELAXED):
        cmd = wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device, record_cmd=True)

    test.assertIsInstance(cmd, wp.Launch)


def test_unified_memory_verify_rejects_gpu_reading_cpu_when_unsupported(test, device):
    """Opt-in launch verification catches unsupported GPU access to CPU memory."""

    if device.is_cpu_memory_access_from_gpu_supported:
        test.skipTest(f"{device} can access CPU memory")

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device="cpu")
    dst = wp.empty(4, dtype=wp.float32, device=device)

    with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
        with test.assertRaisesRegex(RuntimeError, "array allocation is not accessible or cannot be verified"):
            wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device, record_cmd=True)


def test_unified_memory_verify_rejects_cpu_reading_gpu_when_unsupported(test, device):
    """Warp default CUDA allocations are not treated as CPU-accessible managed memory.

    CUDA exposes a host-to-managed-memory capability, but Warp's built-in CUDA
    arrays are allocated with default device allocation APIs. Checked launch
    verification must therefore reject CPU launches that receive those arrays.
    """

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device=device)
    dst = wp.empty(4, dtype=wp.float32, device="cpu")

    with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
        with test.assertRaisesRegex(RuntimeError, "array allocation is not accessible or cannot be verified"):
            wp.launch(read_gpu_write_cpu, dim=src.size, inputs=[src], outputs=[dst], device="cpu", record_cmd=True)


def test_unified_memory_relaxed_allows_cpu_launch_with_gpu_array(test, device):
    """Relaxed mode performs no launch array access check, even for CPU launches."""

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device=device)
    dst = wp.empty(4, dtype=wp.float32, device="cpu")

    with launch_verification_mode(wp.LaunchVerificationMode.RELAXED):
        cmd = wp.launch(read_gpu_write_cpu, dim=src.size, inputs=[src], outputs=[dst], device="cpu", record_cmd=True)

    test.assertIsInstance(cmd, wp.Launch)


def test_unified_memory_strict_rejects_cuda_launch_with_pinned_cpu_array(test, device):
    """Strict mode restores the original same-device rule."""

    if not device.is_uva:
        test.skipTest(f"{device} does not support unified virtual addressing")

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device="cpu", pinned=True)
    dst = wp.empty(4, dtype=wp.float32, device=device)

    with launch_verification_mode(wp.LaunchVerificationMode.STRICT):
        with test.assertRaisesRegex(RuntimeError, "is on device=cpu"):
            wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device, record_cmd=True)


def test_unified_memory_cuda_launch_reads_cpu_array_when_supported(test, device):
    """On coherent systems, GPU kernels can read ordinary CPU arrays directly."""

    if not device.is_cpu_memory_access_from_gpu_supported:
        test.skipTest(f"{device} cannot access CPU memory")

    src_np = np.arange(8, dtype=np.float32)
    src = wp.array(src_np, dtype=wp.float32, device="cpu")
    dst = wp.empty(src.size, dtype=wp.float32, device=device)

    with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
        wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device)

    np.testing.assert_allclose(dst.numpy(), src_np * 2.0)


def test_unified_memory_cuda_launch_writes_cpu_array_when_supported(test, device):
    """On coherent systems, GPU kernels can write ordinary CPU arrays directly."""

    if not device.is_cpu_memory_access_from_gpu_supported:
        test.skipTest(f"{device} cannot access CPU memory")

    dst = wp.empty(8, dtype=wp.float32, device="cpu")

    with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
        wp.launch(write_output_array, dim=dst.size, outputs=[dst], device=device)

    # dst is CPU memory written by the GPU; CPU-backed .numpy() does not synchronize the launch.
    wp.synchronize_device(device)
    np.testing.assert_allclose(dst.numpy(), np.arange(8, dtype=np.float32) + 10.0)


def test_unified_memory_cuda_launch_reads_pinned_cpu_array_when_uva_supported(test, device):
    """Pinned CPU arrays are GPU-accessible on CUDA devices with unified virtual addressing."""

    if not device.is_uva:
        test.skipTest(f"{device} does not support unified virtual addressing")

    src_np = np.arange(8, dtype=np.float32)
    src = wp.array(src_np, dtype=wp.float32, device="cpu", pinned=True)
    dst = wp.empty(src.size, dtype=wp.float32, device=device)

    test.assertTrue(src.pinned)

    with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
        wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device)

    np.testing.assert_allclose(dst.numpy(), src_np * 2.0)


def test_unified_memory_cuda_launch_writes_pinned_cpu_array_when_uva_supported(test, device):
    """Pinned CPU output arrays are valid GPU launch targets on UVA CUDA devices."""

    if not device.is_uva:
        test.skipTest(f"{device} does not support unified virtual addressing")

    dst = wp.empty(8, dtype=wp.float32, device="cpu", pinned=True)

    test.assertTrue(dst.pinned)

    with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
        wp.launch(write_output_array, dim=dst.size, outputs=[dst], device=device)

    # dst is CPU memory written by the GPU; CPU-backed .numpy() does not synchronize the launch.
    wp.synchronize_device(device)
    np.testing.assert_allclose(dst.numpy(), np.arange(8, dtype=np.float32) + 10.0)


def test_unified_memory_array_view_allocator_lookup_uses_parent_array(test, device):
    """Array views must use the base allocation when launch verification checks access.

    Sliced arrays do not own the allocation and may not carry an allocator
    directly. Launch verification needs to walk back to the parent array so
    slices inherit the same cross-device access rules as the storage owner.
    """

    src = wp.array(np.arange(8, dtype=np.float32), dtype=wp.float32, device=device)
    src_slice = src[1:]

    test.assertIs(wp._src.context._get_array_allocator(src_slice), src._allocator)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestUnifiedMemory(unittest.TestCase):
    def test_unified_memory_unknown_access_warning_cache_is_bounded(self):
        """Unknown-access warning deduplication should not grow without bound."""

        cache = warp_context._launch_array_access_warnings_seen
        saved_cache = cache.copy()
        cache.clear()
        try:
            kernel = type("Kernel", (), {"key": "warning-cache-bounds"})()
            value = wp.empty(1, dtype=wp.float32, device="cpu")
            device = wp.get_device("cpu")
            cache_size = warp_context._LAUNCH_ARRAY_ACCESS_WARNING_CACHE_SIZE

            with patch("warp._src.context.log_warning"):
                for i in range(cache_size + 1):
                    warp_context._warn_unknown_launch_array_access(kernel, f"arg{i}", value, device)

            self.assertLessEqual(len(cache), cache_size)
        finally:
            cache.clear()
            cache.update(saved_cache)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_unified_memory_checked_warns_once_for_custom_allocator(self):
        """CHECKED warns once for unknown custom allocator provenance.

        The delegating allocator returns an ordinary CUDA pointer, but the
        launch verifier only sees an unknown custom allocator. Checked mode
        should warn about the unverified cross-device launch pattern without
        warning again for the same kernel/argument/device combination.
        """

        device = wp.get_device("cuda:0")
        cpu = wp.get_device("cpu")
        n = 4
        alloc = DelegatingAllocator(device)

        with wp.ScopedAllocator(device, alloc):
            src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=device)
        dst = wp.empty(n, dtype=wp.float32, device=cpu)

        self.assertFalse(wp.can_access(cpu, src))

        with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
            with patch("warp._src.context.log_warning") as mock_log_warning:
                cmd0 = wp.launch(read_gpu_write_cpu, dim=n, inputs=[src], outputs=[dst], device=cpu, record_cmd=True)
                cmd1 = wp.launch(read_gpu_write_cpu, dim=n, inputs=[src], outputs=[dst], device=cpu, record_cmd=True)

        self.assertIsInstance(cmd0, wp.Launch)
        self.assertIsInstance(cmd1, wp.Launch)
        matching = [
            call for call in mock_log_warning.call_args_list if "cannot verify cross-device access" in call.args[0]
        ]
        self.assertEqual(len(matching), 1)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_unified_memory_strict_rejects_custom_allocator_cross_device(self):
        """STRICT still rejects cross-device arrays with custom allocators.

        Strict mode intentionally restores Warp's old same-device policy before
        allocator-specific reachability matters. This keeps custom allocator
        provenance from creating a loophole in strict validation.
        """

        device = wp.get_device("cuda:0")
        cpu = wp.get_device("cpu")
        n = 4
        alloc = DelegatingAllocator(device)

        with wp.ScopedAllocator(device, alloc):
            src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=device)
        dst = wp.empty(n, dtype=wp.float32, device=cpu)

        with launch_verification_mode(wp.LaunchVerificationMode.STRICT):
            with self.assertRaisesRegex(RuntimeError, "is on device="):
                wp.launch(read_gpu_write_cpu, dim=n, inputs=[src], outputs=[dst], device=cpu, record_cmd=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_unified_memory_relaxed_does_not_warn_for_custom_allocator(self):
        """RELAXED keeps passing unknown custom allocator launches through silently.

        Relaxed mode is the default pass-through policy for users who already
        know their hardware and allocation are valid. Unknown custom allocator
        provenance should not emit the checked-mode diagnostic in this mode.
        """

        device = wp.get_device("cuda:0")
        cpu = wp.get_device("cpu")
        n = 4
        alloc = DelegatingAllocator(device)

        with wp.ScopedAllocator(device, alloc):
            src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=device)
        dst = wp.empty(n, dtype=wp.float32, device=cpu)

        with launch_verification_mode(wp.LaunchVerificationMode.RELAXED):
            with patch("warp._src.context.log_warning") as mock_log_warning:
                cmd = wp.launch(read_gpu_write_cpu, dim=n, inputs=[src], outputs=[dst], device=cpu, record_cmd=True)

        self.assertIsInstance(cmd, wp.Launch)
        matching = [
            call for call in mock_log_warning.call_args_list if "cannot verify cross-device access" in call.args[0]
        ]
        self.assertEqual(len(matching), 0)

    @unittest.skipUnless(get_cuda_device_pair_with_peer_access_support(), "Requires devices with peer access support")
    def test_unified_memory_verify_uses_peer_access_for_default_cuda_allocations(self):
        """Default CUDA allocations use peer-access state for cross-GPU verification.

        Peer access and mempool access are separate CUDA capabilities. When the
        source array was allocated through Warp's default CUDA allocator,
        checked launch verification should accept the launch based on peer
        access even if mempool access is disabled.
        """

        target_device, peer_device = get_cuda_device_pair_with_peer_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_mempool_access_enabled(target_device, peer_device, False)
            wp.set_peer_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, False), wp.ScopedMempool(peer_device, False):
                src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=target_device)
                dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src._allocator).__name__, "CudaDefaultAllocator")
            self.assertTrue(wp.can_access(peer_device, src))

            wp.load_module(device=peer_device)
            peer_device.stream.wait_stream(target_device.stream)
            with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
                wp.launch(read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device)

            np.testing.assert_allclose(dst.numpy(), np.arange(n, dtype=np.float32) * 2.0)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(get_cuda_device_pair_with_peer_access_support(), "Requires devices with peer access support")
    def test_unified_memory_verify_uses_parent_allocator_for_default_cuda_slices(self):
        """Slices of default CUDA allocations should follow the base array's allocator.

        This exercises the same peer-access path as a full default CUDA array,
        but through a view that does not own storage. The verifier must inspect
        the parent allocation instead of treating the slice as unknown.
        """

        target_device, peer_device = get_cuda_device_pair_with_peer_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_mempool_access_enabled(target_device, peer_device, False)
            wp.set_peer_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, False), wp.ScopedMempool(peer_device, False):
                src_base = wp.array(np.arange(n + 1, dtype=np.float32), dtype=wp.float32, device=target_device)
                src = src_base[1:]
                dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src_base._allocator).__name__, "CudaDefaultAllocator")
            self.assertIs(src._ref, src_base)

            wp.load_module(device=peer_device)
            peer_device.stream.wait_stream(target_device.stream)
            with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
                wp.launch(read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device)

            np.testing.assert_allclose(dst.numpy(), np.arange(1, n + 1, dtype=np.float32) * 2.0)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(
        get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support"
    )
    def test_unified_memory_device_can_access_uses_mempool_state_when_target_mempools_enabled(self):
        """Device.can_access() follows the target device's current built-in allocator mode.

        This is a coarse device-level query, not an existing-allocation query.
        If the target device would currently allocate through CUDA mempools, the
        answer must come from mempool access state even when peer access is
        enabled for default CUDA allocations.
        """

        target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_peer_access_enabled(target_device, peer_device, True)
            wp.set_mempool_access_enabled(target_device, peer_device, False)

            with wp.ScopedMempool(target_device, True):
                self.assertTrue(target_device.is_mempool_enabled)
                self.assertFalse(peer_device.can_access(target_device))

            with wp.ScopedMempool(target_device, False):
                self.assertFalse(target_device.is_mempool_enabled)
                self.assertTrue(peer_device.can_access(target_device))
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(
        get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support"
    )
    def test_unified_memory_verify_uses_mempool_access_for_cuda_mempool_allocations(self):
        """CUDA mempool allocations use mempool-access state for cross-GPU verification.

        An array allocated while the source device's mempool is enabled needs
        the CUDA mempool access predicate. The companion rejection test keeps
        peer access enabled while mempool access is disabled, so the pair
        isolates the allocation-specific mempool rule without executing this
        peer kernel in a recently changed pool-access state with peer access
        disabled.
        """

        target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_peer_access_enabled(target_device, peer_device, True)
            wp.set_mempool_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, True):
                src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=target_device)
            dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src._allocator).__name__, "CudaMempoolAllocator")
            self.assertTrue(wp.can_access(peer_device, src))

            wp.load_module(device=peer_device)
            peer_device.stream.wait_stream(target_device.stream)
            with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
                wp.launch(read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device)

            np.testing.assert_allclose(dst.numpy(), np.arange(n, dtype=np.float32) * 2.0)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(
        get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support"
    )
    def test_unified_memory_verify_uses_parent_allocator_for_cuda_mempool_slices(self):
        """Slices of CUDA mempool allocations should follow the base array's allocator.

        This covers the view case for CUDA mempool-backed storage. Checked
        verification must follow the slice's parent allocation and then apply
        mempool access rules, rather than falling back to unknown provenance.
        """

        target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_peer_access_enabled(target_device, peer_device, True)
            wp.set_mempool_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, True):
                src_base = wp.array(np.arange(n + 1, dtype=np.float32), dtype=wp.float32, device=target_device)
                src = src_base[1:]
            dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src_base._allocator).__name__, "CudaMempoolAllocator")
            self.assertIs(src._ref, src_base)

            wp.load_module(device=peer_device)
            peer_device.stream.wait_stream(target_device.stream)
            with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
                wp.launch(read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device)

            np.testing.assert_allclose(dst.numpy(), np.arange(1, n + 1, dtype=np.float32) * 2.0)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(
        get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support"
    )
    def test_unified_memory_verify_rejects_mempool_allocation_without_mempool_access(self):
        """Peer access alone should not validate cross-GPU CUDA mempool allocations.

        Default CUDA allocations and CUDA mempool allocations have different
        cross-device access switches. A mempool-backed source array should be
        rejected in checked mode when Warp observes mempool access as disabled,
        even if normal peer access between the devices is enabled.

        The mempool access predicate is mocked instead of toggling the actual
        CUDA pool access state because CUDA recommends keeping pool
        accessibility stable over the lifetime of the pool.
        """

        target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        try:
            wp.set_peer_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, True):
                src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=target_device)
            dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src._allocator).__name__, "CudaMempoolAllocator")

            # Do not toggle the real CUDA default pool access here. CUDA recommends keeping pool accessibility
            # stable, and toggling it in this rejection test can make later real peer-read tests flaky on some drivers.
            with patch.object(warp_context, "is_mempool_access_enabled", return_value=False) as mock_access:
                self.assertFalse(wp.can_access(peer_device, src))
                mock_access.assert_called_with(target_device, peer_device)

                with launch_verification_mode(wp.LaunchVerificationMode.CHECKED):
                    mock_access.reset_mock()
                    with self.assertRaisesRegex(
                        RuntimeError, "array allocation is not accessible or cannot be verified"
                    ):
                        wp.launch(
                            read_cpu_write_gpu,
                            dim=n,
                            inputs=[src],
                            outputs=[dst],
                            device=peer_device,
                            record_cmd=True,
                        )
                    mock_access.assert_called_with(target_device, peer_device)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)


add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_device_capabilities",
    test_unified_memory_device_capabilities,
    devices=devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_launch_verification_mode_config",
    test_unified_memory_launch_verification_mode_config,
    devices=[wp.get_device("cpu")],
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_can_access",
    test_unified_memory_can_access,
    devices=devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_checked_rejects_indexedarray_with_inaccessible_indices",
    test_unified_memory_checked_rejects_indexedarray_with_inaccessible_indices,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_record_cmd_skips_default_access_check",
    test_unified_memory_record_cmd_skips_default_access_check,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_verify_rejects_gpu_reading_cpu_when_unsupported",
    test_unified_memory_verify_rejects_gpu_reading_cpu_when_unsupported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_verify_rejects_cpu_reading_gpu_when_unsupported",
    test_unified_memory_verify_rejects_cpu_reading_gpu_when_unsupported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_relaxed_allows_cpu_launch_with_gpu_array",
    test_unified_memory_relaxed_allows_cpu_launch_with_gpu_array,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_strict_rejects_cuda_launch_with_pinned_cpu_array",
    test_unified_memory_strict_rejects_cuda_launch_with_pinned_cpu_array,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cuda_launch_reads_cpu_array_when_supported",
    test_unified_memory_cuda_launch_reads_cpu_array_when_supported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cuda_launch_writes_cpu_array_when_supported",
    test_unified_memory_cuda_launch_writes_cpu_array_when_supported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cuda_launch_reads_pinned_cpu_array_when_uva_supported",
    test_unified_memory_cuda_launch_reads_pinned_cpu_array_when_uva_supported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cuda_launch_writes_pinned_cpu_array_when_uva_supported",
    test_unified_memory_cuda_launch_writes_pinned_cpu_array_when_uva_supported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_array_view_allocator_lookup_uses_parent_array",
    test_unified_memory_array_view_allocator_lookup_uses_parent_array,
    devices=devices,
)
if __name__ == "__main__":
    unittest.main(verbosity=2)
