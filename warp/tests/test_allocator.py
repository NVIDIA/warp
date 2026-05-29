# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np

import warp as wp
from warp._src.context import Allocator
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices

wp.init()

cuda_test_devices = get_cuda_test_devices(mode="basic")

# RMM imports NVIDIA's top-level ``cuda`` package. Keep this probe after
# ``unittest_utils`` normalizes direct-test ``sys.path`` so ``warp/tests/cuda``
# does not shadow the installed CUDA package when this file is run by path.
try:
    import rmm

    rmm_available = True
except ImportError:
    rmm_available = False

_TORCH_CUDA_PROBE_EXCEPTIONS = (RuntimeError, OSError, ValueError)


def _try_import_torch():
    try:
        import torch  # noqa: PLC0415
    except (ImportError, OSError):
        return None, False

    return torch, True


torch, torch_available = _try_import_torch()


class CountingAllocator:
    """Test allocator that counts allocations and delegates to the built-in."""

    def __init__(self, device):
        self._inner = device.default_allocator
        self.alloc_count = 0
        self.dealloc_count = 0

    def allocate(self, size_in_bytes):
        self.alloc_count += 1
        return self._inner.allocate(size_in_bytes)

    def deallocate(self, ptr, size_in_bytes):
        self.dealloc_count += 1
        self._inner.deallocate(ptr, size_in_bytes)


class FailAllocator:
    """Test allocator that always raises on allocate()."""

    def allocate(self, size_in_bytes):
        raise RuntimeError("deliberate failure")

    def deallocate(self, ptr, size_in_bytes):
        pass


class TorchCachingAllocatorForWarp:
    """Test allocator that routes CUDA allocations through PyTorch's cache."""

    def __init__(self):
        self._active_allocations: dict[int, int] = {}
        self.alloc_count = 0
        self.dealloc_count = 0

    @staticmethod
    def _get_warp_device_and_stream() -> tuple[int, int]:
        device = wp.get_cuda_device()
        stream = device.stream.cuda_stream
        return device.ordinal, int(stream) if stream is not None else 0

    def allocate(self, size_in_bytes: int) -> int:
        if size_in_bytes == 0:
            return 0

        device, stream = self._get_warp_device_and_stream()
        ptr = torch.cuda.caching_allocator_alloc(size_in_bytes, device=device, stream=stream)
        if not ptr:
            raise RuntimeError(f"Failed to allocate {size_in_bytes} bytes on CUDA device {device} using PyTorch")

        ptr = int(ptr)
        self.alloc_count += 1
        self._active_allocations[ptr] = size_in_bytes
        return ptr

    def deallocate(self, ptr: int, size_in_bytes: int) -> None:
        if ptr == 0:
            return

        allocated_size = self._active_allocations.get(ptr)
        if allocated_size is None:
            raise RuntimeError(
                f"{type(self).__name__}.deallocate called with unrecognized pointer {ptr:#x} (size={size_in_bytes})"
            )
        if allocated_size != size_in_bytes:
            raise RuntimeError(
                f"{type(self).__name__}.deallocate size mismatch for pointer {ptr:#x}: "
                f"allocated {allocated_size}, deallocating {size_in_bytes}"
            )
        del self._active_allocations[ptr]
        self.dealloc_count += 1

        torch.cuda.caching_allocator_delete(ptr)

    def __repr__(self):
        return f"{type(self).__name__}(active_allocations={len(self._active_allocations)})"


def _get_torch_cuda_allocator_test_devices():
    if not torch_available:
        return []
    if not torch.cuda.is_available():
        return []
    if not hasattr(torch.cuda, "caching_allocator_alloc") or not hasattr(torch.cuda, "caching_allocator_delete"):
        return []

    devices = []
    for device in cuda_test_devices:
        try:
            torch.empty(1, device=wp.device_to_torch(device))
        except _TORCH_CUDA_PROBE_EXCEPTIONS as e:
            print(f"Skipping PyTorch allocator test on device '{device}' due to exception: {e}")
        else:
            devices.append(device)

    return devices


# -- Protocol conformance ---------------------------------------------------


class TestAllocatorProtocol(unittest.TestCase):
    def test_protocol_conformance_cpu(self):
        """Built-in CPU allocators satisfy the Allocator protocol."""
        cpu = wp.get_device("cpu")
        self.assertIsInstance(cpu.default_allocator, Allocator)
        self.assertIsInstance(cpu.pinned_allocator, Allocator)


def test_protocol_conformance_cuda(test, device):
    """Built-in CUDA allocators satisfy the Allocator protocol."""
    device = wp.get_device(device)
    test.assertIsInstance(device.default_allocator, Allocator)
    if device.is_mempool_supported:
        test.assertIsInstance(device.mempool_allocator, Allocator)


add_function_test(
    TestAllocatorProtocol, "test_protocol_conformance_cuda", test_protocol_conformance_cuda, devices=cuda_test_devices
)


# -- Custom allocator -------------------------------------------------------


class TestCustomAllocator(unittest.TestCase):
    @unittest.skipUnless(wp.get_cuda_device_count() >= 2, "Multi-GPU not available")
    def test_set_cuda_allocator_broadcasts_to_all_devices(self):
        """set_cuda_allocator() applies the allocator to every available CUDA device."""
        dev0 = wp.get_device("cuda:0")
        dev1 = wp.get_device("cuda:1")
        alloc = CountingAllocator(dev0)
        wp.set_cuda_allocator(alloc)
        try:
            self.assertIs(wp.get_device_allocator(dev0), alloc)
            self.assertIs(wp.get_device_allocator(dev1), alloc)
        finally:
            wp.set_cuda_allocator(None)

    @unittest.skipUnless(wp.get_cuda_device_count() >= 2, "Multi-GPU not available")
    def test_per_device_isolation(self):
        """Setting allocator on one device does not affect another."""
        dev0 = wp.get_device("cuda:0")
        dev1 = wp.get_device("cuda:1")
        alloc0 = CountingAllocator(dev0)
        alloc1 = CountingAllocator(dev1)
        wp.set_device_allocator(dev0, alloc0)
        wp.set_device_allocator(dev1, alloc1)
        try:
            wp.zeros(100, dtype=wp.float32, device=dev0)
            wp.zeros(200, dtype=wp.float32, device=dev1)
            self.assertEqual(alloc0.alloc_count, 1)
            self.assertEqual(alloc1.alloc_count, 1)
        finally:
            wp.set_device_allocator(dev0, None)
            wp.set_device_allocator(dev1, None)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
    def test_set_device_allocator_cpu_raises(self):
        """set_device_allocator() raises for CPU devices."""
        with self.assertRaises(RuntimeError):
            wp.set_device_allocator("cpu", CountingAllocator(wp.get_device("cuda:0")))


def test_set_cuda_allocator(test, device):
    """set_cuda_allocator() routes array allocations through the custom allocator."""
    device = wp.get_device(device)
    alloc = CountingAllocator(device)
    wp.set_cuda_allocator(alloc)
    try:
        a = wp.zeros(100, dtype=wp.float32, device=device)
        b = wp.empty(100, dtype=wp.float32, device=device)
        c = wp.full(100, value=1.0, dtype=wp.float32, device=device)
        test.assertEqual(alloc.alloc_count, 3)
        del a
        del b
        del c
        test.assertEqual(alloc.dealloc_count, 3)
    finally:
        wp.set_cuda_allocator(None)


def test_set_device_allocator(test, device):
    """set_device_allocator() sets allocator on a specific device."""
    device = wp.get_device(device)
    alloc = CountingAllocator(device)
    wp.set_device_allocator(device, alloc)
    try:
        a = wp.zeros(100, dtype=wp.float32, device=device)
        test.assertEqual(alloc.alloc_count, 1)
    finally:
        wp.set_device_allocator(device, None)


def test_get_device_allocator(test, device):
    """get_device_allocator() returns the effective allocator."""
    device = wp.get_device(device)
    alloc = CountingAllocator(device)
    wp.set_device_allocator(device, alloc)
    try:
        test.assertIs(wp.get_device_allocator(device), alloc)
    finally:
        wp.set_device_allocator(device, None)


def test_reset_to_default(test, device):
    """set_cuda_allocator(None) restores the built-in allocator."""
    device = wp.get_device(device)
    original = wp.get_device_allocator(device)
    wp.set_cuda_allocator(CountingAllocator(device))
    wp.set_cuda_allocator(None)
    test.assertIs(wp.get_device_allocator(device), original)


def test_scoped_allocator(test, device):
    """ScopedAllocator restores the previous allocator on exit."""
    device = wp.get_device(device)
    alloc = CountingAllocator(device)
    original = wp.get_device_allocator(device)
    with wp.ScopedAllocator(device, alloc):
        test.assertIs(wp.get_device_allocator(device), alloc)
        wp.zeros(10, dtype=wp.float32, device=device)
        test.assertEqual(alloc.alloc_count, 1)
    test.assertIs(wp.get_device_allocator(device), original)


def test_scoped_allocator_restores_on_exception(test, device):
    """ScopedAllocator restores allocator even if body raises."""
    device = wp.get_device(device)
    alloc = CountingAllocator(device)
    original = wp.get_device_allocator(device)
    with test.assertRaises(ValueError):
        with wp.ScopedAllocator(device, alloc):
            raise ValueError("test")
    test.assertIs(wp.get_device_allocator(device), original)


def test_allocator_swap_with_live_arrays(test, device):
    """Arrays allocated with a custom allocator survive allocator reset."""
    device = wp.get_device(device)
    alloc = CountingAllocator(device)
    wp.set_device_allocator(device, alloc)
    a = wp.zeros(100, dtype=wp.float32, device=device)
    test.assertEqual(alloc.alloc_count, 1)
    wp.set_device_allocator(device, None)
    del a
    test.assertEqual(alloc.dealloc_count, 1)


def test_allocate_failure(test, device):
    """Allocation failure in custom allocator propagates cleanly."""
    device = wp.get_device(device)
    wp.set_device_allocator(device, FailAllocator())
    try:
        with test.assertRaises(RuntimeError):
            wp.zeros(100, dtype=wp.float32, device=device)
    finally:
        wp.set_device_allocator(device, None)


def test_zero_size_allocation(test, device):
    """Custom allocator is not invoked for zero-size arrays."""
    device = wp.get_device(device)
    alloc = CountingAllocator(device)
    wp.set_device_allocator(device, alloc)
    try:
        a = wp.zeros(0, dtype=wp.float32, device=device)
        test.assertEqual(alloc.alloc_count, 0)
        del a
        test.assertEqual(alloc.dealloc_count, 0)
    finally:
        wp.set_device_allocator(device, None)


for fn in [
    test_set_cuda_allocator,
    test_set_device_allocator,
    test_get_device_allocator,
    test_reset_to_default,
    test_scoped_allocator,
    test_scoped_allocator_restores_on_exception,
    test_allocator_swap_with_live_arrays,
    test_allocate_failure,
    test_zero_size_allocation,
]:
    add_function_test(TestCustomAllocator, fn.__name__, fn, devices=cuda_test_devices)


# -- RMM allocator ----------------------------------------------------------


class TestRmmAllocator(unittest.TestCase):
    @unittest.skipUnless(rmm_available, "rmm not installed")
    @unittest.skipUnless(wp.get_cuda_device_count() >= 2, "Multi-GPU not available")
    def test_rmm_allocator_multi_gpu(self):
        """A single AllocatorRmm instance works across multiple CUDA devices."""
        rmm.reinitialize(pool_allocator=True, initial_pool_size=2**26)

        alloc = wp.utils.AllocatorRmm()
        dev0 = wp.get_device("cuda:0")
        dev1 = wp.get_device("cuda:1")
        wp.set_cuda_allocator(alloc)
        try:
            a0 = wp.zeros(500, dtype=wp.float32, device=dev0)
            a1 = wp.zeros(500, dtype=wp.float32, device=dev1)
            self.assertIn(a0.ptr, alloc._buffers)
            self.assertIn(a1.ptr, alloc._buffers)
            a0.fill_(1.0)
            a1.fill_(2.0)
            np.testing.assert_allclose(a0.numpy(), 1.0)
            np.testing.assert_allclose(a1.numpy(), 2.0)
        finally:
            wp.set_cuda_allocator(None)

    @unittest.skipIf(rmm_available, "rmm is installed")
    def test_rmm_allocator_import_error(self):
        """AllocatorRmm raises ImportError when rmm is not installed."""
        with self.assertRaises(ImportError):
            wp.utils.AllocatorRmm()


def test_rmm_allocator_basic(test, device):
    """AllocatorRmm routes allocations through RMM."""
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2**26)

    device = wp.get_device(device)
    alloc = wp.utils.AllocatorRmm()
    wp.set_device_allocator(device, alloc)
    try:
        a = wp.zeros(1000, dtype=wp.float32, device=device)
        test.assertIn(a.ptr, alloc._buffers)
        a.fill_(42.0)
        np.testing.assert_allclose(a.numpy(), 42.0)
        ptr = a.ptr
        del a
        test.assertNotIn(ptr, alloc._buffers)
    finally:
        wp.set_device_allocator(device, None)


def test_rmm_allocator_interop_torch(test, device):
    """RMM-allocated Warp array can be exported to PyTorch."""
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError:
        test.skipTest("torch not installed")

    rmm.reinitialize(pool_allocator=True, initial_pool_size=2**26)

    device = wp.get_device(device)
    alloc = wp.utils.AllocatorRmm()
    wp.set_device_allocator(device, alloc)
    try:
        a = wp.zeros(100, dtype=wp.float32, device=device)
        a.fill_(7.0)
        t = wp.to_torch(a)
        test.assertEqual(t.shape[0], 100)
        np.testing.assert_allclose(t.cpu().numpy(), 7.0)
    finally:
        wp.set_device_allocator(device, None)


def test_rmm_allocator_double_free(test, device):
    """deallocate() raises RuntimeError for an already-freed or unknown pointer."""
    device = wp.get_device(device)
    alloc = wp.utils.AllocatorRmm()
    wp.set_device_allocator(device, alloc)
    try:
        ptr = alloc.allocate(256)
        test.assertIn(ptr, alloc._buffers)
        alloc.deallocate(ptr, 256)
        test.assertNotIn(ptr, alloc._buffers)
        # Second call with the same pointer must raise.
        with test.assertRaises(RuntimeError):
            alloc.deallocate(ptr, 256)
        # Completely unknown pointer must also raise.
        with test.assertRaises(RuntimeError):
            alloc.deallocate(0xDEADBEEF, 128)
    finally:
        wp.set_device_allocator(device, None)


if rmm_available:
    for fn in [
        test_rmm_allocator_basic,
        test_rmm_allocator_interop_torch,
        test_rmm_allocator_double_free,
    ]:
        add_function_test(TestRmmAllocator, fn.__name__, fn, devices=cuda_test_devices)


# -- PyTorch allocator ------------------------------------------------------


class TestTorchAllocator(unittest.TestCase):
    def test_torch_import_probe_handles_os_error(self):
        """Torch DLL load failures make Torch allocator tests unavailable."""

        real_import = __import__

        def import_with_torch_os_error(name, *args, **kwargs):
            if name == "torch":
                raise OSError("DLL initialization failed")

            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_with_torch_os_error):
            imported_torch, is_available = _try_import_torch()

        self.assertIsNone(imported_torch)
        self.assertFalse(is_available)

    def test_torch_import_probe_propagates_unexpected_errors(self):
        """Unexpected Torch import failures still fail test collection."""

        real_import = __import__

        def import_with_torch_runtime_error(name, *args, **kwargs):
            if name == "torch":
                raise RuntimeError("unexpected import failure")

            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_with_torch_runtime_error):
            with self.assertRaisesRegex(RuntimeError, "unexpected import failure"):
                _try_import_torch()


def test_torch_caching_allocator(test, device):
    """Warp arrays can allocate from PyTorch's CUDA caching allocator."""
    device = wp.get_device(device)
    allocator = TorchCachingAllocatorForWarp()
    baseline_allocated = torch.cuda.memory_allocated(device.ordinal)
    a = None
    t = None

    wp.set_cuda_allocator(allocator)
    try:
        a = wp.empty(128, dtype=wp.float32, device=device)
        test.assertEqual(allocator.alloc_count, 1)
        test.assertIn(a.ptr, allocator._active_allocations)
        test.assertEqual(allocator._active_allocations[a.ptr], a.capacity)
        test.assertGreaterEqual(torch.cuda.memory_allocated(device.ordinal) - baseline_allocated, a.capacity)

        a.fill_(5.0)
        wp.synchronize_device(device)

        t = wp.to_torch(a)
        test.assertEqual(t.data_ptr(), a.ptr)
        np.testing.assert_allclose(t.cpu().numpy(), 5.0)

        ptr = a.ptr
        t = None
        a = None
        torch.cuda.synchronize(device.ordinal)

        test.assertEqual(allocator.dealloc_count, 1)
        test.assertNotIn(ptr, allocator._active_allocations)
        test.assertEqual(torch.cuda.memory_allocated(device.ordinal), baseline_allocated)
    finally:
        t = None
        a = None
        wp.set_cuda_allocator(None)


add_function_test(
    TestTorchAllocator,
    "test_torch_caching_allocator",
    test_torch_caching_allocator,
    devices=_get_torch_cuda_allocator_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
