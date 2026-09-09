# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the deterministic native CPU block scheduler against a scalar model."""

import ctypes
import itertools
import os
import subprocess
import sys
import tempfile
import unittest

import warp as wp


@wp.kernel(module="unique")
def _allocation_failure_kernel(output: wp.array[wp.int32]):
    output[wp.tid()] = wp.block_dim()


def _warp_lib_path():
    bindir = os.path.join(os.path.dirname(wp.__file__), "bin")
    if sys.platform == "win32":
        return os.path.join(bindir, "warp.dll")
    if sys.platform == "darwin":
        return os.path.join(bindir, "libwarp.dylib")
    return os.path.join(bindir, "warp.so")


def _setup_runtime():
    wp.init()
    lib = ctypes.CDLL(_warp_lib_path())
    lib.wp_cpu_test_schedule.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.wp_cpu_test_schedule.restype = ctypes.c_int
    lib.wp_cpu_block_pool_size.restype = ctypes.c_size_t
    lib.wp_cpu_test_fail_next_worker_allocation.argtypes = []
    lib.wp_cpu_test_fail_next_worker_allocation.restype = None
    return lib


def _scalar_schedule(barrier_counts):
    """Return the lowest-lane deterministic schedule from a scalar oracle."""
    active_count = len(barrier_counts)
    generations = [0] * active_count
    barrier_indices = [0] * active_count
    waiting_generations = [None] * active_count
    finished = [False] * active_count
    frontier = 0
    events = []
    current = 0

    while current is not None:
        own_generation = waiting_generations[current]
        if own_generation is not None:
            if frontier > own_generation:
                waiting_generations[current] = None
                continue
            laggards = [
                lane for lane in range(active_count) if not finished[lane] and generations[lane] < own_generation
            ]
            if laggards:
                current = laggards[0]
                continue
            waiting_generations[current] = None
            continue

        if barrier_indices[current] < barrier_counts[current]:
            events.append(current)
            barrier_indices[current] += 1
            generations[current] += 1
            frontier = max(frontier, generations[current])
            waiting_generations[current] = generations[current]
            continue

        events.append(active_count + current)
        finished[current] = True
        unfinished = [lane for lane in range(active_count) if not finished[lane]]
        current = unfinished[0] if unfinished else None

    return events


def _native_schedule(lib, block_dim, barrier_counts):
    active_count = len(barrier_counts)
    counts = (ctypes.c_uint32 * active_count)(*barrier_counts)
    capacity = active_count + sum(barrier_counts)
    events = (ctypes.c_int * capacity)()
    event_count = ctypes.c_size_t()
    result = lib.wp_cpu_test_schedule(
        block_dim,
        active_count,
        counts,
        events,
        capacity,
        ctypes.byref(event_count),
    )
    if not result:
        raise RuntimeError(f"native scheduler rejected block_dim={block_dim}, active_count={active_count}")
    return list(events[: event_count.value])


def _check_pool_reuse():
    lib = _setup_runtime()
    if lib.wp_cpu_block_pool_size() != 0:
        raise RuntimeError("the CPU block fiber pool was not initially empty")

    cases = (
        (1024, [2, 0, 1]),
        (2, [4, 4]),
        (64, [0, 3, 1, 4, 0, 2, 5, 1]),
        (8, [1, 0, 2, 0, 3, 0, 4, 0]),
        (511, [7, 1, 0, 5]),
    )
    expected_pool_sizes = (3, 3, 8, 8, 8)
    for (block_dim, counts), expected_pool_size in zip(cases, expected_pool_sizes, strict=True):
        actual = _native_schedule(lib, block_dim, counts)
        expected = _scalar_schedule(counts)
        if actual != expected:
            raise RuntimeError(f"reused fiber schedule mismatch: {actual} != {expected}")
        pool_size = lib.wp_cpu_block_pool_size()
        if pool_size != expected_pool_size:
            raise RuntimeError(f"unexpected CPU block fiber pool size: {pool_size} != {expected_pool_size}")

    print("CPU block fiber pool reuse probe passed")


def _check_allocation_failure():
    lib = _setup_runtime()
    cpu = wp.get_device("cpu")
    previous = wp.config.enable_cpu_blocks
    wp.config.enable_cpu_blocks = True
    try:
        output = wp.zeros(3, dtype=wp.int32, device=cpu)
        message = "Warp failed to allocate a reusable CPU block fiber"

        lib.wp_cpu_test_fail_next_worker_allocation()
        try:
            wp.launch(_allocation_failure_kernel, dim=2, outputs=[output], block_dim=2, device=cpu)
        except RuntimeError as error:
            if message not in str(error):
                raise RuntimeError(f"unexpected direct-launch error: {error}") from error
        else:
            raise RuntimeError("worker allocation failure did not reach the direct CPU launch")

        command = wp.launch(
            _allocation_failure_kernel,
            dim=2,
            outputs=[output],
            block_dim=2,
            device=cpu,
            record_cmd=True,
        )
        lib.wp_cpu_test_fail_next_worker_allocation()
        try:
            command.launch()
        except RuntimeError as error:
            if message not in str(error):
                raise RuntimeError(f"unexpected recorded-launch error: {error}") from error
        else:
            raise RuntimeError("worker allocation failure did not reach the recorded CPU launch")

        # The error is catchable and consumed: retrying the same command grows
        # the pool normally and executes the kernel.
        command.launch()
        if output.numpy()[:2].tolist() != [2, 2]:
            raise RuntimeError("recorded CPU launch did not recover after the injected allocation failure")

        # APIC replay uses the same native launch bridge and must propagate the
        # failure rather than silently completing a partially executed graph.
        _allocation_failure_kernel.module.load(cpu, block_dim=3)
        output.zero_()
        with wp.ScopedCapture(device=cpu, apic=True, force_module_load=False) as capture:
            wp.launch(_allocation_failure_kernel, dim=3, outputs=[output], block_dim=3, device=cpu)

        lib.wp_cpu_test_fail_next_worker_allocation()
        try:
            wp.capture_launch(capture.graph)
        except RuntimeError as error:
            if message not in str(error):
                raise RuntimeError(f"unexpected APIC replay error: {error}") from error
        else:
            raise RuntimeError("worker allocation failure did not reach CPU APIC replay")

        wp.capture_launch(capture.graph)
        if output.numpy().tolist() != [3, 3, 3]:
            raise RuntimeError("CPU APIC replay did not recover after the injected allocation failure")

        # A saved graph loads its kernel from a copied object file. Confirm that
        # this independently loaded code also reports through the core runtime.
        saved_output = wp.zeros(4, dtype=wp.int32, device=cpu)
        _allocation_failure_kernel.module.load(cpu, block_dim=4)
        with wp.ScopedCapture(device=cpu, apic=True, force_module_load=False) as saved_capture:
            wp.launch(_allocation_failure_kernel, dim=4, outputs=[saved_output], block_dim=4, device=cpu)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cpu_block_allocation_failure")
            wp.capture_save(saved_capture.graph, path, outputs={"output": saved_output})
            loaded = wp.capture_load(path, device=cpu)

            lib.wp_cpu_test_fail_next_worker_allocation()
            try:
                wp.capture_launch(loaded)
            except RuntimeError as error:
                if message not in str(error):
                    raise RuntimeError(f"unexpected loaded APIC replay error: {error}") from error
            else:
                raise RuntimeError("worker allocation failure did not reach loaded CPU APIC replay")

            wp.capture_launch(loaded)
            loaded_output = wp.empty_like(saved_output)
            loaded.get_param("output", loaded_output)
            if loaded_output.numpy().tolist() != [4, 4, 4, 4]:
                raise RuntimeError("loaded CPU APIC replay did not recover after the injected allocation failure")
    finally:
        wp.config.enable_cpu_blocks = previous

    print("CPU block allocation failure probe passed")


class TestCpuBlockRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = _setup_runtime()

    def assertSchedule(self, block_dim, barrier_counts):
        expected = _scalar_schedule(barrier_counts)
        actual = _native_schedule(self.lib, block_dim, barrier_counts)
        self.assertEqual(actual, expected)
        self.assertEqual(_native_schedule(self.lib, block_dim, barrier_counts), actual)

    def test_exhaustive_small_schedules(self):
        for active_count in range(1, 6):
            for barrier_counts in itertools.product(range(4), repeat=active_count):
                with self.subTest(active_count=active_count, barrier_counts=barrier_counts):
                    self.assertSchedule(active_count, barrier_counts)

    def test_bitset_boundaries_and_partial_prefixes(self):
        for block_dim in (2, 31, 32, 63, 64, 65, 255, 256, 257, 511, 512, 1023, 1024):
            active_counts = {1, block_dim, max(1, block_dim - 1), min(block_dim, 65)}
            for active_count in sorted(active_counts):
                counts = [(lane * 7 + active_count) % 4 for lane in range(active_count)]
                with self.subTest(block_dim=block_dim, active_count=active_count):
                    self.assertSchedule(block_dim, counts)

    def test_many_generations(self):
        self.assertSchedule(4, [2000, 2000, 2000, 2000])

    def test_early_returns_and_sparse_survivors(self):
        cases = (
            [0, 4, 4, 4],
            [1, 0, 3, 0, 3],
            [2, 2, 0, 2, 0, 2],
            [0, 0, 0, 8],
            [8, 0, 0, 0],
            [3],
        )
        for counts in cases:
            with self.subTest(counts=counts):
                self.assertSchedule(len(counts), counts)

    def test_varied_schedules_through_1024_lanes(self):
        block_dims = (
            # Powers of two in the supported cooperative range.
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            # Primes spanning small through near-maximum blocks.
            3,
            31,
            127,
            509,
            1021,
            # Multiples of ten spanning the same range.
            10,
            30,
            100,
            510,
            1000,
        )
        for block_dim in block_dims:
            active_count = max(1, block_dim - block_dim // 4)
            counts = [(lane * 7 + block_dim) % 9 for lane in range(active_count)]
            with self.subTest(block_dim=block_dim, active_count=active_count):
                self.assertSchedule(block_dim, counts)

    def test_worker_pool_reuse_and_growth(self):
        result = subprocess.run(
            [sys.executable, __file__, "--pool-reuse-probe"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("CPU block fiber pool reuse probe passed", result.stdout)

    def test_worker_pool_allocation_failure_is_catchable(self):
        result = subprocess.run(
            [sys.executable, __file__, "--allocation-failure-probe"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("CPU block allocation failure probe passed", result.stdout)


if __name__ == "__main__":
    if "--pool-reuse-probe" in sys.argv:
        _check_pool_reuse()
    elif "--allocation-failure-probe" in sys.argv:
        _check_allocation_failure()
    else:
        unittest.main(verbosity=2, failfast=True)
