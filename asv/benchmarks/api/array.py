# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
from typing import NamedTuple

import numpy as np

import warp as wp

from ..benchmarks_utils import setup_once

N = 8192


class ArrayEmpty:
    """Benchmark wp.empty()."""

    repeat = 1000  # Number of samples to run
    number = 1  # Number of timed calls per sample

    def setup(self):
        wp.init()
        self.alloc = None
        gc.disable()

    def teardown(self):
        gc.enable()
        self.alloc = None
        wp.synchronize_device("cuda:0")

    def time_empty(self):
        self.alloc = wp.empty(N, dtype=float, device="cuda:0")


class ArrayZeros:
    """Benchmark wp.zeros()."""

    repeat = 1000
    number = 1

    def setup(self):
        wp.init()
        self.alloc = None
        gc.disable()

    def teardown(self):
        gc.enable()
        self.alloc = None
        wp.synchronize_device("cuda:0")

    def time_zeros(self):
        self.alloc = wp.zeros(N, dtype=float, device="cuda:0")


class ArrayFree:
    """Benchmark array free including GPU work."""

    repeat = 100
    number = 1

    def setup(self):
        wp.init()
        self.test_array = wp.empty(N, dtype=float, device="cuda:0")
        self.allocs = [None] * 10
        for i in range(len(self.allocs)):
            self.allocs[i] = wp.empty(N, dtype=float, device="cuda:0")
        wp.synchronize_device("cuda:0")

    def time_ten_free_sync(self):
        for i in range(len(self.allocs)):
            self.allocs[i] = None
        wp.synchronize_device("cuda:0")


class _CustomSequence:
    """A sequence that is neither a list nor a tuple."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _Point3(NamedTuple):
    """A point record, which NumPy reads like a tuple of 3 floats."""

    x: float
    y: float
    z: float


def _make_sequence(kind, count):
    """Return a Python sequence of ``count`` values of the given kind for ``wp.array()``."""
    if kind == "floats":
        return [float(i) for i in range(count)], wp.float32
    if kind == "tuples":
        return [(float(i), 1.0, 2.0) for i in range(count)], wp.vec3
    if kind == "vec3f":
        return [wp.vec3f(i, 1.0, 2.0) for i in range(count)], wp.vec3f
    if kind == "vec3h":
        return [wp.vec3h(i % 1000, 1.0, 2.0) for i in range(count)], wp.vec3h
    if kind == "mat33f":
        return [wp.mat33f(*range(i, i + 9)) for i in range(count)], wp.mat33f
    if kind == "nested vec3f":
        return [[wp.vec3f(i, 1.0, 2.0)] * 3 for i in range(count // 3)], wp.vec3f
    if kind == "named tuples":
        return [_Point3(float(i), 1.0, 2.0) for i in range(count)], wp.vec3f
    if kind == "NumPy 3x3 arrays":
        return [np.full((3, 3), i, dtype=np.float32) for i in range(count)], wp.mat33f
    if kind == "vec3f custom sequence":
        return _CustomSequence([wp.vec3f(i, 1.0, 2.0) for i in range(count)]), wp.vec3f
    raise ValueError(kind)


class ArrayFromList:
    """Benchmark wp.array() from long Python lists."""

    params = (
        "floats",
        "tuples",
        "named tuples",
        "vec3f",
        "vec3h",
        "mat33f",
        "NumPy 3x3 arrays",
        "nested vec3f",
        "vec3f custom sequence",
    )
    param_names = ("kind",)
    repeat = 15  # Samples have low variance at 1-60 ms each.
    number = 1

    @setup_once
    def setup(self, kind):
        wp.init()
        self.data, self.dtype = _make_sequence(kind, 100_000)
        self.alloc = None

    def time_array_from_list(self, kind):
        self.alloc = wp.array(self.data, dtype=self.dtype, device="cpu")


class ArrayFromShortList:
    """Benchmark the fixed cost of wp.array() from short Python lists."""

    params = ("3 floats", "4 tuples", "1 vec3f", "4 vec3f", "2x2 nested vec3f", "4x3 NumPy array")
    param_names = ("kind",)
    repeat = 31
    number = 128

    @setup_once
    def setup(self, kind):
        wp.init()
        if kind == "3 floats":
            self.data, self.dtype = [1.0, 2.0, 3.0], wp.float32
        elif kind == "4 tuples":
            self.data, self.dtype = [(1.0, 2.0, 3.0)] * 4, wp.vec3
        elif kind == "1 vec3f":
            self.data, self.dtype = [wp.vec3f(1.0, 2.0, 3.0)], wp.vec3f
        elif kind == "4 vec3f":
            self.data, self.dtype = [wp.vec3f(1.0, 2.0, 3.0)] * 4, wp.vec3f
        elif kind == "2x2 nested vec3f":
            self.data, self.dtype = [[wp.vec3f(1.0, 2.0, 3.0)] * 2] * 2, wp.vec3f
        else:
            self.data, self.dtype = np.ones((4, 3), dtype=np.float32), wp.vec3f
        self.alloc = None

    def time_array_from_short_list(self, kind):
        self.alloc = wp.array(self.data, dtype=self.dtype, device="cpu")


class ArrayFromListMemory:
    """Benchmark the peak memory of wp.array() from a long Python list of vectors."""

    def setup(self):
        wp.init()
        self.data, self.dtype = _make_sequence("vec3f", 1_000_000)

    def peakmem_array_from_vec3f_list(self):
        wp.array(self.data, dtype=self.dtype, device="cpu")
