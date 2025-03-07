# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_DIM = 64
TILE_M = 16
TILE_N = 32
TILE_O = 8


@wp.kernel
def test_tile_view_kernel(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    # load whole source into local memory
    a = wp.tile_load(src, shape=(TILE_M, TILE_N))

    # copy the source array row by row
    for i in range(TILE_M):
        # create a view on original array and store
        row = a[i]
        wp.tile_store(dst[i], row)


def test_tile_view(test, device):
    rng = np.random.default_rng(42)

    a = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.zeros((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_view_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a.numpy())
    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones_like(a.numpy()))


@wp.kernel
def test_tile_assign_1d_kernel(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    # load whole source into local memory
    a = wp.tile_load(src, shape=(TILE_M, TILE_N))
    b = wp.tile_zeros(dtype=float, shape=(TILE_M, TILE_N))

    # copy the source array row by row
    for i in range(int(TILE_M)):
        # create views onto source and dest rows
        row_src = a[i]
        row_dst = b[i]

        # copy onto dest row
        wp.tile_assign(row_dst, row_src)

    wp.tile_store(dst, b)


def test_tile_assign_1d(test, device):
    rng = np.random.default_rng(42)

    a = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.zeros((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_assign_1d_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a.numpy())
    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones_like(a.numpy()))


@wp.kernel
def test_tile_assign_2d_kernel(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float)):
    # load whole source into local memory
    a = wp.tile_load(src, shape=(TILE_M, TILE_N, TILE_O))
    b = wp.tile_zeros(dtype=float, shape=(TILE_M, TILE_N, TILE_O))

    # copy the source array slice by slice
    for i in range(TILE_M):
        # create views onto source and dest slice
        row_src = a[i]
        row_dst = b[i]

        # copy onto dest slice
        wp.tile_assign(row_dst, row_src)

    wp.tile_store(dst, b)


def test_tile_assign_2d(test, device):
    rng = np.random.default_rng(42)

    a = wp.array(rng.random((TILE_M, TILE_N, TILE_O), dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.zeros((TILE_M, TILE_N, TILE_O), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_assign_2d_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a.numpy())
    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones_like(a.numpy()))


@wp.kernel
def test_tile_view_offset_kernel(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    # load whole source into local memory
    a = wp.tile_load(src, shape=(TILE_M, TILE_N))
    b = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)

    # copy the source array slice by slice
    for i in range(TILE_M // 4):
        # create views onto source and dest slice 4 rows at a time
        v = wp.tile_view(a, offset=(i * 4, 0), shape=(4, TILE_N))

        # copy onto dest slice
        wp.tile_assign(b, v, offset=(i * 4, 0))

    wp.tile_store(dst, b)


def test_tile_view_offset(test, device):
    rng = np.random.default_rng(42)

    a = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.zeros((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_view_offset_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a.numpy())
    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones_like(a.numpy()))


devices = get_test_devices()


class TestTileView(unittest.TestCase):
    pass


add_function_test(TestTileView, "test_tile_view", test_tile_view, devices=devices)
add_function_test(TestTileView, "test_tile_view_offset", test_tile_view_offset, devices=devices)
add_function_test(TestTileView, "test_tile_assign_1d", test_tile_assign_1d, devices=devices)
add_function_test(TestTileView, "test_tile_assign_2d", test_tile_assign_2d, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
