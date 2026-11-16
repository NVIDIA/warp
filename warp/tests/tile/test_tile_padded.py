# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise tile addressing when composite elements have inter-element padding."""

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices

TILE_SIZE = 3
BLOCK_DIM = 32
PADDING = -999.0
PERMUTATION = np.array([2, 0, 1], dtype=np.int32)
COMPONENT_SHAPES = {wp.vec3: (3,), wp.mat33: (3, 3)}


@wp.kernel
def copy_register(src: wp.array[Any], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="register")
    wp.tile_store(dst, t)


@wp.kernel
def copy_register_unchecked(src: wp.array[Any], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="register", bounds_check=False)
    wp.tile_store(dst, t, bounds_check=False)


@wp.kernel
def copy_shared(src: wp.array[Any], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="shared")
    wp.tile_store(dst, t)


@wp.kernel
def gather_register(src: wp.array[Any], indices: wp.array[int], dst: wp.array[Any]):
    index_tile = wp.tile_load(indices, shape=TILE_SIZE, storage="shared")
    t = wp.tile_load_indexed(src, indices=index_tile, shape=(TILE_SIZE,), storage="register")
    wp.tile_store(dst, t)


@wp.kernel
def gather_shared(src: wp.array[Any], indices: wp.array[int], dst: wp.array[Any]):
    index_tile = wp.tile_load(indices, shape=TILE_SIZE, storage="shared")
    t = wp.tile_load_indexed(src, indices=index_tile, shape=(TILE_SIZE,), storage="shared")
    wp.tile_store(dst, t)


@wp.kernel
def scatter_register(src: wp.array[Any], indices: wp.array[int], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="register")
    index_tile = wp.tile_load(indices, shape=TILE_SIZE, storage="shared")
    wp.tile_store_indexed(dst, indices=index_tile, t=t)


@wp.kernel
def scatter_shared(src: wp.array[Any], indices: wp.array[int], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="shared")
    index_tile = wp.tile_load(indices, shape=TILE_SIZE, storage="shared")
    wp.tile_store_indexed(dst, indices=index_tile, t=t)


@wp.kernel
def atomic_register(src: wp.array[Any], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="register")
    wp.tile_atomic_add(dst, t)


@wp.kernel
def atomic_shared(src: wp.array[Any], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="shared")
    wp.tile_atomic_add(dst, t)


@wp.kernel
def atomic_indexed_register(src: wp.array[Any], indices: wp.array[int], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="register")
    index_tile = wp.tile_load(indices, shape=TILE_SIZE, storage="shared")
    wp.tile_atomic_add_indexed(dst, indices=index_tile, t=t)


@wp.kernel
def atomic_indexed_shared(src: wp.array[Any], indices: wp.array[int], dst: wp.array[Any]):
    t = wp.tile_load(src, shape=TILE_SIZE, storage="shared")
    index_tile = wp.tile_load(indices, shape=TILE_SIZE, storage="shared")
    wp.tile_atomic_add_indexed(dst, indices=index_tile, t=t)


COPY_CASES = (
    ("register_load", copy_register, "load", "copy"),
    ("register_store", copy_register, "store", "copy"),
    ("unchecked_register_load", copy_register_unchecked, "load", "copy"),
    ("unchecked_register_store", copy_register_unchecked, "store", "copy"),
    ("shared_load", copy_shared, "load", "copy"),
    ("shared_store", copy_shared, "store", "copy"),
    ("indexed_register_load", gather_register, "load", "gather"),
    ("indexed_shared_load", gather_shared, "load", "gather"),
    ("indexed_register_store", scatter_register, "store", "scatter"),
    ("indexed_shared_store", scatter_shared, "store", "scatter"),
)
ATOMIC_CASES = (
    ("register", atomic_register, "copy"),
    ("shared", atomic_shared, "copy"),
    ("indexed_register", atomic_indexed_register, "scatter"),
    ("indexed_shared", atomic_indexed_shared, "scatter"),
)

for dtype in COMPONENT_SHAPES:
    for kernel in (
        copy_register,
        copy_register_unchecked,
        copy_shared,
        gather_register,
        gather_shared,
        scatter_register,
        scatter_shared,
        atomic_register,
        atomic_shared,
        atomic_indexed_register,
        atomic_indexed_shared,
    ):
        wp.overload(kernel, {"src": wp.array[dtype], "dst": wp.array[dtype]})


def make_view(values, dtype, stride_components, device):
    """Keep the owning scalar array alive and expose a strided composite view."""
    components = int(np.prod(COMPONENT_SHAPES[dtype]))
    rows = np.full((TILE_SIZE, stride_components), PADDING, dtype=np.float32)
    rows[:, :components] = values.reshape(TILE_SIZE, components)
    backing = wp.array(rows, dtype=wp.float32, device=device)
    view = wp.array(
        ptr=backing.ptr,
        dtype=dtype,
        shape=(TILE_SIZE,),
        strides=(backing.strides[0],),
        device=device,
        copy=False,
    )
    return view, backing, rows


def launch(kernel, src, dst, indices, device):
    inputs = [src]
    if kernel not in (copy_register, copy_register_unchecked, copy_shared, atomic_register, atomic_shared):
        inputs.append(indices)
    wp.launch_tiled(kernel, dim=1, inputs=inputs, outputs=[dst], block_dim=BLOCK_DIM, device=device)


def expected_copy(values, mode):
    if mode == "gather":
        return values[PERMUTATION]
    if mode == "scatter":
        expected = np.empty_like(values)
        expected[PERMUTATION] = values
        return expected
    return values


def test_tile_padded_copy(test, device):
    indices = wp.array(PERMUTATION, dtype=int, device=device)
    for dtype, component_shape in COMPONENT_SHAPES.items():
        components = int(np.prod(component_shape))
        values = np.arange(TILE_SIZE * components, dtype=np.float32).reshape((TILE_SIZE, *component_shape))
        for stride in (components, components + 1, components * 2):
            # The final case has padding but an exact element-sized stride multiple.
            for name, kernel, direction, mode in COPY_CASES:
                with test.subTest(dtype=dtype, stride=stride, operation=name):
                    src_stride = stride if direction == "load" else components
                    dst_stride = stride if direction == "store" else components
                    src, src_backing, _ = make_view(values, dtype, src_stride, device)
                    dst, dst_backing, expected_rows = make_view(np.zeros_like(values), dtype, dst_stride, device)
                    launch(kernel, src, dst, indices, device)
                    expected = expected_copy(values, mode)
                    expected_rows[:, :components] = expected.reshape(TILE_SIZE, components)
                    np.testing.assert_array_equal(dst.numpy(), expected)
                    # A correct store must preserve padding, not just its visible values.
                    np.testing.assert_array_equal(dst_backing.numpy(), expected_rows)
                    del src_backing


def test_tile_padded_atomic(test, device):
    indices = wp.array(PERMUTATION, dtype=int, device=device)
    for dtype, component_shape in COMPONENT_SHAPES.items():
        components = int(np.prod(component_shape))
        values = np.arange(TILE_SIZE * components, dtype=np.float32).reshape((TILE_SIZE, *component_shape))
        for name, kernel, mode in ATOMIC_CASES:
            with test.subTest(dtype=dtype, operation=name):
                src = wp.array(values, dtype=dtype, device=device)
                dst, dst_backing, expected_rows = make_view(np.ones_like(values), dtype, components + 1, device)
                launch(kernel, src, dst, indices, device)
                expected = expected_copy(values, mode) + 1.0
                expected_rows[:, :components] = expected.reshape(TILE_SIZE, components)
                np.testing.assert_array_equal(dst.numpy(), expected)
                np.testing.assert_array_equal(dst_backing.numpy(), expected_rows)


def test_tile_padded_backward(test, device):
    indices = wp.array(PERMUTATION, dtype=int, device=device)
    cases = [
        (name, kernel, mode, False)
        for name, kernel, direction, mode in COPY_CASES
        if direction == "load" or mode == "scatter"
    ]
    cases += [(name, kernel, mode, True) for name, kernel, mode in ATOMIC_CASES]
    for dtype, component_shape in COMPONENT_SHAPES.items():
        components = int(np.prod(component_shape))
        values = np.arange(TILE_SIZE * components, dtype=np.float32).reshape((TILE_SIZE, *component_shape))
        seeds = values + 1.0
        for name, kernel, mode, atomic in cases:
            with test.subTest(dtype=dtype, operation=name):
                src, src_backing, _ = make_view(values, dtype, components + 1, device)
                dst, dst_backing, _ = make_view(np.zeros_like(values), dtype, components + 1, device)
                src_grad, src_grad_backing, expected_src_rows = make_view(
                    np.zeros_like(values), dtype, components + 1, device
                )
                dst_grad, dst_grad_backing, expected_dst_rows = make_view(seeds, dtype, components + 1, device)
                src.grad = src_grad
                dst.grad = dst_grad
                with wp.Tape() as tape:
                    launch(kernel, src, dst, indices, device)
                tape.backward()

                if mode == "gather":
                    expected_grad = np.empty_like(seeds)
                    expected_grad[PERMUTATION] = seeds
                elif mode == "scatter":
                    expected_grad = seeds[PERMUTATION]
                else:
                    expected_grad = seeds
                expected_src_rows[:, :components] = expected_grad.reshape(TILE_SIZE, components)
                # Stores consume destination adjoints; atomic addition preserves them.
                if not atomic:
                    expected_dst_rows[:, :components] = 0.0
                np.testing.assert_array_equal(src_grad_backing.numpy(), expected_src_rows)
                np.testing.assert_array_equal(dst_grad_backing.numpy(), expected_dst_rows)
                del src_backing, dst_backing


class TestTilePadded(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestTilePadded, "test_tile_padded_copy", test_tile_padded_copy, devices=devices)
add_function_test(TestTilePadded, "test_tile_padded_atomic", test_tile_padded_atomic, devices=devices)
add_function_test(TestTilePadded, "test_tile_padded_backward", test_tile_padded_backward, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
