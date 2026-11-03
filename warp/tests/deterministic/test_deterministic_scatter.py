# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import sys
import unittest

import numpy as np

import warp as wp
from warp.tests.deterministic.common import (
    DeterministicTestBase,
    _bfloat16_numpy_bits,
    _bfloat16_numpy_to_float32,
    _reference_scatter_add_float32,
    all_devices,
    assert_equal_repeated,
    bfloat16_cuda_devices,
    cuda_devices,
)
from warp.tests.unittest_utils import add_function_test

_THIS_MODULE = sys.modules[__name__]


def _set_test_module_options(options):
    wp.set_module_options(options, module=_THIS_MODULE)


def _get_test_module_options():
    return wp.get_module_options(module=_THIS_MODULE)


def _make_float_scatter_inputs(n, out_size, seed, device):
    rng = np.random.default_rng(seed)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    return data_np, indices_np, data, indices


def _launch_scatter_once(kernel, n, data, indices, out_size, device):
    output = wp.zeros(out_size, dtype=wp.float32, device=device)
    wp.launch(kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
    return output.numpy().copy()


@wp.kernel
def scatter_add_kernel(
    data: wp.array[wp.float32],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Each thread atomically adds to output[dest_indices[tid]]."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, data[tid])


@wp.kernel
def augassign_add_kernel(
    data: wp.array[wp.float32],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Implement ``scatter_add_kernel`` with augmented-assignment syntax."""
    tid = wp.tid()
    idx = dest_indices[tid]
    output[idx] += data[tid]


@wp.kernel
def multi_array_atomic_kernel(
    data: wp.array[wp.float32],
    dest_indices: wp.array[wp.int32],
    out_a: wp.array[wp.float32],
    out_b: wp.array[wp.float32],
    out_c: wp.array[wp.float32],
):
    """Atomic add to three different output arrays from the same kernel."""
    tid = wp.tid()
    idx = dest_indices[tid]
    val = data[tid]
    wp.atomic_add(out_a, idx, val)
    wp.atomic_add(out_b, idx, val * 2.0)
    wp.atomic_add(out_c, idx, val * 3.0)


@wp.kernel
def atomic_sub_kernel(
    data: wp.array[wp.float32],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Atomic sub test."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_sub(output, idx, data[tid])


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("mass",),
)
def deferred_tree_reduce_kernel(parent: wp.array[wp.int32], level: wp.array[wp.int32], mass: wp.array[wp.float32]):
    """Accumulate one level of a tree using safe pre-atomic reads."""
    body = level[wp.tid()]
    wp.atomic_add(mass, parent[body], mass[body])


@wp.func
def deferred_tree_reduce_helper(
    parent: wp.array[wp.int32], level: wp.array[wp.int32], mass: wp.array[wp.float32], tid: int
):
    """Tree-level reduction helper whose argument is remapped by its caller."""
    body = level[tid]
    wp.atomic_add(mass, parent[body], mass[body])


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("mass",),
)
def deferred_tree_reduce_helper_kernel(
    parent: wp.array[wp.int32], level: wp.array[wp.int32], mass: wp.array[wp.float32]
):
    """Call the tree-level helper with a declared deferred target."""
    deferred_tree_reduce_helper(parent, level, mass, wp.tid())


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def declared_deferred_read_after_kernel(out: wp.array[wp.float32], observed: wp.array[wp.float32]):
    """Unsafe read-after-atomic pattern that must remain rejected."""
    wp.atomic_add(out, 0, 1.0)
    observed[0] = out[0]


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def declared_deferred_store_kernel(out: wp.array[wp.float32]):
    """Direct stores remain unsafe even when pre-atomic reads are declared."""
    out[1] = 5.0
    wp.atomic_add(out, 0, 1.0)


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def declared_deferred_dynamic_for_read_kernel(count: int, out: wp.array[wp.float32]):
    """Exercise a repeated read-after-update hazard in a dynamic loop."""
    for _ in range(count):
        value = out[0]
        wp.atomic_add(out, 0, value)


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def declared_deferred_dynamic_while_read_kernel(count: int, out: wp.array[wp.float32]):
    """Exercise the repeated read-after-update hazard in a while loop."""
    i = int(0)
    while i < count:
        value = out[0]
        wp.atomic_add(out, 0, value)
        i = i + 1


@wp.func
def deferred_repeated_read_helper(out: wp.array[wp.float32]):
    """Read a value before atomically accumulating it into the same array."""
    value = out[0]
    wp.atomic_add(out, 0, value)


@wp.func
def deferred_atomic_then_read_helper(
    src: wp.array[wp.float32], dst: wp.array[wp.float32], observed: wp.array[wp.float32]
):
    """Read through a separate parameter after a deferred atomic update."""
    wp.atomic_add(dst, 0, 1.0)
    observed[0] = src[0]


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def declared_deferred_helper_read_after_kernel(out: wp.array[wp.float32], observed: wp.array[wp.float32]):
    """Map a helper's read-after-atomic parameters to the same target."""
    deferred_atomic_then_read_helper(out, out, observed)


@wp.func
def deferred_nested_atomic_then_read_helper(
    src: wp.array[wp.float32], dst: wp.array[wp.float32], observed: wp.array[wp.float32]
):
    """Forward an ordered deferred access pattern through another helper."""
    deferred_atomic_then_read_helper(src, dst, observed)


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def declared_deferred_nested_helper_read_after_kernel(out: wp.array[wp.float32], observed: wp.array[wp.float32]):
    """Map nested helper parameters to the same declared target."""
    deferred_nested_atomic_then_read_helper(out, out, observed)


@wp.func
def deferred_nested_scatter_helper(out: wp.array[wp.float32]):
    """Emit one deterministic accumulation update."""
    wp.atomic_add(out, 0, 1.0)


@wp.func
def deferred_direct_and_nested_scatter_helper(out: wp.array[wp.float32]):
    """Emit one direct and one nested deterministic accumulation update."""
    wp.atomic_add(out, 0, 1.0)
    deferred_nested_scatter_helper(out)


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
def nested_helper_scatter_kernel(out: wp.array[wp.float32]):
    """Exercise buffer sizing for updates imported through nested helpers."""
    deferred_direct_and_nested_scatter_helper(out)


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def declared_deferred_dynamic_helper_read_kernel(count: int, out: wp.array[wp.float32]):
    """Exercise the repeated-read hazard through a helper called in a loop."""
    for _ in range(count):
        deferred_repeated_read_helper(out)


@wp.func
def deferred_separate_loop_helper(src: wp.array[wp.float32], dst: wp.array[wp.float32], count: int):
    """Read one array while reducing into another from a dynamic helper loop."""
    for _ in range(count):
        value = src[0]
        wp.atomic_add(dst, 0, value)


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN, "deterministic_max_records": 2},
    deterministic_deferred_atomics=("a", "b"),
)
def declared_deferred_separate_helper_calls_kernel(
    a: wp.array[wp.float32], b: wp.array[wp.float32], c: wp.array[wp.float32], count: int
):
    """Use separate helper calls whose local loops must not share a scope."""
    deferred_separate_loop_helper(a, b, count)
    deferred_separate_loop_helper(c, a, count)


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def declared_deferred_tile_load_kernel(out: wp.array[wp.float32], observed: wp.array[wp.float32]):
    """Read a declared target through a tile before accumulating into it."""
    wp.tile_store(observed, wp.tile_load(out, shape=32))
    wp.atomic_add(out, 0, 1.0)


@wp.kernel(
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
    deterministic_deferred_atomics=("out",),
)
def deferred_tree_reduce_alias_kernel(
    parent: wp.array[wp.int32], level: wp.array[wp.int32], values: wp.array[wp.float32], out: wp.array[wp.float32]
):
    """Read tree values through an input that may alias the deferred output."""
    body = level[wp.tid()]
    wp.atomic_add(out, parent[body], values[body])


@wp.kernel
def atomic_add_2d_kernel(
    data: wp.array[wp.float32],
    row_indices: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    output: wp.array2d[wp.float32],
):
    """Atomic add to a 2D array."""
    tid = wp.tid()
    r = row_indices[tid]
    c = col_indices[tid]
    wp.atomic_add(output, r, c, data[tid])


@wp.kernel
def negative_index_atomic_add_kernel(output: wp.array[wp.float32]):
    """Atomic add using a negative 1D index."""
    wp.atomic_add(output, -1, 1.0)


@wp.kernel
def negative_index_atomic_add_2d_kernel(output: wp.array2d[wp.float32]):
    """Atomic add using negative 2D indices."""
    wp.atomic_add(output, -1, -1, 1.0)


@wp.kernel
def sliced_2d_atomic_add_kernel(
    data: wp.array[wp.float32],
    row_indices: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    output: wp.array2d[wp.float32],
):
    """Atomic add through a sliced ``output[row]`` view."""
    tid = wp.tid()
    row = row_indices[tid]
    col = col_indices[tid]
    wp.atomic_add(output[row], col, data[tid])


@wp.kernel
def sliced_3d_atomic_add_kernel(
    data: wp.array[wp.float32],
    row_indices: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    depth_indices: wp.array[wp.int32],
    output: wp.array3d[wp.float32],
):
    """Atomic add through a sliced ``output[row, col]`` view."""
    tid = wp.tid()
    row = row_indices[tid]
    col = col_indices[tid]
    depth = depth_indices[tid]
    wp.atomic_add(output[row, col], depth, data[tid])


@wp.kernel
def atomic_half_kernel(
    data: wp.array[wp.float16],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.float16],
):
    """Atomic add with float16."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, data[tid])


@wp.kernel
def atomic_bfloat16_kernel(
    data: wp.array[wp.bfloat16],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.bfloat16],
):
    """Atomic add with bfloat16."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, data[tid])


@wp.kernel
def atomic_bfloat16_minmax_kernel(
    data: wp.array[wp.bfloat16],
    dest_indices: wp.array[wp.int32],
    output_min: wp.array[wp.bfloat16],
    output_max: wp.array[wp.bfloat16],
):
    """Atomic min/max with bfloat16."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_min(output_min, idx, data[tid])
    wp.atomic_max(output_max, idx, data[tid])


@wp.kernel
def atomic_double_kernel(
    data: wp.array[wp.float64],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.float64],
):
    """Atomic add with float64."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, data[tid])


@wp.kernel
def vec3_scatter_add_kernel(
    data: wp.array[wp.vec3],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.vec3],
):
    """Atomic add with ``wp.vec3`` values."""
    tid = wp.tid()
    wp.atomic_add(output, dest_indices[tid], data[tid])


@wp.kernel
def vec3_atomic_minmax_kernel(
    points: wp.array[wp.vec3],
    out_min: wp.array[wp.vec3],
    out_max: wp.array[wp.vec3],
):
    """Component-wise deterministic min/max for bounding-box style reductions."""
    tid = wp.tid()
    p = points[tid]
    wp.atomic_min(out_min, 0, p)
    wp.atomic_max(out_max, 0, p)


@wp.kernel
def mat33_scatter_add_kernel(
    data: wp.array[wp.mat33],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.mat33],
):
    """Atomic add with ``wp.mat33`` values."""
    tid = wp.tid()
    wp.atomic_add(output, dest_indices[tid], data[tid])


@wp.func
def _det_closure_transform_a(x: wp.float32) -> wp.float32:
    return x + wp.float32(1.0)


@wp.func
def _det_closure_transform_b(x: wp.float32) -> wp.float32:
    return x + wp.float32(2.0)


@wp.func
def _det_func_scatter_add_leaf(arr: wp.array[wp.float32], idx: int, value: wp.float32):
    wp.atomic_add(arr, idx, value)


@wp.func
def _det_func_scatter_add_wrapper(dst: wp.array[wp.float32], idx: int, value: wp.float32):
    _det_func_scatter_add_leaf(dst, idx, value)


@wp.struct
class _DetNameCollisionStruct:
    b: wp.array[wp.float32]


@wp.func
def _det_increment_array(output: wp.array[wp.float32], index: int):
    output[index] = output[index] + 1.0


def _make_deterministic_closure_kernel(transform_func):
    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def _deterministic_closure_kernel(
        data: wp.array[wp.float32],
        output: wp.array[wp.float32],
    ):
        tid = wp.tid()
        wp.atomic_add(output, tid % 8, transform_func(data[tid]))

    return _deterministic_closure_kernel


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
def helper_name_collision_kernel(
    a_b: wp.array[wp.float32],
    a: _DetNameCollisionStruct,
    data: wp.array[wp.float32],
):
    tid = wp.tid()
    wp.atomic_add(a_b, 0, data[tid])
    wp.atomic_add(a.b, 0, data[tid] * wp.float32(2.0))


@wp.kernel
def func_scatter_add_kernel(
    data: wp.array[wp.float32],
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    tid = wp.tid()
    _det_func_scatter_add_leaf(output, dest_indices[tid], data[tid])


@wp.kernel
def nested_func_scatter_add_kernel(
    data: wp.array[wp.float32],
    dest_indices: wp.array[wp.int32],
    accum: wp.array[wp.float32],
):
    tid = wp.tid()
    _det_func_scatter_add_wrapper(accum, dest_indices[tid], data[tid])


@wp.kernel
def triple_scatter_add_kernel(
    data: wp.array[wp.float32],
    output: wp.array[wp.float32],
):
    """Emit three deterministic scatter records per thread to the same target."""
    tid = wp.tid()
    val = data[tid]
    wp.atomic_add(output, 0, val)
    wp.atomic_add(output, 0, val * 2.0)
    wp.atomic_add(output, 0, val * 3.0)


@wp.kernel(
    module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN, "deterministic_max_records": 4}
)
def loop_scatter_add_kernel(
    data: wp.array[wp.float32],
    counts: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Emit a data-dependent number of scatter records to the same target."""
    tid = wp.tid()
    val = data[tid]
    count = counts[tid]
    for _ in range(count):
        wp.atomic_add(output, 0, val)


@wp.kernel(
    module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN, "deterministic_max_records": 1}
)
def underprovisioned_loop_scatter_kernel(
    data: wp.array[wp.float32],
    counts: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    tid = wp.tid()
    count = counts[tid]
    for _ in range(count):
        wp.atomic_add(output, 0, data[tid])


def test_scatter_add_reproducibility(test, device):
    """Verify that float atomic_add produces bit-exact identical results across runs."""
    n = 4096
    out_size = 64
    _, _, data, indices = _make_float_scatter_inputs(n, out_size, 42, device)

    assert_equal_repeated(
        lambda: _launch_scatter_once(scatter_add_kernel, n, data, indices, out_size, device),
        err_msg="deterministic mode should be bit-exact",
    )


def test_gpu_to_gpu_matches_canonical_float32_reference(test, device):
    """Verify ``gpu_to_gpu`` is reproducible and matches the canonical order."""
    data_np = np.array(
        [
            1.0e20,
            1.0,
            -1.0e20,
            3.5,
            -2.25,
            2.0**-20,
            1.0e10,
            -1.0e10,
            7.0,
            -7.0,
            9.0,
            1.0e-7,
        ],
        dtype=np.float32,
    )
    indices_np = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2], dtype=np.int32)
    out_size = 3

    expected = _reference_scatter_add_float32(data_np, indices_np, out_size)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    old_det = _get_test_module_options()["deterministic"]
    try:
        _set_test_module_options({"deterministic": wp.DeterministicMode.GPU_TO_GPU})
        result = assert_equal_repeated(
            lambda: _launch_scatter_once(scatter_add_kernel, data_np.shape[0], data, indices, out_size, device)
        )
    finally:
        _set_test_module_options({"deterministic": old_det})

    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def test_augassign_add_reproducibility(test, device):
    """Verify += syntax (desugars to atomic_add) is also deterministic."""
    n = 2048
    out_size = 32
    _, _, data, indices = _make_float_scatter_inputs(n, out_size, 123, device)

    assert_equal_repeated(lambda: _launch_scatter_once(augassign_add_kernel, n, data, indices, out_size, device))


def test_scatter_add_correctness(test, device):
    """Compare deterministic GPU results against CPU sequential execution."""
    n = 2048
    out_size = 32
    rng = np.random.default_rng(99)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    # CPU sequential reference (guaranteed deterministic).
    expected = np.zeros(out_size, dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(out_size, dtype=wp.float32, device=device)

    wp.launch(
        scatter_add_kernel,
        dim=n,
        inputs=[data, indices],
        outputs=[output],
        device=device,
    )

    result = output.numpy()
    # Deterministic sum order may differ from Python loop order, so exact
    # match is not guaranteed.  Check within reasonable tolerance.
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


def test_multi_array_atomic(test, device):
    """Verify deterministic mode works with multiple target arrays."""
    n = 1024
    out_size = 16
    _, _, data, indices = _make_float_scatter_inputs(n, out_size, 77, device)

    def launch_once():
        out_a = wp.zeros(out_size, dtype=wp.float32, device=device)
        out_b = wp.zeros(out_size, dtype=wp.float32, device=device)
        out_c = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            multi_array_atomic_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[out_a, out_b, out_c],
            device=device,
        )
        return out_a.numpy().copy(), out_b.numpy().copy(), out_c.numpy().copy()

    assert_equal_repeated(launch_once)


def test_atomic_sub_deterministic(test, device):
    """Verify atomic_sub is deterministic."""
    n = 2048
    out_size = 32
    _, _, data, indices = _make_float_scatter_inputs(n, out_size, 55, device)

    assert_equal_repeated(lambda: _launch_scatter_once(atomic_sub_kernel, n, data, indices, out_size, device))


def test_atomic_add_2d(test, device):
    """Verify deterministic mode with 2D array indexing."""
    n = 1024
    rows, cols = 8, 8
    rng = np.random.default_rng(88)

    data_np = rng.random(n, dtype=np.float32)
    row_np = rng.integers(0, rows, size=n, dtype=np.int32)
    col_np = rng.integers(0, cols, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    row_idx = wp.array(row_np, dtype=wp.int32, device=device)
    col_idx = wp.array(col_np, dtype=wp.int32, device=device)

    def launch_once():
        output = wp.zeros(shape=(rows, cols), dtype=wp.float32, device=device)
        wp.launch(
            atomic_add_2d_kernel,
            dim=n,
            inputs=[data, row_idx, col_idx],
            outputs=[output],
            device=device,
        )
        return output.numpy().copy()

    assert_equal_repeated(launch_once)


def test_atomic_add_negative_indices(test, device):
    """Verify deterministic flat indices match Warp negative-index semantics."""
    output = wp.zeros(4, dtype=wp.float32, device=device)
    wp.launch(negative_index_atomic_add_kernel, dim=1, outputs=[output], device=device)
    np.testing.assert_array_equal(output.numpy(), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    output_2d = wp.zeros((3, 4), dtype=wp.float32, device=device)
    wp.launch(negative_index_atomic_add_2d_kernel, dim=1, outputs=[output_2d], device=device)
    expected = np.zeros((3, 4), dtype=np.float32)
    expected[-1, -1] = 1.0
    np.testing.assert_array_equal(output_2d.numpy(), expected)


def test_sliced_2d_array_atomic_add(test, device):
    """Verify deterministic atomics through a sliced ``arr[row]`` view."""
    n = 2048
    rows, cols = 16, 16
    rng = np.random.default_rng(101)

    data_np = rng.random(n, dtype=np.float32)
    row_np = rng.integers(0, rows, size=n, dtype=np.int32)
    col_np = rng.integers(0, cols, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    row_idx = wp.array(row_np, dtype=wp.int32, device=device)
    col_idx = wp.array(col_np, dtype=wp.int32, device=device)

    expected = np.zeros((rows, cols), dtype=np.float32)
    for i in range(n):
        expected[row_np[i], col_np[i]] = np.float32(expected[row_np[i], col_np[i]] + data_np[i])

    def launch_once():
        output = wp.zeros(shape=(rows, cols), dtype=wp.float32, device=device)
        wp.launch(
            sliced_2d_atomic_add_kernel,
            dim=n,
            inputs=[data, row_idx, col_idx],
            outputs=[output],
            device=device,
        )
        return output.numpy().copy()

    np.testing.assert_allclose(assert_equal_repeated(launch_once), expected, rtol=1e-5, atol=1e-5)


def test_sliced_3d_array_atomic_add(test, device):
    """Verify deterministic atomics through a sliced ``arr[row, col]`` view."""
    n = 2048
    rows, cols, depth = 8, 8, 8
    rng = np.random.default_rng(102)

    data_np = rng.random(n, dtype=np.float32)
    row_np = rng.integers(0, rows, size=n, dtype=np.int32)
    col_np = rng.integers(0, cols, size=n, dtype=np.int32)
    depth_np = rng.integers(0, depth, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    row_idx = wp.array(row_np, dtype=wp.int32, device=device)
    col_idx = wp.array(col_np, dtype=wp.int32, device=device)
    depth_idx = wp.array(depth_np, dtype=wp.int32, device=device)

    expected = np.zeros((rows, cols, depth), dtype=np.float32)
    for i in range(n):
        expected[row_np[i], col_np[i], depth_np[i]] = np.float32(
            expected[row_np[i], col_np[i], depth_np[i]] + data_np[i]
        )

    def launch_once():
        output = wp.zeros(shape=(rows, cols, depth), dtype=wp.float32, device=device)
        wp.launch(
            sliced_3d_atomic_add_kernel,
            dim=n,
            inputs=[data, row_idx, col_idx, depth_idx],
            outputs=[output],
            device=device,
        )
        return output.numpy().copy()

    np.testing.assert_allclose(assert_equal_repeated(launch_once), expected, rtol=1e-5, atol=1e-5)


def test_strided_1d_view_atomic_add(test, device):
    """Preserve odd slots during atomic addition through ``base[::2]``."""
    n = 1024
    base_size = 128
    view_size = base_size // 2
    rng = np.random.default_rng(303)

    data_np = rng.random(n, dtype=np.float32)
    dest_np = rng.integers(0, view_size, size=n, dtype=np.int32)

    expected_view = np.zeros(view_size, dtype=np.float32)
    for i in range(n):
        expected_view[dest_np[i]] = np.float32(expected_view[dest_np[i]] + data_np[i])

    sentinel = np.float32(-7777.0)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    dest = wp.array(dest_np, dtype=wp.int32, device=device)

    def launch_once():
        base = wp.full(shape=(base_size,), value=sentinel, dtype=wp.float32, device=device)
        view = base[::2]
        test.assertFalse(view.is_contiguous)
        wp.launch(scatter_add_kernel, dim=n, inputs=[data, dest], outputs=[view], device=device)
        return base.numpy().copy()

    result = assert_equal_repeated(launch_once)
    np.testing.assert_allclose(result[0::2], sentinel + expected_view, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(result[1::2], np.full(view_size, sentinel, dtype=np.float32))

    base = wp.zeros(base_size, dtype=wp.float32, device=device)
    reversed_view = wp.array(
        ptr=base.ptr + (view_size - 1) * base.strides[0],
        shape=(view_size,),
        strides=(-base.strides[0],),
        dtype=base.dtype,
        device=device,
    )
    with test.assertRaisesRegex(wp.WarpDeterminismError, "negative-stride"):
        wp.launch(scatter_add_kernel, dim=n, inputs=[data, dest], outputs=[reversed_view], device=device)


def test_zero_stride_view_atomic_add(test, device):
    """Verify that atomic add through a zero-stride view reduces to the one physical slot."""
    n = 1024
    logical_size = 8
    rng = np.random.default_rng(306)

    data_np = rng.random(n, dtype=np.float32)
    dest_np = rng.integers(0, logical_size, size=n, dtype=np.int32)

    sentinel = np.float32(-11.0)
    expected = sentinel
    for value in data_np:
        expected = np.float32(expected + value)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    dest = wp.array(dest_np, dtype=wp.int32, device=device)

    def launch_once():
        base = wp.full(shape=(1,), value=sentinel, dtype=wp.float32, device=device)
        view = wp.array(
            ptr=base.ptr,
            dtype=wp.float32,
            shape=(logical_size,),
            strides=(0,),
            capacity=4,
            device=device,
        )
        test.assertFalse(view.is_contiguous)
        wp.launch(scatter_add_kernel, dim=n, inputs=[data, dest], outputs=[view], device=device)
        return base.numpy().copy()

    np.testing.assert_allclose(
        assert_equal_repeated(launch_once), np.array([expected], dtype=np.float32), rtol=1e-5, atol=1e-5
    )


def test_column_slice_atomic_add(test, device):
    """Preserve other columns during atomic addition through ``base[:, col]``."""
    n = 1024
    rows, cols = 64, 4
    target_col = 2
    rng = np.random.default_rng(304)

    data_np = rng.random(n, dtype=np.float32)
    dest_np = rng.integers(0, rows, size=n, dtype=np.int32)

    expected_col = np.zeros(rows, dtype=np.float32)
    for i in range(n):
        expected_col[dest_np[i]] = np.float32(expected_col[dest_np[i]] + data_np[i])

    sentinel = np.float32(-3333.0)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    dest = wp.array(dest_np, dtype=wp.int32, device=device)

    def launch_once():
        base = wp.full(shape=(rows, cols), value=sentinel, dtype=wp.float32, device=device)
        view = base[:, target_col]
        test.assertFalse(view.is_contiguous)
        wp.launch(scatter_add_kernel, dim=n, inputs=[data, dest], outputs=[view], device=device)
        return base.numpy().copy()

    result = assert_equal_repeated(launch_once)
    np.testing.assert_allclose(result[:, target_col], sentinel + expected_col, rtol=1e-5, atol=1e-5)
    other_cols = np.delete(result, target_col, axis=1)
    np.testing.assert_array_equal(other_cols, np.full(other_cols.shape, sentinel, dtype=np.float32))


def test_transposed_2d_atomic_add(test, device):
    """Map transposed-view atomic additions to the corresponding base elements."""
    n = 2048
    base_rows, base_cols = 16, 8
    view_rows, view_cols = base_cols, base_rows
    rng = np.random.default_rng(305)

    data_np = rng.random(n, dtype=np.float32)
    row_np = rng.integers(0, view_rows, size=n, dtype=np.int32)
    col_np = rng.integers(0, view_cols, size=n, dtype=np.int32)

    expected_view = np.zeros((view_rows, view_cols), dtype=np.float32)
    for i in range(n):
        expected_view[row_np[i], col_np[i]] = np.float32(expected_view[row_np[i], col_np[i]] + data_np[i])

    data = wp.array(data_np, dtype=wp.float32, device=device)
    row_idx = wp.array(row_np, dtype=wp.int32, device=device)
    col_idx = wp.array(col_np, dtype=wp.int32, device=device)

    def launch_once():
        base = wp.zeros(shape=(base_rows, base_cols), dtype=wp.float32, device=device)
        view = base.transpose([1, 0])
        test.assertFalse(view.is_contiguous)
        wp.launch(
            atomic_add_2d_kernel,
            dim=n,
            inputs=[data, row_idx, col_idx],
            outputs=[view],
            device=device,
        )
        return base.numpy().copy()

    np.testing.assert_allclose(assert_equal_repeated(launch_once), expected_view.T, rtol=1e-5, atol=1e-5)


def test_atomic_half_deterministic(test, device):
    """Verify deterministic mode with float16 atomics."""
    n = 1024
    out_size = 16
    rng = np.random.default_rng(78)

    data_np = rng.random(n, dtype=np.float32).astype(np.float16)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float16, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    def launch_once():
        output = wp.zeros(out_size, dtype=wp.float16, device=device)
        wp.launch(
            atomic_half_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        return output.numpy().copy()

    assert_equal_repeated(launch_once)


def test_atomic_bfloat16_deterministic(test, device):
    """Verify deterministic mode with bfloat16 atomics."""
    if device.arch < 80:
        test.skipTest("bfloat16 atomics require CUDA architecture >= 80")

    n = 64
    out_size = 8
    rng = np.random.default_rng(91)

    # Small integers are exactly representable in bfloat16, so correctness does
    # not depend on emulating bfloat16 rounding in the host reference.
    data_np = rng.integers(1, 5, size=n).astype(np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    expected_add = np.zeros(out_size, dtype=np.float32)
    expected_min = np.full(out_size, 16.0, dtype=np.float32)
    expected_max = np.zeros(out_size, dtype=np.float32)
    for value, index in zip(data_np, indices_np, strict=True):
        expected_add[index] += value
        expected_min[index] = min(expected_min[index], value)
        expected_max[index] = max(expected_max[index], value)

    data = wp.array(data_np, dtype=wp.bfloat16, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    def launch_add_once():
        output = wp.zeros(out_size, dtype=wp.bfloat16, device=device)
        wp.launch(
            atomic_bfloat16_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        values = output.numpy()
        return _bfloat16_numpy_bits(values), _bfloat16_numpy_to_float32(values)

    _, add_values = assert_equal_repeated(launch_add_once)
    np.testing.assert_allclose(add_values, expected_add, rtol=0.0, atol=0.0)

    test.assertTrue(
        any(target.scalar_dtype is wp.bfloat16 for target in atomic_bfloat16_kernel.adj.det_meta.scatter_targets)
    )

    def launch_minmax_once():
        output_min = wp.full(out_size, value=wp.bfloat16(16.0), dtype=wp.bfloat16, device=device)
        output_max = wp.zeros(out_size, dtype=wp.bfloat16, device=device)
        wp.launch(
            atomic_bfloat16_minmax_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output_min, output_max],
            device=device,
        )
        min_values = output_min.numpy()
        max_values = output_max.numpy()
        return (
            _bfloat16_numpy_bits(min_values),
            _bfloat16_numpy_bits(max_values),
            _bfloat16_numpy_to_float32(min_values),
            _bfloat16_numpy_to_float32(max_values),
        )

    _, _, min_values, max_values = assert_equal_repeated(launch_minmax_once)
    np.testing.assert_allclose(min_values, expected_min, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(max_values, expected_max, rtol=0.0, atol=0.0)

    test.assertTrue(
        any(target.scalar_dtype is wp.bfloat16 for target in atomic_bfloat16_minmax_kernel.adj.det_meta.scatter_targets)
    )


def test_atomic_double_deterministic(test, device):
    """Verify deterministic mode with float64 atomics."""
    n = 2048
    out_size = 32
    rng = np.random.default_rng(66)

    data_np = rng.random(n).astype(np.float64)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float64, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    def launch_once():
        output = wp.zeros(out_size, dtype=wp.float64, device=device)
        wp.launch(
            atomic_double_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        return output.numpy().copy()

    assert_equal_repeated(launch_once)


def test_vec3_atomic_add_deterministic(test, device):
    """Verify deterministic mode for composite ``wp.vec3`` atomic adds."""
    n = 1024
    out_size = 16
    rng = np.random.default_rng(67)

    data_np = rng.standard_normal((n, 3), dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    def launch_once():
        output = wp.zeros(out_size, dtype=wp.vec3, device=device)
        wp.launch(vec3_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
        return output.numpy().copy()

    expected = np.zeros((out_size, 3), dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    np.testing.assert_allclose(assert_equal_repeated(launch_once), expected, rtol=1e-5, atol=1e-5)


def test_vec3_atomic_minmax_deterministic(test, device):
    """Verify deterministic component-wise ``wp.vec3`` min/max reductions."""
    n = 2048
    rng = np.random.default_rng(68)
    points_np = rng.standard_normal((n, 3), dtype=np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)

    def launch_once():
        out_min = wp.empty(1, dtype=wp.vec3, device=device)
        out_max = wp.empty(1, dtype=wp.vec3, device=device)
        out_min.fill_(wp.vec3(np.inf, np.inf, np.inf))
        out_max.fill_(wp.vec3(-np.inf, -np.inf, -np.inf))
        wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)
        return out_min.numpy().copy(), out_max.numpy().copy()

    expected_min = np.min(points_np, axis=0, keepdims=True)
    expected_max = np.max(points_np, axis=0, keepdims=True)
    result_min, result_max = assert_equal_repeated(launch_once)

    np.testing.assert_allclose(result_min, expected_min, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result_max, expected_max, rtol=0.0, atol=0.0)


def test_mat33_atomic_add_deterministic(test, device):
    """Verify deterministic mode for composite ``wp.mat33`` atomic adds."""
    n = 512
    out_size = 8
    rng = np.random.default_rng(69)

    data_np = rng.standard_normal((n, 3, 3), dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.mat33, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    def launch_once():
        output = wp.zeros(out_size, dtype=wp.mat33, device=device)
        wp.launch(mat33_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
        return output.numpy().copy()

    expected = np.zeros((out_size, 3, 3), dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    np.testing.assert_allclose(assert_equal_repeated(launch_once), expected, rtol=1e-5, atol=1e-5)


def test_triple_scatter_capacity_estimate(test, device):
    """Verify kernels with >2 scatters per thread do not overflow the buffer."""
    n = 512
    rng = np.random.default_rng(12)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    def launch_once():
        output = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(triple_scatter_add_kernel, dim=n, inputs=[data], outputs=[output], device=device)
        return output.numpy().copy()

    expected = np.array([6.0 * data_np.sum()], dtype=np.float32)
    np.testing.assert_allclose(assert_equal_repeated(launch_once), expected, rtol=1e-5, atol=1e-5)


def test_loop_scatter_max_records_override(test, device):
    """Verify ``deterministic_max_records`` handles dynamic loop emission counts."""
    n = 256
    rng = np.random.default_rng(71)
    data_np = rng.random(n, dtype=np.float32)
    counts_np = np.full(n, 4, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts = wp.array(counts_np, dtype=wp.int32, device=device)

    expected = np.array([np.dot(data_np, counts_np).astype(np.float32)], dtype=np.float32)

    def launch_once():
        output = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(loop_scatter_add_kernel, dim=n, inputs=[data, counts], outputs=[output], device=device)
        return output.numpy().copy()

    np.testing.assert_allclose(assert_equal_repeated(launch_once), expected, rtol=1e-5, atol=1e-5)


def test_scatter_overflow_reports_error(test, device):
    """Verify an underprovisioned dynamic scatter reports overflow to the host."""
    n = 2048
    data = wp.ones(n, dtype=wp.float32, device=device)
    counts = wp.full(n, value=2, dtype=wp.int32, device=device)
    output = wp.zeros(1, dtype=wp.float32, device=device)

    with test.assertRaisesRegex(wp.WarpDeterminismError, "more transformed atomic calls"):
        wp.launch(underprovisioned_loop_scatter_kernel, dim=n, inputs=[data, counts], outputs=[output], device=device)


def test_scatter_large_launch_rejected(test, device):
    """Verify scatter key packing fails clearly before oversized launches."""
    data = wp.ones(1, dtype=wp.float32, device=device)
    indices = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(1, dtype=wp.float32, device=device)

    with test.assertRaisesRegex(wp.WarpDeterminismError, "supports at most 2\\^32 threads"):
        wp.launch(
            scatter_add_kernel,
            dim=(65536, 65537),
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )


def test_mixed_reduce_ops_same_array(test, device):
    """Verify mixed reduction families on one array are rejected in deterministic mode."""
    data_np = np.full(4, 0.05, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    output = wp.zeros(1, dtype=wp.float32, device=device)

    with test.assertRaisesRegex(wp.WarpDeterminismError, "receives both 'add' and 'max' atomic operations"):

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def mixed_reduce_op_same_array_local_kernel(
            data: wp.array[wp.float32],
            output: wp.array[wp.float32],
        ):
            tid = wp.tid()
            wp.atomic_add(output, 0, data[tid])
            wp.atomic_max(output, 0, 1.0)

        wp.launch(
            mixed_reduce_op_same_array_local_kernel,
            dim=data_np.shape[0],
            inputs=[data],
            outputs=[output],
            device=device,
        )


def test_scatter_target_aliasing_rejected(test, device):
    """Reject overlapping scatter-target arguments but accept disjoint interleaved views."""

    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def two_targets_kernel(output_a: wp.array[wp.float32], output_b: wp.array[wp.float32]):
        wp.atomic_add(output_a, 0, 1.0)
        wp.atomic_add(output_b, 0, 2.0)

    base = wp.zeros(4, dtype=wp.float32, device=device)
    wp.launch(two_targets_kernel, dim=1, outputs=[base[0::2], base[1::2]], device=device)
    np.testing.assert_array_equal(base.numpy(), [1.0, 2.0, 0.0, 0.0])

    with test.assertRaisesRegex(wp.WarpDeterminismError, "arguments 'output_a' and 'output_b' overlap"):
        wp.launch(two_targets_kernel, dim=1, outputs=[base, base[1:]], device=device)


def test_scatter_target_direct_access_aliasing_rejected(test, device):
    """Reject a scatter target aliased through a directly read argument."""

    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def read_kernel(direct_input: wp.array[wp.float32], atomic_output: wp.array[wp.float32]):
        wp.atomic_add(atomic_output, 0, direct_input[0])

    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def indexed_read_kernel(direct_input: wp.indexedarray[wp.float32], atomic_output: wp.array[wp.float32]):
        wp.atomic_add(atomic_output, 0, direct_input[0])

    output = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(read_kernel, dim=1, inputs=[wp.full(1, 7.0, device=device)], outputs=[output], device=device)
    np.testing.assert_array_equal(output.numpy(), [7.0])

    indexed_output = wp.indexedarray(output, [wp.zeros(1, dtype=wp.int32, device=device)])
    for kernel, alias in ((read_kernel, output), (indexed_read_kernel, indexed_output)):
        with test.subTest(kernel=kernel.key):
            with test.assertRaisesRegex(wp.WarpDeterminismError, "overlaps argument 'direct_input'.*read directly"):
                wp.launch(kernel, dim=1, inputs=[alias], outputs=[output], device=device)


def test_declared_deferred_atomic_pre_read(test, device):
    """Allow a declared tree reduction, including a read-only launch-time alias."""
    parent = wp.array([-1, 0, 0, 1, 1, 2, 2], dtype=wp.int32, device=device)
    levels = [
        wp.array([3, 4, 5, 6], dtype=wp.int32, device=device),
        wp.array([1, 2], dtype=wp.int32, device=device),
    ]
    expected = np.array([7.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def launch_once(kernel, *, alias):
        mass = wp.ones(7, dtype=wp.float32, device=device)
        for level in levels:
            if alias:
                wp.launch(kernel, dim=level.shape[0], inputs=[parent, level, mass], outputs=[mass], device=device)
            else:
                wp.launch(kernel, dim=level.shape[0], inputs=[parent, level], outputs=[mass], device=device)
        return mass.numpy().copy()

    for kernel, alias in (
        (deferred_tree_reduce_kernel, False),
        (deferred_tree_reduce_helper_kernel, False),
        (deferred_tree_reduce_alias_kernel, True),
    ):
        with test.subTest(kernel=kernel.key):
            np.testing.assert_array_equal(
                assert_equal_repeated(lambda kernel=kernel, alias=alias: launch_once(kernel, alias=alias)), expected
            )


def test_declared_deferred_tile_load_pre_read(test, device):
    """Allow a declared target to be tile-loaded before its atomic update."""
    out = wp.ones(32, dtype=wp.float32, device=device)
    observed = wp.zeros_like(out)

    wp.launch_tiled(
        declared_deferred_tile_load_kernel,
        dim=1,
        inputs=[out],
        outputs=[observed],
        block_dim=32,
        device=device,
    )

    np.testing.assert_array_equal(observed.numpy(), np.ones(32, dtype=np.float32))
    np.testing.assert_array_equal(out.numpy(), np.array([33.0] + [1.0] * 31, dtype=np.float32))


def test_declared_deferred_separate_helper_calls(test, device):
    """Allow safe reads and updates from separate calls to a looping helper."""
    a = wp.full(1, value=2.0, dtype=wp.float32, device=device)
    b = wp.zeros(1, dtype=wp.float32, device=device)
    c = wp.full(1, value=3.0, dtype=wp.float32, device=device)

    wp.launch(
        declared_deferred_separate_helper_calls_kernel,
        dim=1,
        inputs=[a, b, c, 2],
        device=device,
    )

    np.testing.assert_array_equal(a.numpy(), [8.0])
    np.testing.assert_array_equal(b.numpy(), [4.0])


def test_nested_helper_scatter_capacity(test, device):
    """Allocate enough deterministic scatter storage for nested helper calls."""
    thread_count = 1024
    out = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(nested_helper_scatter_kernel, dim=thread_count, outputs=[out], device=device)

    np.testing.assert_array_equal(out.numpy(), [2.0 * thread_count])


def test_scatter_tile_access_rejected(test, device):
    """Validate deferred targets against tile reads, stores, and ordinary tile atomics."""

    def make_kernel(operation):
        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def tile_access(output: wp.array[float], other: wp.array[float], observed: wp.array[float]):
            wp.atomic_add(output, 0, 1.0)
            if wp.static(operation == "load"):
                wp.tile_store(observed, wp.tile_load(other, shape=32))
            elif wp.static(operation == "store"):
                wp.tile_store(other, wp.tile_ones(shape=32, dtype=float))
            else:
                wp.tile_atomic_add(other, wp.tile_ones(shape=32, dtype=float))

        return tile_access

    output = wp.zeros(32, device=device)
    other = wp.zeros_like(output)
    observed = wp.zeros_like(output)

    for operation in ("load", "store", "atomic"):
        with test.subTest(operation=operation):
            tile_access = make_kernel(operation)
            wp.launch_tiled(tile_access, dim=1, inputs=[output, other, observed], block_dim=32, device=device)
            with test.assertRaisesRegex(wp.WarpDeterminismError, "overlaps argument 'other'"):
                wp.launch_tiled(tile_access, dim=1, inputs=[output, output, observed], block_dim=32, device=device)


def test_helper_name_collision(test, device):
    """Verify deterministic helpers stay unique for labels with the same sanitized form."""
    data_np = np.linspace(0.25, 2.0, 32, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    direct = wp.zeros(1, dtype=wp.float32, device=device)
    field = wp.zeros(1, dtype=wp.float32, device=device)
    holder = _DetNameCollisionStruct()
    holder.b = field

    wp.launch(helper_name_collision_kernel, dim=data_np.shape[0], inputs=[direct, holder, data], device=device)

    np.testing.assert_allclose(direct.numpy(), np.array([data_np.sum()], dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(field.numpy(), np.array([2.0 * data_np.sum()], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_deterministic_closure_kernel(test, device):
    """Verify deterministic closure kernels remain reproducible and distinct."""
    kernel_a = _make_deterministic_closure_kernel(_det_closure_transform_a)
    kernel_b = _make_deterministic_closure_kernel(_det_closure_transform_b)

    test.assertIsNot(kernel_a, kernel_b)
    test.assertNotEqual(kernel_a.module.name, kernel_b.module.name)

    n = 512
    rng = np.random.default_rng(30)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    def launch_once():
        out_a = wp.zeros(8, dtype=wp.float32, device=device)
        out_b = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(kernel_a, dim=n, inputs=[data], outputs=[out_a], device=device)
        wp.launch(kernel_b, dim=n, inputs=[data], outputs=[out_b], device=device)
        return out_a.numpy().copy(), out_b.numpy().copy()

    result_a, result_b = assert_equal_repeated(launch_once)
    test.assertFalse(np.array_equal(result_a, result_b))


def test_deterministic_func_kernels(test, device):
    """Verify deterministic atomics inside direct and nested ``@wp.func`` calls."""
    n = 512
    out_size = 16
    data_np, indices_np, data, indices = _make_float_scatter_inputs(n, out_size, 74, device)
    expected = _reference_scatter_add_float32(data_np, indices_np, out_size)

    for kernel in (func_scatter_add_kernel, nested_func_scatter_add_kernel):
        with test.subTest(kernel=kernel.key):
            result = assert_equal_repeated(
                lambda kernel=kernel: _launch_scatter_once(kernel, n, data, indices, out_size, device)
            )
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


class TestDeterministicScatter(DeterministicTestBase):
    """Test deterministic scatter/reduce atomic lowering."""

    def test_overlap_check_matrix_blocks(self):
        """Distinguish disjoint and overlapping column blocks of one matrix."""
        from warp._src.deterministic import _deterministic_arrays_overlap  # noqa: PLC0415

        base = wp.zeros((2, 4), dtype=wp.float32, device="cpu")
        self.assertFalse(_deterministic_arrays_overlap(base[:, :2], base[:, 2:]))
        self.assertTrue(_deterministic_arrays_overlap(base[:, :3], base[:, 2:]))

    def test_scatter_target_access_rejected(self):
        """Reject direct access to a deferred scatter target, including through views.

        Compile-time checks apply to the whole array because indices are runtime values.
        """

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def read_after_kernel(output: wp.array2d[wp.float32], observed: wp.array[wp.float32]):
            wp.atomic_add(output, 0, 0, 1.0)
            observed[0] = output[1, 0]

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def store_before_kernel(output: wp.array2d[wp.float32]):
            output[1, 0] = 5.0
            wp.atomic_add(output, 0, 0, 1.0)

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def row_read_kernel(output: wp.array2d[wp.float32], observed: wp.array[wp.float32]):
            wp.atomic_add(output, 0, 0, 1.0)
            row = output[1]
            observed[0] = row[0]

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def slice_store_kernel(output: wp.array2d[wp.float32]):
            wp.atomic_add(output, 0, 0, 1.0)
            column = output[:, 1]
            column[0] = 5.0

        for kernel, operation in (
            (read_after_kernel, "read directly"),
            (store_before_kernel, "written directly"),
            (row_read_kernel, "read directly"),
            (slice_store_kernel, "written directly"),
        ):
            with self.subTest(kernel=kernel.key):
                with self.assertRaisesRegex(
                    wp.WarpDeterminismError,
                    f"array 'output' receives deterministic atomic updates and is also {operation}",
                ):
                    wp.load_module(module=kernel.module, device="cpu")

    def test_declared_deferred_atomic_accesses(self):
        """Permit safe pre-atomic reads but reject unsafe accesses and dynamic loops."""
        wp.load_module(module=deferred_tree_reduce_kernel.module, device="cpu")
        wp.load_module(module=deferred_tree_reduce_helper_kernel.module, device="cpu")
        wp.load_module(module=declared_deferred_separate_helper_calls_kernel.module, device="cpu")

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def undeclared_pre_atomic_read_kernel(out: wp.array[wp.float32]):
            value = out[1]
            wp.atomic_add(out, 0, value)

        for kernel, error in (
            (declared_deferred_read_after_kernel, "read directly"),
            (declared_deferred_helper_read_after_kernel, "read directly"),
            (declared_deferred_nested_helper_read_after_kernel, "read directly"),
            (declared_deferred_store_kernel, "written directly"),
            (undeclared_pre_atomic_read_kernel, "read directly"),
            (declared_deferred_dynamic_for_read_kernel, "same dynamic loop"),
            (declared_deferred_dynamic_while_read_kernel, "same dynamic loop"),
            (declared_deferred_dynamic_helper_read_kernel, "same dynamic loop"),
        ):
            with self.subTest(kernel=kernel.key):
                with self.assertRaisesRegex(
                    wp.WarpDeterminismError,
                    error,
                ):
                    wp.load_module(module=kernel.module, device="cpu")

    def test_deferred_atomic_declaration_validation(self):
        """Validate declared targets when the kernel is defined and built."""
        with self.assertRaisesRegex(ValueError, "unknown argument"):

            @wp.kernel(deterministic_deferred_atomics=("missing",))
            def unknown_deferred_atomic_argument(out: wp.array[wp.float32]):
                wp.atomic_add(out, 0, 1.0)

        with self.assertRaisesRegex(TypeError, "requires array argument"):

            @wp.kernel(deterministic_deferred_atomics=("scale",))
            def scalar_deferred_atomic_argument(out: wp.array[wp.float32], scale: float):
                wp.atomic_add(out, 0, scale)

        @wp.kernel(
            module="unique",
            module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN},
            deterministic_deferred_atomics=("out",),
        )
        def unused_deferred_atomic_argument(out: wp.array[wp.float32]):
            out[0] = 1.0

        with self.assertRaisesRegex(wp.WarpDeterminismError, "without deterministic accumulation atomics"):
            wp.load_module(module=unused_deferred_atomic_argument.module, device="cpu")


def _add(name, devices=cuda_devices, **kwargs):
    add_function_test(TestDeterministicScatter, name, globals()[name], devices=devices, **kwargs)


for _name in (
    "test_scatter_add_reproducibility",
    "test_gpu_to_gpu_matches_canonical_float32_reference",
    "test_augassign_add_reproducibility",
    "test_multi_array_atomic",
    "test_atomic_sub_deterministic",
    "test_atomic_add_2d",
    "test_atomic_add_negative_indices",
    "test_sliced_2d_array_atomic_add",
    "test_sliced_3d_array_atomic_add",
    "test_strided_1d_view_atomic_add",
    "test_zero_stride_view_atomic_add",
    "test_column_slice_atomic_add",
    "test_transposed_2d_atomic_add",
    "test_atomic_half_deterministic",
    "test_atomic_double_deterministic",
    "test_vec3_atomic_add_deterministic",
    "test_vec3_atomic_minmax_deterministic",
    "test_mat33_atomic_add_deterministic",
    "test_triple_scatter_capacity_estimate",
    "test_loop_scatter_max_records_override",
    "test_scatter_overflow_reports_error",
    "test_scatter_large_launch_rejected",
    "test_mixed_reduce_ops_same_array",
    "test_scatter_target_aliasing_rejected",
    "test_scatter_target_direct_access_aliasing_rejected",
    "test_declared_deferred_atomic_pre_read",
    "test_declared_deferred_tile_load_pre_read",
    "test_declared_deferred_separate_helper_calls",
    "test_nested_helper_scatter_capacity",
    "test_scatter_tile_access_rejected",
    "test_helper_name_collision",
    "test_deterministic_closure_kernel",
    "test_deterministic_func_kernels",
):
    _add(_name)

_add("test_scatter_add_correctness", devices=all_devices)
_add("test_atomic_bfloat16_deterministic", devices=bfloat16_cuda_devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
