# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for representative composite-component slot writes.

Covers the critical paths for array-rooted writes where the access chain from
``arr[i]`` lands on a composite slot, including plain assignment, selected
augmented assignments, and struct-field array aliases.

See ``asv/benchmarks/codegen/composite_component_*.py`` for the
performance contract that these tests complement.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

mat22i = wp.types.matrix(shape=(2, 2), dtype=wp.int32)
mat22s = wp.types.matrix(shape=(2, 2), dtype=wp.int16)
mat23 = wp.types.matrix(shape=(2, 3), dtype=float)


@wp.struct
class Scalar:
    a: wp.float32
    b: wp.float32


@wp.struct
class StateStruct:
    position: wp.vec3
    velocity: wp.vec3


@wp.kernel
def seed_scalar_grad_kernel(g: wp.array[Scalar], a_val: wp.float32, b_val: wp.float32):
    i = wp.tid()
    g[i] = Scalar(a_val, b_val)


@wp.kernel
def seed_state_grad_kernel(g: wp.array[StateStruct], pos: wp.vec3, vel: wp.vec3):
    i = wp.tid()
    s = StateStruct()
    s.position = pos
    s.velocity = vel
    g[i] = s


@wp.kernel
def read_state_position_kernel(src: wp.array[StateStruct], dst: wp.array[wp.vec3]):
    dst[0] = src[0].position


# --------- kernels under test ---------


@wp.kernel
def gh583_kernel(dst: wp.array[wp.vec3], src: wp.array[wp.float32]):
    i = wp.tid()
    dst[i].y = src[i]


@wp.kernel
def gh248_kernel(y: wp.array[wp.vec3], x: wp.array[wp.float32]):
    tid = wp.tid()
    y[tid].x = x[tid] * 2.0
    y[tid].y = x[tid] * 3.0
    y[tid].z = x[tid] * 4.0


@wp.kernel
def gh1174_kernel(y: wp.array[Scalar], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i].a = x[i]


@wp.kernel
def mat33_element_kernel(y: wp.array[wp.mat33], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i][1, 2] = x[i]


@wp.kernel
def mat33_row_kernel(y: wp.array[wp.mat33], x: wp.array[wp.vec3]):
    i = wp.tid()
    y[i][1] = x[i]


@wp.kernel
def array_rooted_vec3_subscript_component_iadd_kernel(src: wp.array[wp.float32], dst: wp.array[wp.vec3]):
    dst[0][0] += src[0]


@wp.kernel
def array_rooted_vec3_attribute_component_iadd_kernel(src: wp.array[wp.float32], dst: wp.array[wp.vec3]):
    dst[0].x += src[0]


@wp.kernel(module="unique")
def array_rooted_vec3_component_imul_backward_enabled_kernel(src: wp.array[wp.float32], dst: wp.array[wp.vec3]):
    dst[0].x *= src[0]


@wp.kernel
def array_rooted_mat23_row_store_kernel(src: wp.array[wp.vec3], dst: wp.array[mat23]):
    dst[0][0] = src[0]


@wp.kernel
def mat33_row_iadd_kernel(y: wp.array[wp.mat33], x: wp.array[wp.vec3]):
    i = wp.tid()
    y[i][1] += x[i]


@wp.kernel(module="unique")
def update_inline_sliced_arrays(
    components: wp.array[wp.vec3],
    source: wp.array[float],
    start: int,
    stop: int,
    step: int,
):
    i = wp.tid()
    components[start:stop:step][i].x = source[i]
    components[start:stop:step][i][1] += source[i]


def test_inline_slice_updates_preserve_values_and_gradients(test, device):
    """Write through inline slices and route gradients to the selected elements."""
    for start, stop, step, selected in ((0, 4, 2, [0, 2]), (3, -5, -2, [3, 1])):
        with test.subTest(step=step):
            components = wp.full(4, wp.vec3(7.0), dtype=wp.vec3, requires_grad=True, device=device)
            source = wp.array([3.0, 5.0], dtype=float, requires_grad=True, device=device)
            with wp.Tape() as tape:
                wp.launch(
                    update_inline_sliced_arrays,
                    dim=2,
                    inputs=[components, source, start, stop, step],
                    device=device,
                )

            expected_store = np.full(4, 7.0)
            expected_store[selected] = [3.0, 5.0]
            expected_add = np.full(4, 7.0)
            expected_add[selected] = [10.0, 12.0]
            expected_components = np.full((4, 3), 7.0)
            expected_components[:, 0] = expected_store
            expected_components[:, 1] = expected_add
            np.testing.assert_array_equal(components.numpy(), expected_components)

            tape.backward(
                grads={
                    components: wp.full(4, wp.vec3(1.0), dtype=wp.vec3, device=device),
                }
            )
            expected_store_grad = np.ones(4)
            expected_store_grad[selected] = 0.0
            expected_component_grad = np.ones((4, 3))
            expected_component_grad[:, 0] = expected_store_grad
            np.testing.assert_array_equal(components.grad.numpy(), expected_component_grad)
            np.testing.assert_array_equal(source.grad.numpy(), [2.0, 2.0])
            np.testing.assert_array_equal(components.numpy(), expected_components)


@wp.func
def choose_slot_and_mutate_src(src: wp.array[wp.float32]) -> int:
    # Side-effecting slot-index expression: mutates the array the rhs reads.
    src[0] = 7.0
    return 0


@wp.kernel
def slot_write_rhs_eval_order_kernel(dst: wp.array[wp.vec3], src: wp.array[wp.float32]):
    # Plain assignment: the rhs (``src[0]``) must be read before the slot
    # index expression (which mutates ``src[0]``) runs.
    dst[choose_slot_and_mutate_src(src)].x = src[0]


@wp.kernel
def slot_write_reference_index_kernel(dst: wp.array[wp.vec3], idx: wp.array[wp.int32]):
    # The array index ``idx[0]`` is a reference (an array read). The composite
    # slot fast path must load it to a value before emitting it into
    # ``wp::index(...)``; otherwise a pointer reaches an integer parameter and
    # the kernel fails to compile.
    dst[idx[0]].x = 9.0


@wp.kernel(module="unique")
def slot_write_float_array_index_kernel(dst: wp.array[wp.vec3]):
    dst[0.0].x = 1.0


@wp.kernel(module="unique")
def slot_augassign_float_array_index_kernel(dst: wp.array[wp.vec3], idx: wp.array[wp.float32]):
    dst[idx[0]].x += 1.0


@wp.kernel
def mat22_row_iadd_atomic_kernel(values: wp.array[wp.vec2], out: wp.array[wp.mat22]):
    tid = wp.tid()
    out[0][1] += values[tid]


@wp.kernel
def vec3_component_iadd_atomic_kernel(values: wp.array[wp.float32], out: wp.array[wp.vec3]):
    tid = wp.tid()
    out[0][1] += values[tid]


@wp.kernel
def vec2_component_iadd_replay_kernel(
    vec_y: wp.array[wp.vec2],
    scalar_y: wp.array[wp.float32],
    vec_x: wp.array[wp.float32],
    scalar_x: wp.array[wp.float32],
    vec_out: wp.array[wp.float32],
    scalar_out: wp.array[wp.float32],
):
    vec_y[0].x += vec_x[0]
    scalar_y[0] += scalar_x[0]

    vec_out[0] = vec_y[0].x * vec_y[0].x
    scalar_out[0] = scalar_y[0] * scalar_y[0]


@wp.kernel
def vec2_component_isub_replay_kernel(
    vec_y: wp.array[wp.vec2],
    scalar_y: wp.array[wp.float32],
    vec_x: wp.array[wp.float32],
    scalar_x: wp.array[wp.float32],
    vec_out: wp.array[wp.float32],
    scalar_out: wp.array[wp.float32],
):
    vec_y[0].x -= vec_x[0]
    scalar_y[0] -= scalar_x[0]

    vec_out[0] = vec_y[0].x * vec_y[0].x
    scalar_out[0] = scalar_y[0] * scalar_y[0]


@wp.kernel
def mat22i_row_iand_atomic_kernel(values: wp.array[wp.vec2i], out: wp.array[mat22i]):
    tid = wp.tid()
    out[0][1] &= values[tid]


@wp.kernel
def vec2s_component_iadd_non_atomic_slot_kernel(values: wp.array[wp.int16], out: wp.array[wp.vec2s]):
    i = wp.tid()
    out[i].x += values[i]


@wp.kernel
def mat22s_row_iadd_non_atomic_slot_kernel(values: wp.array[wp.vec2s], out: wp.array[mat22s]):
    i = wp.tid()
    out[i][1] += values[i]


@wp.kernel(module="unique", enable_backward=False)
def vec3_component_idiv_reference_array_index_kernel(out: wp.array[wp.vec3], idx: wp.array[wp.int32]):
    out[idx[0]][0] /= 2.0


@wp.kernel(module="unique", enable_backward=False)
def vec3_component_idiv_reference_component_index_kernel(out: wp.array[wp.vec3], comp: wp.array[wp.int32]):
    out[0][comp[0]] /= 2.0


def test_array_composite_slot_int16_augassign(test, device):
    """Update ``int16`` vector components and matrix rows in place."""
    scalar_values = wp.array(np.array([7], dtype=np.int16), dtype=wp.int16, device=device)
    vec_out = wp.zeros(1, dtype=wp.vec2s, device=device)
    wp.launch(vec2s_component_iadd_non_atomic_slot_kernel, 1, inputs=[scalar_values, vec_out], device=device)
    assert_np_equal(vec_out.numpy(), np.array([[7, 0]], dtype=np.int16))

    row_values = wp.array(np.array([[1, 2]], dtype=np.int16), dtype=wp.vec2s, device=device)
    mat_out = wp.zeros(1, dtype=mat22s, device=device)
    wp.launch(mat22s_row_iadd_non_atomic_slot_kernel, 1, inputs=[row_values, mat_out], device=device)
    expected_mat = np.zeros((1, 2, 2), dtype=np.int16)
    expected_mat[0, 1, :] = [1, 2]
    assert_np_equal(mat_out.numpy(), expected_mat)


def test_array_composite_slot_augassign_with_array_indices(test, device):
    """Update composite slots selected by array-valued indices."""
    values = np.array([[8.0, 6.0, 4.0], [10.0, 12.0, 14.0]], dtype=np.float32)

    idx = wp.array([1], dtype=wp.int32, device=device)
    out = wp.array(values, dtype=wp.vec3, device=device)
    wp.launch(vec3_component_idiv_reference_array_index_kernel, 1, inputs=[out, idx], device=device)

    expected = values.copy()
    expected[1, 0] = 5.0
    assert_np_equal(out.numpy(), expected)

    comp = wp.array([2], dtype=wp.int32, device=device)
    out = wp.array(values, dtype=wp.vec3, device=device)
    wp.launch(vec3_component_idiv_reference_component_index_kernel, 1, inputs=[out, comp], device=device)

    expected = values.copy()
    expected[0, 2] = 2.0
    assert_np_equal(out.numpy(), expected)


@wp.kernel
def transform_p_kernel(y: wp.array[wp.transformf], p: wp.array[wp.vec3]):
    i = wp.tid()
    y[i].p = p[i]


@wp.kernel
def struct_vec_field_kernel(y: wp.array[StateStruct], p: wp.array[wp.vec3]):
    i = wp.tid()
    y[i].position = p[i]


@wp.struct
class VecArrayFieldHolder:
    values: wp.array[wp.vec3]


@wp.struct
class MatArrayFieldHolder:
    values: wp.array[wp.mat33]


@wp.kernel
def embedded_array_slot_write_kernel(holders: wp.array[VecArrayFieldHolder], src: wp.array[float]):
    holders[0].values[0].x = src[0]


def test_embedded_array_without_outer_gradient(test, device):
    """Differentiate an embedded component store without an outer gradient array."""
    holder = VecArrayFieldHolder()
    holder.values = wp.array([[2.0, 3.0, 4.0]], dtype=wp.vec3, device=device, requires_grad=True)
    holders = wp.array([holder], dtype=VecArrayFieldHolder, device=device)
    src = wp.array([5.0], dtype=float, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(embedded_array_slot_write_kernel, 1, [holders, src], device=device)
    holder.values.grad.fill_(wp.vec3(1.0))
    tape.backward()
    np.testing.assert_array_equal(holder.values.numpy(), [[5.0, 3.0, 4.0]])
    np.testing.assert_array_equal(src.grad.numpy(), [1.0])
    np.testing.assert_array_equal(holder.values.grad.numpy(), [[0.0, 1.0, 1.0]])


@wp.kernel
def struct_field_slot_write_rhs_eval_order_kernel(holder: VecArrayFieldHolder, src: wp.array[wp.float32]):
    holder.values[choose_slot_and_mutate_src(src)].x = src[0]


@wp.kernel
def _k_struct_field_vec_component_write(holder: VecArrayFieldHolder, x: wp.array[wp.float32]):
    holder.values[0].x = x[0]


@wp.kernel
def _k_struct_field_array_alias_component_write(holder: VecArrayFieldHolder, x: wp.array[wp.float32]):
    values = holder.values
    values[0].x = x[0]


@wp.kernel
def _k_array_view_component_write(values: wp.array2d[wp.vec3], x: wp.array[wp.float32]):
    view = values[0]
    view[0].x = x[0]


@wp.kernel(module="unique")
def nested_view_component_write(values: wp.array3d[wp.vec3], indices: wp.array[int], src: wp.array[float]):
    plane = values[0]
    row = plane[indices[0]]
    row[0].x = src[0]


def test_nested_view_component_gradient(test, device):
    """Route component gradients through chained views with an array-valued index."""
    values = wp.zeros((1, 2, 1), dtype=wp.vec3, device=device, requires_grad=True)
    indices = wp.array([1], dtype=int, device=device)
    src = wp.array([3.0], dtype=float, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(nested_view_component_write, 1, [values, indices, src], device=device)
    values.grad = wp.full_like(values, wp.vec3(2.0, 3.0, 4.0))
    tape.backward()
    expected = np.zeros((1, 2, 1, 3), dtype=np.float32)
    expected[0, 1, 0, 0] = 3.0
    np.testing.assert_array_equal(values.numpy(), expected)
    np.testing.assert_array_equal(src.grad.numpy(), [2.0])
    expected_grad = np.tile([2.0, 3.0, 4.0], (1, 2, 1, 1))
    expected_grad[0, 1, 0, 0] = 0.0
    np.testing.assert_array_equal(values.grad.numpy(), expected_grad)


@wp.kernel
def _k_struct_field_mat_row_write(holder: MatArrayFieldHolder, x: wp.array[wp.vec3]):
    holder.values[0][1] = x[0]


@wp.kernel
def _k_struct_field_vec_component_iadd_replay(
    holder: VecArrayFieldHolder,
    x: wp.array[wp.float32],
    out: wp.array[wp.float32],
):
    holder.values[0].x += x[0]
    out[0] = holder.values[0].x * holder.values[0].x


@wp.kernel
def _k_struct_field_array_alias_component_iadd_replay(
    holder: VecArrayFieldHolder,
    x: wp.array[wp.float32],
    out: wp.array[wp.float32],
):
    values = holder.values
    values[0].x += x[0]
    out[0] = values[0].x * values[0].x


@wp.kernel(module="unique", enable_backward=False)
def _k_struct_field_vec_component_iadd_atomic(holder: VecArrayFieldHolder, x: wp.array[wp.float32]):
    i = wp.tid()
    holder.values[0].x += x[i]


@wp.struct
class AdjVecHolder:
    vec: wp.vec3


@wp.func
def _adj_sum_vec(x: AdjVecHolder):
    return x.vec[0] + x.vec[1] + x.vec[2]


@wp.func_grad(_adj_sum_vec)
def _adj_sum_vec_grad(x: AdjVecHolder, adj_ret: wp.float32):
    # Plain component writes to an adjoint should use normal adjoint-store behavior.
    wp.adjoint[x.vec].y = adj_ret * 5.0


@wp.kernel
def _k_adjoint_component_write(xs: wp.array[AdjVecHolder], out: wp.array[wp.float32]):
    i = wp.tid()
    out[i] = _adj_sum_vec(xs[i])


@wp.func
def _lookup_vec_x(values: wp.array[wp.vec3], index: int) -> wp.float32:
    return values[index].x


@wp.func_grad(_lookup_vec_x)
def _adj_lookup_vec_x(values: wp.array[wp.vec3], index: int, adj_ret: wp.float32):
    wp.adjoint[values][index].x += adj_ret


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.NOT_GUARANTEED})
def _k_adjoint_component_atomic(values: wp.array[wp.vec3], indices: wp.array[wp.int32], out: wp.array[wp.float32]):
    i = wp.tid()
    out[i] = _lookup_vec_x(values, indices[i])


# --------- tests ---------


class TestCompositeComponentAdjoint(unittest.TestCase):
    pass


def test_gh583_array_vec_component(test, device):
    """Verify array vector component writes propagate scalar gradients."""
    n = 4
    src = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device)
    dst = wp.zeros(n, dtype=wp.vec3, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(gh583_kernel, n, inputs=[dst, src], device=device)

    dst.grad = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    tape.backward()

    assert_np_equal(src.grad.numpy(), np.ones(n, dtype=np.float32))


def test_gh248_sequential_component_writes(test, device):
    """Verify sequential vector component writes propagate gradients."""
    n = 3
    x = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device)
    y = wp.zeros(n, dtype=wp.vec3, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(gh248_kernel, n, inputs=[y, x], device=device)

    y.grad = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    tape.backward()

    assert_np_equal(x.grad.numpy(), np.full(n, 9.0, dtype=np.float32))


def test_gh1174_array_struct_field(test, device):
    """Verify array struct field writes propagate scalar gradients."""
    n = 3
    x = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device)
    y = wp.zeros(n, dtype=Scalar, requires_grad=True, device=device)

    wp.launch(seed_scalar_grad_kernel, n, inputs=[y.grad, 1.0, 0.0], device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(gh1174_kernel, n, inputs=[y, x], device=device)
    tape.backward()

    assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))


def test_mat33_element_write(test, device):
    """Verify mat33 element writes propagate scalar gradients."""
    n = 2
    x = wp.array(np.full(n, 7.0, dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device)
    y = wp.zeros(n, dtype=wp.mat33, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat33_element_kernel, n, inputs=[y, x], device=device)
    y.grad = wp.array(np.ones((n, 3, 3), dtype=np.float32), dtype=wp.mat33, device=device)
    tape.backward()

    assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))


def test_transform_translation_write(test, device):
    """Verify array transform translation writes propagate vector gradients."""
    n = 2
    p = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, requires_grad=True, device=device)
    y = wp.zeros(n, dtype=wp.transformf, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(transform_p_kernel, n, inputs=[y, p], device=device)
    y.grad = wp.array(np.ones((n, 7), dtype=np.float32), dtype=wp.transformf, device=device)
    tape.backward()

    assert_np_equal(p.grad.numpy(), np.ones((n, 3), dtype=np.float32))


def test_struct_vec_field_write(test, device):
    """Verify struct vec field writes propagate vector gradients."""
    n = 2
    p = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, requires_grad=True, device=device)
    y = wp.zeros(n, dtype=StateStruct, requires_grad=True, device=device)

    wp.launch(seed_state_grad_kernel, n, inputs=[y.grad, wp.vec3(1.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0)], device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(struct_vec_field_kernel, n, inputs=[y, p], device=device)
    tape.backward()

    assert_np_equal(p.grad.numpy(), np.ones((n, 3), dtype=np.float32))


def test_struct_field_array_composite_slot_write(test, device):
    """Verify struct-field array slot writes propagate gradients."""
    with test.subTest("direct struct field"):
        scalar_src = wp.array([2.0], dtype=wp.float32, requires_grad=True, device=device)
        vec_values = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=device)
        vec_holder = VecArrayFieldHolder()
        vec_holder.values = vec_values

        tape = wp.Tape()
        with tape:
            wp.launch(_k_struct_field_vec_component_write, 1, inputs=[vec_holder, scalar_src], device=device)

        vec_values.grad = wp.array([[3.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        tape.backward()

        assert_np_equal(vec_values.numpy(), np.array([[2.0, 0.0, 0.0]], dtype=np.float32))
        assert_np_equal(scalar_src.grad.numpy(), np.array([3.0], dtype=np.float32))

    with test.subTest("array alias"):
        alias_src = wp.array([4.0], dtype=wp.float32, requires_grad=True, device=device)
        alias_values = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=device)
        alias_holder = VecArrayFieldHolder()
        alias_holder.values = alias_values

        tape = wp.Tape()
        with tape:
            wp.launch(_k_struct_field_array_alias_component_write, 1, inputs=[alias_holder, alias_src], device=device)

        alias_values.grad = wp.array([[5.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        tape.backward()

        assert_np_equal(alias_values.numpy(), np.array([[4.0, 0.0, 0.0]], dtype=np.float32))
        assert_np_equal(alias_src.grad.numpy(), np.array([5.0], dtype=np.float32))

    with test.subTest("view"):
        view_src = wp.array([6.0], dtype=wp.float32, requires_grad=True, device=device)
        view_values = wp.zeros((1, 1), dtype=wp.vec3, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_array_view_component_write, 1, inputs=[view_values, view_src], device=device)

        view_values.grad = wp.array([[[7.0, 0.0, 0.0]]], dtype=wp.vec3, device=device)
        tape.backward()

        assert_np_equal(view_values.numpy(), np.array([[[6.0, 0.0, 0.0]]], dtype=np.float32))
        assert_np_equal(view_src.grad.numpy(), np.array([7.0], dtype=np.float32))

    with test.subTest("matrix row"):
        row_src = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, requires_grad=True, device=device)
        mat_values = wp.zeros(1, dtype=wp.mat33, requires_grad=True, device=device)
        mat_holder = MatArrayFieldHolder()
        mat_holder.values = mat_values

        tape = wp.Tape()
        with tape:
            wp.launch(_k_struct_field_mat_row_write, 1, inputs=[mat_holder, row_src], device=device)

        grad_seed = np.zeros((1, 3, 3), dtype=np.float32)
        grad_seed[0, 1, :] = [4.0, 5.0, 6.0]
        mat_values.grad = wp.array(grad_seed, dtype=wp.mat33, device=device)
        tape.backward()

        expected = np.zeros((1, 3, 3), dtype=np.float32)
        expected[0, 1, :] = [1.0, 2.0, 3.0]
        assert_np_equal(mat_values.numpy(), expected)
        assert_np_equal(row_src.grad.numpy(), np.array([[4.0, 5.0, 6.0]], dtype=np.float32))


def test_composite_slot_type_mismatch_reports_error(test, device):
    """Report incompatible values assigned to composite slots."""
    with wp.ScopedDevice(device):

        @wp.kernel(module="unique")
        def _k_mismatch(y: wp.array[wp.vec3], src: wp.array[wp.vec3]):
            i = wp.tid()
            y[i].y = src[i]  # scalar slot (.y) <- vec3 reference rhs: type mismatch

        n = 1
        y = wp.zeros(n, dtype=wp.vec3)
        src = wp.zeros(n, dtype=wp.vec3)

        with test.assertRaisesRegex(TypeError, "Composite slot assignment expects value of type"):
            wp.launch(_k_mismatch, n, inputs=[y, src])


def test_wp_adjoint_composite_component_write(test, device):
    """Verify custom adjoint component writes use normal adjoint stores."""
    with wp.ScopedDevice(device):
        n = 2
        x = AdjVecHolder()
        x.vec = wp.vec3(1.0, 2.0, 3.0)
        xs = wp.array([x] * n, dtype=AdjVecHolder, requires_grad=True)
        out = wp.zeros(n, dtype=wp.float32)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_adjoint_component_write, n, inputs=[xs], outputs=[out])
        tape.backward(grads={out: wp.ones(n, dtype=wp.float32)})

        grad_vec = xs.grad.numpy()["vec"]
        expected = np.tile(np.array([0.0, 5.0, 0.0], dtype=np.float32), (n, 1))
        assert_np_equal(grad_vec, expected)


def test_wp_adjoint_array_component_accumulates_cuda(test, device):
    """Accumulate contended array-component gradients on CUDA."""
    n = 4096
    value_count = 31
    rng = np.random.default_rng(307)
    values_np = rng.random((value_count, 3), dtype=np.float32)
    indices_np = rng.integers(0, value_count, size=n, dtype=np.int32)

    values = wp.array(values_np, dtype=wp.vec3, device=device, requires_grad=True)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    out = wp.zeros(n, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(_k_adjoint_component_atomic, dim=n, inputs=[values, indices], outputs=[out], device=device)
    tape.backward(grads={out: wp.ones(n, dtype=wp.float32, device=device)})

    expected = np.zeros((value_count, 3), dtype=np.float32)
    expected[:, 0] = np.bincount(indices_np, minlength=value_count).astype(np.float32)
    assert_np_equal(values.grad.numpy(), expected)


def test_array_mat33_row_write_backward(test, device):
    """Verify mat33 row writes propagate row gradients."""
    n = 2
    src = wp.array(np.tile([1.0, 2.0, 3.0], (n, 1)), dtype=wp.vec3, requires_grad=True, device=device)
    dst = wp.zeros(n, dtype=wp.mat33, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat33_row_kernel, n, inputs=[dst, src], device=device)

    grad_seed = np.zeros((n, 3, 3), dtype=np.float32)
    grad_seed[:, 1, :] = [10.0, 20.0, 30.0]
    dst.grad = wp.array(grad_seed, dtype=wp.mat33, device=device)
    tape.backward()

    expected = np.tile([10.0, 20.0, 30.0], (n, 1))
    assert_np_equal(src.grad.numpy(), expected)


def test_array_rooted_composite_slot_gradients(test, device):
    """Verify array-rooted slot writes propagate gradients."""
    component_kernels = (
        array_rooted_vec3_subscript_component_iadd_kernel,
        array_rooted_vec3_attribute_component_iadd_kernel,
    )

    for kernel in component_kernels:
        src = wp.array([2.0], dtype=wp.float32, requires_grad=True, device=device)
        dst = wp.array([[5.0, 0.0, 0.0]], dtype=wp.vec3, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[src, dst], device=device)

        dst.grad = wp.array(np.ones((1, 3), dtype=np.float32), dtype=wp.vec3, device=device)
        tape.backward()

        assert_np_equal(dst.numpy(), np.array([[7.0, 0.0, 0.0]], dtype=np.float32))
        assert_np_equal(src.grad.numpy(), np.array([1.0], dtype=np.float32))

    src = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, requires_grad=True, device=device)
    dst = wp.zeros(1, dtype=mat23, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(array_rooted_mat23_row_store_kernel, dim=1, inputs=[src, dst], device=device)

    dst.grad = wp.array(np.ones((1, 2, 3), dtype=np.float32), dtype=mat23, device=device)
    tape.backward()

    expected = np.zeros((1, 2, 3), dtype=np.float32)
    expected[0, 0, :] = [1.0, 2.0, 3.0]
    assert_np_equal(dst.numpy(), expected)
    assert_np_equal(src.grad.numpy(), np.ones((1, 3), dtype=np.float32))


def test_array_composite_slot_multiply_assign_backward_rejected(test, device):
    """Verify unsupported differentiable slot ``*=`` fails with backward enabled."""
    src = wp.array([2.0], dtype=wp.float32, requires_grad=True, device=device)
    dst = wp.array([[5.0, 0.0, 0.0]], dtype=wp.vec3, requires_grad=True, device=device)

    with test.assertRaisesRegex(Exception, "Differentiable composite slot augmented assignments"):
        wp.launch(array_rooted_vec3_component_imul_backward_enabled_kernel, dim=1, inputs=[src, dst], device=device)


def test_slot_write_rhs_eval_order(test, device):
    """Verify plain slot writes evaluate RHS before target indices."""
    # The index expression mutates ``src[0]``, but plain assignment must store
    # the original RHS value.
    src = wp.array([3.0], dtype=wp.float32, device=device)
    dst = wp.zeros(1, dtype=wp.vec3, device=device)

    wp.launch(slot_write_rhs_eval_order_kernel, dim=1, inputs=[dst, src], device=device)

    assert_np_equal(dst.numpy()[0, 0], np.float32(3.0))
    assert_np_equal(src.numpy()[0], np.float32(7.0))

    struct_src = wp.array([5.0], dtype=wp.float32, device=device)
    struct_dst = wp.zeros(1, dtype=wp.vec3, device=device)
    holder = VecArrayFieldHolder()
    holder.values = struct_dst

    wp.launch(struct_field_slot_write_rhs_eval_order_kernel, dim=1, inputs=[holder, struct_src], device=device)

    assert_np_equal(struct_dst.numpy()[0, 0], np.float32(5.0))
    assert_np_equal(struct_src.numpy()[0], np.float32(7.0))


def test_slot_write_with_array_index(test, device):
    """Verify slot writes load reference-typed root array indices."""
    # Reference-typed array indices must be loaded before raw slot access.
    dst = wp.zeros(2, dtype=wp.vec3, device=device)
    idx = wp.zeros(1, dtype=wp.int32, device=device)  # idx[0] == 0

    wp.launch(slot_write_reference_index_kernel, dim=1, inputs=[dst, idx], device=device)

    expected = np.zeros((2, 3), dtype=np.float32)
    expected[0, 0] = 9.0
    assert_np_equal(dst.numpy(), expected)


def test_slot_write_rejects_non_integer_array_index(test, device):
    """Verify slot writes reject non-integer root array indices during codegen."""
    with test.assertRaisesRegex(TypeError, "Array slot write indices must be integers"):
        slot_write_float_array_index_kernel.module.load(device=device)

    with test.assertRaisesRegex(TypeError, "Array slot write indices must be integers"):
        slot_augassign_float_array_index_kernel.module.load(device=device)


def test_contended_array_composite_slot_augassign_cuda(test, device):
    """Update contended vector components and matrix rows on CUDA."""
    n = 4096

    row_values = wp.array(np.ones((n, 2), dtype=np.float32), dtype=wp.vec2, device=device)
    scalar_values = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, device=device)

    mat_out = wp.zeros(1, dtype=wp.mat22, device=device)
    wp.launch(mat22_row_iadd_atomic_kernel, n, inputs=[row_values, mat_out], device=device)
    expected_mat = np.zeros((1, 2, 2), dtype=np.float32)
    expected_mat[0, 1, :] = float(n)
    assert_np_equal(mat_out.numpy(), expected_mat)

    vec_out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(vec3_component_iadd_atomic_kernel, n, inputs=[scalar_values, vec_out], device=device)
    expected_vec = np.zeros((1, 3), dtype=np.float32)
    expected_vec[0, 1] = float(n)
    assert_np_equal(vec_out.numpy(), expected_vec)

    struct_vec_out = wp.zeros(1, dtype=wp.vec3, device=device)
    struct_holder = VecArrayFieldHolder()
    struct_holder.values = struct_vec_out
    wp.launch(_k_struct_field_vec_component_iadd_atomic, n, inputs=[struct_holder, scalar_values], device=device)
    expected_vec = np.zeros((1, 3), dtype=np.float32)
    expected_vec[0, 0] = float(n)
    assert_np_equal(struct_vec_out.numpy(), expected_vec)

    row_bits = wp.array(np.tile([0x0F, 0xF0], (n, 1)).astype(np.int32), dtype=wp.vec2i, device=device)

    mat_bits = np.zeros((1, 2, 2), dtype=np.int32)
    mat_bits[0, 1, :] = [-1, -1]
    mat_iand_out = wp.array(mat_bits, dtype=mat22i, device=device)
    wp.launch(mat22i_row_iand_atomic_kernel, n, inputs=[row_bits, mat_iand_out], device=device)
    expected_iand = mat_bits.copy()
    expected_iand[0, 1, :] = [0x0F, 0xF0]
    assert_np_equal(mat_iand_out.numpy(), expected_iand)


def test_array_composite_slot_augassign_backward(test, device):
    """Match scalar gradients for composite-slot ``+=`` and ``-=``."""
    for label, kernel, initial_y in (
        ("add", vec2_component_iadd_replay_kernel, 2.0),
        ("sub", vec2_component_isub_replay_kernel, 8.0),
    ):
        with test.subTest(label):
            vec_y = wp.array(
                np.array([[initial_y, 0.0]], dtype=np.float32), dtype=wp.vec2, requires_grad=True, device=device
            )
            scalar_y = wp.array(
                np.array([initial_y], dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device
            )
            vec_x = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device)
            scalar_x = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device)
            vec_out = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
            scalar_out = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)

            tape = wp.Tape()
            with tape:
                wp.launch(
                    kernel,
                    1,
                    inputs=[vec_y, scalar_y, vec_x, scalar_x, vec_out, scalar_out],
                    device=device,
                )

            assert_np_equal(vec_y.numpy()[:, 0], scalar_y.numpy())
            assert_np_equal(vec_out.numpy(), scalar_out.numpy())

            vec_out.grad = wp.ones(1, dtype=wp.float32, device=device)
            scalar_out.grad = wp.ones(1, dtype=wp.float32, device=device)
            tape.backward()

            assert_np_equal(vec_y.numpy()[:, 0], scalar_y.numpy())
            assert_np_equal(vec_x.grad.numpy(), scalar_x.grad.numpy())
            assert_np_equal(vec_y.grad.numpy()[:, 0], scalar_y.grad.numpy())

    for kernel in (
        _k_struct_field_vec_component_iadd_replay,
        _k_struct_field_array_alias_component_iadd_replay,
    ):
        x = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device)
        values = wp.array(
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, requires_grad=True, device=device
        )
        holder = VecArrayFieldHolder()
        holder.values = values
        out = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel, 1, inputs=[holder, x], outputs=[out], device=device)

        assert_np_equal(values.numpy(), np.array([[5.0, 0.0, 0.0]], dtype=np.float32))
        assert_np_equal(out.numpy(), np.array([25.0], dtype=np.float32))

        tape.backward(grads={out: wp.ones(1, dtype=wp.float32, device=device)})

        assert_np_equal(x.grad.numpy(), np.array([10.0], dtype=np.float32))
        assert_np_equal(values.grad.numpy(), np.array([[10.0, 0.0, 0.0]], dtype=np.float32))


def test_array_mat33_row_add_assign_backward(test, device):
    """Verify mat33 row ``+=`` propagates row gradients."""
    n = 2
    src = wp.array(np.tile([1.0, 2.0, 3.0], (n, 1)), dtype=wp.vec3, requires_grad=True, device=device)
    init = np.zeros((n, 3, 3), dtype=np.float32)
    init[:, 1, :] = [5.0, 5.0, 5.0]
    dst = wp.array(init, dtype=wp.mat33, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat33_row_iadd_kernel, n, inputs=[dst, src], device=device)

    # seed adj on dst: row 1 of each mat33 receives [10, 20, 30]
    grad_seed = np.zeros((n, 3, 3), dtype=np.float32)
    grad_seed[:, 1, :] = [10.0, 20.0, 30.0]
    dst.grad = wp.array(grad_seed, dtype=wp.mat33, device=device)
    tape.backward()

    # augassign d/dsrc(y[i][1] += x[i]) = identity, so adj_src = seeded row
    expected = np.tile([10.0, 20.0, 30.0], (n, 1))
    assert_np_equal(src.grad.numpy(), expected)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

add_function_test(
    TestCompositeComponentAdjoint,
    "test_nested_view_component_gradient",
    test_nested_view_component_gradient,
    devices=devices,
)

add_function_test(
    TestCompositeComponentAdjoint,
    "test_inline_slice_updates_preserve_values_and_gradients",
    test_inline_slice_updates_preserve_values_and_gradients,
    devices=devices,
)


add_function_test(
    TestCompositeComponentAdjoint,
    "test_embedded_array_without_outer_gradient",
    test_embedded_array_without_outer_gradient,
    devices=devices,
)


add_function_test(
    TestCompositeComponentAdjoint,
    "test_array_composite_slot_int16_augassign",
    test_array_composite_slot_int16_augassign,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_array_composite_slot_augassign_with_array_indices",
    test_array_composite_slot_augassign_with_array_indices,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint, "test_gh583_array_vec_component", test_gh583_array_vec_component, devices=devices
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_gh248_sequential_component_writes",
    test_gh248_sequential_component_writes,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint, "test_gh1174_array_struct_field", test_gh1174_array_struct_field, devices=devices
)
add_function_test(TestCompositeComponentAdjoint, "test_mat33_element_write", test_mat33_element_write, devices=devices)
add_function_test(
    TestCompositeComponentAdjoint, "test_transform_translation_write", test_transform_translation_write, devices=devices
)
add_function_test(
    TestCompositeComponentAdjoint, "test_struct_vec_field_write", test_struct_vec_field_write, devices=devices
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_struct_field_array_composite_slot_write",
    test_struct_field_array_composite_slot_write,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_composite_slot_type_mismatch_reports_error",
    test_composite_slot_type_mismatch_reports_error,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_wp_adjoint_composite_component_write",
    test_wp_adjoint_composite_component_write,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_wp_adjoint_array_component_accumulates_cuda",
    test_wp_adjoint_array_component_accumulates_cuda,
    devices=cuda_devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_array_mat33_row_write_backward",
    test_array_mat33_row_write_backward,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_array_rooted_composite_slot_gradients",
    test_array_rooted_composite_slot_gradients,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_array_composite_slot_multiply_assign_backward_rejected",
    test_array_composite_slot_multiply_assign_backward_rejected,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_slot_write_rhs_eval_order",
    test_slot_write_rhs_eval_order,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_slot_write_with_array_index",
    test_slot_write_with_array_index,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_slot_write_rejects_non_integer_array_index",
    test_slot_write_rejects_non_integer_array_index,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_contended_array_composite_slot_augassign_cuda",
    test_contended_array_composite_slot_augassign_cuda,
    devices=cuda_devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_array_composite_slot_augassign_backward",
    test_array_composite_slot_augassign_backward,
    devices=devices,
)
add_function_test(
    TestCompositeComponentAdjoint,
    "test_array_mat33_row_add_assign_backward",
    test_array_mat33_row_add_assign_backward,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
