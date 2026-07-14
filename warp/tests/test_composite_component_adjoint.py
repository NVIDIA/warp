# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for adjoints of in-place composite-component writes.

Covers the shapes the v2 lowering currently handles — array-rooted writes
where the access chain from ``arr[i]`` lands on a composite slot, single
assignment (``=``). Augmented assignments, nested struct chains, and
local-struct composite-field writes are tracked as follow-ups.

See ``asv/benchmarks/codegen/composite_component_*.py`` for the
performance contract that these tests complement.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.struct
class Scalar:
    a: wp.float32
    b: wp.float32


@wp.struct
class StateStruct:
    position: wp.vec3
    velocity: wp.vec3


@wp.kernel
def _seed_scalar_grad(g: wp.array[Scalar], a_val: wp.float32, b_val: wp.float32):
    i = wp.tid()
    g[i] = Scalar(a_val, b_val)


@wp.kernel
def _seed_state_grad(g: wp.array[StateStruct], pos: wp.vec3, vel: wp.vec3):
    i = wp.tid()
    s = StateStruct()
    s.position = pos
    s.velocity = vel
    g[i] = s


# --------- kernels under test ---------


@wp.kernel
def _k_gh583(dst: wp.array[wp.vec3], src: wp.array[wp.float32]):
    i = wp.tid()
    dst[i].y = src[i]


@wp.kernel
def _k_gh248(y: wp.array[wp.vec3], x: wp.array[wp.float32]):
    tid = wp.tid()
    y[tid].x = x[tid] * 2.0
    y[tid].y = x[tid] * 3.0
    y[tid].z = x[tid] * 4.0


@wp.kernel
def _k_gh1174(y: wp.array[Scalar], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i].a = x[i]


@wp.kernel
def _k_vec2_component(y: wp.array[wp.vec2], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i].y = x[i]


@wp.kernel
def _k_vec4_component(y: wp.array[wp.vec4], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i].z = x[i]


@wp.kernel
def _k_quat_component(y: wp.array[wp.quatf], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i].x = x[i]


@wp.kernel
def _k_mat22_element(y: wp.array[wp.mat22], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i][0, 1] = x[i]


@wp.kernel
def _k_mat33_element(y: wp.array[wp.mat33], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i][1, 2] = x[i]


@wp.kernel
def _k_transform_p(y: wp.array[wp.transformf], p: wp.array[wp.vec3]):
    i = wp.tid()
    y[i].p = p[i]


@wp.kernel
def _k_transform_q(y: wp.array[wp.transformf], q: wp.array[wp.quatf]):
    i = wp.tid()
    y[i].q = q[i]


@wp.kernel
def _k_struct_vec_field(y: wp.array[StateStruct], p: wp.array[wp.vec3]):
    i = wp.tid()
    y[i].position = p[i]


@wp.kernel
def _k_vec3_subscript(y: wp.array[wp.vec3], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i][1] = x[i]


@wp.kernel
def _k_quat_subscript(y: wp.array[wp.quatf], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i][0] = x[i]


@wp.struct
class Inner:
    a: wp.float32
    b: wp.float32


@wp.struct
class Outer:
    inner: Inner
    tag: wp.float32


@wp.kernel
def _k_nested_struct_scalar(out: wp.array[Outer], src: wp.array[wp.float32]):
    i = wp.tid()
    out[i].inner.a = src[i]


@wp.struct
class MatHolder:
    m: wp.mat33
    tag: wp.float32


@wp.kernel
def _k_struct_mat_elem(out: wp.array[MatHolder], src: wp.array[wp.float32]):
    i = wp.tid()
    out[i].m[1, 2] = src[i]


@wp.kernel
def _k_struct_vec_component(out: wp.array[StateStruct], src: wp.array[wp.float32]):
    i = wp.tid()
    out[i].position.y = src[i]


@wp.kernel
def _seed_outer_grad(g: wp.array[Outer], a_val: wp.float32):
    i = wp.tid()
    t = Outer()
    t.inner = Inner(a_val, 0.0)
    t.tag = 0.0
    g[i] = t


@wp.kernel
def _seed_matholder_grad(g: wp.array[MatHolder]):
    i = wp.tid()
    t = MatHolder()
    t.m = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    t.tag = 0.0
    g[i] = t


@wp.kernel
def _k_array2d(y: wp.array2d[wp.vec3], x: wp.array2d[wp.float32]):
    i, j = wp.tid()
    y[i, j].y = x[i, j]


@wp.kernel
def _k_array3d(y: wp.array3d[wp.vec3], x: wp.array3d[wp.float32]):
    i, j, k = wp.tid()
    y[i, j, k].z = x[i, j, k]


@wp.struct
class ArrayFieldHolder:
    v: wp.array[wp.float32]


@wp.kernel
def _k_struct_array_field(holder: ArrayFieldHolder, x: wp.array[wp.float32]):
    i = wp.tid()
    holder.v[i] = x[i] * 2.0


@wp.kernel
def _k_indexedarray_component(y: wp.indexedarray[wp.vec3], x: wp.array[wp.float32]):
    i = wp.tid()
    y[i].y = x[i]


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


# --------- tests ---------


class TestCompositeComponentAdjoint(unittest.TestCase):
    def test_gh583_array_vec_component(self):
        n = 4
        src = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        dst = wp.zeros(n, dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_gh583, n, inputs=[dst, src])

        dst.grad = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        assert_np_equal(src.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_gh248_sequential_component_writes(self):
        n = 3
        x = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_gh248, n, inputs=[y, x])

        y.grad = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.full(n, 9.0, dtype=np.float32))

    def test_gh1174_array_struct_field(self):
        n = 3
        x = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=Scalar, requires_grad=True)

        wp.launch(_seed_scalar_grad, n, inputs=[y.grad, 1.0, 0.0])

        tape = wp.Tape()
        with tape:
            wp.launch(_k_gh1174, n, inputs=[y, x])
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_vec2_component_write(self):
        n = 2
        x = wp.array(np.full(n, 2.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.vec2, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_vec2_component, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 2), dtype=np.float32), dtype=wp.vec2)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_vec4_component_write(self):
        n = 2
        x = wp.array(np.full(n, 1.5, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.vec4, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_vec4_component, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 4), dtype=np.float32), dtype=wp.vec4)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_quaternion_component_write(self):
        n = 2
        x = wp.array(np.full(n, 0.5, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.quatf, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_quat_component, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 4), dtype=np.float32), dtype=wp.quatf)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_mat22_element_write(self):
        n = 2
        x = wp.array(np.full(n, 3.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.mat22, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_mat22_element, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 2, 2), dtype=np.float32), dtype=wp.mat22)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_mat33_element_write(self):
        n = 2
        x = wp.array(np.full(n, 7.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.mat33, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_mat33_element, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 3, 3), dtype=np.float32), dtype=wp.mat33)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_transform_translation_write(self):
        n = 2
        p = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, requires_grad=True)
        y = wp.zeros(n, dtype=wp.transformf, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_transform_p, n, inputs=[y, p])
        y.grad = wp.array(np.ones((n, 7), dtype=np.float32), dtype=wp.transformf)
        tape.backward()

        assert_np_equal(p.grad.numpy(), np.ones((n, 3), dtype=np.float32))

    def test_transform_rotation_write(self):
        n = 2
        q = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32), dtype=wp.quatf, requires_grad=True)
        y = wp.zeros(n, dtype=wp.transformf, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_transform_q, n, inputs=[y, q])
        y.grad = wp.array(np.ones((n, 7), dtype=np.float32), dtype=wp.transformf)
        tape.backward()

        assert_np_equal(q.grad.numpy(), np.ones((n, 4), dtype=np.float32))

    def test_struct_vec_field_write(self):
        n = 2
        p = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, requires_grad=True)
        y = wp.zeros(n, dtype=StateStruct, requires_grad=True)

        wp.launch(_seed_state_grad, n, inputs=[y.grad, wp.vec3(1.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0)])

        tape = wp.Tape()
        with tape:
            wp.launch(_k_struct_vec_field, n, inputs=[y, p])
        tape.backward()

        assert_np_equal(p.grad.numpy(), np.ones((n, 3), dtype=np.float32))

    def test_array2d_composite_component(self):
        rows, cols = 2, 3
        x = wp.array(np.ones((rows, cols), dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros((rows, cols), dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_array2d, (rows, cols), inputs=[y, x])
        y.grad = wp.array(np.ones((rows, cols, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones((rows, cols), dtype=np.float32))

    def test_vec3_subscript_write(self):
        """``arr[i][1] = rhs`` — vec3 scalar subscript (exercises the
        ``[k]`` access path rather than ``.y`` attribute)."""
        n = 3
        x = wp.array(np.full(n, 4.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_vec3_subscript, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_quaternion_subscript_write(self):
        """``arr[i][0] = rhs`` — quat scalar subscript via ``operator[]``."""
        n = 2
        x = wp.array(np.full(n, 0.25, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.quatf, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_quat_subscript, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 4), dtype=np.float32), dtype=wp.quatf)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_nested_struct_scalar_field(self):
        """``arr[i].inner.a = rhs`` — two-level struct chain terminating
        in a scalar field."""
        n = 3
        src = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        out = wp.zeros(n, dtype=Outer, requires_grad=True)
        wp.launch(_seed_outer_grad, n, inputs=[out.grad, 1.0])

        tape = wp.Tape()
        with tape:
            wp.launch(_k_nested_struct_scalar, n, inputs=[out, src])
        tape.backward()

        assert_np_equal(src.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_struct_mat_element_write(self):
        """``arr[i].m[r, c] = rhs`` — struct field descending into a
        matrix element (composite-valued field crossed into, terminating
        in a scalar slot)."""
        n = 2
        src = wp.array(np.full(n, 7.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        out = wp.zeros(n, dtype=MatHolder, requires_grad=True)
        wp.launch(_seed_matholder_grad, n, inputs=[out.grad])

        tape = wp.Tape()
        with tape:
            wp.launch(_k_struct_mat_elem, n, inputs=[out, src])
        tape.backward()

        assert_np_equal(src.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_struct_vec_component_write(self):
        """``arr[i].position.y = rhs`` — struct field descending into a
        vec3 component."""
        n = 3
        src = wp.array(np.full(n, 2.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        out = wp.zeros(n, dtype=StateStruct, requires_grad=True)
        # Seed upstream adj_out[i].position.y = 1.
        wp.launch(_seed_state_grad, n, inputs=[out.grad, wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 0.0)])

        tape = wp.Tape()
        with tape:
            wp.launch(_k_struct_vec_component, n, inputs=[out, src])
        tape.backward()

        assert_np_equal(src.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_array3d_composite_component(self):
        a, b, c = 2, 2, 2
        x = wp.array(np.ones((a, b, c), dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros((a, b, c), dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_array3d, (a, b, c), inputs=[y, x])
        y.grad = wp.array(np.ones((a, b, c, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        assert_np_equal(x.grad.numpy(), np.ones((a, b, c), dtype=np.float32))

    def test_struct_array_field_write(self):
        # Struct-held arrays should still compile and write through the regular
        # array assignment path.
        n = 4
        x = wp.array(np.full(n, 3.0, dtype=np.float32), dtype=wp.float32)
        v = wp.zeros(n, dtype=wp.float32)
        holder = ArrayFieldHolder()
        holder.v = v

        wp.launch(_k_struct_array_field, n, inputs=[holder, x])
        wp.synchronize_device()

        assert_np_equal(v.numpy(), np.full(n, 6.0, dtype=np.float32))

    def test_indexedarray_composite_component_write(self):
        # Indexed arrays should keep the generic lowering and preserve forward
        # write behavior.
        n = 4
        base = wp.zeros(n, dtype=wp.vec3, requires_grad=True)
        indices = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32)
        y = wp.indexedarray(base, [indices])
        x = wp.array(np.full(n, 5.0, dtype=np.float32), dtype=wp.float32)

        wp.launch(_k_indexedarray_component, n, inputs=[y, x])
        wp.synchronize_device()

        expected = np.zeros((n, 3), dtype=np.float32)
        expected[:, 1] = 5.0
        assert_np_equal(base.numpy(), expected)

    def test_slot_type_mismatch_reference_rhs_reports_error(self):
        # Assigning a vector reference into a scalar component should report the
        # normal type mismatch and emit no slot-store call.
        @wp.kernel
        def _k_mismatch(y: wp.array[wp.vec3], src: wp.array[wp.vec3]):
            i = wp.tid()
            y[i].y = src[i]  # scalar slot (.y) <- vec3 reference rhs: type mismatch

        # Match the message so an unrelated runtime failure cannot satisfy this assertion.
        with self.assertRaisesRegex(RuntimeError, "must be of the same type as the reference"):
            _k_mismatch.adj.build(builder=None)

        self.assertTrue(_k_mismatch.adj.blocks, "Expected build() to initialize codegen blocks before failing.")
        forward = "\n".join(_k_mismatch.adj.blocks[0].body_forward)
        reverse = "\n".join(_k_mismatch.adj.blocks[0].body_reverse)
        self.assertNotIn("adj_array_store_slot", forward + reverse)

    def test_wp_adjoint_composite_component_write(self):
        # Plain component writes to ``wp.adjoint`` inside a custom grad should
        # behave like normal adjoint stores.
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
