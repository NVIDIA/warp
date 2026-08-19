# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

BLOCK_DIM = 8
ADJ_CLEAR_SYNC_BLOCK_DIM = 64
ADJ_CLEAR_SYNC_WARP_OFFSET = ADJ_CLEAR_SYNC_BLOCK_DIM // 2
# Delay one warp so unsynchronized adjoint clears race with earlier accumulation.
ADJ_CLEAR_SYNC_DELAY_ITERS = 2000


@wp.kernel
def tile_permuted_overwrite_adj_sync_kernel(
    inp: wp.array[wp.float32],
    src: wp.array[wp.float32],
    delay: wp.array[wp.float32],
    out: wp.array[wp.float32],
):
    i = wp.tid()

    base = wp.tile_zeros(shape=ADJ_CLEAR_SYNC_BLOCK_DIM, dtype=wp.float32)
    base[i] = inp[i]

    update = wp.tile_zeros(shape=ADJ_CLEAR_SYNC_BLOCK_DIM, dtype=wp.float32)
    j = (i + ADJ_CLEAR_SYNC_WARP_OFFSET) % ADJ_CLEAR_SYNC_BLOCK_DIM

    scale = float(1.0)
    if i >= ADJ_CLEAR_SYNC_WARP_OFFSET:
        for _ in range(ADJ_CLEAR_SYNC_DELAY_ITERS):
            scale = scale * (float(1.0) + delay[0])

    update[j] = (src[i] - inp[j]) * scale

    wp.tile_atomic_add(out, base, offset=(0,))
    wp.tile_atomic_add(out, update, offset=(0,))


def test_tile_assign_adjoint_clear_sync(test, device):
    """Verify overwritten tile adjoints are cleared after cooperative assignment."""
    inp_np = np.arange(ADJ_CLEAR_SYNC_BLOCK_DIM, dtype=np.float32)
    src_np = np.arange(100, 100 + ADJ_CLEAR_SYNC_BLOCK_DIM, dtype=np.float32)
    expected_out = np.roll(src_np, ADJ_CLEAR_SYNC_WARP_OFFSET)
    expected_inp_grad = np.zeros(ADJ_CLEAR_SYNC_BLOCK_DIM, dtype=np.float32)
    expected_src_grad = np.ones(ADJ_CLEAR_SYNC_BLOCK_DIM, dtype=np.float32)

    for _ in range(5):
        inp = wp.array(inp_np, dtype=wp.float32, requires_grad=True, device=device)
        src = wp.array(src_np, dtype=wp.float32, requires_grad=True, device=device)
        delay = wp.zeros(1, dtype=wp.float32, device=device)
        out = wp.zeros(ADJ_CLEAR_SYNC_BLOCK_DIM, dtype=wp.float32, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(
                tile_permuted_overwrite_adj_sync_kernel,
                dim=ADJ_CLEAR_SYNC_BLOCK_DIM,
                inputs=[inp, src, delay],
                outputs=[out],
                block_dim=ADJ_CLEAR_SYNC_BLOCK_DIM,
                device=device,
            )

        assert_np_equal(out.numpy(), expected_out)

        out.grad = wp.ones(ADJ_CLEAR_SYNC_BLOCK_DIM, dtype=wp.float32, device=device)
        tape.backward()

        assert_np_equal(inp.grad.numpy(), expected_inp_grad)
        assert_np_equal(src.grad.numpy(), expected_src_grad)


@wp.kernel
def tile_mat33_row_read_1d_kernel(
    inp: wp.array[wp.mat33],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    t = wp.tile_load(inp, shape=(BLOCK_DIM,))
    out[i] = t[i][1]


def test_tile_mat33_row_read_1d(test, device):
    """Verify row reads from a 1D tile of matrices."""
    n = BLOCK_DIM
    data = np.tile(np.arange(1.0, 10.0, dtype=np.float32).reshape(3, 3), (n, 1, 1))
    inp = wp.array(data, dtype=wp.mat33, device=device)
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(tile_mat33_row_read_1d_kernel, dim=BLOCK_DIM, inputs=[inp, out], block_dim=BLOCK_DIM, device=device)
    expected = np.tile([4.0, 5.0, 6.0], (n, 1))
    assert_np_equal(out.numpy(), expected)


@wp.kernel
def tile_mat33_negative_row_read_1d_kernel(
    inp: wp.array[wp.mat33],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    t = wp.tile_load(inp, shape=(BLOCK_DIM,))
    out[i] = t[i][-1]


def test_tile_mat33_negative_row_read_1d(test, device):
    """Verify negative row reads from a 1D tile of matrices."""
    n = BLOCK_DIM
    data = np.tile(np.arange(1.0, 10.0, dtype=np.float32).reshape(3, 3), (n, 1, 1))
    inp = wp.array(data, dtype=wp.mat33, device=device)
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(
        tile_mat33_negative_row_read_1d_kernel,
        dim=BLOCK_DIM,
        inputs=[inp, out],
        block_dim=BLOCK_DIM,
        device=device,
    )
    expected = np.tile([7.0, 8.0, 9.0], (n, 1))
    assert_np_equal(out.numpy(), expected)


BLOCK_DIM_2D = 4


@wp.kernel
def tile_mat33_row_read_2d_kernel(
    inp: wp.array2d[wp.mat33],
    out: wp.array2d[wp.vec3],
):
    row, col = wp.tid()
    t = wp.tile_load(inp, shape=(2, 2))
    out[row, col] = t[row, col][2]


def test_tile_mat33_row_read_2d(test, device):
    """Verify row reads from a 2D tile of matrices."""
    rows, cols = 2, 2
    data = np.tile(np.arange(1.0, 10.0, dtype=np.float32).reshape(3, 3), (rows, cols, 1, 1))
    inp = wp.array(data, dtype=wp.mat33, device=device)
    out = wp.zeros((rows, cols), dtype=wp.vec3, device=device)
    wp.launch(
        tile_mat33_row_read_2d_kernel,
        dim=(rows, cols),
        inputs=[inp, out],
        block_dim=BLOCK_DIM_2D,
        device=device,
    )
    expected = np.tile([7.0, 8.0, 9.0], (rows, cols, 1))
    assert_np_equal(out.numpy(), expected)


D0_3D = 2
D1_3D = 2
D2_3D = 2
BLOCK_DIM_3D = D0_3D * D1_3D * D2_3D


@wp.kernel
def tile_mat33_row_read_3d_kernel(
    inp: wp.array3d[wp.mat33],
    out: wp.array3d[wp.vec3],
):
    i, j, k = wp.tid()
    t = wp.tile_load(inp, shape=(D0_3D, D1_3D, D2_3D))
    out[i, j, k] = t[i, j, k][1]


def test_tile_mat33_row_read_3d(test, device):
    """Verify row reads from a 3D tile of matrices."""
    data = np.tile(
        np.arange(1.0, 10.0, dtype=np.float32).reshape(3, 3),
        (D0_3D, D1_3D, D2_3D, 1, 1),
    )
    inp = wp.array(data, dtype=wp.mat33, device=device)
    out = wp.zeros((D0_3D, D1_3D, D2_3D), dtype=wp.vec3, device=device)
    wp.launch(
        tile_mat33_row_read_3d_kernel,
        dim=(D0_3D, D1_3D, D2_3D),
        inputs=[inp, out],
        block_dim=BLOCK_DIM_3D,
        device=device,
    )
    expected = np.tile([4.0, 5.0, 6.0], (D0_3D, D1_3D, D2_3D, 1))
    assert_np_equal(out.numpy(), expected)


D0_4D = 2
D1_4D = 2
D2_4D = 2
D3_4D = 2
BLOCK_DIM_4D = D0_4D * D1_4D * D2_4D * D3_4D


@wp.kernel
def tile_mat22_row_read_4d_kernel(
    inp: wp.array4d[wp.mat22],
    out: wp.array4d[wp.vec2],
):
    i, j, k, l = wp.tid()
    t = wp.tile_load(inp, shape=(D0_4D, D1_4D, D2_4D, D3_4D))
    out[i, j, k, l] = t[i, j, k, l][0]


def test_tile_mat22_row_read_4d(test, device):
    """Verify row reads from a 4D tile of matrices."""
    data = np.tile(
        np.arange(1.0, 5.0, dtype=np.float32).reshape(2, 2),
        (D0_4D, D1_4D, D2_4D, D3_4D, 1, 1),
    )
    inp = wp.array(data, dtype=wp.mat22, device=device)
    out = wp.zeros((D0_4D, D1_4D, D2_4D, D3_4D), dtype=wp.vec2, device=device)
    wp.launch(
        tile_mat22_row_read_4d_kernel,
        dim=(D0_4D, D1_4D, D2_4D, D3_4D),
        inputs=[inp, out],
        block_dim=BLOCK_DIM_4D,
        device=device,
    )
    expected = np.tile([1.0, 2.0], (D0_4D, D1_4D, D2_4D, D3_4D, 1))
    assert_np_equal(out.numpy(), expected)


@wp.kernel
def tile_mat22_row_read_adj_kernel(
    inp: wp.array[wp.mat22],
    out: wp.array[wp.vec2],
):
    i = wp.tid()
    t = wp.tile_load(inp, shape=(BLOCK_DIM,))
    out[i] = t[i][0]


def test_tile_mat22_row_read_backward(test, device):
    """Verify adjoints for matrix row reads from tiles."""
    n = BLOCK_DIM
    init = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)
    inp = wp.array(init, dtype=wp.mat22, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.vec2, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(tile_mat22_row_read_adj_kernel, dim=n, inputs=[inp, out], block_dim=n, device=device)

    out.grad = wp.array(np.tile([1.0, 1.0], (n, 1)), dtype=wp.vec2, device=device)
    tape.backward()

    expected = np.zeros_like(init)
    expected[:, 0, :] = 1.0
    assert_np_equal(inp.grad.numpy(), expected)


@wp.kernel
def tile_mat33_row_write_1d_kernel(
    out: wp.array[wp.mat33],
):
    i = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat33, shape=(BLOCK_DIM,))
    t[i][1] = wp.vec3(10.0, 20.0, 30.0)
    wp.tile_atomic_add(out, t, offset=(0,))


def test_tile_mat33_row_write_1d(test, device):
    """Verify row writes to a 1D tile of matrices."""
    out = wp.zeros(BLOCK_DIM, dtype=wp.mat33, device=device)
    wp.launch(tile_mat33_row_write_1d_kernel, dim=BLOCK_DIM, inputs=[out], block_dim=BLOCK_DIM, device=device)
    expected = np.zeros((BLOCK_DIM, 3, 3), dtype=np.float32)
    expected[:, 1, :] = [10.0, 20.0, 30.0]
    assert_np_equal(out.numpy(), expected)


@wp.kernel
def tile_mat33_negative_row_write_1d_kernel(
    out: wp.array[wp.mat33],
):
    i = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat33, shape=(BLOCK_DIM,))
    t[i][-1] = wp.vec3(10.0, 20.0, 30.0)
    wp.tile_atomic_add(out, t, offset=(0,))


def test_tile_mat33_negative_row_write_1d(test, device):
    """Verify negative row writes to a 1D tile of matrices."""
    out = wp.zeros(BLOCK_DIM, dtype=wp.mat33, device=device)
    wp.launch(
        tile_mat33_negative_row_write_1d_kernel,
        dim=BLOCK_DIM,
        inputs=[out],
        block_dim=BLOCK_DIM,
        device=device,
    )
    expected = np.zeros((BLOCK_DIM, 3, 3), dtype=np.float32)
    expected[:, 2, :] = [10.0, 20.0, 30.0]
    assert_np_equal(out.numpy(), expected)


@wp.kernel
def tile_mat33_row_write_2d_kernel(
    out: wp.array2d[wp.mat33],
):
    row, col = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat33, shape=(2, 2))
    t[row, col][0] = wp.vec3(1.0, 2.0, 3.0)
    wp.tile_atomic_add(out, t, offset=(0, 0))


def test_tile_mat33_row_write_2d(test, device):
    """Verify row writes to a 2D tile of matrices."""
    rows, cols = 2, 2
    out = wp.zeros((rows, cols), dtype=wp.mat33, device=device)
    wp.launch(
        tile_mat33_row_write_2d_kernel,
        dim=(rows, cols),
        inputs=[out],
        block_dim=BLOCK_DIM_2D,
        device=device,
    )
    expected = np.zeros((rows, cols, 3, 3), dtype=np.float32)
    expected[:, :, 0, :] = [1.0, 2.0, 3.0]
    assert_np_equal(out.numpy(), expected)


@wp.kernel
def tile_mat33_row_write_3d_kernel(
    out: wp.array3d[wp.mat33],
):
    i, j, k = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat33, shape=(D0_3D, D1_3D, D2_3D))
    t[i, j, k][2] = wp.vec3(7.0, 8.0, 9.0)
    wp.tile_atomic_add(out, t, offset=(0, 0, 0))


def test_tile_mat33_row_write_3d(test, device):
    """Verify row writes to a 3D tile of matrices."""
    out = wp.zeros((D0_3D, D1_3D, D2_3D), dtype=wp.mat33, device=device)
    wp.launch(
        tile_mat33_row_write_3d_kernel,
        dim=(D0_3D, D1_3D, D2_3D),
        inputs=[out],
        block_dim=BLOCK_DIM_3D,
        device=device,
    )
    expected = np.zeros((D0_3D, D1_3D, D2_3D, 3, 3), dtype=np.float32)
    expected[:, :, :, 2, :] = [7.0, 8.0, 9.0]
    assert_np_equal(out.numpy(), expected)


@wp.kernel
def tile_mat22_row_write_4d_kernel(
    out: wp.array4d[wp.mat22],
):
    i, j, k, l = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat22, shape=(D0_4D, D1_4D, D2_4D, D3_4D))
    t[i, j, k, l][1] = wp.vec2(3.0, 4.0)
    wp.tile_atomic_add(out, t, offset=(0, 0, 0, 0))


def test_tile_mat22_row_write_4d(test, device):
    """Verify row writes to a 4D tile of matrices."""
    out = wp.zeros((D0_4D, D1_4D, D2_4D, D3_4D), dtype=wp.mat22, device=device)
    wp.launch(
        tile_mat22_row_write_4d_kernel,
        dim=(D0_4D, D1_4D, D2_4D, D3_4D),
        inputs=[out],
        block_dim=BLOCK_DIM_4D,
        device=device,
    )
    expected = np.zeros((D0_4D, D1_4D, D2_4D, D3_4D, 2, 2), dtype=np.float32)
    expected[:, :, :, :, 1, :] = [3.0, 4.0]
    assert_np_equal(out.numpy(), expected)


@wp.kernel
def tile_mat22_row_write_adj_kernel(
    src_rows: wp.array[wp.vec2],
    out: wp.array[wp.mat22],
):
    i = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat22, shape=(BLOCK_DIM,))
    t[i][1] = src_rows[i]
    wp.tile_atomic_add(out, t, offset=(0,))


def test_tile_mat22_row_write_backward(test, device):
    """Verify adjoints for matrix row writes to tiles."""
    n = BLOCK_DIM
    src_data = np.ones((n, 2), dtype=np.float32)
    src_rows = wp.array(src_data, dtype=wp.vec2, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(tile_mat22_row_write_adj_kernel, dim=n, inputs=[src_rows, out], block_dim=n, device=device)

    grad_out = np.zeros((n, 2, 2), dtype=np.float32)
    grad_out[:, 1, :] = [2.0, 3.0]
    out.grad = wp.array(grad_out, dtype=wp.mat22, device=device)
    tape.backward()

    expected = np.tile([2.0, 3.0], (n, 1))
    assert_np_equal(src_rows.grad.numpy(), expected)


@wp.kernel
def tile_mat22_row_write_overwrite_adj_kernel(
    inp: wp.array[wp.mat22],
    src_rows: wp.array[wp.vec2],
    out: wp.array[wp.mat22],
):
    i = wp.tid()
    t = wp.tile_load(inp, shape=(BLOCK_DIM,))
    t[i][1] = src_rows[i]
    out[i] = t[i]


def test_tile_mat22_row_write_overwrite_backward(test, device):
    """Verify row-write adjoints zero overwritten matrix rows."""
    n = BLOCK_DIM
    init = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)
    inp = wp.array(init, dtype=wp.mat22, requires_grad=True, device=device)
    src_rows = wp.array(np.ones((n, 2), dtype=np.float32), dtype=wp.vec2, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            tile_mat22_row_write_overwrite_adj_kernel,
            dim=n,
            inputs=[inp, src_rows, out],
            block_dim=n,
            device=device,
        )

    out.grad = wp.array(np.ones((n, 2, 2), dtype=np.float32), dtype=wp.mat22, device=device)
    tape.backward()

    expected_inp_grad = np.ones_like(init)
    expected_inp_grad[:, 1, :] = 0.0
    assert_np_equal(inp.grad.numpy(), expected_inp_grad)

    expected_src_grad = np.ones((n, 2), dtype=np.float32)
    assert_np_equal(src_rows.grad.numpy(), expected_src_grad)


@wp.kernel
def tile_mat22_scalar_write_overwrite_adj_kernel(
    inp: wp.array[wp.mat22],
    src_values: wp.array[wp.float32],
    out: wp.array[wp.mat22],
):
    i = wp.tid()
    t = wp.tile_load(inp, shape=(BLOCK_DIM,))
    t[i][1, 0] = src_values[i]
    out[i] = t[i]


def test_tile_mat22_scalar_write_overwrite_backward(test, device):
    """Verify scalar-write adjoints zero overwritten matrix elements."""
    n = BLOCK_DIM
    init = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)
    inp = wp.array(init, dtype=wp.mat22, requires_grad=True, device=device)
    src_values = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            tile_mat22_scalar_write_overwrite_adj_kernel,
            dim=n,
            inputs=[inp, src_values, out],
            block_dim=n,
            device=device,
        )

    out.grad = wp.array(np.ones((n, 2, 2), dtype=np.float32), dtype=wp.mat22, device=device)
    tape.backward()

    expected_inp_grad = np.ones_like(init)
    expected_inp_grad[:, 1, 0] = 0.0
    assert_np_equal(inp.grad.numpy(), expected_inp_grad)

    expected_src_grad = np.ones(n, dtype=np.float32)
    assert_np_equal(src_values.grad.numpy(), expected_src_grad)


@wp.kernel
def tile_mat22_full_write_overwrite_adj_kernel(
    inp: wp.array[wp.mat22],
    src_mats: wp.array[wp.mat22],
    out: wp.array[wp.mat22],
):
    i = wp.tid()
    t = wp.tile_load(inp, shape=(BLOCK_DIM,))
    t[i] = src_mats[i]
    out[i] = t[i]


def test_tile_mat22_full_write_overwrite_backward(test, device):
    """Verify full-matrix write adjoints zero overwritten matrix values."""
    n = BLOCK_DIM
    init = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)
    inp = wp.array(init, dtype=wp.mat22, requires_grad=True, device=device)
    src_data = np.ones((n, 2, 2), dtype=np.float32)
    src_mats = wp.array(src_data, dtype=wp.mat22, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            tile_mat22_full_write_overwrite_adj_kernel,
            dim=n,
            inputs=[inp, src_mats, out],
            block_dim=n,
            device=device,
        )

    out.grad = wp.array(np.ones((n, 2, 2), dtype=np.float32), dtype=wp.mat22, device=device)
    tape.backward()

    expected_inp_grad = np.zeros_like(init)
    assert_np_equal(inp.grad.numpy(), expected_inp_grad)

    expected_src_grad = np.ones_like(src_data)
    assert_np_equal(src_mats.grad.numpy(), expected_src_grad)


devices = get_test_devices()


class TestTileCompositeRow(unittest.TestCase):
    pass


add_function_test(
    TestTileCompositeRow, "test_tile_assign_adjoint_clear_sync", test_tile_assign_adjoint_clear_sync, devices=devices
)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_read_1d", test_tile_mat33_row_read_1d, devices=devices)
add_function_test(
    TestTileCompositeRow, "test_tile_mat33_negative_row_read_1d", test_tile_mat33_negative_row_read_1d, devices=devices
)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_read_2d", test_tile_mat33_row_read_2d, devices=devices)
add_function_test(
    TestTileCompositeRow, "test_tile_mat22_row_read_backward", test_tile_mat22_row_read_backward, devices=devices
)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_read_3d", test_tile_mat33_row_read_3d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat22_row_read_4d", test_tile_mat22_row_read_4d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_write_1d", test_tile_mat33_row_write_1d, devices=devices)
add_function_test(
    TestTileCompositeRow,
    "test_tile_mat33_negative_row_write_1d",
    test_tile_mat33_negative_row_write_1d,
    devices=devices,
)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_write_2d", test_tile_mat33_row_write_2d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_write_3d", test_tile_mat33_row_write_3d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat22_row_write_4d", test_tile_mat22_row_write_4d, devices=devices)
add_function_test(
    TestTileCompositeRow, "test_tile_mat22_row_write_backward", test_tile_mat22_row_write_backward, devices=devices
)
add_function_test(
    TestTileCompositeRow,
    "test_tile_mat22_row_write_overwrite_backward",
    test_tile_mat22_row_write_overwrite_backward,
    devices=devices,
)
add_function_test(
    TestTileCompositeRow,
    "test_tile_mat22_scalar_write_overwrite_backward",
    test_tile_mat22_scalar_write_overwrite_backward,
    devices=devices,
)
add_function_test(
    TestTileCompositeRow,
    "test_tile_mat22_full_write_overwrite_backward",
    test_tile_mat22_full_write_overwrite_backward,
    devices=devices,
)
if __name__ == "__main__":
    unittest.main(verbosity=2)
