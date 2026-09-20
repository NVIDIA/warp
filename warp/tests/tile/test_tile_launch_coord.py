# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for tiled CUDA launch coordinate reconstruction (#1361).

``wp.launch_tiled`` appends the block dimension as a trailing launch extent, which
``_build_launch_bounds_from_tuple`` folds into ``launch_bounds_t.coord_mult``. The
generated CUDA reaches ``wp.tid()`` through ``wp::launch_coord_tile``, which recovers
the coord from the block-derived tile index instead of dividing the linear thread
index back down.

The expected coord is stated from the launch shape alone, independently of Warp's own
unraveling, so a fast path that aliases or drops a logical tile fails here even when
the tile math downstream stays self-consistent.

The CPU device has no ``blockIdx`` and keeps the original path, so the same kernels are
also asserted there as the non-folded reference.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Tile extents; the 2D cases below deliberately use a shape[1] (nj) much smaller than
# the folded coord_mult so a lane added to coord.j would overflow it.
TILE_M = 8
TILE_N = 4

# Block dims exercised for the folded-axis path. Tile counts are chosen so the tile
# grid never divides the block grid evenly, which is where an off-by-one in the coord
# reconstruction would show up.
BLOCK_DIMS = [32, 64, 128, 256]


@wp.kernel
def tid1d_kernel(out: wp.array[int]):
    i = wp.tid()
    out[i] = i


@wp.kernel
def tid2d_kernel(nj: int, out: wp.array[int]):
    i, j = wp.tid()
    out[i * nj + j] = i * nj + j


@wp.kernel
def tid3d_kernel(nj: int, nk: int, out: wp.array[int]):
    i, j, k = wp.tid()
    out[(i * nj + j) * nk + k] = (i * nj + j) * nk + k


@wp.kernel
def tile2d_binary_map_kernel(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
    # Two-input 2D tile map, mirroring the tile suite's binary-map test. Distinct inputs
    # and a sentinel output make a coord that aliases tile rows show up as a value
    # mismatch rather than as an index assertion.
    i, j = wp.tid()
    ta = wp.tile_load(a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    tb = wp.tile_load(b, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(out, wp.tile_map(wp.add, ta, tb), offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def tile2d_writer_kernel(a: wp.array2d[float], out: wp.array2d[float]):
    # Every thread of a tile stores the same tile, so the primal is race free by
    # construction and a coord that aliases tiles leaves sentinel values behind. All
    # threads also run the store's adjoint, which is why the gradient below is asserted
    # for uniformity across tiles rather than for a fixed constant.
    i, j = wp.tid()
    ta = wp.tile_load(a, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(out, ta, offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def tid_early_return_kernel(limit: int, out: wp.array[int]):
    i = wp.tid()
    if i < limit:
        out[i] = i


def expected_1d(ntiles):
    """One logical tile per block; the coord equals the linear tile index."""
    return np.arange(ntiles, dtype=np.int32)


def tiled(kernel, args, dim, block_dim, device, **kwargs):
    wp.launch_tiled(kernel, dim=dim, inputs=args, block_dim=block_dim, device=device, **kwargs)


def plain(kernel, args, dim, block_dim, device, **kwargs):
    wp.launch(kernel, dim=dim, inputs=args, block_dim=block_dim, device=device, **kwargs)


def test_folded_axis_matches_tile_index(test, device, block_dim):
    """1D tiled launch: coord == block index, for every tile count."""
    for ntiles in (1, 7, 33, 513):
        out = wp.zeros(ntiles, dtype=int, device=device)
        tiled(tid1d_kernel, [out], [ntiles], block_dim, device)
        assert_np_equal(out.numpy(), expected_1d(ntiles))  # f"ntiles={ntiles} block_dim={block_dim}"


def test_unfolded_axis_is_unchanged(test, device, block_dim):
    """The same kernel launched without a folded axis still covers every thread."""
    for ntiles in (1, 7, 33, 513):
        out = wp.zeros(ntiles, dtype=int, device=device)
        plain(tid1d_kernel, [out], [ntiles], block_dim, device)
        assert_np_equal(out.numpy(), expected_1d(ntiles))  # f"ntiles={ntiles} block_dim={block_dim}"


def test_2d_tile_grid(test, device, block_dim):
    """2D tiled launch: the coord matches the rectangular tile grid exactly."""
    for ni, nj in ((7, 11), (3, 5), (1, 9), (16, 1), (13, 7)):
        out = wp.zeros(ni * nj, dtype=int, device=device)
        tiled(tid2d_kernel, [nj, out], [ni, nj], block_dim, device)
        assert_np_equal(out.numpy(), np.arange(ni * nj, dtype=np.int32))  # f"grid={ni}x{nj} block_dim={block_dim}"


def test_2d_tile_values_do_not_alias_rows(test, device, block_dim):
    """Every element of a 2D tiled grid must carry its own tile's data.

    This is the case a coord that conflates ``coord_mult`` with ``shape[1]`` breaks:
    ``coord.j`` runs past ``shape[1]``, sibling tiles alias, and most output elements
    keep whatever the buffer was initialized to.
    """
    for ni, nj in ((7, 5), (3, 5), (4, 4), (1, 9)):
        M, N = ni * TILE_M, nj * TILE_N
        rng = np.random.default_rng(7)
        a_np = rng.random((M, N), dtype=np.float32) + 0.5
        b_np = rng.random((M, N), dtype=np.float32) + 0.5
        a = wp.array(a_np, dtype=float, device=device)
        b = wp.array(b_np, dtype=float, device=device)
        # sentinel: an aliased or never-written tile keeps this value
        out = wp.full((M, N), -12345.0, dtype=float, device=device)

        wp.launch_tiled(
            tile2d_binary_map_kernel,
            dim=[ni, nj],
            inputs=[a, b, out],
            block_dim=block_dim,
            device=device,
        )
        expected = (a_np + b_np).astype(np.float32)
        got = out.numpy()
        assert_np_equal(got, expected, tol=1e-5)


def test_2d_tile_gradients(test, device, block_dim):
    """Forward values and the adjoint both have to cover every 2D tile."""
    ni, nj = 3, 5
    M, N = ni * TILE_M, nj * TILE_N
    rng = np.random.default_rng(11)
    a_np = rng.random((M, N), dtype=np.float32)
    a = wp.array(a_np, dtype=float, device=device, requires_grad=True)
    # sentinel: a tile the launch never reaches keeps this value
    out = wp.full((M, N), -7.0, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(tile2d_writer_kernel, dim=[ni, nj], inputs=[a, out], block_dim=block_dim, device=device)

    assert_np_equal(out.numpy(), a_np)
    out.grad = wp.ones_like(out, device=device)
    tape.backward()

    grad = a.grad.numpy()
    assert np.all(grad > 0.0), "some tiles received no gradient"
    assert_np_equal(grad, np.full((M, N), grad[0, 0], dtype=np.float32))


def test_3d_tile_grid(test, device, block_dim):
    """3D tiled launch stays on the linear path and must not regress."""
    ni, nj, nk = 3, 5, 7
    out = wp.zeros(ni * nj * nk, dtype=int, device=device)
    tiled(tid3d_kernel, [nj, nk, out], [ni, nj, nk], block_dim, device)
    assert_np_equal(out.numpy(), np.arange(ni * nj * nk, dtype=np.int32))


def test_grid_stride_visits_every_tile_once(test, device, block_dim):
    """Capping the block count must not skip or duplicate a tile."""
    for ntiles, max_blocks in ((16, 1), (16, 2), (33, 7), (200, 3), (512, 5)):
        out = wp.full(ntiles, -1, dtype=int, device=device)
        tiled(tid1d_kernel, [out], [ntiles], block_dim, device, max_blocks=max_blocks)
        assert_np_equal(out.numpy(), expected_1d(ntiles))  # f"ntiles={ntiles} max_blocks={max_blocks}"


def test_grid_stride_early_return(test, device, block_dim):
    """A branch in the loop body must not shorten the remaining grid-stride tiles."""
    ntiles = 64
    limit = ntiles // 2
    expected = np.full(ntiles, -1, dtype=np.int32)
    expected[:limit] = np.arange(limit, dtype=np.int32)
    for max_blocks in (1, 3, 8):
        out = wp.full(ntiles, -1, dtype=int, device=device)
        tiled(tid_early_return_kernel, [limit, out], [ntiles], block_dim, device, max_blocks=max_blocks)
        assert_np_equal(out.numpy(), expected)  # f"max_blocks={max_blocks}"


def test_tiled_equals_explicit_block_dim_axis(test, device, block_dim):
    """A tiled launch equals the explicit dim=[ntiles, block_dim] launch."""
    ntiles = 37
    folded = wp.zeros(ntiles, dtype=int, device=device)
    explicit = wp.zeros(ntiles, dtype=int, device=device)
    tiled(tid1d_kernel, [folded], [ntiles], block_dim, device)
    wp.launch(tid1d_kernel, dim=[ntiles, block_dim], inputs=[explicit], block_dim=block_dim, device=device)
    assert_np_equal(folded.numpy(), explicit.numpy())


def test_tiled_and_plain_launch_share_a_kernel(test, device, block_dim):
    """One kernel used both ways must not have one launch form poison the other."""
    ntiles = 41
    for _ in range(2):
        folded = wp.zeros(ntiles, dtype=int, device=device)
        unfolded = wp.zeros(ntiles, dtype=int, device=device)
        tiled(tid1d_kernel, [folded], [ntiles], block_dim, device)
        plain(tid1d_kernel, [unfolded], [ntiles], block_dim, device)
        assert_np_equal(folded.numpy(), expected_1d(ntiles))
        assert_np_equal(unfolded.numpy(), expected_1d(ntiles))


def test_tiled_launch_inside_cuda_graph(test, device, block_dim):
    """Graph capture/replay must not change the coord mapping."""
    if not wp.get_device(device).is_cuda:
        return
    ntiles = 64
    out = wp.zeros(ntiles, dtype=int, device=device)

    def body():
        wp.launch_tiled(tid1d_kernel, dim=[ntiles], inputs=[out], block_dim=block_dim, device=device)

    wp.synchronize()
    with wp.ScopedCapture(device=device) as capture:
        body()
    out.zero_()
    wp.synchronize()
    wp.capture_launch(capture.graph)
    wp.synchronize()
    assert_np_equal(out.numpy(), expected_1d(ntiles))


devices = get_test_devices()


class TestTileLaunchCoord(unittest.TestCase):
    pass


for _block_dim in BLOCK_DIMS:
    for _name, _fn in (
        ("test_folded_axis_matches_tile_index", test_folded_axis_matches_tile_index),
        ("test_unfolded_axis_is_unchanged", test_unfolded_axis_is_unchanged),
        ("test_2d_tile_grid", test_2d_tile_grid),
        ("test_2d_tile_values_do_not_alias_rows", test_2d_tile_values_do_not_alias_rows),
        ("test_2d_tile_gradients", test_2d_tile_gradients),
        ("test_3d_tile_grid", test_3d_tile_grid),
        ("test_grid_stride_visits_every_tile_once", test_grid_stride_visits_every_tile_once),
        ("test_grid_stride_early_return", test_grid_stride_early_return),
        ("test_tiled_equals_explicit_block_dim_axis", test_tiled_equals_explicit_block_dim_axis),
        ("test_tiled_and_plain_launch_share_a_kernel", test_tiled_and_plain_launch_share_a_kernel),
        ("test_tiled_launch_inside_cuda_graph", test_tiled_launch_inside_cuda_graph),
    ):
        add_function_test(TestTileLaunchCoord, f"{_name}_bd{_block_dim}", _fn, devices=devices, block_dim=_block_dim)


if __name__ == "__main__":
    unittest.main(verbosity=2)
