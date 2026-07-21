# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def pnoise(kernel_seed: int, W: int, px: int, py: int, noise_values: wp.array[float], pixel_values: wp.array[float]):
    tid = wp.tid()

    state = wp.rand_init(kernel_seed)

    x = (float(tid % W) + 0.5) * 0.2
    y = (float(tid / W) + 0.5) * 0.2
    p = wp.vec2(x, y)

    n = wp.pnoise(state, p, px, py)
    noise_values[tid] = n

    g = ((n + 1.0) / 2.0) * 255.0
    pixel_values[tid] = g


@wp.kernel
def curlnoise(kernel_seed: int, W: int, noise_coords: wp.array[wp.vec2], noise_vectors: wp.array[wp.vec2]):
    tid = wp.tid()

    state = wp.rand_init(kernel_seed)

    x = (float(tid % W) + 0.5) * 0.2
    y = (float(tid / W) + 0.5) * 0.2

    p = wp.vec2(x, y)
    v = wp.curlnoise(state, p)

    noise_coords[tid] = p
    noise_vectors[tid] = v


def test_pnoise(test, device):
    # image dim
    W = 256
    H = 256
    N = W * H
    seed = 42

    # periodic perlin noise test
    px = 16
    py = 16

    noise_values = wp.zeros(N, dtype=float, device=device)
    pixel_values = wp.zeros(N, dtype=float, device=device)

    wp.launch(kernel=pnoise, dim=N, inputs=[seed, W, px, py, noise_values, pixel_values], outputs=[], device=device)

    # Perlin theoretical range is [-0.5*sqrt(n), 0.5*sqrt(n)] for n dimensions
    n = noise_values.numpy()
    # max = np.max(n)
    # min = np.min(n)

    img = pixel_values.numpy()
    img = np.reshape(img, (W, H))

    ### Figure viewing ###
    # img = img.astype(np.uint8)
    # imgplot = plt.imshow(img, 'gray')
    # plt.savefig("pnoise_test.png")

    ### Generating pnoise_test_result_true.npy ###
    # np.save(os.path.join(os.path.dirname(__file__), "assets/pnoise_golden.npy"), img)

    ### Golden image comparison ###
    img_true = np.load(os.path.join(os.path.dirname(__file__), "assets/pnoise_golden.npy"))
    test.assertTrue(img.shape == img_true.shape)
    err = np.max(np.abs(img - img_true))
    tolerance = 1.5e-3
    test.assertTrue(err < tolerance, f"err is {err} which is >= {tolerance}")


def test_curlnoise(test, device):
    # image dim
    W = 128
    H = 128
    N = W * H
    seed = 42

    # curl noise test
    quiver_coords_host = wp.zeros(N, dtype=wp.vec2, device="cpu")
    quiver_coords = wp.zeros(N, dtype=wp.vec2, device=device)

    quiver_arrows_host = wp.zeros(N, dtype=wp.vec2, device="cpu")
    quiver_arrows = wp.zeros(N, dtype=wp.vec2, device=device)

    wp.launch(kernel=curlnoise, dim=N, inputs=[seed, W, quiver_coords, quiver_arrows], outputs=[], device=device)

    wp.copy(quiver_coords_host, quiver_coords)
    wp.copy(quiver_arrows_host, quiver_arrows)

    wp.synchronize()

    xy_coords = quiver_coords_host.numpy()
    uv_coords = quiver_arrows_host.numpy()

    # normalize
    norms = uv_coords[:, 0] * uv_coords[:, 0] + uv_coords[:, 1] * uv_coords[:, 1]
    uv_coords = uv_coords / np.sqrt(np.max(norms))

    X = xy_coords[:, 0]
    Y = xy_coords[:, 1]
    U = uv_coords[:, 0]
    V = uv_coords[:, 1]

    ### Figure viewing ###
    # fig, ax = plt.subplots(figsize=(25,25))
    # ax.quiver(X, Y, U, V)
    # ax.axis([0.0, 25.0, 0.0, 25.0])
    # ax.set_aspect('equal')
    # plt.savefig("curlnoise_test.png")

    ### Generating curlnoise_test_result_true.npy ###
    result = np.stack((xy_coords, uv_coords))
    # np.save(os.path.join(os.path.dirname(__file__), "assets/curlnoise_golden.npy"), result)

    ### Golden image comparison ###
    result_true = np.load(os.path.join(os.path.dirname(__file__), "assets/curlnoise_golden.npy"))
    test.assertTrue(result.shape, result_true.shape)
    err = np.max(np.abs(result - result_true))
    test.assertTrue(err < 1e-04)


@wp.kernel
def noise_loss_kernel(
    kernel_seed: int,
    query_positions: wp.array[wp.vec2],
    noise_values: wp.array[float],
    noise_loss: wp.array[float],
):
    tid = wp.tid()
    state = wp.rand_init(kernel_seed)

    p = query_positions[tid]

    n = wp.noise(state, p)
    noise_values[tid] = n

    wp.atomic_add(noise_loss, 0, n)


@wp.kernel
def noise_cd(kernel_seed: int, query_positions: wp.array[wp.vec2], gradients: wp.array[wp.vec2]):
    tid = wp.tid()
    state = wp.rand_init(kernel_seed)
    p = query_positions[tid]

    eps = 1.0e-3

    pl = wp.vec2(p[0] - eps, p[1])
    pr = wp.vec2(p[0] + eps, p[1])
    pd = wp.vec2(p[0], p[1] - eps)
    pu = wp.vec2(p[0], p[1] + eps)

    nl = wp.noise(state, pl)
    nr = wp.noise(state, pr)
    nd = wp.noise(state, pd)
    nu = wp.noise(state, pu)

    gx = (nr - nl) / (2.0 * eps)
    gy = (nu - nd) / (2.0 * eps)

    gradients[tid] = wp.vec2(gx, gy)


def test_adj_noise(test, device):
    # grid dim
    N = 9
    seed = 42

    tape = wp.Tape()

    positions = np.array(
        [
            [-0.1, -0.1],
            [0.0, -0.1],
            [0.1, -0.1],
            [-0.1, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
            [-0.1, 0.1],
            [0.0, 0.1],
            [0.1, 0.1],
        ]
    )

    with tape:
        query_positions = wp.array(positions, dtype=wp.vec2, device=device, requires_grad=True)
        noise_values = wp.zeros(N, dtype=float, device=device)
        noise_loss = wp.zeros(n=1, dtype=float, device=device, requires_grad=True)

        wp.launch(
            kernel=noise_loss_kernel, dim=N, inputs=[seed, query_positions, noise_values, noise_loss], device=device
        )

    # analytic
    tape.backward(loss=noise_loss)
    analytic = tape.gradients[query_positions].numpy().reshape((3, 3, 2))

    # central difference
    gradients = wp.zeros(N, dtype=wp.vec2, device=device)
    wp.launch(kernel=noise_cd, dim=N, inputs=[seed, query_positions, gradients], device=device)

    gradients_host = gradients.numpy().reshape((3, 3, 2))
    diff = analytic - gradients_host
    result = np.sum(diff * diff, axis=2)

    err = np.where(result > 1.0e-3, result, 0).sum()
    test.assertTrue(err < 1.0e-8)


# -------------- curlnoise adjoint tests --------------
#
# These verify the analytic ``adj_curlnoise`` implementations against
# central-difference approximations of the same forward pass. The forward
# is untouched (``wp.curlnoise`` as shipped); only the reverse is under
# test. Mirrors the existing ``test_adj_noise`` pattern.
#
# Test positions are chosen away from integer cell boundaries at every
# octave (pt_k = freq_k * xy avoids integer multiples), so the central
# difference doesn't straddle a cell and introduce discretisation error.


@wp.kernel
def curlnoise_2d_loss(
    seed: int,
    xy: wp.array[wp.vec2],
    octaves: int,
    lac: float,
    gain: float,
    w: wp.vec2,
    loss: wp.array[float],
):
    tid = wp.tid()
    state = wp.rand_init(seed)
    v = wp.curlnoise(state, xy[tid], wp.uint32(octaves), lac, gain)
    wp.atomic_add(loss, 0, v[0] * w[0] + v[1] * w[1])


@wp.kernel
def curlnoise_2d_cd(
    seed: int,
    xy: wp.array[wp.vec2],
    octaves: int,
    lac: float,
    gain: float,
    w: wp.vec2,
    eps: float,
    grad: wp.array[wp.vec2],
):
    tid = wp.tid()
    state = wp.rand_init(seed)
    p = xy[tid]
    p_xp = wp.vec2(p[0] + eps, p[1])
    p_xm = wp.vec2(p[0] - eps, p[1])
    p_yp = wp.vec2(p[0], p[1] + eps)
    p_ym = wp.vec2(p[0], p[1] - eps)
    v_xp = wp.curlnoise(state, p_xp, wp.uint32(octaves), lac, gain)
    v_xm = wp.curlnoise(state, p_xm, wp.uint32(octaves), lac, gain)
    v_yp = wp.curlnoise(state, p_yp, wp.uint32(octaves), lac, gain)
    v_ym = wp.curlnoise(state, p_ym, wp.uint32(octaves), lac, gain)
    L_xp = v_xp[0] * w[0] + v_xp[1] * w[1]
    L_xm = v_xm[0] * w[0] + v_xm[1] * w[1]
    L_yp = v_yp[0] * w[0] + v_yp[1] * w[1]
    L_ym = v_ym[0] * w[0] + v_ym[1] * w[1]
    grad[tid] = wp.vec2((L_xp - L_xm) / (2.0 * eps), (L_yp - L_ym) / (2.0 * eps))


def test_adj_curlnoise_2d(test, device):
    seed = 42
    octaves = 3
    lac = 2.0
    gain = 0.5
    w = wp.vec2(1.0, -0.5)
    # Positions with irrational-looking decimals to avoid cell boundaries
    # at every octave multiplier.
    positions = np.array(
        [[0.17, 0.23], [1.33, 2.77], [-0.43, 0.61], [3.19, -2.11]],
        dtype=np.float32,
    )
    n = len(positions)

    tape = wp.Tape()
    with tape:
        xy = wp.array(positions, dtype=wp.vec2, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        wp.launch(curlnoise_2d_loss, n, inputs=[seed, xy, octaves, lac, gain, w, loss], device=device)
    tape.backward(loss=loss)
    analytic = xy.grad.numpy()

    fd = wp.zeros(n, dtype=wp.vec2, device=device)
    wp.launch(curlnoise_2d_cd, n, inputs=[seed, xy, octaves, lac, gain, w, 1e-3, fd], device=device)
    fd_host = fd.numpy()

    # Central difference has O(eps^2) error; accept a few parts in 10^3.
    np.testing.assert_allclose(analytic, fd_host, rtol=0, atol=5e-3, err_msg="adj curlnoise 2D")


@wp.kernel
def curlnoise_3d_loss(
    seed: int,
    xyz: wp.array[wp.vec3],
    octaves: int,
    lac: float,
    gain: float,
    w: wp.vec3,
    loss: wp.array[float],
):
    tid = wp.tid()
    state = wp.rand_init(seed)
    v = wp.curlnoise(state, xyz[tid], wp.uint32(octaves), lac, gain)
    wp.atomic_add(loss, 0, v[0] * w[0] + v[1] * w[1] + v[2] * w[2])


@wp.kernel
def curlnoise_3d_cd(
    seed: int,
    xyz: wp.array[wp.vec3],
    octaves: int,
    lac: float,
    gain: float,
    w: wp.vec3,
    eps: float,
    grad: wp.array[wp.vec3],
):
    tid = wp.tid()
    state = wp.rand_init(seed)
    p = xyz[tid]
    p_xp = wp.vec3(p[0] + eps, p[1], p[2])
    p_xm = wp.vec3(p[0] - eps, p[1], p[2])
    p_yp = wp.vec3(p[0], p[1] + eps, p[2])
    p_ym = wp.vec3(p[0], p[1] - eps, p[2])
    p_zp = wp.vec3(p[0], p[1], p[2] + eps)
    p_zm = wp.vec3(p[0], p[1], p[2] - eps)
    v_xp = wp.curlnoise(state, p_xp, wp.uint32(octaves), lac, gain)
    v_xm = wp.curlnoise(state, p_xm, wp.uint32(octaves), lac, gain)
    v_yp = wp.curlnoise(state, p_yp, wp.uint32(octaves), lac, gain)
    v_ym = wp.curlnoise(state, p_ym, wp.uint32(octaves), lac, gain)
    v_zp = wp.curlnoise(state, p_zp, wp.uint32(octaves), lac, gain)
    v_zm = wp.curlnoise(state, p_zm, wp.uint32(octaves), lac, gain)
    L_xp = v_xp[0] * w[0] + v_xp[1] * w[1] + v_xp[2] * w[2]
    L_xm = v_xm[0] * w[0] + v_xm[1] * w[1] + v_xm[2] * w[2]
    L_yp = v_yp[0] * w[0] + v_yp[1] * w[1] + v_yp[2] * w[2]
    L_ym = v_ym[0] * w[0] + v_ym[1] * w[1] + v_ym[2] * w[2]
    L_zp = v_zp[0] * w[0] + v_zp[1] * w[1] + v_zp[2] * w[2]
    L_zm = v_zm[0] * w[0] + v_zm[1] * w[1] + v_zm[2] * w[2]
    grad[tid] = wp.vec3(
        (L_xp - L_xm) / (2.0 * eps),
        (L_yp - L_ym) / (2.0 * eps),
        (L_zp - L_zm) / (2.0 * eps),
    )


def test_adj_curlnoise_3d(test, device):
    seed = 42
    octaves = 3
    lac = 2.0
    gain = 0.5
    w = wp.vec3(1.0, -0.5, 0.7)
    positions = np.array(
        [[0.17, 0.23, 0.41], [1.33, 2.77, 0.29], [-0.43, 0.61, 1.13]],
        dtype=np.float32,
    )
    n = len(positions)

    tape = wp.Tape()
    with tape:
        xyz = wp.array(positions, dtype=wp.vec3, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        wp.launch(curlnoise_3d_loss, n, inputs=[seed, xyz, octaves, lac, gain, w, loss], device=device)
    tape.backward(loss=loss)
    analytic = xyz.grad.numpy()

    fd = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(curlnoise_3d_cd, n, inputs=[seed, xyz, octaves, lac, gain, w, 1e-3, fd], device=device)
    fd_host = fd.numpy()

    np.testing.assert_allclose(analytic, fd_host, rtol=0, atol=5e-3, err_msg="adj curlnoise 3D")


@wp.kernel
def curlnoise_4d_loss(
    seed: int,
    xyzt: wp.array[wp.vec4],
    octaves: int,
    lac: float,
    gain: float,
    w: wp.vec3,
    loss: wp.array[float],
):
    tid = wp.tid()
    state = wp.rand_init(seed)
    v = wp.curlnoise(state, xyzt[tid], wp.uint32(octaves), lac, gain)
    wp.atomic_add(loss, 0, v[0] * w[0] + v[1] * w[1] + v[2] * w[2])


@wp.kernel
def curlnoise_4d_cd(
    seed: int,
    xyzt: wp.array[wp.vec4],
    octaves: int,
    lac: float,
    gain: float,
    w: wp.vec3,
    eps: float,
    grad: wp.array[wp.vec4],
):
    tid = wp.tid()
    state = wp.rand_init(seed)
    p = xyzt[tid]
    p_xp = wp.vec4(p[0] + eps, p[1], p[2], p[3])
    p_xm = wp.vec4(p[0] - eps, p[1], p[2], p[3])
    p_yp = wp.vec4(p[0], p[1] + eps, p[2], p[3])
    p_ym = wp.vec4(p[0], p[1] - eps, p[2], p[3])
    p_zp = wp.vec4(p[0], p[1], p[2] + eps, p[3])
    p_zm = wp.vec4(p[0], p[1], p[2] - eps, p[3])
    p_tp = wp.vec4(p[0], p[1], p[2], p[3] + eps)
    p_tm = wp.vec4(p[0], p[1], p[2], p[3] - eps)
    v_xp = wp.curlnoise(state, p_xp, wp.uint32(octaves), lac, gain)
    v_xm = wp.curlnoise(state, p_xm, wp.uint32(octaves), lac, gain)
    v_yp = wp.curlnoise(state, p_yp, wp.uint32(octaves), lac, gain)
    v_ym = wp.curlnoise(state, p_ym, wp.uint32(octaves), lac, gain)
    v_zp = wp.curlnoise(state, p_zp, wp.uint32(octaves), lac, gain)
    v_zm = wp.curlnoise(state, p_zm, wp.uint32(octaves), lac, gain)
    v_tp = wp.curlnoise(state, p_tp, wp.uint32(octaves), lac, gain)
    v_tm = wp.curlnoise(state, p_tm, wp.uint32(octaves), lac, gain)
    L_xp = v_xp[0] * w[0] + v_xp[1] * w[1] + v_xp[2] * w[2]
    L_xm = v_xm[0] * w[0] + v_xm[1] * w[1] + v_xm[2] * w[2]
    L_yp = v_yp[0] * w[0] + v_yp[1] * w[1] + v_yp[2] * w[2]
    L_ym = v_ym[0] * w[0] + v_ym[1] * w[1] + v_ym[2] * w[2]
    L_zp = v_zp[0] * w[0] + v_zp[1] * w[1] + v_zp[2] * w[2]
    L_zm = v_zm[0] * w[0] + v_zm[1] * w[1] + v_zm[2] * w[2]
    L_tp = v_tp[0] * w[0] + v_tp[1] * w[1] + v_tp[2] * w[2]
    L_tm = v_tm[0] * w[0] + v_tm[1] * w[1] + v_tm[2] * w[2]
    grad[tid] = wp.vec4(
        (L_xp - L_xm) / (2.0 * eps),
        (L_yp - L_ym) / (2.0 * eps),
        (L_zp - L_zm) / (2.0 * eps),
        (L_tp - L_tm) / (2.0 * eps),
    )


def test_adj_curlnoise_4d(test, device):
    seed = 42
    octaves = 3
    lac = 2.0
    gain = 0.5
    w = wp.vec3(1.0, -0.5, 0.7)
    positions = np.array(
        [[0.17, 0.23, 0.41, 0.67], [1.33, 2.77, 0.29, 1.83], [-0.43, 0.61, 1.13, 0.37]],
        dtype=np.float32,
    )
    n = len(positions)

    tape = wp.Tape()
    with tape:
        xyzt = wp.array(positions, dtype=wp.vec4, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        wp.launch(curlnoise_4d_loss, n, inputs=[seed, xyzt, octaves, lac, gain, w, loss], device=device)
    tape.backward(loss=loss)
    analytic = xyzt.grad.numpy()

    fd = wp.zeros(n, dtype=wp.vec4, device=device)
    wp.launch(curlnoise_4d_cd, n, inputs=[seed, xyzt, octaves, lac, gain, w, 1e-3, fd], device=device)
    fd_host = fd.numpy()

    np.testing.assert_allclose(analytic, fd_host, rtol=0, atol=5e-3, err_msg="adj curlnoise 4D")


# -------------- Independent verification via Python reimplementation --------------
#
# This mirrors the forward curlnoise (2D) entirely in Warp Python @wp.func
# building blocks — no call to ``wp.curlnoise``. Because the Python
# reimplementation is built from operations whose adjoints Warp already
# knows (add, mul, floor, cos, sin, randf, ...), autodiff through the
# Python forward gives analytic gradients that are independent of the
# derivation used in ``adj_curlnoise``. Comparing the two analytic paths
# is a tighter check than central difference.
#
# 2D is covered with an autodiff-of-reimplementation test; the forward
# reimplementation mirrors ``noise_2d_gradient`` and ``curlnoise`` from
# ``warp/native/noise.h`` line-for-line, so its autodiff-derived Jacobian
# gives an implementation-independent ground truth for the native adjoint.
# 3D and 4D reimplementation tests were prototyped but run into a Warp
# autodiff interaction with state-mutating builtins (``rand_init``) inside
# a loop: Warp's reverse pass doesn't correctly replay the per-octave
# state chain used by each noise-field call. The central-difference tests
# above already cover 3D / 4D at 5e-3 tolerance, which is sufficient
# verification for the analytic adjoints.


@wp.func
def _corner_grad_2d_ref(state: wp.uint32, ix: int, iy: int) -> wp.vec2:
    # Matches native ``random_gradient_2d``: hash (ix, iy, state) into a
    # seed via prime products and XOR, take two ``randf`` samples as a
    # point in the unit square centred on the origin, then normalize to
    # project onto the unit circle (``sample_unit_square`` + ``normalize``).
    idx = (wp.uint32(ix) * wp.uint32(73856093)) ^ (wp.uint32(iy) * wp.uint32(19349663) + state)
    x = wp.randf(idx) - 0.5
    y = wp.randf(idx) - 0.5
    return wp.normalize(wp.vec2(x, y))


@wp.func
def _noise_2d_gradient_ref(state: wp.uint32, x: float, y: float) -> wp.vec2:
    # Analytic spatial gradient of Perlin 2D noise, expressed as
    # composable operations on the smooth inputs ``x`` and ``y``. The
    # corner-index / corner-gradient branches are non-differentiable by
    # construction (int floors, hash, cos/sin of hashed angle).
    ix0 = int(wp.floor(x))
    iy0 = int(wp.floor(y))
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    dx = x - float(ix0)
    dy = y - float(iy0)

    g00 = _corner_grad_2d_ref(state, ix0, iy0)
    g10 = _corner_grad_2d_ref(state, ix1, iy0)
    g01 = _corner_grad_2d_ref(state, ix0, iy1)
    g11 = _corner_grad_2d_ref(state, ix1, iy1)

    v00 = dx * g00[0] + dy * g00[1]
    v10 = (dx - 1.0) * g10[0] + dy * g10[1]
    v01 = dx * g01[0] + (dy - 1.0) * g01[1]
    v11 = (dx - 1.0) * g11[0] + (dy - 1.0) * g11[1]

    s = dx * dx * dx * (dx * (dx * 6.0 - 15.0) + 10.0)
    s1 = 30.0 * dx * dx * (dx * (dx - 2.0) + 1.0)
    T = dy * dy * dy * (dy * (dy * 6.0 - 15.0) + 10.0)
    T1 = 30.0 * dy * dy * (dy * (dy - 2.0) + 1.0)

    xi0 = v00 + (v10 - v00) * s
    xi1 = v01 + (v11 - v01) * s

    gxi0_x = g00[0] + (g10[0] - g00[0]) * s + (v10 - v00) * s1
    gxi0_y = g00[1] + (g10[1] - g00[1]) * s
    gxi1_x = g01[0] + (g11[0] - g01[0]) * s + (v11 - v01) * s1
    gxi1_y = g01[1] + (g11[1] - g01[1]) * s

    gN_x = gxi0_x + (gxi1_x - gxi0_x) * T
    gN_y = gxi0_y + (gxi1_y - gxi0_y) * T + (xi1 - xi0) * T1

    return wp.vec2(gN_x, gN_y)


@wp.kernel
def curlnoise_2d_reimpl_loss(
    seed: int,
    xy: wp.array[wp.vec2],
    octaves: int,
    lac: float,
    gain: float,
    w: wp.vec2,
    loss: wp.array[float],
):
    tid = wp.tid()
    state = wp.rand_init(seed)
    p = xy[tid]
    curl_x = float(0.0)
    curl_y = float(0.0)
    freq = float(1.0)
    amp = float(1.0)
    for _ in range(octaves):
        gx = _noise_2d_gradient_ref(state, freq * p[0], freq * p[1])
        curl_x = curl_x + amp * gx[0]
        curl_y = curl_y + amp * gx[1]
        amp = amp * gain
        freq = freq * lac
    out_x = -curl_y
    out_y = curl_x
    wp.atomic_add(loss, 0, out_x * w[0] + out_y * w[1])


def test_adj_curlnoise_2d_autodiff_reimpl(test, device):
    """Independent verification: autodiff of a pure-Python-kernel
    reimplementation of 2D curlnoise vs the native ``adj_curlnoise``.
    Both compute analytic gradients, so they should agree to FP noise."""
    seed = 42
    octaves = 3
    lac = 2.0
    gain = 0.5
    w_np = np.array([1.0, -0.5], dtype=np.float32)
    w = wp.vec2(w_np[0], w_np[1])
    positions = np.array([[0.17, 0.23], [1.33, 2.77], [-0.43, 0.61]], dtype=np.float32)
    n = len(positions)

    # Path 1: native wp.curlnoise + native adj.
    tape1 = wp.Tape()
    with tape1:
        xy1 = wp.array(positions, dtype=wp.vec2, device=device, requires_grad=True)
        loss1 = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        wp.launch(curlnoise_2d_loss, n, inputs=[seed, xy1, octaves, lac, gain, w, loss1], device=device)
    tape1.backward(loss=loss1)
    native_adj = xy1.grad.numpy()

    # Path 2: Python-kernel reimplementation + Warp autodiff.
    tape2 = wp.Tape()
    with tape2:
        xy2 = wp.array(positions, dtype=wp.vec2, device=device, requires_grad=True)
        loss2 = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        wp.launch(curlnoise_2d_reimpl_loss, n, inputs=[seed, xy2, octaves, lac, gain, w, loss2], device=device)
    tape2.backward(loss=loss2)
    reimpl_adj = xy2.grad.numpy()

    # Both are analytic; the only disagreement is FP ordering in the two
    # code paths. Tight tolerance.
    np.testing.assert_allclose(native_adj, reimpl_adj, rtol=0, atol=1e-4, err_msg="autodiff adj curlnoise 2D")


devices = get_test_devices()


class TestNoise(unittest.TestCase):
    pass


add_function_test(TestNoise, "test_pnoise", test_pnoise, devices=devices)
add_function_test(TestNoise, "test_curlnoise", test_curlnoise, devices=devices)
add_function_test(TestNoise, "test_adj_noise", test_adj_noise, devices=devices)
add_function_test(TestNoise, "test_adj_curlnoise_2d", test_adj_curlnoise_2d, devices=devices)
add_function_test(TestNoise, "test_adj_curlnoise_3d", test_adj_curlnoise_3d, devices=devices)
add_function_test(TestNoise, "test_adj_curlnoise_4d", test_adj_curlnoise_4d, devices=devices)
add_function_test(
    TestNoise,
    "test_adj_curlnoise_2d_autodiff_reimpl",
    test_adj_curlnoise_2d_autodiff_reimpl,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
