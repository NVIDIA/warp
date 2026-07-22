# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

epsilon = 0.00001


@wp.kernel
def closest_point_edge_edge_kernel(
    p1: wp.array[wp.vec3],
    q1: wp.array[wp.vec3],
    p2: wp.array[wp.vec3],
    q2: wp.array[wp.vec3],
    epsilon: float,
    st0: wp.array[wp.vec3],
    c1: wp.array[wp.vec3],
    c2: wp.array[wp.vec3],
):
    tid = wp.tid()
    st = wp.closest_point_edge_edge(p1[tid], q1[tid], p2[tid], q2[tid], epsilon)
    s = st[0]
    t = st[1]
    st0[tid] = st
    c1[tid] = p1[tid] + (q1[tid] - p1[tid]) * s
    c2[tid] = p2[tid] + (q2[tid] - p2[tid]) * t


def closest_point_edge_edge_launch(p1, q1, p2, q2, epsilon, st0, c1, c2, device):
    n = len(p1)
    wp.launch(
        kernel=closest_point_edge_edge_kernel,
        dim=n,
        inputs=[p1, q1, p2, q2, epsilon],
        outputs=[st0, c1, c2],
        device=device,
    )


def run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device):
    p1 = wp.array(p1_h, dtype=wp.vec3, device=device)
    q1 = wp.array(q1_h, dtype=wp.vec3, device=device)
    p2 = wp.array(p2_h, dtype=wp.vec3, device=device)
    q2 = wp.array(q2_h, dtype=wp.vec3, device=device)
    st0 = wp.empty_like(p1)
    c1 = wp.empty_like(p1)
    c2 = wp.empty_like(p1)

    closest_point_edge_edge_launch(p1, q1, p2, q2, epsilon, st0, c1, c2, device)

    wp.synchronize()
    view = st0.numpy()
    return view


def test_edge_edge_middle_crossing(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[0, 1, 0]])
    q2_h = np.array([[1, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.5)  # s value
    test.assertAlmostEqual(st0[1], 0.5)  # t value


def test_edge_edge_near_parallel_middle_crossing(test, device):
    """Verify the forward solve finds the midpoint intersection of two near-parallel segments.

    The two unit segments cross at their midpoints with ``sin^2(theta)`` just below the adjoint stability tolerance
    (1e-6). The forward must still find the intersection. The relative-tolerance gate is for the adjoint only, not for
    forward correctness.
    """
    p1_h = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    q1_h = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    p2_h = np.array([[0.0, -4.9e-4, 0.0]], dtype=np.float32)
    q2_h = np.array([[1.0, 4.9e-4, 0.0]], dtype=np.float32)

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.5, delta=1.0e-2)
    test.assertAlmostEqual(st0[1], 0.5, delta=1.0e-2)
    test.assertAlmostEqual(st0[2], 0.0, delta=1.0e-5)


def test_edge_edge_parallel_s1_t0(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[2, 2, 0]])
    q2_h = np.array([[3, 3, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 1.0)  # s value
    test.assertAlmostEqual(st0[1], 0.0)  # t value


def test_edge_edge_parallel_s0_t1(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[-2, -2, 0]])
    q2_h = np.array([[-1, -1, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 1.0)  # t value


def test_edge_edge_both_degenerate_case(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[0, 0, 0]])
    p2_h = np.array([[1, 1, 1]])
    q2_h = np.array([[1, 1, 1]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 0.0)  # t value


def test_edge_edge_degenerate_first_edge(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[0, 0, 0]])
    p2_h = np.array([[0, 1, 0]])
    q2_h = np.array([[1, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 0.5)  # t value


def test_edge_edge_degenerate_first_edge_projection_outside(test, device):
    """Clamp the projection parameter to 1 when a degenerate edge projects past the segment end.

    Edge 1 is a point at x=10; edge 2 is the segment from x=0 to x=1. The projection lands at t=10 on the
    infinite line, which must clamp to 1.
    """
    p1_h = np.array([[10, 0, 0]])
    q1_h = np.array([[10, 0, 0]])
    p2_h = np.array([[0, 0, 0]])
    q2_h = np.array([[1, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 1.0)  # t value (clamped from 10)


def test_edge_edge_degenerate_second_edge(test, device):
    p1_h = np.array([[1, 0, 0]])
    q1_h = np.array([[0, 1, 0]])
    p2_h = np.array([[1, 1, 0]])
    q2_h = np.array([[1, 1, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.5)  # s value
    test.assertAlmostEqual(st0[1], 0.0)  # t value


def test_edge_edge_parallel(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 0, 0]])
    p2_h = np.array([[-0.5, 1, 0]])
    q2_h = np.array([[0.5, 1, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 0.5)  # t value


def test_edge_edge_perpendicular_s1_t0(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[10, 1, 0]])
    q2_h = np.array([[11, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 1.0)  # s value
    test.assertAlmostEqual(st0[1], 0.0)  # t value


def test_edge_edge_perpendicular_s0_t1(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[-11, -1, 0]])
    q2_h = np.array([[-5, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 1.0)  # t value


@wp.func
def check_edge_closest_point_sufficient_necessary(c1: wp.vec3, c2: wp.vec3, t: float, p: wp.vec3, q: wp.vec3):
    """
    This is a sufficient and necessary condition of closest point
    c1: closest point on the other edge
    c2: closest point on edge p-q
    t: c2 = (1.0-t) * p + t * q
    e1, e2: end points of the edge
    """
    eps = 1e-5
    e = p - q
    if t == 0.0:
        wp.expect_eq(wp.dot(c1 - p, p - q) > -eps, True)
        wp.expect_eq(wp.abs(wp.length(c2 - p)) < eps, True)
    elif t == 1.0:
        wp.expect_eq(wp.dot(c1 - q, q - p) > -eps, True)
        wp.expect_eq(wp.abs(wp.length(c2 - q)) < eps, True)
    else:
        # interior closest point, c1c2 must be perpendicular to e
        c1c2 = c1 - c2
        wp.expect_eq(wp.abs(wp.dot(c1c2, e)) < eps, True)


@wp.kernel
def check_edge_closest_point_sufficient_necessary_kernel(
    p1s: wp.array[wp.vec3],
    q1s: wp.array[wp.vec3],
    p2s: wp.array[wp.vec3],
    q2s: wp.array[wp.vec3],
    epsilon: float,
):
    tid = wp.tid()

    p1 = p1s[tid]
    q1 = q1s[tid]
    p2 = p2s[tid]
    q2 = q2s[tid]

    st = wp.closest_point_edge_edge(p1, q1, p2, q2, epsilon)
    s = st[0]
    t = st[1]
    c1 = p1 + (q1 - p1) * s
    c2 = p2 + (q2 - p2) * t

    check_edge_closest_point_sufficient_necessary(c1, c2, t, p2, q2)
    check_edge_closest_point_sufficient_necessary(c2, c1, s, p1, q1)


def check_edge_closest_point_random(test, device):
    num_tests = 100000
    rng = np.random.default_rng(123)
    p1 = wp.array(rng.standard_normal(size=(num_tests, 3)), dtype=wp.vec3, device=device)
    q1 = wp.array(rng.standard_normal(size=(num_tests, 3)), dtype=wp.vec3, device=device)

    p2 = wp.array(rng.standard_normal(size=(num_tests, 3)), dtype=wp.vec3, device=device)
    q2 = wp.array(rng.standard_normal(size=(num_tests, 3)), dtype=wp.vec3, device=device)

    wp.launch(
        kernel=check_edge_closest_point_sufficient_necessary_kernel,
        dim=num_tests,
        inputs=[p1, q1, p2, q2, epsilon],
        device=device,
    )

    # parallel edges
    p1 = rng.standard_normal(size=(num_tests, 3))
    q1 = rng.standard_normal(size=(num_tests, 3))

    shifts = rng.standard_normal(size=(num_tests, 3))

    p2 = p1 + shifts
    q2 = q1 + shifts

    p1 = wp.array(p1, dtype=wp.vec3, device=device)
    q1 = wp.array(q1, dtype=wp.vec3, device=device)

    p2 = wp.array(p2, dtype=wp.vec3, device=device)
    q2 = wp.array(q2, dtype=wp.vec3, device=device)

    wp.launch(
        kernel=check_edge_closest_point_sufficient_necessary_kernel,
        dim=num_tests,
        inputs=[p1, q1, p2, q2, epsilon],
        device=device,
    )


@wp.kernel
def cpee_dist_kernel(
    p1: wp.array[wp.vec3],
    q1: wp.array[wp.vec3],
    p2: wp.array[wp.vec3],
    q2: wp.array[wp.vec3],
    eps: float,
    dist: wp.array[float],
):
    tid = wp.tid()
    out = wp.closest_point_edge_edge(p1[tid], q1[tid], p2[tid], q2[tid], eps)
    dist[tid] = out[2]


def _analytic_dist_grads(p1_np, q1_np, p2_np, q2_np, device):
    """Backprop ∂(sum dist)/∂(p1, q1, p2, q2) via Warp's tape."""
    p1 = wp.array(p1_np, dtype=wp.vec3, device=device, requires_grad=True)
    q1 = wp.array(q1_np, dtype=wp.vec3, device=device, requires_grad=True)
    p2 = wp.array(p2_np, dtype=wp.vec3, device=device, requires_grad=True)
    q2 = wp.array(q2_np, dtype=wp.vec3, device=device, requires_grad=True)
    n = len(p1_np)
    dist = wp.zeros(n, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(cpee_dist_kernel, dim=n, inputs=[p1, q1, p2, q2, epsilon, dist], device=device)
    dist.grad.fill_(1.0)
    tape.backward()
    return p1.grad.numpy().copy(), q1.grad.numpy().copy(), p2.grad.numpy().copy(), q2.grad.numpy().copy()


def _fd_dist_grads(p1_np, q1_np, p2_np, q2_np, device, h=1e-3):
    """Central-difference gradient of dist w.r.t. each (p1, q1, p2, q2) input."""
    n = len(p1_np)

    def forward(p1, q1, p2, q2):
        p1w = wp.array(p1, dtype=wp.vec3, device=device)
        q1w = wp.array(q1, dtype=wp.vec3, device=device)
        p2w = wp.array(p2, dtype=wp.vec3, device=device)
        q2w = wp.array(q2, dtype=wp.vec3, device=device)
        dw = wp.zeros(n, dtype=float, device=device)
        wp.launch(cpee_dist_kernel, dim=n, inputs=[p1w, q1w, p2w, q2w, epsilon, dw], device=device)
        return dw.numpy().copy()

    inputs = (p1_np, q1_np, p2_np, q2_np)
    grads = []
    for which, arr in enumerate(inputs):
        g = np.zeros_like(arr)
        for axis in range(3):
            plus = arr.copy()
            minus = arr.copy()
            plus[:, axis] += h
            minus[:, axis] -= h
            args_plus = [plus if i == which else inputs[i] for i in range(4)]
            args_minus = [minus if i == which else inputs[i] for i in range(4)]
            d_plus = forward(*args_plus)
            d_minus = forward(*args_minus)
            g[:, axis] = (d_plus - d_minus) / (2.0 * h)
        grads.append(g)
    return tuple(grads)


def _well_conditioned_mask(p1, q1, p2, q2):
    """Accept inputs that fall *firmly inside* one of the smooth branches
    (D, E, or F) of the forward function. Rejects:
      - degenerate edges (epsilon branches)
      - near-parallel edges (boundary between D and D_DEGEN)
      - inputs near a clamp boundary or t-correction boundary, where finite
        differences would straddle a kink.

    Coverage is sample-by-sample: D needs s_init_uc and t_init both interior;
    E needs t_init firmly < 0 and -c/a interior; F needs t_init firmly > 1
    and (b-c)/a interior."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    b = np.einsum("ij,ij->i", d1, d2)
    n = np.cross(d1, d2)
    denom = np.einsum("ij,ij->i", n, n)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    ae = a * e

    not_degen = (a > 0.1) & (e > 0.1)
    not_parallel = denom > 1e-3 * np.maximum(ae, 1e-30)
    safe_denom = np.where(denom > 1e-30, denom, 1.0)
    safe_a = np.where(a > 1e-30, a, 1.0)

    s_init_uc = (b * f - c * e) / safe_denom
    s_init = np.clip(s_init_uc, 0.0, 1.0)
    t_init = (b * s_init + f) / e

    margin = 0.05

    # CASE_D: t_init in interior of [0, 1] AND s_init_uc interior
    in_D = (t_init > margin) & (t_init < 1.0 - margin) & (s_init_uc > margin) & (s_init_uc < 1.0 - margin)

    # CASE_E: t firmly clamped to 0; s_uc = -c/a must be interior
    s_E_uc = -c / safe_a
    in_E = (t_init < -margin) & (s_E_uc > margin) & (s_E_uc < 1.0 - margin)

    # CASE_F: t firmly clamped to 1; s_uc = (b-c)/a must be interior
    s_F_uc = (b - c) / safe_a
    in_F = (t_init > 1.0 + margin) & (s_F_uc > margin) & (s_F_uc < 1.0 - margin)

    return not_degen & not_parallel & (in_D | in_E | in_F)


def test_edge_edge_grad_finite_difference(test, device):
    """Analytic gradients of dist match central finite differences across all
    forward branches at well-conditioned inputs (away from boundary transitions)."""
    rng = np.random.default_rng(2024)
    n = 500
    p1 = rng.standard_normal((n, 3)).astype(np.float32)
    q1 = rng.standard_normal((n, 3)).astype(np.float32)
    p2 = rng.standard_normal((n, 3)).astype(np.float32)
    q2 = rng.standard_normal((n, 3)).astype(np.float32)

    keep = _well_conditioned_mask(p1, q1, p2, q2)
    test.assertGreater(int(keep.sum()), 50, msg="too few well-conditioned cases")

    p1, q1, p2, q2 = p1[keep], q1[keep], p2[keep], q2[keep]

    g_p1, g_q1, g_p2, g_q2 = _analytic_dist_grads(p1, q1, p2, q2, device)
    fd_p1, fd_q1, fd_p2, fd_q2 = _fd_dist_grads(p1, q1, p2, q2, device, h=1e-3)

    for _name, g, fd in [("p1", g_p1, fd_p1), ("q1", g_q1, fd_q1), ("p2", g_p2, fd_p2), ("q2", g_q2, fd_q2)]:
        assert_np_equal(g, fd, tol=2e-3)


def test_edge_edge_grad_bounded_near_parallel(test, device):
    """At near-parallel edges, the analytic gradient must remain finite and
    physically bounded. The previous codegen'd adjoint produced gradients up
    to ~1e6 here because it divided by a denom near machine epsilon; the
    cross-product formulation keeps the chain rule well-conditioned.

    Sampling: log-uniform sin θ ∈ [1e-7, 1e-2] covers both the parallel
    branch (sin θ < sqrt(REL_PARALLEL_TOL) ≈ 1e-3) and the near-threshold
    region where the previous formulation was most pathological. Half the
    samples concentrate at sin θ ≈ 1e-3 to specifically stress the threshold.
    """
    rng = np.random.default_rng(7)
    n = 400

    p1 = rng.standard_normal((n, 3)).astype(np.float32)
    q1 = p1 + rng.standard_normal((n, 3)).astype(np.float32)
    d1 = q1 - p1
    shift = rng.standard_normal((n, 3)).astype(np.float32)

    log_amp = np.concatenate(
        [
            rng.uniform(-7.0, -2.0, n // 2),  # full range
            rng.uniform(-3.05, -2.95, n - n // 2),  # right at the threshold
        ]
    )
    eps_amp = (10.0**log_amp).astype(np.float32)
    eps_perp = (eps_amp[:, None] * rng.standard_normal((n, 3))).astype(np.float32)
    d2 = (d1 * rng.uniform(0.5, 2.0, n)[:, None].astype(np.float32)) + eps_perp
    p2 = p1 + shift
    q2 = p2 + d2

    g_p1, g_q1, g_p2, g_q2 = _analytic_dist_grads(p1, q1, p2, q2, device)

    # ∂dist/∂(any single input vec3) is geometrically bounded by 1 for unit
    # adj_dist (each input contributes linearly to c1 or c2 with coefficient
    # in [0, 1]). Empirically max is 1.0; we leave 10x headroom for legitimate
    # near-boundary amplification while still flagging any 1e3+ blow-up.
    bound = 10.0
    for name, g in [("p1", g_p1), ("q1", g_q1), ("p2", g_p2), ("q2", g_q2)]:
        test.assertTrue(np.all(np.isfinite(g)), msg=f"{name} has non-finite gradient")
        max_abs = float(np.abs(g).max())
        test.assertLess(max_abs, bound, msg=f"{name} gradient magnitude {max_abs:.3g} exceeds bound {bound}")


devices = get_test_devices()


class TestClosestPointEdgeEdgeMethods(unittest.TestCase):
    pass


add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_middle_crossing",
    test_edge_edge_middle_crossing,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_near_parallel_middle_crossing",
    test_edge_edge_near_parallel_middle_crossing,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel_s1_t0", test_edge_edge_parallel_s1_t0, devices=devices
)
add_function_test(
    TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel_s0_t1", test_edge_edge_parallel_s0_t1, devices=devices
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_both_degenerate_case",
    test_edge_edge_both_degenerate_case,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_degenerate_first_edge",
    test_edge_edge_degenerate_first_edge,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_degenerate_first_edge_projection_outside",
    test_edge_edge_degenerate_first_edge_projection_outside,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_degenerate_second_edge",
    test_edge_edge_degenerate_second_edge,
    devices=devices,
)
add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel", test_edge_edge_parallel, devices=devices)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_perpendicular_s1_t0",
    test_edge_edge_perpendicular_s1_t0,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_perpendicular_s0_t1",
    test_edge_edge_perpendicular_s0_t1,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_closest_point_random",
    check_edge_closest_point_random,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_grad_finite_difference",
    test_edge_edge_grad_finite_difference,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_grad_bounded_near_parallel",
    test_edge_edge_grad_bounded_near_parallel,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
