# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import unittest

import warp as wp
from warp.tests.test_base import *

wp.init()
epsilon = 0.00001

@wp.kernel
def closest_point_edge_edge_kernel(
    p1: wp.array(dtype=wp.vec3),
    q1: wp.array(dtype=wp.vec3),
    p2: wp.array(dtype=wp.vec3),
    q2: wp.array(dtype=wp.vec3),
    epsilon: float,
    st0: wp.array(dtype=wp.vec3),
    c1: wp.array(dtype=wp.vec3),
    c2: wp.array(dtype=wp.vec3),
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

    closest_point_edge_edge_launch(
        p1, q1, p2, q2, epsilon, st0, c1, c2, device
    )

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


def register(parent):
        
    devices = wp.get_devices()

    class TestClosestPointEdgeEdgeMethods(parent):
        pass

    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_middle_crossing", test_edge_edge_middle_crossing, devices=devices)
    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel_s1_t0", test_edge_edge_parallel_s1_t0, devices=devices)
    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel_s0_t1", test_edge_edge_parallel_s0_t1, devices=devices)
    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_both_degenerate_case", test_edge_edge_both_degenerate_case, devices=devices)
    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_degenerate_first_edge", test_edge_edge_degenerate_first_edge, devices=devices)
    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_degenerate_second_edge", test_edge_edge_degenerate_second_edge, devices=devices)
    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel", test_edge_edge_parallel, devices=devices)
    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_perpendicular_s1_t0", test_edge_edge_perpendicular_s1_t0, devices=devices)
    add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_perpendicular_s0_t1", test_edge_edge_perpendicular_s0_t1, devices=devices)

    return TestClosestPointEdgeEdgeMethods

if __name__ == '__main__':    
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)