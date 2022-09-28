# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from warp.tests.test_base import *
from warp.sim import ModelBuilder

import numpy as np

wp.init()

devices = wp.get_devices()

def register(parent):

    class TestModel(parent):
        def test_add_triangles(self):
            pts = np.random.randn(5,3)
            tris = np.array([[0,3,1],[1,3,4],[1,4,2],[1,4,1]])
            
            builder1 = ModelBuilder()
            builder2 = ModelBuilder()
            for pt in pts:
                builder1.add_particle(pt, [0.0,0.0,0.0], 1.0)
                builder2.add_particle(pt, [0.0,0.0,0.0], 1.0)
            
            # test add_triangle(s) with default arguments:
            for t in tris:
                builder1.add_triangle(t[0],t[1],t[2])
            builder2.add_triangles( tris[:,0], tris[:,1], tris[:,2])
            
            # test add_triangle(s) with non default arguments:
            tri_ke = np.random.randn(3)
            tri_ka = np.random.randn(3)
            tri_kd = np.random.randn(3)
            tri_drag = np.random.randn(3)
            tri_lift = np.random.randn(3)
            for i in range(3):
                t = tris[i]
                builder1.add_triangle(t[0],t[1],t[2],tri_ke[i],tri_ka[i],tri_kd[i],tri_drag[i],tri_lift[i],)
            builder2.add_triangles( tris[:,0], tris[:,1], tris[:,2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift )

            assert_np_equal(np.array(builder1.tri_indices), np.array(builder2.tri_indices))
            assert_np_equal(np.array(builder1.tri_poses), np.array(builder2.tri_poses), tol=1.e-6)
            assert_np_equal(np.array(builder1.tri_activations), np.array(builder2.tri_activations))
            assert_np_equal(np.array(builder1.tri_materials), np.array(builder2.tri_materials))

            
        def test_add_edges(self):
            pts = np.random.randn(5,3)
            edges = np.array([[0,4,3,1],[3,2,4,1]])
            
            builder1 = ModelBuilder()
            builder2 = ModelBuilder()
            for pt in pts:
                builder1.add_particle(pt, [0.0,0.0,0.0], 1.0)
                builder2.add_particle(pt, [0.0,0.0,0.0], 1.0)
            
            # test defaults:
            for i in range(2):
                builder1.add_edge(edges[i,0],edges[i,1],edges[i,2],edges[i,3])
            builder2.add_edges(edges[:,0],edges[:,1],edges[:,2],edges[:,3])

            # test non defaults:
            rest = np.random.randn(2)
            edge_ke = np.random.randn(2)
            edge_kd = np.random.randn(2)
            for i in range(2):
                builder1.add_edge(edges[i,0],edges[i,1],edges[i,2],edges[i,3],rest[i],edge_ke[i],edge_kd[i])
            builder2.add_edges(edges[:,0],edges[:,1],edges[:,2],edges[:,3],rest,edge_ke,edge_kd)

            assert_np_equal(np.array(builder1.edge_indices), np.array(builder2.edge_indices))
            assert_np_equal(np.array(builder1.edge_rest_angle), np.array(builder2.edge_rest_angle), tol=1.e-4)
            assert_np_equal(np.array(builder1.edge_bending_properties), np.array(builder2.edge_bending_properties))

            

    return TestModel

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
