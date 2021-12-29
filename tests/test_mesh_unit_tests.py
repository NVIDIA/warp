import unittest

import warp as wp
import numpy as np

# wp.config.verbose = True
# wp.config.mode = "debug"
# wp.config.verify_cuda = True

wp.init()

@wp.func
def min_vec3(a:wp.vec3, b:wp.vec3):
    return wp.vec3(wp.min(a[0],b[0]), wp.min(a[1],b[1]), wp.min(a[2],b[2]))

@wp.func
def max_vec3(a:wp.vec3, b:wp.vec3):
    return wp.vec3(wp.max(a[0],b[0]), wp.max(a[1],b[1]), wp.max(a[2],b[2]))

@wp.kernel
def compute_bounds(indices: wp.array(dtype=int), positions: wp.array(dtype=wp.vec3), lowers:wp.array(dtype=wp.vec3), uppers: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    i = wp.load(indices, tid * 3 + 0)
    j = wp.load(indices, tid * 3 + 1)
    k = wp.load(indices, tid * 3 + 2)

    x0 = wp.load(positions, i)        # point zero
    x1 = wp.load(positions, j)        # point one
    x2 = wp.load(positions, k)        # point two

    lower = min_vec3(min_vec3(x0,x1),x2)
    upper = max_vec3(max_vec3(x0,x1),x2)

    wp.store(lowers, tid, lower)
    wp.store(uppers, tid, upper)



@wp.kernel
def compute_num_contacts(lowers: wp.array(dtype=wp.vec3), uppers: wp.array(dtype=wp.vec3), mesh_id: wp.uint64, counts: wp.array(dtype=int)):
    
    face_index = int(0)
    face_u = float(0.0)  
    face_v = float(0.0)
    sign = float(0.0)

    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    query = wp.mesh_query_aabb(mesh_id, lower, upper)

    index = int(-1)
    count = int(0)
    
    while(wp.mesh_query_aabb_next(query, index)):
        count = count+1
        
    counts[tid] = count


class TestMeshQueryAABBMethods(unittest.TestCase):

    def test_compute_bounds(self):
        device = "cuda"
        #create two triangles.
        points = np.array([[0,0,0],[1,0,0],[0,1,0], [-1,-1,1]])
        indices = np.array([0,1,2,1,2,3])
        m = wp.Mesh(wp.array(points, dtype=wp.vec3, device=device), None, wp.array(indices, dtype=int, device=device))

        num_tris = int(len(indices)/3)

        lowers = wp.empty(n=num_tris,dtype=wp.vec3,device=device)
        uppers = wp.empty_like(lowers)
        wp.launch(
            kernel=compute_bounds, 
            dim=num_tris, 
            inputs=[m.indices, m.points],
            outputs=[lowers,uppers],
            device=device)

        lower_view = lowers.numpy()
        upper_view = uppers.numpy()
        wp.synchronize()

        self.assertTrue(lower_view[0][0] == 0)
        self.assertTrue(lower_view[0][1] == 0)
        self.assertTrue(lower_view[0][2] == 0)

        self.assertTrue(upper_view[0][0] == 1)
        self.assertTrue(upper_view[0][1] == 1)
        self.assertTrue(upper_view[0][2] == 0)

        self.assertTrue(lower_view[1][0] == -1)
        self.assertTrue(lower_view[1][1] == -1)
        self.assertTrue(lower_view[1][2] == 0)

        self.assertTrue(upper_view[1][0] == 1)
        self.assertTrue(upper_view[1][1] == 1)
        self.assertTrue(upper_view[1][2] == 1)


    def test_mesh_query_aabb_count_overlap(self):
        device = "cuda"
        #create two triangles.
        points = np.array([[0,0,0],[1,0,0],[0,1,0], [-1,-1,1]])
        indices = np.array([0,1,2,1,2,3])
        m = wp.Mesh(wp.array(points, dtype=wp.vec3, device=device), None, wp.array(indices, dtype=int, device=device))

        num_tris = int(len(indices)/3)
        
        lowers = wp.empty(n=num_tris,dtype=wp.vec3,device=device)
        uppers = wp.empty_like(lowers)
        wp.launch(
            kernel=compute_bounds, 
            dim=num_tris, 
            inputs=[m.indices, m.points],
            outputs=[lowers,uppers],
            device=device)

        counts = wp.empty(n=num_tris,dtype=int,device=device)

        wp.launch(
            kernel=compute_num_contacts, 
            dim=num_tris, 
            inputs=[lowers, uppers, m.id],
            outputs=[counts],
            device=device)

        wp.synchronize()

        view = counts.numpy()

        for c in view:
            self.assertTrue(c == 2)
        
    def test_mesh_query_aabb_count_nonoverlap(self):
        device = "cuda"
        #create two triangles.
        points = np.array([[0,0,0],[1,0,0],[0,1,0], [10,0,0], [10,1,0], [10,0,1]])
        indices = np.array([0,1,2,3,4,5])
        m = wp.Mesh(wp.array(points, dtype=wp.vec3, device=device), None, wp.array(indices, dtype=int, device=device))

        num_tris = int(len(indices)/3)
        
        lowers = wp.empty(n=num_tris,dtype=wp.vec3,device=device)
        uppers = wp.empty_like(lowers)
        wp.launch(
            kernel=compute_bounds, 
            dim=num_tris, 
            inputs=[m.indices, m.points],
            outputs=[lowers,uppers],
            device=device)

        counts = wp.empty(n=num_tris,dtype=int,device=device)

        wp.launch(
            kernel=compute_num_contacts, 
            dim=num_tris, 
            inputs=[lowers, uppers, m.id],
            outputs=[counts],
            device=device)

        wp.synchronize()

        view = counts.numpy()

        for c in view:
            self.assertTrue(c == 1)
        



if __name__ == '__main__':
    unittest.main()