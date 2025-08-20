import numpy as np
import warp as wp

from warp.tests.test_utils import get_test_devices

@wp.kernel
def compute_group_roots(bvh: wp.uint64, group_roots: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    root = wp.bvh_get_group_root(bvh, wp.int32(tid))
    group_roots[tid] = root

@wp.kernel
def debug_group_info(bvh: wp.uint64, group_id: wp.int32, first_leaf: wp.array(dtype=wp.int32), last_leaf: wp.array(dtype=wp.int32)):
    """Debug kernel to see what first and last leaf indices are being computed"""
    first = wp.bvh_get_group_root(bvh, group_id)
    # Note: This is a simplified version - in reality we'd need to extract the first/last computation
    first_leaf[0] = first
    last_leaf[0] = first  # Placeholder

def test_group_roots_simple():
    """Simple test to debug group root computation"""
    devices = get_test_devices()
    # Prefer CUDA for grouped BVH construction
    device = devices[0]
    for d in devices:
        if "cuda" in str(d):
            device = d
            break
    print(f"Testing on device: {device}")

    # Create a very simple test case
    # Group 0: 2 boxes at (0,0,0)-(1,1,1) and (2,2,2)-(3,3,3)
    # Group 1: 2 boxes at (4,4,4)-(5,5,5) and (6,6,6)-(7,7,7)
    
    lowers = np.array([
        [0.0, 0.0, 0.0],  # Group 0, Box 0
        [2.0, 2.0, 2.0],  # Group 0, Box 1
        [4.0, 4.0, 4.0],  # Group 1, Box 2
        [6.0, 6.0, 6.0],  # Group 1, Box 3
    ], dtype=np.float32)
    
    uppers = np.array([
        [1.0, 1.0, 1.0],  # Group 0, Box 0
        [3.0, 3.0, 3.0],  # Group 0, Box 1
        [5.0, 5.0, 5.0],  # Group 1, Box 2
        [7.0, 7.0, 7.0],  # Group 1, Box 3
    ], dtype=np.float32)
    
    groups = np.array([0, 0, 1, 1], dtype=np.int32)
    
    print("Test setup:")
    print(f"  Total bounds: {len(lowers)}")
    print(f"  Groups: {groups}")
    print(f"  Group 0 bounds: {np.sum(groups == 0)}")
    print(f"  Group 1 bounds: {np.sum(groups == 1)}")
    
    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    device_groups = wp.array(groups, dtype=wp.int32, device=device)

    print('\nBuilding BVH...')
    bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups)
    print('BVH built successfully')

    # Test group root computation
    print('\nTesting group roots...')
    group_roots = wp.zeros(shape=(2), dtype=wp.int32, device=device)
    
    wp.launch(compute_group_roots,
        dim=2,
        inputs=[bvh.id, group_roots],
        device=device)
    
    roots = group_roots.numpy()
    print(f"Group roots: {roots}")
    assert all(r != -1 for r in roots)
    assert len(set(roots)) == 2
    
    # Analyze results
    for i in range(2):
        root = roots[i]
        print(f"  Group {i}:")
        print(f"    Root index: {root}")
        if root == -1:
            print(f"    ERROR: Group {i} has root -1!")
        elif root == roots[1-i]:  # Same as other group
            print(f"    WARNING: Group {i} has same root as Group {1-i}!")
        else:
            print(f"    OK: Group {i} has unique root {root}")
    
    # Test individual group queries
    print('\nTesting individual group queries...')
    for group_id in range(2):
        first_leaf = wp.zeros(shape=(1), dtype=wp.int32, device=device)
        last_leaf = wp.zeros(shape=(1), dtype=wp.int32, device=device)
        
        wp.launch(debug_group_info,
            dim=1,
            inputs=[bvh.id, group_id, first_leaf, last_leaf],
            device=device)
        
        print(f"  Group {group_id} debug info:")
        print(f"    First leaf: {first_leaf.numpy()[0]}")

def test_group_roots_edge_cases():
    """Test edge cases for group root computation"""
    devices = get_test_devices()
    device = devices[0]
    for d in devices:
        if "cuda" in str(d):
            device = d
            break
    print(f"\nTesting edge cases on device: {device}")

    # Test case 1: Single group
    print("\nTest 1: Single group")
    lowers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    uppers = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    groups = np.array([0, 0], dtype=np.int32)
    
    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    device_groups = wp.array(groups, dtype=wp.int32, device=device)
    
    bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups)
    
    group_roots = wp.zeros(shape=(1), dtype=wp.int32, device=device)
    wp.launch(compute_group_roots, dim=1, inputs=[bvh.id, group_roots], device=device)
    print(f"Single group root: {group_roots.numpy()[0]}")

    # Test case 2: Many groups with single boxes
    print("\nTest 2: Many groups with single boxes")
    num_groups = 5
    lowers = np.array([[i*10.0, i*10.0, i*10.0] for i in range(num_groups)], dtype=np.float32)
    uppers = np.array([[i*10.0+1.0, i*10.0+1.0, i*10.0+1.0] for i in range(num_groups)], dtype=np.float32)
    groups = np.arange(num_groups, dtype=np.int32)
    
    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    device_groups = wp.array(groups, dtype=wp.int32, device=device)
    
    bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups)
    
    group_roots = wp.zeros(shape=(num_groups), dtype=wp.int32, device=device)
    wp.launch(compute_group_roots, dim=num_groups, inputs=[bvh.id, group_roots], device=device)
    roots = group_roots.numpy()
    print(f"Many groups roots: {roots}")
    assert all(r != -1 for r in roots)
    assert len(set(roots)) == num_groups


def test_group_roots_two_worlds():
    """Two adjacent worlds near origin; roots should be valid and unique."""
    devices = get_test_devices()
    device = devices[0]
    for d in devices:
        if "cuda" in str(d):
            device = d
            break

    lowers = np.array([
        [-1.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
    ], dtype=np.float32)
    uppers = np.array([
        [-0.5,  0.5,  0.5],
        [ 1.5,  0.5,  0.5],
    ], dtype=np.float32)
    groups = np.array([0, 1], dtype=np.int32)

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    device_groups = wp.array(groups, dtype=wp.int32, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups)

    group_roots = wp.zeros(shape=(2), dtype=wp.int32, device=device)
    wp.launch(compute_group_roots, dim=2, inputs=[bvh.id, group_roots], device=device)
    roots = group_roots.numpy()
    print(f"Two-world roots: {roots}")
    assert all(r != -1 for r in roots)
    assert len(set(roots)) == 2

if __name__ == "__main__":
    test_group_roots_simple()
    test_group_roots_edge_cases() 
    test_group_roots_two_worlds()