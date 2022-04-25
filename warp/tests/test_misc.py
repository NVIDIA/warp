import warp as wp
import numpy as np

from warp.types import float32

wp.init()

a1 = np.array([1.0, 2.0, 3.0, 4.0])

@wp.kernel
def test1d(a: wp.array(dtype=float32)):

    print(a[wp.tid()])

a2 = np.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
               [9.0, 10.0, 11.0, 12.0],
               [13.0, 14.0, 15.0, 16.0]])

@wp.kernel
def test2d(a: wp.array(dtype=float, ndim=2)):

    i = wp.tid()//4
    j = wp.tid()%4


    print(a[i,j])
    print(a[i][j])


device = "cpu"

arr1d = wp.array(a1, dtype=float, device=device)
arr2d = wp.array(a2, dtype=float, device=device)

wp.launch(test1d, dim=arr1d.size, inputs=[arr1d], device=device)
wp.launch(test2d, dim=arr2d.size, inputs=[arr2d], device=device)