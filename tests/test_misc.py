import numpy as np
import ctypes

import warp as wp

class mat22(ctypes.Array):
    
    _length_ = 4
    _type_ = ctypes.c_float
    
    # def __init__(self):
    #     pass

    
m = mat22(0, 1, 2, 3)
print(m[0])
print(m[1])
print(m[2])
print(m[3])

wp.init()

class A:

    d = {}

    def __init__(self):
        self.d["hi"] = 1
        A.d["test"] = 3

A.d["ro"] = 2
print(A.d)

a = A()
print(a.d)

print(A.d)



# decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x
@wp.func
def quat_decompose(q: wp.quat):

    R = wp.mat33(
            wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
            wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
            wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))

    # https://www.sedris.org/wg8home/Documents/WG80485.pdf
    phi = wp.atan2(R[1, 2], R[2, 2])
    theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])

    return -wp.vec3(phi, theta, psi)


@wp.kernel
def test_decompose():

    theta_0 = 0.9
    theta_1 = -0.1
    theta_2 = 0.25

    axis_0 = wp.vec3(1.0, 0.0, 0.0)
    q_0 = wp.quat_from_axis_angle(axis_0, theta_0)

    axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
    #axis_1 = wp.vec3(0.0, 1.0, 0.0)
    q_1 = wp.quat_from_axis_angle(axis_1, theta_1)

    axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))
    #axis_2 = wp.vec3(0.0, 0.0, 1.0)
    q_2 = wp.quat_from_axis_angle(axis_2, theta_2)
   
    angles = wp.quat_decompose(q_2*q_1*q_0)

    print(axis_0)
    print(axis_1)
    print(axis_2)


    print(theta_0)
    print(theta_1)
    print(theta_2)
    print(angles)

    q_final = q_2*q_1*q_0
    q_reconstruct = wp.quat_from_axis_angle(axis_2, angles[2])*wp.quat_from_axis_angle(axis_1, angles[1])*wp.quat_from_axis_angle(axis_0, angles[0])

    print(q_final)
    print(q_reconstruct)






wp.launch(kernel=test_decompose, dim=1, inputs=[], device="cpu")

