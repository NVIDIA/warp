# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import math
import timeit
import cProfile
import numpy as np
from typing import Union, Tuple, Any

import warp as wp
import warp.types


def length(a):
    return np.linalg.norm(a)


def length_sq(a):
    return np.dot(a, a)


def cross(a, b):
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]), dtype=np.float32)


# NumPy has no normalize() method..
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# math utils
# def quat(i, j, k, w):
#     return np.array([i, j, k, w])


def quat_identity():
    return np.array((0.0, 0.0, 0.0, 1.0))


def quat_inverse(q):
    return np.array((-q[0], -q[1], -q[2], q[3]))


def quat_from_axis_angle(axis, angle):
    v = normalize(np.array(axis))

    half = angle * 0.5
    w = math.cos(half)

    sin_theta_over_two = math.sin(half)
    v *= sin_theta_over_two

    return np.array((v[0], v[1], v[2], w))


def quat_to_axis_angle(quat):
    w2 = quat[3] * quat[3]
    if w2 > 1 - 1e-7:
        return np.zeros(3), 0.0

    angle = 2 * np.arccos(quat[3])
    xyz = quat[:3] / np.sqrt(1 - w2)
    return xyz, angle


# quat_rotate a vector
def quat_rotate(q, x):
    x = np.array(x)
    axis = np.array((q[0], q[1], q[2]))
    return x * (2.0 * q[3] * q[3] - 1.0) + np.cross(axis, x) * q[3] * 2.0 + axis * np.dot(axis, x) * 2.0


# multiply two quats
def quat_multiply(a, b):
    return np.array(
        (
            a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - b[1] * a[2],
            a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - b[2] * a[0],
            a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - b[0] * a[1],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        )
    )


# convert to mat33
def quat_to_matrix(q):
    c1 = quat_rotate(q, np.array((1.0, 0.0, 0.0)))
    c2 = quat_rotate(q, np.array((0.0, 1.0, 0.0)))
    c3 = quat_rotate(q, np.array((0.0, 0.0, 1.0)))

    return np.array([c1, c2, c3]).T


def quat_rpy(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return (x, y, z, w)


def quat_from_matrix(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    h = 0.0

    if tr >= 0.0:
        h = math.sqrt(tr + 1.0)
        w = 0.5 * h
        h = 0.5 / h

        x = (m[2, 1] - m[1, 2]) * h
        y = (m[0, 2] - m[2, 0]) * h
        z = (m[1, 0] - m[0, 1]) * h

    else:
        i = 0
        if m[1, 1] > m[0, 0]:
            i = 1
        if m[2, 2] > m[i, i]:
            i = 2

        if i == 0:
            h = math.sqrt((m[0, 0] - (m[1, 1] + m[2, 2])) + 1.0)
            x = 0.5 * h
            h = 0.5 / h

            y = (m[0, 1] + m[1, 0]) * h
            z = (m[2, 0] + m[0, 2]) * h
            w = (m[2, 1] - m[1, 2]) * h

        elif i == 1:
            h = math.sqrt((m[1, 1] - (m[2, 2] + m[0, 0])) + 1.0)
            y = 0.5 * h
            h = 0.5 / h

            z = (m[1, 2] + m[2, 1]) * h
            x = (m[0, 1] + m[1, 0]) * h
            w = (m[0, 2] - m[2, 0]) * h

        elif i == 2:
            h = math.sqrt((m[2, 2] - (m[0, 0] + m[1, 1])) + 1.0)
            z = 0.5 * h
            h = 0.5 / h

            x = (m[2, 0] + m[0, 2]) * h
            y = (m[1, 2] + m[2, 1]) * h
            w = (m[1, 0] - m[0, 1]) * h

    return normalize(np.array([x, y, z, w]))


@wp.func
def quat_between_vectors(a: wp.vec3, b: wp.vec3) -> wp.quat:
    """
    Compute the quaternion that rotates vector a to vector b
    """
    a = wp.normalize(a)
    b = wp.normalize(b)
    c = wp.cross(a, b)
    d = wp.dot(a, b)
    q = wp.quat(c[0], c[1], c[2], 1.0 + d)
    return wp.normalize(q)


# rigid body transform


# def transform(x, r):
#     return (np.array(x), np.array(r))


def transform_identity():
    return wp.transform(np.array((0.0, 0.0, 0.0)), quat_identity())


# se(3) -> SE(3), Park & Lynch pg. 105, screw in [w, v] normalized form
def transform_exp(s, angle):
    w = np.array(s[0:3])
    v = np.array(s[3:6])

    if length(w) < 1.0:
        r = quat_identity()
    else:
        r = quat_from_axis_angle(w, angle)

    t = v * angle + (1.0 - math.cos(angle)) * np.cross(w, v) + (angle - math.sin(angle)) * np.cross(w, np.cross(w, v))

    return (t, r)


def transform_inverse(t):
    q_inv = quat_inverse(t.q)
    return wp.transform(-quat_rotate(q_inv, t.p), q_inv)


def transform_vector(t, v):
    return quat_rotate(t.q, v)


def transform_point(t, p):
    return np.array(t.p) + quat_rotate(t.q, p)


def transform_multiply(t, u):
    return wp.transform(quat_rotate(t.q, u.p) + t.p, quat_multiply(t.q, u.q))


# flatten an array of transforms (p,q) format to a 7-vector
def transform_flatten(t):
    return np.array([*t.p, *t.q])


# expand a 7-vec to a tuple of arrays
def transform_expand(t):
    return wp.transform(np.array(t[0:3]), np.array(t[3:7]))


# convert array of transforms to a array of 7-vecs
def transform_flatten_list(xforms):
    exp = lambda t: transform_flatten(t)
    return list(map(exp, xforms))


def transform_expand_list(xforms):
    exp = lambda t: transform_expand(t)
    return list(map(exp, xforms))


def transform_inertia(m, I, p, q):
    """
    Transforms the inertia tensor described by the given mass and 3x3 inertia
    matrix to a new frame described by the given position and orientation.
    """
    R = quat_to_matrix(q)

    # Steiner's theorem
    return R @ I @ R.T + m * (np.dot(p, p) * np.eye(3) - np.outer(p, p))


# spatial operators


# AdT
def spatial_adjoint(t):
    R = quat_to_matrix(t.q)
    w = skew(t.p)

    A = np.zeros((6, 6))
    A[0:3, 0:3] = R
    A[3:6, 0:3] = np.dot(w, R)
    A[3:6, 3:6] = R

    return A


# (AdT)^-T
def spatial_adjoint_dual(t):
    R = quat_to_matrix(t.q)
    w = skew(t.p)

    A = np.zeros((6, 6))
    A[0:3, 0:3] = R
    A[0:3, 3:6] = np.dot(w, R)
    A[3:6, 3:6] = R

    return A


# AdT*s
def transform_twist(t_ab, s_b):
    return np.dot(spatial_adjoint(t_ab), s_b)


# AdT^{-T}*s
def transform_wrench(t_ab, f_b):
    return np.dot(spatial_adjoint_dual(t_ab), f_b)


# transform spatial inertia (6x6) in b frame to a frame
def transform_spatial_inertia(t_ab, I_b):
    t_ba = transform_inverse(t_ab)

    # todo: write specialized method
    I_a = np.dot(np.dot(spatial_adjoint(t_ba).T, I_b), spatial_adjoint(t_ba))
    return I_a


def translate_twist(p_ab, s_b):
    w = s_b[0:3]
    v = np.cross(p_ab, s_b[0:3]) + s_b[3:6]

    return np.array((*w, *v))


def translate_wrench(p_ab, s_b):
    w = s_b[0:3] + np.cross(p_ab, s_b[3:6])
    v = s_b[3:6]

    return np.array((*w, *v))


# def spatial_vector(v=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
#     return np.array(v)


# ad_V pg. 289 L&P, pg. 25 Featherstone
def spatial_cross(a, b):
    w = np.cross(a[0:3], b[0:3])
    v = np.cross(a[3:6], b[0:3]) + np.cross(a[0:3], b[3:6])

    return np.array((*w, *v))


# ad_V^T pg. 290 L&P,  pg. 25 Featurestone, note this does not includes the sign flip in the definition
def spatial_cross_dual(a, b):
    w = np.cross(a[0:3], b[0:3]) + np.cross(a[3:6], b[3:6])
    v = np.cross(a[0:3], b[3:6])

    return np.array((*w, *v))


def spatial_dot(a, b):
    return np.dot(a, b)


def spatial_outer(a, b):
    return np.outer(a, b)


# def spatial_matrix():
#     return np.zeros((6, 6))


def spatial_matrix_from_inertia(I, m):
    G = spatial_matrix()

    G[0:3, 0:3] = I
    G[3, 3] = m
    G[4, 4] = m
    G[5, 5] = m

    return G


# solves x = I^(-1)b
def spatial_solve(I, b):
    return np.dot(np.linalg.inv(I), b)


# helper to retrive body angular velocity from a twist v_s in se(3)
def get_body_angular_velocity(v_s):
    return v_s[0:3]


# helper to compute velocity of a point p on a body given it's spatial twist v_s
def get_body_linear_velocity(v_s, p):
    dpdt = v_s[3:6] + np.cross(v_s[0:3], p)
    return dpdt


# helper to build a body twist given the angular and linear velocity of
# the center of mass specified in the world frame, returns the body
# twist with respect to the origin (v_s)
def get_body_twist(w_m, v_m, p_m):
    lin = v_m + np.cross(p_m, w_m)
    return (*w_m, *lin)


def array_scan(in_array, out_array, inclusive=True):
    if in_array.device != out_array.device:
        raise RuntimeError("Array storage devices do not match")

    if in_array.size != out_array.size:
        raise RuntimeError("Array storage sizes do not match")

    if in_array.dtype != out_array.dtype:
        raise RuntimeError("Array data types do not match")

    from warp.context import runtime

    if in_array.device.is_cpu:
        if in_array.dtype == wp.int32:
            runtime.core.array_scan_int_host(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        elif in_array.dtype == wp.float32:
            runtime.core.array_scan_float_host(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        else:
            raise RuntimeError("Unsupported data type")
    elif in_array.device.is_cuda:
        if in_array.dtype == wp.int32:
            runtime.core.array_scan_int_device(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        elif in_array.dtype == wp.float32:
            runtime.core.array_scan_float_device(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        else:
            raise RuntimeError("Unsupported data type")


def radix_sort_pairs(keys, values, count: int):
    if keys.device != values.device:
        raise RuntimeError("Array storage devices do not match")

    if keys.size < 2 * count or values.size < 2 * count:
        raise RuntimeError("Array storage must be large enough to contain 2*count elements")

    from warp.context import runtime

    if keys.device.is_cpu:
        if keys.dtype == wp.int32 and values.dtype == wp.int32:
            runtime.core.radix_sort_pairs_int_host(keys.ptr, values.ptr, count)
        else:
            raise RuntimeError("Unsupported data type")
    elif keys.device.is_cuda:
        if keys.dtype == wp.int32 and values.dtype == wp.int32:
            runtime.core.radix_sort_pairs_int_device(keys.ptr, values.ptr, count)
        else:
            raise RuntimeError("Unsupported data type")


def runlength_encode(values, run_values, run_lengths, run_count=None, value_count=None):
    if run_values.device != values.device or run_lengths.device != values.device:
        raise RuntimeError("Array storage devices do not match")

    if value_count is None:
        value_count = values.size

    if run_values.size < value_count or run_lengths.size < value_count:
        raise RuntimeError("Output array storage sizes must be at least equal to value_count")

    if values.dtype != run_values.dtype:
        raise RuntimeError("values and run_values data types do not match")

    if run_lengths.dtype != wp.int32:
        raise RuntimeError("run_lengths array must be of type int32")

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if run_count is None:
        host_return = True
        run_count = wp.empty(shape=(1,), dtype=int, device=values.device)
    else:
        host_return = False
        if run_count.device != values.device:
            raise RuntimeError("run_count storage devices does not match other arrays")
        if run_count.dtype != wp.int32:
            raise RuntimeError("run_count array must be of type int32")

    from warp.context import runtime

    if values.device.is_cpu:
        if values.dtype == wp.int32:
            runtime.core.runlength_encode_int_host(
                values.ptr, run_values.ptr, run_lengths.ptr, run_count.ptr, value_count
            )
        else:
            raise RuntimeError("Unsupported data type")
    elif values.device.is_cuda:
        if values.dtype == wp.int32:
            runtime.core.runlength_encode_int_device(
                values.ptr, run_values.ptr, run_lengths.ptr, run_count.ptr, value_count
            )
        else:
            raise RuntimeError("Unsupported data type")

    if host_return:
        return int(run_count.numpy()[0])


def array_sum(values, out=None, value_count=None, axis=None):
    if value_count is None:
        if axis is None:
            value_count = values.size
        else:
            value_count = values.shape[axis]

    if axis is None:
        output_shape = (1,)
    else:

        def output_dim(ax, dim):
            return 1 if ax == axis else dim

        output_shape = tuple(output_dim(ax, dim) for ax, dim in enumerate(values.shape))

    type_length = wp.types.type_length(values.dtype)
    scalar_type = wp.types.type_scalar_type(values.dtype)

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if out is None:
        host_return = True
        out = wp.empty(shape=output_shape, dtype=values.dtype, device=values.device)
    else:
        host_return = False
        if out.device != values.device:
            raise RuntimeError("out storage device should match values array")
        if out.dtype != values.dtype:
            raise RuntimeError(f"out array should have type {values.dtype.__name__}")
        if out.shape != output_shape:
            raise RuntimeError(f"out array should have shape {output_shape}")

    from warp.context import runtime

    if values.device.is_cpu:
        if scalar_type == wp.float32:
            native_func = runtime.core.array_sum_float_host
        elif scalar_type == wp.float64:
            native_func = runtime.core.array_sum_double_host
        else:
            raise RuntimeError("Unsupported data type")
    elif values.device.is_cuda:
        if scalar_type == wp.float32:
            native_func = runtime.core.array_sum_float_device
        elif scalar_type == wp.float64:
            native_func = runtime.core.array_sum_double_device
        else:
            raise RuntimeError("Unsupported data type")

    if axis is None:
        stride = wp.types.type_size_in_bytes(values.dtype)
        native_func(values.ptr, out.ptr, value_count, stride, type_length)

        if host_return:
            return out.numpy()[0]
    else:
        stride = values.strides[axis]
        for idx in np.ndindex(output_shape):
            out_offset = sum(i * s for i, s in zip(idx, out.strides))
            val_offset = sum(i * s for i, s in zip(idx, values.strides))

            native_func(
                values.ptr + val_offset,
                out.ptr + out_offset,
                value_count,
                stride,
                type_length,
            )

        if host_return:
            return out


def array_inner(a, b, out=None, count=None, axis=None):
    if a.size != b.size:
        raise RuntimeError("Array storage sizes do not match")

    if a.device != b.device:
        raise RuntimeError("Array storage sizes do not match")

    if a.dtype != b.dtype:
        raise RuntimeError("Array data types do not match")

    if count is None:
        if axis is None:
            count = a.size
        else:
            count = a.shape[axis]

    if axis is None:
        output_shape = (1,)
    else:

        def output_dim(ax, dim):
            return 1 if ax == axis else dim

        output_shape = tuple(output_dim(ax, dim) for ax, dim in enumerate(a.shape))

    type_length = wp.types.type_length(a.dtype)
    scalar_type = wp.types.type_scalar_type(a.dtype)

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if out is None:
        host_return = True
        out = wp.empty(shape=output_shape, dtype=scalar_type, device=a.device)
    else:
        host_return = False
        if out.device != a.device:
            raise RuntimeError("out storage device should match values array")
        if out.dtype != scalar_type:
            raise RuntimeError(f"out array should have type {scalar_type.__name__}")
        if out.shape != output_shape:
            raise RuntimeError(f"out array should have shape {output_shape}")

    from warp.context import runtime

    if a.device.is_cpu:
        if scalar_type == wp.float32:
            native_func = runtime.core.array_inner_float_host
        elif scalar_type == wp.float64:
            native_func = runtime.core.array_inner_double_host
        else:
            raise RuntimeError("Unsupported data type")
    elif a.device.is_cuda:
        if scalar_type == wp.float32:
            native_func = runtime.core.array_inner_float_device
        elif scalar_type == wp.float64:
            native_func = runtime.core.array_inner_double_device
        else:
            raise RuntimeError("Unsupported data type")

    if axis is None:
        stride_a = wp.types.type_size_in_bytes(a.dtype)
        stride_b = wp.types.type_size_in_bytes(b.dtype)
        native_func(a.ptr, b.ptr, out.ptr, count, stride_a, stride_b, type_length)

        if host_return:
            return out.numpy()[0]
    else:
        stride_a = a.strides[axis]
        stride_b = b.strides[axis]

        for idx in np.ndindex(output_shape):
            out_offset = sum(i * s for i, s in zip(idx, out.strides))
            a_offset = sum(i * s for i, s in zip(idx, a.strides))
            b_offset = sum(i * s for i, s in zip(idx, b.strides))

            native_func(
                a.ptr + a_offset,
                b.ptr + b_offset,
                out.ptr + out_offset,
                count,
                stride_a,
                stride_b,
                type_length,
            )

        if host_return:
            return out


_copy_kernel_cache = dict()


def array_cast(in_array, out_array, count=None):
    def make_copy_kernel(dest_dtype, src_dtype):
        import re
        import warp.context

        def copy_kernel(
            dest: Any,
            src: Any,
        ):
            dest[wp.tid()] = dest_dtype(src[wp.tid()])

        module = wp.get_module(copy_kernel.__module__)
        key = f"{copy_kernel.__name__}_{warp.context.type_str(src_dtype)}_{warp.context.type_str(dest_dtype)}"
        key = re.sub("[^0-9a-zA-Z_]+", "", key)

        if key not in _copy_kernel_cache:
            _copy_kernel_cache[key] = wp.Kernel(func=copy_kernel, key=key, module=module)
        return _copy_kernel_cache[key]

    if in_array.device != out_array.device:
        raise RuntimeError("Array storage devices do not match")

    in_array_data_shape = getattr(in_array.dtype, "_shape_", ())
    out_array_data_shape = getattr(out_array.dtype, "_shape_", ())

    if in_array.ndim != out_array.ndim or in_array_data_shape != out_array_data_shape:
        # Number of dimensions or data type shape do not match.
        # Flatten arrays and do cast at the scalar level
        in_array = in_array.flatten()
        out_array = out_array.flatten()

        in_array_data_length = warp.types.type_length(in_array.dtype)
        out_array_data_length = warp.types.type_length(out_array.dtype)
        in_array_scalar_type = wp.types.type_scalar_type(in_array.dtype)
        out_array_scalar_type = wp.types.type_scalar_type(out_array.dtype)

        in_array = wp.array(
            data=None,
            ptr=in_array.ptr,
            capacity=in_array.capacity,
            owner=False,
            device=in_array.device,
            dtype=in_array_scalar_type,
            shape=in_array.shape[0] * in_array_data_length,
        )

        out_array = wp.array(
            data=None,
            ptr=out_array.ptr,
            capacity=out_array.capacity,
            owner=False,
            device=out_array.device,
            dtype=out_array_scalar_type,
            shape=out_array.shape[0] * out_array_data_length,
        )

        if count is not None:
            count *= in_array_data_length

    if count is None:
        count = in_array.size

    if in_array.ndim == 1:
        dim = count
    elif count < in_array.size:
        raise RuntimeError("Partial cast is not supported for arrays with more than one dimension")
    else:
        dim = in_array.shape

    if in_array.dtype == out_array.dtype:
        # Same data type, can simply copy
        wp.copy(dest=out_array, src=in_array, count=count)
    else:
        copy_kernel = make_copy_kernel(src_dtype=in_array.dtype, dest_dtype=out_array.dtype)
        wp.launch(kernel=copy_kernel, dim=dim, inputs=[out_array, in_array], device=out_array.device)


# code snippet for invoking cProfile
# cp = cProfile.Profile()
# cp.enable()
# for i in range(1000):
#     self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

# cp.disable()
# cp.print_stats(sort='tottime')
# exit(0)


# represent an edge between v0, v1 with connected faces f0, f1, and opposite vertex o0, and o1
# winding is such that first tri can be reconstructed as {v0, v1, o0}, and second tri as { v1, v0, o1 }
class MeshEdge:
    def __init__(self, v0, v1, o0, o1, f0, f1):
        self.v0 = v0  # vertex 0
        self.v1 = v1  # vertex 1
        self.o0 = o0  # opposite vertex 1
        self.o1 = o1  # opposite vertex 2
        self.f0 = f0  # index of tri1
        self.f1 = f1  # index of tri2


class MeshAdjacency:
    def __init__(self, indices, num_tris):
        # map edges (v0, v1) to faces (f0, f1)
        self.edges = {}
        self.indices = indices

        for index, tri in enumerate(indices):
            self.add_edge(tri[0], tri[1], tri[2], index)
            self.add_edge(tri[1], tri[2], tri[0], index)
            self.add_edge(tri[2], tri[0], tri[1], index)

    def add_edge(self, i0, i1, o, f):  # index1, index2, index3, index of triangle
        key = (min(i0, i1), max(i0, i1))
        edge = None

        if key in self.edges:
            edge = self.edges[key]

            if edge.f1 != -1:
                print("Detected non-manifold edge")
                return
            else:
                # update other side of the edge
                edge.o1 = o
                edge.f1 = f
        else:
            # create new edge with opposite yet to be filled
            edge = MeshEdge(i0, i1, o, -1, f, -1)

        self.edges[key] = edge

    def opposite_vertex(self, edge):
        pass


def mem_report():
    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation"""
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            # print('%s\t\t%s\t\t%.2f' % (
            #     element_type,
            #     size,
            #     mem) )
        print("Type: %s Total Tensors: %d \tUsed Memory Space: %.2f MBytes" % (mem_type, total_numel, total_mem))

    import gc
    import torch

    gc.collect()

    LEN = 65
    objects = gc.get_objects()
    # print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, "GPU")
    _mem_report(host_tensors, "CPU")
    print("=" * LEN)


def lame_parameters(E, nu):
    l = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    return (l, mu)


class ScopedDevice:
    def __init__(self, device):
        self.device = wp.get_device(device)

    def __enter__(self):
        # save the previous default device
        self.saved_device = self.device.runtime.default_device

        # make this the default device
        self.device.runtime.default_device = self.device

        # make it the current CUDA device so that device alias "cuda" will evaluate to this device
        self.device.context_guard.__enter__()

        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        # restore original CUDA context
        self.device.context_guard.__exit__(exc_type, exc_value, traceback)

        # restore original target device
        self.device.runtime.default_device = self.saved_device


class ScopedStream:
    def __init__(self, stream):
        self.stream = stream
        if stream is not None:
            self.device = stream.device
            self.device_scope = ScopedDevice(self.device)

    def __enter__(self):
        if self.stream is not None:
            self.device_scope.__enter__()
            self.saved_stream = self.device.stream
            self.device.stream = self.stream

        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self.device.stream = self.saved_stream
            self.device_scope.__exit__(exc_type, exc_value, traceback)


# timer utils
class ScopedTimer:
    indent = -1

    enabled = True

    def __init__(
        self,
        name,
        active=True,
        print=True,
        detailed=False,
        dict=None,
        use_nvtx=False,
        color="rapids",
        synchronize=False,
    ):
        """Context manager object for a timer

        Parameters:
            name (str): Name of timer
            active (bool): Enables this timer
            print (bool): At context manager exit, print elapsed time to sys.stdout
            detailed (bool): Collects additional profiling data using cProfile and calls ``print_stats()`` at context exit
            dict (dict): A dictionary of lists to which the elapsed time will be appended using ``name`` as a key
            use_nvtx (bool): If true, timing functionality is replaced by an NVTX range
            color (int or str): ARGB value (e.g. 0x00FFFF) or color name (e.g. 'cyan') associated with the NVTX range
            synchronize (bool): Synchronize the CPU thread with any outstanding CUDA work to return accurate GPU timings

        Attributes:
            elapsed (float): The duration of the ``with`` block used with this object
        """
        self.name = name
        self.active = active and self.enabled
        self.print = print
        self.detailed = detailed
        self.dict = dict
        self.use_nvtx = use_nvtx
        self.color = color
        self.synchronize = synchronize
        self.elapsed = 0.0

        if self.dict is not None:
            if name not in self.dict:
                self.dict[name] = []

    def __enter__(self):
        if self.active:
            if self.synchronize:
                wp.synchronize()

            if self.use_nvtx:
                import nvtx

                self.nvtx_range_id = nvtx.start_range(self.name, color=self.color)
                return

            self.start = timeit.default_timer()
            ScopedTimer.indent += 1

            if self.detailed:
                self.cp = cProfile.Profile()
                self.cp.clear()
                self.cp.enable()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            if self.synchronize:
                wp.synchronize()

            if self.use_nvtx:
                import nvtx

                nvtx.end_range(self.nvtx_range_id)
                return

            if self.detailed:
                self.cp.disable()
                self.cp.print_stats(sort="tottime")

            self.elapsed = (timeit.default_timer() - self.start) * 1000.0

            if self.dict is not None:
                self.dict[self.name].append(self.elapsed)

            indent = ""
            for i in range(ScopedTimer.indent):
                indent += "\t"

            if self.print:
                print("{}{} took {:.2f} ms".format(indent, self.name, self.elapsed))

            ScopedTimer.indent -= 1
