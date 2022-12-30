# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from . context import add_builtin

from warp.types import *

from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from typing import Callable


def sametype_value_func(default):
    def fn(args,_):
        if args is None:
            return default
        if not all( types_equal(args[0].type,a.type) for a in args[1:] ):
            raise RuntimeError(f"Input types must be exactly the same, {[a.type for a in args]}")
        return args[0].type
    return fn

#---------------------------------
# Scalar Math

add_builtin("min", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar), doc="Return the minimum of two integers.", group="Scalar Math")

add_builtin("max", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar), doc="Return the maximum of two integers.", group="Scalar Math")

add_builtin("clamp", input_types={"x": Scalar, "a": Scalar, "b": Scalar}, value_func=sametype_value_func(Scalar), doc="Clamp the value of x to the range [a, b].", group="Scalar Math")

add_builtin("abs", input_types={"x": Scalar}, value_func=sametype_value_func(Scalar), doc="Return the absolute value of x.", group="Scalar Math")
add_builtin("sign", input_types={"x": Scalar}, value_func=sametype_value_func(Scalar), doc="Return -1 if x < 0, return 1 otherwise.", group="Scalar Math")

add_builtin("step", input_types={"x": Scalar}, value_func=sametype_value_func(Scalar), doc="Return 1.0 if x < 0.0, return 0.0 otherwise.", group="Scalar Math")
add_builtin("nonzero", input_types={"x": Scalar}, value_func=sametype_value_func(Scalar), doc="Return 1.0 if x is not equal to zero, return 0.0 otherwise.", group="Scalar Math")

for T in [float16,float32,float64]:
    add_builtin("sin", input_types={"x": T}, value_type=T, doc="Return the sine of x in radians.", group="Scalar Math")
    add_builtin("cos", input_types={"x": T}, value_type=T, doc="Return the cosine of x in radians.", group="Scalar Math")
    add_builtin("acos", input_types={"x": T}, value_type=T, doc="Return arccos of x in radians. Inputs are automatically clamped to [-1.0, 1.0].", group="Scalar Math")
    add_builtin("asin", input_types={"x": T}, value_type=T, doc="Return arcsin of x in radians. Inputs are automatically clamped to [-1.0, 1.0].", group="Scalar Math")
    add_builtin("sqrt", input_types={"x": T}, value_type=T, doc="Return the sqrt of x, where x is positive.", group="Scalar Math")
    add_builtin("tan", input_types={"x": T}, value_type=T, doc="Return tangent of x in radians.", group="Scalar Math")
    add_builtin("atan", input_types={"x": T}, value_type=T, doc="Return arctan of x.", group="Scalar Math")
    add_builtin("atan2", input_types={"y": T, "x": T}, value_type=T, doc="Return atan2 of x.", group="Scalar Math")
    add_builtin("sinh", input_types={"x": T}, value_type=T, doc="Return the sinh of x.", group="Scalar Math")
    add_builtin("cosh", input_types={"x": T}, value_type=T, doc="Return the cosh of x.", group="Scalar Math")
    add_builtin("tanh", input_types={"x": T}, value_type=T, doc="Return the tanh of x.", group="Scalar Math")

    add_builtin("log", input_types={"x": T}, value_type=T, doc="Return the natural log (base-e) of x, where x is positive.", group="Scalar Math")
    add_builtin("log2", input_types={"x": T}, value_type=T, doc="Return the natural log (base-2) of x, where x is positive.", group="Scalar Math")
    add_builtin("log10", input_types={"x": T}, value_type=T, doc="Return the natural log (base-10) of x, where x is positive.", group="Scalar Math")
    add_builtin("exp", input_types={"x": T}, value_type=T, doc="Return base-e exponential, e^x.", group="Scalar Math")
    add_builtin("pow", input_types={"x": T, "y": T}, value_type=T, doc="Return the result of x raised to power of y.", group="Scalar Math")

    add_builtin("round", input_types={"x": T}, value_type=T, group="Scalar Math",
        doc="""Calculate the nearest integer value, rounding halfway cases away from zero.
    This is the most intuitive form of rounding in the colloquial sense, but can be slower than other options like ``warp.rint()``.
    Differs from ``numpy.round()``, which behaves the same way as ``numpy.rint()``.""")

    add_builtin("rint", input_types={"x": T}, value_type=T, group="Scalar Math",
        doc="""Calculate the nearest integer value, rounding halfway cases to nearest even integer.
    It is generally faster than ``warp.round()``.
    Equivalent to ``numpy.rint()``.""")

    add_builtin("trunc", input_types={"x": T}, value_type=T, group="Scalar Math",
        doc="""Calculate the nearest integer that is closer to zero than x.
    In other words, it discards the fractional part of x.
    It is similar to casting ``float(int(x))``, but preserves the negative sign when x is in the range [-0.0, -1.0).
    Equivalent to ``numpy.trunc()`` and ``numpy.fix()``.""")

    add_builtin("floor", input_types={"x": T}, value_type=T, group="Scalar Math",
        doc="""Calculate the largest integer that is less than or equal to x.""")

    add_builtin("ceil", input_types={"x": T}, value_type=T, group="Scalar Math",
        doc="""Calculate the smallest integer that is greater than or equal to x.""")

def infer_scalar_type(args):
    if args is None:
        return Scalar

    def iterate_scalar_types(args):
        for a in args:
            if hasattr(a.type,"_wp_scalar_type_"):
                yield a.type._wp_scalar_type_
            elif a.type in scalar_types:
                yield a.type

    scalarTypes = set(iterate_scalar_types(args))
    if len(scalarTypes) > 1:
        raise RuntimeError(f"Couldn't figure out return type as arguments have multiple precisions: {list(scalarTypes)}")
    return list(scalarTypes)[0]


def sametype_scalar_value_func(args,_):
    if args is None:
        return Scalar
    if not all( types_equal(args[0].type,a.type) for a in args[1:] ):
        raise RuntimeError(f"Input types must be exactly the same, {[a.type for a in args]}")
    
    return infer_scalar_type(args)

#---------------------------------
# Vector Math

add_builtin("dot", input_types={"x": vec(length=Any,type=Scalar), "y": vec(length=Any,type=Scalar)}, value_func=sametype_scalar_value_func, group="Vector Math",
    doc="Compute the dot product between two vectors.")
add_builtin("ddot", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": mat(shape=(Any,Any),type=Scalar)}, value_func=sametype_scalar_value_func, group="Vector Math",
    doc="Compute the double dot product between two matrices.")

def value_func_outer(args,_):

    if args is None:
        return mat(shape=(Any,Any),type=Scalar)
    
    scalarType = infer_scalar_type(args)
    vectorLengths = [i.type._length_ for i in args]
    return mat(shape=(vectorLengths),type=scalarType)

add_builtin("outer", input_types={"x": vec(length=Any,type=Scalar), "y": vec(length=Any,type=Scalar)}, value_func=value_func_outer, group="Vector Math",
    doc="Compute the outer product x*y^T for two vec2 objects.")

add_builtin("cross", input_types={"x": vec(length=3,type=Scalar), "y": vec(length=3,type=Scalar)}, value_func=sametype_value_func(vec(length=3,type=Scalar)), group="Vector Math",
    doc="Compute the cross product of two 3d vectors.")
add_builtin("skew", input_types={"x": vec(length=3,type=Scalar)}, value_func=lambda args,_: mat(shape=(3,3),type=args[0].type._wp_scalar_type_), group="Vector Math",
    doc="Compute the skew symmetric matrix for a 3d vector.")

add_builtin("length", input_types={"x": vec(length=Any,type=Scalar)}, value_func=sametype_scalar_value_func, group="Vector Math",
    doc="Compute the length of a vector.")
add_builtin("length", input_types={"x": quaternion(type=Scalar)}, value_func=sametype_scalar_value_func, group="Vector Math",
    doc="Compute the length of a quaternion.")
add_builtin("length_sq", input_types={"x": vec(length=Any,type=Scalar)}, value_func=sametype_scalar_value_func, group="Vector Math",
    doc="Compute the squared length of a 2d vector.")
add_builtin("length_sq", input_types={"x": quaternion(type=Scalar)}, value_func=sametype_scalar_value_func, group="Vector Math",
    doc="Compute the squared length of a quaternion.")
add_builtin("normalize", input_types={"x": vec(length=Any,type=Scalar)}, value_func=sametype_value_func(vec(length=Any,type=Scalar)), group="Vector Math",
    doc="Compute the normalized value of x, if length(x) is 0 then the zero vector is returned.")
add_builtin("normalize", input_types={"x": quaternion(type=Scalar)}, value_func=sametype_value_func(quaternion(type=Scalar)), group="Vector Math",
    doc="Compute the normalized value of x, if length(x) is 0 then the zero quat is returned.")

add_builtin("transpose", input_types={"m": mat(shape=(Any,Any),type=Scalar)}, value_func=lambda args,_: mat(shape=(args[0].type._shape_[1],args[0].type._shape_[0]),type=args[0].type._wp_scalar_type_), group="Vector Math",
    doc="Return the transpose of the matrix m")
add_builtin("transpose", input_types={"m": spatial_matrix_t(type=Scalar)}, value_func=sametype_value_func(spatial_matrix_t(type=Scalar)), group="Vector Math",
    doc="Return the transpose of the matrix m")

def value_func_mat_inv(args,_):
    if args is None:
        return mat(shape=(Any,Any),type=Scalar)
    if args[0].type._shape_[0] != args[0].type._shape_[1]:
        raise RuntimeError(f"Matrix shape is {args[0].type._shape_}. Cannot invert non square matrices")
    if args[0].type._shape_[0] not in [2,3,4]:
        raise RuntimeError(f"Matrix inverse only supported for sizes 2,3 and 4 right now")
    return args[0].type

add_builtin("inverse", input_types={"m": mat(shape=(Any,Any),type=Scalar)}, value_func=value_func_mat_inv, group="Vector Math",
    doc="Return the inverse of the matrix m")

def value_func_mat_det(args,_):
    if args is None:
        return Scalar
    if args[0].type._shape_[0] != args[0].type._shape_[1]:
        raise RuntimeError(f"Matrix shape is {args[0].type._shape_}. Cannot find the determinant of non square matrices")
    if args[0].type._shape_[0] not in [2,3,4]:
        raise RuntimeError(f"Matrix determinant only supported for sizes 2,3 and 4 right now")
    return args[0].type._wp_scalar_type_

add_builtin("determinant", input_types={"m": mat(shape=(Any,Any),type=Scalar)}, value_func=value_func_mat_det, group="Vector Math",
    doc="Return the determinant of the matrix m")

def value_func_mat_trace(args,_):
    if args is None:
        return Scalar
    if args[0].type._shape_[0] != args[0].type._shape_[1]:
        raise RuntimeError(f"Matrix shape is {args[0].type._shape_}. Cannot find the trace of non square matrices")
    return args[0].type._wp_scalar_type_

add_builtin("trace", input_types={"m": mat(shape=(Any,Any),type=Scalar)}, value_func=value_func_mat_trace, group="Vector Math",
    doc="Return the trace of the matrix m")

add_builtin("diag", input_types={"d": vec(length=Any,type=Scalar)}, value_func=lambda args,_: mat(shape=(args[0].type._length_,args[0].type._length_),type=args[0].type._wp_scalar_type_), group="Vector Math",
    doc="Returns a matrix with the components of the vector d on the diagonal")

add_builtin("cw_mul", input_types={"x": vec(length=Any,type=Scalar), "y": vec(length=Any,type=Scalar)}, value_func=sametype_value_func(vec(length=Any,type=Scalar)), group="Vector Math",
     doc="Component wise multiply of two 2d vectors.")
add_builtin("cw_div", input_types={"x": vec(length=Any,type=Scalar), "y": vec(length=Any,type=Scalar)}, value_func=sametype_value_func(vec(length=Any,type=Scalar)), group="Vector Math",
    doc="Component wise division of two 2d vectors.")

add_builtin("cw_mul", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": mat(shape=(Any,Any),type=Scalar)}, value_func=sametype_value_func(mat(shape=(Any,Any),type=Scalar)), group="Vector Math",
     doc="Component wise multiply of two 2d vectors.")
add_builtin("cw_div", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": mat(shape=(Any,Any),type=Scalar)}, value_func=sametype_value_func(mat(shape=(Any,Any),type=Scalar)), group="Vector Math",
    doc="Component wise division of two 2d vectors.")


# type construtors between all storage / compute types 
scalar_types_all = [*scalar_types, int, float]
for t in scalar_types_all:
    for u in scalar_types_all:
        add_builtin(t.__name__, input_types={"u": u}, value_type=t, doc="", hidden=True, group="Scalar Math", export=False)

def vec_constructor_value_func(args,templates):
    if templates is None:
        return vec(length=Any,type=Scalar)
    
    veclen,vectype = templates
    if len(args) and args[0].type != vectype:
        raise RuntimeError("Wrong scalar type for vec<{}> constructor".format( ",".join(map(str,templates)) ))
    
    return vec(length=veclen,type=vectype)

add_builtin("vec", input_types={}, value_func=vec_constructor_value_func, doc="Construct a zero-initialized vector.", group="Vector Math", export=False)
add_builtin("vec", input_types={"s": Scalar}, value_func=vec_constructor_value_func, doc="Construct a vector with all components set to s.", group="Vector Math", export=False)

def vec_component_constructor_value_func(args,templates):
    if templates is None:
        return vec(length=Any,type=Scalar)
    
    # check to make sure the user entered the correct number and type of components given the template arguments:
    veclen,vectype = templates
    if veclen != len(args):
        raise RuntimeError("Wrong number of arguments for vec<{}> constructor".format( ",".join(map(str,templates)) ))
    if not all( vectype == a.type for a in args ):
        raise RuntimeError("Wrong type of arguments for vec<{}> constructor".format( ",".join(map(str,templates)) ))
    return vec(length=veclen,type=vectype)

add_builtin("vec", input_types={}, variadic=True, initializer_list=True, value_func=vec_component_constructor_value_func, doc="Construct a vector from scalar compontents.", group="Vector Math", export=False)

def mat_constructor_value_func(args,templates):
    if templates is None:
        return mat(shape=(Any,Any),type=Scalar)
    
    matrows,matcols,mattype = templates
    if len(args) and args[0].type != mattype:
        raise RuntimeError("Wrong scalar type for mat<{}> constructor".format( ",".join(map(str,templates)) ))
        
    return mat(shape=(matrows,matcols),type=mattype)

add_builtin("mat", input_types={}, value_func=mat_constructor_value_func, doc="Construct a zero matrix", group="Vector Math", export=False)
add_builtin("mat", input_types={"s": Scalar}, value_func=mat_constructor_value_func, doc="Construct a matrix with all components set to s", group="Vector Math", export=False)

# The following constructor accepts both scalars and vectors
def mat_initlist_constructor_value_func(args,templates):

    if templates is None:
        return mat(shape=(Any,Any),type=Scalar)
    
    matrows,matcols,mattype = templates
    if infer_scalar_type(args) != mattype:
        raise RuntimeError("Wrong scalar type for mat<{}> constructor".format( ",".join(map(str,templates)) ))

    types = [a.type for a in args]
    if all( hasattr(a,"_wp_generic_type_str_") and a._wp_generic_type_str_ == "vec" for a in types ):
        cols = len(types)
        if matcols != cols:
            raise RuntimeError("Wrong number of vectors when attempting to construct a matrix with column vectors")
        
        if not all( a._length_ == matrows for a in types ):
            raise RuntimeError("Wrong vector row count when attempting to construct a matrix with column vectors")

    else:
        ncmps = matrows * matcols
        if len(types) != ncmps:
            raise RuntimeError("Wrong number of scalars when attempting to construct a matrix from a list of components")

    return mat(shape=(matrows,matcols),type=mattype)

add_builtin("mat", input_types={}, variadic=True, initializer_list=True, value_func=mat_initlist_constructor_value_func, doc="Construct a matrix from components or column vectors.", group="Vector Math", export=False)

def mat_quat_constructor_value_func(args,templates):
    
    if templates is None:
        return mat(shape=(Any,Any),type=Scalar)
    
    matrows,matcols,mattype = templates
    if (matrows,matcols) != (4,4):
        raise RuntimeError("Can only construct 4x4 matrices with position, rotation and scale")
    if infer_scalar_type(args) != mattype:
        raise RuntimeError("Wrong scalar type for mat<{}> constructor".format( ",".join(map(str,templates)) ))
    
    return mat(shape=(4,4),type=mattype)

add_builtin("mat", input_types={"pos": vec(length=3,type=Scalar), "rot": quaternion(type=Scalar), "scale": vec(length=3,type=Scalar)}, value_func=mat_quat_constructor_value_func, 
    doc="""Construct a 4x4 transformation matrix that applies the transformations as Translation(pos)*Rotation(rot)*Scale(scale) when applied to column vectors, i.e.: y = (TRS)*x""", group="Vector Math", export=False)

add_builtin("svd3", input_types={"A": mat(shape=(3,3),type=Scalar), "U":mat(shape=(3,3),type=Scalar), "sigma":vec(length=3,type=Scalar), "V":mat(shape=(3,3),type=Scalar)}, value_type=None, group="Vector Math",
    doc="""Compute the SVD of a 3x3 matrix. The singular values are returned in sigma, 
   while the left and right basis vectors are returned in U and V.""")

add_builtin("qr3", input_types={"A": mat(shape=(3,3),type=Scalar), "Q":mat(shape=(3,3),type=Scalar), "R":mat(shape=(3,3),type=Scalar)}, value_type=None, group="Vector Math",
    doc="""Compute the QR decomposition of a 3x3 matrix. The orthogonal matrix is returned in Q, while the upper triangular matrix is returend in R.""")

add_builtin("eig3", input_types={"A": mat(shape=(3,3),type=Scalar), "Q":mat(shape=(3,3),type=Scalar), "d":vec(length=3,type=Scalar)}, value_type=None, group="Vector Math",
    doc="""Compute the eigen decomposition of a 3x3 marix. The eigen vectors are returned as the columns of Q, while the corresponding eigen values are returned in d.""")

#---------------------------------
# Quaternion Math

def quat_constructor_value_func(args,templates):
    if templates is None:
        return quaternion(type=Scalar)
    
    quattype = templates[0]
    if len(args) and infer_scalar_type(args) != quattype:
        raise RuntimeError("Wrong scalar type for quaternion<{}> constructor".format( ",".join(map(str,templates)) ))
    
    return quaternion(type=quattype)

add_builtin("quaternion", input_types={}, value_func=quat_constructor_value_func, group="Quaternion Math", 
    doc="""Construct a zero-initialized quaternion, quaternions are laid out as
   [ix, iy, iz, r], where ix, iy, iz are the imaginary part, and r the real part.""", export=False)
add_builtin("quaternion", input_types={"x": Scalar, "y": Scalar, "z": Scalar, "w": Scalar}, value_func=quat_constructor_value_func, group="Quaternion Math",
    doc="Construct a quarternion from its components x, y, z are the imaginary parts, w is the real part.", export=False)
add_builtin("quaternion", input_types={"i": vec(length=3,type=Scalar), "r": Scalar}, value_func=quat_constructor_value_func, group="Quaternion Math",
    doc="Construct a quaternion from it's imaginary components i, and real part r", export=False)

add_builtin("quat_identity", input_types={}, value_type=quat, group="Quaternion Math",
    doc="Construct a float32 identity quaternion with zero imaginary part and real part of 1.0")

add_builtin("quat_from_axis_angle", input_types={"axis": vec(length=3,type=Scalar), "angle": Scalar}, value_func=lambda args,_: quaternion(type=infer_scalar_type(args)), group="Quaternion Math",
    doc="Construct a quaternion representing a rotation of angle radians around the given axis.")
add_builtin("quat_to_axis_angle", input_types={"q": quaternion(type=Scalar), "axis": vec(length=3,type=Scalar), "angle": Scalar}, value_type=None, group="Quaternion Math",
    doc="Extract the rotation axis and angle radians a quaternion represents.")
add_builtin("quat_from_matrix", input_types={"m": mat(shape=(3,3),type=Scalar)}, value_func=lambda args,_: quaternion(type=infer_scalar_type(args)), group="Quaternion Math",
    doc="Construct a quaternion from a 3x3 matrix.")
add_builtin("quat_rpy", input_types={"roll": Scalar, "pitch": Scalar, "yaw": Scalar}, value_func=lambda args,_: quaternion(type=infer_scalar_type(args)), group="Quaternion Math",
    doc="Construct a quaternion representing a combined roll (z), pitch (x), yaw rotations (y) in radians.")
add_builtin("quat_inverse", input_types={"q": quaternion(type=Scalar)}, value_func=lambda args,_: quaternion(type=infer_scalar_type(args)), group="Quaternion Math",
    doc="Compute quaternion conjugate.")
add_builtin("quat_rotate", input_types={"q": quaternion(type=Scalar), "p": vec(length=3,type=Scalar)}, value_func=lambda args,_: vec(length=3,type=infer_scalar_type(args)), group="Quaternion Math",
    doc="Rotate a vector by a quaternion.")
add_builtin("quat_rotate_inv", input_types={"q": quaternion(type=Scalar), "p": vec(length=3,type=Scalar)}, value_func=lambda args,_: vec(length=3,type=infer_scalar_type(args)), group="Quaternion Math",
    doc="Rotate a vector the inverse of a quaternion.")
add_builtin("quat_slerp", input_types={"q0": quaternion(type=Scalar), "q1": quaternion(type=Scalar), "t": Scalar}, value_func=lambda args: quaternion(type=infer_scalar_type(args)), group="Quaternion Math",
    doc="Linearly interpolate between two quaternions.")
add_builtin("quat_to_matrix", input_types={"q": quaternion(type=Scalar)}, value_func=lambda args,_: mat(shape=(3,3),type=infer_scalar_type(args)), group="Quaternion Math",
    doc="Convert a quaternion to a 3x3 rotation matrix.")

#---------------------------------
# Transformations 

def transform_constructor_value_func(args,templates):
    if templates is None:
        return transform_t(type=Scalar)
    
    transtype = templates[0]
    if infer_scalar_type(args) != transtype:
        raise RuntimeError("Wrong scalar type for transform_t<{}> constructor".format( ",".join(map(str,templates)) ))
    
    return transform_t(type=transtype)

add_builtin("transform_t", input_types={"p": vec(length=3,type=Scalar), "q": quaternion(type=Scalar)}, value_func=transform_constructor_value_func, group="Transformations",
    doc="Construct a rigid body transformation with translation part p and rotation q.", export=False)

add_builtin("transform_identity", input_types={}, value_type=transform, group="Transformations",
    doc="Construct a float32 identity transform with zero translation and identity rotation.")

add_builtin("transform_get_translation", input_types={"t": transform_t(type=Scalar)}, value_func=lambda args,_: vec(length=3,type=infer_scalar_type(args)), group="Transformations",
    doc="Return the translational part of a transform.")
add_builtin("transform_get_rotation", input_types={"t": transform_t(type=Scalar)}, value_func=lambda args,_: quaternion(type=infer_scalar_type(args)), group="Transformations",
    doc="Return the rotational part of a transform.")
add_builtin("transform_multiply", input_types={"a": transform_t(type=Scalar), "b": transform_t(type=Scalar)}, value_func=lambda args,_: transform_t(type=infer_scalar_type(args)), group="Transformations",
    doc="Multiply two rigid body transformations together.")
add_builtin("transform_point", input_types={"t": transform_t(type=Scalar), "p": vec(length=3,type=Scalar)}, value_func=lambda args,_: vec(length=3,type=infer_scalar_type(args)), group="Transformations",
    doc="Apply the transform to a point p treating the homogenous coordinate as w=1 (translation and rotation).")
add_builtin("transform_point", input_types={"m": mat(shape=(4,4),type=Scalar), "p": vec(length=3,type=Scalar)}, value_func=lambda args,_: vec(length=3,type=infer_scalar_type(args)), group="Vector Math",
    doc="""Apply the transform to a point ``p`` treating the homogenous coordinate as w=1. The transformation is applied treating ``p`` as a column vector, e.g.: ``y = M*p``
   note this is in contrast to some libraries, notably USD, which applies transforms to row vectors, ``y^T = p^T*M^T``. If the transform is coming from a library that uses row-vectors
   then users should transpose the tranformation matrix before calling this method.""")
add_builtin("transform_vector", input_types={"t": transform_t(type=Scalar), "v": vec(length=3,type=Scalar)}, value_func=lambda args,_: vec(length=3,type=infer_scalar_type(args)), group="Transformations",
    doc="Apply the transform to a vector v treating the homogenous coordinate as w=0 (rotation only).")
add_builtin("transform_vector", input_types={"m": mat(shape=(4,4),type=Scalar), "v": vec(length=3,type=Scalar)}, value_func=lambda args,_: vec(length=3,type=infer_scalar_type(args)), group="Vector Math",
    doc="""Apply the transform to a vector ``v`` treating the homogenous coordinate as w=0. The transformation is applied treating ``v`` as a column vector, e.g.: ``y = M*v``
   note this is in contrast to some libraries, notably USD, which applies transforms to row vectors, ``y^T = v^T*M^T``. If the transform is coming from a library that uses row-vectors
   then users should transpose the tranformation matrix before calling this method.""")
add_builtin("transform_inverse", input_types={"t": transform_t(type=Scalar)}, value_func=sametype_value_func(transform_t(type=Scalar)), group="Transformations",
    doc="Compute the inverse of the transform.")
#---------------------------------
# Spatial Math 

def spatial_vector_constructor_value_func(args,templates):
    if templates is None:
        return spatial_vector_t(type=Scalar)
    
    vectype = templates[0]
    if len(args) and infer_scalar_type(args) != vectype:
        raise RuntimeError("Wrong scalar type for spatial_vector_t<{}> constructor".format( ",".join(map(str,templates)) ))
    
    return spatial_vector_t(type=vectype)

add_builtin("spatial_vector_t", input_types={}, value_func=spatial_vector_constructor_value_func, group="Spatial Math",
    doc="Construct a zero-initialized 6d screw vector. Screw vectors may be used to represent rigid body wrenches and twists (velocites).", export=False)
add_builtin("spatial_vector_t", input_types={"a": Scalar, "b": Scalar, "c": Scalar, "d": Scalar, "e": Scalar, "f": Scalar}, value_func=spatial_vector_constructor_value_func, group="Spatial Math",
    doc="Construct a 6d screw vector from it's components.", export=False)
add_builtin("spatial_vector_t", input_types={"w": vec(length=3,type=Scalar), "v": vec(length=3,type=Scalar)}, value_func=spatial_vector_constructor_value_func, group="Spatial Math",
    doc="Construct a 6d screw vector from two 3d vectors.", export=False)
add_builtin("spatial_vector_t", input_types={"s": Scalar}, value_func=spatial_vector_constructor_value_func, group="Spatial Math",
    doc="Construct a 6d screw vector with all components set to s", export=False)

def spatial_matrix_constructor_value_func(args,templates):
    if templates is None:
        return spatial_matrix_t(type=Scalar)
    
    mattype = templates[0]
    return spatial_matrix_t(type=mattype)

add_builtin("spatial_matrix_t", input_types={}, value_func=spatial_matrix_constructor_value_func, group="Spatial Math",
    doc="Construct a 6x6 zero-initialized spatial inertia matrix", export=False)

add_builtin("spatial_adjoint", input_types={"r": mat(shape=(3,3),type=Scalar), "s": mat(shape=(3,3),type=Scalar)}, value_func=lambda args,_: spatial_matrix_t(type=infer_scalar_type(args)), group="Spatial Math",
    doc="Construct a 6x6 spatial inertial matrix from two 3x3 diagonal blocks.", export=False)
add_builtin("spatial_dot", input_types={"a": spatial_vector_t(type=Scalar), "b": spatial_vector_t(type=Scalar)}, value_func=sametype_scalar_value_func, group="Spatial Math",
    doc="Compute the dot product of two 6d screw vectors.")
add_builtin("spatial_cross", input_types={"a": spatial_vector_t(type=Scalar), "b": spatial_vector_t(type=Scalar)}, value_func=sametype_value_func(spatial_vector_t(type=Scalar)), group="Spatial Math",
    doc="Compute the cross-product of two 6d screw vectors.")
add_builtin("spatial_cross_dual", input_types={"a": spatial_vector_t(type=Scalar), "b": spatial_vector_t(type=Scalar)}, value_func=sametype_value_func(spatial_vector_t(type=Scalar)), group="Spatial Math",
    doc="Compute the dual cross-product of two 6d screw vectors.")

add_builtin("spatial_top", input_types={"a": spatial_vector_t(type=Scalar)}, value_func=lambda args,_: vec(length=3,type=args[0].type._wp_scalar_type_), group="Spatial Math",
    doc="Return the top (first) part of a 6d screw vector.")
add_builtin("spatial_bottom", input_types={"a": spatial_vector_t(type=Scalar)}, value_func=lambda args,_: vec(length=3,type=args[0].type._wp_scalar_type_), group="Spatial Math",
    doc="Return the bottom (second) part of a 6d screw vector.")

add_builtin("spatial_jacobian",
     input_types={"S": array(dtype=spatial_vector_t(type=Scalar)), 
                  "joint_parents": array(dtype=int),
                  "joint_qd_start": array(dtype=int),
                  "joint_start": int,
                  "joint_count": int,
                  "J_start": int,
                  "J_out": array(dtype=Scalar)}, value_type=None, doc="", group="Spatial Math")

add_builtin("spatial_mass",
     input_types={"I_s": array(dtype=spatial_matrix_t(type=Scalar)),
                  "joint_start": int,
                  "joint_count": int,
                  "M_start": int,
                  "M": array(dtype=Scalar)}, value_type=None, doc="", group="Spatial Math")

#---------------------------------
# Linear Algebra

add_builtin("dense_gemm", 
    input_types={"m": int, 
                 "n": int, 
                 "p": int, 
                 "t1": int, 
                 "t2": int, 
                 "A": array(dtype=float), 
                 "B": array(dtype=float), 
                 "C": array(dtype=float) }, value_type=None, doc="", group="Utility", hidden=True)

add_builtin("dense_gemm_batched", 
    input_types={"m": array(dtype=int), 
                 "n": array(dtype=int), 
                 "p": array(dtype=int), 
                 "t1": int, 
                 "t2": int, 
                 "A_start": array(dtype=int), 
                 "B_start": array(dtype=int), 
                 "C_start": array(dtype=int), 
                 "A": array(dtype=float), 
                 "B": array(dtype=float), 
                 "C": array(dtype=float)}, value_type=None, doc="", group="Utility", hidden=True)


add_builtin("dense_chol",
    input_types={"n": int, 
                 "A": array(dtype=float), 
                 "regularization": float, 
                 "L": array(dtype=float)}, value_type=None, doc="WIP", group="Utility", hidden=True)

add_builtin("dense_chol_batched",
    input_types={"A_start": array(dtype=int),
                 "A_dim": array(dtype=int),
                 "A": array(dtype=float),
                 "regularization": float,
                 "L": array(dtype=float)}, value_type=None, doc="WIP", group="Utility", hidden=True)

add_builtin("dense_subs", 
    input_types={"n": int, 
                 "L": array(dtype=float), 
                 "b": array(dtype=float), 
                 "x": array(dtype=float)}, value_type=None, doc="WIP", group="Utility", hidden=True)

add_builtin("dense_solve", 
    input_types={"n": int, 
                 "A": array(dtype=float), 
                 "L": array(dtype=float), 
                 "b": array(dtype=float), 
                 "x": array(dtype=float)}, value_type=None, doc="WIP", group="Utility", hidden=True)

add_builtin("dense_solve_batched",
    input_types={"b_start": array(dtype=int), 
                 "A_start": array(dtype=int),
                 "A_dim": array(dtype=int),
                 "A": array(dtype=float),
                 "L": array(dtype=float),
                 "b": array(dtype=float),
                 "x": array(dtype=float)}, value_type=None, doc="WIP", group="Utility", hidden=True)


add_builtin("mlp", input_types={"weights": array(dtype=float, ndim=2), "bias": array(dtype=float, ndim=1), "activation": Callable, "index": int, "x": array(dtype=float, ndim=2), "out": array(dtype=float, ndim=2)}, value_type=None, skip_replay=True, 
    doc="""Evaluate a multi-layer perceptron (MLP) layer in the form: ``out = act(weights*x + bias)``. 

   :param weights: A layer's network weights with dimensions ``(m, n)``.
   :param bias: An array with dimensions ``(n)``.
   :param activation: A ``wp.func`` function that takes a single scalar float as input and returns a scalar float as output
   :param index: The batch item to process, typically each thread will process 1 item in the batch, in this case index should be ``wp.tid()``
   :param x: The feature matrix with dimensions ``(n, b)``
   :param out: The network output with dimensions ``(m, b)``

   :note: Feature and output matrices are transposed compared to some other frameworks such as PyTorch. All matrices are assumed to be stored in flattened row-major memory layout (NumPy default).""", group="Utility")


#---------------------------------
# Geometry

add_builtin("bvh_query_aabb", input_types={"id": uint64, "lower": vec3, "upper": vec3}, value_type=bvh_query_t, group="Geometry",
    doc="""Construct an axis-aligned bounding box query against a bvh object. This query can be used to iterate over all bounds
   inside a bvh. Returns an object that is used to track state during bvh traversal.
    
   :param id: The bvh identifier
   :param lower: The lower bound of the bounding box in bvh space
   :param upper: The upper bound of the bounding box in bvh space""")

add_builtin("bvh_query_ray", input_types={"id": uint64, "start": vec3, "dir": vec3}, value_type=bvh_query_t, group="Geometry",
    doc="""Construct a ray query against a bvh object. This query can be used to iterate over all bounds
   that intersect the ray. Returns an object that is used to track state during bvh traversal.
    
   :param id: The bvh identifier
   :param start: The start of the ray in bvh space
   :param dir: The direction of the ray in bvh space""")

add_builtin("bvh_query_next", input_types={"query": bvh_query_t, "index": int}, value_type=bool, group="Geometry",
    doc="""Move to the next bound returned by the query. The index of the current bound is stored in ``index``, returns ``False``
   if there are no more overlapping bound.""")

add_builtin("mesh_query_point", input_types={"id": uint64, "point": vec3, "max_dist": float, "inside": float, "face": int, "bary_u": float, "bary_v": float}, value_type=bool, group="Geometry",
    doc="""Computes the closest point on the mesh with identifier `id` to the given point in space. Returns ``True`` if a point < ``max_dist`` is found.

   :param id: The mesh identifier
   :param point: The point in space to query
   :param max_dist: Mesh faces above this distance will not be considered by the query
   :param inside: Returns a value < 0 if query point is inside the mesh, >=0 otherwise. Note that mesh must be watertight for this to be robust
   :param face: Returns the index of the closest face
   :param bary_u: Returns the barycentric u coordinate of the closest point
   :param bary_v: Retruns the barycentric v coordinate of the closest point""")

add_builtin("mesh_query_ray", input_types={"id": uint64, "start": vec3, "dir": vec3, "max_t": float, "t": float, "bary_u": float, "bary_v": float, "sign": float, "normal": vec3, "face": int}, value_type=bool, group="Geometry",
    doc="""Computes the closest ray hit on the mesh with identifier `id`, returns ``True`` if a point < ``max_t`` is found.

   :param id: The mesh identifier
   :param start: The start point of the ray
   :param dir: The ray direction (should be normalized)
   :param max_t: The maximum distance along the ray to check for intersections
   :param t: Returns the distance of the closest hit along the ray
   :param bary_u: Returns the barycentric u coordinate of the closest hit
   :param bary_v: Returns the barycentric v coordinate of the closest hit
   :param sign: Returns a value > 0 if the hit ray hit front of the face, returns < 0 otherwise
   :param normal: Returns the face normal
   :param face: Returns the index of the hit face""")

add_builtin("mesh_query_aabb", input_types={"id": uint64, "lower": vec3, "upper": vec3}, value_type=mesh_query_aabb_t, group="Geometry",
    doc="""Construct an axis-aligned bounding box query against a mesh object. This query can be used to iterate over all triangles
   inside a volume. Returns an object that is used to track state during mesh traversal.
    
   :param id: The mesh identifier
   :param lower: The lower bound of the bounding box in mesh space
   :param upper: The upper bound of the bounding box in mesh space""")

add_builtin("mesh_query_aabb_next", input_types={"query": mesh_query_aabb_t, "index": int}, value_type=bool, group="Geometry",
    doc="""Move to the next triangle overlapping the query bounding box. The index of the current face is stored in ``index``, returns ``False``
   if there are no more overlapping triangles.""")

add_builtin("mesh_eval_position", input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float}, value_type=vec3, group="Geometry",
    doc="""Evaluates the position on the mesh given a face index, and barycentric coordinates.""")

add_builtin("mesh_eval_velocity", input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float}, value_type=vec3, group="Geometry",
    doc="""Evaluates the velocity on the mesh given a face index, and barycentric coordinates.""")

add_builtin("hash_grid_query", input_types={"id": uint64, "point": vec3, "max_dist": float}, value_type=hash_grid_query_t, group="Geometry",
    doc="""Construct a point query against a hash grid. This query can be used to iterate over all neighboring points withing a 
   fixed radius from the query point. Returns an object that is used to track state during neighbor traversal.""")

add_builtin("hash_grid_query_next", input_types={"query": hash_grid_query_t, "index": int}, value_type=bool, group="Geometry",
    doc="""Move to the next point in the hash grid query. The index of the current neighbor is stored in ``index``, returns ``False``
   if there are no more neighbors.""")

add_builtin("hash_grid_point_id", input_types={"id": uint64, "index": int}, value_type=int, group="Geometry",
    doc="""Return the index of a point in the grid, this can be used to re-order threads such that grid 
   traversal occurs in a spatially coherent order.""")

add_builtin("intersect_tri_tri", input_types={"v0": vec3, "v1": vec3, "v2": vec3, "u0": vec3, "u1": vec3, "u2": vec3}, value_type=int, group="Geometry", 
    doc="Tests for intersection between two triangles (v0, v1, v2) and (u0, u1, u2) using Moller's method. Returns > 0 if triangles intersect.")


add_builtin("mesh_get", input_types={"id": uint64}, value_type=Mesh, group="Geometry",
    doc="""Retrieves the mesh given its index.""")

add_builtin("mesh_eval_face_normal", input_types={"id": uint64, "face": int}, value_type=vec3, group="Geometry",
    doc="""Evaluates the face normal the mesh given a face index.""")

add_builtin("mesh_get_point", input_types={"id": uint64, "index": int}, value_type=vec3, group="Geometry",
    doc="""Returns the point of the mesh given a index.""")

add_builtin("mesh_get_velocity", input_types={"id": uint64, "index": int}, value_type=vec3, group="Geometry",
    doc="""Returns the velocity of the mesh given a index.""")

add_builtin("mesh_get_index", input_types={"id": uint64, "index": int}, value_type=int, group="Geometry",
    doc="""Returns the point-index of the mesh given a face-vertex index.""")


add_builtin("closest_point_edge_edge", input_types={"p1":vec3, "q1":vec3, "p2":vec3, "q2":vec3, "epsilon":float}, value_type=vec3, group="Geometry",
    doc="""Finds the closest points between two edges. Returns barycentric weights to the points on each edge, as well as the closest distance between the edges.

   :param p1: First point of first edge
   :param q1: Second point of first edge
   :param p2: First point of second edge
   :param q2: Second point of second edge
   :param epsilon: Zero tolerance for determining if points in an edge are degenerate.
   :param out: vec3 output containing (s,t,d), where `s` in [0,1] is the barycentric weight for the first edge, `t` is the barycentric weight for the second edge, and `d` is the distance between the two edges at these two closest points.""")

#---------------------------------
# Ranges

add_builtin("range", input_types={"end": int}, value_type=range_t, group="Utility", export=False, hidden=True)
add_builtin("range", input_types={"start": int, "end": int}, value_type=range_t, group="Utility", export=False, hidden=True)
add_builtin("range", input_types={"start": int, "end": int, "step": int}, value_type=range_t, group="Utility", export=False, hidden=True)

#---------------------------------
# Iterators

add_builtin("iter_next", input_types={"range": range_t}, value_type=int, group="Utility", hidden=True)
add_builtin("iter_next", input_types={"query": hash_grid_query_t}, value_type=int, group="Utility", hidden=True)
add_builtin("iter_next", input_types={"query": mesh_query_aabb_t}, value_type=int, group="Utility", hidden=True)

#---------------------------------
# Volumes 

add_builtin("volume_sample_f", input_types={"id": uint64, "uvw": vec3, "sampling_mode": int}, value_type=float, group="Volumes",
    doc="""Sample the volume given by ``id`` at the volume local-space point ``uvw``. Interpolation should be ``wp.Volume.CLOSEST``, or ``wp.Volume.LINEAR.``""")

add_builtin("volume_lookup_f", input_types={"id": uint64, "i": int, "j": int, "k": int}, value_type=float, group="Volumes",
    doc="""Returns the value of voxel with coordinates ``i``, ``j``, ``k``, if the voxel at this index does not exist this function returns the background value""")

add_builtin("volume_store_f", input_types={"id": uint64, "i": int, "j": int, "k": int, "value": float}, group="Volumes",
    doc="""Store the value at voxel with coordinates ``i``, ``j``, ``k``.""")

add_builtin("volume_sample_v", input_types={"id": uint64, "uvw": vec3, "sampling_mode": int}, value_type=vec3, group="Volumes",
    doc="""Sample the vector volume given by ``id`` at the volume local-space point ``uvw``. Interpolation should be ``wp.Volume.CLOSEST``, or ``wp.Volume.LINEAR.``""")

add_builtin("volume_lookup_v", input_types={"id": uint64, "i": int, "j": int, "k": int}, value_type=vec3, group="Volumes",
    doc="""Returns the vector value of voxel with coordinates ``i``, ``j``, ``k``, if the voxel at this index does not exist this function returns the background value""")

add_builtin("volume_store_v", input_types={"id": uint64, "i": int, "j": int, "k": int, "value": vec3}, group="Volumes",
    doc="""Store the value at voxel with coordinates ``i``, ``j``, ``k``.""")

add_builtin("volume_sample_i", input_types={"id": uint64, "uvw": vec3}, value_type=int, group="Volumes",
    doc="""Sample the int32 volume given by ``id`` at the volume local-space point ``uvw``. """)

add_builtin("volume_lookup_i", input_types={"id": uint64, "i": int, "j": int, "k": int}, value_type=int, group="Volumes",
    doc="""Returns the int32 value of voxel with coordinates ``i``, ``j``, ``k``, if the voxel at this index does not exist this function returns the background value""")

add_builtin("volume_store_i", input_types={"id": uint64, "i": int, "j": int, "k": int, "value": int}, group="Volumes",
    doc="""Store the value at voxel with coordinates ``i``, ``j``, ``k``.""")

add_builtin("volume_index_to_world", input_types={"id": uint64, "uvw": vec3}, value_type=vec3, group="Volumes",
    doc="""Transform a point defined in volume index space to world space given the volume's intrinsic affine transformation.""")
add_builtin("volume_world_to_index", input_types={"id": uint64, "xyz": vec3}, value_type=vec3, group="Volumes",
    doc="""Transform a point defined in volume world space to the volume's index space, given the volume's intrinsic affine transformation.""")
add_builtin("volume_index_to_world_dir", input_types={"id": uint64, "uvw": vec3}, value_type=vec3, group="Volumes",
    doc="""Transform a direction defined in volume index space to world space given the volume's intrinsic affine transformation.""")
add_builtin("volume_world_to_index_dir", input_types={"id": uint64, "xyz": vec3}, value_type=vec3, group="Volumes",
    doc="""Transform a direction defined in volume world space to the volume's index space, given the volume's intrinsic affine transformation.""")


#---------------------------------
# Random 

add_builtin("rand_init", input_types={"seed": int}, value_type=uint32, group="Random", 
    doc="Initialize a new random number generator given a user-defined seed. Returns a 32-bit integer representing the RNG state.")

add_builtin("rand_init", input_types={"seed": int, "offset": int}, value_type=uint32, group="Random", 
    doc="""Initialize a new random number generator given a user-defined seed and an offset. 
   This alternative constructor can be useful in parallel programs, where a kernel as a whole should share a seed,
   but each thread should generate uncorrelated values. In this case usage should be ``r = rand_init(seed, tid)``""")

add_builtin("randi", input_types={"state": uint32}, value_type=int, group="Random", 
    doc="Return a random integer between [0, 2^32)")
add_builtin("randi", input_types={"state": uint32, "min": int, "max": int}, value_type=int, group="Random", 
    doc="Return a random integer between [min, max)")
add_builtin("randf", input_types={"state": uint32}, value_type=float, group="Random", 
    doc="Return a random float between [0.0, 1.0)")
add_builtin("randf", input_types={"state": uint32, "min": float, "max": float}, value_type=float, group="Random", 
    doc="Return a random float between [min, max)")
add_builtin("randn", input_types={"state": uint32}, value_type=float, group="Random", 
    doc="Sample a normal distribution")

add_builtin("sample_cdf", input_types={"state": uint32, "cdf": array(dtype=float)}, value_type=int, group="Random",
    doc="Inverse transform sample a cumulative distribution function")
add_builtin("sample_triangle", input_types={"state": uint32}, value_type=vec2, group="Random",
    doc="Uniformly sample a triangle. Returns sample barycentric coordinates")
add_builtin("sample_unit_ring", input_types={"state": uint32}, value_type=vec2, group="Random",
    doc="Uniformly sample a ring in the xy plane")
add_builtin("sample_unit_disk", input_types={"state": uint32}, value_type=vec2, group="Random",
    doc="Uniformly sample a disk in the xy plane")
add_builtin("sample_unit_sphere_surface", input_types={"state": uint32}, value_type=vec3, group="Random",
    doc="Uniformly sample a unit sphere surface")
add_builtin("sample_unit_sphere", input_types={"state": uint32}, value_type=vec3, group="Random",
    doc="Uniformly sample a unit sphere")
add_builtin("sample_unit_hemisphere_surface", input_types={"state": uint32}, value_type=vec3, group="Random",
    doc="Uniformly sample a unit hemisphere surface")
add_builtin("sample_unit_hemisphere", input_types={"state": uint32}, value_type=vec3, group="Random",
    doc="Uniformly sample a unit hemisphere")
add_builtin("sample_unit_square", input_types={"state": uint32}, value_type=vec2, group="Random",
    doc="Uniformly sample a unit square")
add_builtin("sample_unit_cube", input_types={"state": uint32}, value_type=vec3, group="Random",
    doc="Uniformly sample a unit cube")

add_builtin("noise", input_types={"state": uint32, "x": float}, value_type=float, group="Random",
    doc="Non-periodic Perlin-style noise in 1d.")
add_builtin("noise", input_types={"state": uint32, "xy": vec2}, value_type=float, group="Random",
    doc="Non-periodic Perlin-style noise in 2d.")
add_builtin("noise", input_types={"state": uint32, "xyz": vec3}, value_type=float, group="Random",
    doc="Non-periodic Perlin-style noise in 3d.")
add_builtin("noise", input_types={"state": uint32, "xyzt": vec4}, value_type=float, group="Random",
    doc="Non-periodic Perlin-style noise in 4d.")

add_builtin("pnoise", input_types={"state": uint32, "x": float, "px": int}, value_type=float, group="Random",
    doc="Periodic Perlin-style noise in 1d.")
add_builtin("pnoise", input_types={"state": uint32, "xy": vec2, "px": int, "py": int}, value_type=float, group="Random",
    doc="Periodic Perlin-style noise in 2d.")
add_builtin("pnoise", input_types={"state": uint32, "xyz": vec3, "px": int, "py": int, "pz": int}, value_type=float, group="Random",
    doc="Periodic Perlin-style noise in 3d.")
add_builtin("pnoise", input_types={"state": uint32, "xyzt": vec4, "px": int, "py": int, "pz": int, "pt": int}, value_type=float, group="Random",
    doc="Periodic Perlin-style noise in 4d.")

add_builtin("curlnoise", input_types={"state": uint32, "xy": vec2}, value_type=vec2, group="Random",
    doc="Divergence-free vector field based on the gradient of a Perlin noise function.", missing_grad=True)
add_builtin("curlnoise", input_types={"state": uint32, "xyz": vec3}, value_type=vec3, group="Random",
    doc="Divergence-free vector field based on the curl of three Perlin noise functions.", missing_grad=True)
add_builtin("curlnoise", input_types={"state": uint32, "xyzt": vec4}, value_type=vec3, group="Random",
    doc="Divergence-free vector field based on the curl of three Perlin noise functions.", missing_grad=True)

# note printf calls directly to global CRT printf (no wp:: namespace prefix)
add_builtin("printf", input_types={}, namespace="", variadic=True, group="Utility",
    doc="Allows printing formatted strings, using C-style format specifiers.")

add_builtin("print", input_types={"value": Any}, doc="Print variable to stdout", export=False, group="Utility")

# helpers
add_builtin("tid", input_types={}, value_type=int, group="Utility",
    doc="""Return the current thread index. Note that this is the *global* index of the thread in the range [0, dim) 
   where dim is the parameter passed to kernel launch.""")

add_builtin("tid", input_types={}, value_type=[int, int], group="Utility",
    doc="""Return the current thread indices for a 2d kernel launch. Use ``i,j = wp.tid()`` syntax to retrieve the coordinates inside the kernel thread grid.""")

add_builtin("tid", input_types={}, value_type=[int, int, int], group="Utility",
    doc="""Return the current thread indices for a 3d kernel launch. Use ``i,j,k = wp.tid()`` syntax to retrieve the coordinates inside the kernel thread grid.""")

add_builtin("tid", input_types={}, value_type=[int, int, int, int], group="Utility",
    doc="""Return the current thread indices for a 4d kernel launch. Use ``i,j,k,l = wp.tid()`` syntax to retrieve the coordinates inside the kernel thread grid.""")


add_builtin("copy", variadic=True, hidden=True, export=False, group="Utility")
add_builtin("select", input_types={"cond": bool, "arg1": Any, "arg2": Any}, value_func=lambda args,_: args[1].type, doc="Select between two arguments, if cond is false then return ``arg1``, otherwise return ``arg2``", group="Utility")

# does argument checking and type progagation for load()
def load_value_func(args,_):

    if (type(args[0].type) != array):
        raise RuntimeError("load() argument 0 must be an array")

    num_indices = len(args[1:])
    num_dims = args[0].type.ndim

    if num_indices < num_dims:
        raise RuntimeError("Num indices < num dimensions for array load, this is a codegen error, should have generated a view instead")

    if num_indices > num_dims:
        raise RuntimeError(f"Num indices > num dimensions for array load, received {num_indices}, but array only has {num_dims}")

    # check index types
    for a in args[1:]:
        if type_is_int(a.type) == False:
            raise RuntimeError(f"load() index arguments must be of integer type, got index of type {a.type}")

    return args[0].type.dtype

# does argument checking and type progagation for view()
def view_value_func(args,_):
    if (type(args[0].type) != array):
        raise RuntimeError("view() argument 0 must be an array")

    # check array dim big enough to support view
    num_indices = len(args[1:])
    num_dims = args[0].type.ndim

    if num_indices >= num_dims:
        raise RuntimeError(f"Trying to create an array view with {num_indices} indices, but the array only has {num_dims} dimension(s). Ensure that the argument type on the function or kernel specifies the expected number of dimensions, e.g.: def func(param: wp.array3d(dtype=float):")

    # check index types
    for a in args[1:]:
        if type_is_int(a.type) == False:
            raise RuntimeError(f"view() index arguments must be of integer type, got index of type {a.type}")

    # create an array view with leading dimensions removed
    import copy
    view_type = copy.copy(args[0].type)
    view_type.ndim -= num_indices
    
    return view_type

# does argument checking and type progagation for store()
def store_value_func(args,_):
    # check target type
    if (type(args[0].type) != array):
        raise RuntimeError("store() argument 0 must be an array")
    
    num_indices = len(args[1:-1])
    num_dims = args[0].type.ndim

    # if this happens we should have generated a view instead of a load during code gen
    if num_indices < num_dims:
        raise RuntimeError("Num indices < num dimensions for array store")

    if num_indices > num_dims:
        raise RuntimeError(f"Num indices > num dimensions for array store, received {num_indices}, but array only has {num_dims}")

    # check index types
    for a in args[1:-1]:
        if type_is_int(a.type) == False:
            raise RuntimeError(f"store() index arguments must be of integer type, got index of type {a.type}")

    # check value type
    if not types_equal(args[-1].type, args[0].type.dtype):
        raise RuntimeError(f"store() value argument type ({args[2].type}) must be of the same type as the array ({args[0].type.dtype})")

    return None


add_builtin("load", variadic=True, hidden=True, value_func=load_value_func, group="Utility")
add_builtin("view", variadic=True, hidden=True, value_func=view_value_func, group="Utility")
add_builtin("store", variadic=True, hidden=True, value_func=store_value_func, skip_replay=True, group="Utility")

def atomic_op_value_type(args,_):

    # check target type
    if (type(args[0].type) != array):
        raise RuntimeError("atomic() operation argument 0 must be an array")
    
    num_indices = len(args[1:-1])
    num_dims = args[0].type.ndim

    # if this happens we should have generated a view instead of a load during code gen
    if num_indices < num_dims:
        raise RuntimeError("Num indices < num dimensions for atomic array operation")

    if num_indices > num_dims:
        raise RuntimeError(f"Num indices > num dimensions for atomic array operation, received {num_indices}, but array only has {num_dims}")

    # check index types
    for a in args[1:-1]:
        if type_is_int(a.type) == False:
            raise RuntimeError(f"atomic() operation index arguments must be of integer type, got index of type {a.type}")

    if not types_equal(args[-1].type, args[0].type.dtype):
        raise RuntimeError(f"atomic() value argument ({args[-1].type}) must be of the same type as the array ({args[0].type.dtype})")

    return args[0].type.dtype


add_builtin("atomic_add", input_types={"a": array(dtype=Any), "i": int, "value": Any}, value_func=atomic_op_value_type, doc="Atomically add ``value`` onto the array at location given by index.", group="Utility", skip_replay=True)
add_builtin("atomic_add", input_types={"a": array(dtype=Any), "i": int, "j": int, "value": Any}, value_func=atomic_op_value_type, doc="Atomically add ``value`` onto the array at location given by indices.", group="Utility", skip_replay=True)
add_builtin("atomic_add", input_types={"a": array(dtype=Any), "i": int, "j": int, "k": int, "value": Any}, value_func=atomic_op_value_type, doc="Atomically add ``value`` onto the array at location given by indices.", group="Utility", skip_replay=True)
add_builtin("atomic_add", input_types={"a": array(dtype=Any), "i": int, "j": int, "k": int, "l": int, "value": Any}, value_func=atomic_op_value_type, doc="Atomically add ``value`` onto the array at location given by indices.", group="Utility", skip_replay=True)

add_builtin("atomic_sub", input_types={"a": array(dtype=Any), "i": int, "value": Any}, value_func=atomic_op_value_type, doc="Atomically subtract ``value`` onto the array at location given by index.", group="Utility", skip_replay=True)
add_builtin("atomic_sub", input_types={"a": array(dtype=Any), "i": int, "j": int, "value": Any}, value_func=atomic_op_value_type, doc="Atomically subtract ``value`` onto the array at location given by indices.", group="Utility", skip_replay=True)
add_builtin("atomic_sub", input_types={"a": array(dtype=Any), "i": int, "j": int, "k":int, "value": Any}, value_func=atomic_op_value_type, doc="Atomically subtract ``value`` onto the array at location given by indices.", group="Utility", skip_replay=True)
add_builtin("atomic_sub", input_types={"a": array(dtype=Any), "i": int, "j": int, "k":int, "l": int, "value": Any}, value_func=atomic_op_value_type, doc="Atomically subtract ``value`` onto the array at location given by indices.", group="Utility", skip_replay=True)

add_builtin("atomic_min", input_types={"a": array(dtype=Any), "i": int, "value": Any}, value_func=atomic_op_value_type, doc="Compute the minimum of ``value`` and ``array[index]`` and atomically update the array. Note that for vectors and matrices the operation is only atomic on a per-component basis.", group="Utility", skip_replay=True)
add_builtin("atomic_min", input_types={"a": array(dtype=Any), "i": int, "j": int, "value": Any}, value_func=atomic_op_value_type, doc="Compute the minimum of ``value`` and ``array[index]`` and atomically update the array. Note that for vectors and matrices the operation is only atomic on a per-component basis.", group="Utility", skip_replay=True)
add_builtin("atomic_min", input_types={"a": array(dtype=Any), "i": int, "j": int, "k": int, "value": Any}, value_func=atomic_op_value_type, doc="Compute the minimum of ``value`` and ``array[index]`` and atomically update the array. Note that for vectors and matrices the operation is only atomic on a per-component basis.", group="Utility", skip_replay=True)
add_builtin("atomic_min", input_types={"a": array(dtype=Any), "i": int, "j": int, "k": int, "l": int, "value": Any}, value_func=atomic_op_value_type, doc="Compute the minimum of ``value`` and ``array[index]`` and atomically update the array. Note that for vectors and matrices the operation is only atomic on a per-component basis.", group="Utility", skip_replay=True)

add_builtin("atomic_max", input_types={"a": array(dtype=Any), "i": int, "value": Any}, value_func=atomic_op_value_type, doc="Compute the maximum of ``value`` and ``array[index]`` and atomically update the array. Note that for vectors and matrices the operation is only atomic on a per-component basis.", group="Utility", skip_replay=True)
add_builtin("atomic_max", input_types={"a": array(dtype=Any), "i": int, "j": int, "value": Any}, value_func=atomic_op_value_type, doc="Compute the maximum of ``value`` and ``array[index]`` and atomically update the array. Note that for vectors and matrices the operation is only atomic on a per-component basis.", group="Utility", skip_replay=True)
add_builtin("atomic_max", input_types={"a": array(dtype=Any), "i": int, "j": int, "k":int, "value": Any}, value_func=atomic_op_value_type, doc="Compute the maximum of ``value`` and ``array[index]`` and atomically update the array. Note that for vectors and matrices the operation is only atomic on a per-component basis.", group="Utility", skip_replay=True)
add_builtin("atomic_max", input_types={"a": array(dtype=Any), "i": int, "j": int, "k":int, "l": int, "value": Any}, value_func=atomic_op_value_type, doc="Compute the maximum of ``value`` and ``array[index]`` and atomically update the array. Note that for vectors and matrices the operation is only atomic on a per-component basis.", group="Utility", skip_replay=True)


# used to index into builtin types, i.e.: y = vec3[1]
def index_value_func(args,_):
    return args[0].type._wp_scalar_type_

add_builtin("index", input_types={"a": vec(length=Any,type=Scalar), "i": int}, value_func=index_value_func, hidden=True, group="Utility")
add_builtin("index", input_types={"a": quaternion(type=Scalar), "i": int}, value_func=index_value_func, hidden=True, group="Utility")

add_builtin("index", input_types={"a": mat(shape=(Any,Any),type=Scalar), "i": int}, value_func=lambda args,_: vec(length=args[0].type._shape_[0],type=args[0].type._wp_scalar_type_), hidden=True, group="Utility")
add_builtin("index", input_types={"a": mat(shape=(Any,Any),type=Scalar), "i": int, "j": int}, value_func=index_value_func, hidden=True, group="Utility")

add_builtin("index", input_types={"a": spatial_matrix_t(type=Scalar), "i": int, "j": int}, value_func=index_value_func, hidden=True, group="Utility")

add_builtin("index", input_types={"a": spatial_vector_t(type=Scalar), "i": int}, value_func=index_value_func, hidden=True, group="Utility")
add_builtin("index", input_types={"a": transform_t(type=Scalar), "i": int}, value_func=index_value_func, hidden=True, group="Utility")

add_builtin("index", input_types={"s": shape_t, "i": int}, value_type=int, hidden=True, group="Utility")

for t in scalar_types + vector_types:
    if t in [vec2,vec3,vec4,mat22,mat33,mat44]:
        continue
    add_builtin("expect_eq", input_types={"arg1": t, "arg2": t}, value_type=None, doc="Prints an error to stdout if arg1 and arg2 are not equal", group="Utility")

def expect_eq_val_func(args,_):
    if not types_equal(args[0].type,args[1].type):
        raise RuntimeError("Can't test equality for objects with different types")
    return None

add_builtin("expect_eq", input_types={"arg1": vec(length=Any,type=Scalar), "arg2": vec(length=Any,type=Scalar)}, value_func=expect_eq_val_func, doc="Prints an error to stdout if arg1 and arg2 are not equal", group="Utility")
add_builtin("expect_neq", input_types={"arg1": vec(length=Any,type=Scalar), "arg2": vec(length=Any,type=Scalar)}, value_func=expect_eq_val_func, doc="Prints an error to stdout if arg1 and arg2 are equal", group="Utility")

add_builtin("expect_eq", input_types={"arg1": mat(shape=(Any,Any),type=Scalar), "arg2": mat(shape=(Any,Any),type=Scalar)}, value_func=expect_eq_val_func, doc="Prints an error to stdout if arg1 and arg2 are not equal", group="Utility")
add_builtin("expect_neq", input_types={"arg1": mat(shape=(Any,Any),type=Scalar), "arg2": mat(shape=(Any,Any),type=Scalar)}, value_func=expect_eq_val_func, doc="Prints an error to stdout if arg1 and arg2 are equal", group="Utility")
    
for t in [float16,float32,float64,]:
    
    add_builtin("lerp", input_types={"a": t, "b": t, "t": t}, value_type=t, doc="Linearly interpolate two values a and b using factor t, computed as ``a*(1-t) + b*t``", group="Utility")
    add_builtin("smoothstep", input_types={"edge0": t, "edge1": t, "x": t}, value_type=t, doc="Smoothly interpolate between two values edge0 and edge1 using a factor x, and return a result between 0 and 1 using a cubic Hermite interpolation after clamping", group="Utility")

for typ in vector_types:
    t = typ._wp_scalar_type_
    if t not in [float16,float32,float64,]:
        continue

    add_builtin("lerp", input_types={"a": typ, "b": typ, "t": t}, value_type=typ, doc="Linearly interpolate two values a and b using factor t, computed as ``a*(1-t) + b*t``", group="Utility")

# fuzzy compare for float values
add_builtin("expect_near", input_types={"arg1": float, "arg2": float, "tolerance": float}, value_type=None, doc="Prints an error to stdout if arg1 and arg2 are not closer than tolerance in magnitude", group="Utility")
add_builtin("expect_near", input_types={"arg1": vec3, "arg2": vec3, "tolerance": float}, value_type=None, doc="Prints an error to stdout if any element of arg1 and arg2 are not closer than tolerance in magnitude", group="Utility")

#---------------------------------
# Algorithms

add_builtin("lower_bound", input_types={"arr": array(dtype=int), "value": int}, value_type=int, doc="Search a sorted array for the closest element greater than or equal to value.")
add_builtin("lower_bound", input_types={"arr": array(dtype=float), "value": float}, value_type=int, doc="Search a sorted array for the closest element greater than or equal to value.")

#---------------------------------
# Operators

add_builtin("add", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar), doc="", group="Operators")
add_builtin("add", input_types={"x": vec(length=Any,type=Scalar), "y": vec(length=Any,type=Scalar)}, value_func=sametype_value_func(vec(length=Any,type=Scalar)), doc="", group="Operators")
add_builtin("add", input_types={"x": quaternion(type=Scalar), "y": quaternion(type=Scalar)}, value_func=sametype_value_func(quaternion(type=Scalar)), doc="", group="Operators")
add_builtin("add", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": mat(shape=(Any,Any),type=Scalar)}, value_func=sametype_value_func(mat(shape=(Any,Any),type=Scalar)), doc="", group="Operators")
add_builtin("add", input_types={"x": spatial_vector_t(type=Scalar), "y": spatial_vector_t(type=Scalar)}, value_func=sametype_value_func(spatial_vector_t(type=Scalar)), doc="", group="Operators")
add_builtin("add", input_types={"x": spatial_matrix_t(type=Scalar), "y": spatial_matrix_t(type=Scalar)}, value_func=sametype_value_func(spatial_matrix_t(type=Scalar)), doc="", group="Operators")

add_builtin("sub", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar), doc="", group="Operators")
add_builtin("sub", input_types={"x": vec(length=Any,type=Scalar), "y": vec(length=Any,type=Scalar)}, value_func=sametype_value_func(vec(length=Any,type=Scalar)), doc="", group="Operators")
add_builtin("sub", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": mat(shape=(Any,Any),type=Scalar)}, value_func=sametype_value_func(mat(shape=(Any,Any),type=Scalar)), doc="", group="Operators")
add_builtin("sub", input_types={"x": spatial_vector_t(type=Scalar), "y": spatial_vector_t(type=Scalar)}, value_func=sametype_value_func(spatial_vector_t(type=Scalar)), doc="", group="Operators")
add_builtin("sub", input_types={"x": spatial_matrix_t(type=Scalar), "y": spatial_matrix_t(type=Scalar)}, value_func=sametype_value_func(spatial_matrix_t(type=Scalar)), doc="", group="Operators")

def scalar_mul_value_func(default):
    def fn(args,_):
        if args is None:
            return default
        scalar = [a.type for a in args if a.type in scalar_types][0]
        compound = [a.type for a in args if a.type not in scalar_types][0]
        if scalar != compound._wp_scalar_type_:
            raise RuntimeError("Object and coefficient must have the same scalar type when multiplying by scalar")
        return compound
    return fn

def mul_matvec_value_func(args,_):
    if args is None:
        return vec(length=Any,type=Scalar)

    if args[0].type._wp_scalar_type_ != args[1].type._wp_scalar_type_:
        raise RuntimeError(f"Can't multiply matrix and vector with different types {args[0].type._wp_scalar_type_}, {args[1].type._wp_scalar_type_}")

    if args[0].type._shape_[1] != args[1].type._length_:
        raise RuntimeError(f"Can't multiply matrix of shape {args[0].type._shape_} and vector with length {args[1].type._length_}")

    return vec(length=args[0].type._shape_[0],type=args[0].type._wp_scalar_type_)

def mul_matmat_value_func(args,_):
    if args is None:
        return mat(length=Any,type=Scalar)

    if args[0].type._wp_scalar_type_ != args[1].type._wp_scalar_type_:
        raise RuntimeError(f"Can't multiply matrices with different types {args[0].type._wp_scalar_type_}, {args[1].type._wp_scalar_type_}")

    if args[0].type._shape_[1] != args[1].type._shape_[0]:
        raise RuntimeError(f"Can't multiply matrix of shapes {args[0].type._shape_} and {args[1].type._shape_}")

    return mat(shape=(args[0].type._shape_[0],args[1].type._shape_[1]),type=args[0].type._wp_scalar_type_)

def mul_spatial_matvec_value_func(args,_):
    if args is None:
        return spatial_vector_t(type=Scalar)

    if args[0].type._wp_scalar_type_ != args[1].type._wp_scalar_type_:
        raise RuntimeError(f"Can't multiply spatial matrix and spatial vector with different types {args[0].type._wp_scalar_type_}, {args[1].type._wp_scalar_type_}")

    return args[1].type

add_builtin("mul", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar), doc="", group="Operators")
add_builtin("mul", input_types={"x": vec(length=Any,type=Scalar), "y": Scalar}, value_func=scalar_mul_value_func(vec(length=Any,type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": Scalar, "y": vec(length=Any,type=Scalar)}, value_func=scalar_mul_value_func(vec(length=Any,type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": quaternion(type=Scalar), "y": Scalar}, value_func=scalar_mul_value_func(quaternion(type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": Scalar, "y": quaternion(type=Scalar)}, value_func=scalar_mul_value_func(quaternion(type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": quaternion(type=Scalar), "y": quaternion(type=Scalar)}, value_func=sametype_value_func(quaternion(type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": Scalar, "y": mat(shape=(Any,Any),type=Scalar)}, value_func=scalar_mul_value_func(mat(shape=(Any,Any),type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": Scalar}, value_func=scalar_mul_value_func(mat(shape=(Any,Any),type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": vec(length=Any,type=Scalar)}, value_func=mul_matvec_value_func, doc="", group="Operators")
add_builtin("mul", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": mat(shape=(Any,Any),type=Scalar)}, value_func=mul_matmat_value_func, doc="", group="Operators")
add_builtin("mul", input_types={"x": spatial_vector_t(type=Scalar), "y": Scalar}, value_func=scalar_mul_value_func(spatial_vector_t(type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": spatial_matrix_t(type=Scalar), "y": spatial_matrix_t(type=Scalar)}, value_func=sametype_value_func(spatial_matrix_t(type=Scalar)), doc="", group="Operators")
add_builtin("mul", input_types={"x": spatial_matrix_t(type=Scalar), "y": spatial_vector_t(type=Scalar)}, value_func=mul_spatial_matvec_value_func, doc="", group="Operators")
add_builtin("mul", input_types={"x": transform_t(type=Scalar), "y": transform_t(type=Scalar)}, value_func=sametype_value_func(transform_t(type=Scalar)), doc="", group="Operators")

add_builtin("mod", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar), doc="", group="Operators")

add_builtin("div", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar), doc="", group="Operators")
add_builtin("div", input_types={"x": vec(length=Any,type=Scalar), "y": Scalar}, value_func=scalar_mul_value_func(vec(length=Any,type=Scalar)), doc="", group="Operators")
add_builtin("div", input_types={"x": mat(shape=(Any,Any),type=Scalar), "y": Scalar}, value_func=scalar_mul_value_func(mat(shape=(Any,Any),type=Scalar)), doc="", group="Operators")

add_builtin("floordiv", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar), doc="", group="Operators")

add_builtin("neg", input_types={"x": Scalar}, value_func=sametype_value_func(Scalar), doc="", group="Operators")
add_builtin("neg", input_types={"x": vec(length=Any,type=Scalar)}, value_func=sametype_value_func(vec(length=Any,type=Scalar)), doc="", group="Operators")
add_builtin("neg", input_types={"x": quaternion(type=Scalar)}, value_func=sametype_value_func(quaternion(type=Scalar)), doc="", group="Operators")
add_builtin("neg", input_types={"x": mat(shape=(Any,Any),type=Scalar)}, value_func=sametype_value_func(mat(shape=(Any,Any),type=Scalar)), doc="", group="Operators")

add_builtin("unot", input_types={"b": bool}, value_type=bool, doc="", group="Operators")
