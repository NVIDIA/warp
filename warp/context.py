from copy import deepcopy
import math
import os
import sys
import imp
import subprocess
import typing
import timeit
import cProfile
import inspect
import hashlib

from ctypes import*

from warp.types import *
from warp.utils import *

import warp.codegen
import warp.build
import warp.config

# represents either a built-in or user-defined function
class Function:

    def __init__(self, func, key, namespace, input_types={}, value_type=None, module=None, doc="", group=""):
        
        self.func = func   # points to Python function decorated with @wp.func, may be None for builtins
        self.key = key
        self.namespace = namespace
        self.value_type = value_type
        self.input_types = input_types
        self.doc = doc
        self.group = group
        self.module = module

        if (func):
            self.adj = warp.codegen.Adjoint(func)

        if (module):
            module.register_function(self)

# caches source and compiled entry points for a kernel (will be populated after module loads)
class Kernel:
    
    def __init__(self, func, key, module):

        self.func = func
        self.module = module
        self.key = key

        self.forward_cpu = None
        self.backward_cpu = None
        
        self.forward_cuda = None
        self.backward_cuda = None

        self.adj = warp.codegen.Adjoint(func)

        if (module):
            module.register_kernel(self)

    # lookup and cache entry points based on name, called after compilation / module load
    def hook(self):

        dll = self.module.dll
        cuda = self.module.cuda

        if (dll):

            try:
                self.forward_cpu = eval("dll." + self.func.__name__ + "_cpu_forward")
                self.backward_cpu = eval("dll." + self.func.__name__ + "_cpu_backward")
            except:
                print(f"Could not load CPU methods for kernel {self.func.__name__}")

        if (cuda):

            try:
                self.forward_cuda = runtime.core.cuda_get_kernel(self.module.cuda, (self.func.__name__ + "_cuda_kernel_forward").encode('utf-8'))
                self.backward_cuda = runtime.core.cuda_get_kernel(self.module.cuda, (self.func.__name__ + "_cuda_kernel_backward").encode('utf-8'))
            except:
                print(f"Could not load CUDA methods for kernel {self.func.__name__}")


#----------------------

# decorator to register function, @func
def func(f):

    m = get_module(f.__module__)
    f = Function(func=f, key=f.__name__, namespace="", module=m, value_type=None)   # value_type not known yet, will be infered during code-gen

    return f

# decorator to register kernel, @kernel
def kernel(f):

    m = get_module(f.__module__)
    k = Kernel(func=f, key=f.__name__, module=m)

    return k


builtin_functions = {}

# decorator to register a built-in function @builtin
def builtin(key):
    def insert(c):
        
        func = Function(func=None, key=key, namespace="wp::", value_type=c.value_type)
        builtin_functions[key] = func

    return insert

def add_builtin(key, input_types={}, value_type=None, doc="", group="Other"):

    # lambda to return a constant type for an overload
    def value_func(arg):
        return value_type

    func = Function(func=None, key=key, namespace="wp::", input_types=input_types, value_type=value_func, doc=doc, group=group)

    if key in builtin_functions:
        builtin_functions[key].append(func)
    else:
        builtin_functions[key] = [func]

#---------------------------------
# built-in operators

add_builtin("add", input_types={"x": int, "y": int}, value_type=int, doc="", group="Operators")
add_builtin("add", input_types={"x": float, "y": float}, value_type=float, doc="", group="Operators")
add_builtin("add", input_types={"x": vec3, "y": vec3}, value_type=vec3, doc="", group="Operators")
add_builtin("add", input_types={"x": vec4, "y": vec4}, value_type=vec4, doc="", group="Operators")
add_builtin("add", input_types={"x": quat, "y": quat}, value_type=quat, doc="", group="Operators")
add_builtin("add", input_types={"x": mat22, "y": mat22}, value_type=mat22, doc="", group="Operators")
add_builtin("add", input_types={"x": mat33, "y": mat33}, value_type=mat33, doc="", group="Operators")
add_builtin("add", input_types={"x": mat44, "y": mat44}, value_type=mat44, doc="", group="Operators")
add_builtin("add", input_types={"x": spatial_vector, "y": spatial_vector}, value_type=spatial_vector, doc="", group="Operators")
add_builtin("add", input_types={"x": spatial_matrix, "y": spatial_matrix}, value_type=spatial_matrix, doc="", group="Operators")

add_builtin("sub", input_types={"x": int, "y": int}, value_type=int, doc="", group="Operators")
add_builtin("sub", input_types={"x": float, "y": float}, value_type=float, doc="", group="Operators")
add_builtin("sub", input_types={"x": vec3, "y": vec3}, value_type=vec3, doc="", group="Operators")
add_builtin("sub", input_types={"x": vec4, "y": vec4}, value_type=vec4, doc="", group="Operators")
add_builtin("sub", input_types={"x": mat22, "y": mat22}, value_type=mat22, doc="", group="Operators")
add_builtin("sub", input_types={"x": mat33, "y": mat33}, value_type=mat33, doc="", group="Operators")
add_builtin("sub", input_types={"x": mat44, "y": mat44}, value_type=mat44, doc="", group="Operators")
add_builtin("sub", input_types={"x": spatial_vector, "y": spatial_vector}, value_type=spatial_vector, doc="", group="Operators")
add_builtin("sub", input_types={"x": spatial_matrix, "y": spatial_matrix}, value_type=spatial_matrix, doc="", group="Operators")

add_builtin("mul", input_types={"x": int, "y": int}, value_type=int, doc="", group="Operators")
add_builtin("mul", input_types={"x": float, "y": float}, value_type=float, doc="", group="Operators")
add_builtin("mul", input_types={"x": float, "y": vec3}, value_type=vec3, doc="", group="Operators")
add_builtin("mul", input_types={"x": float, "y": vec4}, value_type=vec4, doc="", group="Operators")
add_builtin("mul", input_types={"x": vec3, "y": float}, value_type=vec3, doc="", group="Operators")
add_builtin("mul", input_types={"x": vec4, "y": float}, value_type=vec4, doc="", group="Operators")
add_builtin("mul", input_types={"x": quat, "y": float}, value_type=quat, doc="", group="Operators")
add_builtin("mul", input_types={"x": quat, "y": quat}, value_type=quat, doc="", group="Operators")
add_builtin("mul", input_types={"x": mat22, "y": float}, value_type=mat22, doc="", group="Operators")
add_builtin("mul", input_types={"x": mat33, "y": float}, value_type=mat33, doc="", group="Operators")
add_builtin("mul", input_types={"x": mat33, "y": vec3}, value_type=vec3, doc="", group="Operators")
add_builtin("mul", input_types={"x": mat33, "y": mat33}, value_type=mat33, doc="", group="Operators")
add_builtin("mul", input_types={"x": mat44, "y": float}, value_type=mat44, doc="", group="Operators")
add_builtin("mul", input_types={"x": mat44, "y": vec4}, value_type=vec4, doc="", group="Operators")
add_builtin("mul", input_types={"x": mat44, "y": mat44}, value_type=mat44, doc="", group="Operators")
add_builtin("mul", input_types={"x": spatial_vector, "y": float}, value_type=spatial_vector, doc="", group="Operators")
add_builtin("mul", input_types={"x": spatial_matrix, "y": spatial_matrix}, value_type=spatial_matrix, doc="", group="Operators")
add_builtin("mul", input_types={"x": spatial_matrix, "y": spatial_vector}, value_type=spatial_vector, doc="", group="Operators")

add_builtin("mod", input_types={"x": int, "y": int}, value_type=int, doc="", group="Operators")
add_builtin("mod", input_types={"x": float, "y": float}, value_type=float, doc="", group="operators")

add_builtin("div", input_types={"x": int, "y": int}, value_type=int, doc="", group="Operators")
add_builtin("div", input_types={"x": float, "y": float}, value_type=float, doc="", group="Operators")
add_builtin("div", input_types={"x": vec3, "y": float}, value_type=vec3, doc="", group="Operators")

add_builtin("neg", input_types={"x": int}, value_type=int, doc="", group="Operators")
add_builtin("neg", input_types={"x": float}, value_type=float, doc="", group="Operators")
add_builtin("neg", input_types={"x": vec3}, value_type=vec3, doc="", group="Operators")
add_builtin("neg", input_types={"x": vec4}, value_type=vec4, doc="", group="Operators")
add_builtin("neg", input_types={"x": quat}, value_type=quat, doc="", group="Operators")
add_builtin("neg", input_types={"x": mat33}, value_type=mat33, doc="", group="Operators")
add_builtin("neg", input_types={"x": mat44}, value_type=mat44, doc="", group="Operators")

add_builtin("min", input_types={"x": int, "y": int}, value_type=int, doc="", group="Scalar Math")
add_builtin("min", input_types={"x": float, "y": float}, value_type=float, doc="", group="Scalar Math")

add_builtin("max", input_types={"x": int, "y": int}, value_type=int, doc="", group="Scalar Math")
add_builtin("max", input_types={"x": float, "y": float}, value_type=float, doc="", group="Scalar Math")

add_builtin("clamp", input_types={"x": float, "a": float, "b": float}, value_type=float, doc="", group="Scalar Math")
add_builtin("clamp", input_types={"x": int, "a": int, "b": int}, value_type=int, doc="", group="Scalar Math")

add_builtin("step", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")
add_builtin("nonzero", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")
add_builtin("sign", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")
add_builtin("abs", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")
add_builtin("sin", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")
add_builtin("cos", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")
add_builtin("acos", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")
add_builtin("sqrt", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")

add_builtin("dot", input_types={"x": vec3, "y": vec3}, value_type=float, doc="", group="Vector Math")
add_builtin("dot", input_types={"x": vec4, "y": vec4}, value_type=float, doc="", group="Vector Math")
add_builtin("dot", input_types={"x": quat, "y": quat}, value_type=float, doc="", group="Vector Math")

add_builtin("cross", input_types={"x": vec3, "y": vec3}, value_type=vec3, doc="", group="Vector Math")
add_builtin("skew", input_types={"x": vec3}, value_type=mat33, doc="", group="Vector Math"),

add_builtin("length", input_types={"x": vec3}, value_type=float, doc="", group="Vector Math")
add_builtin("normalize", input_types={"x": vec3}, value_type=vec3, doc="", group="Vector Math")
add_builtin("normalize", input_types={"x": vec4}, value_type=vec4, doc="", group="Vector Math")
add_builtin("normalize", input_types={"x": quat}, value_type=quat, doc="", group="Vector Math")

add_builtin("rotate", input_types={"q": quat, "p": vec3}, value_type=vec3, doc="", group="Vector Math")
add_builtin("rotate_inv", input_types={"q": quat, "p": vec3}, value_type=vec3, doc="", group="Vector Math")

add_builtin("determinant", input_types={"m": mat22}, value_type=float, doc="", group="Vector Math")
add_builtin("determinant", input_types={"m": mat33}, value_type=float, doc="", group="Vector Math")
add_builtin("transpose", input_types={"m": mat22}, value_type=mat22, doc="", group="Vector Math")
add_builtin("transpose", input_types={"m": mat33}, value_type=mat33, doc="", group="Vector Math")
add_builtin("transpose", input_types={"m": mat44}, value_type=mat44, doc="", group="Vector Math")
add_builtin("transpose", input_types={"m": spatial_matrix}, value_type=spatial_matrix, doc="", group="Vector Math")

# type construtors
add_builtin("int", input_types={"x": int}, value_type=int, doc="", group="Scalar Math")
add_builtin("int", input_types={"x": float}, value_type=int, doc="", group="Scalar Math")

add_builtin("float", input_types={"x": int}, value_type=float, doc="", group="Scalar Math")
add_builtin("float", input_types={"x": float}, value_type=float, doc="", group="Scalar Math")

add_builtin("vec3", input_types={}, value_type=vec3, doc="", group="Vector Math")
add_builtin("vec3", input_types={"x": float, "y": float, "z": float}, value_type=vec3, doc="", group="Vector Math")
add_builtin("vec3", input_types={"s": float}, value_type=vec3, doc="", group="Vector Math")

add_builtin("vec4", input_types={}, value_type=vec4, doc="", group="Vector Math")
add_builtin("vec4", input_types={"x": float, "y": float, "z": float, "w": float}, value_type=vec4, doc="", group="Vector Math")
add_builtin("vec4", input_types={"s": float}, value_type=vec4, doc="", group="Vector Math")

add_builtin("quat", input_types={}, value_type=quat, doc="", group="Quaternion Math")
add_builtin("quat", input_types={"x": float, "y": float, "z": float, "w": float}, value_type=quat, doc="", group="Quaternion Math")
add_builtin("quat", input_types={"i": vec3, "r": float}, value_type=quat, doc="", group="Quaternion Math")
add_builtin("quat_identity", input_types={}, value_type=quat, doc="", group="Quaternion Math")
add_builtin("quat_from_axis_angle", input_types={"axis": vec3, "angle": float}, value_type=quat, doc="", group="Quaternion Math")
add_builtin("quat_inverse", input_types={"q": quat}, value_type=quat, doc="", group="Quaternion Math")

add_builtin("mat22", input_types={"m00": float, "m01": float, "m10": float, "m11": float}, value_type=mat22, doc="", group="Vector Math")
add_builtin("mat33", input_types={"c0": vec3, "c1": vec3, "c2": vec3 }, value_type=mat33, doc="", group="Vector Math")
add_builtin("mat44", input_types={"c0": vec4, "c1": vec4, "c2": vec4, "c3": vec4 }, value_type=mat44, doc="", group="Vector Math")

add_builtin("spatial_vector", input_types={}, value_type=spatial_vector, doc="", group="Spatial Math")
add_builtin("spatial_vector", input_types={"a": float, "b": float, "c": float, "d": float, "e": float, "f": float}, value_type=spatial_vector, doc="", group="Spatial Math")
add_builtin("spatial_vector", input_types={"w": vec3, "v": vec3}, value_type=spatial_vector, doc="", group="Spatial Math")
add_builtin("spatial_vector", input_types={"s": float}, value_type=spatial_vector, doc="", group="Spatial Math"),

add_builtin("spatial_transform", input_types={"p": vec3, "q": quat}, value_type=spatial_transform, doc="", group="Spatial Math")
add_builtin("spatial_transform_identity", input_types={}, value_type=spatial_transform, doc="", group="Spatial Math")

add_builtin("spatial_transform_get_translation", input_types={"t": spatial_transform}, value_type=vec3, doc="", group="Spatial Math")
add_builtin("spatial_transform_get_rotation", input_types={"t": spatial_transform}, value_type=quat, doc="", group="Spatial Math")
add_builtin("spatial_transform_multiply", input_types={"a": spatial_transform, "b": spatial_transform}, value_type=spatial_transform, doc="", group="Spatial Math")

add_builtin("spatial_adjoint", input_types={"r": mat33, "s": mat33}, value_type=spatial_matrix, doc="", group="Spatial Math")
add_builtin("spatial_dot", input_types={"a": spatial_vector, "b": spatial_vector}, value_type=float, doc="", group="Spatial Math")
add_builtin("spatial_cross", input_types={"a": spatial_vector, "b": spatial_vector}, value_type=spatial_vector, doc="", group="Spatial Math")
add_builtin("spatial_cross_dual", input_types={"a": spatial_vector, "b": spatial_vector}, value_type=spatial_vector, doc="", group="Spatial Math")

add_builtin("spatial_transform_point", input_types={"t": spatial_transform, "p": vec3}, value_type=vec3, doc="Apply the transform to p treating the homogenous coordinate as w=1 (translation and rotation)", group="Spatial Math")
add_builtin("spatial_transform_vector", input_types={"t": spatial_transform, "v": vec3}, value_type=vec3, doc="Apply the transform to v treating the homogenous coordinate as w=0 (rotation only)", group="Spatial Math")

add_builtin("spatial_top", input_types={"a": spatial_vector}, value_type=vec3, doc="", group="Spatial Math")
add_builtin("spatial_bottom", input_types={"a": spatial_vector}, value_type=vec3, doc="", group="Spatial Math")

add_builtin("spatial_jacobian",
     input_types={"S": array(dtype=spatial_vector), 
                  "joint_parents": array(dtype=int),
                  "joint_qd_start": array(dtype=int),
                  "joint_start": int,
                  "joint_count": int,
                  "J_start": int,
                  "J_out": array(dtype=float)}, value_type=None, doc="", group="Spatial Math")

add_builtin("spatial_mass", input_types={"I_s": array(dtype=spatial_matrix), "joint_start": int, "joint_count": int, "M_start": int, "M": array(float)}, value_type=None, doc="", group="Spatial Math")

add_builtin("dense_gemm", 
    input_types={"m": int, 
                 "n": int, 
                 "p": int, 
                 "t1": int, 
                 "t2": int, 
                 "A": array(dtype=float), 
                 "B": array(dtype=float), 
                 "C": array(dtype=float) }, value_type=None, doc="", group="Linear Algebra")

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
                 "C": array(dtype=float)}, value_type=None, doc="", group="Linear Algebra")


add_builtin("dense_chol",
    input_types={"n": int, 
                 "A": array(dtype=float), 
                 "regularization": float, 
                 "L": array(dtype=float)}, value_type=None, doc="", group="Linear Algebra")

add_builtin("dense_chol_batched",
    input_types={"A_start": array(dtype=int),
                 "A_dim": array(dtype=int),
                 "A": array(dtype=float),
                 "regularization": float,
                 "L": array(dtype=float)}, value_type=None, doc="", group="Linear Algebra")

add_builtin("dense_subs", 
    input_types={"n": int, 
                 "L": array(dtype=float), 
                 "b": array(dtype=float), 
                 "x": array(dtype=float)}, value_type=None, doc="", group="Linear Algebra")

add_builtin("dense_solve", 
    input_types={"n": int, 
                 "A": array(dtype=float), 
                 "L": array(dtype=float), 
                 "b": array(dtype=float), 
                 "x": array(dtype=float)}, value_type=None, doc="", group="Linear Algebra")

add_builtin("dense_solve_batched",
    input_types={"b_start": array(dtype=int), 
                 "A_start": array(dtype=int),
                 "A_dim": array(dtype=int),
                 "A": array(dtype=float),
                 "L": array(dtype=float),
                 "b": array(dtype=float),
                 "x": array(dtype=float)}, value_type=None, doc="", group="Linear Algebra")

add_builtin("transform_point", input_types={"m": mat44, "p": vec3}, value_type=vec3, doc="", group="Vector Math")
add_builtin("transform_vector", input_types={"m": mat44, "v": vec3}, value_type=vec3, doc="", group="Vector Math")

add_builtin("mesh_query_point", input_types={"id": uint64, "point": vec3, "max_dist": float, "inside": float, "face": int, "bary_u": float, "bary_v": float}, value_type=bool, doc="", group="Geometry")
add_builtin("mesh_query_ray", input_types={"id": uint64, "start": vec3, "dir": vec3, "max_t": float, "t": float, "bary_u": float, "bary_v": float, "sign": float, "normal": vec3, "face": int}, value_type=bool, doc="", group="Geometry")

add_builtin("mesh_eval_position", input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float}, value_type=vec3, doc="", group="Geometry")
add_builtin("mesh_eval_velocity", input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float}, value_type=vec3, doc="", group="Geometry")

# helpers

add_builtin("tid", input_types={}, value_type=int, doc="Return the current thread id.", group="Utility")

@builtin("select")
class SelectFunc:
    @staticmethod
    def value_type(args):
        return args[1].type

@builtin("copy")
class CopyFunc:
    @staticmethod
    def value_type(args):
        return None

@builtin("load")
class LoadFunc:
    @staticmethod
    def value_type(args):
        if (type(args[0].type) != warp.types.array):
            raise Exception("load() argument 0 must be a array")
        if (args[1].type != int and args[1].type != warp.types.int32 and args[1].type != warp.types.int64 and args[1].type != warp.types.uint64):
            raise Exception("load() argument input 1 must be an integer type")

        return args[0].type.dtype


@builtin("store")
class StoreFunc:
    @staticmethod
    def value_type(args):
        if (type(args[0].type) != warp.types.array):
            raise Exception("store() argument 0 must be a array")
        if (args[1].type != int and args[1].type != warp.types.int32 and args[1].type != warp.types.int64 and args[1].type != warp.types.uint64):
            raise Exception("store() argument input 1 must be an integer type")
        if (args[2].type != args[0].type.dtype):
            raise Exception("store() argument input 2 ({}) must be of the same type as the array ({})".format(args[2].type, args[0].type.dtype))

        return None


@builtin("atomic_add")
class AtomicAddFunc:
    @staticmethod
    def value_type(args):

        if (type(args[0].type) != warp.types.array):
            raise Exception("store() argument 0 must be a array")
        if (args[1].type != int and args[1].type != warp.types.int32 and args[1].type != warp.types.int64 and args[1].type != warp.types.uint64):
            raise Exception("store() argument input 1 must be an integer type")
        if (args[2].type != args[0].type.dtype):
            raise Exception("store() argument input 2 ({}) must be of the same type as the array ({})".format(args[2].type, args[0].type.dtype))

        return args[0].type.dtype


@builtin("atomic_sub")
class AtomicSubFunc:
    @staticmethod
    def value_type(args):

        if (type(args[0].type) != warp.types.array):
            raise Exception("store() argument 0 must be a array")
        if (args[1].type != int and args[1].type != warp.types.int32 and args[1].type != warp.types.int64 and args[1].type != warp.types.uint64):
            raise Exception("store() argument input 1 must be an integer type")
        if (args[2].type != args[0].type.dtype):
            raise Exception("store() argument input 2 ({}) must be of the same type as the array ({})".format(args[2].type, args[0].type.dtype))

        return args[0].type.dtype


@builtin("index")
class IndexFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("print")
class PrintFunc:
    @staticmethod
    def value_type(args):
        return None

@builtin("expect_eq")
class ExpectEqFunc:
    @staticmethod
    def value_type(args):
        return None


def wrap(adj):
    def value_type(args):
        if (adj.return_var):
            return adj.return_var.type
        else:
            return None

    return value_type


# global dictionary of modules
user_modules = {}

def get_module(m):

    if (m not in user_modules):
        user_modules[m] = warp.context.Module(str(m))

    return user_modules[m]


#---------------------------------------------
# stores all functions and kernels for a module
# create a hash of the function to use for checking build cache

class Module:

    def __init__(self, name):

        self.name = name
        self.kernels = {}
        self.functions = {}

        self.dll = None
        self.cuda = None

        self.loaded = False


    def register_kernel(self, kernel):

        if kernel.key in self.kernels:
            
            # if kernel is replacing an old one then assume it has changed and 
            # force a rebuild / reload of the dynamic libary 
            if (self.dll):
                warp.build.unload_dll(self.dll)

            if (self.cuda):
                runtime.core.cuda_unload_module(self.cuda)
                
            self.dll = None
            self.cuda = None
            self.loaded = False

        # register new kernel
        self.kernels[kernel.key] = kernel


    def register_function(self, func):
        self.functions[func.key] = func

    def hash_module(self):
        
        h = hashlib.sha256()

        for func in self.functions.values():
            s = func.adj.source
            h.update(bytes(s, 'utf-8'))
            
        for kernel in self.kernels.values():       
            s = kernel.adj.source
            h.update(bytes(s, 'utf-8'))

        # append any configuration parameters
        h.update(bytes(warp.config.mode, 'utf-8'))

        return h.digest()

    def load(self):

        with ScopedTimer(f"Module {self.name} load"):

            enable_cpu = warp.is_cpu_available()
            enable_cuda = warp.is_cuda_available()

            module_name = "wp_" + self.name

            include_path = os.path.dirname(os.path.realpath(__file__))
            build_path = os.path.dirname(os.path.realpath(__file__)) + "/bin"
            gen_path = os.path.dirname(os.path.realpath(__file__)) + "/gen"

            cache_path = build_path + "/" + module_name + ".hash"
            module_path = build_path + "/" + module_name

            ptx_path = module_path + ".ptx"
            dll_path = module_path + ".dll"

            if (os.path.exists(build_path) == False):
                os.mkdir(build_path)

            # test cache
            module_hash = self.hash_module()

            if (os.path.exists(cache_path)):

                f = open(cache_path, 'rb')
                cache_hash = f.read()
                f.close()

                if (cache_hash == module_hash):
                    
                    if (warp.config.verbose):
                        print("Warp: Using cached kernels for module {}".format(self.name))
                        
                    if (enable_cpu):
                        self.dll = warp.build.load_dll(dll_path)
                        
                        if (self.dll == None):
                            raise Exception(f"Could not load dll from cache {dll_path}")

                    if (enable_cuda):
                        self.cuda = warp.build.load_cuda(ptx_path)

                        if (self.cuda == None):
                            raise Exception(f"Could not load ptx from cache path: {ptx_path}")

                    self.loaded = True
                    return

            if (warp.config.verbose):
                print("Warp: Rebuilding kernels for module {}".format(self.name))


            # generate kernel source
            if (enable_cpu):
                cpp_path = gen_path + "/" + module_name + ".cpp"
                cpp_source = warp.codegen.cpu_module_header
            
            if (enable_cuda):
                cu_path = gen_path + "/" + module_name + ".cu"        
                cu_source = warp.codegen.cuda_module_header

            # kernels
            entry_points = []

            # functions
            for name, func in self.functions.items():
            
                func.adj.build(builtin_functions, self.functions)
                
                if (enable_cpu):
                    cpp_source += warp.codegen.codegen_func(func.adj, device="cpu")

                if (enable_cuda):
                    cu_source += warp.codegen.codegen_func(func.adj, device="cuda")

                # complete the function return type after we have analyzed it (infered from return statement in ast)
                func.value_type = wrap(func.adj)


            # kernels
            for kernel in self.kernels.values():

                kernel.adj.build(builtin_functions, self.functions)

                # each kernel gets an entry point in the module
                if (enable_cpu):
                    entry_points.append(kernel.func.__name__ + "_cpu_forward")
                    entry_points.append(kernel.func.__name__ + "_cpu_backward")

                    cpp_source += warp.codegen.codegen_module_decl(kernel.adj, device="cpu")
                    cpp_source += warp.codegen.codegen_kernel(kernel.adj, device="cpu")
                    cpp_source += warp.codegen.codegen_module(kernel.adj, device="cpu")

                if (enable_cuda):                
                    entry_points.append(kernel.func.__name__ + "_cuda_forward")
                    entry_points.append(kernel.func.__name__ + "_cuda_backward")

                    cpp_source += warp.codegen.codegen_module_decl(kernel.adj, device="cuda")
                    cu_source += warp.codegen.codegen_kernel(kernel.adj, device="cuda")
                    cu_source += warp.codegen.codegen_module(kernel.adj, device="cuda")


            # write cpp sources
            if (enable_cpu):
                cpp_file = open(cpp_path, "w")
                cpp_file.write(cpp_source)
                cpp_file.close()

            # write cuda sources
            if (enable_cuda):
                cu_file = open(cu_path, "w")
                cu_file.write(cu_source)
                cu_file.close()
        
            try:
                
                if (enable_cpu):
                    with ScopedTimer("Compile x86", active=warp.config.verbose):
                        warp.build.build_dll(cpp_path, None, dll_path, config=warp.config.mode)

                if (enable_cuda):
                    with ScopedTimer("Compile CUDA", active=warp.config.verbose):
                        warp.build.build_cuda(cu_path, ptx_path, config=warp.config.mode)

                # update cached output
                f = open(cache_path, 'wb')
                f.write(module_hash)
                f.close()

            except Exception as e:

                print(e)
                raise(e)

            if (enable_cpu):
                self.dll = warp.build.load_dll(dll_path)

            if (enable_cuda):
                self.cuda = warp.build.load_cuda(ptx_path)

            self.loaded = True

#-------------------------------------------
# exectution context
from ctypes import *


class Runtime:

    def __init__(self):

        # in Python 3.8 we should use os.add_dll_directory() since adding to the PATH is not supported
        bin_path = os.path.dirname(os.path.realpath(__file__)) + "/bin"
        os.environ["PATH"] += os.pathsep + bin_path

        self.core = warp.build.load_dll("warp.dll")

        # setup c-types for warp.dll
        self.core.alloc_host.restype = c_void_p
        self.core.alloc_device.restype = c_void_p
        
        self.core.mesh_create_host.restype = c_uint64
        self.core.mesh_create_host.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int]

        self.core.mesh_create_device.restype = c_uint64
        self.core.mesh_create_device.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int]

        self.core.mesh_destroy_host.argtypes = [c_uint64]
        self.core.mesh_destroy_device.argtypes = [c_uint64]

        self.core.mesh_refit_host.argtypes = [c_uint64]
        self.core.mesh_refit_device.argtypes = [c_uint64]

        self.core.cuda_check_device.restype = c_uint64
        self.core.cuda_get_context.restype = c_void_p
        self.core.cuda_get_stream.restype = c_void_p
        self.core.cuda_graph_end_capture.restype = c_void_p
        self.core.cuda_get_device_name.restype = c_char_p

        self.core.cuda_compile_program.argtypes = [c_char_p, c_char_p, c_bool, c_bool, c_char_p]
        self.core.cuda_compile_program.restype = c_size_t

        self.core.cuda_load_module.argtypes = [c_char_p]
        self.core.cuda_load_module.restype = c_void_p

        self.core.cuda_unload_module.argtypes = [c_void_p]

        self.core.cuda_get_kernel.argtypes = [c_void_p, c_char_p]
        self.core.cuda_get_kernel.restype = c_void_p
        
        self.core.cuda_launch_kernel.argtypes = [c_void_p, c_size_t, POINTER(c_void_p)]
        self.core.cuda_launch_kernel.restype = c_size_t

        self.core.init.restype = c_int
        
        error = self.core.init()

        if (error > 0):
            raise Exception("Warp Initialization failed, CUDA not found")

        # save context
        self.cuda_device = self.core.cuda_get_context()
        self.cuda_stream = self.core.cuda_get_stream()

        # initialize host build env
        if (warp.config.host_compiler == None):
            warp.config.host_compiler = warp.build.find_host_compiler()

        # print device and version information
        print("Warp initialized:")
        print("   Version: {}".format(warp.config.version))
        print("   Using CUDA device: {}".format(self.core.cuda_get_device_name().decode()))
        print("   Using CPU compiler: {}".format(warp.config.host_compiler.rstrip()))

        # global tape
        self.tape = None

    # host functions
    def alloc_host(self, num_bytes):
        ptr = self.core.alloc_host(num_bytes)       
        return ptr

    def free_host(self, ptr):
        self.core.free_host(cast(ptr,POINTER(c_int)))

    # device functions
    def alloc_device(self, num_bytes):
        ptr = self.core.alloc_device(c_size_t(num_bytes))
        return ptr

    def free_device(self, ptr):
        self.core.free_device(cast(ptr,POINTER(c_int)))

    def verify_device(self):

        if warp.config.verify_cuda:

            context = self.core.cuda_get_context()
            if (context != self.cuda_device):
                raise RuntimeError("Unexpected CUDA device, original {} current: {}".format(self.cuda_device, context))

            err = self.core.cuda_check_device()
            if (err != 0):
                raise RuntimeError("CUDA error detected: {}".format(err))



# global entry points 
def is_cpu_available():
    return warp.config.host_compiler != None

def is_cuda_available():
    return runtime.cuda_device != None


def capture_begin():
    # ensure that all modules are loaded, this is necessary
    # since cuLoadModule() is not permitted during capture
    for m in user_modules.values():
        m.load()

    runtime.core.cuda_graph_begin_capture()

def capture_end():
    return runtime.core.cuda_graph_end_capture()

def capture_launch(graph):
    runtime.core.cuda_graph_launch(c_void_p(graph))


def copy(dest, src):

    num_bytes = src.length*type_size_in_bytes(src.dtype)

    if (src.device == "cpu" and dest.device == "cuda"):
        runtime.core.memcpy_h2d(c_void_p(dest.data), c_void_p(src.data), c_size_t(num_bytes))

    elif (src.device == "cuda" and dest.device == "cpu"):
        runtime.core.memcpy_d2h(c_void_p(dest.data), c_void_p(src.data), c_size_t(num_bytes))

    elif (src.device == "cpu" and dest.device == "cpu"):
        runtime.core.memcpy_h2h(c_void_p(dest.data), c_void_p(src.data), c_size_t(num_bytes))

    elif (src.device == "cuda" and dest.device == "cuda"):
        runtime.core.memcpy_d2d(c_void_p(dest.data), c_void_p(src.data), c_size_t(num_bytes))
    
    else:
        raise RuntimeError("Unexpected source and destination combination")


def zeros(n, dtype=float, device="cpu", requires_grad=False):

    num_bytes = n*warp.types.type_size_in_bytes(dtype)

    if (device == "cpu"):
        ptr = runtime.core.alloc_host(num_bytes) 
        runtime.core.memset_host(cast(ptr,POINTER(c_int)), c_int(0), c_size_t(num_bytes))

    if( device == "cuda"):
        ptr = runtime.alloc_device(num_bytes)
        runtime.core.memset_device(cast(ptr,POINTER(c_int)), c_int(0), c_size_t(num_bytes))

    if (ptr == None and num_bytes > 0):
        raise RuntimeError("Memory allocation failed on device: {} for {} bytes".format(device, num_bytes))
    else:
        # construct array
        return warp.types.array(dtype=dtype, length=n, capacity=num_bytes, data=ptr, context=runtime, device=device, owner=True, requires_grad=requires_grad)

def zeros_like(src, requires_grad=False):

    arr = zeros(len(src), dtype=src.dtype, device=src.device, requires_grad=requires_grad)
    return arr

def clone(src):
    dest = empty(len(src), dtype=src.dtype, device=src.device, requires_grad=src.requires_grad)
    copy(dest, src)

    return dest

def empty(n, dtype=float, device="cpu", requires_grad=False):

    # todo: implement uninitialized allocation
    return zeros(n, dtype, device, requires_grad=requires_grad)  

def empty_like(src, requires_grad=False):

    arr = empty(len(src), dtype=src.dtype, device=src.device, requires_grad=requires_grad)
    return arr


def from_numpy(arr, dtype, device="cpu", requires_grad=False):

    return warp.array(data=arr, dtype=dtype, device=device, requires_grad=requires_grad)


def synchronize():
    runtime.core.synchronize()


def launch(kernel, dim, inputs, outputs=[], adj_inputs=[], adj_outputs=[], device="cpu", adjoint=False):

    if (dim > 0):

        # delay load modules
        if (kernel.module.loaded == False):
            kernel.module.load()

        # first param is the number of threads
        params = []
        params.append(c_long(dim))

        # converts arguments to expected types and packs into params to passed to kernel def.
        def pack_args(args, params):

            for i, a in enumerate(args):

                arg_type = kernel.adj.args[i].type

                if (isinstance(arg_type, warp.types.array)):

                    if (a is None):
                        
                        # allow for NULL arrays
                        params.append(c_int64(0))

                    else:

                        # check subtype
                        if (a.dtype != arg_type.dtype):
                            raise RuntimeError("Array dtype {} does not match kernel signature {} for param: {}".format(a.dtype, arg_type.dtype, kernel.adj.args[i].label))

                        # check device
                        if (a.device != device):
                            raise RuntimeError("Launching kernel on device={} where input array is on device={}. Arrays must live on the same device".format(device, i.device))
            
                        params.append(c_int64(a.data))

                # try and convert arg to correct type
                elif (arg_type == warp.types.float32):
                    params.append(c_float(a))

                elif (arg_type == warp.types.int32):
                    params.append(c_int32(a))

                elif (arg_type == warp.types.int64):
                    params.append(c_int64(a))

                elif (arg_type == warp.types.uint64):
                    params.append(c_uint64(a))
                
                elif isinstance(a, np.ndarray) or isinstance(a, tuple):

                    # force conversion to ndarray (handle tuple case)
                    a = np.array(a)

                    # flatten to 1D array
                    v = a.flatten()
                    if (len(v) != arg_type.length()):
                        raise RuntimeError(f"Kernel parameter {kernel.adj.args[i].label} has incorrect value length {len(v)}, expected {arg_type.length()}")

                    # try and convert numpy array to builtin numeric type vec3, vec4, mat33, etc
                    x = arg_type()
                    for i in range(arg_type.length()):
                        x.value[i] = v[i]

                    params.append(x)

                else:
                    raise RuntimeError(f"Unknown parameter type {type(a)} for param {kernel.adj.args[i].label}, expected {arg_type}")

        fwd_args = inputs + outputs
        adj_args = adj_inputs + adj_outputs

        pack_args(fwd_args, params)
        pack_args(adj_args, params)

        # late bind
        if (kernel.forward_cpu == None or kernel.forward_cuda == None):
            kernel.hook()

        # run kernel
        if device == 'cpu':

            if (adjoint):
                kernel.backward_cpu(*params)
            else:
                kernel.forward_cpu(*params)

        
        elif device.startswith("cuda"):
            kernel_args = [c_void_p(addressof(x)) for x in params]
            kernel_params = (c_void_p * len(kernel_args))(*kernel_args)

            if (adjoint):
                runtime.core.cuda_launch_kernel(kernel.backward_cuda, dim, kernel_params)
            else:
                runtime.core.cuda_launch_kernel(kernel.forward_cuda, dim, kernel_params)

            runtime.verify_device()

    # record on tape if one is active
    if (runtime.tape):
        runtime.tape.record(kernel, dim, inputs, outputs, device)


def type_str(t):
    if (t == None):
        return "None"
    if (isinstance(t, array)):
        #s = type(t.dtype)
        return f"array({t.dtype.__name__})"
    else:
        return t.__name__
        
def print_function(f):
    args = ", ".join(f"{k}: {type_str(v)}" for k,v in f.input_types.items())

    return_type = ""

    try:
        return_type = " -> " + type_str(f.value_type(None))
    except:
        pass

    print(f".. function:: {f.key}({args}){return_type}")
    print("")
    
    if (f.doc != ""):
        print(f"   {f.doc}")
        print("")

    print()
    
def print_builtins():

    # build dictionary of all functions by group
    groups = {}

    for k, f in builtin_functions.items():
        
        g = None

        if (isinstance(f, list)):
            g = f[0].group  # assumes all overloads have the same group
        else:
            g = f.group

        if (g not in groups):
            groups[g] = []
        
        groups[g].append(f)

    for k, g in groups.items():
        print("\n")
        print(k)
        print("---------------")

        for f in g:            
            if isinstance(f, list):
                for x in f:
                    print_function(x)
            else:
                print_function(f)


# ensures that correct CUDA is set for the guards lifetime
# restores the previous CUDA context on exit
class ScopedCudaGuard:

    def __init__(self):
        pass

    def __enter__(self):
        runtime.core.cuda_acquire_context()

    def __exit__(self, exc_type, exc_value, traceback):
        runtime.core.cuda_restore_context()


# initialize global runtime
runtime = None

def init():
    global runtime

    if (runtime == None):
        runtime = Runtime()


