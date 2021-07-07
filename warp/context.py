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

    def __init__(self, func, key, namespace, module=None, value_type=None):
        
        self.func = func   # may be None for built-in funcs
        self.key = key
        self.namespace = namespace
        self.value_type = value_type
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

    # cache entry points based on name, called after compilation / module load
    def hook(self):

        dll = self.module.dll
        cuda = self.module.cuda

        if (dll):

            try:
                self.forward_cpu = eval("dll." + self.func.__name__ + "_cpu_forward")
                self.backward_cpu = eval("dll." + self.func.__name__ + "_cpu_backward")
            except:
                print("Could not load CPU methods for kernel {}".format(self.func.__name__))

        if (cuda):

            # try:
            #     self.forward_cuda = eval("dll." + self.func.__name__ + "_cuda_forward")
            #     self.backward_cuda = eval("dll." + self.func.__name__ + "_cuda_backward")
            # except:
            #     print("Could not load CUDA methods for kernel {}".format(self.func.__name__))

            self.forward_cuda = runtime.core.cuda_get_kernel(self.module.cuda, (self.func.__name__ + "_cuda_kernel_forward").encode('utf-8'))
            self.backward_cuda = runtime.core.cuda_get_kernel(self.module.cuda, (self.func.__name__ + "_cuda_kernel_backward").encode('utf-8'))


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


#---------------------------------
# built-in operators +,-,*,/


@builtin("add")
class AddFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


@builtin("sub")
class SubFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


@builtin("mod")
class ModFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


@builtin("mul")
class MulFunc:
    @staticmethod
    def value_type(args):

        # int x int
        if (warp.types.type_is_int(args[0].type) and warp.types.type_is_int(args[1].type)):
            return int

        # float x int
        elif (warp.types.type_is_float(args[1].type) and warp.types.type_is_int(args[0].type)):
            return float

        # int x float
        elif (warp.types.type_is_int(args[0].type) and warp.types.type_is_float(args[1].type)):
            return float

        # scalar x object
        elif (warp.types.type_is_float(args[0].type)):
            return args[1].type

        # object x scalar
        elif (warp.types.type_is_float(args[1].type)):
            return args[0].type

        # mat33 x vec3
        elif (args[0].type == mat33 and args[1].type == vec3):
            return vec3

        # mat33 x mat33
        elif(args[0].type == mat33 and args[1].type == mat33):
            return mat33

        # mat66 x vec6
        if (args[0].type == spatial_matrix and args[1].type == spatial_vector):
            return spatial_vector

        # mat66 x mat66        
        if (args[0].type == spatial_matrix and args[1].type == spatial_matrix):
            return spatial_matrix

        # quat x quat
        if (args[0].type == quat and args[1].type == quat):
            return quat

        else:
            raise Exception("Unrecognized types for multiply operator *, got {} and {}".format(args[0].type, args[1].type))


@builtin("div")
class DivFunc:
    @staticmethod
    def value_type(args):
        
        # int / int
        if (warp.types.type_is_int(args[0].type) and warp.types.type_is_int(args[1].type)):
            return int

        # float / int
        elif (warp.types.type_is_float(args[0].type) and warp.types.type_is_int(args[1].type)):
            return float

        # int / float
        elif (warp.types.type_is_int(args[0].type) and warp.types.type_is_float(args[1].type)):
            return float

        # vec3 / float
        elif (args[0].type == vec3 and warp.types.type_is_float(args[1].type)):
            return vec3

        # object / float
        elif (warp.types.type_is_float(args[0].type)):
            return args[1].type

        else:
            raise Exception("Unrecognized types for division operator /, got {} and {}".format(args[0].type, args[1].type))


@builtin("neg")
class NegFunc:
    @staticmethod
    def value_type(args):
        return args[0].type

#----------------------
# built-in functions



@builtin("min")
class MinFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


@builtin("max")
class MaxFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


@builtin("leaky_max")
class LeakyMaxFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("leaky_min")
class LeakyMinFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("clamp")
class ClampFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


@builtin("step")
class StepFunc:
    @staticmethod
    def value_type(args):
        return float

@builtin("nonzero")
class NonZeroFunc:
    @staticmethod
    def value_type(args):
        return float

@builtin("sign")
class SignFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("abs")
class AbsFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("sin")
class SinFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("cos")
class CosFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("acos")
class ACosFunc:
    @staticmethod
    def value_type(args):
        return float

@builtin("sqrt")
class SqrtFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("dot")
class DotFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("cross")
class CrossFunc:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("skew")
class SkewFunc:
    @staticmethod
    def value_type(args):
        return mat33


@builtin("length")
class LengthFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("normalize")
class NormalizeFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


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

@builtin("rotate")
class RotateFunc:
    @staticmethod
    def value_type(args):
        return vec3


@builtin("rotate_inv")
class RotateInvFunc:
    @staticmethod
    def value_type(args):
        return vec3


@builtin("determinant")
class DeterminantFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("transpose")
class TransposeFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


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
        return args[0].type.dtype


@builtin("atomic_sub")
class AtomicSubFunc:
    @staticmethod
    def value_type(args):
        return args[0].type.dtype


@builtin("tid")
class ThreadIdFunc:
    @staticmethod
    def value_type(args):
        return int


# type construtors

@builtin("float")
class FloatFunc:
    @staticmethod
    def value_type(args):
        return float

@builtin("int")
class IntFunc:
    @staticmethod
    def value_type(args):
        return int

@builtin("vec3")
class vec3Func:
    @staticmethod
    def value_type(args):
        return vec3


@builtin("quat")
class QuatFunc:
    @staticmethod
    def value_type(args):
        return quat


@builtin("quat_identity")
class QuatIdentityFunc:
    @staticmethod
    def value_type(args):
        return quat


@builtin("quat_from_axis_angle")
class QuatAxisAngleFunc:
    @staticmethod
    def value_type(args):
        return quat


@builtin("mat22")
class Mat22Func:
    @staticmethod
    def value_type(args):
        return mat22


@builtin("mat33")
class Mat33Func:
    @staticmethod
    def value_type(args):
        return mat33

@builtin("mat44")
class Mat44Func:
    @staticmethod
    def value_type(args):
        return mat44


@builtin("spatial_vector")
class SpatialVectorFunc:
    @staticmethod
    def value_type(args):
        return spatial_vector


# built-in spatial builtin_operators
@builtin("spatial_transform")
class TransformFunc:
    @staticmethod
    def value_type(args):
        return spatial_transform


@builtin("spatial_transform_identity")
class TransformIdentity:
    @staticmethod
    def value_type(args):
        return spatial_transform

@builtin("inverse")
class Inverse:
    @staticmethod
    def value_type(args):
        return quat


# @builtin("spatial_transform_inverse")
# class TransformInverse:
#     @staticmethod
#     def value_type(args):
#         return spatial_transform


@builtin("spatial_transform_get_translation")
class TransformGetTranslation:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("spatial_transform_get_rotation")
class TransformGetRotation:
    @staticmethod
    def value_type(args):
        return quat

@builtin("spatial_transform_multiply")
class TransformMulFunc:
    @staticmethod
    def value_type(args):
        return spatial_transform

# @builtin("spatial_transform_inertia")
# class TransformInertiaFunc:
#     @staticmethod
#     def value_type(args):
#         return spatial_matrix

@builtin("spatial_adjoint")
class SpatialAdjoint:
    @staticmethod
    def value_type(args):
        return spatial_matrix

@builtin("spatial_dot")
class SpatialDotFunc:
    @staticmethod
    def value_type(args):
        return float

@builtin("spatial_cross")
class SpatialDotFunc:
    @staticmethod
    def value_type(args):
        return spatial_vector

@builtin("spatial_cross_dual")
class SpatialDotFunc:
    @staticmethod
    def value_type(args):
        return spatial_vector

@builtin("spatial_transform_point")
class SpatialTransformPointFunc:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("spatial_transform_vector")
class SpatialTransformVectorFunc:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("spatial_top")
class SpatialTopFunc:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("spatial_bottom")
class SpatialBottomFunc:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("spatial_jacobian")
class SpatialJacobian:
    @staticmethod
    def value_type(args):
        return None
    
@builtin("spatial_mass")
class SpatialMass:
    @staticmethod
    def value_type(args):
        return None

@builtin("dense_gemm")
class DenseGemm:
    @staticmethod
    def value_type(args):
        return None

@builtin("dense_gemm_batched")
class DenseGemmBatched:
    @staticmethod
    def value_type(args):
        return None        

@builtin("dense_chol")
class DenseChol:
    @staticmethod
    def value_type(args):
        return None

@builtin("dense_chol_batched")
class DenseCholBatched:
    @staticmethod
    def value_type(args):
        return None        

@builtin("dense_subs")
class DenseSubs:
    @staticmethod
    def value_type(args):
        return None

@builtin("dense_solve")
class DenseSolve:
    @staticmethod
    def value_type(args):
        return None

@builtin("dense_solve_batched")
class DenseSolveBatched:
    @staticmethod
    def value_type(args):
        return None        


# mat44 point/vec transforms
@builtin("transform_point")
class TransformPointFunc:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("transform_vector")
class TransformVectorFunc:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("mesh_query_point")
class MeshQueryPoint:
    @staticmethod
    def value_type(args):
        return bool

@builtin("mesh_query_ray")
class MeshQueryRay:
    @staticmethod
    def value_type(args):
        return bool

@builtin("mesh_eval_position")
class MeshEvalPosition:
    @staticmethod
    def value_type(args):
        return vec3

@builtin("mesh_eval_velocity")
class MeshEvalVelocity:
    @staticmethod
    def value_type(args):
        return vec3

# helpers

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


def rename(name, return_type):
    def func(cls):
        cls.__name__ = name
        cls.key = name
        cls.prefix = ""
        cls.return_type = return_type
        return cls

    return func

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
                self.dll = None

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

                if (enable_cuda):
                    self.cuda = warp.build.load_cuda(ptx_path)

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

        if (enable_cuda):
            cu_file = open(cu_path, "w")
            cu_file.write(cu_source)
            cu_file.close()
       
        try:
                       
            if (enable_cuda):
                warp.build.build_dll(cpp_path, None, dll_path, config=warp.config.mode)

            if (enable_cpu):
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

        # load core
        self.core = warp.build.load_dll(os.path.dirname(os.path.realpath(__file__)) + "/bin/warp.dll")

        # setup c-types for core.dll
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
        print("   Using CPU compiler: {}".format(warp.config.host_compiler))

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

            with ScopedTimer("Module {} load".format(kernel.module.name)):
                kernel.module.load()

        # first param is the number of threads
        params = []
        params.append(c_long(dim))

        # converts arguments to expected types and packs into params to passed to kernel def.
        def pack_args(args, params):

            for i, a in enumerate(args):

                arg_type = kernel.adj.args[i].type

                if (isinstance(arg_type, warp.types.array)):

                    if (a == None):
                        
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
                        raise RuntimeError("Kernel parameter {} has incorrect value length {}, expected {}".format(kernel.adj.args[i].label, len(v), arg_type.length()))

                    # try and convert numpy array to builtin numeric type vec3, vec4, mat33, etc
                    x = arg_type()
                    for i in range(arg_type.length()):
                        x.value[i] = v[i]

                    params.append(x)

                else:
                    raise RuntimeError("Unknown parameter type {} for param {}, expected {}".format(type(a), kernel.adj.args[i].label, arg_type))

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


def print_builtins():

    s = ""
    for items in builtin_functions.items():
        s += str(items[0]) + ", "

    print(s)

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


