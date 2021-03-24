import math
import os
import imp
import subprocess
import timeit
import cProfile

from ctypes import*


from oglang.types import *

import oglang.codegen


class ScopedTimer:

    indent = -1

    enabled = True

    def __init__(self, name, active=True, detailed=False):
        self.name = name
        self.active = active and self.enabled
        self.detailed = detailed

    def __enter__(self):

        if (self.active):
            self.start = timeit.default_timer()
            ScopedTimer.indent += 1

            if (self.detailed):
                self.cp = cProfile.Profile()
                self.cp.clear()
                self.cp.enable()


    def __exit__(self, exc_type, exc_value, traceback):

        if (self.detailed):
            self.cp.disable()
            self.cp.print_stats(sort='tottime')

        if (self.active):
            elapsed = (timeit.default_timer() - self.start) * 1000.0

            indent = ""
            for i in range(ScopedTimer.indent):
                indent += "\t"

            print("{}{} took {:.2f} ms".format(indent, self.name, elapsed))

            ScopedTimer.indent -= 1


# runs vcvars and copies back the build environment
def set_build_env():

    def find_vcvars_path():
        import glob
        for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
            paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Auxiliary\Build\vcvars64.bat" % edition), reverse=True)
            if paths:
                return paths[0]

    if os.name == 'nt':

        vcvars_path = find_vcvars_path()

        # merge vcvars with our env
        s = '"{}" && set'.format(vcvars_path)
        output = os.popen(s).read()
        for line in output.splitlines():
            pair = line.split("=", 1)
            if (len(pair) >= 2):
                os.environ[pair[0]] = pair[1]




# represents either a built-in or user-defined function
class Function:

    def __init__(self, func, key, namespace, module=None, value_type=None):
        
        self.func = func   # may be None for built-in funcs
        self.key = key
        self.namespace = namespace
        self.value_type = value_type
        self.module = module

        if (module):
            module.register_function(self)

# caches source and compiled entry points for a kernel (will be populated after module loads)
class Kernel:
    
    def __init__(self, func, module):

        self.func = func
        self.module = module

        if (module):
            module.register_kernel(self)

    # cache entry points based on name, called after compilation / module load
    def hook(self, dll):

        try:
            self.forward_cpu = eval("dll." + self.func.__name__ + "_cpu_forward")
            self.backward_cpu = eval("dll." + self.func.__name__ + "_cpu_backward")
        except:
            print("Could not find load CPU methods for kernel {}".format(self.func.__name__))

        try:
            self.forward_cuda = eval("dll." + self.func.__name__ + "_cuda_forward")
            self.backward_cuda = eval("dll." + self.func.__name__ + "_cuda_backward")
        except:
            print("Could not find load CUDA methods for kernel {}".format(self.func.__name__))


#----------------------

# register function, @func
def func(f):

    m = get_module(f.__module__)
    f = Function(func=f, key=f.__name__, namespace="", module=m, value_type=None)   # value_type not known yet, will be infered during code-gen

    return f

# register kernel, @kernel
def kernel(f):

    m = get_module(f.__module__)
    k = Kernel(func=f, module=m)

    return k


builtin_functions = {}

# register built-in function @builtin
def builtin(key):
    def insert(c):
        
        func = Function(func=None, key=key, namespace="og::", value_type=c.value_type)
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
        # todo: encode type operator type globally
        if (args[0].type == mat33 and args[1].type == float3):            
            return float3
        if (args[0].type == spatial_matrix and args[1].type == spatial_vector):
            return spatial_vector
        else:
            return args[0].type


@builtin("div")
class DivFunc:
    @staticmethod
    def value_type(args):
        return args[0].type


#----------------------
# built-in builtin_functions



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
        return float3

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


@builtin("rotate")
class RotateFunc:
    @staticmethod
    def value_type(args):
        return float3


@builtin("rotate_inv")
class RotateInvFunc:
    @staticmethod
    def value_type(args):
        return float3


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
        if (type(args[0].type) != oglang.types.array):
            raise Exception("Load input 0 must be a array")
        if (args[1].type != int):
            raise Exception("Load input 1 must be a int")

        return args[0].type.type


@builtin("store")
class StoreFunc:
    @staticmethod
    def value_type(args):
        if (type(args[0].type) != oglang.types.array):
            raise Exception("Store input 0 must be a array")
        if (args[1].type != int):
            raise Exception("Store input 1 must be a int")
        if (args[2].type != args[0].type.type):
            raise Exception("Store input 2 must be of the same type as the array")

        return None


@builtin("atomic_add")
class AtomicAddFunc:
    @staticmethod
    def value_type(args):
        return None


@builtin("atomic_sub")
class AtomicSubFunc:
    @staticmethod
    def value_type(args):
        return None


@builtin("tid")
class ThreadIdFunc:
    @staticmethod
    def value_type(args):
        return int


# type construtors

@builtin("float")
class floatFunc:
    @staticmethod
    def value_type(args):
        return float


@builtin("float3")
class float3Func:
    @staticmethod
    def value_type(args):
        return float3


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
        return float3

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
        return float3

@builtin("spatial_transform_vector")
class SpatialTransformVectorFunc:
    @staticmethod
    def value_type(args):
        return float3

@builtin("spatial_top")
class SpatialTopFunc:
    @staticmethod
    def value_type(args):
        return float3

@builtin("spatial_bottom")
class SpatialBottomFunc:
    @staticmethod
    def value_type(args):
        return float3

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
class DenseSolve:
    @staticmethod
    def value_type(args):
        return None        

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



def rename(name, return_type):
    def func(cls):
        cls.__name__ = name
        cls.key = name
        cls.prefix = ""
        cls.return_type = return_type
        return cls

    return func


# global dictionary of modules
user_modules = {}

def get_module(m):

    if (m not in user_modules):
        user_modules[m] = oglang.context.Module(str(m))

    return user_modules[m]


#---------------------------------------------
# stores all functions and kernels for a module

def wrap(adj):
    def value_type(args):
        if (adj.return_var):
            return adj.return_var.type
        else:
            return None

    return value_type

class Module:

    def __init__(self, name):

        self.name = name
        self.kernels = []
        self.functions = {}

        self.dll = None

    def register_kernel(self, kernel):
        self.kernels.append(kernel)

    def register_function(self, func):
        self.functions[func.key] = func

    def load(self):

        use_cuda = False
        if not use_cuda:
            print("[INFO] CUDA support not found. Disabling CUDA kernel compilation.")

        cpp_source = ""
        cu_source = ""

        cpp_source += oglang.codegen.cpu_module_header
        cu_source += oglang.codegen.cuda_module_header

        # kernels
        entry_points = []

        # functions
        for name, func in self.functions.items():

            adj = oglang.codegen.Adjoint(func.func, builtin_functions, self.functions, device='cpu')
            cpp_source += oglang.codegen.codegen_func(adj, device='cpu')

            adj = oglang.codegen.Adjoint(func.func, builtin_functions, self.functions, device='cuda')
            cu_source += oglang.codegen.codegen_func(adj, device='cuda')

            # complete the function return type after we have analyzed it (infered from return statement in ast)            
            func.value_type = wrap(adj)


            # import pdb
            # pdb.set_trace()

            # import copy

            # @rename(func.__name__ + "_cpu_func", adj.return_var.type)
            # class Func:
            #     @classmethod
            #     def value_type(cls, *args):
            #         return cls.return_type

            # functions[func.__name__] = Func

            # @rename(func.__name__ + "_cuda_func", adj.return_var.type)
            # class CUDAFunc:
            #     @classmethod
            #     def value_type(cls, *args):
            #         return cls.return_type

            # cuda_functions[func.__name__] = CUDAFunc

        for kernel in self.kernels:

            if use_cuda:
                # each kernel gets an entry point in the module
                entry_points.append(kernel.func.__name__ + "_cuda_forward")
                entry_points.append(kernel.func.__name__ + "_cuda_backward")

            # each kernel gets an entry point in the module
            entry_points.append(kernel.func.__name__ + "_cpu_forward")
            entry_points.append(kernel.func.__name__ + "_cpu_backward")

            if use_cuda:
                adj = oglang.codegen.Adjoint(kernel.func, builtin_functions, self.functions, device='cuda')
                cpp_source += oglang.codegen.codegen_module_decl(adj, device='cuda')
                cu_source += oglang.codegen.codegen_kernel(adj, device='cuda')
                cu_source += oglang.codegen.codegen_module(adj, device='cuda')

            adj = oglang.codegen.Adjoint(kernel.func, builtin_functions, self.functions, device='cpu')
            cpp_source += oglang.codegen.codegen_module_decl(adj, device='cpu')
            cpp_source += oglang.codegen.codegen_kernel(adj, device='cpu')
            cpp_source += oglang.codegen.codegen_module(adj, device='cpu')

        module_name = "og_" + self.name

        include_path = os.path.dirname(os.path.realpath(__file__))
        build_path = os.path.dirname(os.path.realpath(__file__)) + "/kernels"
        cache_path = build_path + "/" + module_name + ".gen"
        
        cpp_path = build_path + "/" + module_name + ".cpp"
        cu_path = build_path + "/" + module_name + ".ccu"
        dll_path = build_path + "/" + module_name + ".dll"

        if (os.path.exists(build_path) == False):
            os.mkdir(build_path)

        # test cache
        if (os.path.exists(cache_path)):

            f = open(cache_path, 'r')

            cache_string = f.read()
            f.close()

            if (cache_string == cpp_source):
                print("Using cached kernels")
                self.dll = cdll.LoadLibrary(dll_path)

                # register kernel methods
                for k in self.kernels:
                    k.hook(self.dll)

                return


        # write cpp sources
        cpp_file = open(cpp_path, "w")
        cpp_file.write(cpp_source)
        cpp_file.close()

        cu_file = open(cu_path, "w")
        cu_file.write(cu_source)
        cu_file.close()

        # print("ignoring rebuild, using stale kernels")
        # module = import_module("kernels", build_path)
        # return module

        # cache stale, rebuild
        print("Rebuilding kernels")

        set_build_env()

        # debug config
        #module = torch.utils.cpp_extension.load_inline('kernels', [cpp_source], None, entry_points, extra_cflags=["/Zi", "/Od"], extra_loglags=["/DEBUG"], build_directory=build_path, extra_include_paths=[include_path], verbose=True)

        if os.name == 'nt':
            cxx = "cl"
            nvcc = "nvcc"
            link = "link"

            cpp_flags = "/Ox -DNDEBUG /fp:fast"
            ld_flags = "-DNDEBUG"

    #        cpp_flags = ["/Zi", "/Od", "/DEBUG"]
    #        ld_flags = ["/DEBUG"]
        else:
            cxx = "gcc"
            nvcc = "nvcc"
            link = "gcc"

            cpp_flags = "-Z", "-O2", "-DNDEBUG"
            ld_flags = "-DNDEBUG"

        # just use minimum to ensure compatability
        cuda_flags = ['-gencode=arch=compute_35,code=compute_35']


        with ScopedTimer("build"):
            build_cmd = "cl.exe {cflags} -c {cpp_path} /Fo{cpp_path}.obj ".format(cflags=cpp_flags, cpp_path=cpp_path)
            print(build_cmd)
            subprocess.call(build_cmd)

        with ScopedTimer("link"):
            link_cmd = "link.exe {cpp_path}.obj /dll /out:{dll_path}".format(cpp_path=cpp_path, dll_path=dll_path)
            print(link_cmd)
            subprocess.call(link_cmd)

       
        self.dll = cdll.LoadLibrary(dll_path)

        # update cached output
        f = open(cache_path, 'w')
        f.write(cpp_source)
        f.close()

        # register kernel methods
        for k in self.kernels:
            k.hook(self.dll)


#-------------------------------------------
# exectution context
from ctypes import *


class Context:

    def __init__(self, device="cpu"):
        
        self.device = device

    def alloc(self, num_bytes):
        dll = CDLL('msvcrt')
        dll.malloc.restype = c_void_p
        ptr = dll.malloc(num_bytes)
        
        # always clear
        dll.memset(cast(ptr,POINTER(c_int)), 0, num_bytes)
        
        return ptr

    def free(self, ptr):
        dll = CDLL('msvcrt')
        dll.free(cast(ptr,POINTER(c_int)))

    def wrap(self, n, dtype, data_ptr):
        pass

    def empty(self, n, dtype=float):
        pass

    def zeros(self, n, dtype=float):

        # todo concrete builtin types
        ptr = self.alloc(n*4) 

        return oglang.types.array(dtype, length=n, data=ptr, context=self, owner=True)


    def empty_like(self, array):
        pass
    
    def zeros_like(self, array):
        pass


    def launch(self, kernel, dim, inputs, outputs):

        if (dim > 0):

            # delay load modules
            if (kernel.module.dll == None):

                with ScopedTimer("Module load"):
                    kernel.module.load()

            # build params
            params = [dim]

            # todo: verify argument types against the kernel definition, perform automatic conversion for simple types

            for i in inputs:
                if type(i) is oglang.types.array:
                    params.append(c_int64(i.data))
                elif type(i) is int:
                    params.append(c_int32(i))
                elif type(i) is float:
                    params.append(c_float(i))
                else:
                    # todo: add support for other built-types as kernel arguments (float3, quat, etc)
                    print("Unknown parameter type")

            # run kernel
            if self.device == 'cpu':
                kernel.forward_cpu(*params)
            elif self.device.startswith('cuda'):
                kernel.forward_cuda(*params)

        