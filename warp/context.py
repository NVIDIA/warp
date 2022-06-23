# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import sys
import inspect
import hashlib
import ctypes

from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from typing import Callable

import warp
import warp.utils
import warp.codegen
import warp.build
import warp.config

import numpy as np

# represents either a built-in or user-defined function
class Function:

    def __init__(self,
                 func,
                 key,
                 namespace,
                 input_types=None,
                 value_func=None,
                 module=None,
                 variadic=False,
                 export=False,
                 doc="",
                 group="",
                 hidden=False,
                 skip_replay=False):
        
        self.func = func   # points to Python function decorated with @wp.func, may be None for builtins
        self.key = key
        self.namespace = namespace
        self.value_func = value_func    # a function that takes a list of args and returns the value type, e.g.: load(array, index) returns the type of value being loaded
        self.input_types = {}
        self.export = export
        self.doc = doc
        self.group = group
        self.module = module
        self.variadic = variadic        # function can take arbitrary number of inputs, e.g.: printf()
        self.hidden = hidden            # function will not be listed in docs
        self.skip_replay = skip_replay  # whether or not operation will be performed during the forward replay in the backward pass

        self.overloads = [self]

        if func:

            # user defined (Python) function
            self.adj = warp.codegen.Adjoint(func)

            # record input types    
            self.input_types = {}
            for name, type in self.adj.arg_types.items():
                
                if name == "return":
                    def value_func(args):
                        return type
                    self.value_func = value_func
                
                else:
                    self.input_types[name] = type

        else:

            # builtin (native) function, canonicalize argument types
            for k, v in input_types.items():
                self.input_types[k] = warp.types.type_to_warp(v)

            # cache mangled name
            if self.is_simple():
                self.mangled_name = self.mangle()
            else:
                self.mangled_name = None

        # add to current module
        if module:
            module.register_function(self)
            

    def __call__(self, *args, **kwargs):
        # handles calling a builtin (native) function
        # as if it was a Python function, i.e.: from
        # within the CPython interpreter rather than 
        # from within a kernel (experimental).

        if self.is_builtin() and self.mangled_name:

            # store last error during overload resolution
            error = None

            for f in self.overloads:
                    
                # try and find builtin in the warp.dll            
                if hasattr(warp.context.runtime.core, f.mangled_name) == False:
                    raise RuntimeError(f"Couldn't find function {self.key} with mangled name {self.mangled_name} in the Warp native library")
                
                try:
                    # try and pack args into what the function expects
                    params = []
                    for i, (arg_name, arg_type) in enumerate(f.input_types.items()):

                        a = args[i]

                        # try to convert to a value type (vec3, mat33, etc)
                        if issubclass(arg_type, ctypes.Array):

                            # force conversion to ndarray first (handles tuple / list, Gf.Vec3 case)
                            a = np.array(a)

                            # flatten to 1D array
                            v = a.flatten()
                            if (len(v) != arg_type._length_):
                                raise RuntimeError(f"Error calling function '{f.key}', parameter for argument '{arg_name}' has length {len(v)}, but expected {arg_type._length_}. Could not convert parameter to {arg_type}.")

                            # wrap the arg_type (which is an ctypes.Array) in a structure
                            # to ensure parameter is passed to the .dll by value rather than reference
                            class ValueArg(ctypes.Structure):
                                _fields_ = [ ('value', arg_type)]

                            x = ValueArg()
                            for i in range(arg_type._length_):
                                x.value[i] = v[i]

                            params.append(x)

                        else:
                            try:
                                # try to pack as a scalar type
                                params.append(arg_type._type_(a))
                            except:
                                raise RuntimeError(f"Error calling function {f.key}, unable to pack function parameter type {type(a)} for param {arg_name}, expected {arg_type}")

                    # returns the corresponding ctype for a scalar or vector warp type
                    def type_ctype(dtype):

                        if dtype == float:
                            return ctypes.c_float
                        elif dtype == int:
                            return ctypes.c_int32
                        elif issubclass(dtype, ctypes.Array):
                            return dtype
                        elif issubclass(dtype, ctypes.Structure):
                            return dtype
                        else:
                            # scalar type
                            return dtype._type_

                    value_type = type_ctype(f.value_func(None))

                    # construct return value (passed by address)
                    ret = value_type()
                    ret_addr = ctypes.c_void_p(ctypes.addressof(ret))

                    params.append(ret_addr)

                    c_func = getattr(warp.context.runtime.core, f.mangled_name)
                    c_func(*params)

                    if issubclass(value_type, ctypes.Array) or issubclass(value_type, ctypes.Structure):
                        # return vector types as ctypes 
                        return ret
                    else:
                        # return scalar types as int/float
                        return ret.value
                
                except Exception as e:
                    # couldn't pack values to match this overload
                    # store error and move onto the next one
                    error = e
                    continue

            # overload resolution or call failed
            # raise the last exception encountered
            raise error

        else:
            raise RuntimeError(f"Error, functions decorated with @wp.func can only be called from within Warp kernels (trying to call {self.key}())")
       

    def is_builtin(self):
        return self.func == None

    def is_simple(self):
        
        if self.variadic:
           return False

        # only export simple types that don't use arrays
        for k, v in self.input_types.items():
            if isinstance(v, warp.array) or v == Any or v == Callable or v == Tuple:
                return False

        return_type = ""

        try:
            # todo: construct a default value for each of the functions args
            # so we can generate the return type for overloaded functions
            return_type = type_str(self.value_func(None))
        except:
            return False

        if return_type.startswith("Tuple"):
            return False

        return True

    def mangle(self):       
        # builds a mangled name for the C-exported 
        # function, e.g.: builtin_normalize_vec3()

        name = "builtin_" + self.key
        
        types = []
        for t in self.input_types.values():
            types.append(t.__name__)

        return "_".join([name, *types])

    def add_overload(self, f):
        self.overloads.append(f)

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
                self.forward_cpu = eval("dll." + self.key + "_cpu_forward")
                self.backward_cpu = eval("dll." + self.key + "_cpu_backward")
            except:
                print(f"Could not load CPU methods for kernel {self.key}")

        if (cuda):

            try:
                self.forward_cuda = runtime.core.cuda_get_kernel(self.module.cuda, (self.key + "_cuda_kernel_forward").encode('utf-8'))
                self.backward_cuda = runtime.core.cuda_get_kernel(self.module.cuda, (self.key + "_cuda_kernel_backward").encode('utf-8'))
            except:
                print(f"Could not load CUDA methods for kernel {self.key}")


#----------------------

# decorator to register function, @func
def func(f):

    m = get_module(f.__module__)
    func = Function(func=f, key=f.__name__, namespace="", module=m, value_func=None)   # value_type not known yet, will be inferred during Adjoint.build()

    # if the function already exists in the module
    # then add an overload and return original
    for x in m.functions:
        if x.key == f.__name__:
            x.add_overload(func)
            return x

    return func

# decorator to register kernel, @kernel, custom_name may be a string
# that creates a kernel with a different name from the actual function
def kernel(f):
    
    m = get_module(f.__module__)
    k = Kernel(func=f, key=f.__name__, module=m)

    return k


builtin_functions = {}


def add_builtin(key, input_types={}, value_type=None, value_func=None, doc="", namespace="wp::", variadic=False, export=True, group="Other", hidden=False, skip_replay=False):

    # wrap simple single-type functions with a value_func()
    if value_func == None:
        def value_func(args):
            return value_type
   
    func = Function(func=None,
                    key=key,
                    namespace=namespace,
                    input_types=input_types,
                    value_func=value_func,
                    variadic=variadic,
                    export=export,
                    doc=doc,
                    group=group,
                    hidden=hidden,
                    skip_replay=skip_replay)

    if key in builtin_functions:
        builtin_functions[key].add_overload(func)
    else:
        builtin_functions[key] = func

        if export == True:
            
            if hasattr(warp, key):
                
                # check that we haven't already created something at this location
                # if it's just an overload stub for auto-complete then overwrite it
                if getattr(warp, key).__name__ != "_overload_dummy":
                    raise RuntimeError(f"Trying to register builtin function '{key}' that would overwrite existing object.")

            setattr(warp, key, func)
            

# global dictionary of modules
user_modules = {}

def get_module(m):

    if (m not in user_modules):
        user_modules[m] = warp.context.Module(str(m))

    return user_modules[m]


class ModuleBuilder:

    def __init__(self, module, options):
            
        self.functions = {}
        self.options = options
        self.module = module

        # build all functions declared in the module
        for func in module.functions:
            self.build_function(func)

        # build all kernel entry points
        for kernel in module.kernels:
            self.build_kernel(kernel)


    def build_kernel(self, kernel):
        kernel.adj.build(self)

    def build_function(self, func):

        if func in self.functions:
            return
        else:

            func.adj.build(self)

            # complete the function return type after we have analyzed it (inferred from return statement in ast)
            if not func.value_func:

                def wrap(adj):
                    def value_type(args):
                        if (adj.return_var):
                            return adj.return_var.type
                        else:
                            return None

                    return value_type

                func.value_func = wrap(func.adj)

            # use dict to preserve import order
            self.functions[func] = None

    def codegen_cpu(self):

        cpp_source = ""
        cu_source = ""

        # code-gen all imported functions
        for func in self.functions.keys():
            cpp_source += warp.codegen.codegen_func(func.adj, device="cpu")

        for kernel in self.module.kernels:

            # each kernel gets an entry point in the module
            cpp_source += warp.codegen.codegen_kernel(kernel, device="cpu")
            cpp_source += warp.codegen.codegen_module(kernel, device="cpu")

        # add headers
        cpp_source = warp.codegen.cpu_module_header + cpp_source

        return cpp_source

    def codegen_cuda(self):

        cu_source = ""

        # code-gen all imported functions
        for func in self.functions.keys():
            cu_source += warp.codegen.codegen_func(func.adj, device="cuda") 

        for kernel in self.module.kernels:

            cu_source += warp.codegen.codegen_kernel(kernel, device="cuda")
            cu_source += warp.codegen.codegen_module(kernel, device="cuda")

        # add headers
        cu_source = warp.codegen.cuda_module_header + cu_source

        return cu_source

#-----------------------------------------------------
# stores all functions and kernels for a Python module
# creates a hash of the function to use for checking
# build cache

class Module:

    def __init__(self, name):

        self.name = name

        self.kernels = []
        self.functions = []
        self.constants = []

        self.dll = None
        self.cuda = None

        self.build_failed = False

        self.options = {"max_unroll": 16,
                        "mode": warp.config.mode}

    def register_kernel(self, kernel):

        if kernel.key in self.kernels:
            
            # if kernel is replacing an old one then assume it has changed and 
            # force a rebuild / reload of the dynamic library 
            if (self.dll):
                warp.build.unload_dll(self.dll)

            if (self.cuda):
                runtime.core.cuda_unload_module(self.cuda)
                
            self.dll = None
            self.cuda = None

        # register new kernel
        self.kernels.append(kernel)


    def register_function(self, func):
        self.functions.append(func)


    def hash_module(self):
        
        h = hashlib.sha256()

        # functions source
        for func in self.functions:
            s = func.adj.source
            h.update(bytes(s, 'utf-8'))
            
        # kernel source
        for kernel in self.kernels:
            s = kernel.adj.source
            h.update(bytes(s, 'utf-8'))

         # configuration parameters
        for k in sorted(self.options.keys()):
            s = f"{k}={self.options[k]}"
            h.update(bytes(s, 'utf-8'))
       
        # # compile-time constants (global)
        if warp.types.constant._hash:
            h.update(warp.constant._hash.digest())

        return h.digest()

    def load(self, device):

        # early out to avoid repeatedly attemping to rebuild
        if self.build_failed:
            return False

        build_cpu = False
        build_cuda = False

        if device == "cpu":
            if self.dll:
                return True
            else:
                build_cpu = warp.is_cpu_available()

        # early out if cuda already loaded
        if device == "cuda":
            if self.cuda:
                return True
            else:
                build_cuda = warp.is_cuda_available()


        with warp.utils.ScopedTimer(f"Module {self.name} load"):

            module_name = "wp_" + self.name

            build_path = warp.build.kernel_bin_dir
            gen_path = warp.build.kernel_gen_dir

            cpu_hash_path = os.path.join(build_path, module_name + ".cpu.hash")
            cuda_hash_path = os.path.join(build_path, module_name + ".cuda.hash")

            module_path = os.path.join(build_path, module_name)

            ptx_path = module_path + ".ptx"

            if (os.name == 'nt'):
                dll_path = module_path + ".dll"
            else:
                dll_path = module_path + ".so"

            if (os.path.exists(build_path) == False):
                os.mkdir(build_path)

            # test cache
            module_hash = self.hash_module()


            # check CPU cache
            if build_cpu and warp.config.cache_kernels and os.path.exists(cpu_hash_path):

                f = open(cpu_hash_path, 'rb')
                cache_hash = f.read()
                f.close()

                if cache_hash == module_hash:
                    if os.path.isfile(dll_path):
                        self.dll = warp.build.load_dll(dll_path)
                        if self.dll is not None:
                            return True

            # check GPU cache
            if build_cuda and warp.config.cache_kernels and os.path.exists(cuda_hash_path):

                f = open(cuda_hash_path, 'rb')
                cache_hash = f.read()
                f.close()

                if cache_hash == module_hash:
                    if os.path.isfile(ptx_path):
                        self.cuda = warp.build.load_cuda(ptx_path)
                        if self.cuda is not None:
                            return True


            if warp.config.verbose:
                print("Warp: Rebuilding kernels for module {}".format(self.name))

            builder = ModuleBuilder(self, self.options)
            
            cpp_path = os.path.join(gen_path, module_name + ".cpp")
            cu_path = os.path.join(gen_path, module_name + ".cu")

            # write cpp sources
            if build_cpu:
                cpp_source = builder.codegen_cpu()

                cpp_file = open(cpp_path, "w")
                cpp_file.write(cpp_source)
                cpp_file.close()

            # write cuda sources
            if build_cuda:
                cu_source = builder.codegen_cuda()
                
                cu_file = open(cu_path, "w")
                cu_file.write(cu_source)
                cu_file.close()
        
            try:
                if build_cpu:
                    with warp.utils.ScopedTimer("Compile x86", active=warp.config.verbose):
                        warp.build.build_dll(cpp_path, None, dll_path, config=self.options["mode"])

                if build_cuda:
                    with warp.utils.ScopedTimer("Compile CUDA", active=warp.config.verbose):
                        warp.build.build_cuda(cu_path, ptx_path, config=self.options["mode"])

                # update cpu hash
                if build_cpu:
                    f = open(cpu_hash_path, 'wb')
                    f.write(module_hash)
                    f.close()

                # update cuda hash
                if build_cuda:
                    f = open(cuda_hash_path, 'wb')
                    f.write(module_hash)
                    f.close()

            except Exception as e:
                self.build_failed = True
                print(e)
                raise(e)

            if build_cpu:
                self.dll = warp.build.load_dll(dll_path)
                if self.dll is None:
                    raise Exception("Failed to load CPU module")


            if build_cuda:
                self.cuda = warp.build.load_cuda(ptx_path)
                if self.cuda is None:
                    raise Exception("Failed to load CUDA module")

            return True

#-------------------------------------------
# execution context

# a simple pooled allocator that caches allocs based 
# on size to avoid hitting the system allocator
class Allocator:

    def __init__(self, alloc_func, free_func):

        # map from sizes to array of allocs
        self.alloc_func = alloc_func
        self.free_func = free_func

        # map from size->list[allocs]
        self.pool = {}

    def __del__(self):
        self.clear()       

    def alloc(self, size_in_bytes):
        
        p = self.alloc_func(size_in_bytes)
        return p

        # if size_in_bytes in self.pool and len(self.pool[size_in_bytes]) > 0:            
        #     return self.pool[size_in_bytes].pop()
        # else:
        #     return self.alloc_func(size_in_bytes)

    def free(self, addr, size_in_bytes):

        addr = ctypes.cast(addr, ctypes.c_void_p)
        self.free_func(addr)
        return

        # if size_in_bytes not in self.pool:
        #     self.pool[size_in_bytes] = [addr,]
        # else:
        #     self.pool[size_in_bytes].append(addr)

    def print(self):
        
        total_size = 0

        for k,v in self.pool.items():
            print(f"alloc size: {k} num allocs: {len(v)}")
            total_size += k*len(v)

        print(f"total size: {total_size}")


    def clear(self):
        for s in self.pool.values():
            for a in s:
                self.free_func(a)
        
        self.pool = {}

class Runtime:

    def __init__(self):


        bin_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin")
        
        if (os.name == 'nt'):

            if (sys.version_info[0] > 3 or
                sys.version_info[0] == 3 and sys.version_info[1] >= 8):
                
                # Python >= 3.8 this method to add dll search paths
                os.add_dll_directory(bin_path)

            else:
                # Python < 3.8 we add dll directory to path
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]


            warp_lib = "warp.dll"
            self.core = warp.build.load_dll(os.path.join(bin_path, warp_lib))

        elif sys.platform == "darwin":

            warp_lib = os.path.join(bin_path, "warp.dylib")
            self.core = warp.build.load_dll(warp_lib)

        else:

            warp_lib = os.path.join(bin_path, "warp.so")
            self.core = warp.build.load_dll(warp_lib)

        # setup c-types for warp.dll
        self.core.alloc_host.restype = ctypes.c_void_p
        self.core.alloc_device.restype = ctypes.c_void_p
        
        self.core.mesh_create_host.restype = ctypes.c_uint64
        self.core.mesh_create_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

        self.core.mesh_create_device.restype = ctypes.c_uint64
        self.core.mesh_create_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

        self.core.mesh_destroy_host.argtypes = [ctypes.c_uint64]
        self.core.mesh_destroy_device.argtypes = [ctypes.c_uint64]

        self.core.mesh_refit_host.argtypes = [ctypes.c_uint64]
        self.core.mesh_refit_device.argtypes = [ctypes.c_uint64]

        self.core.hash_grid_create_host.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.core.hash_grid_create_host.restype = ctypes.c_uint64
        self.core.hash_grid_destroy_host.argtypes = [ctypes.c_uint64]
        self.core.hash_grid_update_host.argtypes = [ctypes.c_uint64, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        self.core.hash_grid_reserve_host.argtypes = [ctypes.c_uint64, ctypes.c_int]

        self.core.hash_grid_create_device.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.core.hash_grid_create_device.restype = ctypes.c_uint64
        self.core.hash_grid_destroy_device.argtypes = [ctypes.c_uint64]
        self.core.hash_grid_update_device.argtypes = [ctypes.c_uint64, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        self.core.hash_grid_reserve_device.argtypes = [ctypes.c_uint64, ctypes.c_int]

        self.core.volume_create_host.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.core.volume_create_host.restype = ctypes.c_uint64
        self.core.volume_get_buffer_info_host.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64)]
        self.core.volume_destroy_host.argtypes = [ctypes.c_uint64]

        self.core.volume_create_device.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.core.volume_create_device.restype = ctypes.c_uint64
        self.core.volume_get_buffer_info_device.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64)]
        self.core.volume_destroy_device.argtypes = [ctypes.c_uint64]

        # load CUDA entry points on supported platforms
        self.core.cuda_check_device.restype = ctypes.c_uint64
        self.core.cuda_get_context.restype = ctypes.c_void_p
        self.core.cuda_get_stream.restype = ctypes.c_void_p
        self.core.cuda_graph_end_capture.restype = ctypes.c_void_p
        self.core.cuda_get_device_name.restype = ctypes.c_char_p

        self.core.cuda_compile_program.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_char_p]
        self.core.cuda_compile_program.restype = ctypes.c_size_t

        self.core.cuda_load_module.argtypes = [ctypes.c_char_p]
        self.core.cuda_load_module.restype = ctypes.c_void_p

        self.core.cuda_unload_module.argtypes = [ctypes.c_void_p]

        self.core.cuda_get_kernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.core.cuda_get_kernel.restype = ctypes.c_void_p
        
        self.core.cuda_launch_kernel.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.core.cuda_launch_kernel.restype = ctypes.c_size_t

        self.core.init.restype = ctypes.c_int
        
        error = self.core.init()

        if (error > 0):
            raise Exception("Warp Initialization failed, CUDA not found")

        # allocation functions, these are function local to 
        # force other classes to go through the allocator objects
        def alloc_host(num_bytes):
            ptr = self.core.alloc_host(num_bytes)       
            return ptr

        def free_host(ptr):
            self.core.free_host(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int)))

        def alloc_device(num_bytes):
            ptr = self.core.alloc_device(ctypes.c_size_t(num_bytes))
            return ptr

        def free_device(ptr):
            # must be careful to not call any globals here
            # since this may be called during destruction / shutdown
            # even ctypes module may no longer exist
            self.core.free_device(ptr)

        self.host_allocator = Allocator(alloc_host, free_host)
        self.device_allocator = Allocator(alloc_device, free_device)

        # save context
        self.cuda_device = self.core.cuda_get_context()
        self.cuda_stream = self.core.cuda_get_stream()

        # initialize kernel cache        
        warp.build.init_kernel_cache(warp.config.kernel_cache_dir)

        # print device and version information
        print("Warp initialized:")
        print("   Version: {}".format(warp.config.version))
        print("   CUDA device: {}".format(self.core.cuda_get_device_name().decode()))
        print("   Kernel cache: {}".format(warp.config.kernel_cache_dir))

        # global tape
        self.tape = None


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
    
    # initialize host build env (do this lazily) since
    # it takes 5secs to run all the batch files to locate MSVC
    if warp.config.host_compiler == None:
        warp.config.host_compiler = warp.build.find_host_compiler()

    return warp.config.host_compiler != ""

def is_cuda_available():
    return runtime.cuda_device != None

def is_device_available(device):
    return device in get_devices()

def get_devices():
    """Returns a list of device strings supported in this environment.
    """
    devices = []
    if (is_cpu_available()):
        devices.append("cpu")
    if (is_cuda_available()):
        devices.append("cuda")

    return devices

def get_preferred_device():
    """Returns the preferred compute device, will return "cuda" if available, "cpu" otherwise.
    """
    if is_cuda_available():
        return "cuda"
    elif is_cpu_available():
        return "cpu"
    else:
        return None


def zeros(shape: Tuple=None, dtype=float, device: str="cpu", requires_grad: bool=False, **kwargs)-> warp.array:
    """Return a zero-initialized array

    Args:
        shape: Array dimensions
        dtype: Type of each element, e.g.: warp.vec3, warp.mat33, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation

    Returns:
        A warp.array object representing the allocation                
    """

    if runtime == None:
        raise RuntimeError("Warp not initialized, call wp.init() before use")

    if device != "cpu" and device != "cuda":
        raise RuntimeError(f"Trying to allocate array on unknown device {device}")

    if device == "cuda" and not is_cuda_available():
        raise RuntimeError("Trying to allocate CUDA buffer without GPU support")


    # backwards compatability for case where users did wp.zeros(n, dtype=..), or wp.zeros(n=length, dtype=..)
    if isinstance(shape, int):
        shape = (shape,)
    elif "n" in kwargs:
        shape = (kwargs["n"], )
    
    # compute num els
    num_elements = 1
    for d in shape:
        num_elements *= d

    num_bytes = num_elements*warp.types.type_size_in_bytes(dtype)

    if device == "cpu":
        ptr = runtime.host_allocator.alloc(num_bytes) 
        runtime.core.memset_host(ctypes.cast(ptr,ctypes.POINTER(ctypes.c_int)), ctypes.c_int(0), ctypes.c_size_t(num_bytes))

    elif device == "cuda":
        ptr = runtime.device_allocator.alloc(num_bytes)
        runtime.core.memset_device(ctypes.cast(ptr,ctypes.POINTER(ctypes.c_int)), ctypes.c_int(0), ctypes.c_size_t(num_bytes))

    if (ptr == None and num_bytes > 0):
        raise RuntimeError("Memory allocation failed on device: {} for {} bytes".format(device, num_bytes))
    else:
        # construct array
        return warp.types.array(dtype=dtype, shape=shape, capacity=num_bytes, ptr=ptr, device=device, owner=True, requires_grad=requires_grad)

def zeros_like(src: warp.array) -> warp.array:
    """Return a zero-initialized array with the same type and dimension of another array

    Args:
        src: The template array to use for length, data type, and device

    Returns:
        A warp.array object representing the allocation
    """

    arr = zeros(shape=src.shape, dtype=src.dtype, device=src.device, requires_grad=src.requires_grad)
    return arr

def clone(src: warp.array) -> warp.array:
    """Clone an existing array, allocates a copy of the src memory

    Args:
        src: The source array to copy

    Returns:
        A warp.array object representing the allocation
    """

    dest = empty(len(src), dtype=src.dtype, device=src.device, requires_grad=src.requires_grad)
    copy(dest, src)

    return dest

def empty(shape: Tuple=None, dtype=float, device:str="cpu", requires_grad:bool=False, **kwargs) -> warp.array:
    """Returns an uninitialized array

    Args:
        n: Number of elements
        dtype: Type of each element, e.g.: `warp.vec3`, `warp.mat33`, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation

    Returns:
        A warp.array object representing the allocation
    """

    # todo: implement uninitialized allocation
    return zeros(shape, dtype, device, requires_grad=requires_grad, **kwargs)  

def empty_like(src: warp.array, requires_grad:bool=False) -> warp.array:
    """Return an uninitialized array with the same type and dimension of another array

    Args:
        src: The template array to use for length, data type, and device
        requires_grad: Whether the array will be tracked for back propagation

    Returns:
        A warp.array object representing the allocation
    """
    arr = empty(shape=src.shape, dtype=src.dtype, device=src.device, requires_grad=requires_grad)
    return arr


def from_numpy(arr, dtype, device="cpu", requires_grad=False):

    return warp.array(data=arr, dtype=dtype, device=device, requires_grad=requires_grad)


def launch(kernel, dim: Tuple[int], inputs:List, outputs:List=[], adj_inputs:List=[], adj_outputs:List=[], device:str="cpu", adjoint=False):
    """Launch a Warp kernel on the target device

    Kernel launches are asynchronous with respect to the calling Python thread. 

    Args:
        kernel: The name of a Warp kernel function, decorated with the ``@wp.kernel`` decorator
        dim: The number of threads to launch the kernel, can be an integer, or a Tuple of ints with max of 4 dimensions
        inputs: The input parameters to the kernel
        outputs: The output parameters (optional)
        adj_inputs: The adjoint inputs (optional)
        adj_outputs: The adjoint outputs (optional)
        device: The device to launch on
        adjoint: Whether to run forward or backward pass (typically use False)
    """

    

    # check device available
    if is_device_available(device) == False:
        raise RuntimeError(f"Error launching kernel, device '{device}' is not available.")

    # check function is a Kernel
    if isinstance(kernel, Kernel) == False:
        raise RuntimeError("Error launching kernel, can only launch functions decorated with @wp.kernel.")

    # debugging aid
    if (warp.config.print_launches):
        print(f"kernel: {kernel.key} dim: {dim} inputs: {inputs} outputs: {outputs} device: {device}")

    # delay load modules
    success = kernel.module.load(device)
    if (success == False):
        return

    # construct launch bounds
    bounds = warp.types.launch_bounds_t(dim)

    if (bounds.size > 0):

        # first param is the number of threads
        params = []
        params.append(bounds)

        # converts arguments to kernel's expected ctypes and packs into params
        def pack_args(args, params):

            for i, a in enumerate(args):

                arg_type = kernel.adj.args[i].type
                arg_name = kernel.adj.args[i].label

                if (isinstance(arg_type, warp.types.array)):

                    if (a is None):
                        
                        # allow for NULL arrays
                        params.append(warp.types.array_t())

                    else:

                        # check for array value
                        if (isinstance(a, warp.types.array) == False):
                            raise RuntimeError(f"Error launching kernel '{kernel.key}', argument '{arg_name}' expects an array, but passed value has type {type(a)}.")
                        
                        # check subtype
                        if (a.dtype != arg_type.dtype):
                            raise RuntimeError(f"Error launching kernel '{kernel.key}', argument '{arg_name}' expects an array with dtype={arg_type.dtype} but passed array has dtype={a.dtype}.")

                        # check dimensions
                        if (a.ndim != arg_type.ndim):
                            raise RuntimeError(f"Error launching kernel '{kernel.key}', argument '{arg_name}' expects an array with {arg_type.ndim} dimension(s) but the passed array has {a.ndim} dimension(s).")

                        # check device
                        if (a.device != device):
                            raise RuntimeError(f"Error launching kernel '{kernel.key}', trying to launch on device='{device}', but input array for argument '{arg_name}' is on device={a.device}.")
                        
                        params.append(a.__ctype__())

                # try to convert to a value type (vec3, mat33, etc)
                elif issubclass(arg_type, ctypes.Array):

                    # force conversion to ndarray first (handles tuple / list, Gf.Vec3 case)
                    a = np.array(a)

                    # flatten to 1D array
                    v = a.flatten()
                    if (len(v) != arg_type._length_):
                        raise RuntimeError(f"Error launching kernel '{kernel.key}', parameter for argument '{arg_name}' has length {len(v)}, but expected {arg_type._length_}. Could not convert parameter to {arg_type}.")

                    # wrap the arg_type (which is an ctypes.Array) in a structure
                    # to ensure parameter is passed to the .dll by value rather than reference
                    class ValueArg(ctypes.Structure):
                        _fields_ = [ ('value', arg_type)]

                    x = ValueArg()
                    for i in range(arg_type._length_):
                        x.value[i] = v[i]

                    params.append(x)

                else:
                    try:
                        # try to pack as a scalar type
                        params.append(arg_type._type_(a))
                    except:
                        raise RuntimeError(f"Error launching kernel, unable to pack kernel parameter type {type(a)} for param {arg_name}, expected {arg_type}")


        fwd_args = inputs + outputs
        adj_args = adj_inputs + adj_outputs

        if (len(fwd_args)) != (len(kernel.adj.args)): 
            raise RuntimeError(f"Error launching kernel '{kernel.key}', passed {len(fwd_args)} arguments but kernel requires {len(kernel.adj.args)}.")

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

        
        elif device == "cuda":

            kernel_args = [ctypes.c_void_p(ctypes.addressof(x)) for x in params]
            kernel_params = (ctypes.c_void_p * len(kernel_args))(*kernel_args)

            if (adjoint):
                runtime.core.cuda_launch_kernel(kernel.backward_cuda, bounds.size, kernel_params)
            else:
                runtime.core.cuda_launch_kernel(kernel.forward_cuda, bounds.size, kernel_params)

            try:
                runtime.verify_device()            
            except Exception as e:
                print(f"Error launching kernel: {kernel.key} on device {device}")
                raise e

    # record on tape if one is active
    if (runtime.tape):
        runtime.tape.record(kernel, dim, inputs, outputs, device)

def synchronize():
    """Manually synchronize the calling CPU thread with any outstanding CUDA work

    This method allows the host application code to ensure that any kernel launches
    or memory copies have completed.
    """

    runtime.core.synchronize()

    runtime.verify_device()


def force_load():
    """Force all user-defined kernels to be compiled
    """

    devices = get_devices()

    for d in devices:
        for m in user_modules.values():
            m.load(d)

def set_module_options(options: Dict[str, Any]):
    """Set options for the current module.

    Options can be used to control runtime compilation and code-generation
    for the current module individually. Available options are listed below.

    * **mode**: The compilation mode to use, can be "debug", or "release", defaults to the value of ``warp.config.mode``.
    * **max_unroll**: The maximum fixed-size loop to unroll (default 16)

    Args:

        options: Set of key-value option pairs
    """
   
    import inspect
    m = inspect.getmodule(inspect.stack()[1][0])

    get_module(m.__name__).options.update(options)

def get_module_options() -> Dict[str, Any]:
    """Returns a list of options for the current module.
    """
    import inspect
    m = inspect.getmodule(inspect.stack()[1][0])

    return get_module(m.__name__).options


def capture_begin():
    """Begin capture of a CUDA graph

    Captures all subsequent kernel launches and memory operations on CUDA devices.
    This can be used to record large numbers of kernels and replay them with low-overhead.
    """

    if warp.config.verify_cuda == True:
        raise RuntimeError("Cannot use CUDA error verification during graph capture")

    # ensure that all modules are loaded, this is necessary
    # since cuLoadModule() is not permitted during capture
    for m in user_modules.values():
        m.load("cuda")    

    runtime.core.cuda_graph_begin_capture()

def capture_end()->int:
    """Ends the capture of a CUDA graph

    Returns:
        A handle to a CUDA graph object that can be launched with :func:`~warp.capture_launch()`
    """

    graph = runtime.core.cuda_graph_end_capture()
    
    if graph == None:
        raise RuntimeError("Error occurred during CUDA graph capture. This could be due to an unintended allocation or CPU/GPU synchronization event.")
    else:
        return graph

def capture_launch(graph: int):
    """Launch a previously captured CUDA graph

    Args:
        graph: A handle to the graph as returned by :func:`~warp.capture_end`
    """

    runtime.core.cuda_graph_launch(ctypes.c_void_p(graph))


def copy(dest: warp.array, src: warp.array, dest_offset: int = 0, src_offset: int = 0, count: int = 0):
    """Copy array contents from src to dest

    Args:
        dest: Destination array, must be at least as big as source buffer
        src: Source array
        dest_offset: Element offset in the destination array
        src_offset: Element offset in the source array
        count: Number of array elements to copy (will copy all elements if set to 0)

    """

    # backwards compatability, if count is zero then copy entire src array
    if count <= 0:
        count = src.size

    if count == 0:
        return

    bytes_to_copy = count * warp.types.type_size_in_bytes(src.dtype)

    src_size_in_bytes = src.size * warp.types.type_size_in_bytes(src.dtype)
    dst_size_in_bytes = dest.size * warp.types.type_size_in_bytes(dest.dtype)

    src_offset_in_bytes = src_offset * warp.types.type_size_in_bytes(src.dtype)
    dst_offset_in_bytes = dest_offset * warp.types.type_size_in_bytes(dest.dtype)

    src_ptr = src.ptr + src_offset_in_bytes
    dst_ptr = dest.ptr + dst_offset_in_bytes

    if src_offset_in_bytes + bytes_to_copy > src_size_in_bytes:
        raise RuntimeError(f"Trying to copy source buffer with size ({bytes_to_copy}) from offset ({src_offset_in_bytes}) is larger than source size ({src_size_in_bytes})")

    if dst_offset_in_bytes + bytes_to_copy > dst_size_in_bytes:
        raise RuntimeError(f"Trying to copy source buffer with size ({bytes_to_copy}) to offset ({dst_offset_in_bytes}) is larger than destination size ({dst_size_in_bytes})")

    if (src.device == "cpu" and dest.device == "cuda"):
        runtime.core.memcpy_h2d(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr), ctypes.c_size_t(bytes_to_copy))

    elif (src.device == "cuda" and dest.device == "cpu"):
        runtime.core.memcpy_d2h(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr), ctypes.c_size_t(bytes_to_copy))

    elif (src.device == "cpu" and dest.device == "cpu"):
        runtime.core.memcpy_h2h(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr), ctypes.c_size_t(bytes_to_copy))

    elif (src.device == "cuda" and dest.device == "cuda"):
        runtime.core.memcpy_d2d(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr), ctypes.c_size_t(bytes_to_copy))
    
    else:
        raise RuntimeError("Unexpected source and destination combination")


def type_str(t):
    if t == None:
        return "None"
    elif t == Any:
        return "Any"
    elif t == Callable:
        return "Callable"
    elif isinstance(t, List):
        return "Tuple[" + ", ".join(map(type_str, t)) + "]"
    elif isinstance(t, warp.array):
        return f"array[{type_str(t.dtype)}]"
    else:
        return t.__name__

def print_function(f, file):

    if f.hidden:
        return

    args = ", ".join(f"{k}: {type_str(v)}" for k,v in f.input_types.items())

    return_type = ""

    try:

        # todo: construct a default value for each of the functions args
        # so we can generate the return type for overloaded functions
        return_type = " -> " + type_str(f.value_func(None))
    except:
        pass

    print(f".. function:: {f.key}({args}){return_type}", file=file)
    print("", file=file)
    
    if (f.doc != ""):
        print(f"   {f.doc}", file=file)
        print("", file=file)

    print(file=file)
    

def print_builtins(file):

    header = ("..\n"
              "   Autogenerated File - Do not edit. Run build_docs.py to generate.\n"
              "\n"
              ".. functions:\n"
              ".. currentmodule:: warp\n"
              "\n"
              "Kernel Reference\n"
              "================")

    print(header, file=file)

    # type definitions of all functions by group
    print("Scalar Types", file=file)
    print("------------", file=file)

    for t in warp.types.scalar_types:
        print(f".. autoclass:: {t.__name__}", file=file)

    print("Vector Types", file=file)
    print("------------", file=file)

    for t in warp.types.vector_types:
        print(f".. autoclass:: {t.__name__}", file=file)


    # build dictionary of all functions by group
    groups = {}

    for k, f in builtin_functions.items():

        # build dict of groups
        if f.group not in groups:
            groups[f.group] = []
        
        # append all overloads to the group
        for o in f.overloads:
            groups[f.group].append(o)

    for k, g in groups.items():
        print("\n", file=file)
        print(k, file=file)
        print("---------------", file=file)

        for f in g:
            print_function(f, file=file)


def export_stubs(file):
    """ Generates stub file for auto-complete of builtin functions"""

    import textwrap

    print("# Autogenerated file, do not edit, this file provides stubs for builtins autocomplete in VSCode, PyCharm, etc", file=file)
    print("", file=file)
    print("from typing import Any", file=file)
    print("from typing import Tuple", file=file)
    print("from typing import Callable", file=file)
    print("from typing import overload", file=file)

    print("from warp.types import array, array2d, array3d, array4d, constant", file=file)
    print("from warp.types import int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64", file=file)
    print("from warp.types import vec2, vec3, vec4, mat22, mat33, mat44, quat, transform, spatial_vector, spatial_matrix", file=file)
    print("from warp.types import mesh_query_aabb_t, hash_grid_query_t", file=file)


    #print("from warp.types import *", file=file)
    print("\n", file=file)

    for k, g in builtin_functions.items():

        for f in g.overloads:

            args = ", ".join(f"{k}: {type_str(v)}" for k,v in f.input_types.items())

            return_str = ""

            if f.export == False or f.hidden == True:
                continue
            
            try:
                       
                # todo: construct a default value for each of the functions args
                # so we can generate the return type for overloaded functions
                return_type = f.value_func(None)
                if return_type:
                    return_str = " -> " + type_str(return_type)

            except:
                pass

            print("@overload", file=file)
            print(f"def {f.key}({args}){return_str}:", file=file)
            print(f'   """', file=file)
            print(textwrap.indent(text=f.doc, prefix="   "), file=file)
            print(f'   """', file=file)
            print(f"   ...\n", file=file)
            


def export_builtins(file):

    for k, g in builtin_functions.items():

        for f in g.overloads:

            if f.export == False:
                continue

            simple = True
            for k, v in f.input_types.items():
                if isinstance(v, warp.array) or v == Any or v == Callable or v == Tuple:
                    simple = False
                    break

            # only export simple types that don't use arrays
            # or templated types
            if not simple or f.variadic:
                continue

            args = ", ".join(f"{type_str(v)} {k}" for k,v in f.input_types.items())
            params = ", ".join(f.input_types.keys())

            return_type = ""

            try:
                # todo: construct a default value for each of the functions args
                # so we can generate the return type for overloaded functions
                return_type = type_str(f.value_func(None))
            except:
                pass

            if return_type.startswith("Tuple"):
                continue

            if args == "":
                print(f"WP_API void {f.mangled_name}({return_type}* ret) {{ *ret = wp::{f.key}({params}); }}", file=file)
            elif return_type == "None":
                print(f"WP_API void {f.mangled_name}({args}) {{ wp::{f.key}({params}); }}", file=file)
            else:
                print(f"WP_API void {f.mangled_name}({args}, {return_type}* ret) {{ *ret = wp::{f.key}({params}); }}", file=file)




# initialize global runtime
runtime = None

def init():
    """Initialize the Warp runtime. This function must be called before any other API call. If an error occurs an exception will be raised.
    """
    global runtime

    if (runtime == None):
        runtime = Runtime()

