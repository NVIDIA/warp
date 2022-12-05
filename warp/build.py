# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import sys
import imp
import subprocess
import ctypes
import _ctypes

import warp.config
import warp.utils
from warp.utils import ScopedTimer
from warp.thirdparty import appdirs

# return from funtions without type -> C++ compile error
# array[i,j] += x -> augassign not handling target of subscript




def run_cmd(cmd, capture=False):
    
    if (warp.config.verbose):
        print(cmd)

    try:
        return subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output.decode())
        raise(e)

# cut-down version of vcvars64.bat that allows using
# custom toolchain locations
def set_msvc_compiler(msvc_path, sdk_path):

    if "INCLUDE" not in os.environ:
        os.environ["INCLUDE"] = ""

    if "LIB" not in os.environ:
        os.environ["LIB"] = ""

    msvc_path = os.path.abspath(msvc_path)
    sdk_path = os.path.abspath(sdk_path)

    os.environ["INCLUDE"] += os.pathsep + os.path.join(msvc_path, "include")
    os.environ["INCLUDE"] += os.pathsep + os.path.join(sdk_path, "include/winrt")
    os.environ["INCLUDE"] += os.pathsep + os.path.join(sdk_path, "include/um")
    os.environ["INCLUDE"] += os.pathsep + os.path.join(sdk_path, "include/ucrt")
    os.environ["INCLUDE"] += os.pathsep + os.path.join(sdk_path, "include/shared")

    os.environ["LIB"] += os.pathsep + os.path.join(msvc_path, "lib/x64")
    os.environ["LIB"] += os.pathsep + os.path.join(sdk_path, "lib/ucrt/x64")
    os.environ["LIB"] += os.pathsep + os.path.join(sdk_path, "lib/um/x64")

    os.environ["PATH"] += os.pathsep + os.path.join(msvc_path, "bin/HostX64/x64")

    warp.config.host_compiler = os.path.join(msvc_path, "bin", "HostX64", "x64", "cl.exe")


def find_host_compiler():

    if (os.name == 'nt'):

        try:

            # try and find an installed host compiler (msvc)
            # runs vcvars and copies back the build environment

            vswhere_path = r"%ProgramFiles(x86)%/Microsoft Visual Studio/Installer/vswhere.exe"
            vswhere_path = os.path.expandvars(vswhere_path)
            if not os.path.exists(vswhere_path):
                return ""

            vs_path = run_cmd('"{}" -latest -property installationPath'.format(vswhere_path)).decode().rstrip()
            vsvars_path = os.path.join(vs_path, "VC\\Auxiliary\\Build\\vcvars64.bat")

            output = run_cmd('"{}" && set'.format(vsvars_path)).decode()
            
            for line in output.splitlines():
                pair = line.split("=", 1)
                if (len(pair) >= 2):
                    os.environ[pair[0]] = pair[1]
                
            cl_path = run_cmd("where cl.exe").decode("utf-8").rstrip()
            cl_version = os.environ["VCToolsVersion"].split(".")

            # ensure at least VS2017 version, see list of MSVC versions here https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B
            cl_required_major = 14
            cl_required_minor = 1

            if ((int(cl_version[0]) < cl_required_major) or
                (int(cl_version[0]) == cl_required_major) and int(cl_version[1]) < cl_required_minor):
                
                print(f"Warp: MSVC found but compiler version too old, found {cl_version[0]}.{cl_version[1]}, but must be {cl_required_major}.{cl_required_minor} or higher, kernel host compilation will be disabled.")
                return ""
                
            return cl_path
        
        except Exception as e:

            # couldn't find host compiler
            return ""
    else:
        
        # try and find g++
        try:
            return run_cmd("which g++").decode()
        except:
            return ""





# See PyTorch for reference on how to find nvcc.exe more robustly, https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension
def find_cuda():
    
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    return cuda_home
    

def get_cuda_toolkit_version(cuda_home):
    
    try:
        # the toolkit version can be obtained by running "nvcc --version"
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        nvcc_version_output = subprocess.check_output([nvcc_path, "--version"]).decode('utf-8')
        # search for release substring (e.g., "release 11.5")
        import re
        m = re.search(r"(?<=release )\d+\.\d+", nvcc_version_output)
        if m is not None:
            return tuple(int(x) for x in m.group(0).split("."))
        else:
            raise Exception("Failed to parse NVCC output")

    except Exception as e:
        print(f"Failed to determine CUDA Toolkit version: {e}")


# builds cuda source to PTX or CUBIN using NVRTC (output type determined by output_path extension)
def build_cuda(cu_path, arch, output_path, config="release", force=False, verify_fp=False, fast_math=False):

    src_file = open(cu_path)
    src = src_file.read().encode('utf-8')
    src_file.close()

    inc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "native").encode('utf-8')
    output_path = output_path.encode('utf-8')

    err = warp.context.runtime.core.cuda_compile_program(src, arch, inc_path, config=="debug", warp.config.verbose, verify_fp, fast_math, output_path)
    if (err):
        raise Exception("CUDA build failed")


# load PTX or CUBIN as a CUDA runtime module (input type determined by input_path extension)
def load_cuda(input_path, device):

    if not device.is_cuda:
        raise("Not a CUDA device")

    return warp.context.runtime.core.cuda_load_module(device.context, input_path.encode('utf-8'))


def quote(path):
    return "\"" + path + "\""

def build_dll(cpp_path, cu_path, dll_path, config="release", verify_fp=False, fast_math=False, force=False):

    cuda_home = warp.config.cuda_path
    cuda_cmd = None

    import pathlib
    warp_home_path = pathlib.Path(__file__).parent
    warp_home = warp_home_path.resolve()
    nanovdb_home = (warp_home_path.parent / "_build/host-deps/nanovdb/include")

    if (cu_path != None and cuda_home == None):
        print("CUDA toolchain not found, skipping CUDA build")
    
    if(force == False):

        if (os.path.exists(dll_path) == True):

            dll_time = os.path.getmtime(dll_path)
            cache_valid = True

            # check if output exists and is newer than source
            if (cu_path):                   
                cu_time = os.path.getmtime(cu_path)
                if (cu_time > dll_time):
                    
                    if (warp.config.verbose):
                        print(f"cu_time: {cu_time} > dll_time: {dll_time} invaliding cache for {cu_path}")

                    cache_valid = False

            if (cpp_path):
                cpp_time = os.path.getmtime(cpp_path)
                if (cpp_time > dll_time):
                    
                    if (warp.config.verbose):
                        print(f"cpp_time: {cpp_time} > dll_time: {dll_time} invaliding cache for {cpp_path}")

                    cache_valid = False
                
            if (cache_valid):
                
                if (warp.config.verbose):
                    print("Skipping build of {} since outputs newer than inputs".format(dll_path))
                
                return True

    # ensure that dll is not loaded in the process
    force_unload_dll(dll_path)

    # output stale, rebuild
    if (warp.config.verbose):
        print("Building {}".format(dll_path))

    native_dir = os.path.join(warp_home, "native")

    # is the library being built with CUDA enabled?
    if cuda_home is None or cuda_home == "":
        cuda_disabled = 1
    else:
        cuda_disabled = 0

        # check CUDA Toolkit version
        min_ctk_version = (11, 5)
        ctk_version = get_cuda_toolkit_version(cuda_home) or min_ctk_version
        if ctk_version < min_ctk_version:
            raise Exception(f"CUDA Toolkit version {min_ctk_version[0]}.{min_ctk_version[1]}+ is required (found {ctk_version[0]}.{ctk_version[1]} in {cuda_home})")

        # generate code for all supported architectures
        gencode_opts = [
            # SASS for supported desktop/datacenter architectures
            "-gencode=arch=compute_52,code=sm_52",  # Maxwell
            "-gencode=arch=compute_60,code=sm_60",  # Pascal
            "-gencode=arch=compute_61,code=sm_61",
            "-gencode=arch=compute_70,code=sm_70",  # Volta
            "-gencode=arch=compute_75,code=sm_75",  # Turing
            "-gencode=arch=compute_80,code=sm_80",  # Ampere
            "-gencode=arch=compute_86,code=sm_86",

            # SASS for supported mobile architectures (e.g. Tegra/Jetson)
            # "-gencode=arch=compute_53,code=sm_53",
            # "-gencode=arch=compute_62,code=sm_62",
            # "-gencode=arch=compute_72,code=sm_72",
            # "-gencode=arch=compute_87,code=sm_87",
        ]

        # support for Ada and Hopper is available with CUDA Toolkit 11.8+
        if ctk_version < (11, 8):
            gencode_opts += [
                # PTX for future compatibility
                "-gencode=arch=compute_86,code=compute_86",
            ]
        else: # ctk_version >= (11, 8)
            gencode_opts += [
                "-gencode=arch=compute_89,code=sm_89",  # Ada
                "-gencode=arch=compute_90,code=sm_90",  # Hopper

                # PTX for future compatibility
                "-gencode=arch=compute_90,code=compute_90",
            ]

        nvcc_opts = gencode_opts + [
            "-t0", # multithreaded compilation
            "--extended-lambda",
        ]

        if fast_math:
            nvcc_opts.append("--use_fast_math")

    if os.name == 'nt':

        if not warp.config.host_compiler:
            raise RuntimeError("Warp build error: Host compiler was not found")
        
        host_linker = os.path.join(os.path.dirname(warp.config.host_compiler), "link.exe")

        cpp_out = cpp_path + ".obj"

        if cuda_disabled:
            cuda_includes = ""
        else:
            cuda_includes = f' /I"{cuda_home}/include"'

        if (config == "debug"):
            cpp_flags = f'/MTd /Zi /Od /D "_DEBUG" /D "WP_CPU" /D "WP_DISABLE_CUDA={cuda_disabled}" /D "_ITERATOR_DEBUG_LEVEL=0" /I"{native_dir}" /I"{nanovdb_home}" {cuda_includes}'
            ld_flags = '/DEBUG /dll'
            ld_inputs = []

        elif (config == "release"):
            cpp_flags = f'/Ox /D "NDEBUG" /D "WP_CPU" /D "WP_DISABLE_CUDA={cuda_disabled}" /D "_ITERATOR_DEBUG_LEVEL=0" /I"{native_dir}" /I"{nanovdb_home}" {cuda_includes}'
            ld_flags = '/dll'
            ld_inputs = []

        else:
            raise RuntimeError("Unrecognized build configuration (debug, release), got: {}".format(config))

        if verify_fp:
            cpp_flags += ' /D "WP_VERIFY_FP"'

        if fast_math:
            cpp_flags += " /fp:fast"


        with ScopedTimer("build", active=warp.config.verbose):
            cpp_cmd = f'"{warp.config.host_compiler}" {cpp_flags} -c "{cpp_path}" /Fo"{cpp_out}"'
            run_cmd(cpp_cmd)

            ld_inputs.append(quote(cpp_out))

        if cu_path and not cuda_disabled:

            cu_out = cu_path + ".o"

            if (config == "debug"):
                cuda_cmd = f'"{cuda_home}/bin/nvcc" --compiler-options=/MTd,/Zi,/Od -g -G -O0 -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -I"{native_dir}" -I"{nanovdb_home}" -line-info {" ".join(nvcc_opts)} -DWP_CUDA -o "{cu_out}" -c "{cu_path}"'

            elif (config == "release"):
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -O3 {" ".join(nvcc_opts)} -I"{native_dir}" -I"{nanovdb_home}" -DNDEBUG -DWP_CUDA -o "{cu_out}" -c "{cu_path}"'

            with ScopedTimer("build_cuda", active=warp.config.verbose):
                run_cmd(cuda_cmd)
                ld_inputs.append(quote(cu_out))
                ld_inputs.append("cudart_static.lib nvrtc_static.lib nvrtc-builtins_static.lib nvptxcompiler_static.lib ws2_32.lib user32.lib /LIBPATH:{}/lib/x64".format(quote(cuda_home)))

        with ScopedTimer("link", active=warp.config.verbose):
            link_cmd = f'"{host_linker}" {" ".join(ld_inputs)} {ld_flags} /out:"{dll_path}"'
            run_cmd(link_cmd)
        
    else:

        cpp_out = cpp_path + ".o"

        if cuda_disabled:
            cuda_includes = ""
        else:
            cuda_includes = f' -I"{cuda_home}/include"'

        if (config == "debug"):
            cpp_flags = f'-O0 -g -D_DEBUG -DWP_CPU -DWP_DISABLE_CUDA={cuda_disabled} -fPIC -fvisibility=hidden --std=c++11 -fkeep-inline-functions -I"{native_dir}" {cuda_includes}'
            ld_flags = "-D_DEBUG"
            ld_inputs = []

        if (config == "release"):
            cpp_flags = f'-O3 -DNDEBUG -DWP_CPU -DWP_DISABLE_CUDA={cuda_disabled} -fPIC -fvisibility=hidden --std=c++11 -I"{native_dir}" {cuda_includes}'
            ld_flags = "-DNDEBUG"
            ld_inputs = []

        if verify_fp:
            cpp_flags += ' -DWP_VERIFY_FP'

        if fast_math:
            cpp_flags += ' -ffast-math'

        with ScopedTimer("build", active=warp.config.verbose):
            build_cmd = f'g++ {cpp_flags} -c "{cpp_path}" -o "{cpp_out}"'
            run_cmd(build_cmd)

            ld_inputs.append(quote(cpp_out))

        if cu_path and not cuda_disabled:

            cu_out = cu_path + ".o"

            if (config == "debug"):
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -g -G -O0 --compiler-options -fPIC,-fvisibility=hidden -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -line-info {" ".join(nvcc_opts)} -DWP_CUDA -I"{native_dir}" -o "{cu_out}" -c "{cu_path}"'

            elif (config == "release"):
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -O3 --compiler-options -fPIC,-fvisibility=hidden {" ".join(nvcc_opts)} -DNDEBUG -DWP_CUDA -I"{native_dir}" -o "{cu_out}" -c "{cu_path}"'

            with ScopedTimer("build_cuda", active=warp.config.verbose):
                run_cmd(cuda_cmd)

                ld_inputs.append(quote(cu_out))
                ld_inputs.append('-L"{cuda_home}/lib64" -lcudart_static -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static -lpthread -ldl -lrt'.format(cuda_home=cuda_home))

        if sys.platform == 'darwin':
            opt_no_undefined = "-Wl,-undefined,error"
            opt_exclude_libs = ""
        else:
            opt_no_undefined = "-Wl,--no-undefined"
            opt_exclude_libs = "-Wl,--exclude-libs,ALL"

        with ScopedTimer("link", active=warp.config.verbose):
            link_cmd = f"g++ -shared -Wl,-rpath,'$ORIGIN' {opt_no_undefined} {opt_exclude_libs} -o '{dll_path}' {' '.join(ld_inputs)}"
            run_cmd(link_cmd)

    
def load_dll(dll_path):    
    try:
        if (sys.version_info[0] > 3 or
            sys.version_info[0] == 3 and sys.version_info[1] >= 8):
                dll = ctypes.CDLL(dll_path, winmode=0)
        else:
            dll = ctypes.CDLL(dll_path)
    except OSError:
        raise RuntimeError("Failed to load the shared library '{}'".format(dll_path))
    return dll

def unload_dll(dll):

    handle = dll._handle
    del dll

    # force garbage collection to eliminate any Python references to the dll
    import gc
    gc.collect()

    # platform dependent unload, removes *all* references to the dll
    # note this should only be performed if you know there are no dangling
    # refs to the dll inside the Python program
    if os.name == "nt":
        max_attempts = 100
        for i in range(max_attempts):
            result = ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            if result != 0:
                return
    else:
        _ctypes.dlclose(handle)


def force_unload_dll(dll_path):

    try:
        # force load/unload of the dll from the process 
        dll = load_dll(dll_path)
        unload_dll(dll)

    except Exception as e:
        return


kernel_bin_dir = None
kernel_gen_dir = None


def init_kernel_cache(path=None):
    """Initialize kernel cache directory.

    This function is used during Warp initialization, but it can also be called directly to change the cache location.
    If the path is not explicitly specified, a default location will be chosen based on OS-specific conventions.

    To change the default cache location, set warp.config.kernel_cache_dir before calling warp.init().
    """

    warp_root_dir = os.path.dirname(os.path.realpath(__file__))
    warp_bin_dir = os.path.join(warp_root_dir, "bin")

    if path is not None:
        cache_root_dir = os.path.realpath(path)
    else:
        cache_root_dir = appdirs.user_cache_dir(appname="warp", appauthor="NVIDIA Corporation", version=warp.config.version)

    cache_bin_dir = os.path.join(cache_root_dir, "bin")
    cache_gen_dir = os.path.join(cache_root_dir, "gen")

    if not os.path.isdir(cache_root_dir):
        #print("Creating cache directory '%s'" % cache_root_dir)
        os.makedirs(cache_root_dir, exist_ok=True)

    if not os.path.isdir(cache_gen_dir):
        #print("Creating codegen directory '%s'" % cache_gen_dir)
        os.makedirs(cache_gen_dir, exist_ok=True)

    if not os.path.isdir(cache_bin_dir):
        #print("Creating binary directory '%s'" % cache_bin_dir)
        os.makedirs(cache_bin_dir, exist_ok=True)

    warp.config.kernel_cache_dir = cache_root_dir
    
    global kernel_bin_dir, kernel_gen_dir
    kernel_bin_dir = cache_bin_dir
    kernel_gen_dir = cache_gen_dir


def clear_kernel_cache():
    """Clear the kernel cache."""

    import glob

    paths = []

    if kernel_bin_dir is not None and os.path.isdir(kernel_bin_dir):
        pattern = os.path.join(kernel_bin_dir, "wp_*")
        paths += glob.glob(pattern)

    if kernel_gen_dir is not None and os.path.isdir(kernel_gen_dir):
        pattern = os.path.join(kernel_gen_dir, "wp_*")
        paths += glob.glob(pattern)

    for p in paths:
        if os.path.isfile(p):
            os.remove(p)
