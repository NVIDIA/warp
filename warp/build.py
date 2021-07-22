import math
import os
import sys
import imp
import subprocess
from ctypes import *

import warp.config
from warp.utils import ScopedTimer

def run_cmd(cmd, capture=False):
    
    if (warp.config.verbose):
        print(cmd)

    try:
        return subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output.decode())
        raise(e)
        

# runs vcvars and copies back the build environment
def find_host_compiler():

    if (os.name == 'nt'):

        if os.system("where cl.exe >nul 2>nul") != 0:

            # cl.exe not on path then set vcvars
            def find_vcvars_path():
                import glob
                for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                    paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Auxiliary\Build\vcvars64.bat" % edition), reverse=True)
                    if paths:
                        return paths[0]

            vcvars_path = find_vcvars_path()

            # merge vcvars with our env
            s = '"{}" && set'.format(vcvars_path)
            output = os.popen(s).read()
            for line in output.splitlines():
                pair = line.split("=", 1)
                if (len(pair) >= 2):
                    os.environ[pair[0]] = pair[1]

        # try and find cl.exe
        try:
            return subprocess.check_output("where cl.exe").decode()
        except:
            return None
    else:
        
        # try and find g++
        try:
            return run_cmd("which g++").decode()
        except:
            return None





# See PyTorch for reference on how to find nvcc.exe more robustly, https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension
def find_cuda():
    
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    return cuda_home
    

# builds cuda->ptx using NVRTC
def build_cuda(cu_path, ptx_path, config="release", force=False):

    src_file = open(cu_path)
    src = src_file.read().encode('utf-8')
    src_file.close()

    inc_path = os.path.dirname(cu_path).encode('utf-8')
    ptx_path = ptx_path.encode('utf-8')

    warp.context.runtime.core.cuda_compile_program(src, inc_path, False, warp.config.verbose, ptx_path)

# load ptx to a CUDA runtime module    
def load_cuda(ptx_path):

    module = warp.context.runtime.core.cuda_load_module(ptx_path.encode('utf-8'))
    return module


def quote(path):
    return "\"" + path + "\""

def build_dll(cpp_path, cu_path, dll_path, config="release", force=False):

    cuda_home = warp.config.cuda_path
    cuda_cmd = None

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
                    cache_valid = False

            if (cpp_path):
                cpp_time = os.path.getmtime(cpp_path)
                if (cpp_time > dll_time):
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

    if os.name == 'nt':

        cpp_out = cpp_path + ".obj"

        if (config == "debug"):
            cpp_flags = '/MTd /Zi /Od /D "_DEBUG" /D "WP_CPU" /D "_ITERATOR_DEBUG_LEVEL=0" '
            ld_flags = '/DEBUG /dll'
            ld_inputs = []

        elif (config == "release"):
            cpp_flags = '/Ox /D "NDEBUG" /D "WP_CPU" /D "_ITERATOR_DEBUG_LEVEL=0" /fp:fast'
            ld_flags = '/dll'
            ld_inputs = []

        else:
            raise RuntimeError("Unrecognized build configuration (debug, release), got: {}".format(config))


        with ScopedTimer("build", active=warp.config.verbose):
            cpp_cmd = 'cl.exe {cflags} -c "{cpp_path}" /Fo"{cpp_out}"'.format(cflags=cpp_flags, cpp_out=cpp_out, cpp_path=cpp_path)
            run_cmd(cpp_cmd)

            ld_inputs.append(quote(cpp_out))

        if (cuda_home and cu_path):

            cu_out = cu_path + ".o"

            if (config == "debug"):
                cuda_cmd = '"{cuda_home}/bin/nvcc" --compiler-options=/MTd,/Zi,/Od -g -G -O0 -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -line-info -gencode=arch=compute_35,code=compute_35 -DWP_CUDA -o "{cu_out}" -c "{cu_path}"'.format(cuda_home=cuda_home, cu_out=cu_out, cu_path=cu_path)

            elif (config == "release"):
                cuda_cmd = '"{cuda_home}/bin/nvcc" -O3 -gencode=arch=compute_35,code=compute_35 --use_fast_math -DWP_CUDA -o "{cu_out}" -c "{cu_path}"'.format(cuda_home=cuda_home, cu_out=cu_out, cu_path=cu_path)

            with ScopedTimer("build_cuda", active=warp.config.verbose):
                run_cmd(cuda_cmd)
                ld_inputs.append(quote(cu_out))
                ld_inputs.append("cudart.lib cuda.lib nvrtc.lib /LIBPATH:{}/lib/x64".format(quote(cuda_home)))

        with ScopedTimer("link", active=warp.config.verbose):
            link_cmd = 'link.exe {inputs} {flags} /out:"{dll_path}"'.format(inputs=' '.join(ld_inputs), flags=ld_flags, dll_path=dll_path)
            run_cmd(link_cmd)
        
    else:

        cpp_out = cpp_path + ".o"

        if (config == "debug"):
            cpp_flags = "-O0 -g -D_DEBUG -DWP_CPU -fPIC --std=c++11"
            ld_flags = "-D_DEBUG"
            ld_inputs = []

        if (config == "release"):
            cpp_flags = "-O3 -DNDEBUG -DWP_CPU  -fPIC --std=c++11"
            ld_flags = "-DNDEBUG"
            ld_inputs = []


        with ScopedTimer("build", active=warp.config.verbose):
            build_cmd = 'g++ {cflags} -c "{cpp_path}" -o "{cpp_out}"'.format(cflags=cpp_flags, cpp_out=cpp_out, cpp_path=cpp_path)
            run_cmd(build_cmd)

            ld_inputs.append(quote(cpp_out))

        if (cuda_home and cu_path):

            cu_out = cu_path + ".o"

            cuda_cmd = '"{cuda_home}/bin/nvcc" -gencode=arch=compute_35,code=compute_35 -DWP_CUDA --compiler-options -fPIC -o "{cu_out}" -c "{cu_path}"'.format(cuda_home=cuda_home, cu_out=cu_out, cu_path=cu_path)

            with ScopedTimer("build_cuda", active=warp.config.verbose):
                run_cmd(cuda_cmd)

                ld_inputs.append(quote(cu_out))
                ld_inputs.append('-L"{cuda_home}/lib64" -lcudart')

        with ScopedTimer("link", active=warp.config.verbose):
            link_cmd = 'g++ -shared -o "{dll_path}" {inputs}'.format(cuda_home=cuda_home, inputs=' '.join(ld_inputs), dll_path=dll_path)            
            run_cmd(link_cmd)

    
def load_dll(dll_path):
    
    dll = CDLL(dll_path)
    return dll

def unload_dll(dll):
    
    handle = dll._handle
    del dll

    import _ctypes
    
    # platform dependent unload, removes *all* references to the dll
    # note this should only be performed if you know there are no dangling
    # refs to the dll inside the Python prwpram 
    try:
        while (True):
            _ctypes.FreeLibrary(handle)
    except:
        return

def force_unload_dll(dll_path):

    try:
        # force load/unload of the dll from the process 
        dll = load_dll(dll_path)
        unload_dll(dll)
    except:
        return
