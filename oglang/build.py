import math
import os
import sys
import imp
import subprocess
from ctypes import *

from oglang.utils import ScopedTimer

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



# See PyTorch for reference on how to find nvcc.exe more robustly, https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension
def find_cuda():
    
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    return cuda_home
    


def build_module(cpp_path, cu_path, dll_path, config="release", load=True, force=False):

    set_build_env()

    cuda_home = find_cuda()
    cuda_cmd = None
    
    if(force == False):

        if (os.path.exists(dll_path) == True):

            # check if output exists and is newer than source
            cu_time = os.path.getmtime(cu_path)
            cpp_time = os.path.getmtime(cpp_path)
            dll_time = os.path.getmtime(dll_path)

            if (cu_time < dll_time and cpp_time < dll_time):
                # output valid, skip build
                print("Skipping build of {} since outputs newer than inputs".format(dll_path))
                return True

    # output stale, rebuild
    print("Building {}".format(dll_path))

    if os.name == 'nt':

        cpp_out = cpp_path + ".obj"
        cu_out = cu_path + ".o"

        if (config == "debug"):
            cpp_flags = "/Zi, /Od, /DEBUG"
            ld_flags = "/DEBUG /dll"
            ld_inputs = []

        if (config == "release"):
            cpp_flags = "/Ox, -DNDEBUG, /fp:fast"
            ld_flags = "/dll"
            ld_inputs = []


        with ScopedTimer("build"):
            cpp_cmd = "cl.exe {cflags} -DCPU -c {cpp_path} /Fo{cpp_path}.obj ".format(cflags=cpp_flags, cpp_path=cpp_path)
            print(cpp_cmd)
            err = subprocess.call(cpp_cmd)

            if (err):
                raise RuntimeError("cpp build failed")

            ld_inputs.append(cpp_out)

        if (cuda_home):

            if (config == "debug"):
                cuda_cmd = "{cuda_home}/bin/nvcc --compiler-options=/Zi,/Od -g -G -O0 -line-info -gencode=arch=compute_35,code=compute_35 -DCUDA -o {cu_path}.o -c {cu_path}".format(cuda_home=cuda_home, cu_path=cu_path)

            elif (config == "release"):
                cuda_cmd = "{cuda_home}/bin/nvcc -O3 -gencode=arch=compute_35,code=compute_35 --use_fast_math -DCUDA -o {cu_path}.o -c {cu_path}".format(cuda_home=cuda_home, cu_path=cu_path)

            with ScopedTimer("build_cuda"):
                print(cuda_cmd)
                err = subprocess.call(cuda_cmd)

                if (err):
                    raise RuntimeError("cuda build failed")

                ld_inputs.append(cu_out)

        with ScopedTimer("link"):
            link_cmd = 'link.exe {inputs} cudart.lib {flags} /LIBPATH:"{cuda_home}/lib/x64" /out:{dll_path}'.format(inputs=' '.join(ld_inputs), cuda_home=cuda_home, flags=ld_flags, dll_path=dll_path)
            print(link_cmd)

            err = subprocess.call(link_cmd)
            if (err):
                raise RuntimeError("Link failed")

        
    else:

        cpp_out = cpp_path + ".o"
        cu_out = cu_path + ".o"

        if (config == "debug"):
            cpp_flags = "-O0 -g -D_DEBUG -fPIC --std=c++11"
            ld_flags = "-D_DEBUG"
            ld_inputs = []

        if (config == "release"):
            cpp_flags = "-O3 -DNDEBUG -fPIC --std=c++11"
            ld_flags = "-DNDEBUG"
            ld_inputs = []


        with ScopedTimer("build"):
            build_cmd = "g++ {cflags} -c -o {cpp_path}.o {cpp_path}".format(cflags=cpp_flags, cpp_path=cpp_path)
            print(build_cmd)
            err = subprocess.call(build_cmd, shell=True)

            if (err):
                raise RuntimeError("C++ build failed")

            ld_inputs.append(cpp_out)

        if (cuda_home):

            cuda_cmd = "{cuda_home}/bin/nvcc -gencode=arch=compute_35,code=compute_35 -DCUDA --compiler-options -fPIC -o {cu_path}.o -c {cu_path}".format(cuda_home=cuda_home, cu_path=cu_path)

            with ScopedTimer("build_cuda"):
                print(cuda_cmd)
                err = subprocess.call(cuda_cmd, shell=True)

                if (err):
                    raise RuntimeError("cuda build failed")

                ld_inputs.append(cu_out)
                ld_inputs.append("-L{cuda_home}/lib64 -lcudart")

        with ScopedTimer("link"):
            link_cmd = "g++ -shared -o {dll_path} {inputs}".format(cuda_home=cuda_home, inputs=' '.join(ld_inputs), dll_path=dll_path)
            print(link_cmd)
            err = subprocess.call(link_cmd, shell=True)

            if (err):
                raise RuntimeError("link failed")


    
def load_module(dll_path):
    
    dll = CDLL(dll_path)
    return dll

