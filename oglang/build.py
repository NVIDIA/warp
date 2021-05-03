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
    


def build_module(cpp_path, cu_path, dll_path, config="release", load=True):

    # cache stale, rebuild
    print("Building {}".format(dll_path))

    set_build_env()

    cuda_home = find_cuda()
    cuda_cmd = None

    try:
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

                if (err == False):
                    ld_inputs.append(cpp_out)
                else:
                    raise RuntimeError("cpp build failed")

            if (cuda_home):
                cuda_cmd = "{cuda_home}/bin/nvcc -O3 -gencode=arch=compute_35,code=compute_35 --use_fast_math -DCUDA -o {cu_path}.o -c {cu_path}".format(cuda_home=cuda_home, cu_path=cu_path)
                #cuda_cmd = "{cuda_home}/bin/nvcc --compiler-options=/Zi,/Od -g -G -O0 -line-info -gencode=arch=compute_35,code=compute_35 -DCUDA -o {cu_path}.o -c {cu_path}".format(cuda_home=cuda_home, cu_path=cu_path)

                with ScopedTimer("build_cuda"):
                    print(cuda_cmd)
                    err = subprocess.call(cuda_cmd)

                    if (err == False):
                        ld_inputs.append(cu_out)
                    else:
                        raise RuntimeError("cuda build failed")


            with ScopedTimer("link"):
                link_cmd = 'link.exe {inputs} cudart.lib {flags} /LIBPATH:"{cuda_home}/lib/x64" /out:{dll_path}'.format(inputs=' '.join(ld_inputs), cuda_home=cuda_home, flags=ld_flags, dll_path=dll_path)
                print(link_cmd)

                err = subprocess.call(link_cmd)
                if (err):
                    raise RuntimeError("Link failed")                    

            
        else:

            if (config == "debug"):
                cpp_flags = "-Z -O0 -g -D_DEBUG -fPIC --std=c++11 -DCPU"
                ld_flags = "-D_DEBUG"

            if (config == "release"):
                cpp_flags = "-Z -O3 -DNDEBUG -fPIC --std=c++11 -DCPU"
                ld_flags = "-DNDEBUG"


            with ScopedTimer("build"):
                build_cmd = "g++ {cflags} -c -o {cpp_path}.o {cpp_path}".format(cflags=cpp_flags, cpp_path=cpp_path)
                print(build_cmd)
                subprocess.call(build_cmd, shell=True)

            if (cuda_home):

                cuda_cmd = "{cuda_home}/bin/nvcc -gencode=arch=compute_35,code=compute_35 --compiler-options -fPIC -DCUDA -o {cu_path}.o -c {cu_path}".format(cuda_home=cuda_home, cu_path=cu_path)

                with ScopedTimer("build_cuda"):
                    print(cuda_cmd)
                    err = subprocess.call(cuda_cmd, shell=True)

                    if (err):
                        raise RuntimeError("cuda build failed")

            with ScopedTimer("link"):
                link_cmd = "g++ -shared -L{cuda_home}/lib64 -o {dll_path} {cpp_path}.o {cu_path}.o -lcudart".format(cuda_home=cuda_home, cpp_path=cpp_path, cu_path=cu_path, dll_path=dll_path)
                print(link_cmd)
                subprocess.call(link_cmd, shell=True)

    except Exception as e:

        # print error 
        print("Build failed, using cached binaries (if available")
        print(e)

    
def load_module(dll_path):
    
    dll = CDLL(dll_path)
    return dll

