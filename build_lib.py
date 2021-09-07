# This script is an 'offline' build of the core warp runtime libraries
# designed to be executed as part of CI / developer workflows, not 
# as part of the user runtime (since it requires CUDA toolkit, etc)

import os
import warp

# set build output path off this file
build_path = os.path.dirname(os.path.realpath(__file__)) + "/warp"

# find host compiler
if (warp.config.host_compiler == None):
    warp.config.host_compiler = warp.build.find_host_compiler()

# no host toolchain not found
if (warp.config.host_compiler == None):
    raise Exception("Warp: Could not find host compiler (MSVC, GCC, etc), ensure that the compiler is present in the environment")

# check for CUDA
if (warp.config.cuda_path == None):
    warp.config.cuda_path = warp.build.find_cuda()

# no CUDA toolchain not found
if (warp.config.cuda_path == None):
    print("Warning: building without CUDA support")

    warp.build.build_dll(
                    cpp_path=build_path + "/native/warp.cpp", 
                    cu_path=None, 
                    dll_path=build_path + "/bin/warp.so",
                    config=warp.config.mode,
                    force=True)

else:
    if (os.name == 'nt'):
        dll_path = build_path + "/bin/warp.dll"
    else:
        dll_path = build_path + "/bin/warp.so"

    warp.build.build_dll(
                    cpp_path=build_path + "/native/warp.cpp", 
                    cu_path=build_path + "/native/warp.cu", 
                    dll_path=dll_path,
                    config=warp.config.mode,
                    force=True)
                    
