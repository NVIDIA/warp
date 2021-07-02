import os
import warp

# set build output path off this file
build_path = os.path.dirname(os.path.realpath(__file__)) + "/warp"

# check for CUDA
if (warp.config.cuda_path == None):
    warp.config.cuda_path = warp.build.find_cuda()

# still no CUDA toolchain not found
if (warp.config.cuda_path == None):
    raise Exception("Warp: Could not find CUDA toolkit, ensure that the CUDA_PATH environment variable is set or specify manually in warp.config.cuda_path before initialization")

try:

    warp.build.build_dll(
                    cpp_path=build_path + "/native/warp.cpp", 
                    cu_path=build_path + "/native/warp.cu", 
                    dll_path=build_path + "/bin/warp.dll",
                    config=warp.config.mode,
                    force=True)
                    
except Exception as e:

    raise Exception("Could not build core library.")
