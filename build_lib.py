# This script is an 'offline' build of the core warp runtime libraries
# designed to be executed as part of CI / developer workflows, not 
# as part of the user runtime (since it requires CUDA toolkit, etc)

import os
import sys
import argparse

import warp.config
import warp.build

parser = argparse.ArgumentParser(description="Warp build script")
parser.add_argument('--msvc_path', type=str, help='Path to MSVC compiler (optional if already on PATH)')
parser.add_argument('--sdk_path', type=str, help='Path to WinSDK (optional if already on PATH)')
parser.add_argument('--cuda_path', type=str, help='Path to CUDA SDK')
parser.add_argument('--mode', type=str, default="release", help="Build configuration, either 'release' or 'debug'")
parser.add_argument('--verbose', type=bool, default=True, help="Verbose building output, default True")
args = parser.parse_args()

# set build output path off this file
build_path = os.path.dirname(os.path.realpath(__file__)) + "/warp"

print(args)

warp.config.verbose = args.verbose
warp.config.mode = args.mode

# setup CUDA paths
if sys.platform == 'darwin':

    warp.config.cuda_path = None

else:

    if args.cuda_path:
        warp.config.cuda_path = args.cuda_path
    else:
        warp.config.cuda_path = warp.build.find_cuda()


# setup MSVC and WinSDK paths
if os.name == 'nt':
    
    if args.sdk_path and args.msvc_path:
        # user provided MSVC
        warp.build.set_msvc_compiler(msvc_path=args.msvc_path, sdk_path=args.sdk_path)
    else:
        
        # attempt to find MSVC in environment (will set vcvars)
        cl_path = warp.build.find_host_compiler()
        
        if (cl_path == None):
            print("Could not find MSVC compiler in path")
            sys.exit(1)


try:

    if (os.name == 'nt'):
        dll_path = build_path + "/bin/warp.dll"
    else:
        dll_path = build_path + "/bin/warp.so"

    # no CUDA toolchain found
    if (warp.config.cuda_path == None):
        print("Warning: building without CUDA support")

        warp.build.build_dll(
                        cpp_path=build_path + "/native/warp.cpp", 
                        cu_path=None, 
                        dll_path=dll_path,
                        config=warp.config.mode,
                        force=True)

    else:

        warp.build.build_dll(
                        cpp_path=build_path + "/native/warp.cpp", 
                        cu_path=build_path + "/native/warp.cu", 
                        dll_path=dll_path,
                        config=warp.config.mode,
                        force=True)
                    
except Exception as e:

    # output build error
    print(e)

    # report error
    sys.exit(1)
