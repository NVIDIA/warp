# This script is an 'offline' build of the core warp runtime libraries
# designed to be executed as part of CI / developer workflows, not 
# as part of the user runtime (since it requires CUDA toolkit, etc)

import sys
if sys.version_info[0] < 3:
    raise Exception("Warp requires Python 3.x minimum")

import os
import argparse

import warp.config
import warp.build

parser = argparse.ArgumentParser(description="Warp build script")
parser.add_argument('--msvc_path', type=str, help='Path to MSVC compiler (optional if already on PATH)')
parser.add_argument('--sdk_path', type=str, help='Path to WinSDK (optional if already on PATH)')
parser.add_argument('--cuda_path', type=str, help='Path to CUDA SDK')
parser.add_argument('--mode', type=str, default="release", help="Build configuration, either 'release' or 'debug'")
parser.add_argument('--verbose', type=bool, default=True, help="Verbose building output, default True")
parser.add_argument('--verify_fp', type=bool, default=False, help="Verify kernel inputs and outputs are finite after each launch, default False")
parser.add_argument('--fast_math', type=bool, default=False, help="Enable fast math on library, default False")
parser.add_argument('--quick', action='store_true', help="Only generate PTX code, disable CUTLASS ops")
parser.add_argument('--build_llvm', type=bool, default=False, help="Build a bundled Clang/LLVM compiler")
parser.add_argument('--standalone', type=bool, default=True, help="Use standalone LLVM-based JIT")
args = parser.parse_args()

# set build output path off this file
base_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(base_path, "warp")

print(args)

warp.config.verbose = args.verbose
warp.config.mode = args.mode
warp.config.verify_fp = args.verify_fp
warp.config.fast_math = args.fast_math


# See PyTorch for reference on how to find nvcc.exe more robustly, https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension
def find_cuda():
    
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    return cuda_home


# setup CUDA paths
if sys.platform == 'darwin':

    warp.config.cuda_path = None

else:

    if args.cuda_path:
        warp.config.cuda_path = args.cuda_path
    else:
        warp.config.cuda_path = find_cuda()


# setup MSVC and WinSDK paths
if os.name == 'nt':
    
    if args.sdk_path and args.msvc_path:
        # user provided MSVC
        warp.build.set_msvc_compiler(msvc_path=args.msvc_path, sdk_path=args.sdk_path)
    else:
        
        # attempt to find MSVC in environment (will set vcvars)
        warp.config.host_compiler = warp.build.find_host_compiler()
        
        if not warp.config.host_compiler:
            print("Warp build error: Could not find MSVC compiler")
            sys.exit(1)


if args.build_llvm:
    
    import subprocess
    from git import Repo

    llvm_project_dir = "external/llvm-project"
    llvm_project_path = os.path.join(base_path, llvm_project_dir)
    repo_url = "https://github.com/llvm/llvm-project.git"

    if not os.path.exists(llvm_project_path):
        print(f"Cloning LLVM project...")
        shallow_clone = True  # https://github.blog/2020-12-21-get-up-to-speed-with-partial-clone-and-shallow-clone/
        if shallow_clone:
            repo = Repo.clone_from(repo_url, to_path=llvm_project_path, single_branch=True, branch="llvmorg-15.0.7", depth=1)
        else:
            repo = Repo.clone_from(repo_url, to_path=llvm_project_path)
            repo.git.checkout("tags/llvmorg-15.0.7", "-b", "llvm-15.0.7")
    else:
        print(f"Found existing {llvm_project_dir} directory")
        repo = Repo(llvm_project_path)

    # CMake supports Debug, Release, RelWithDebInfo, and MinSizeRel builds
    if warp.config.mode == "release":
        cmake_build_type = "MinSizeRel"  # prefer smaller size over aggressive speed
    else:
        cmake_build_type = "Debug"

    llvm_path = os.path.join(llvm_project_path, "llvm")
    llvm_build_path = os.path.join(llvm_project_path, f"out/build/{warp.config.mode}")
    llvm_install_path = os.path.join(llvm_project_path, f"out/install/{warp.config.mode}")

    # Location of cmake and ninja installed through pip (see build.bat / build.sh)
    python_bin = "python/Scripts" if sys.platform == "win32" else "python/bin"
    os.environ["PATH"] = os.path.join(base_path, "_build/target-deps/" + python_bin) + os.pathsep + os.environ["PATH"]

    # Build LLVM and Clang
    cmake_gen = ["cmake", "-S", llvm_path,
                          "-B", llvm_build_path,
                          "-G", "Ninja",
                          "-D", f"CMAKE_BUILD_TYPE={cmake_build_type}",
                          "-D", "LLVM_USE_CRT_RELEASE=MT",
                          "-D", "LLVM_USE_CRT_MINSIZEREL=MT",
                          "-D", "LLVM_USE_CRT_DEBUG=MTd",
                          "-D", "LLVM_USE_CRT_RELWITHDEBINFO=MTd",
                          "-D", "LLVM_TARGETS_TO_BUILD=X86",
                          "-D", "LLVM_ENABLE_PROJECTS=clang",
                          "-D", "LLVM_ENABLE_ZLIB=FALSE",
                          "-D", "LLVM_ENABLE_ZSTD=FALSE",
                          "-D", "LLVM_ENABLE_TERMINFO=FALSE",
                          "-D", "LLVM_BUILD_LLVM_C_DYLIB=FALSE",
                          "-D", "LLVM_BUILD_RUNTIME=FALSE",
                          "-D", "LLVM_BUILD_RUNTIMES=FALSE",
                          "-D", "LLVM_BUILD_TOOLS=FALSE",
                          "-D", "LLVM_BUILD_UTILS=FALSE",
                          "-D", "LLVM_INCLUDE_BENCHMARKS=FALSE",
                          "-D", "LLVM_INCLUDE_DOCS=FALSE",
                          "-D", "LLVM_INCLUDE_EXAMPLES=FALSE",
                          "-D", "LLVM_INCLUDE_RUNTIMES=FALSE",
                          "-D", "LLVM_INCLUDE_TESTS=FALSE",
                          "-D", "LLVM_INCLUDE_TOOLS=TRUE",  # Needed by Clang
                          "-D", "LLVM_INCLUDE_UTILS=FALSE",
                          "-D", "CMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0",  # The pre-C++11 ABI is still the default on the CentOS 7 toolchain
                          "-D", f"CMAKE_INSTALL_PREFIX={llvm_install_path}",
                          ]
    ret = subprocess.check_call(cmake_gen, stderr=subprocess.STDOUT)
    
    cmake_build = ["cmake", "--build", llvm_build_path]
    ret = subprocess.check_call(cmake_build, stderr=subprocess.STDOUT)
    
    cmake_install = ["cmake", "--install", llvm_build_path]
    ret = subprocess.check_call(cmake_install, stderr=subprocess.STDOUT)


# return platform specific shared library name
def lib_name(name):
    if sys.platform == "win32":
        return f"{name}.dll"
    elif sys.platform == "darwin":
        return f"lib{name}.dylib"
    else:
        return f"{name}.so"


try:

    if args.standalone:
        # build warp-clang.dll
        cpp_sources = [
            "clang/clang.cpp",
            "native/crt.cpp",
        ]
        clang_cpp_paths = [os.path.join(build_path, cpp) for cpp in cpp_sources]

        clang_dll_path = os.path.join(build_path, f"bin/{lib_name('warp-clang')}")

        if args.build_llvm:
            # obtain Clang and LLVM libraries from the local build
            libpath = os.path.join(llvm_install_path, "lib")
        else:
            # obtain Clang and LLVM libraries from packman
            assert os.path.exists("_build/host-deps/llvm-project"), "run build.bat / build.sh"
            libpath = os.path.join(base_path, "_build/host-deps/llvm-project/lib")

        for (_, _, libs) in os.walk(libpath):
            break  # just the top level contains library files

        if os.name == 'nt':
            libs.append("Version.lib")
            libs.append(f'/LIBPATH:"{libpath}"')
        else:
            libs = [f"-l{lib[3:-2]}" for lib in libs if os.path.splitext(lib)[1] == ".a"]
            if sys.platform == "darwin":
                libs += libs  # prevents unresolved symbols due to link order
            else:
                libs.insert(0, "-Wl,--start-group")
                libs.append("-Wl,--end-group")
            libs.append(f"-L{libpath}")
            libs.append("-lpthread")
            libs.append("-ldl")

        warp.build.build_dll(
                        dll_path=clang_dll_path,
                        cpp_paths=clang_cpp_paths,
                        cu_path=None,
                        libs=libs,
                        mode=warp.config.mode,
                        verify_fp=warp.config.verify_fp,
                        fast_math=args.fast_math)

    # build warp.dll
    cpp_sources = [
        "native/warp.cpp",
        "native/crt.cpp",
        "native/cuda_util.cpp",
        "native/mesh.cpp",
        "native/hashgrid.cpp",
        "native/sort.cpp",
        "native/volume.cpp",
        "native/marching.cpp",
        "native/cutlass_gemm.cpp",
    ]
    warp_cpp_paths = [os.path.join(build_path, cpp) for cpp in cpp_sources]

    if (warp.config.cuda_path is None):
        print("Warning: CUDA toolchain not found, building without CUDA support")
        warp_cu_path = None
    else:
        warp_cu_path = os.path.join(build_path, "native/warp.cu")

    warp_dll_path = os.path.join(build_path, f"bin/{lib_name('warp')}")

    warp.build.build_dll(
                    dll_path=warp_dll_path,
                    cpp_paths=warp_cpp_paths,
                    cu_path=warp_cu_path,
                    mode=warp.config.mode,
                    verify_fp=warp.config.verify_fp,
                    fast_math=args.fast_math,
                    quick=args.quick)
                    
except Exception as e:

    # output build error
    print(f"Warp build error: {e}")

    # report error
    sys.exit(1)
