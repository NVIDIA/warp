import sys
import os
import subprocess

import warp.build

# set build output path off this file
base_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(base_path, "warp")

llvm_project_dir = "external/llvm-project"
llvm_project_path = os.path.join(base_path, llvm_project_dir)
llvm_path = os.path.join(llvm_project_path, "llvm")
llvm_build_path = os.path.join(llvm_project_path, f"out/build/{warp.config.mode}")
llvm_install_path = os.path.join(llvm_project_path, f"out/install/{warp.config.mode}")

# Fetch prebuilt Clang/LLVM libraries
if os.name == "nt":
    subprocess.check_call(
        [
            "tools\\packman\\packman.cmd",
            "install",
            "-l",
            "_build/host-deps/llvm-project",
            "clang+llvm-warp",
            "15.0.7-windows-x86_64-vs142",
        ]
    )
elif sys.platform == "darwin":
    subprocess.check_call(
        [
            "./tools/packman/packman",
            "install",
            "-l",
            "./_build/host-deps/llvm-project",
            "clang+llvm-warp",
            "15.0.7-darwin-x86_64-macos11",
        ]
    )
else:
    subprocess.check_call(
        [
            "./tools/packman/packman",
            "install",
            "-l",
            "./_build/host-deps/llvm-project",
            "clang+llvm-warp",
            "15.0.7-linux-x86_64-gcc7.5-cxx11abi0",
        ]
    )


def build_from_source():
    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cmake"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja"])

    from git import Repo

    repo_url = "https://github.com/llvm/llvm-project.git"

    if not os.path.exists(llvm_project_path):
        print("Cloning LLVM project...")
        shallow_clone = True  # https://github.blog/2020-12-21-get-up-to-speed-with-partial-clone-and-shallow-clone/
        if shallow_clone:
            repo = Repo.clone_from(
                repo_url, to_path=llvm_project_path, single_branch=True, branch="llvmorg-15.0.7", depth=1
            )
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

    # Location of cmake and ninja installed through pip (see build.bat / build.sh)
    python_bin = "python/Scripts" if sys.platform == "win32" else "python/bin"
    os.environ["PATH"] = os.path.join(base_path, "_build/target-deps/" + python_bin) + os.pathsep + os.environ["PATH"]

    # Build LLVM and Clang
    cmake_gen = [
        "cmake",
        "-S",
        llvm_path,
        "-B",
        llvm_build_path,
        "-G",
        "Ninja",
        "-D",
        f"CMAKE_BUILD_TYPE={cmake_build_type}",
        "-D",
        "LLVM_USE_CRT_RELEASE=MT",
        "-D",
        "LLVM_USE_CRT_MINSIZEREL=MT",
        "-D",
        "LLVM_USE_CRT_DEBUG=MTd",
        "-D",
        "LLVM_USE_CRT_RELWITHDEBINFO=MTd",
        "-D",
        "LLVM_TARGETS_TO_BUILD=X86",
        "-D",
        "LLVM_ENABLE_PROJECTS=clang",
        "-D",
        "LLVM_ENABLE_ZLIB=FALSE",
        "-D",
        "LLVM_ENABLE_ZSTD=FALSE",
        "-D",
        "LLVM_ENABLE_TERMINFO=FALSE",
        "-D",
        "LLVM_BUILD_LLVM_C_DYLIB=FALSE",
        "-D",
        "LLVM_BUILD_RUNTIME=FALSE",
        "-D",
        "LLVM_BUILD_RUNTIMES=FALSE",
        "-D",
        "LLVM_BUILD_TOOLS=FALSE",
        "-D",
        "LLVM_BUILD_UTILS=FALSE",
        "-D",
        "LLVM_INCLUDE_BENCHMARKS=FALSE",
        "-D",
        "LLVM_INCLUDE_DOCS=FALSE",
        "-D",
        "LLVM_INCLUDE_EXAMPLES=FALSE",
        "-D",
        "LLVM_INCLUDE_RUNTIMES=FALSE",
        "-D",
        "LLVM_INCLUDE_TESTS=FALSE",
        "-D",
        "LLVM_INCLUDE_TOOLS=TRUE",  # Needed by Clang
        "-D",
        "LLVM_INCLUDE_UTILS=FALSE",
        "-D",
        "CMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0",  # The pre-C++11 ABI is still the default on the CentOS 7 toolchain
        "-D",
        f"CMAKE_INSTALL_PREFIX={llvm_install_path}",
    ]
    ret = subprocess.check_call(cmake_gen, stderr=subprocess.STDOUT)

    cmake_build = ["cmake", "--build", llvm_build_path]
    ret = subprocess.check_call(cmake_build, stderr=subprocess.STDOUT)

    cmake_install = ["cmake", "--install", llvm_build_path]
    ret = subprocess.check_call(cmake_install, stderr=subprocess.STDOUT)


# build warp-clang.dll
def build_warp_clang(build_llvm, lib_name):
    try:
        cpp_sources = [
            "clang/clang.cpp",
            "native/crt.cpp",
        ]
        clang_cpp_paths = [os.path.join(build_path, cpp) for cpp in cpp_sources]

        clang_dll_path = os.path.join(build_path, f"bin/{lib_name}")

        if build_llvm:
            # obtain Clang and LLVM libraries from the local build
            libpath = os.path.join(llvm_install_path, "lib")
        else:
            # obtain Clang and LLVM libraries from packman
            assert os.path.exists("_build/host-deps/llvm-project"), "run build.bat / build.sh"
            libpath = os.path.join(base_path, "_build/host-deps/llvm-project/lib")

        for _, _, libs in os.walk(libpath):
            break  # just the top level contains library files

        if os.name == "nt":
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
            mode=warp.config.mode if build_llvm else "release",
            verify_fp=warp.config.verify_fp,
            fast_math=warp.config.fast_math,
        )

    except Exception as e:
        # output build error
        print(f"Warp Clang/LLVM build error: {e}")

        # report error
        sys.exit(1)
