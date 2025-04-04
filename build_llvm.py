# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to build Clang/LLVM from source and to build the CPU-only Warp library."""

import os
import subprocess
import sys

from warp.build_dll import *

# set build output path off this file
base_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(base_path, "warp")

llvm_project_path = os.path.join(base_path, "external/llvm-project")
llvm_build_path = os.path.join(llvm_project_path, "out/build/")
llvm_install_path = os.path.join(llvm_project_path, "out/install/")


# Fetch prebuilt Clang/LLVM libraries
def fetch_prebuilt_libraries(arch):
    if os.name == "nt":
        packman = "tools\\packman\\packman.cmd"
        packages = {"x86_64": "15.0.7-windows-x86_64-ptx-vs142"}
    else:
        packman = "./tools/packman/packman"
        if sys.platform == "darwin":
            packages = {
                "aarch64": "15.0.7-darwin-aarch64-macos11",
                "x86_64": "15.0.7-darwin-x86_64-macos11",
            }
        else:
            packages = {
                "aarch64": "15.0.7-linux-aarch64-gcc7.5",
                "x86_64": "18.1.3-linux-x86_64-gcc9.4",
            }

    try:
        subprocess.check_output(
            [
                packman,
                "install",
                "-l",
                f"./_build/host-deps/llvm-project/release-{arch}",
                "clang+llvm-warp",
                packages[arch],
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise e


def build_llvm_clang_from_source_for_arch(args, arch: str, llvm_source: str) -> None:
    """Build Clang/LLVM from source for a given architecture.

    Args:
        args: Command line arguments
        arch: Architecture to build for ("aarch64" or "x86_64")
        llvm_source: Path to the LLVM source code
    """

    # Check out the LLVM project Git repository, unless it already exists
    if not os.path.exists(llvm_source):
        # Install dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cmake"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja"])

        from git import Repo

        repo_url = "https://github.com/llvm/llvm-project.git"
        print(f"Cloning LLVM project from {repo_url}...")

        shallow_clone = True  # https://github.blog/2020-12-21-get-up-to-speed-with-partial-clone-and-shallow-clone/
        version = "18.1.3"
        if shallow_clone:
            repo = Repo.clone_from(
                repo_url,
                to_path=llvm_source,
                single_branch=True,
                branch=f"llvmorg-{version}",
                depth=1,
            )
        else:
            repo = Repo.clone_from(repo_url, to_path=llvm_source)
            repo.git.checkout(f"tags/llvmorg-{version}", "-b", f"llvm-{version}")

    print(f"Using LLVM project source from {llvm_source}")

    # CMake supports Debug, Release, RelWithDebInfo, and MinSizeRel builds
    if args.mode == "release":
        msvc_runtime = "MultiThreaded"
        # prefer smaller size over aggressive speed
        cmake_build_type = "MinSizeRel"
    else:
        msvc_runtime = "MultiThreadedDebug"
        # When args.mode == "debug" we build a Debug version of warp.dll but
        # we generally don't want warp-clang.dll to be a slow Debug version.
        if args.debug_llvm:
            cmake_build_type = "Debug"
        else:
            # The GDB/LLDB debugger observes the __jit_debug_register_code symbol
            # defined by the LLVM JIT, for which it needs debug info.
            cmake_build_type = "RelWithDebInfo"

    # Location of cmake and ninja installed through pip (see build.bat / build.sh)
    python_bin = "python/Scripts" if sys.platform == "win32" else "python/bin"
    os.environ["PATH"] = os.path.join(base_path, "_build/target-deps/" + python_bin) + os.pathsep + os.environ["PATH"]

    if arch == "aarch64":
        target_backend = "AArch64"
    else:
        target_backend = "X86"

    if sys.platform == "darwin":
        host_triple = f"{arch}-apple-macos11"
        osx_architectures = arch  # build one architecture only
        abi_version = ""
    elif os.name == "nt":
        host_triple = f"{arch}-pc-windows"
        osx_architectures = ""
        abi_version = ""
    else:
        host_triple = f"{arch}-pc-linux"
        osx_architectures = ""
        abi_version = "-fabi-version=13"  # GCC 8.2+

    llvm_path = os.path.join(llvm_source, "llvm")
    build_path = os.path.join(llvm_build_path, f"{args.mode}-{arch}")
    install_path = os.path.join(llvm_install_path, f"{args.mode}-{arch}")

    # Build LLVM and Clang
    # fmt: off
    cmake_gen = [
        "cmake",
        "-S", llvm_path,
        "-B", build_path,
        "-G", "Ninja",
        "-D", f"CMAKE_BUILD_TYPE={cmake_build_type}",
        "-D", f"CMAKE_MSVC_RUNTIME_LIBRARY={msvc_runtime}",
        "-D", f"LLVM_TARGETS_TO_BUILD={target_backend};NVPTX",
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
        "-D", f"CMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 {abi_version}",  # The pre-C++11 ABI is still the default on the CentOS 7 toolchain
        "-D", f"CMAKE_INSTALL_PREFIX={install_path}",
        "-D", f"LLVM_HOST_TRIPLE={host_triple}",
        "-D", f"CMAKE_OSX_ARCHITECTURES={osx_architectures}",

        # Disable unused tools and features
        "-D", "CLANG_BUILD_TOOLS=FALSE",
        "-D", "LLVM_ENABLE_PLUGINS=FALSE",
        "-D", "CLANG_PLUGIN_SUPPORT=FALSE",
        "-D", "CLANG_ENABLE_ARCMT=FALSE",
        "-D", "CLANG_ENABLE_STATIC_ANALYZER=FALSE",
        "-D", "CLANG_TOOLING_BUILD_AST_INTROSPECTION=FALSE",
        "-D", "CLANG_TOOL_AMDGPU_ARCH_BUILD=FALSE",
        "-D", "CLANG_TOOL_APINOTES_TEST_BUILD=FALSE",
        "-D", "CLANG_TOOL_ARCMT_TEST_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_CHECK_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_DIFF_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_EXTDEF_MAPPING_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_FORMAT_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_FORMAT_VS_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_FUZZER_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_IMPORT_TEST_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_LINKER_WRAPPER_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_NVLINK_WRAPPER_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_OFFLOAD_BUNDLER_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_OFFLOAD_PACKAGER_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_OFFLOAD_WRAPPER_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_REFACTOR_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_RENAME_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_REPL_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_SCAN_DEPS_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_SHLIB_BUILD=FALSE",
        "-D", "CLANG_TOOL_C_ARCMT_TEST_BUILD=FALSE",
        "-D", "CLANG_TOOL_C_INDEX_TEST_BUILD=FALSE",
        "-D", "CLANG_TOOL_DIAGTOOL_BUILD=FALSE",
        "-D", "CLANG_TOOL_DRIVER_BUILD=FALSE",
        "-D", "CLANG_TOOL_LIBCLANG_BUILD=FALSE",
        "-D", "CLANG_TOOL_SCAN_BUILD_BUILD=FALSE",
        "-D", "CLANG_TOOL_SCAN_BUILD_PY_BUILD=FALSE",
        "-D", "CLANG_TOOL_CLANG_OFFLOAD_BUNDLER_BUILD=FALSE",
        "-D", "CLANG_TOOL_SCAN_VIEW_BUILD=FALSE",
        "-D", "LLVM_ENABLE_BINDINGS=FALSE",
        "-D", "LLVM_ENABLE_OCAMLDOC=FALSE",
        "-D", "LLVM_TOOL_BUGPOINT_BUILD=FALSE",
        "-D", "LLVM_TOOL_BUGPOINT_PASSES_BUILD=FALSE",
        "-D", "LLVM_TOOL_CLANG_BUILD=FALSE",
        "-D", "LLVM_TOOL_DSYMUTIL_BUILD=FALSE",
        "-D", "LLVM_TOOL_DXIL_DIS_BUILD=FALSE",
        "-D", "LLVM_TOOL_GOLD_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLC_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLDB_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLI_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_AR_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_AS_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_AS_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_BCANALYZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_CAT_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_CFI_VERIFY_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_CONFIG_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_COV_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_CVTRES_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_CXXDUMP_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_CXXFILT_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_CXXMAP_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_C_TEST_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DEBUGINFOD_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DEBUGINFOD_FIND_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DIFF_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DIS_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DIS_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DLANG_DEMANGLE_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DWARFDUMP_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DWARFUTIL_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_DWP_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_EXEGESIS_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_EXTRACT_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_GO_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_GSYMUTIL_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_IFS_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_ISEL_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_ITANIUM_DEMANGLE_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_JITLINK_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_JITLISTENER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_LIBTOOL_DARWIN_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_LINK_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_LIPO_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_LTO2_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_LTO_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_MCA_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_MC_ASSEMBLE_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_MC_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_MC_DISASSEMBLE_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_MICROSOFT_DEMANGLE_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_ML_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_MODEXTRACT_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_MT_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_NM_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_OBJCOPY_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_OBJDUMP_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_OPT_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_OPT_REPORT_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_PDBUTIL_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_PROFDATA_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_PROFGEN_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_RC_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_READOBJ_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_REDUCE_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_REMARK_SIZE_DIFF_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_RTDYLD_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_RUST_DEMANGLE_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_SHLIB_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_SIM_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_SIZE_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_SPECIAL_CASE_LIST_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_SPLIT_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_STRESS_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_STRINGS_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_SYMBOLIZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_TAPI_DIFF_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_TLI_CHECKER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_UNDNAME_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_XRAY_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_YAML_NUMERIC_PARSER_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LLVM_YAML_PARSER_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_LTO_BUILD=FALSE",
        "-D", "LLVM_TOOL_OBJ2YAML_BUILD=FALSE",
        "-D", "LLVM_TOOL_OPT_BUILD=FALSE",
        "-D", "LLVM_TOOL_OPT_VIEWER_BUILD=FALSE",
        "-D", "LLVM_TOOL_REMARKS_SHLIB_BUILD=FALSE",
        "-D", "LLVM_TOOL_SANCOV_BUILD=FALSE",
        "-D", "LLVM_TOOL_SANSTATS_BUILD=FALSE",
        "-D", "LLVM_TOOL_SPLIT_FILE_BUILD=FALSE",
        "-D", "LLVM_TOOL_VERIFY_USELISTORDER_BUILD=FALSE",
        "-D", "LLVM_TOOL_VFABI_DEMANGLE_FUZZER_BUILD=FALSE",
        "-D", "LLVM_TOOL_XCODE_TOOLCHAIN_BUILD=FALSE",
        "-D", "LLVM_TOOL_YAML2OBJ_BUILD=FALSE",
    ]
    # fmt: on
    subprocess.check_call(cmake_gen, stderr=subprocess.STDOUT)

    cmake_build = ["cmake", "--build", build_path]
    subprocess.check_call(cmake_build, stderr=subprocess.STDOUT)

    cmake_install = ["cmake", "--install", build_path]
    subprocess.check_call(cmake_install, stderr=subprocess.STDOUT)


def build_llvm_clang_from_source(args) -> None:
    """Build Clang/LLVM from source."""

    print("Building Clang/LLVM from source...")

    if args.llvm_source_path is not None:
        llvm_source = args.llvm_source_path
    else:
        llvm_source = llvm_project_path

    # build for the machine's architecture
    build_llvm_clang_from_source_for_arch(args, machine_architecture(), llvm_source)

    # for Apple systems also cross-compile for building a universal binary
    if sys.platform == "darwin":
        if machine_architecture() == "x86_64":
            build_llvm_clang_from_source_for_arch(args, "aarch64", llvm_source)
        else:
            build_llvm_clang_from_source_for_arch(args, "x86_64", llvm_source)


# build warp-clang.dll
def build_warp_clang_for_arch(args, lib_name: str, arch: str) -> None:
    try:
        cpp_sources = [
            "native/clang/clang.cpp",
            "native/crt.cpp",
        ]
        clang_cpp_paths = [os.path.join(build_path, cpp) for cpp in cpp_sources]

        clang_dll_path = os.path.join(build_path, f"bin/{lib_name}")

        if args.build_llvm:
            # obtain Clang and LLVM libraries from the local build
            install_path = os.path.join(llvm_install_path, f"{args.mode}-{arch}")
            libpath = os.path.join(install_path, "lib")
        else:
            # obtain Clang and LLVM libraries from packman
            fetch_prebuilt_libraries(arch)
            libpath = os.path.join(base_path, f"_build/host-deps/llvm-project/release-{arch}/lib")

        libs = []

        for _, _, libraries in os.walk(libpath):
            libs.extend(libraries)
            break  # just the top level contains library files

        if os.name == "nt":
            libs.append("Version.lib")
            libs.append("Ws2_32.lib")
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
            if sys.platform != "darwin":
                libs.append("-lrt")

        build_dll_for_arch(
            args,
            dll_path=clang_dll_path,
            cpp_paths=clang_cpp_paths,
            cu_path=None,
            arch=arch,
            libs=libs,
            mode=args.mode if args.build_llvm else "release",
        )

    except Exception as e:
        # output build error
        print(f"Warp Clang/LLVM build error: {e}")

        # report error
        sys.exit(1)


def build_warp_clang(args, lib_name: str) -> None:
    """Build the CPU-only Warp library using Clang/LLVM."""

    if sys.platform == "darwin":
        # create a universal binary by combining x86-64 and AArch64 builds
        build_warp_clang_for_arch(args, lib_name + "-x86_64", "x86_64")
        build_warp_clang_for_arch(args, lib_name + "-aarch64", "aarch64")

        dylib_path = os.path.join(build_path, f"bin/{lib_name}")
        run_cmd(f"lipo -create -output {dylib_path} {dylib_path}-x86_64 {dylib_path}-aarch64")
        os.remove(f"{dylib_path}-x86_64")
        os.remove(f"{dylib_path}-aarch64")

    else:
        build_warp_clang_for_arch(args, lib_name, machine_architecture())
