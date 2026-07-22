# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conan recipe producing Warp's embedded LLVM/Clang CPU-JIT compiler SDK.

The recipe hardcodes Warp's flavor (clang-only static libraries with the
NVPTX backend, self-contained, size-optimized) rather than exposing the
upstream option matrix. Consumers never use Conan: the deployed tree
(see deployers/llvm_sdk.py) is the product.
"""

import os
import shutil
import subprocess

from conan import ConanFile
from conan.errors import ConanException, ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd, cross_building
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import (
    apply_conandata_patches,
    collect_libs,
    copy,
    export_conandata_patches,
    get,
    load,
    rm,
    rmdir,
    save,
)
from conan.tools.microsoft import is_msvc
from conan.tools.scm import Version

# Host tools LLVM runs mid-build; when cross-compiling these must be built
# for the build machine first and passed via LLVM_NATIVE_TOOL_DIR.
NATIVE_TABLEGEN_TARGETS = ["llvm-min-tblgen", "llvm-tblgen", "clang-tblgen"]

HOST_BACKENDS = {"x86_64": "X86", "armv8": "AArch64"}

# test_package needs CMake >= 3.24; LLVM 21 itself needs >= 3.20.
CMAKE_MIN_VERSION = (3, 24)
# Ninja 1.10 matches the upstream floor.
NINJA_MIN_VERSION = (1, 10)


class ClangWarp(ConanFile):
    name = "clang-warp"
    description = (
        "Warp's embedded LLVM/Clang CPU-JIT compiler flavor: minimal clang-only "
        "static libraries with the NVPTX backend, self-contained (no external deps)"
    )
    url = "https://github.com/NVIDIA/warp"
    homepage = "https://github.com/llvm/llvm-project"
    license = "Apache-2.0 WITH LLVM-exception"
    topics = ("llvm", "clang", "jit", "warp")

    settings = "os", "arch", "compiler", "build_type"
    options = {"targets": [None, "ANY"]}  # noqa: RUF012 -- Conan option declarations are class-level dicts
    default_options = {"targets": None}  # noqa: RUF012 -- None -> host backend + NVPTX

    no_copy_source = True

    def export_sources(self):
        export_conandata_patches(self)

    def layout(self):
        cmake_layout(self, src_folder="source")

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    @property
    def _targets(self):
        if self.options.targets:
            return str(self.options.targets)
        return f"{HOST_BACKENDS[str(self.settings.arch)]};NVPTX"

    def validate(self):
        if str(self.settings.os) not in ("Linux", "Windows", "Macos"):
            raise ConanInvalidConfiguration(f"Unsupported OS: {self.settings.os}")
        if str(self.settings.arch) not in HOST_BACKENDS:
            raise ConanInvalidConfiguration(f"Unsupported arch: {self.settings.arch}")
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, "17")

        if self.settings.os == "Linux":
            if self.settings.compiler != "gcc":
                raise ConanInvalidConfiguration("Linux SDKs must be built with GCC (the manylinux toolchain)")
            if self.settings.compiler.libcxx != "libstdc++":
                raise ConanInvalidConfiguration(
                    "Linux SDKs require compiler.libcxx=libstdc++ (pre-C++11 ABI) so the "
                    "static libraries stay linkable into manylinux wheels"
                )

        if self.settings.os == "Macos" and not self.settings.get_safe("os.version"):
            raise ConanInvalidConfiguration("macOS SDKs must pin os.version (the deployment target)")

        if self.settings.os == "Windows":
            if not is_msvc(self):
                raise ConanInvalidConfiguration("Windows SDKs must be built with MSVC")
            if self.settings.compiler.runtime != "static":
                raise ConanInvalidConfiguration("Windows SDKs must use the static CRT (compiler.runtime=static)")
            msvc_version = str(self.settings.compiler.version)
            if self.settings.arch == "x86_64" and msvc_version != "192":
                raise ConanInvalidConfiguration(
                    "windows-x86_64 SDKs must be built with the v142 toolset "
                    "(compiler.version=192): MSVC link compatibility is directional, and a "
                    "newer toolset would break Warp's documented Visual Studio 2019 floor"
                )
            if self.settings.arch == "armv8" and Version(msvc_version) < "193":
                raise ConanInvalidConfiguration("windows-arm64 SDKs require the v143 toolset or newer")

    def _check_build_tool(self, tool, min_version):
        """Fail early if a system build tool is missing or too old.

        The recipe deliberately declares no tool_requires: the container,
        runner, or developer machine provides cmake and ninja, and nothing
        is fetched from a Conan registry.
        """
        path = shutil.which(tool)
        if path is None:
            raise ConanException(f"{tool} not found on PATH; install it first (e.g. pip install {tool})")
        out = subprocess.check_output([tool, "--version"], text=True)
        digits = next((tok for tok in out.split() if tok[0].isdigit()), None)
        if digits is None:
            first_line = out.splitlines()[0] if out else ""
            raise ConanException(f"could not parse {tool} version from: {first_line!r}")
        found = tuple(int(p) for p in digits.split(".")[:2])
        if found < min_version:
            wanted = ".".join(str(p) for p in min_version)
            raise ConanException(f"{tool} {digits} is too old; need >= {wanted}")

    def generate(self):
        tc = CMakeToolchain(self, generator="Ninja")
        # These libraries ship inside Warp's PyPI wheels: approximate MinSizeRel
        # so code size wins over compiler throughput (kernels are JIT-compiled
        # once and cached). The per-config Release flags must be overridden --
        # appended global flags lose to CMAKE_<LANG>_FLAGS_RELEASE, which CMake
        # places last on the compile line.
        if self.settings.build_type == "Release":
            if is_msvc(self):
                minsize_flags = "/O1 /Ob1 /DNDEBUG"
            elif str(self.settings.compiler) in ("clang", "apple-clang"):
                minsize_flags = "-Oz -DNDEBUG"
            else:
                minsize_flags = "-Os -DNDEBUG"
            tc.cache_variables["CMAKE_C_FLAGS_RELEASE"] = minsize_flags
            tc.cache_variables["CMAKE_CXX_FLAGS_RELEASE"] = minsize_flags
        if self.settings.os == "Linux":
            # Stable GCC C++ ABI for manylinux consumers
            # (_GLIBCXX_USE_CXX11_ABI=0 comes from compiler.libcxx=libstdc++).
            tc.extra_cxxflags.append("-fabi-version=13")
        tc.generate()

    def _host_triple(self):
        if self.settings.os == "Windows" and str(self.settings.arch) == "armv8":
            return "aarch64-pc-windows-msvc"
        raise ConanInvalidConfiguration(
            f"clang-warp cross builds only support Windows armv8 targets, got {self.settings.os}/{self.settings.arch}"
        )

    def _native_tools_bindir(self):
        return os.path.join(self.build_folder, "native-tools", "bin")

    def _write_native_vcvars(self):
        """Derive a build-machine (x64 native) vcvars wrapper from the Conan-generated cross one."""
        cross_vcvars = os.path.join(self.generators_folder, "conanvcvars.bat")
        if not os.path.exists(cross_vcvars):
            raise ConanException(
                "conanvcvars.bat not found; cross builds require an MSVC profile pair (see tools/llvm/README.md)"
            )
        content = load(self, cross_vcvars)
        if "amd64_arm64" not in content:
            raise ConanException(
                "conanvcvars.bat does not select the amd64_arm64 cross toolchain; "
                "cannot derive the native x64 environment for the tablegen pass"
            )
        # Clear VSCMD_VER so vcvarsall fully re-initializes instead of
        # refusing to run inside an already-configured environment.
        native = "set VSCMD_VER=\n" + content.replace("amd64_arm64", "amd64")
        native_vcvars = os.path.join(self.generators_folder, "conanvcvars_native.bat")
        save(self, native_vcvars, native)
        return native_vcvars

    def _build_native_tablegens(self):
        """Pass 1 of a cross build: build the tablegen tools for the build machine."""
        self.output.highlight("Cross build detected: building native tablegen tools (pass 1)")
        native_vcvars = self._write_native_vcvars()
        source = os.path.join(self.source_folder, "llvm")
        build_dir = os.path.join(self.build_folder, "native-tools")
        configure = (
            f'cmake -S "{source}" -B "{build_dir}" -G Ninja'
            " -DCMAKE_BUILD_TYPE=Release"
            " -DLLVM_ENABLE_PROJECTS=clang"
            " -DLLVM_TARGETS_TO_BUILD=X86"
            " -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF"
        )
        build = f'cmake --build "{build_dir}" --target {" ".join(NATIVE_TABLEGEN_TARGETS)}'
        # env=None skips the conanbuild wrapper (which selects the arm64 cross
        # toolchain); the derived native vcvars provides the x64 environment.
        self.run(f'"{native_vcvars}" && {configure}', env=None)
        self.run(f'"{native_vcvars}" && {build}', env=None)

    def _configured_cmake(self):
        cmake = CMake(self)
        definitions = {
            # Warp's flavor: clang only, static libs, no tools, no external deps.
            "LLVM_ENABLE_PROJECTS": "clang",
            "LLVM_TARGETS_TO_BUILD": self._targets,
            "LLVM_TARGET_ARCH": "host",
            "LLVM_ENABLE_PIC": True,
            "LLVM_ENABLE_EH": False,
            "LLVM_ENABLE_RTTI": False,
            "LLVM_ENABLE_THREADS": True,
            "LLVM_ENABLE_ZLIB": False,
            "LLVM_ENABLE_ZSTD": False,
            "LLVM_ENABLE_LIBXML2": False,
            "LLVM_ENABLE_CURL": False,
            "LLVM_ENABLE_HTTPLIB": False,
            "LLVM_ENABLE_FFI": False,
            "LLVM_ENABLE_Z3_SOLVER": False,
            "LLVM_ENABLE_LIBEDIT": False,
            "LLVM_ENABLE_LIBPFM": False,
            "LLVM_ENABLE_BINDINGS": False,
            "LLVM_ENABLE_PLUGINS": False,
            "LLVM_ENABLE_ASSERTIONS": self.settings.build_type == "Debug",
            # Security: no git repository info embedded in binaries.
            "LLVM_APPEND_VC_REV": False,
            "LLVM_INCLUDE_TOOLS": True,  # needed by clang's build
            "LLVM_INCLUDE_TESTS": False,
            "LLVM_INCLUDE_EXAMPLES": False,
            "LLVM_INCLUDE_BENCHMARKS": False,
            "LLVM_INCLUDE_DOCS": False,
            "LLVM_INCLUDE_RUNTIMES": False,
            "LLVM_INCLUDE_UTILS": False,
            "LLVM_BUILD_TOOLS": False,
            "LLVM_BUILD_UTILS": False,
            "LLVM_BUILD_RUNTIME": False,
            "LLVM_BUILD_RUNTIMES": False,
            "LLVM_BUILD_LLVM_C_DYLIB": False,
            "LLVM_TOOL_LTO_BUILD": False,
            "LLVM_TOOL_REMARKS_SHLIB_BUILD": False,
            "CLANG_BUILD_TOOLS": False,
            "CLANG_PLUGIN_SUPPORT": False,
            "CLANG_ENABLE_STATIC_ANALYZER": False,
            "CLANG_TOOL_LIBCLANG_BUILD": False,
            "CLANG_TOOL_C_INDEX_TEST_BUILD": False,
            # Skip the ~130 MB libclang-cpp dylib Warp never links.
            "CLANG_TOOL_CLANG_SHLIB_BUILD": False,
        }
        if is_msvc(self):
            definitions["LLVM_ENABLE_DIA_SDK"] = False
        if cross_building(self):
            definitions["LLVM_NATIVE_TOOL_DIR"] = self._native_tools_bindir().replace("\\", "/")
            definitions["LLVM_HOST_TRIPLE"] = self._host_triple()
        else:
            # Speeds the build up noticeably; needs a runnable tablegen, so
            # host builds only.
            definitions["LLVM_OPTIMIZED_TABLEGEN"] = True
        cmake.configure(
            variables=definitions,
            build_script_folder=os.path.join(self.source_folder, "llvm"),
        )
        return cmake

    def build(self):
        for tool, min_version in (("cmake", CMAKE_MIN_VERSION), ("ninja", NINJA_MIN_VERSION)):
            self._check_build_tool(tool, min_version)
        apply_conandata_patches(self)
        if cross_building(self):
            self._build_native_tablegens()
        cmake = self._configured_cmake()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        for project in ("llvm", "clang"):
            copy(
                self,
                "LICENSE.TXT",
                src=os.path.join(self.source_folder, project),
                dst=os.path.join(self.package_folder, "licenses", project),
                keep_path=False,
            )
        # Nothing in bin/, share/, libexec/, or LLVM's CMake package files is
        # part of the SDK; Warp links the static libraries directly.
        for subdir in ("bin", "share", "libexec", os.path.join("lib", "cmake")):
            rmdir(self, os.path.join(self.package_folder, subdir))
        # Clang builds some subsystems unconditionally (no CMake option skips
        # compiling them). The static analyzer archives were verified unused by
        # removing them and relinking warp-clang: the binary comes out
        # byte-identical, i.e. the linker pulls zero objects from them. They
        # are pure weight in the SDK archive (~28 MB Linux / ~42 MB Windows).
        rm(self, "*clangStaticAnalyzer*", os.path.join(self.package_folder, "lib"))

    def package_info(self):
        self.cpp_info.libs = collect_libs(self)
        if self.settings.os == "Linux":
            self.cpp_info.system_libs = ["pthread", "dl", "rt", "m"]
            # Consumers must compile against the headers with the same ABI the
            # archives were built with.
            self.cpp_info.defines = ["_GLIBCXX_USE_CXX11_ABI=0"]
        elif self.settings.os == "Windows":
            self.cpp_info.system_libs = [
                "version",
                "ws2_32",
                "ntdll",
                "advapi32",
                "ole32",
                "psapi",
                "shell32",
                "uuid",
            ]
