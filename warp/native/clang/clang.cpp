// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../native/crt.h"
#include "../version.h"
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#if LLVM_VERSION_MAJOR >= 18
#include <llvm/Frontend/Debug/Options.h>
#else
#include <llvm/Support/CodeGen.h>
#endif
#if LLVM_VERSION_MAJOR == 21
#include <llvm/Support/VirtualFileSystem.h>
#endif
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h>
#include <llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/InitializePasses.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#if LLVM_VERSION_MAJOR >= 16
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif

#if defined(_WIN64)
extern "C" void __chkstk();
#elif defined(__APPLE__)
extern "C" void _bzero(void* s, size_t n) { memset(s, 0, n); }
extern "C" void __bzero(void* s, size_t n) { memset(s, 0, n); }
extern "C" void _memset_pattern16(void* s, const void* pattern, size_t n);
extern "C" void __memset_pattern16(void* s, const void* pattern, size_t n);
extern "C" __double2 __sincos_stret(double);
extern "C" __float2 __sincosf_stret(float);
#endif  // defined(__APPLE__)

extern "C" {

// GDB and LLDB support debugging of JIT-compiled code by observing calls to __jit_debug_register_code()
// by putting a breakpoint on it, and retrieving the debug info through __jit_debug_descriptor.
// On Linux it suffices for these symbols not to be stripped out, while for Windows a .pdb has to contain
// their information. LLVM defines them, but we don't want a huge .pdb with all LLVM source code's debug
// info. By forward-declaring them here it suffices to compile this file with /Zi.
extern struct jit_descriptor __jit_debug_descriptor;
extern void __jit_debug_register_code();
}

namespace wp {

#if defined(_WIN32)
// Windows defaults to using the COFF binary format (aka. "msvc" in the target triple).
// Override it to use the ELF format to support DWARF debug info, but keep using the
// Microsoft calling convention (see also https://llvm.org/docs/DebuggingJITedCode.html).
static const char* target_triple = "x86_64-pc-windows-elf";
#else
static const char* target_triple = LLVM_DEFAULT_TARGET_TRIPLE;
#endif

// Minimum CUDA compute capability that supports all of Warp's features.
// Since we always emit PTX (forward-compatible), targeting the minimum
// ensures the output runs on all supported GPUs.
static const char* cuda_target_arch = "sm_75";

static void initialize_llvm()
{
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
}

struct HostCpuInfo {
    std::string name;  // e.g. "znver5", "apple-m2", or "generic"
    std::string features;  // comma-separated, for TargetMachine: "+avx2,+fma,..."
    std::vector<std::string> feature_list;  // individual flags, for -target-feature args
};

// Thread-safe: C++11 guarantees local static initialization is synchronized.
static const HostCpuInfo& get_host_cpu_info()
{
    static HostCpuInfo info = []() {
        HostCpuInfo result;
        result.name = llvm::sys::getHostCPUName().str();

        llvm::StringMap<bool> feature_map;
        llvm::sys::getHostCPUFeatures(feature_map);

        for (const auto& f : feature_map) {
            // Skip avx10.1-256: getHostCPUFeatures() reports it on Intel Granite Rapids,
            // but Clang treats it as an invalid feature combination and warns that it
            // will be promoted to avx10.1-512 (one warning per compile).
            if (f.first() == "avx10.1-256") {
                continue;
            }
            std::string flag = (f.second ? "+" : "-") + f.first().str();
            result.feature_list.push_back(flag);
            if (!result.features.empty())
                result.features += ",";
            result.features += flag;
        }

        return result;
    }();
    return info;
}

static std::unique_ptr<clang::CompilerInstance> create_compiler(
    const std::string& input_file,
    const char* include_dir,
    bool is_cuda,
    bool debug,
    bool verify_fp,
    bool tiles_in_stack_memory,
    const char** extra_flags = nullptr,  // null-terminated array of flag strings, or nullptr for none
    int optimization_level = 3
)
{
    auto compiler_instance = std::make_unique<clang::CompilerInstance>();
    // Compilation arguments
    std::vector<const char*> args;
    args.push_back(input_file.c_str());

    args.push_back("-I");
    args.push_back(include_dir);

    if (debug) {
        args.push_back("-O0");
    } else {
        switch (optimization_level) {
        case 0:
            args.push_back("-O0");
            break;
        case 1:
            args.push_back("-O1");
            break;
        case 2:
            args.push_back("-O2");
            break;
        default:
            args.push_back("-O3");
            break;
        }
    }

    if (is_cuda) {
        args.push_back("-triple");
        args.push_back("nvptx64-nvidia-cuda");

        args.push_back("-target-cpu");
        args.push_back(cuda_target_arch);
    } else {
        args.push_back("-triple");
        args.push_back(target_triple);

        // Append extra flags to args. Our "driver" expands -march=native inline
        // into -target-cpu and -target-feature flags, preserving flag order.
        // Other flags are passed through to the Clang frontend as-is.
        if (extra_flags) {
            for (const char** flag = extra_flags; *flag; ++flag) {
                if (strcmp(*flag, "-march=native") == 0) {
                    const auto& cpu = get_host_cpu_info();
                    if (cpu.name != "generic") {
                        args.push_back("-target-cpu");
                        args.push_back(cpu.name.c_str());
                    }
                    for (const auto& feat : cpu.feature_list) {
                        args.push_back("-target-feature");
                        args.push_back(feat.c_str());
                    }
                } else {
                    args.push_back(*flag);
                }
            }
        }

#if defined(__x86_64__) || defined(_M_X64)
        // F16C is required for _Float16 conversions in builtin.h. Duplicate
        // flags are harmless (last-wins semantics), so add unconditionally.
        args.push_back("-target-feature");
        args.push_back("+f16c");
#endif

#if defined(__aarch64__)
        if (tiles_in_stack_memory) {
            // Static memory support is broken on AArch64 CPUs. As a workaround we reserve some stack memory on kernel
            // entry, and point the callee-saved x28 register to it so we can access it anywhere. See
            // tile_shared_storage_t in tile.h.
            args.push_back("-target-feature");
            args.push_back("+reserve-x28");
        }
#endif
    }

#if LLVM_VERSION_MAJOR >= 21
    clang::DiagnosticOptions diagnostic_options;
    std::unique_ptr<clang::TextDiagnosticPrinter> text_diagnostic_printer
        = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), diagnostic_options);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids;
    std::unique_ptr<clang::DiagnosticsEngine> diagnostic_engine = std::make_unique<clang::DiagnosticsEngine>(
        diagnostic_ids, diagnostic_options, text_diagnostic_printer.release()
    );
#else
    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options = new clang::DiagnosticOptions();
    std::unique_ptr<clang::TextDiagnosticPrinter> text_diagnostic_printer
        = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &*diagnostic_options);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids;
    std::unique_ptr<clang::DiagnosticsEngine> diagnostic_engine = std::make_unique<clang::DiagnosticsEngine>(
        diagnostic_ids, &*diagnostic_options, text_diagnostic_printer.release()
    );
#endif

    auto& compiler_invocation = compiler_instance->getInvocation();
    clang::CompilerInvocation::CreateFromArgs(compiler_invocation, args, *diagnostic_engine);

    if (debug) {
#if LLVM_VERSION_MAJOR >= 18
        compiler_invocation.getCodeGenOpts().setDebugInfo(llvm::codegenoptions::FullDebugInfo);
#else
        compiler_invocation.getCodeGenOpts().setDebugInfo(clang::codegenoptions::FullDebugInfo);
#endif
    }

    if (!debug) {
        compiler_instance->getPreprocessorOpts().addMacroDef("NDEBUG");
    }

    if (is_cuda) {
        // According to https://llvm.org/docs/CompileCudaWithLLVM.html, "Both clang and nvcc define `__CUDACC__` during
        // CUDA compilation." But this normally happens in the __clang_cuda_runtime_wrapper.h header, which we don't
        // include. The __CUDA__ and __CUDA_ARCH__ macros are internally defined by
        // llvm-project/clang/lib/Frontend/InitPreprocessor.cpp
        compiler_instance->getPreprocessorOpts().addMacroDef("__CUDACC__");

        compiler_instance->getLangOpts().CUDA = 1;
        compiler_instance->getLangOpts().CUDAIsDevice = 1;
    } else {
        if (verify_fp) {
            compiler_instance->getPreprocessorOpts().addMacroDef("WP_VERIFY_FP");
        }

        if (tiles_in_stack_memory) {
            compiler_instance->getPreprocessorOpts().addMacroDef("WP_ENABLE_TILES_IN_STACK_MEMORY");
        }

        compiler_instance->getLangOpts().MicrosoftExt = 1;  // __forceinline / __int64
        compiler_instance->getLangOpts().DeclSpecKeyword = 1;  // __declspec
    }

    // For LLVM >= 21, transfer ownership of the DiagnosticConsumer to the
    // CompilerInstance so it outlives create_compiler's scope. First release
    // the DiagnosticsEngine's ownership to avoid a double-free.
    // For LLVM < 21, passing nullptr makes createDiagnostics create its own
    // internal printer (text_diagnostic_printer was already released into
    // diagnostic_engine above).
#if LLVM_VERSION_MAJOR >= 21
    diagnostic_engine->setClient(diagnostic_engine->getClient(), /*ShouldOwnClient=*/false);
#endif
#if LLVM_VERSION_MAJOR >= 22
    compiler_instance->createDiagnostics(diagnostic_engine->getClient(), true);
#elif LLVM_VERSION_MAJOR == 21
    compiler_instance->createDiagnostics(*llvm::vfs::getRealFileSystem(), diagnostic_engine->getClient(), true);
#else
    compiler_instance->createDiagnostics(text_diagnostic_printer.get(), false);
#endif

    return compiler_instance;
}

static bool generate_pch(
    const char* include_dir,
    const std::string& pch_path,
    bool debug,
    bool verify_fp,
    bool tiles_in_stack_memory,
    const char** extra_flags,
    bool verbose,
    int block_dim
)
{
    if (verbose) {
        std::cout << "Warp: Generating precompiled header: " << pch_path << std::endl;
    }

    std::string input_file = "pch_gen.cpp";

    auto compiler
        = create_compiler(input_file, include_dir, false, debug, verify_fp, tiles_in_stack_memory, extra_flags);

    // Create a source buffer that includes the main header.
    // WP_NO_CRT skips system headers (assert.h, math.h, etc.) which aren't
    // available in our embedded Clang — matching codegen.py's module headers.
    // WP_TILE_BLOCK_DIM must match the value used by the module to avoid
    // template instantiation mismatches in tile.h.
    std::string pch_src = "#define WP_TILE_BLOCK_DIM " + std::to_string(block_dim)
        + "\n"
          "#define WP_NO_CRT\n"
          "#include \"builtin.h\"\n";
    std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBufferCopy(pch_src);
    compiler->getInvocation().getPreprocessorOpts().addRemappedFile(input_file.c_str(), buffer.get());

    compiler->getFrontendOpts().OutputFile = pch_path;

    clang::GeneratePCHAction generate_pch_action;
    bool success = compiler->ExecuteAction(generate_pch_action);
    (void)buffer.release();

    if (success && verbose) {
        std::cout << "Warp: Precompiled header generated successfully" << std::endl;
    }

    return success;
}

static std::unique_ptr<llvm::Module> source_to_llvm(
    bool is_cuda,
    const std::string& input_file,
    const char* cpp_src,
    const char* include_dir,
    bool debug,
    bool verify_fp,
    llvm::LLVMContext& context,
    bool tiles_in_stack_memory,
    const char** extra_flags = nullptr,  // null-terminated array of flag strings, or nullptr for none
    int optimization_level = 3,
    const char* pch_path = nullptr
)
{
    auto compiler = create_compiler(
        input_file, include_dir, is_cuda, debug, verify_fp, tiles_in_stack_memory, extra_flags, optimization_level
    );

    // Map code to a MemoryBuffer
    std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBufferCopy(cpp_src);
    compiler->getInvocation().getPreprocessorOpts().addRemappedFile(input_file.c_str(), buffer.get());

    // Use precompiled header if available (CPU path only)
    if (pch_path && !is_cuda) {
        compiler->getPreprocessorOpts().ImplicitPCHInclude = pch_path;
        // Suppress macro redefinition warnings — both the PCH and the module
        // source define macros like WP_TILE_BLOCK_DIM to the same value
        // (matching block_dim), and Clang warns on any redefinition.
        compiler->getDiagnostics().setSeverityForGroup(
            clang::diag::Flavor::WarningOrError, "macro-redefined", clang::diag::Severity::Ignored
        );
    }

    clang::EmitLLVMOnlyAction emit_llvm_only_action(&context);
    bool success = compiler->ExecuteAction(emit_llvm_only_action);

    // Ownership of the buffer was transferred to the SourceManager during
    // ExecuteAction() (RetainRemappedFileBuffers defaults to false).
    // Release the unique_ptr to avoid a double-free.
    (void)buffer.release();

    return success ? std::move(emit_llvm_only_action.takeModule()) : nullptr;
}

extern "C" {

WP_API int wp_compile_cpp(
    const char* cpp_src,
    const char* input_file,
    const char* include_dir,
    const char* output_file,
    bool debug,
    bool verify_fp,
    bool fuse_fp,
    bool tiles_in_stack_memory,
    const char** extra_flags,
    int optimization_level,
    bool verbose,
    bool use_precompiled_headers,
    const char* pch_dir,
    int block_dim
)
{
    initialize_llvm();

    // Determine PCH path if requested.
    // Each block_dim value gets its own PCH file because tile.h templates
    // are instantiated with WP_TILE_BLOCK_DIM baked into the PCH.
    std::string pch_path_str;
    const char* pch_path = nullptr;
    if (use_precompiled_headers && pch_dir) {
        // Encode preprocessor-affecting flags into the filename so that
        // modules with different settings get separate PCH files.
        // Note: extra_flags are not encoded — they are assumed constant
        // within a session. If they differ, Clang rejects the PCH and
        // the fallback path handles it.
        pch_path_str = std::string(pch_dir) + "/builtin_bd" + std::to_string(block_dim) + (verify_fp ? "_vfp" : "")
            + (debug ? "_dbg" : "") + (tiles_in_stack_memory ? "_tis" : "") + ".pch";

        // Check if the PCH file already exists
        FILE* f = fopen(pch_path_str.c_str(), "rb");
        if (f) {
            fclose(f);
            if (verbose) {
                std::cout << "Warp: Using existing precompiled header: " << pch_path_str << std::endl;
            }
        } else {
            // Generate the PCH file
            if (!generate_pch(
                    include_dir, pch_path_str, debug, verify_fp, tiles_in_stack_memory, extra_flags, verbose, block_dim
                )) {
                std::cerr << "Warp: PCH generation failed, compiling without precompiled headers" << std::endl;
                remove(pch_path_str.c_str());
                pch_path_str.clear();
            }
        }

        if (!pch_path_str.empty()) {
            pch_path = pch_path_str.c_str();
        }
    }

    // Use a unique_ptr so we can replace the context on fallback retry.
    // The LLVMContext must outlive the module through codegen.
    auto llvm_context = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> module = source_to_llvm(
        false, input_file, cpp_src, include_dir, debug, verify_fp, *llvm_context, tiles_in_stack_memory, extra_flags,
        optimization_level, pch_path
    );

    // Fallback: if compilation failed with PCH, retry without it
    if (!module && pch_path) {
        std::cerr << "Warp: Compilation with PCH failed, retrying without precompiled headers" << std::endl;
        // Delete the stale PCH so subsequent calls don't hit it again
        if (remove(pch_path_str.c_str()) != 0) {
            std::cerr << "Warp: Failed to remove stale PCH file: " << pch_path_str << std::endl;
        }

        // Need a fresh LLVMContext for the retry
        llvm_context = std::make_unique<llvm::LLVMContext>();
        module = source_to_llvm(
            false, input_file, cpp_src, include_dir, debug, verify_fp, *llvm_context, tiles_in_stack_memory,
            extra_flags, optimization_level, nullptr
        );
    }

    if (!module) {
        return -1;
    }

    std::string error;
#if LLVM_VERSION_MAJOR >= 22
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(llvm::Triple(target_triple), error);
#else
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, error);
#endif

    // Check if -march=native was requested to set the backend target accordingly.
    bool use_native = false;
    if (extra_flags) {
        for (const char** flag = extra_flags; *flag; ++flag) {
            if (strcmp(*flag, "-march=native") == 0) {
                use_native = true;
                break;
            }
        }
    }

    const char* CPU = "generic";
    const char* features = "";
    if (use_native) {
        const auto& cpu = get_host_cpu_info();
        CPU = cpu.name.c_str();
        features = cpu.features.c_str();
    }
    llvm::TargetOptions target_options;
    if (fuse_fp)
        target_options.AllowFPOpFusion = llvm::FPOpFusion::Standard;
    else
        target_options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
    llvm::Reloc::Model relocation_model = llvm::Reloc::PIC_;  // Position Independent Code
    llvm::CodeModel::Model code_model = llvm::CodeModel::Large;  // Don't make assumptions about displacement sizes

#if LLVM_VERSION_MAJOR >= 18
    llvm::CodeGenOptLevel codegen_opt;
#else
    llvm::CodeGenOpt::Level codegen_opt;
#define CodeGenOptLevel CodeGenOpt
#endif
    if (debug) {
        codegen_opt = llvm::CodeGenOptLevel::None;
    } else {
        switch (optimization_level) {
        case 0:
            codegen_opt = llvm::CodeGenOptLevel::None;
            break;
        case 1:
            codegen_opt = llvm::CodeGenOptLevel::Less;
            break;
        case 2:
            codegen_opt = llvm::CodeGenOptLevel::Default;
            break;
        default:
            codegen_opt = llvm::CodeGenOptLevel::Aggressive;
            break;
        }
    }

#if LLVM_VERSION_MAJOR >= 20
    llvm::TargetMachine* target_machine = target->createTargetMachine(
        llvm::Triple(target_triple), CPU, features, target_options, relocation_model, code_model, codegen_opt
    );
#else
    llvm::TargetMachine* target_machine = target->createTargetMachine(
        target_triple, CPU, features, target_options, relocation_model, code_model, codegen_opt
    );
#endif

    module->setDataLayout(target_machine->createDataLayout());

    std::error_code error_code;
    llvm::raw_fd_ostream output(output_file, error_code, llvm::sys::fs::OF_None);

    llvm::legacy::PassManager pass_manager;
#if LLVM_VERSION_MAJOR >= 18
    llvm::CodeGenFileType file_type = llvm::CodeGenFileType::ObjectFile;
#else
    llvm::CodeGenFileType file_type = llvm::CGFT_ObjectFile;
#endif
    target_machine->addPassesToEmitFile(pass_manager, output, nullptr, file_type);

    pass_manager.run(*module);
    output.flush();

    delete target_machine;

    return 0;
}

WP_API int wp_compile_cuda(
    const char* cpp_src, const char* input_file, const char* include_dir, const char* output_file, bool debug
)
{
    initialize_llvm();

    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module
        = source_to_llvm(true, input_file, cpp_src, include_dir, debug, false, context, false);

    if (!module) {
        return -1;
    }

    std::string error;

#if LLVM_VERSION_MAJOR >= 22
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(llvm::Triple("nvptx64-nvidia-cuda"), error);
#else
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget("nvptx64-nvidia-cuda", error);
#endif
    const char* features = "+ptx75";  // Warp requires CUDA 11.5, which supports PTX ISA 7.5
    llvm::TargetOptions target_options;
    llvm::Reloc::Model relocation_model = llvm::Reloc::PIC_;
#if LLVM_VERSION_MAJOR >= 20
    llvm::TargetMachine* target_machine = target->createTargetMachine(
        llvm::Triple("nvptx64-nvidia-cuda"), cuda_target_arch, features, target_options, relocation_model
    );
#else
    llvm::TargetMachine* target_machine = target->createTargetMachine(
        "nvptx64-nvidia-cuda", cuda_target_arch, features, target_options, relocation_model
    );
#endif

    module->setDataLayout(target_machine->createDataLayout());

    // Link libdevice
    llvm::SMDiagnostic diagnostic;
    std::string libdevice_path = std::string(include_dir) + "/libdevice/libdevice.10.bc";
    std::unique_ptr<llvm::Module> libdevice(llvm::parseIRFile(libdevice_path, diagnostic, context));
    if (!libdevice) {
        return -1;
    }

    llvm::Linker linker(*module.get());
    if (linker.linkInModule(std::move(libdevice), llvm::Linker::Flags::LinkOnlyNeeded) == true) {
        return -1;
    }

    std::error_code error_code;
    llvm::raw_fd_ostream output(output_file, error_code, llvm::sys::fs::OF_None);

    llvm::legacy::PassManager pass_manager;
#if LLVM_VERSION_MAJOR >= 18
    llvm::CodeGenFileType file_type = llvm::CodeGenFileType::AssemblyFile;
#else
    llvm::CodeGenFileType file_type = llvm::CGFT_AssemblyFile;
#endif
    target_machine->addPassesToEmitFile(pass_manager, output, nullptr, file_type);

    pass_manager.run(*module);
    output.flush();

    delete target_machine;

    return 0;
}

// Two JIT instances: one for JITLink (default) and one for RTDyld (legacy).
// Both are created lazily on first use and kept alive so that modules loaded
// with either linker remain valid when the user switches between them.
static llvm::orc::LLJIT* jit_default = nullptr;
static llvm::orc::LLJIT* jit_legacy = nullptr;

// Return the JIT instance for the given linker mode, creating it if needed.
// Note: not thread-safe.  The caller (wp_load_obj) is serialized by Python's
// Module._compile / Module.load, but if parallel loading is ever introduced at
// the C++ level a mutex would be needed here.
static llvm::orc::LLJIT* get_or_create_jit(bool use_legacy_linker)
{
    if (use_legacy_linker && jit_legacy)
        return jit_legacy;
    if (!use_legacy_linker && jit_default)
        return jit_default;

    initialize_llvm();

    llvm::orc::LLJITBuilder builder;

    if (use_legacy_linker) {
        builder.setObjectLinkingLayerCreator(
#if LLVM_VERSION_MAJOR >= 21
            [](llvm::orc::ExecutionSession& session)
#else
            [](llvm::orc::ExecutionSession& session, const llvm::Triple& triple)
#endif
                -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
#if LLVM_VERSION_MAJOR >= 21
                auto get_memory_manager = [](const llvm::MemoryBuffer&) {
#else
                auto get_memory_manager = []() {
#endif
                    return std::make_unique<llvm::SectionMemoryManager>();
                };
                auto layer
                    = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, std::move(get_memory_manager));

                layer->registerJITEventListener(*llvm::JITEventListener::createGDBRegistrationListener());

                // Make sure the debug info sections aren't stripped.
                layer->setProcessAllSections(true);

                return layer;
            }
        );
    } else {
        builder.setObjectLinkingLayerCreator(
#if LLVM_VERSION_MAJOR >= 21
            [](llvm::orc::ExecutionSession& session)
#else
            [](llvm::orc::ExecutionSession& session, const llvm::Triple& triple)
#endif
                -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
                auto layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);

                if (WP_ENABLE_DEBUG) {
                    // Register debug-object plugin for GDB/LLDB JIT debugging support.
                    auto registrar = llvm::orc::createJITLoaderGDBRegistrar(session);
                    if (registrar) {
#if LLVM_VERSION_MAJOR >= 21
                        layer->addPlugin(
                            std::make_shared<llvm::orc::DebugObjectManagerPlugin>(
                                session, std::move(*registrar), true, true
                            )
                        );
#else
                        layer->addPlugin(
                            std::make_unique<llvm::orc::DebugObjectManagerPlugin>(session, std::move(*registrar))
                        );
#endif
                    } else {
                        llvm::consumeError(registrar.takeError());
                        std::cout << "Warp notice: JIT debug support is not available with "
                                     "this LLVM build. Step-through debugging of CPU kernels "
                                     "requires building Warp with --build-llvm, or setting "
                                     "wp.config.legacy_cpu_linker = True to use the legacy "
                                     "RTDyld linker."
                                  << std::endl;
                    }
                }

                return layer;
            }
        );
    }

    auto jit_expected = builder.create();

    if (!jit_expected) {
        std::cerr << "Failed to create JIT instance: " << toString(jit_expected.takeError()) << std::endl;
        return nullptr;
    }

    auto* jit = (*jit_expected).release();
    if (use_legacy_linker)
        jit_legacy = jit;
    else
        jit_default = jit;
    return jit;
}

// Find which JIT instance owns a module by name.
static llvm::orc::LLJIT* find_jit_for_module(const char* module_name)
{
    if (jit_default && jit_default->getJITDylibByName(module_name))
        return jit_default;
    if (jit_legacy && jit_legacy->getJITDylibByName(module_name))
        return jit_legacy;
    return nullptr;
}

// Load an object file into an in-memory DLL named `module_name`.
// When `use_legacy_linker` is true, the legacy RTDyld linker is used instead
// of JITLink; this provides debug support with pre-built LLVM but is less
// robust against virtual address space fragmentation.
// Debug support (GDB/LLDB step-through) is enabled automatically in debug
// builds (WP_ENABLE_DEBUG=1).
WP_API int wp_load_obj(const char* object_file, const char* module_name, bool use_legacy_linker)
{
    auto* jit = get_or_create_jit(use_legacy_linker);
    if (!jit)
        return -1;

    auto dll = jit->createJITDylib(module_name);

    if (!dll) {
        std::cerr << "Failed to create JITDylib: " << toString(dll.takeError()) << std::endl;
        return -1;
    }

    // Define symbols for Warp's CRT functions subset
    {
#if defined(__APPLE__)
#define MANGLING_PREFIX "_"
#else
#define MANGLING_PREFIX ""
#endif

        const auto flags = llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute;
#if LLVM_VERSION_MAJOR >= 18
#define SYMBOL(sym) { jit->getExecutionSession().intern(MANGLING_PREFIX #sym), { llvm::orc::ExecutorAddr::fromPtr(&::sym), flags} }
#define SYMBOL_T(sym, T) { jit->getExecutionSession().intern(MANGLING_PREFIX #sym), { llvm::orc::ExecutorAddr::fromPtr(static_cast<T>(&::sym)), flags} }

        auto error = dll->define(llvm::orc::absoluteSymbols(llvm::orc::SymbolMap({
#else
#define SYMBOL(sym) { jit->getExecutionSession().intern(MANGLING_PREFIX #sym), { llvm::pointerToJITTargetAddress(&::sym), flags} }
#define SYMBOL_T(sym, T) { jit->getExecutionSession().intern(MANGLING_PREFIX #sym), { llvm::pointerToJITTargetAddress(static_cast<T>(&::sym)), flags} }

        auto error = dll->define(llvm::orc::absoluteSymbols({
#endif
            SYMBOL(printf), SYMBOL(puts), SYMBOL(putchar), SYMBOL_T(abs, int (*)(int)), SYMBOL(llabs), SYMBOL(fmodf),
                SYMBOL_T(fmod, double (*)(double, double)), SYMBOL(logf), SYMBOL_T(log, double (*)(double)),
                SYMBOL(log2f), SYMBOL_T(log2, double (*)(double)), SYMBOL(log10f), SYMBOL_T(log10, double (*)(double)),
                SYMBOL(expf), SYMBOL_T(exp, double (*)(double)), SYMBOL(sqrtf), SYMBOL_T(sqrt, double (*)(double)),
                SYMBOL(cbrtf), SYMBOL_T(cbrt, double (*)(double)), SYMBOL(powf),
                SYMBOL_T(pow, double (*)(double, double)), SYMBOL(floorf), SYMBOL_T(floor, double (*)(double)),
                SYMBOL(ceilf), SYMBOL_T(ceil, double (*)(double)), SYMBOL(fabsf), SYMBOL_T(fabs, double (*)(double)),
                SYMBOL(roundf), SYMBOL_T(round, double (*)(double)), SYMBOL(truncf),
                SYMBOL_T(trunc, double (*)(double)), SYMBOL(rintf), SYMBOL_T(rint, double (*)(double)), SYMBOL(acosf),
                SYMBOL_T(acos, double (*)(double)), SYMBOL(asinf), SYMBOL_T(asin, double (*)(double)), SYMBOL(atanf),
                SYMBOL_T(atan, double (*)(double)), SYMBOL(atan2f), SYMBOL_T(atan2, double (*)(double, double)),
                SYMBOL(cosf), SYMBOL_T(cos, double (*)(double)), SYMBOL(sinf), SYMBOL_T(sin, double (*)(double)),
                SYMBOL(tanf), SYMBOL_T(tan, double (*)(double)), SYMBOL(sinhf), SYMBOL_T(sinh, double (*)(double)),
                SYMBOL(coshf), SYMBOL_T(cosh, double (*)(double)), SYMBOL(tanhf), SYMBOL_T(tanh, double (*)(double)),
                SYMBOL(fmaf), SYMBOL_T(fma, double (*)(double, double, double)), SYMBOL(erff),
                SYMBOL_T(erf, double (*)(double)), SYMBOL(erfcf), SYMBOL_T(erfc, double (*)(double)), SYMBOL(erfinvf),
                SYMBOL_T(erfinv, double (*)(double)), SYMBOL(erfcinvf), SYMBOL_T(erfcinv, double (*)(double)),
                SYMBOL(memcpy), SYMBOL(memset), SYMBOL(memmove), SYMBOL(_wp_assert), SYMBOL(_wp_isfinite),
                SYMBOL(_wp_isnan), SYMBOL(_wp_isinf),
#if defined(_WIN64)
                // For functions with large stack frames the compiler will emit a call to
                // __chkstk() to linearly touch each memory page. This grows the stack without
                // triggering the stack overflow guards.
                SYMBOL(__chkstk),
#elif defined(__APPLE__)
            SYMBOL(bzero),
            SYMBOL(_bzero),
            SYMBOL(memset_pattern16),
            SYMBOL(__sincos_stret), SYMBOL(__sincosf_stret),
#else
            SYMBOL(sincosf), SYMBOL_T(sincos, void (*)(double, double*, double*)),
#endif
#if LLVM_VERSION_MAJOR >= 18
        })));
#else
        }));
#endif

        if (error) {
            std::cerr << "Failed to define symbols: " << llvm::toString(std::move(error)) << std::endl;
            return -1;
        }
    }

    // Load the object file into a memory buffer
    auto buffer = llvm::MemoryBuffer::getFile(object_file);
    if (!buffer) {
        std::cerr << "Failed to load object file: " << buffer.getError().message() << std::endl;
        return -1;
    }

    auto err = jit->addObjectFile(*dll, std::move(*buffer));
    if (err) {
        std::cerr << "Failed to add object file: " << llvm::toString(std::move(err)) << std::endl;
        return -1;
    }

    return 0;
}

WP_API int wp_unload_obj(const char* module_name)
{
    auto* jit = find_jit_for_module(module_name);
    if (!jit)
        return 0;

    auto* dll = jit->getJITDylibByName(module_name);
    llvm::Error error = jit->getExecutionSession().removeJITDylib(*dll);

    if (error) {
        std::cerr << "Failed to unload: " << llvm::toString(std::move(error)) << std::endl;
        return -1;
    }

    return 0;
}

WP_API uint64_t wp_lookup(const char* dll_name, const char* function_name)
{
    auto* jit = find_jit_for_module(dll_name);
    if (!jit) {
        std::cerr << "Failed to find module: " << dll_name << std::endl;
        return 0;
    }

    auto* dll = jit->getJITDylibByName(dll_name);

    auto func = jit->lookup(*dll, function_name);

    if (!func) {
        std::cerr << "Failed to lookup symbol: " << llvm::toString(func.takeError()) << std::endl;
        return 0;
    }

    return func->getValue();
}

WP_API const char* wp_warp_clang_version() { return WP_VERSION_STRING; }

WP_API const char* wp_llvm_version()
{
    static char version[64];
    snprintf(version, sizeof(version), "%d.%d.%d", LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR, LLVM_VERSION_PATCH);
    return version;
}

WP_API const char* wp_get_host_cpu_name() { return get_host_cpu_info().name.c_str(); }

WP_API const char* wp_get_host_cpu_features()
{
    // Build a comma-separated string of only the *enabled* features.
    // cpu.feature_list includes both +enabled and -disabled flags;
    // we filter to only the enabled ones and strip the leading '+'.
    static std::string enabled_features = []() {
        const auto& cpu = get_host_cpu_info();
        std::string result;
        for (const auto& flag : cpu.feature_list) {
            if (!flag.empty() && flag[0] == '+') {
                if (!result.empty())
                    result += ",";
                result += flag.substr(1);  // strip leading '+'
            }
        }
        return result;
    }();
    return enabled_features.c_str();
}

}  // extern "C"

}  // namespace wp
