/** Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "../native/crt.h"

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Lex/PreprocessorOptions.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/PassRegistry.h>
#include <llvm/InitializePasses.h>
#include <llvm/IR/LegacyPassManager.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <vector>
#include <iostream>
#include <string>
#include <cstring>

namespace wp {

static void initialize_llvm()
{
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
}

static std::unique_ptr<llvm::Module> cpp_to_llvm(const std::string& input_file, const char* cpp_src, const char* include_dir, llvm::LLVMContext& context)
{
    // Compilation arguments
    std::vector<const char*> args;
    args.push_back(input_file.c_str());

    args.push_back("-I");
    args.push_back(include_dir);

    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options = new clang::DiagnosticOptions();
    std::unique_ptr<clang::TextDiagnosticPrinter> text_diagnostic_printer =
            std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &*diagnostic_options);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids;
    std::unique_ptr<clang::DiagnosticsEngine> diagnostic_engine =
            std::make_unique<clang::DiagnosticsEngine>(diagnostic_ids, &*diagnostic_options, text_diagnostic_printer.release());

    clang::CompilerInstance compiler_instance;

    auto& compiler_invocation = compiler_instance.getInvocation();
    clang::CompilerInvocation::CreateFromArgs(compiler_invocation, args, *diagnostic_engine.release());

    // Map code to a MemoryBuffer
    std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBufferCopy(cpp_src);
    compiler_invocation.getPreprocessorOpts().addRemappedFile(input_file.c_str(), buffer.get());

    compiler_instance.getPreprocessorOpts().addMacroDef("WP_CPU");

    compiler_instance.getLangOpts().MicrosoftExt = 1;  // __forceinline / __int64
    compiler_instance.getLangOpts().DeclSpecKeyword = 1;  // __declspec

    compiler_instance.createDiagnostics(text_diagnostic_printer.get(), false);

    clang::EmitLLVMOnlyAction emit_llvm_only_action(&context);
    bool success = compiler_instance.ExecuteAction(emit_llvm_only_action);
    buffer.release();

    return success ? std::move(emit_llvm_only_action.takeModule()) : nullptr;
}

extern "C" {

WP_API int compile_cpp(const char* cpp_src, const char* include_dir, const char* output_file)
{
    #if defined (_WIN32)
        const char* obj_ext = ".obj";
    #else
        const char* obj_ext = ".o";
    #endif

    std::string input_file = std::string(output_file).substr(0, std::strlen(output_file) - std::strlen(obj_ext));

    initialize_llvm();

    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module = cpp_to_llvm(input_file, cpp_src, include_dir, context);

    if(!module)
    {
        return -1;
    }

    std::string target_triple = llvm::sys::getDefaultTargetTriple();
    std::string Error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, Error);

    const char* CPU = "generic";
    const char* features = "";
    llvm::TargetOptions target_options;
    llvm::Reloc::Model relocation_model = llvm::Reloc::PIC_;  // DLLs need Position Independent Code
    llvm::CodeModel::Model code_model = llvm::CodeModel::Large;  // Don't make assumptions about displacement sizes
    llvm::TargetMachine* target_machine = target->createTargetMachine(target_triple, CPU, features, target_options, relocation_model, code_model);

    module->setDataLayout(target_machine->createDataLayout());

    std::error_code error_code;
    llvm::raw_fd_ostream output(output_file, error_code, llvm::sys::fs::OF_None);

    llvm::legacy::PassManager pass_manager;
    llvm::CodeGenFileType file_type = llvm::CGFT_ObjectFile;
    target_machine->addPassesToEmitFile(pass_manager, output, nullptr, file_type);

    pass_manager.run(*module);
    output.flush();

    delete target_machine;

    return 0;
}

// Global JIT instance
static llvm::orc::LLJIT* jit = nullptr;

// Load an object file into an in-memory DLL named `module_name`
WP_API int load_obj(const char* object_file, const char* module_name)
{
    if(!jit)
    {
        initialize_llvm();

        auto jit_expected = llvm::orc::LLJITBuilder().create();

        if(!jit_expected)
        {
            std::cerr << "Failed to create JIT instance: " << toString(jit_expected.takeError()) << std::endl;
            return -1;
        }

        jit = (*jit_expected).release();
    }

    auto dll = jit->createJITDylib(module_name);

    if(!dll)
    {
        std::cerr << "Failed to create JITDylib: " << toString(dll.takeError()) << std::endl;
        return -1;
    }

    // Enable searching for external symbols in the calling module
    {
        #if defined (_WIN32)
            const char* clang_dll = "warp-clang.dll";
            char global_prefix = '\0';
        #elif defined(__APPLE__)
            const char* clang_dll = "libwarp-clang.dylib";
            char global_prefix = '_';
        #else
            const char* clang_dll = "warp-clang.so";
            char global_prefix = '\0';
        #endif

        auto search = llvm::orc::DynamicLibrarySearchGenerator::Load(clang_dll, global_prefix);

        if(!search)
        {
            std::cerr << "Failed to create generator: " << toString(search.takeError()) << std::endl;
            return -1;
        }

        dll->addGenerator(std::move(*search));
    }

    // Load the object file into a memory buffer
    auto buffer = llvm::MemoryBuffer::getFile(object_file);
    if(!buffer)
    {
        std::cerr << "Failed to load object file: " << buffer.getError().message() << std::endl;
        return -1;
    }

    auto err = jit->addObjectFile(*dll, std::move(*buffer));
    if(err)
    {
        std::cerr << "Failed to add object file: " << llvm::toString(std::move(err)) << std::endl;
        return -1;
    }

    return 0;
}

 WP_API int unload_obj(const char* module_name)
{
    if(!jit)  // If there's no JIT instance there are no object files loaded
    {
        return 0;
    }

    auto* dll = jit->getJITDylibByName(module_name);
    llvm::Error error = jit->getExecutionSession().removeJITDylib(*dll);

    if(error)
    {
        std::cerr << "Failed to unload: " << llvm::toString(std::move(error)) << std::endl;
        return -1;
    }

    return 0;
}

WP_API uint64_t lookup(const char* dll_name, const char* function_name)
{
    auto* dll = jit->getJITDylibByName(dll_name);

    auto func = jit->lookup(*dll, function_name);

    if(!func)
    {
        std::cerr << "Failed to lookup symbol: " << llvm::toString(func.takeError()) << std::endl;
        return -1;
    }

    return func->getValue();
}

}  // extern "C"

}  // namespace wp

