/** Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "../native/builtin.h"

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

#include <lld/Common/Driver.h>

#include <vector>
#include <iostream>
#include <string>
#include <cstring>

namespace wp {

extern "C" {

std::unique_ptr<llvm::Module> cpp_to_llvm(const std::string &input_file, const char* cpp_src, const char* include_dir, llvm::LLVMContext& context)
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

WP_API int compile_cpp(const char* cpp_src, const char* include_dir, const char* output_file)
{
    std::string input_file = std::string(output_file).substr(0, std::strlen(output_file) - std::strlen(".obj"));

    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

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
    llvm::TargetMachine* target_machine = target->createTargetMachine(target_triple, CPU, features, target_options, relocation_model);

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

WP_API int link(int argc, const char** argv)
{
    std::vector<const char*> args = {"lld-link.exe"};

	for(int i = 0; i < argc; i++)
	{
	    args.push_back(argv[i]);
	}    

    bool success = lld::coff::link(args, llvm::outs(), llvm::errs(), /*exitEarly*/ false, /*disableOutput*/ false);

    lld::CommonLinkerContext::destroy();

    return success ? 0 : -1;
}

}  // extern "C"

}  // namespace wp