// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Build @add(i32, i32) with IRBuilder, JIT it with LLJIT, and execute it.
// This exercises target initialization, codegen, and the ORC JIT the same
// way Warp's CPU backend does, without driving a full C++ frontend. Clang
// linkage is proven by printing the clang version string.

#include <cstdio>
#include <memory>
#include <string>

#include <clang/Basic/Version.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/Triple.h>

int main() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeNativeTargetAsmPrinter();

    // The SDK must include the NVPTX backend (Warp emits PTX through it).
    std::string error;
    if (!llvm::TargetRegistry::lookupTarget(llvm::Triple("nvptx64-nvidia-cuda"), error)) {
        std::fprintf(stderr, "NVPTX backend missing: %s\n", error.c_str());
        return 1;
    }

    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("test", *context);
    auto* int32 = llvm::Type::getInt32Ty(*context);
    auto* fn_type = llvm::FunctionType::get(int32, {int32, int32}, false);
    auto* fn = llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage, "add", module.get());
    auto* block = llvm::BasicBlock::Create(*context, "entry", fn);
    llvm::IRBuilder<> builder(block);
    builder.CreateRet(builder.CreateAdd(fn->getArg(0), fn->getArg(1)));

    auto jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());
    llvm::cantFail(jit->addIRModule(
        llvm::orc::ThreadSafeModule(std::move(module), std::move(context))));
    auto addr = llvm::cantFail(jit->lookup("add"));
    auto* add = addr.toPtr<int (*)(int, int)>();

    const int result = add(2, 3);
    if (result != 5) {
        std::fprintf(stderr, "JIT returned %d, expected 5\n", result);
        return 1;
    }
    std::printf("clang %s: JIT add(2, 3) == %d\n", clang::getClangFullVersion().c_str(), result);
    return 0;
}
