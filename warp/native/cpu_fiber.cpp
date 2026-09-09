// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Per-platform context-switch primitives for the cooperative-fiber CPU
// dispatcher. There are two distinct backends:
//
//   POSIX x86-64 SysV (Linux + macOS x86-64) and AArch64 (Linux + macOS arm64):
//     mmap'd stack with a leading PROT_NONE guard page, a custom asm
//     trampoline (`wp_fiber_switch_asm`) that saves callee-saved regs of the
//     outgoing fiber onto its own stack, and an entry-trampoline that fresh
//     fibers `ret` into. ~50?100 ns per switch.
//
//   Windows (x86-64 and ARM64):
//     Win32 fibers (`ConvertThreadToFiberEx` + `CreateFiberEx` +
//     `SwitchToFiber`).
//     Microsoft built this for exactly this use case; using it avoids needing
//     a separate MASM trampoline (MSVC has no inline asm on x64). Per-switch
//     cost is higher (~1 ?s measured) but it's correct and self-contained.
//
// Truly unsupported architectures (everything not covered above) fail the
// build with `#error` rather than getting a silent runtime fallback ? the
// cooperative-fiber dispatcher is required for `block_dim>1` on CPU and the
// silent-fallback path was removed in Phase A.5 follow-up. See
// design/cpu-block-dim-fibers.md.

#include "cpu_fiber.h"

#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// Per-platform fiber state. The asm trampoline (POSIX x86-64 / AArch64) only
// reads `rsp` at offset 0, so we keep that field first on those platforms. On
// Windows we never invoke the asm; the struct just stores the OS handle plus
// bookkeeping.
struct wp_fiber {
#if defined(_WIN32)
    LPVOID win_handle = nullptr;  // from CreateFiber, or the converted-thread handle for the main fiber
    int is_main = 0;  // 1 ? don't DeleteFiber on destroy (the OS thread's implicit main fiber)
#else
    void* rsp = nullptr;  // saved stack pointer (offset 0; the only field the asm reads)
    void* stack_base = nullptr;  // base of mmap'd region (incl. guard page)
    size_t stack_size = 0;  // total mapped bytes (incl. guard page)
#endif
    void (*entry)(void*) = nullptr;
    void* entry_arg = nullptr;
    int finished = 0;  // set to 1 after `entry` returns, before parking
};

namespace {
thread_local wp_fiber_t* g_active_fiber = nullptr;
thread_local wp_fiber_t g_main_fiber {};  // implicit fiber for the OS thread
}

#if !defined(_WIN32)
// asm-side context switch (POSIX backends only). Saves callee-saved regs of
// the caller onto its current stack, sets the caller's `wp_fiber::rsp` to the
// new SP, loads `to->rsp`, pops the new fiber's saved regs, and `ret`s into
// its resumed PC. Implemented as top-level inline asm below.
extern "C" void wp_fiber_switch_asm(wp_fiber_t* from, wp_fiber_t* to);

// Trampoline that fresh fibers `ret` into on first switch-in. C linkage so the
// asm can take its address without name mangling.
extern "C" void wp_fiber_entry_trampoline();
#endif

#if defined(_WIN32)
// =====================================================================
// Windows backend (Win32 fibers API)
// =====================================================================

// Entry callback CreateFiber jumps into when a fiber is first switched-to.
// Sets the per-thread "current fiber" pointer (the moral equivalent of what
// the asm trampoline + wp_fiber_entry_trampoline do on POSIX), runs the user's
// entry, marks finished, then parks on the main fiber forever ? the block
// dispatcher never reschedules a finished fiber, but the OS will terminate
// the thread if a fiber's entry returns, so we have to loop.
//
// Goes through `wp_fiber_switch` (not raw SwitchToFiber) so `g_active_fiber`
// stays consistent with whoever is actually running. The POSIX entry
// trampoline uses the same idiom for the same reason.
extern "C" WP_API void wp_fiber_switch(wp_fiber_t* to);
static VOID CALLBACK win_fiber_entry(PVOID raw)
{
    wp_fiber_t* self = (wp_fiber_t*)raw;
    g_active_fiber = self;
    self->entry(self->entry_arg);
    self->finished = 1;
    for (;;) {
        wp_fiber_switch(&g_main_fiber);
    }
}

#elif defined(__x86_64__)
// =====================================================================
// POSIX x86-64 SysV backend (Linux + macOS x86-64)
// =====================================================================
//
// 6 callee-saved GPRs (RBX, RBP, R12-R15), plus the ABI-preserved MXCSR
// control bits and x87 control word, live on the outgoing fiber's stack.
__asm__(
    ".text\n"
#if defined(__APPLE__)
    ".globl _wp_fiber_switch_asm\n"
    ".private_extern _wp_fiber_switch_asm\n"
    "_wp_fiber_switch_asm:\n"
#else
    ".globl wp_fiber_switch_asm\n"
    ".hidden wp_fiber_switch_asm\n"
    ".type wp_fiber_switch_asm, @function\n"
    "wp_fiber_switch_asm:\n"
#endif
    "    pushq %rbx\n"
    "    pushq %rbp\n"
    "    pushq %r12\n"
    "    pushq %r13\n"
    "    pushq %r14\n"
    "    pushq %r15\n"
    "    subq $16, %rsp\n"
    "    stmxcsr (%rsp)\n"
    "    fnstcw 4(%rsp)\n"
    "    movq %rsp, (%rdi)\n"  // from->rsp = current rsp
    "    movq (%rsi), %rsp\n"  // rsp = to->rsp
    "    ldmxcsr (%rsp)\n"
    "    fldcw 4(%rsp)\n"
    "    addq $16, %rsp\n"
    "    popq %r15\n"
    "    popq %r14\n"
    "    popq %r13\n"
    "    popq %r12\n"
    "    popq %rbp\n"
    "    popq %rbx\n"
    "    ret\n"
#if !defined(__APPLE__)
    ".size wp_fiber_switch_asm, .-wp_fiber_switch_asm\n"
    // Mark stack non-executable, then switch back to .text so the C++
    // functions following this top-level asm aren't emitted into the note
    // section.
    ".section .note.GNU-stack,\"\",@progbits\n"
    ".text\n"
#endif
);

#elif defined(__aarch64__)
// =====================================================================
// POSIX AArch64 backend (Linux arm64 + macOS arm64)
// =====================================================================
//
// AAPCS64 callee-saved set: x19?x28 (10 GPRs) + x29 (FP) + x30 (LR) +
// d8..d15 (8 vector regs, lower 64 bits of v8..v15), FPCR, and FPSR.
// 12*8 + 8*8 + 2*8 = 176 bytes
// of saved state per swap. Layout (low ? high offset from saved-frame base):
//
//     [ 0]  x19   [ 8]  x20
//     [16]  x21   [24]  x22
//     [32]  x23   [40]  x24
//     [48]  x25   [56]  x26
//     [64]  x27   [72]  x28
//     [80]  x29   [88]  x30   ? LR; freshly-init'd fibers preload this with
//                                 wp_fiber_entry_trampoline (see init_fiber_stack)
//     [96]  d8    [104] d9
//     [112] d10   [120] d11
//     [128] d12   [136] d13
//     [144] d14   [152] d15
//
// `ret` jumps to x30, which we just `ldp`-restored from offset 88.
__asm__(
    ".text\n"
#if defined(__APPLE__)
    ".globl _wp_fiber_switch_asm\n"
    ".private_extern _wp_fiber_switch_asm\n"
    "_wp_fiber_switch_asm:\n"
#else
    ".globl wp_fiber_switch_asm\n"
    ".hidden wp_fiber_switch_asm\n"
    ".type wp_fiber_switch_asm, @function\n"
    "wp_fiber_switch_asm:\n"
#endif
    "    sub sp, sp, #176\n"
    "    stp x19, x20, [sp, #0]\n"
    "    stp x21, x22, [sp, #16]\n"
    "    stp x23, x24, [sp, #32]\n"
    "    stp x25, x26, [sp, #48]\n"
    "    stp x27, x28, [sp, #64]\n"
    "    stp x29, x30, [sp, #80]\n"
    "    stp d8, d9,   [sp, #96]\n"
    "    stp d10, d11, [sp, #112]\n"
    "    stp d12, d13, [sp, #128]\n"
    "    stp d14, d15, [sp, #144]\n"
    "    mrs x9, fpcr\n"
    "    str x9, [sp, #160]\n"
    "    mrs x9, fpsr\n"
    "    str x9, [sp, #168]\n"
    "    mov x9, sp\n"  // from->rsp = sp (x0 = from; rsp at offset 0)
    "    str x9, [x0]\n"
    "    ldr x9, [x1]\n"  // sp = to->rsp (x1 = to)
    "    mov sp, x9\n"
    "    ldr x9, [sp, #168]\n"
    "    msr fpsr, x9\n"
    "    ldr x9, [sp, #160]\n"
    "    msr fpcr, x9\n"
    "    ldp d14, d15, [sp, #144]\n"
    "    ldp d12, d13, [sp, #128]\n"
    "    ldp d10, d11, [sp, #112]\n"
    "    ldp d8, d9,   [sp, #96]\n"
    "    ldp x29, x30, [sp, #80]\n"
    "    ldp x27, x28, [sp, #64]\n"
    "    ldp x25, x26, [sp, #48]\n"
    "    ldp x23, x24, [sp, #32]\n"
    "    ldp x21, x22, [sp, #16]\n"
    "    ldp x19, x20, [sp, #0]\n"
    "    add sp, sp, #176\n"
    "    ret\n"
#if !defined(__APPLE__)
    ".size wp_fiber_switch_asm, .-wp_fiber_switch_asm\n"
    ".section .note.GNU-stack,\"\",@progbits\n"
    ".text\n"
#endif
);

#else
#error "wp_fiber: unsupported architecture; supported: POSIX x86-64 SysV, POSIX AArch64, Windows."
#endif

#if !defined(_WIN32)
// Trampoline that runs as the entry point of a fresh fiber on POSIX backends.
// The asm trampoline `ret`s into here when a fiber is first switched-to;
// subsequent re-entries land back in `wp_fiber_switch_asm` (one frame above
// the original caller of `wp_fiber_switch`).
extern "C" void wp_fiber_entry_trampoline()
{
    wp_fiber_t* self = g_active_fiber;
    self->entry(self->entry_arg);
    self->finished = 1;
    // Park forever on g_main_fiber. The block dispatcher never schedules a
    // finished fiber, so the loop body runs at most once in practice; it
    // exists as defense for the case where the dispatcher is bypassed.
    for (;;) {
        wp_fiber_switch(&g_main_fiber);
    }
}
#endif

#if !defined(_WIN32)
namespace {

// AArch64 Linux supports 4 KB, 16 KB, and 64 KB pages depending on
// `CONFIG_ARM64_PAGE_SIZE`. Hardcoding 4 KB would under-round the usable
// region and silently shrink the PROT_NONE guard on 16 KB / 64 KB kernels.
// macOS arm64 is always 16 KB, so the runtime value is needed there too.
inline size_t page_size()
{
    static const size_t kPageSize = (size_t)sysconf(_SC_PAGESIZE);
    return kPageSize;
}

inline size_t round_up_page(size_t n)
{
    const size_t ps = page_size();
    return (n + ps - 1) & ~(ps - 1);
}

// Allocate a fiber stack with a leading PROT_NONE guard page. Returns the
// base of the mapping (low address). Stack-usable region is
// [base + page_size, base + total_size).
void* alloc_stack_with_guard(size_t usable_size, size_t* out_total)
{
    const size_t ps = page_size();
    size_t total = round_up_page(usable_size) + ps;
    *out_total = total;

    void* p = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        return nullptr;
    if (mprotect(p, ps, PROT_NONE) != 0) {
        munmap(p, total);
        return nullptr;
    }
    return p;
}

void free_stack(void* base, size_t total) { munmap(base, total); }

// Initialize a fiber's stack so the first `wp_fiber_switch_asm` into it lands
// in `wp_fiber_entry_trampoline`. Per-arch layout details:
//
//   x86-64 SysV: trampoline addr lives ABOVE the 6 saved-callee-saved-GPR
//   slots; `ret` pops it off the stack into RIP. We add 8 bytes of padding
//   above the trampoline addr so that, after `ret`, RSP is the standard
//   "16-byte-aligned-minus-8" expected at function entry.
//
//   AArch64 (AAPCS64): trampoline addr lives INSIDE the saved-reg block at
//   the x30 (LR) slot ? offset 88 from the saved-frame base. The asm's `ldp
//   x29, x30, [sp, #80]` restores it; `ret` jumps to x30.
void* init_fiber_stack(void* stack_top, size_t* slots_used)
{
    uint8_t* sp = (uint8_t*)stack_top;
    sp = (uint8_t*)((uintptr_t)sp & ~(uintptr_t)15);  // 16-byte align

#if defined(__x86_64__)
    sp -= 8;  // padding so RSP is 16-aligned-minus-8 at trampoline entry
    sp -= 8;
    *(void**)sp = (void*)&wp_fiber_entry_trampoline;
    sp -= 48;  // 6 callee-saved GPRs (RBX, RBP, R12-R15)
    memset(sp, 0, 48);
    sp -= 16;  // MXCSR and x87 control word
    memset(sp, 0, 16);
    __asm__ volatile("stmxcsr %0" : "=m"(*(uint32_t*)sp));
    __asm__ volatile("fnstcw %0" : "=m"(*(uint16_t*)(sp + 4)));
    *slots_used = 6;
#elif defined(__aarch64__)
    sp -= 176;  // 12 GPRs + 8 vector regs + FPCR/FPSR
    memset(sp, 0, 176);
    *(void**)(sp + 88) = (void*)&wp_fiber_entry_trampoline;  // x30 (LR)
    uint64_t fpcr;
    uint64_t fpsr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr));
    *reinterpret_cast<uint64_t*>(sp + 160) = fpcr;
    *reinterpret_cast<uint64_t*>(sp + 168) = fpsr;
    // The memset intentionally initializes the saved x28 slot to zero. CPU
    // codegen reserves x28 for `shared_tile_storage` (see tile.h), but a fresh
    // fiber is created outside the JIT'd dispatcher and cannot safely inherit
    // that register here. Instead, every generated block-lane thunk calls
    // `tile_shared_storage_t::bind()` before entering kernel code. Keep that
    // bind ahead of any tile access or nested Warp call so both fresh and
    // reused AArch64 fibers receive the current block's arena pointer.
    *slots_used = 12 + 8;
#else
#error "wp_fiber: unsupported architecture"
#endif

    return sp;
}

}  // namespace
#endif  // !_WIN32

extern "C" WP_API wp_fiber_t* wp_fiber_create(void (*entry)(void*), void* arg, size_t stack_size)
{
    if (!entry)
        return nullptr;

#if defined(_WIN32)
    // CreateFiber requires the calling thread to be a fiber; lazy-init via
    // wp_fiber_active() (which itself calls ConvertThreadToFiber if needed).
    if (!wp_fiber_active())
        return nullptr;

    wp_fiber_t* f = new (std::nothrow) wp_fiber_t();
    if (!f)
        return nullptr;
    f->entry = entry;
    f->entry_arg = arg;
    f->finished = 0;
    f->is_main = 0;
    // Floor at 16 KB to match the POSIX backend.
    if (stack_size < 16 * 1024)
        stack_size = 16 * 1024;
    // Reserve the requested capacity while committing only the initial pages.
    // FIBER_FLAG_FLOAT_SWITCH makes floating-point control and nonvolatile
    // vector state part of the native context on both Windows architectures.
    const size_t commit_size = stack_size < 64 * 1024 ? stack_size : 64 * 1024;
    f->win_handle = CreateFiberEx(commit_size, stack_size, FIBER_FLAG_FLOAT_SWITCH, &win_fiber_entry, f);
    if (!f->win_handle) {
        delete f;
        return nullptr;
    }
    return f;
#else
    if (stack_size < 16 * 1024)
        stack_size = 16 * 1024;  // 16 KB floor (rounded up to a page in alloc_stack_with_guard)

    wp_fiber_t* f = new (std::nothrow) wp_fiber_t();
    if (!f)
        return nullptr;

    size_t total = 0;
    void* base = alloc_stack_with_guard(stack_size, &total);
    if (!base) {
        delete f;
        return nullptr;
    }
    f->stack_base = base;
    f->stack_size = total;
    f->entry = entry;
    f->entry_arg = arg;
    f->finished = 0;

    void* stack_top = (uint8_t*)base + total;  // stack grows down from here
    size_t slots_used = 0;
    f->rsp = init_fiber_stack(stack_top, &slots_used);
    return f;
#endif
}

extern "C" WP_API void wp_fiber_destroy(wp_fiber_t* f)
{
    // The main fiber is thread-local storage rather than a heap allocation,
    // and unmapping the currently executing stack is never safe. Enforce
    // both preconditions in release builds rather than relying on assert().
    if (!f || f == g_active_fiber || f == &g_main_fiber)
        return;
#if defined(_WIN32)
    if (f->is_main)
        return;
    if (f->win_handle) {
        DeleteFiber(f->win_handle);
    }
#else
    if (f->stack_base) {
        free_stack(f->stack_base, f->stack_size);
    }
#endif
    delete f;
}

extern "C" WP_API wp_fiber_t* wp_fiber_active(void)
{
    if (!g_active_fiber) {
        g_main_fiber = wp_fiber_t {};
#if defined(_WIN32)
        // The first fiber operation on this OS thread "claims" the thread as
        // the implicit main fiber. `IsThreadAFiber` lets us cooperate with
        // hosts that have already converted (Python's main thread doesn't,
        // but a host might).
        g_main_fiber.is_main = 1;
        if (IsThreadAFiber()) {
            g_main_fiber.win_handle = GetCurrentFiber();
        } else {
            g_main_fiber.win_handle = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
        }
        if (!g_main_fiber.win_handle)
            return nullptr;
#endif
        g_active_fiber = &g_main_fiber;
    }
    return g_active_fiber;
}

extern "C" WP_API void wp_fiber_switch(wp_fiber_t* to)
{
    if (!to)
        return;
#if defined(_WIN32)
    if (!to->win_handle)
        return;
#endif
    wp_fiber_t* from = wp_fiber_active();
    if (!from)
        return;
    if (from == to)
        return;
    g_active_fiber = to;
#if defined(_WIN32)
    SwitchToFiber(to->win_handle);
#else
    wp_fiber_switch_asm(from, to);
#endif
    // Resumed: g_active_fiber has been reset by whoever switched back to us.
}

extern "C" WP_API int wp_fiber_finished(wp_fiber_t* f) { return f ? f->finished : 1; }

namespace {

struct fiber_test_state {
    wp_fiber_t* main_fiber;
    int result;
    int expected_rounding;
};

#if defined(_MSC_VER)
#define WP_FIBER_NOINLINE __declspec(noinline)
#else
#define WP_FIBER_NOINLINE __attribute__((noinline))
#endif

WP_FIBER_NOINLINE int fiber_test_deep_call(int depth, uintptr_t seed)
{
    volatile uintptr_t values[32];
    for (int i = 0; i < 32; ++i)
        values[i] = seed + (uintptr_t)i;
    if (depth == 0)
        return values[17] == seed + 17;
    return fiber_test_deep_call(depth - 1, seed + 37) && values[9] == seed + 9;
}

void fiber_test_abi_entry(void* raw)
{
    fiber_test_state* state = static_cast<fiber_test_state*>(raw);
    alignas(16) volatile uint8_t aligned_value[16] {};
    volatile uint64_t integer_values[] = {
        0x0123456789abcdefull,
        0xfedcba9876543210ull,
        0x55aa55aa55aa55aaull,
    };
    volatile double floating_values[] = { 1.25, -8.5, 1024.125, 0.03125 };

    std::fesetround(FE_DOWNWARD);
    state->expected_rounding = std::fegetround();
    wp_fiber_switch(state->main_fiber);

    const bool aligned = (reinterpret_cast<uintptr_t>(&aligned_value) & 15u) == 0;
    const bool integers_ok = integer_values[0] == 0x0123456789abcdefull && integer_values[1] == 0xfedcba9876543210ull
        && integer_values[2] == 0x55aa55aa55aa55aaull;
    const bool floating_ok = floating_values[0] == 1.25 && floating_values[1] == -8.5 && floating_values[2] == 1024.125
        && floating_values[3] == 0.03125;
    state->result = aligned && integers_ok && floating_ok && std::fegetround() == state->expected_rounding;
}

struct deep_call_state {
    int depth;
    int result;
};

void fiber_test_deep_entry(void* raw)
{
    deep_call_state* state = static_cast<deep_call_state*>(raw);
    state->result = fiber_test_deep_call(state->depth, 0x13579bdu);
}

WP_FIBER_NOINLINE uintptr_t fiber_test_overflow_call(uintptr_t depth)
{
    // Keep each frame smaller than a guard page so successive frames cannot
    // skip over it. Reading the volatile frame after the recursive call also
    // prevents the optimizer from converting the recursion into a loop.
    volatile uintptr_t values[32];
    for (int i = 0; i < 32; ++i)
        values[i] = depth + (uintptr_t)i;
    if (depth == UINTPTR_MAX)
        return values[0];
    const uintptr_t nested = fiber_test_overflow_call(depth + 1);
    return values[depth & 31u] + nested;
}

void fiber_test_overflow_entry(void*) { (void)fiber_test_overflow_call(0); }

}  // namespace

extern "C" WP_API int wp_fiber_test_abi()
{
    const int saved_rounding = std::fegetround();
    std::fesetround(FE_TONEAREST);

    fiber_test_state state { wp_fiber_active(), 0, FE_TONEAREST };
    wp_fiber_t* fiber = wp_fiber_create(&fiber_test_abi_entry, &state, 64 * 1024);
    if (!fiber) {
        std::fesetround(saved_rounding);
        return 0;
    }

    wp_fiber_switch(fiber);
    const bool main_state_ok = std::fegetround() == FE_TONEAREST;
    std::fesetround(FE_UPWARD);
    wp_fiber_switch(fiber);
    const bool resumed_main_state_ok = std::fegetround() == FE_UPWARD;
    const bool finished = wp_fiber_finished(fiber) != 0;
    wp_fiber_destroy(fiber);
    std::fesetround(saved_rounding);
    return state.result && main_state_ok && resumed_main_state_ok && finished;
}

extern "C" WP_API int wp_fiber_test_deep_calls(size_t stack_size, int depth)
{
    deep_call_state state { depth, 0 };
    wp_fiber_t* fiber = wp_fiber_create(&fiber_test_deep_entry, &state, stack_size);
    if (!fiber)
        return 0;
    wp_fiber_switch(fiber);
    const bool finished = wp_fiber_finished(fiber) != 0;
    wp_fiber_destroy(fiber);
    return state.result && finished;
}

extern "C" WP_API void wp_fiber_test_overflow(size_t stack_size)
{
    wp_fiber_t* fiber = wp_fiber_create(&fiber_test_overflow_entry, nullptr, stack_size);
    if (!fiber)
        return;
    wp_fiber_switch(fiber);
    wp_fiber_destroy(fiber);
}
