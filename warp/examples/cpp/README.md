# Warp C++ Integration Examples

This directory contains examples demonstrating how to integrate Warp-compiled kernels into standalone C++ applications.

## Purpose

These examples show how to author and test GPU kernels in Python using Warp, then deploy them in C++ programs without runtime Python dependencies. This workflow enables:

- **Rapid prototyping**: Write and test kernels in Python with Warp's expressive API
- **Production deployment**: Integrate proven kernels into existing C++ codebases
- **Performance**: Leverage Warp's kernel generation without Python runtime overhead
- **Flexibility**: Choose between runtime CUBIN loading or compile-time source inclusion

## Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[00_cubin_launch](00_cubin_launch/)** | Runtime CUBIN loading with CUDA Driver API | SAXPY operation, CUBIN module loading, `cuLaunchKernel()`, architecture-specific binaries |
| **[01_source_include](01_source_include/)** | Static source inclusion with autodiff | Gradient descent, automatic differentiation, forward/backward kernels, `<<<>>>` launch syntax, multi-architecture compilation |

## Quick Start

```bash
# Example 1: Runtime CUBIN loading
cd 00_cubin_launch
make && ./00_cubin_launch

# Example 2: Source inclusion with autodiff
cd 01_source_include
make && ./01_source_include
```

## Build Systems

Both examples support dual build systems:

- **Makefile** - Unix/Linux only (`make` auto-compiles kernels)
- **CMake 3.20+** - Cross-platform: Linux and Windows (requires manual `python compile_kernel.py` first)

**Make Targets**:
- `make` - Build everything, auto-compile kernel if needed
- `make cpp` - Build only C++ code (fast iteration)
- `make clean` - Remove executable only
- `make distclean` - Remove executable and `generated/` directory

**Note**: macOS is **not supported** - these examples require CUDA, which is not available on macOS.

## General Workflow

All examples follow a two-phase workflow:

### 1. Python Phase: Compile Kernels

```python
import warp as wp

@wp.kernel
def my_kernel(x: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0

wp.init()
wp.compile_aot_module("__main__", module_dir="generated/", strip_hash=True)
```

Generates:

- `generated/wp___main__.sm*.cubin` - Compiled kernel binary (for runtime loading)
- `generated/wp___main__.cu` - Generated CUDA source (for static inclusion)

### 2. C++ Phase: Load and Launch

#### Option A: Runtime CUBIN Loading (00_cubin_launch)

```cpp
#include "aot.h"  // Warp types and utilities

// Load CUBIN module
cuModuleLoadData(&module, cubin_data.c_str());
cuModuleGetFunction(&kernel, module, "my_kernel_cuda_kernel_forward");

// Launch with Driver API
void* params[] = {&dim, &arr_x};
cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, nullptr, params, nullptr);
```

#### Option B: Static Source Inclusion (01_source_include)

```cpp
#include "aot.h"  // Warp types and utilities
#include "generated/wp___main__.cu"

// Launch with Runtime API
my_kernel_cuda_kernel_forward<<<grid, block>>>(dim, arr_x);
```

## Key Concepts

### Warp AOT Header

Examples use Warp's AOT (Ahead-Of-Time) header (`warp/native/aot.h`) which provides:

- Automatic CUDA detection and configuration
- Error checking macros (`CHECK_CU`, `CHECK_CUDA`)
- Common Warp type definitions via `builtin.h`:
  - `wp::launch_bounds_t` - Thread count and grid dimensions
  - `wp::array_t<T>` - Array descriptor with pointer, shape, strides
  - `wp::vec_t<N, T>`, `wp::mat_t<N, M, T>` - Vector and matrix types

### CUDA APIs

- **Driver API** (`cuda.h`): CUBIN module loading, `cuLaunchKernel()` (used in 00_cubin_launch)
- **Runtime API** (`cuda_runtime.h`): Memory management, `<<<>>>` launch syntax (used in 01_source_include)

## Prerequisites

These examples are designed to run from within the Warp repository.

### Requirements

- **Python 3.9+**
- **CUDA Toolkit**:
  - `00_cubin_launch`: **12.0+** (Warp's minimum requirement)
  - `01_source_include`: **12.8+** (required for `sm_120` compilation)
- **NVIDIA GPU** with CUDA support
- **Operating System**: Linux or Windows (macOS not supported - no CUDA)
- **Build System**: GNU Make (Unix/Linux) or CMake 3.20+ (cross-platform)

### Setup

1. Clone the Warp repository:
   ```bash
   git clone https://github.com/NVIDIA/warp.git
   cd warp
   ```

2. Build Warp and install:

   **Option A - Using `uv` (recommended):**
   ```bash
   uv run build_lib.py  # Handles dependencies, environment, and installation automatically
   ```

   **Option B - Using `python` directly:**
   ```bash
   # Create a virtual environment (recommended - use venv, conda, etc.)
   python build_lib.py
   pip install -e .
   ```

3. Navigate to the examples:
   ```bash
   cd warp/examples/cpp
   ```

## Future Examples

More sophisticated integration patterns are planned, including:

- **Invoking Warp functions from CUDA kernels** - Calling Warp-generated device functions (and their adjoints) from custom CUDA kernel code
- **Tile programming API integration** - Running kernels authored with Warp's tile API from C++ applications

## Testing

All examples include automated tests via CTest for CI/CD regression testing:

```bash
# Run all example tests
bash test_examples.sh

# Or manually with CMake
cmake -B build && ctest --test-dir build --output-on-failure
```

Tests verify that examples compile and run successfully.
