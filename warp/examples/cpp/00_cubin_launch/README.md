# Warp C++ Integration Example: SAXPY

This example demonstrates a workflow for launching previously compiled Warp kernels from a C++ CUDA program.
It implements SAXPY (Single-Precision A·X Plus Y), a classic BLAS operation that computes `y = alpha * x + y`.

This workflow enables developers to author high-performance kernels in Python using Warp while executing them from existing C++ applications.

## Overview

The example shows a complete two-phase workflow:

1. **Python Phase**: Compile Warp kernels to CUBIN using `wp.compile_aot_module()`
2. **C++ Phase**: Load and launch the compiled kernels using CUDA Driver API

**Scope**: This example demonstrates the simplest case—one module with one kernel, generating a single CUBIN file.
The patterns shown here can be extended to handle multiple kernels or modules.

This approach allows you to:

- Write and test kernels easily in pure Python - iterate quickly before integrating into C++
- Integrate those kernels into production C++ codebases
- Avoid runtime Python dependencies in deployment

## Files

- `compile_kernel.py` - Python script that compiles Warp kernels to CUBIN
- `main.cu` - C++ program that loads and launches the compiled kernel
- `Makefile` / `CMakeLists.txt` - Build systems (Make for Unix, CMake for cross-platform)
- `generated/` - Directory containing generated files (auto-created by compilation):
  - `*.cubin` - Compiled kernel binary (architecture-specific)
  - `*.cu` - Generated CUDA source code (for reference)
  - `*.meta` - Kernel metadata (not used in this example)

## Quick Start

### Prerequisites

This example runs from within the Warp repository.

**Requirements:**
- **Python 3.9+**
- **CUDA Toolkit 12.0+** (includes `nvcc` compiler)
- **NVIDIA GPU** (Warp automatically compiles for your GPU architecture)
- **Build System**: GNU Make (Unix/Linux) or CMake 3.20+ (cross-platform)
- **Note**: macOS is **not supported** (CUDA not available on macOS)

**Setup:**

Clone and build Warp:
```bash
git clone https://github.com/NVIDIA/warp.git
cd warp

# Build and install (choose one):
uv run build_lib.py  # Option A (recommended - handles everything automatically)
# or: python build_lib.py && pip install -e .  # Option B (use a virtual environment)

cd warp/examples/cpp/00_cubin_launch
```

### Build and Run

**Using Make (Unix/Linux)**:

```bash
make              # Build everything (auto-compiles kernel if needed)
./00_cubin_launch # Run
```

**Make Targets**:

- `make` (default) - Build everything, auto-compile kernel if needed
- `make cpp` - Build only C++ program (fast iteration, assumes CUBIN exists)
- `make clean` - Remove executable only
- `make distclean` - Remove executable and `generated/` directory
- `make help` - Show all available targets

**Using CMake (cross-platform)**:

```bash
python compile_kernel.py                  # Step 1: Compile kernel to CUBIN
cmake -B build -DCMAKE_BUILD_TYPE=Release # Step 2: Configure
cmake --build build --config Release      # Step 3: Build
./build/00_cubin_launch                   # Step 4: Run (Unix) or build\Release\00_cubin_launch.exe (Windows)
```

## How It Works

### 1. Python: Compile Kernel to CUBIN

The Python script compiles a SAXPY kernel to CUBIN:

```python
@wp.kernel
def saxpy(alpha: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
    """SAXPY: Single-Precision A·X Plus Y
    
    Computes: y = alpha * x + y
    """
    tid = wp.tid()
    y[tid] = alpha * x[tid] + y[tid]

# Compile to CUBIN (use_ptx=False for CUBIN, strip_hash=True for predictable kernel names)
wp.compile_aot_module("__main__", module_dir="generated/", use_ptx=False, strip_hash=True)
```

This generates `generated/wp___main__.sm*.cubin` where `*` is the target GPU architecture (e.g., `sm120` for Blackwell).

The build system automatically detects this CUBIN file. No architecture flags are needed for compilation since the C++ launcher is host-only code using the CUDA Driver API—the CUBIN file itself is architecture-specific.

### 2. C++: Load CUBIN and Launch Kernel

The C++ program loads the compiled CUBIN module and launches the SAXPY kernel.
The Makefile passes `CUBIN_FILE` and `KERNEL_NAME` as compile-time defines:

```cpp
#include "aot.h"  // Warp types and utilities

// CUBIN_FILE and KERNEL_NAME are defined by Makefile via -D flags

// 1. Load CUBIN module (using CUDA Driver API)
std::string cubin = read_file(CUBIN_FILE);
cuModuleLoadData(&module, cubin.c_str());
cuModuleGetFunction(&kernel, module, KERNEL_NAME);

// 2. Allocate device memory (using Runtime API)
float *d_x, *d_y;
cudaMalloc(&d_x, N * sizeof(float));
cudaMalloc(&d_y, N * sizeof(float));

// 3. Create Warp data structures (kernel signature: saxpy(alpha, x, y))
wp::launch_bounds_t dim = {.shape = {N, 0, 0, 0}, .ndim = 1, .size = size_t(N)};
wp::float32 alpha = 2.5f;  // Scalar parameter
wp::array_t<wp::float32> arr_x(d_x, N);
wp::array_t<wp::float32> arr_y(d_y, N);

// 4. Launch kernel - parameters match kernel signature order
void* params[] = {&dim, &alpha, &arr_x, &arr_y};
cuLaunchKernel(kernel, grid_dim, 1, 1, block_dim, 1, 1, 0, nullptr, params, nullptr);
```

**Key points:**

- Use Warp's `aot.h` header which automatically detects CUDA compilation and includes `builtin.h`
- Provides `wp::launch_bounds_t` and `wp::array_t<T>` definitions plus error checking macros
- Use CUDA Driver API for CUBIN loading (`cuModuleLoadData`)
- Use CUDA Runtime API for memory management (`cudaMalloc`)
- **All parameters** (scalars, arrays, structs) are passed as pointers in the `params[]` array—this is a
  requirement of `cuLaunchKernel`, which copies the values into kernel parameter space
- Parameter order must match the kernel signature: `dim`, then kernel parameters in order
