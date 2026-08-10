# Warp C++ Integration Examples

This directory contains examples demonstrating how to integrate Warp workloads into standalone C++ applications.

## Purpose

The examples cover two integration patterns:

- **Ahead-of-time (AOT) kernels**: Compile CUDA kernels from Python, then load their CUBIN modules or include their generated source in C++.
- **API Capture (APIC)**: Record and save a supported Warp workload from Python, then load and replay it from C++ on CUDA or CPU.

Both patterns let the deployed application run without a Python runtime.

## Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[00_cubin_launch](00_cubin_launch/)** | Runtime CUBIN loading with CUDA Driver API | SAXPY operation, CUBIN module loading, `cuLaunchKernel()`, architecture-specific binaries |
| **[01_source_include](01_source_include/)** | Static source inclusion with autodiff | Gradient descent, automatic differentiation, forward/backward kernels, `<<<>>>` launch syntax, multi-architecture compilation |
| **[02_apic_visualization](02_apic_visualization/)** | CUDA replay from a saved API Capture (APIC) representation | APIC operation recording, CUDA graph reconstruction, dynamic parameters, real-time GLFW visualization |
| **[03_apic_visualization_cpu](03_apic_visualization_cpu/)** | CPU replay from a saved APIC representation (no CUDA at runtime) | CPU operation-stream replay, dynamic parameters, `warp-clang`, real-time GLFW visualization |

## Quick Start

```bash
# Run from warp/examples/cpp

# Example 1: Runtime CUBIN loading
(cd 00_cubin_launch && make && ./00_cubin_launch)

# Example 2: Source inclusion with autodiff
(cd 01_source_include && make && ./01_source_include)

# Example 3: APIC save/load with CUDA graph replay
(cd 02_apic_visualization && make && ./02_apic_visualization)

# Example 4: APIC save/load with CPU replay
(cd 03_apic_visualization_cpu && make && ./03_apic_visualization_cpu)
```

## Build Systems

All four examples support two build systems:

- **Makefile** - Supported Unix-like platforms (`make` auto-runs the example's Python setup script if needed)
- **CMake 3.20+** - Cross-platform (run the example's Python setup script before `cmake -B build`: `compile_kernel.py` for the AOT examples, `capture_wave.py` for the APIC examples)

**Make Targets**:
- `make` - Build everything and run the example's Python setup if needed
- `make cpp` - Build only C++ code (fast iteration)
- `make clean` - Remove executable only
- `make distclean` - Remove executable and `generated/` directory

The three CUDA examples (`00`, `01`, and `02`) do not support macOS. The CPU APIC example (`03`) does not require CUDA; see its README for platform-specific prerequisites.

## AOT Workflow

Examples `00` and `01` follow this two-phase workflow.

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

## API Capture Workflow

Examples `02` and `03` use `wp.capture_begin(..., apic=True)` and `wp.capture_save()` to write a `.wrp` file plus a companion `_modules` directory. The standalone C++ application loads both artifacts through the APIC API. The CUDA example reconstructs a CUDA graph for replay; the CPU example directly interprets the recorded operation stream.

## Key Concepts

### Warp AOT Header

The CUDA examples use Warp's AOT (Ahead-Of-Time) header (`warp/native/aot.h`), which provides:

- Automatic CUDA detection and configuration
- Error checking macros (`CHECK_CU`, `CHECK_CUDA`)
- Common Warp type definitions via `builtin.h`:
  - `wp::launch_bounds_t<N>` - Thread count and grid dimensions
  - `wp::array_t<T>` - Array descriptor with pointer, shape, strides
  - `wp::vec_t<N, T>`, `wp::mat_t<N, M, T>` - Vector and matrix types

### CUDA APIs

- **Driver API** (`cuda.h`): CUBIN module loading, `cuLaunchKernel()` (used in 00_cubin_launch)
- **Runtime API** (`cuda_runtime.h`): Memory management, `<<<>>>` launch syntax (used in 01_source_include)
- **Graph API**: CUDA graph reconstruction and `cudaGraphLaunch()` (used in 02_apic_visualization)

## Prerequisites

These examples are designed to run from within the Warp repository.

### Requirements

- **Python 3.10+**
- **CUDA Toolkit and NVIDIA GPU** for examples `00`, `01`, and `02`:
  - `00_cubin_launch` and `02_apic_visualization`: **12.0+**
  - `01_source_include`: **12.8+** (required for `sm_120` compilation)
- **Warp LLVM library (`warp-clang`)** for CPU replay in `03_apic_visualization_cpu`
- **Operating System**: Linux or Windows for the CUDA examples; the CPU example also supports macOS
- **Build System**: GNU Make (Unix/Linux) or CMake 3.20+ (cross-platform)

The visualization examples also require OpenGL and GLFW. See each example's README for exact prerequisites and setup.

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
